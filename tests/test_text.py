"""TrueType text helper (``cylvision.ui.text``) and the readout boxes of the panels."""
from __future__ import annotations

import numpy as np
import pytest

from cylvision.ui import text as uitext
from cylvision.ui import theme
from cylvision.detection.panels import draw_readout, readout_size


def _blank(h: int = 60, w: int = 200) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    return img


def test_matplotlib_fonts_resolve() -> None:
    for name in (theme.LABEL_FONT_ZONE, theme.LABEL_FONT_VALUE, "DejaVuSans-Bold"):
        assert uitext.font_path(name) is not None, name
    assert uitext.font_path("NoSuchFont-XYZ") is None
    assert uitext.load_font("NoSuchFont-XYZ", 12) is None


def test_text_size_grows_with_size_and_spacing() -> None:
    w1, h1 = uitext.text_size("FOAM", "DejaVuSans", 11)
    w2, h2 = uitext.text_size("FOAM", "DejaVuSans", 22)
    w3, _ = uitext.text_size("FOAM", "DejaVuSans", 11, letter_spacing=2.0)
    assert 0 < w1 < w2 and 0 < h1 < h2
    assert w3 == pytest.approx(w1 + 3 * 2.0, abs=1.5)
    # Monospaced digits: every digit has the same advance.
    ws = {uitext.text_size(d, "DejaVuSansMono", 17)[0] for d in "0123456789"}
    assert len(ws) == 1


def test_draw_text_marks_pixels_and_returns_box() -> None:
    img = _blank()
    box = uitext.draw_text(img, "440 mL", (10, 10), "DejaVuSansMono", 17, (255, 255, 255), "lt")
    x0, y0, x1, y1 = box
    assert x0 == 10 and y0 == 10 and x1 > x0 and y1 > y0
    assert img[y0:y1, x0:x1].max() > 200          # glyphs were drawn
    assert img[:, :x0].max() == 40                # nothing left of the box
    # Anchors: "rb" puts the box's bottom-right corner at xy.
    img2 = _blank()
    bx = uitext.draw_text(img2, "440 mL", (150, 50), "DejaVuSansMono", 17, (255, 255, 255), "rb")
    assert bx[2] == 150 and bx[3] == 50
    with pytest.raises(ValueError):
        uitext.draw_text(img2, "x", (0, 0), "DejaVuSans", 12, (255, 255, 255), "zz")


def test_draw_text_backing_and_bar() -> None:
    img = _blank()
    box = uitext.draw_text(img, "TOTAL", (30, 20), "DejaVuSans", 11, (255, 255, 255), "lt",
                           letter_spacing=1.5, backing=((0, 0, 0), 0.6), bar=((0, 0, 255), 2))
    x0, y0, x1, y1 = box
    assert x0 < 30 and y0 < 20                    # padded around the text
    # The bar is solid red on the left edge; the backing darkened the rest.
    assert tuple(img[(y0 + y1) // 2, x0]) == (0, 0, 255)
    assert img[y1 - 1, x1 - 1].max() < 40


def test_hershey_fallback_still_draws() -> None:
    img = _blank()
    box = uitext.draw_text(img, "LIQUID 213 mL", (5, 5), "NoSuchFont-XYZ", 14, (255, 255, 255), "lt")
    x0, y0, x1, y1 = box
    assert x1 > x0 and img[y0:y1 + 2, x0:x1 + 2].max() > 200


def test_readout_box_alignment() -> None:
    rows = [("total", "653 mL", theme.UPPER_BGR), ("foam", "431 mL", theme.FOAM_WASH_BGR),
            ("liquid", "83 mL", theme.LOWER_BGR)]
    w, h, ww, vw = readout_size(rows)
    assert w > ww + vw and h > 3 * 10
    w_min = readout_size(rows, min_width=w + 40)[0]
    assert w_min == w + 40
    img = _blank(120, 260)
    box = draw_readout(img, (10, 10), rows, "lt")
    assert box[0] == 10 and box[1] == 10 and box[2] - box[0] + 1 == w and box[3] - box[1] + 1 == h
    # Colour bar of the first row = upper (teal) at the top-left corner.
    assert tuple(img[box[1] + 2, box[0]]) == tuple(theme.UPPER_BGR)
    # Empty rows: nothing drawn, degenerate box.
    assert draw_readout(img, (0, 0), [], "lt") == (0, 0, 0, 0)
