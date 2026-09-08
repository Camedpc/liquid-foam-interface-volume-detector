"""TrueType text on OpenCV images (Pillow), with a Hershey fallback.

OpenCV only ships the Hershey vector fonts, which look like plotter output
next to a photograph. Every label of the overlays goes through this module
instead: the text is rasterised by Pillow with a real TrueType font, drawn
on a small region of the BGR array and pasted back.

Fonts are looked up by *name* in the ``fonts/ttf`` folder that ships with
matplotlib (``DejaVuSans``, ``DejaVuSans-Bold``, ``DejaVuSansMono``,
``DejaVuSerif``, ``STIXGeneral``, ``cmss10``, ``cmtt10``...), which makes
the rendering reproducible on every machine that can import matplotlib. A
name that is an existing ``.ttf`` path is used as is. When Pillow is not
installed or the font cannot be loaded, :func:`draw_text` falls back to
``cv2.putText`` with a Hershey font of the same cap height, so nothing
breaks -- the labels just look plainer.

Public API
----------
:func:`font_path` -> path of a named font (or ``None``).
:func:`load_font` -> cached ``PIL.ImageFont`` (or ``None``).
:func:`text_size` -> ``(w, h)`` in px of a string, letter spacing included.
:func:`draw_text` -> draw a string at ``xy`` with an anchor, optional letter
spacing and an optional translucent backing box; returns the bounding box.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # Pillow is optional at runtime (see the fallback).
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:  # pragma: no cover - exercised only without Pillow
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _HAVE_PIL = False

HERSHEY = cv2.FONT_HERSHEY_SIMPLEX

# ``anchor`` = (horizontal, vertical): l / c / r  x  t / m / b.
_ANCHORS = {"lt", "lm", "lb", "ct", "cm", "cb", "rt", "rm", "rb"}


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _ttf_dir() -> Path | None:
    try:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        return d if d.is_dir() else None
    except Exception:  # pragma: no cover
        return None


def font_path(name: str) -> Path | None:
    """Resolve ``name`` to a ``.ttf`` file: an existing path, or a matplotlib font."""
    p = Path(name)
    if p.suffix.lower() in (".ttf", ".otf") and p.is_file():
        return p
    d = _ttf_dir()
    if d is None:
        return None
    for cand in (d / f"{name}.ttf", d / name, d / f"{name}.otf"):
        if cand.is_file():
            return cand
    return None


@functools.lru_cache(maxsize=64)
def load_font(name: str, size: int) -> Any | None:
    """``PIL.ImageFont.FreeTypeFont`` of ``name`` at ``size`` px, or ``None``."""
    if not _HAVE_PIL:
        return None
    path = font_path(name)
    if path is None:
        return None
    try:
        return ImageFont.truetype(str(path), int(size))
    except Exception:
        return None


def available() -> bool:
    """True when Pillow and the matplotlib font folder are both usable."""
    return _HAVE_PIL and _ttf_dir() is not None


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def _hershey_scale(size: int) -> float:
    # Hershey simplex cap height is about 22 px at scale 1.0.
    return max(0.3, size / 22.0)


def _advance(font: Any, ch: str) -> float:
    try:
        return float(font.getlength(ch))
    except Exception:  # pragma: no cover
        return float(font.getbbox(ch)[2])


def text_size(text: str, font_name: str, size: int, *, letter_spacing: float = 0.0
              ) -> tuple[int, int]:
    """``(width, height)`` in px of ``text``; height = ascent + descent of the font."""
    font = load_font(font_name, size)
    if font is None:
        (tw, th), base = cv2.getTextSize(text, HERSHEY, _hershey_scale(size), 1)
        return int(tw + letter_spacing * max(0, len(text) - 1)), int(th + base)
    asc, desc = font.getmetrics()
    if not text:
        return 0, int(asc + desc)
    if letter_spacing:
        w = sum(_advance(font, ch) for ch in text) + letter_spacing * (len(text) - 1)
    else:
        w = _advance(font, text)
    return int(round(w)), int(asc + desc)


def _origin(xy: tuple[int, int], w: int, h: int, anchor: str) -> tuple[int, int]:
    if anchor not in _ANCHORS:
        raise ValueError(f"anchor must be one of {sorted(_ANCHORS)}, got {anchor!r}")
    x, y = int(xy[0]), int(xy[1])
    hx, vy = anchor[0], anchor[1]
    x0 = x if hx == "l" else (x - w // 2 if hx == "c" else x - w)
    y0 = y if vy == "t" else (y - h // 2 if vy == "m" else y - h)
    return x0, y0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def fill_rect_alpha(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                    color: tuple[int, int, int], alpha: float, *, radius: int = 0) -> None:
    """Blend a (rounded) rectangle ``[x0, x1] x [y0, y1]`` into ``img`` in place."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W - 1, int(x1)), min(H - 1, int(y1))
    if x1 < x0 or y1 < y0 or alpha <= 0.0:
        return
    region = img[y0:y1 + 1, x0:x1 + 1]
    tint = np.empty_like(region)
    tint[:] = color
    blended = cv2.addWeighted(region, 1.0 - alpha, tint, alpha, 0.0)
    r = int(radius)
    if r > 0 and region.shape[0] > 2 * r and region.shape[1] > 2 * r:
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        h, w = mask.shape
        cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx_, cy_ in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(mask, (cx_, cy_), r, 255, -1, cv2.LINE_AA)
        m = (mask.astype(np.float32) / 255.0)[..., None]
        region[:] = np.clip(region * (1.0 - m) + blended * m, 0, 255).astype(np.uint8)
    else:
        region[:] = blended


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], font_name: str, size: int,
              color: tuple[int, int, int], anchor: str = "lt", *,
              letter_spacing: float = 0.0,
              backing: tuple[tuple[int, int, int], float] | None = None,
              backing_pad: tuple[int, int] = (6, 3), backing_radius: int = 3,
              bar: tuple[tuple[int, int, int], int] | None = None,
              shadow: tuple[int, int, int] | None = None) -> tuple[int, int, int, int]:
    """Draw ``text`` on the BGR array ``img`` in place; return its box ``(x0, y0, x1, y1)``.

    Parameters
    ----------
    xy, anchor
        The point of the text box that sits at ``xy``: ``"lt"`` (left/top,
        default), ``"lm"``, ``"rb"``, ``"cm"``... (horizontal l/c/r, then
        vertical t/m/b). The box is the full line box (ascent + descent), so
        strings of the same font align whatever their glyphs.
    letter_spacing
        Extra advance in px between consecutive glyphs (tracking).
    backing
        ``(bgr, alpha)``: a translucent rectangle behind the text, padded by
        ``backing_pad = (px, py)``, corners rounded by ``backing_radius``.
    bar
        ``(bgr, width)``: a solid vertical bar on the left edge of the
        backing box (only with ``backing``).
    shadow
        Colour of a 1 px drop shadow under the glyphs (bottom-right).

    Falls back to a Hershey font when the TrueType font is unavailable.
    """
    if img.ndim != 3:
        raise ValueError("draw_text needs a BGR image")
    H, W = img.shape[:2]
    tw, th = text_size(text, font_name, size, letter_spacing=letter_spacing)
    x0, y0 = _origin(xy, tw, th, anchor)
    box = (x0, y0, x0 + tw, y0 + th)
    if backing is not None:
        px, py = int(backing_pad[0]), int(backing_pad[1])
        bx0, by0, bx1, by1 = x0 - px, y0 - py, x0 + tw + px, y0 + th + py
        fill_rect_alpha(img, bx0, by0, bx1, by1, backing[0], float(backing[1]), radius=backing_radius)
        if bar is not None and bar[1] > 0:
            cv2.rectangle(img, (max(0, bx0), max(0, by0)),
                          (min(W - 1, bx0 + int(bar[1]) - 1), min(H - 1, by1)), bar[0], -1)
        box = (bx0, by0, bx1, by1)
    if not text:
        return box
    font = load_font(font_name, size)
    if font is None:
        _draw_hershey(img, text, x0, y0, th, size, color, letter_spacing, shadow)
        return box
    # Work on the smallest region that holds the glyphs (+ the shadow).
    rx0, ry0 = max(0, x0 - 2), max(0, y0 - 2)
    rx1, ry1 = min(W, x0 + tw + 4), min(H, y0 + th + 4)
    if rx1 <= rx0 or ry1 <= ry0:
        return box
    region = img[ry0:ry1, rx0:rx1]
    pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    sh = None if shadow is None else (int(shadow[2]), int(shadow[1]), int(shadow[0]))
    ox, oy = x0 - rx0, y0 - ry0

    def put(dx: int, dy: int, fill: tuple[int, int, int]) -> None:
        if letter_spacing:
            x = float(ox + dx)
            for ch in text:
                draw.text((x, oy + dy), ch, font=font, fill=fill)
                x += _advance(font, ch) + letter_spacing
        else:
            draw.text((ox + dx, oy + dy), text, font=font, fill=fill)

    if sh is not None:
        put(1, 1, sh)
    put(0, 0, rgb)
    region[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return box


def _draw_hershey(img: np.ndarray, text: str, x0: int, y0: int, th: int, size: int,
                  color: tuple[int, int, int], letter_spacing: float,
                  shadow: tuple[int, int, int] | None) -> None:
    scale = _hershey_scale(size)
    (_, h), base = cv2.getTextSize(text, HERSHEY, scale, 1)
    y_base = y0 + (th + h - base) // 2
    thick = 1 if size < 18 else 2
    if letter_spacing:
        x = float(x0)
        for ch in text:
            if shadow is not None:
                cv2.putText(img, ch, (int(x) + 1, y_base + 1), HERSHEY, scale, shadow, thick, cv2.LINE_AA)
            cv2.putText(img, ch, (int(x), y_base), HERSHEY, scale, color, thick, cv2.LINE_AA)
            x += cv2.getTextSize(ch, HERSHEY, scale, 1)[0][0] + letter_spacing
    else:
        if shadow is not None:
            cv2.putText(img, text, (x0 + 1, y_base + 1), HERSHEY, scale, shadow, thick, cv2.LINE_AA)
        cv2.putText(img, text, (x0, y_base), HERSHEY, scale, color, thick, cv2.LINE_AA)


__all__ = ["font_path", "load_font", "available", "text_size", "fill_rect_alpha", "draw_text"]
