"""Colour palette shared by the Tk control panel and the OpenCV canvases.

The palette is a Catppuccin-Mocha-like dark theme. Two representations are
provided:

* ``PALETTE`` / the ``HEX_*`` constants: ``"#rrggbb"`` strings for Tkinter;
* the ``*_BGR`` constants: ``(b, g, r)`` tuples for OpenCV drawing calls,
  derived from the same hex values through :func:`hex_to_bgr`.

Semantic colours used by every overlay (keep them consistent across the
tuner, the batch figures and the README screenshots):

======================  =========  ==================================
role                    colour     where
======================  =========  ==================================
lower interface         red        per-column dots, mean line, mask
upper interface         teal       per-column dots, mean line, mask
lower band |x-cx|<=r    orange     vertical markers
upper band              yellow     vertical markers
cylinder axis (cx)      mauve      vertical line
gradient mask y_top/... magenta    horizontal lines
upper bound (empirical) yellow     dashed line (uncertainty figures)
lower bound (empirical) cyan       dashed line (uncertainty figures)
======================  =========  ==================================

Section accents of the control panel: cylinder = mauve, gradient = teal,
view = blue, sampling = peach.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Hex palette (Tkinter)
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "base":     "#1e1e2e",
    "mantle":   "#181825",
    "crust":    "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "text":     "#cdd6f4",
    "subtext":  "#a6adc8",
    "overlay":  "#6c7086",
    # accents
    "mauve":    "#cba6f7",
    "yellow":   "#f9e2af",
    "teal":     "#94e2d5",
    "red":      "#f38ba8",
    "green":    "#a6e3a1",
    "blue":     "#89b4fa",
    "peach":    "#fab387",
    "sky":      "#89dceb",
    "pink":     "#f5c2e7",
}

BG = PALETTE["base"]
SURFACE = PALETTE["surface0"]
SURFACE_HI = PALETTE["surface1"]
TEXT = PALETTE["text"]
SUBTEXT = PALETTE["subtext"]

# Section accents of the control panel.
SECTION_CYLINDER = PALETTE["mauve"]
SECTION_GRADIENT = PALETTE["teal"]
SECTION_VIEW = PALETTE["blue"]
SECTION_SAMPLING = PALETTE["peach"]

# Default fonts (Windows names, Tk falls back gracefully elsewhere).
FONT_TITLE = ("Segoe UI Semibold", 11)
FONT_LABEL = ("Segoe UI", 9)
FONT_VALUE = ("Cascadia Mono", 9)
FONT_HOTKEY = ("Segoe UI", 8)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert ``"#rrggbb"`` to an OpenCV ``(b, g, r)`` tuple of ints."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #rrggbb, got {hex_color!r}")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    """Convert an OpenCV ``(b, g, r)`` tuple to ``"#rrggbb"``."""
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# BGR constants (OpenCV)
# ---------------------------------------------------------------------------

BG_BGR = hex_to_bgr(BG)
SURFACE_BGR = hex_to_bgr(SURFACE)
SURFACE_HI_BGR = hex_to_bgr(SURFACE_HI)
TEXT_BGR = hex_to_bgr(TEXT)
SUBTEXT_BGR = hex_to_bgr(SUBTEXT)
OVERLAY_BGR = hex_to_bgr(PALETTE["overlay"])

# Overlays are drawn on photographs: saturated colours read better than the
# pastel Tk accents, so the interface colours are fully saturated while the
# secondary markers keep the palette tint.
LOWER_BGR: tuple[int, int, int] = (40, 40, 255)      # red  - lower interface
UPPER_BGR: tuple[int, int, int] = (230, 220, 0)      # teal - upper interface
BAND_LOWER_BGR: tuple[int, int, int] = (0, 165, 255)  # orange - lower band
BAND_UPPER_BGR: tuple[int, int, int] = (0, 220, 220)  # yellow - upper band
AXIS_BGR = hex_to_bgr(PALETTE["mauve"])               # cylinder axis
MASK_BGR: tuple[int, int, int] = (255, 0, 255)        # magenta - y_top / y_bottom
BOUND_UPPER_BGR: tuple[int, int, int] = (0, 230, 230)  # yellow dashed - upper bound
BOUND_LOWER_BGR: tuple[int, int, int] = (255, 230, 0)  # cyan dashed - lower bound
EXTRA_LOWEST_BGR: tuple[int, int, int] = (0, 200, 255)  # lowest upper dot (legacy)
EXTRA_BLOB_BGR: tuple[int, int, int] = (200, 100, 255)  # lowest blob centroid (legacy)

STRIP_BG_BGR: tuple[int, int, int] = SURFACE_BGR
STRIP_TEXT_BGR: tuple[int, int, int] = TEXT_BGR
SEPARATOR_BGR: tuple[int, int, int] = hex_to_bgr(PALETTE["crust"])

# Scientific-illustration strokes (brace + labels on the specimen panel).
BRACE_BGR: tuple[int, int, int] = (245, 245, 245)
BRACE_OUTLINE_BGR: tuple[int, int, int] = (0, 0, 0)

__all__ = [
    "PALETTE", "BG", "SURFACE", "SURFACE_HI", "TEXT", "SUBTEXT",
    "SECTION_CYLINDER", "SECTION_GRADIENT", "SECTION_VIEW", "SECTION_SAMPLING",
    "FONT_TITLE", "FONT_LABEL", "FONT_VALUE", "FONT_HOTKEY",
    "hex_to_bgr", "bgr_to_hex",
    "BG_BGR", "SURFACE_BGR", "SURFACE_HI_BGR", "TEXT_BGR", "SUBTEXT_BGR", "OVERLAY_BGR",
    "LOWER_BGR", "UPPER_BGR", "BAND_LOWER_BGR", "BAND_UPPER_BGR", "AXIS_BGR",
    "MASK_BGR", "BOUND_UPPER_BGR", "BOUND_LOWER_BGR", "EXTRA_LOWEST_BGR",
    "EXTRA_BLOB_BGR", "STRIP_BG_BGR", "STRIP_TEXT_BGR", "SEPARATOR_BGR",
    "BRACE_BGR", "BRACE_OUTLINE_BGR",
]
