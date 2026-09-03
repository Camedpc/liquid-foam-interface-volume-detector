"""Zoom loupe for pixel-accurate clicks on an OpenCV image.

``make_magnifier`` renders a ``SOURCE_SIZE x SOURCE_SIZE`` neighbourhood of
the cursor enlarged ``ZOOM_FACTOR`` times with nearest-neighbour
interpolation (every source pixel becomes a flat 8x8 block, so the
pointed pixel is unambiguous), with a central crosshair and a one-pixel
box. ``overlay_magnifier`` pastes it in the top-left corner of the display
and jumps to the top-right corner when the cursor would be hidden.
``draw_crosshair`` marks the cursor on the main image.
"""
from __future__ import annotations

import cv2
import numpy as np

ZOOM_FACTOR = 8                              # power of two: 8x8 block per source pixel
SOURCE_SIZE = 45                             # source px shown
MAGNIFIED_SIZE = SOURCE_SIZE * ZOOM_FACTOR   # 360 px on screen

CROSSHAIR_COLOR = (0, 255, 255)              # yellow
BORDER_COLOR = (255, 255, 255)
PAD_VALUE = (40, 40, 40)


def make_magnifier(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Return the loupe centred on ``(cx, cy)`` with the central crosshair."""
    h, w = img.shape[:2]
    half = SOURCE_SIZE // 2

    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    x0c, x1c = max(0, x0), min(w, x1)
    y0c, y1c = max(0, y0), min(h, y1)

    region = img[y0c:y1c, x0c:x1c].copy() if (x1c > x0c and y1c > y0c) else None
    if region is not None and region.ndim == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)

    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - h)
    pad_left = max(0, -x0)
    pad_right = max(0, x1 - w)

    if region is None or region.size == 0:
        region = np.full((SOURCE_SIZE, SOURCE_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    elif any((pad_top, pad_bottom, pad_left, pad_right)):
        region = cv2.copyMakeBorder(region, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=PAD_VALUE)

    magnified = cv2.resize(region, (MAGNIFIED_SIZE, MAGNIFIED_SIZE),
                           interpolation=cv2.INTER_NEAREST)

    # Central crosshair + a box the size of one source pixel.
    c = MAGNIFIED_SIZE // 2
    cv2.line(magnified, (c, 0), (c, MAGNIFIED_SIZE), CROSSHAIR_COLOR, 1)
    cv2.line(magnified, (0, c), (MAGNIFIED_SIZE, c), CROSSHAIR_COLOR, 1)
    cv2.rectangle(magnified,
                  (c - ZOOM_FACTOR // 2, c - ZOOM_FACTOR // 2),
                  (c + ZOOM_FACTOR // 2, c + ZOOM_FACTOR // 2),
                  CROSSHAIR_COLOR, 1)
    cv2.rectangle(magnified, (0, 0), (MAGNIFIED_SIZE - 1, MAGNIFIED_SIZE - 1), BORDER_COLOR, 2)
    cv2.putText(magnified, f"({cx},{cy})  zoom x{ZOOM_FACTOR}", (8, MAGNIFIED_SIZE - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BORDER_COLOR, 1, cv2.LINE_AA)
    return magnified


def overlay_magnifier(display: np.ndarray, magnifier: np.ndarray,
                      cursor: tuple[int, int], margin: int = 12) -> None:
    """Paste the loupe top-left, or top-right when the cursor is under it.

    When the display is smaller than the loupe (narrow portrait crops) the
    loupe is cropped to fit instead of raising a broadcasting error.
    """
    H, W = display.shape[:2]
    h, w = magnifier.shape[:2]
    cx, cy = cursor

    px, py = margin, margin
    if cx < w + 2 * margin and cy < h + 2 * margin:
        px = W - w - margin
    px = max(0, min(px, max(0, W - w)))
    py = max(0, min(py, max(0, H - h)))

    h_avail = max(0, min(h, H - py))
    w_avail = max(0, min(w, W - px))
    if h_avail <= 0 or w_avail <= 0:
        return
    display[py:py + h_avail, px:px + w_avail] = magnifier[:h_avail, :w_avail]


def draw_crosshair(display: np.ndarray, x: int, y: int, size: int = 18,
                   color: tuple[int, int, int] = (0, 0, 255)) -> None:
    """Thin crosshair at the cursor position on the main image."""
    H, W = display.shape[:2]
    if 0 <= x < W and 0 <= y < H:
        cv2.line(display, (x, max(0, y - size)), (x, min(H, y + size)), color, 1)
        cv2.line(display, (max(0, x - size), y), (min(W, x + size), y), color, 1)


__all__ = ["ZOOM_FACTOR", "SOURCE_SIZE", "MAGNIFIED_SIZE", "make_magnifier",
           "overlay_magnifier", "draw_crosshair"]
