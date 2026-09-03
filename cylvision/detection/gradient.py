"""Vertical intensity gradient of a cylinder crop, and the row masks.

Physics / algorithm
-------------------
An interface between two phases (air / foam / liquid) is a horizontal
transition of brightness that spans the whole width of the cylinder.
Bubbles inside the foam are also brightness transitions, but they are
short, isolated and oriented in every direction. The pipeline is built to
keep the former and crush the latter:

1. ``to_gray``: grey conversion, or a single colour channel when one
   channel carries more contrast (e.g. ``"G"`` on a green screen).
2. Gaussian smoothing (``blur_sigma`` px, isotropic) removes pixel noise
   and spreads each interface over a few rows (which later helps the
   connected-component filter).
3. Optional HORIZONTAL box filter (``blur_h`` px wide, 1 px tall). It
   averages every row along ``x``: the horizontal interfaces are
   preserved (they are already constant along ``x``) while the vertical
   texture of the foam (each bubble is a point-like micro-gradient) is
   washed out. ``0`` or ``1`` disables it; an even width is bumped to the
   next odd value because OpenCV requires odd kernel sizes.
4. Sobel derivative along ``y`` (ksize 3) -> ``Gy(x, y) = dI/dy`` as
   ``float32``. The image ``y`` axis points DOWN, so ``Gy > 0`` means the
   image gets brighter going down.

Masks: the glass rim at the top of the cylinder and the base / table at
the bottom create strong spurious gradients. ``apply_top_mask`` zeroes
the rows ``y < y_top`` and ``apply_bottom_mask`` zeroes the rows
``y > y_bottom`` (both bounds INCLUDED in the active area). A zero
gradient never exceeds a strictly positive threshold, so masked rows can
never be detected as an interface. The masks do not crop: ``y`` inside
the gradient equals ``y`` in the original frame.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

Channel = Literal["gray", "B", "G", "R"]
CHANNELS: tuple[str, ...] = ("gray", "B", "G", "R")

_CHANNEL_INDEX = {"B": 0, "G": 1, "R": 2}


def to_gray(crop_bgr: np.ndarray, channel: Channel | str = "gray") -> np.ndarray:
    """Return a single-channel ``uint8`` image from a BGR (or grey) crop.

    ``channel`` selects the grey conversion (``"gray"``) or one of the BGR
    planes (``"B"``, ``"G"``, ``"R"``). A 2-D input is returned unchanged
    whatever ``channel`` says (there is only one plane to pick).
    """
    if crop_bgr.ndim == 2:
        return crop_bgr
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] not in (3, 4):
        raise ValueError(f"expected an HxW or HxWx3 image, got shape {crop_bgr.shape}")
    if channel == "gray":
        return cv2.cvtColor(crop_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    try:
        idx = _CHANNEL_INDEX[channel]
    except KeyError as exc:
        raise ValueError(f"unknown channel {channel!r}, expected one of {CHANNELS}") from exc
    return np.ascontiguousarray(crop_bgr[:, :, idx])


def compute_gradient(crop_gray: np.ndarray, blur_sigma: float,
                     blur_h: int = 0) -> np.ndarray:
    """Gaussian blur + optional horizontal box filter + Sobel-y.

    Parameters
    ----------
    crop_gray
        Single-channel image (``uint8`` or float).
    blur_sigma
        Gaussian sigma in px (``<= 0`` disables the blur).
    blur_h
        Width in px of the horizontal box filter applied AFTER the
        Gaussian. ``0`` or ``1`` disables it; even values are rounded up
        to the next odd value.

    Returns
    -------
    np.ndarray
        ``float32`` array of the same shape, ``Gy = dI/dy`` (Sobel ksize 3,
        so the response to a unit slope is 8, not 1).
    """
    if crop_gray.ndim != 2:
        raise ValueError(f"compute_gradient expects a 2-D image, got shape {crop_gray.shape}")
    if blur_sigma > 0:
        smoothed = cv2.GaussianBlur(crop_gray, (0, 0), sigmaX=float(blur_sigma))
    else:
        smoothed = crop_gray
    if blur_h >= 2:
        kw = int(blur_h) if blur_h % 2 == 1 else int(blur_h) + 1
        smoothed = cv2.boxFilter(smoothed, -1, (kw, 1))
    return cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)


def apply_top_mask(Gy: np.ndarray, y_top: int) -> np.ndarray:
    """Zero the gradient on rows ``y < y_top`` (glass rim exclusion).

    Returns a copy when something is masked, the input array otherwise.
    """
    if y_top is None or y_top <= 0:
        return Gy
    out = Gy.copy()
    out[:int(y_top)] = 0.0
    return out


def apply_bottom_mask(Gy: np.ndarray, y_bottom: int | None) -> np.ndarray:
    """Zero the gradient on rows ``y > y_bottom`` (base / table exclusion).

    The row ``y == y_bottom`` stays active, symmetric to ``apply_top_mask``.
    Returns a copy when something is masked, the input array otherwise.
    """
    if y_bottom is None:
        return Gy
    H = Gy.shape[0]
    if y_bottom >= H - 1:
        return Gy
    out = Gy.copy()
    out[int(y_bottom) + 1:] = 0.0
    return out


__all__ = ["Channel", "CHANNELS", "to_gray", "compute_gradient",
           "apply_top_mask", "apply_bottom_mask"]
