"""Reference intensity ``I0(x, y)`` of the empty, back-lit cylinder.

Physics / algorithm
-------------------
The absorbance of a frame is measured *against* the same view of the
cylinder before the pour: every pixel then carries its own reference
intensity, so the non-uniformities of the back-light (vignetting, texture
of the diffuser, the glass itself) cancel out in the ratio ``I / I0``.

1. ``stack_reference``: per-pixel mean and standard deviation of ``N``
   consecutive frames of the empty cylinder (``N = 10`` in the original
   experiment). The mean is the reference; the standard deviation tells
   how stable the back-light was during those frames.
2. ``dead_pixel_masks``: pixels darker than ``dead_thresh`` in the
   reference (printed graduations and numbers, glass defects, dust) are
   *dead*: their reference is not a measurement of the back-light. The
   mask is dilated by ``dead_dilate`` px (Euclidean, elliptical kernel) so
   the blurred edge of each mark is excluded as well.
3. ``fill_dead_pixels``: the dead pixels are replaced by the unweighted
   mean of the valid pixels in a square window of radius ``r_fill`` (a
   mask-aware box filter). Valid pixels keep their exact value; the same
   fill is applied to the live frames so that ``I`` and ``I0`` are
   treated identically.
4. ``prepare_intensity``: channel selection -> dead-pixel fill ->
   isotropic Gaussian blur of ``light_blur`` px. The reference and every
   live frame go through this same function, so the absorbance compares
   homogeneous quantities.

Coordinates: everything here works on the *crop band*
``frame[:, x_left:x_right + 1]`` (full height), like the gradient
detector, so rows are frame rows.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cylvision.detection.gradient import to_gray

DEFAULT_N_REF_FRAMES = 10


@dataclass
class Reference:
    """Per-pixel statistics of the empty cylinder (BGR crop, ``float32``)."""

    mean_bgr: np.ndarray            # (H, W, 3) float32
    std_bgr: np.ndarray             # (H, W, 3) float32
    frame_start: int                # first frame index used
    frame_count: int                # number of frames averaged

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.mean_bgr.shape[0]), int(self.mean_bgr.shape[1])

    def intensity(self, channel: str = "G") -> np.ndarray:
        """Mean reference on one channel (``"gray"``, ``"B"``, ``"G"``, ``"R"``), ``float32``."""
        return channel_intensity(self.mean_bgr, channel)

    def to_npz_dict(self) -> dict[str, Any]:
        return {
            "mean_bgr": self.mean_bgr.astype(np.float32),
            "std_bgr": self.std_bgr.astype(np.float32),
            "frame_start": int(self.frame_start),
            "frame_count": int(self.frame_count),
        }

    @classmethod
    def from_npz(cls, npz: Any) -> "Reference":
        """Rebuild from ``np.load`` of a file written with :meth:`to_npz_dict`."""
        return cls(
            mean_bgr=np.asarray(npz["mean_bgr"], dtype=np.float32),
            std_bgr=np.asarray(npz["std_bgr"], dtype=np.float32),
            frame_start=int(npz["frame_start"]),
            frame_count=int(npz["frame_count"]),
        )


def channel_intensity(image_bgr: np.ndarray, channel: str = "G") -> np.ndarray:
    """One plane of a BGR image (or the grey level) as ``float32``.

    A ``float32`` BGR image is accepted (the reference mean): the BGR
    planes are picked directly and the grey level is the OpenCV weighted
    sum ``0.299 R + 0.587 G + 0.114 B``.
    """
    if image_bgr.ndim == 2:
        return image_bgr.astype(np.float32, copy=False)
    if image_bgr.dtype == np.uint8:
        return to_gray(image_bgr, channel).astype(np.float32)
    if channel == "gray":
        b, g, r = image_bgr[:, :, 0], image_bgr[:, :, 1], image_bgr[:, :, 2]
        return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)
    return np.ascontiguousarray(to_gray(image_bgr, channel), dtype=np.float32)


def stack_reference(frames: Iterable[np.ndarray], frame_start: int = 0) -> Reference:
    """Per-pixel mean / std of an iterable of BGR crops of the same shape."""
    stack = [np.asarray(f, dtype=np.float32) for f in frames]
    if not stack:
        raise ValueError("stack_reference needs at least one frame")
    arr = np.stack(stack, axis=0)
    if arr.ndim == 3:  # grey frames -> add a channel axis for a uniform layout
        arr = np.repeat(arr[..., None], 3, axis=3)
    return Reference(mean_bgr=arr.mean(axis=0), std_bgr=arr.std(axis=0),
                     frame_start=int(frame_start), frame_count=int(arr.shape[0]))


def reference_from_video(video: Any, start: int, stop: int,
                         crop: tuple[int, int] | None = None) -> Reference:
    """Average the frames ``start .. stop - 1`` of a ``VideoSource``.

    ``crop = (x_left, x_right)`` keeps the columns ``x_left .. x_right``
    (inclusive) of every frame, like the detector's crop band.
    """
    frames = []
    for _idx, fr in video.frames(int(start), int(stop), 1):
        if crop is not None:
            x_left, x_right = int(crop[0]), int(crop[1])
            fr = fr[:, x_left:x_right + 1]
        frames.append(fr)
    if not frames:
        raise OSError(f"no frame could be read in {start}..{stop - 1}")
    return stack_reference(frames, frame_start=int(start))


def dead_pixel_masks(ref_intensity: np.ndarray, dead_thresh: float,
                     dead_dilate: int) -> tuple[np.ndarray, np.ndarray]:
    """``(direct, dilated)`` boolean masks of the dead pixels of a reference.

    ``direct = ref < dead_thresh``; ``dilated`` adds every pixel within a
    Euclidean distance ``dead_dilate`` of a direct dead pixel (``0``
    returns a copy of ``direct``).
    """
    direct = np.asarray(ref_intensity, dtype=np.float32) < float(dead_thresh)
    if dead_dilate <= 0:
        return direct, direct.copy()
    k = 2 * int(dead_dilate) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(direct.astype(np.uint8), kernel).astype(bool)
    return direct, dilated


def fill_dead_pixels(image: np.ndarray, valid: np.ndarray, r_fill: int) -> np.ndarray:
    """Replace the pixels where ``valid`` is False by the local mean of the valid ones.

    The window is a square of radius ``r_fill`` (``2 r + 1`` px wide); a
    pixel with no valid neighbour in its window falls back to the global
    mean of the valid pixels. Valid pixels are returned unchanged. Output is
    ``float32`` with the shape of ``image``.
    """
    img = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != img.shape:
        raise ValueError(f"valid mask shape {valid.shape} != image shape {img.shape}")
    if valid.all():
        return img.copy()
    v = valid.astype(np.float32)
    k = max(1, int(r_fill))
    ksize = (2 * k + 1, 2 * k + 1)
    num = cv2.boxFilter(img * v, -1, ksize, normalize=False, borderType=cv2.BORDER_REPLICATE)
    den = cv2.boxFilter(v, -1, ksize, normalize=False, borderType=cv2.BORDER_REPLICATE)
    n_valid = float(v.sum())
    global_mean = float((img * v).sum() / n_valid) if n_valid > 0 else float(img.mean())
    local = np.where(den > 0.5, num / np.maximum(den, 1e-6), global_mean)
    return np.where(valid, img, local).astype(np.float32)


def prepare_intensity(image_bgr: np.ndarray, channel: str = "G", *,
                      valid: np.ndarray | None = None, r_fill: int = 5,
                      light_blur: float = 1.0) -> np.ndarray:
    """Channel -> dead-pixel fill -> Gaussian blur; the common path of ``I`` and ``I0``."""
    inten = channel_intensity(image_bgr, channel)
    if valid is not None:
        inten = fill_dead_pixels(inten, valid, r_fill)
    if light_blur > 0:
        inten = cv2.GaussianBlur(inten, (0, 0), sigmaX=float(light_blur))
    return np.asarray(inten, dtype=np.float32)


__all__ = [
    "DEFAULT_N_REF_FRAMES", "Reference", "channel_intensity", "stack_reference",
    "reference_from_video", "dead_pixel_masks", "fill_dead_pixels", "prepare_intensity",
]
