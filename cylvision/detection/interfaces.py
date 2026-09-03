"""Two-interface detection (liquid/foam and foam/air) on a vertical gradient.

Vocabulary
----------
The two interfaces are named by POSITION, not by substance:

* **lower interface** = liquid / foam (beer / foam in the original
  experiment). Its mean row is ``y_lower``.
* **upper interface** = foam / air. Its mean row is ``y_upper``.

``y`` grows DOWNWARDS (OpenCV), so ``y_upper < y_lower`` and the volume
read on the graduated cylinder increases when ``y`` decreases.

Polarity
--------
``Gy = dI/dy`` is positive where the image gets brighter going down. The
sign of the gradient at each interface depends only on the lighting:

* ``"lower_darker"`` -- the phase BELOW an interface is darker than the
  phase above it. Front-lit cylinder on a black background: bright foam
  over dark beer, dark air above the foam. Lower interface: ``Gy < 0``
  peak; upper interface: ``Gy > 0``.
* ``"lower_brighter"`` -- the phase below is brighter. Back-lit green
  screen: the beer transmits the light, the foam scatters it and looks
  dark, and the air above the foam is bright. Lower interface: ``Gy > 0``;
  upper interface: ``Gy < 0``.

The two interfaces ALWAYS have opposite gradient signs whatever the
lighting, so a single parameter flips both. Implementation::

    sign = -1 if polarity == "lower_darker" else +1
    S = sign * Gy                      # "S > 0" = lower-interface candidates

and every other line of the algorithm is written for ``S`` (this keeps the
maths of the original detector, written for ``-Gy``, unchanged).

Algorithm (two ordered passes)
------------------------------
1. **Lower interface**, per column: ``argmax_y S`` validated by
   ``max_y S > T_lower``. The valid columns inside the band
   ``|x - cx| <= r_lower`` are averaged -> ``y_lower``. The band keeps the
   measurement away from the cylinder walls, where refraction through the
   glass and parallax bend the interface.
2. **Upper interface**, per column, searched BOTTOM-UP from ``y_lower``:
   the first row ``y < floor(y_lower)`` (scanning upwards) where
   ``-S > T_upper``. Averaged over the band ``|x - cx| <= r_upper``
   (independent of ``r_lower``) -> ``y_upper``.

   Why bottom-up instead of a raw ``argmax``: the foam is porous, every
   bubble edge creates a micro-gradient of the "upper" sign that may beat
   the peak of the true foam/air transition. A per-column ``argmax`` then
   picks a random bubble edge higher in the foam. Scanning upwards from
   the lower interface returns the LOWEST transition above threshold,
   which is the top of the foam, not a bubble 50 px above it.

   **Connected-component filter** ``min_h_upper``: the boolean mask
   ``-S > T_upper`` is labelled (8-connectivity) and only the pixels that
   belong to a component whose bounding-box HEIGHT is at least
   ``min_h_upper`` are kept. The criterion is applied to the whole blob,
   not to the vertical run of the column: the thin ends of an arched foam
   top survive as long as the blob they belong to is tall enough, whereas
   an isolated bubble edge (1-2 px tall) is rejected. ``0`` or ``1``
   disables the filter. Gaussian smoothing spreads the true interface
   over 3-5 rows, so ``3`` is a good default.

   If the lower interface was not found, the search covers the whole
   masked height.

Masks: rows ``y < y_top`` and ``y > y_bottom`` are zeroed before both
passes (see ``gradient.apply_top_mask``). ``r >= W // 2`` means "whole
crop width" for either band.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

import cv2
import numpy as np

from .gradient import apply_bottom_mask, apply_top_mask, compute_gradient, to_gray

Polarity = Literal["lower_darker", "lower_brighter"]
POLARITIES: tuple[str, ...] = ("lower_darker", "lower_brighter")

# Legacy parameter names (interface_tuner state / params.json written by the
# original research scripts). Legacy code was written for the front-lit
# black-background setup, i.e. polarity "lower_darker": the lower interface
# was the NEGATIVE peak (T_neg / r_neg) and the upper interface the POSITIVE
# one (T_pos / r_pos / min_h_pos).
_LEGACY_KEYS: dict[str, str] = {
    "T_neg": "T_lower",
    "T_pos": "T_upper",
    "r_neg": "r_lower",
    "r_pos": "r_upper",
    "min_h_pos": "min_h_upper",
    "blur_horizontal": "blur_h",
}


def polarity_sign(polarity: str) -> int:
    """``-1`` for ``"lower_darker"``, ``+1`` for ``"lower_brighter"``."""
    if polarity == "lower_darker":
        return -1
    if polarity == "lower_brighter":
        return 1
    raise ValueError(f"unknown polarity {polarity!r}, expected one of {POLARITIES}")


@dataclass
class DetectionParams:
    """Every knob of the detector (json-friendly, no global state).

    Thresholds are in Sobel units (``compute_gradient`` responds 8 to a
    unit slope in grey levels / px).
    """

    polarity: Polarity = "lower_darker"
    T_lower: float = 40.0        # gradient threshold, lower (liquid/foam) interface
    T_upper: float = 47.0        # gradient threshold, upper (foam/air) interface
    r_lower: int = 80            # half-width of the averaging band around cx (px)
    r_upper: int = 80
    blur_sigma: float = 5.0      # Gaussian sigma (px)
    blur_h: int = 38             # horizontal box-filter width (px), 0 = off
    min_h_upper: int = 3         # connected-component min height, upper interface (px)
    y_top: int = 0               # gradient mask: rows above are ignored
    y_bottom: int | None = None  # gradient mask: rows below are ignored
    cx: int | None = None        # axis column inside the crop, None = W // 2
    channel: str = "gray"        # "gray" | "B" | "G" | "R"

    def __post_init__(self) -> None:
        if self.polarity not in POLARITIES:
            raise ValueError(f"polarity must be one of {POLARITIES}, got {self.polarity!r}")

    @property
    def sign(self) -> int:
        """Sign applied to ``Gy`` so that the lower interface is a positive peak."""
        return polarity_sign(self.polarity)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict with json-serialisable values."""
        d = asdict(self)
        d["T_lower"] = float(d["T_lower"])
        d["T_upper"] = float(d["T_upper"])
        d["blur_sigma"] = float(d["blur_sigma"])
        for k in ("r_lower", "r_upper", "blur_h", "min_h_upper", "y_top"):
            d[k] = int(d[k])
        d["y_bottom"] = None if d["y_bottom"] is None else int(d["y_bottom"])
        d["cx"] = None if d["cx"] is None else int(d["cx"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DetectionParams":
        """Build from a dict, accepting legacy keys.

        Legacy keys (``T_pos``, ``T_neg``, ``r_pos``, ``r_neg``,
        ``min_h_pos``) are translated with the legacy meaning (negative
        peak = lower interface) and ``polarity`` defaults to
        ``"lower_darker"`` when absent. Unknown keys are ignored.
        """
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in d.items():
            name = _LEGACY_KEYS.get(key, key)
            if name in known and name not in kwargs:
                kwargs[name] = value
        # Explicit new-style keys win over legacy aliases.
        for key, value in d.items():
            if key in known:
                kwargs[key] = value
        if "polarity" not in kwargs or kwargs["polarity"] is None:
            kwargs["polarity"] = "lower_darker"
        p = cls(**kwargs)
        # Normalise numeric types (json may carry ints for floats and vice versa).
        p.T_lower = float(p.T_lower)
        p.T_upper = float(p.T_upper)
        p.blur_sigma = float(p.blur_sigma)
        p.r_lower = int(p.r_lower)
        p.r_upper = int(p.r_upper)
        p.blur_h = int(p.blur_h)
        p.min_h_upper = int(p.min_h_upper)
        p.y_top = int(p.y_top)
        p.y_bottom = None if p.y_bottom is None else int(p.y_bottom)
        p.cx = None if p.cx is None else int(p.cx)
        p.channel = str(p.channel)
        return p

    def resolved_cx(self, W: int) -> int:
        """``cx`` clamped to the crop, or ``W // 2`` when unset."""
        cx = W // 2 if self.cx is None else int(self.cx)
        return max(0, min(W - 1, cx))


@dataclass
class InterfaceResult:
    """Output of :func:`detect_interfaces`. Rows are crop == frame rows."""

    y_lower: float | None
    y_upper: float | None
    lower_rows: np.ndarray          # per-column argmax row (int32, W)
    lower_valid: np.ndarray         # bool (W): above threshold AND inside band
    upper_rows: np.ndarray          # per-column first row above threshold, bottom-up
    upper_valid: np.ndarray         # bool (W): valid AND inside band
    band_lower: tuple[int, int]     # x_lo, x_hi inclusive
    band_upper: tuple[int, int]
    n_lower: int
    n_upper: int
    std_lower: float | None         # std of lower_rows over valid columns
    upper_lowest: int | None        # lowest (max y) valid upper row inside the band
    upper_low_blob: float | None    # centroid y of the lowest connected component
    S: np.ndarray                   # signed gradient used (sign * Gy, masked)
    lower_above_T: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    upper_above_T: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def found_lower(self) -> bool:
        return self.y_lower is not None

    @property
    def found_upper(self) -> bool:
        return self.y_upper is not None

    def to_dict(self) -> dict[str, Any]:
        """Scalar summary (no arrays), json-friendly."""
        return {
            "y_lower": self.y_lower, "y_upper": self.y_upper,
            "n_lower": self.n_lower, "n_upper": self.n_upper,
            "std_lower": self.std_lower,
            "band_lower": list(self.band_lower), "band_upper": list(self.band_upper),
            "upper_lowest": self.upper_lowest, "upper_low_blob": self.upper_low_blob,
        }


def band_columns(W: int, cx: int, r: int) -> tuple[int, int]:
    """Inclusive column range ``[x_lo, x_hi]`` of the band ``|x - cx| <= r``."""
    x_lo = max(0, int(cx) - int(r))
    x_hi = min(W - 1, int(cx) + int(r))
    return x_lo, x_hi


def _band_mask(W: int, x_lo: int, x_hi: int) -> np.ndarray:
    m = np.zeros(W, dtype=bool)
    m[x_lo:x_hi + 1] = True
    return m


def _filter_min_height(mask: np.ndarray, min_h: int) -> np.ndarray:
    """Keep only pixels of 8-connected components at least ``min_h`` rows tall."""
    if min_h < 2 or not mask.any():
        return mask if min_h < 2 else np.zeros_like(mask)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    keep = stats[:, cv2.CC_STAT_HEIGHT] >= int(min_h)
    keep[0] = False  # background
    return keep[labels]


def detect_interfaces(Gy: np.ndarray, params: DetectionParams) -> InterfaceResult:
    """Detect the lower then the upper interface on a vertical gradient.

    ``Gy`` is ``dI/dy`` (see ``gradient.compute_gradient``). The masks
    ``params.y_top`` / ``params.y_bottom`` are applied here (zeroing is
    idempotent, so pre-masked input is fine). See the module docstring for
    the algorithm and the polarity convention.
    """
    if Gy.ndim != 2:
        raise ValueError(f"Gy must be 2-D, got shape {Gy.shape}")
    H, W = Gy.shape
    Gy_m = apply_bottom_mask(apply_top_mask(Gy, params.y_top), params.y_bottom)
    S = (params.sign * Gy_m).astype(np.float32, copy=False)

    cx = params.resolved_cx(W)
    x_lo_l, x_hi_l = band_columns(W, cx, params.r_lower)
    x_lo_u, x_hi_u = band_columns(W, cx, params.r_upper)
    band_lower = _band_mask(W, x_lo_l, x_hi_l)
    band_upper = _band_mask(W, x_lo_u, x_hi_u)

    # --- 1. lower interface: per-column argmax of S, validated by T_lower ---
    lower_rows = np.argmax(S, axis=0).astype(np.int32)
    lower_max = S[lower_rows, np.arange(W)]
    lower_above = lower_max > float(params.T_lower)
    lower_valid = lower_above & band_lower
    if lower_valid.any():
        rows_l = lower_rows[lower_valid].astype(np.float64)
        y_lower: float | None = float(rows_l.mean())
        std_lower: float | None = float(rows_l.std())
    else:
        y_lower, std_lower = None, None

    # --- 2. upper interface: bottom-up from y_lower, first row with -S > T_upper ---
    limit = int(np.floor(y_lower)) if y_lower is not None else H  # exclusive
    upper_rows = np.zeros(W, dtype=np.int32)
    upper_above = np.zeros(W, dtype=bool)
    sub = np.zeros((0, W), dtype=bool)
    if limit > 0:
        sub = (-S[:limit, :]) > float(params.T_upper)
        sub = _filter_min_height(sub, int(params.min_h_upper))
        sub_rev = sub[::-1, :]
        idx_from_bottom = np.argmax(sub_rev, axis=0)
        upper_above = sub_rev.any(axis=0)
        upper_rows = np.where(upper_above, limit - 1 - idx_from_bottom, 0).astype(np.int32)
    upper_valid = upper_above & band_upper

    y_upper: float | None = None
    upper_lowest: int | None = None
    upper_low_blob: float | None = None
    if upper_valid.any():
        rows_u = upper_rows[upper_valid].astype(np.float64)
        y_upper = float(rows_u.mean())
        upper_lowest = int(rows_u.max())
        # Centroid of the lowest connected component inside the band.
        sub_in_band = sub & band_upper[None, :]
        n_lab, _lab, st, cent = cv2.connectedComponentsWithStats(
            sub_in_band.astype(np.uint8), connectivity=8)
        if n_lab > 1:
            bottoms = st[1:, cv2.CC_STAT_TOP] + st[1:, cv2.CC_STAT_HEIGHT] - 1
            idx_low = int(np.argmax(bottoms)) + 1
            upper_low_blob = float(cent[idx_low][1])

    return InterfaceResult(
        y_lower=y_lower, y_upper=y_upper,
        lower_rows=lower_rows, lower_valid=lower_valid,
        upper_rows=upper_rows, upper_valid=upper_valid,
        band_lower=(x_lo_l, x_hi_l), band_upper=(x_lo_u, x_hi_u),
        n_lower=int(lower_valid.sum()), n_upper=int(upper_valid.sum()),
        std_lower=std_lower,
        upper_lowest=upper_lowest, upper_low_blob=upper_low_blob,
        S=S,
        lower_above_T=lower_above, upper_above_T=upper_above,
    )


def detect_from_crop(crop_bgr_or_gray: np.ndarray,
                     params: DetectionParams) -> tuple[InterfaceResult, np.ndarray]:
    """Full pipeline on a crop: grey -> gradient -> masks -> detection.

    Returns ``(result, Gy)`` where ``Gy`` is the masked ``dI/dy`` (before
    the polarity sign; ``result.S`` is the signed version).
    """
    gray = to_gray(crop_bgr_or_gray, params.channel)
    Gy = compute_gradient(gray, params.blur_sigma, params.blur_h)
    Gy = apply_bottom_mask(apply_top_mask(Gy, params.y_top), params.y_bottom)
    return detect_interfaces(Gy, params), Gy


__all__ = [
    "Polarity", "POLARITIES", "polarity_sign", "DetectionParams",
    "InterfaceResult", "band_columns", "detect_interfaces", "detect_from_crop",
]
