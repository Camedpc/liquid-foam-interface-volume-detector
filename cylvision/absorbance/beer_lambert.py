"""Effective optical absorbance of the foam, its profile, the foam top and the liquid content.

Physics / algorithm
-------------------
On a back-lit setup the beer transmits the light and the foam scatters it.
With a per-pixel reference ``I0(x, y)`` of the empty cylinder
(:mod:`cylvision.absorbance.reference`), each live frame gives an
**effective absorbance**::

    A(x, y) = -ln( max(I, 1) / max(I0, 1) )

This is Beer-Lambert in form only: the attenuation in the foam is multiple
scattering by the films and Plateau borders, not absorption, so ``A`` is
an effective extinction ``mu_eff * d`` (``d`` = optical path through the
cylinder), monotone in the amount of liquid films on the path but not a
calibrated absorption. ``I`` is clamped to 1 to avoid ``ln 0``; ``A`` is
NOT clamped from below: a negative value means the live frame is brighter
than the reference, which flags a drifting back-light.

1. ``absorbance_profile``: ``A(z) = mean_{|x - cx| <= r_dens} A(x, z)``.
   The band is narrower than the cylinder so the dark glass walls (an
   absorbance of their own) stay out of the average.
2. ``foam_top_from_profile``: the foam / air interface by a threshold on
   the profile. ``A_max = max A(z)`` over ``[y_top, y_liquid)``, threshold
   ``T = A_max / k``; scanning UPWARDS from the liquid / foam interface,
   the first row with ``A(z) < T`` is the foam top. Robust when the top of
   the foam is diffuse (breathing, oscillating surface), where a gradient
   threshold hesitates; scanning bottom-up returns the LOWEST exit of the
   foam, so foam left on the glass above the true top is ignored (the price
   is that a hole with ``A < T`` inside the foam stops the scan).
3. ``mass_balance_gamma``: the liquid inside the foam is the beer missing
   from the bottom, ``V_inf - V_beer(t)`` where ``V_inf`` is the final
   beer volume once the foam is gone. Writing the liquid content of the
   foam as ``phi(z) = gamma * A(z)`` (volume of liquid per volume of
   foam), integrating over the foam column of section ``S`` gives::

        gamma(t) = (V_inf - V_beer(t)) / ( S * integral_foam A(z) dz )

   with ``S`` in cm^2 and ``dz = 1 / px_per_cm`` cm: ``S * integral A dz``
   is in cm^3 = mL, so ``gamma`` and ``phi`` are dimensionless. ``phi(z)``
   is a liquid volume fraction (mL of liquid per mL of foam); the original
   analysis multiplied it by ``rho_beer`` to quote it in g/mL.
4. ``liquid_content_profile``: ``c(z) = gamma A(z)`` inside the foam
   (``[y_foam_top, y_liquid)``), NaN outside.

The liquid / foam interface itself (``y_liquid``) is NOT measured here: it
comes from the gradient detector (:mod:`cylvision.detection`), which is
sharp on that interface; the absorbance takes over for the foam top and
for what happens inside the foam.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from .reference import Reference, dead_pixel_masks, prepare_intensity

RHO_BEER_G_ML: float = 1.016
"""Density of the beer of the original experiment (254 g / 250 mL)."""

D_INT_CM: float = 6.7
"""Inner diameter of the 1000 mL cylinder of the original experiment (cm)."""

S_CM2: float = float(math.pi * D_INT_CM ** 2 / 4.0)
"""Inner section of that cylinder (cm^2), ``pi D^2 / 4``."""

NAN = float("nan")


@dataclass
class AbsorbanceParams:
    """Every knob of the absorbance analysis (json-friendly)."""

    channel: str = "G"           # plane compared with the reference ("gray", "B", "G", "R")
    light_blur: float = 1.0      # Gaussian sigma (px) applied to I and I0 alike
    r_dens: int = 64             # half-width of the profile band |x - cx| <= r_dens (px)
    foam_k: float = 5.0          # foam-top threshold T = A_max / k
    dead_thresh: float = 178.0   # reference pixels darker than this are dead
    dead_dilate: int = 2         # dilation radius of the dead mask (px)
    px_per_cm: float = 20.0      # vertical scale of the image (px / cm)
    section_cm2: float = S_CM2   # inner section of the cylinder (cm^2)
    y_top: int = 0               # rows above are ignored by the foam-top search
    cx: int | None = None        # axis column inside the crop, None = W // 2

    @property
    def r_fill(self) -> int:
        """Radius of the dead-pixel fill window (always wider than the dilation)."""
        return max(5, int(self.dead_dilate) + 3)

    def resolved_cx(self, W: int) -> int:
        cx = W // 2 if self.cx is None else int(self.cx)
        return max(0, min(W - 1, cx))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("light_blur", "foam_k", "dead_thresh", "px_per_cm", "section_cm2"):
            d[k] = float(d[k])
        for k in ("r_dens", "dead_dilate", "y_top"):
            d[k] = int(d[k])
        d["cx"] = None if d["cx"] is None else int(d["cx"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AbsorbanceParams":
        """Build from a dict; unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PreparedReference:
    """``I0`` ready for :func:`absorbance_map`, plus the dead-pixel masks."""

    I0: np.ndarray                 # (H, W) float32, filled + blurred
    valid: np.ndarray              # (H, W) bool: True where the reference is a real measurement
    dead_direct: np.ndarray        # (H, W) bool: ref < dead_thresh
    raw: np.ndarray                # (H, W) float32, the reference plane before fill / blur
    params: AbsorbanceParams

    @property
    def n_dead(self) -> int:
        return int((~self.valid).sum())


def prepare_reference(ref: Reference, params: AbsorbanceParams) -> PreparedReference:
    """Channel, dead masks and fill + blur of the reference, once per run."""
    raw = ref.intensity(params.channel)
    direct, dilated = dead_pixel_masks(raw, params.dead_thresh, params.dead_dilate)
    valid = ~dilated
    I0 = prepare_intensity(ref.mean_bgr, params.channel, valid=valid,
                           r_fill=params.r_fill, light_blur=params.light_blur)
    return PreparedReference(I0=I0, valid=valid, dead_direct=direct, raw=raw, params=params)


def absorbance_map(intensity: np.ndarray, I0: np.ndarray | float) -> np.ndarray:
    """``A(x, y) = -ln(max(I, 1) / max(I0, 1))`` as ``float32``; ``I0`` may be a scalar."""
    I = np.maximum(np.asarray(intensity, dtype=np.float32), 1.0)
    if np.isscalar(I0):
        return (-np.log(I / max(float(I0), 1.0))).astype(np.float32)
    I0a = np.maximum(np.asarray(I0, dtype=np.float32), 1.0)
    if I0a.shape != I.shape:
        raise ValueError(f"I0 shape {I0a.shape} != I shape {I.shape}")
    return (-np.log(I / I0a)).astype(np.float32)


def absorbance_profile(A: np.ndarray, cx: int, r_dens: int) -> np.ndarray:
    """``A(z)``: mean of ``A(x, z)`` over the columns ``|x - cx| <= r_dens``."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    H, W = A.shape
    x_lo = max(0, int(cx) - int(r_dens))
    x_hi = min(W - 1, int(cx) + int(r_dens))
    if x_hi < x_lo:
        return np.full(H, np.nan, dtype=np.float32)
    return A[:, x_lo:x_hi + 1].mean(axis=1).astype(np.float32)


def foam_top_from_profile(A_z: np.ndarray, y_liquid: float | None, k: float,
                          y_top: int = 0) -> tuple[int | None, float, float]:
    """Foam / air row by threshold on ``A(z)``: ``(y_foam_top, A_max, T)``.

    ``A_max`` is the maximum of the profile over ``[y_top, y_liquid)``,
    ``T = A_max / k``. Scanning upwards from ``round(y_liquid) - 1``, the
    first row with ``A(z) < T`` (NaN counts as below) is the foam top.
    ``y_foam_top`` is ``None`` when the liquid row is unknown, the search
    region is empty or all-NaN, ``A_max <= 0``, or no row drops below ``T``.
    """
    if y_liquid is None or not np.isfinite(y_liquid) or k <= 0:
        return None, 0.0, 0.0
    hi = int(min(len(A_z), int(round(float(y_liquid)))))
    lo = int(max(0, int(y_top)))
    if hi <= lo:
        return None, 0.0, 0.0
    region = np.asarray(A_z[lo:hi], dtype=np.float64)
    finite = np.isfinite(region)
    if not finite.any():
        return None, 0.0, 0.0
    A_max = float(region[finite].max())
    if A_max <= 0:
        return None, A_max, 0.0
    T = A_max / float(k)
    below = (region < T) | ~finite
    rev = below[::-1]
    if not rev.any():
        return None, A_max, T
    return hi - 1 - int(np.argmax(rev)), A_max, T


def foam_integral_cm(A_z: np.ndarray, y_foam_top: int | None, y_liquid: float | None,
                     px_per_cm: float) -> float:
    """``integral_foam A dz`` in cm over the rows ``[y_foam_top, round(y_liquid))``; NaN counts as 0."""
    if y_foam_top is None or y_liquid is None or not np.isfinite(y_liquid) or px_per_cm <= 0:
        return 0.0
    yl = int(round(float(y_liquid)))
    yt = int(y_foam_top)
    if yt >= yl:
        return 0.0
    sl = np.asarray(A_z[max(0, yt):min(len(A_z), yl)], dtype=np.float64)
    return float(np.nansum(sl)) / float(px_per_cm)


def mass_balance_gamma(V_inf_ml: float, V_beer_ml: float, integral_cm: float,
                       section_cm2: float = S_CM2) -> float:
    """``gamma = (V_inf - V_beer) / (S * integral A dz)``; NaN when undefined.

    ``gamma`` converts the absorbance profile into a liquid volume fraction
    ``phi(z) = gamma A(z)``. It is negative when ``V_beer > V_inf`` (a
    transient overshoot of the liquid reading), which callers may treat as
    an invalid frame.
    """
    if not (np.isfinite(V_inf_ml) and np.isfinite(V_beer_ml)) or integral_cm <= 0 or section_cm2 <= 0:
        return NAN
    return float((V_inf_ml - V_beer_ml) / (section_cm2 * integral_cm))


def liquid_content_profile(A_z: np.ndarray, gamma: float, y_foam_top: int | None,
                           y_liquid: float | None) -> np.ndarray:
    """``c(z) = gamma A(z)`` on the foam rows ``[y_foam_top, round(y_liquid))``, NaN elsewhere."""
    c = np.full(len(A_z), np.nan, dtype=np.float32)
    if y_foam_top is None or y_liquid is None or not np.isfinite(y_liquid) or not np.isfinite(gamma):
        return c
    yl = min(len(A_z), int(round(float(y_liquid))))
    yt = max(0, int(y_foam_top))
    if yt >= yl:
        return c
    a = np.nan_to_num(np.asarray(A_z[yt:yl], dtype=np.float32), nan=0.0)
    c[yt:yl] = float(gamma) * a
    return c


@dataclass
class AbsorbanceResult:
    """Output of :func:`analyse_crop` for one frame. Rows are crop == frame rows."""

    A: np.ndarray                  # (H, W) absorbance map
    A_z: np.ndarray                # (H,) radial profile
    intensity: np.ndarray          # (H, W) prepared live intensity (filled + blurred)
    y_liquid: float | None         # liquid / foam row (input, gradient detector)
    y_foam_top: int | None         # foam / air row by absorbance threshold
    A_max: float                   # max of A(z) between y_top and y_liquid
    threshold: float               # A_max / k
    integral_cm: float             # integral of A over the foam, cm
    gamma: float                   # mass-balance factor (NaN when V_inf / V_beer unknown)
    c_z: np.ndarray                # (H,) liquid volume fraction, NaN outside the foam

    @property
    def found_foam_top(self) -> bool:
        return self.y_foam_top is not None

    def to_dict(self) -> dict[str, Any]:
        """Scalar summary (no arrays), json-friendly."""
        return {
            "y_liquid": self.y_liquid, "y_foam_top": self.y_foam_top, "A_max": self.A_max,
            "threshold": self.threshold, "integral_cm": self.integral_cm, "gamma": self.gamma,
        }


def analyse_crop(crop_bgr: np.ndarray, ref: PreparedReference, params: AbsorbanceParams,
                 y_liquid: float | None, *, V_beer_ml: float = NAN,
                 V_inf_ml: float = NAN) -> AbsorbanceResult:
    """Full absorbance pipeline on one crop: ``I`` -> ``A(x, y)`` -> ``A(z)`` -> foam top -> ``gamma`` -> ``c(z)``.

    ``y_liquid`` is the liquid / foam row from the gradient detector.
    ``gamma`` and ``c(z)`` need ``V_beer_ml`` (the liquid volume of this
    frame) and ``V_inf_ml`` (the final liquid volume); pass NaN to skip them
    (``gamma`` is NaN, ``c_z`` all NaN) and call :func:`mass_balance_gamma`
    / :func:`liquid_content_profile` later, once ``V_inf`` is known.
    """
    if crop_bgr.shape[:2] != ref.I0.shape:
        raise ValueError(f"crop shape {crop_bgr.shape[:2]} != reference shape {ref.I0.shape}")
    inten = prepare_intensity(crop_bgr, params.channel, valid=ref.valid,
                              r_fill=params.r_fill, light_blur=params.light_blur)
    A = absorbance_map(inten, ref.I0)
    cx = params.resolved_cx(A.shape[1])
    A_z = absorbance_profile(A, cx, params.r_dens)
    y_top_row, A_max, T = foam_top_from_profile(A_z, y_liquid, params.foam_k, params.y_top)
    integ = foam_integral_cm(A_z, y_top_row, y_liquid, params.px_per_cm)
    gamma = mass_balance_gamma(V_inf_ml, V_beer_ml, integ, params.section_cm2)
    c_z = liquid_content_profile(A_z, gamma, y_top_row, y_liquid)
    return AbsorbanceResult(A=A, A_z=A_z, intensity=inten, y_liquid=y_liquid, y_foam_top=y_top_row,
                            A_max=A_max, threshold=T, integral_cm=integ, gamma=gamma, c_z=c_z)


__all__ = [
    "RHO_BEER_G_ML", "D_INT_CM", "S_CM2", "NAN", "AbsorbanceParams", "PreparedReference",
    "prepare_reference", "absorbance_map", "absorbance_profile", "foam_top_from_profile",
    "foam_integral_cm", "mass_balance_gamma", "liquid_content_profile", "AbsorbanceResult",
    "analyse_crop",
]
