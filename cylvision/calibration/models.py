"""Calibration models mapping an image row ``y_px`` to a volume ``V_mL``.

Physics
-------
A graduated cylinder standing vertically is filmed by a pinhole camera.
Because the liquid height is (to first order) proportional to the volume,
the graduation marks are equally spaced along the cylinder axis. Their
projection onto the sensor is *not* equally spaced when the camera is
pitched with respect to the axis: a point at height ``z`` projects to

    v = cy - f * (z cos a - D sin a) / (z sin a + D cos a)

which is a Möbius (rational, degree 1/1) transform of ``z``, hence of ``V``.
Its inverse ``V(v)`` is Möbius as well. Three models are therefore fitted
in parallel on the clicked graduations ``(y_px, V_mL)``:

* ``pchip``  -- monotone cubic interpolation (PCHIP). Exact at the clicks,
  follows any imperfection of the printed scale, no smoothing.
* ``poly2``  -- second-degree polynomial. Captures the dominant tilt
  effect and smooths the click noise.
* ``mobius`` -- ``V = (a*y + b) / (c*y + 1)``, the exact pinhole + tilt
  form with 3 effective parameters (``c = 0`` is the untilted case).

``best_model`` picks ``poly2`` when its RMS residual is below a threshold
(smooth and robust to a single bad click), otherwise ``pchip`` (which then
honours the actual scale). ``build_model_fn`` rebuilds the recommended
model from a saved :class:`~cylvision.calibration.store.Calibration`, and
``slope_ml_per_px`` gives the local sensitivity ``|dV/dy|`` used by the
pixel-quantisation term of the uncertainty budget.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from cylvision.calibration.store import Calibration

ModelName = Literal["pchip", "poly2", "mobius"]
MODEL_NAMES: tuple[ModelName, ...] = ("pchip", "poly2", "mobius")

ModelFn = Callable[[ArrayLike], np.ndarray]


def mobius_eval(y: ArrayLike, popt: ArrayLike) -> np.ndarray:
    """Evaluate ``V = (a*y + b) / (c*y + 1)`` for ``popt = (a, b, c)``."""
    a, b, c = (float(p) for p in np.asarray(popt, dtype=float))
    y = np.asarray(y, dtype=float)
    return (a * y + b) / (c * y + 1.0)


def poly2_eval(y: ArrayLike, coef: ArrayLike) -> np.ndarray:
    """Evaluate the degree-2 polynomial ``coef`` (numpy ``polyval`` order)."""
    return np.polyval(np.asarray(coef, dtype=float), np.asarray(y, dtype=float))


def pchip_fn(y_px: ArrayLike, V_ml: ArrayLike) -> ModelFn:
    """Build a monotone PCHIP interpolator ``y_px -> V_mL`` (any click order)."""
    y = np.asarray(y_px, dtype=float)
    V = np.asarray(V_ml, dtype=float)
    order = np.argsort(y)
    interp = PchipInterpolator(y[order], V[order], extrapolate=True)
    return lambda yq: np.asarray(interp(np.asarray(yq, dtype=float)), dtype=float)


def _mobius(y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return (a * y + b) / (c * y + 1.0)


def fit_calibration_models(y_px: ArrayLike, V_ml: ArrayLike) -> dict[str, dict[str, Any]]:
    """Fit the three models on the clicked graduations.

    Parameters
    ----------
    y_px, V_ml
        Rows (px, image convention: y grows downward) and volumes (mL) of
        the clicked graduations, in any order. At least 3 points.

    Returns
    -------
    dict
        ``{"pchip": {"fn"}, "poly2": {"fn", "coef"}, "mobius": {"fn", "popt", "pcov"}}``
        where every ``fn`` maps an array of rows to volumes.
    """
    y = np.asarray(y_px, dtype=float)
    V = np.asarray(V_ml, dtype=float)
    if y.shape != V.shape or y.ndim != 1:
        raise ValueError("y_px and V_ml must be 1-D arrays of equal length")
    if y.size < 3:
        raise ValueError(f"at least 3 graduations are required, got {y.size}")

    pchip = pchip_fn(y, V)

    poly2_coef = np.polyfit(y, V, deg=2)

    def poly2(yq: ArrayLike) -> np.ndarray:
        return poly2_eval(yq, poly2_coef)

    # Initial guess: straight line through the clicks, no tilt.
    slope = (V.max() - V.min()) / (y.min() - y.max() + 1e-9)
    intercept = V.mean() - slope * y.mean()
    p0 = [slope, intercept, 1e-6]
    try:
        popt, pcov = curve_fit(_mobius, y, V, p0=p0, maxfev=20000)
    except Exception:  # singular fit (e.g. collinear clicks): keep the line
        popt = np.array(p0, dtype=float)
        pcov = np.eye(3) * 1e9

    def mobius(yq: ArrayLike) -> np.ndarray:
        return mobius_eval(yq, popt)

    return {
        "pchip": {"fn": pchip},
        "poly2": {"fn": poly2, "coef": np.asarray(poly2_coef, dtype=float)},
        "mobius": {"fn": mobius, "popt": np.asarray(popt, dtype=float),
                   "pcov": np.asarray(pcov, dtype=float)},
    }


def evaluate_residuals(models: Mapping[str, Mapping[str, Any]],
                       y_px: ArrayLike, V_ml: ArrayLike) -> dict[str, np.ndarray]:
    """Residual ``model(y) - V`` (mL) of every model at the calibration points."""
    y = np.asarray(y_px, dtype=float)
    V = np.asarray(V_ml, dtype=float)
    return {name: np.asarray(m["fn"](y), dtype=float) - V for name, m in models.items()}


def rms(residual: ArrayLike) -> float:
    """Root-mean-square of a residual vector."""
    r = np.asarray(residual, dtype=float)
    return float(np.sqrt(np.mean(r ** 2))) if r.size else 0.0


def best_model(residuals: Mapping[str, ArrayLike],
               rms_threshold_ml: float = 0.3) -> ModelName:
    """Heuristic choice of the production model.

    ``poly2`` when its RMS residual is at or below ``rms_threshold_ml``
    (smooth model, robust to one noisy click); otherwise ``pchip``, which
    reproduces the clicks exactly and therefore follows a scale whose
    printed marks are not perfectly regular.
    """
    if rms(residuals["poly2"]) <= rms_threshold_ml:
        return "poly2"
    return "pchip"


def _field(calib: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a Calibration dataclass or a plain (legacy) dict."""
    if isinstance(calib, Mapping):
        return calib.get(key, default)
    return getattr(calib, key, default)


def build_model_fn(calib: "Calibration | Mapping[str, Any]",
                   model: ModelName | None = None) -> ModelFn:
    """Rebuild the ``y_px -> V_mL`` function from a saved calibration.

    Uses ``calib.recommended_model`` unless ``model`` overrides it. ``poly2``
    and ``mobius`` are rebuilt from their stored coefficients; ``pchip`` is
    re-interpolated from the stored clicks (an interpolator cannot be
    serialised, the clicks can). A legacy ``calib.json`` dict is accepted.
    """
    name = model or _field(calib, "recommended_model", "poly2")
    if name == "poly2":
        coef = np.asarray(_field(calib, "poly2_coef"), dtype=float)
        return lambda y: poly2_eval(y, coef)
    if name == "mobius":
        popt = np.asarray(_field(calib, "mobius_popt"), dtype=float)
        return lambda y: mobius_eval(y, popt)
    if name != "pchip":
        raise ValueError(f"unknown calibration model {name!r}")
    return pchip_fn(_field(calib, "graduations_px_y"), _field(calib, "graduations_ml"))


def slope_ml_per_px(model_fn: ModelFn, y_px: float, h: float = 1.0) -> float:
    """Local sensitivity ``|dV/dy|`` (mL per pixel) by central difference.

    ``V`` decreases when ``y`` grows (volume is up, ``y`` points down), so
    the raw derivative is negative; the absolute value is returned because
    only the magnitude matters for the uncertainty budget.
    """
    vp = float(np.asarray(model_fn(np.array([y_px + h])), dtype=float)[0])
    vm = float(np.asarray(model_fn(np.array([y_px - h])), dtype=float)[0])
    return abs(vp - vm) / (2.0 * h)


def invert_model(model_fn: ModelFn, y_min: float, y_max: float,
                 V_targets: ArrayLike, n_samples: int = 4000) -> np.ndarray:
    """Invert ``V(y)`` into ``y(V)`` by dense sampling + linear interpolation.

    Works for any of the three models as long as ``V(y)`` is monotone on
    ``[y_min, y_max]``, which the physics guarantees inside the image.
    """
    y_dense = np.linspace(float(y_min), float(y_max), int(n_samples))
    V_dense = np.asarray(model_fn(y_dense), dtype=float)
    if V_dense[0] > V_dense[-1]:  # np.interp needs increasing abscissae
        V_dense, y_dense = V_dense[::-1], y_dense[::-1]
    return np.interp(np.asarray(V_targets, dtype=float), V_dense, y_dense)
