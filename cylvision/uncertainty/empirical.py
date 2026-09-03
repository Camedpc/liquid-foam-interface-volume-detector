"""Empirical (method) uncertainty of a detected interface.

The interface detector returns, for every column of the averaging band, the
row of the strongest signed gradient; the reported level ``y_mean`` is the
mean of those rows over the columns that passed the threshold.  The spread of
the per-column rows is a direct, per-frame measurement of how well the
detection pins the interface: an irregular foam/liquid boundary, bubbles or
a meniscus all widen it.

Procedure (per frame, per interface)
------------------------------------
1. Take the per-column rows ``rows_px`` and their validity flags ``valid``
   (above threshold and inside the band).
2. Convert every valid row to a volume with the calibration model and keep
   only the columns whose volume lies within ``tol_ml`` of ``V(y_mean)``
   (default 7 mL): columns that jumped to a bubble or to the other interface
   are outliers of the *method*, not a property of the interface, and would
   otherwise dominate the range.
3. ``y_up = min(rows kept)`` (highest point in the image) and
   ``y_lo = max(rows kept)`` (lowest point).
4. ``range_ml = |V(y_up) - V(y_lo)|`` and the standard uncertainty is that of
   a uniform distribution over the range, ``u = range / (2 sqrt(3))``.

Over a run the per-frame values are summarised by their median (robust to
the few frames where the foam collapses or a bubble crosses the band); the
budget uses that summary as ``u_method``.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

_SQRT12 = 2.0 * np.sqrt(3.0)


def _eval(model_fn: Callable[..., Any], y: float | np.ndarray) -> np.ndarray:
    return np.asarray(model_fn(np.atleast_1d(np.asarray(y, dtype=float))), dtype=float).ravel()


def _empty(y_mean: float | None, V_mean: float) -> dict:
    nan = float("nan")
    return {"y_mean": nan if y_mean is None else float(y_mean), "V_mean": V_mean,
            "y_up": nan, "y_lo": nan, "V_up": nan, "V_lo": nan,
            "range_ml": nan, "u_ml": nan, "n_kept": 0}


def method_uncertainty(rows_px: np.ndarray, valid: np.ndarray, y_mean: float | None,
                       model_fn: Callable[..., Any], *, tol_ml: float = 7.0) -> dict:
    """Per-frame empirical uncertainty from the per-column interface rows.

    Returns ``{"y_mean", "V_mean", "y_up", "y_lo", "V_up", "V_lo", "range_ml",
    "u_ml", "n_kept"}``; NaN values and ``n_kept = 0`` when nothing usable.
    ``y_up`` is the smallest row (highest in the image), ``y_lo`` the largest.
    """
    if y_mean is None or not np.isfinite(y_mean):
        return _empty(None, float("nan"))
    rows = np.asarray(rows_px, dtype=float).ravel()
    ok = np.asarray(valid, dtype=bool).ravel()
    if rows.size != ok.size:
        raise ValueError("rows_px and valid must have the same length")
    V_mean = float(_eval(model_fn, float(y_mean))[0])
    if not ok.any():
        return _empty(y_mean, V_mean)
    y = rows[ok]
    V = _eval(model_fn, y)
    keep = np.isfinite(V) & (np.abs(V - V_mean) <= float(tol_ml))
    if not keep.any():
        return _empty(y_mean, V_mean)
    y_kept = y[keep]
    y_up, y_lo = float(y_kept.min()), float(y_kept.max())
    V_up, V_lo = (float(v) for v in _eval(model_fn, np.array([y_up, y_lo])))
    rng = abs(V_up - V_lo)
    return {"y_mean": float(y_mean), "V_mean": V_mean, "y_up": y_up, "y_lo": y_lo,
            "V_up": V_up, "V_lo": V_lo, "range_ml": float(rng), "u_ml": float(rng / _SQRT12),
            "n_kept": int(keep.sum())}


def method_uncertainty_from_result(result: Any, model_fn: Callable[..., Any], *,
                                   which: str = "lower", tol_ml: float = 7.0) -> dict:
    """Convenience wrapper on an ``InterfaceResult`` (``which`` = ``"lower"`` or ``"upper"``)."""
    if which not in ("lower", "upper"):
        raise ValueError("which must be 'lower' or 'upper'")
    rows = getattr(result, f"{which}_rows")
    valid = getattr(result, f"{which}_valid")
    y_mean = getattr(result, f"y_{which}")
    return method_uncertainty(rows, valid, y_mean, model_fn, tol_ml=tol_ml)


def summarize_method_uncertainty(rows: Iterable[dict], key: str = "u_method_lower_mL") -> dict:
    """Run-level summary (``n``, ``mean``, ``median``, ``p95``, ``max``) of a per-frame column."""
    vals = np.array([float(r[key]) for r in rows if r.get(key) is not None], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "median": nan, "p95": nan, "max": nan}
    return {"n": int(vals.size), "mean": float(vals.mean()), "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)), "max": float(vals.max())}
