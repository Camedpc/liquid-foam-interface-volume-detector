"""Absorbance over a whole video: profiles ``A(t, z)``, foam tops, mass balance, CSV rows.

Per analysed frame the crop band goes through the gradient detector for
the liquid / foam row (:func:`cylvision.detection.detect_from_crop`) and
through :func:`cylvision.absorbance.analyse_crop` for the absorbance map,
the profile ``A(z)`` and the foam top. The profiles are kept in memory
(``n_frames x H`` floats) so that the mass balance can be closed once
``V_inf`` is known (``V_inf = max V_beer`` over the run when not given) and
the foam top can be re-evaluated for several ``k`` without touching the
video again.

Time axis: ``t_sec = (frame_idx * time_scale - t0_frame) / fps`` where
``t0_frame`` is a SOURCE frame index (the start of the pour) and
``time_scale`` the number of source frames per video frame for a
pre-subsampled video (1 otherwise).

Columns (``ABS_CSV_COLUMNS``)
-----------------------------
``frame_idx, t_sec, y_liquid_px, y_foam_top_px, V_beer_mL, V_foam_mL,
A_max, threshold, integral_A_cm, gamma`` -- NaN when not measured.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cylvision.calibration.store import Calibration
from cylvision.detection.interfaces import DetectionParams, detect_from_crop
from cylvision.pipeline.batch import crop_frame

from .beer_lambert import (
    NAN,
    AbsorbanceParams,
    PreparedReference,
    analyse_crop,
    foam_top_from_profile,
    liquid_content_profile,
    mass_balance_gamma,
)

ModelFn = Callable[[Any], Any]

ABS_CSV_COLUMNS: list[str] = [
    "frame_idx", "t_sec", "y_liquid_px", "y_foam_top_px", "V_beer_mL", "V_foam_mL",
    "A_max", "threshold", "integral_A_cm", "gamma",
]
DEFAULT_KS: tuple[int, ...] = tuple(range(2, 11))
"""``k`` values of the foam-top sweep. ``k = 1`` is degenerate (``T = A_max``: the first row below
the maximum is right above the liquid), so the sweep starts at 2."""


def safe_volume(model_fn: ModelFn, y: float | None) -> float:
    """``model_fn(y)`` as a float, NaN when ``y`` is None / NaN or the model fails."""
    if y is None or not np.isfinite(y):
        return NAN
    try:
        v = float(np.asarray(model_fn(np.array([float(y)])), dtype=float).ravel()[0])
    except Exception:
        return NAN
    return v if math.isfinite(v) else NAN


@dataclass
class AbsorbanceRun:
    """Everything the batch measured (arrays are indexed by analysed frame)."""

    frame_idx: np.ndarray          # (n,) int
    t_sec: np.ndarray              # (n,) float
    A: np.ndarray                  # (n, H) float32 profiles A(z)
    y_liquid: np.ndarray           # (n,) float, NaN when the gradient detector failed
    y_foam_top: np.ndarray         # (n,) float, NaN when the threshold found nothing
    A_max: np.ndarray              # (n,)
    threshold: np.ndarray          # (n,)
    V_beer: np.ndarray             # (n,) mL
    V_foam: np.ndarray             # (n,) mL, V(y_foam_top) - V_beer
    integral_cm: np.ndarray        # (n,) cm
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(0))       # filled by finalize_mass_balance
    c: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))      # (n, H) liquid fraction
    V_inf_ml: float = NAN
    z: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    @property
    def n(self) -> int:
        return int(self.frame_idx.size)

    def rows(self) -> list[dict[str, Any]]:
        """CSV rows (``ABS_CSV_COLUMNS``)."""
        out = []
        for i in range(self.n):
            out.append({
                "frame_idx": int(self.frame_idx[i]), "t_sec": float(self.t_sec[i]),
                "y_liquid_px": float(self.y_liquid[i]), "y_foam_top_px": float(self.y_foam_top[i]),
                "V_beer_mL": float(self.V_beer[i]), "V_foam_mL": float(self.V_foam[i]),
                "A_max": float(self.A_max[i]), "threshold": float(self.threshold[i]),
                "integral_A_cm": float(self.integral_cm[i]),
                "gamma": float(self.gamma[i]) if self.gamma.size else NAN,
            })
        return out

    def to_npz_dict(self) -> dict[str, Any]:
        return {
            "A": self.A.astype(np.float32), "c": self.c.astype(np.float32),
            "t": self.t_sec.astype(np.float64), "frame_idx": self.frame_idx.astype(np.int32),
            "z": self.z.astype(np.int32), "y_liquid": self.y_liquid.astype(np.float64),
            "y_foam_top": self.y_foam_top.astype(np.float64), "gamma": self.gamma.astype(np.float64),
            "V_beer": self.V_beer.astype(np.float64), "V_foam": self.V_foam.astype(np.float64),
            "V_inf_ml": float(self.V_inf_ml),
        }


def process_video_absorbance(video: Any, calib: Calibration, det_params: DetectionParams,
                             abs_params: AbsorbanceParams, ref: PreparedReference, model_fn: ModelFn,
                             *, start: int = 0, end: int | None = None, frame_step: int = 1,
                             t0_frame: float | None = None, time_scale: float = 1.0,
                             progress: bool = True,
                             keep_maps: Sequence[int] = ()) -> tuple[AbsorbanceRun, dict[int, Any]]:
    """Analyse frames ``start .. end`` (inclusive) one in ``frame_step``.

    Returns ``(run, maps)`` where ``maps`` holds the full
    :class:`AbsorbanceResult` of the frames listed in ``keep_maps`` (for
    the measurement figure). ``gamma`` / ``c`` are left empty: call
    :func:`finalize_mass_balance` next.
    """
    n_frames = int(video.n_frames)
    start = max(0, int(start))
    end = n_frames - 1 if end is None else min(int(end), n_frames - 1)
    if end < start:
        raise ValueError(f"empty frame range: start={start} > end={end}")
    fps = float(video.fps) if video.fps and video.fps > 0 else 0.0
    scale = float(time_scale)
    t0 = float(start) * scale if t0_frame is None else float(t0_frame)

    def t_of(idx: int) -> float:
        return (idx * scale - t0) / fps if fps > 0 else float(idx * scale - t0)

    H = ref.I0.shape[0]
    idxs: list[int] = []
    profiles: list[np.ndarray] = []
    cols: dict[str, list[float]] = {k: [] for k in ("y_liquid", "y_foam_top", "A_max", "threshold",
                                                     "V_beer", "V_foam", "integral_cm")}
    maps: dict[int, Any] = {}
    keep = {int(k) for k in keep_maps}
    n_expected = len(range(start, end + 1, max(1, frame_step)))
    print_every = max(1, int(round(n_expected * 0.05)))
    if progress:
        print(f"Absorbance: frames {start}..{end} step {frame_step} ({n_expected} frames)", flush=True)
    for n_done, (idx, frame) in enumerate(video.frames(start, end + 1, max(1, frame_step)), start=1):
        crop = crop_frame(frame, calib)
        det, _Gy = detect_from_crop(crop, det_params)
        y_liq = det.y_lower
        V_beer = safe_volume(model_fn, y_liq)
        res = analyse_crop(crop, ref, abs_params, y_liq, V_beer_ml=V_beer)
        V_top = safe_volume(model_fn, None if res.y_foam_top is None else float(res.y_foam_top))
        idxs.append(idx)
        profiles.append(res.A_z if res.A_z.shape[0] == H else np.resize(res.A_z, H))
        cols["y_liquid"].append(NAN if y_liq is None else float(y_liq))
        cols["y_foam_top"].append(NAN if res.y_foam_top is None else float(res.y_foam_top))
        cols["A_max"].append(float(res.A_max))
        cols["threshold"].append(float(res.threshold))
        cols["V_beer"].append(V_beer)
        cols["V_foam"].append(V_top - V_beer if math.isfinite(V_top) and math.isfinite(V_beer) else NAN)
        cols["integral_cm"].append(float(res.integral_cm))
        if idx in keep:
            maps[idx] = res
        if progress and (n_done % print_every == 0 or n_done == n_expected):
            print(f"  {100.0 * n_done / n_expected:5.1f}%  frame {idx:>6d}  t = {t_of(idx):8.1f} s"
                  f"  y_liquid = {cols['y_liquid'][-1]:7.1f}  foam top = {cols['y_foam_top'][-1]:6.0f}"
                  f"  V_foam = {cols['V_foam'][-1]:6.1f} mL", flush=True)
    if not idxs:
        raise OSError("no frame could be read")
    fi = np.asarray(idxs, dtype=np.int64)
    run = AbsorbanceRun(
        frame_idx=fi, t_sec=np.array([t_of(int(i)) for i in fi], dtype=float),
        A=np.stack(profiles).astype(np.float32),
        y_liquid=np.asarray(cols["y_liquid"]), y_foam_top=np.asarray(cols["y_foam_top"]),
        A_max=np.asarray(cols["A_max"]), threshold=np.asarray(cols["threshold"]),
        V_beer=np.asarray(cols["V_beer"]), V_foam=np.asarray(cols["V_foam"]),
        integral_cm=np.asarray(cols["integral_cm"]), z=np.arange(H, dtype=np.int32),
    )
    return run, maps


def finalize_mass_balance(run: AbsorbanceRun, V_inf_ml: float | None, section_cm2: float,
                          clip_negative: bool = True) -> AbsorbanceRun:
    """Fill ``gamma`` and ``c`` of a run; ``V_inf_ml=None`` uses ``max V_beer``.

    A negative ``gamma`` (``V_beer > V_inf``) is set to 0 when
    ``clip_negative`` (the frame then has an all-zero liquid content inside
    its foam), otherwise kept and its ``c`` left NaN.
    """
    finite = np.isfinite(run.V_beer)
    V_inf = float(np.max(run.V_beer[finite])) if V_inf_ml is None and finite.any() else float(V_inf_ml or NAN)
    n, H = run.A.shape
    gamma = np.full(n, NAN)
    c = np.full((n, H), np.nan, dtype=np.float32)
    for i in range(n):
        g = mass_balance_gamma(V_inf, float(run.V_beer[i]), float(run.integral_cm[i]), section_cm2)
        if np.isfinite(g) and g < 0:
            if not clip_negative:
                gamma[i] = g
                continue
            g = 0.0
        gamma[i] = g
        yt = None if not np.isfinite(run.y_foam_top[i]) else int(run.y_foam_top[i])
        c[i] = liquid_content_profile(run.A[i], g, yt, float(run.y_liquid[i]))
    run.gamma, run.c, run.V_inf_ml = gamma, c, V_inf
    return run


def foam_tops_for_ks(run: AbsorbanceRun, ks: Sequence[int] = DEFAULT_KS,
                     y_top: int = 0) -> dict[int, np.ndarray]:
    """Foam-top row per frame for each ``k`` (NaN when not found), from the stored profiles."""
    out = {int(k): np.full(run.n, NAN) for k in ks}
    for i in range(run.n):
        yl = float(run.y_liquid[i])
        if not np.isfinite(yl):
            continue
        for k in ks:
            y, _A_max, _T = foam_top_from_profile(run.A[i], yl, float(k), y_top)
            if y is not None:
                out[int(k)][i] = float(y)
    return out


def foam_volume_vs_k(run: AbsorbanceRun, i: int, model_fn: ModelFn, ks: Sequence[int] = DEFAULT_KS,
                     y_top: int = 0) -> dict[int, tuple[float, float]]:
    """``{k: (y_foam_top, V_foam_mL)}`` of the analysed frame ``i`` for each ``k``."""
    out: dict[int, tuple[float, float]] = {}
    yl = float(run.y_liquid[i])
    V_beer = float(run.V_beer[i])
    for k in ks:
        y, _A_max, _T = foam_top_from_profile(run.A[i], yl, float(k), y_top)
        V_top = safe_volume(model_fn, None if y is None else float(y))
        out[int(k)] = (NAN if y is None else float(y), V_top - V_beer if math.isfinite(V_top) else NAN)
    return out


def px_per_cm_from_calibration(calib: Calibration, section_cm2: float) -> float:
    """Vertical scale implied by the calibration: ``S / |dV/dy|`` in px per cm.

    ``|dV/dy|`` is the mean slope between the extreme clicked graduations
    (mL per px); dividing the section (cm^2) by it gives the height of one
    row in cm, hence the px per cm. Using it makes ``S * integral A dz``
    consistent with the volumes read on the scale, so that ``c = gamma A``
    is a true volume fraction (``V_inf - V_beer = S * integral c dz``).
    """
    y = np.asarray(calib.graduations_px_y, dtype=float)
    V = np.asarray(calib.graduations_ml, dtype=float)
    if y.size < 2 or float(np.ptp(y)) <= 0:
        raise ValueError("px_per_cm_from_calibration needs at least two graduations at different rows")
    slope = abs(float(V[np.argmin(y)] - V[np.argmax(y)])) / float(np.ptp(y))   # mL / px
    if slope <= 0 or section_cm2 <= 0:
        raise ValueError("degenerate calibration slope or section")
    return float(section_cm2) / slope


def volume_window(calib: Calibration, H: int, margin_top: int = 60,
                  margin_bottom: int = 80) -> tuple[int, int]:
    """Rows ``(y_min, y_max)`` of the calibrated zone plus small margins (the heat-map window)."""
    grads = np.asarray(calib.graduations_px_y, dtype=float)
    y_min = max(0, int(math.floor(grads.min())) - int(margin_top))
    y_max = min(int(H), int(math.ceil(grads.max())) + int(margin_bottom))
    return y_min, y_max


__all__ = [
    "ABS_CSV_COLUMNS", "DEFAULT_KS", "AbsorbanceRun", "safe_volume", "process_video_absorbance",
    "finalize_mass_balance", "foam_tops_for_ks", "foam_volume_vs_k", "px_per_cm_from_calibration",
    "volume_window",
]
