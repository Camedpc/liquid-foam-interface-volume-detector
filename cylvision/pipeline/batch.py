"""Batch processing: video -> one row per frame (levels, volumes, spreads).

For every analysed frame the crop band ``frame[:, x_left:x_right+1]`` is
run through :func:`cylvision.detection.detect_from_crop`; the two mean rows
are converted to volumes with the calibration model::

    V_lower = model_fn(y_lower)      liquid volume        (below the lower interface)
    V_total = model_fn(y_upper)      liquid + foam volume (below the upper interface)
    V_foam  = V_total - V_lower

and the empirical (method) uncertainty of each interface is the per-column
spread of the detection (:func:`cylvision.uncertainty.empirical.method_uncertainty`).

Rows and time axis
------------------
``process_video`` emits ONE row per frame index of ``[start, end]``
(``end`` inclusive), whatever ``frame_step``: frames that are not analysed
(sub-sampling) or that cannot be decoded get a NaN row, so the CSV keeps its
time alignment and ``len(rows) == end - start + 1``. ``t_sec`` is measured
from ``start``::

    t_sec = (frame_idx - start) * time_scale / fps

``time_scale`` is 1 for a normal video. For a video that was pre-subsampled
(one stored frame = ``K`` source frames, e.g. an ``ffmpeg select`` output
that kept the source fps metadata) pass ``time_scale=K`` so that ``t_sec``
is the source time.

Columns (``CSV_COLUMNS``)
-------------------------
``frame_idx, t_sec, y_lower_px, std_lower_px, n_lower, y_upper_px, n_upper,
V_lower_mL, V_total_mL, V_foam_mL, u_method_lower_mL, u_method_upper_mL``.
Every numeric field is NaN when the corresponding interface was not found.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cylvision.calibration.store import Calibration
from cylvision.detection.interfaces import DetectionParams, InterfaceResult, detect_from_crop
from cylvision.io import VideoSource

CSV_COLUMNS: list[str] = [
    "frame_idx", "t_sec", "y_lower_px", "std_lower_px", "n_lower", "y_upper_px", "n_upper",
    "V_lower_mL", "V_total_mL", "V_foam_mL", "u_method_lower_mL", "u_method_upper_mL",
]
_INT_COLUMNS = ("frame_idx", "n_lower", "n_upper")

NAN = float("nan")
ModelFn = Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Single frame
# ---------------------------------------------------------------------------

def _safe_volume(model_fn: ModelFn, y: float | None) -> float:
    """``model_fn(y)`` as a float, NaN when ``y`` is None or the model fails."""
    if y is None or not np.isfinite(y):
        return NAN
    try:
        v = float(np.asarray(model_fn(np.array([float(y)])), dtype=float).ravel()[0])
    except Exception:
        return NAN
    return v if math.isfinite(v) else NAN


def nan_row(frame_idx: int, t_sec: float) -> dict[str, Any]:
    """Row with NaN everywhere except ``frame_idx`` and ``t_sec``."""
    row: dict[str, Any] = {c: NAN for c in CSV_COLUMNS}
    row["frame_idx"] = int(frame_idx)
    row["t_sec"] = float(t_sec)
    row["n_lower"] = 0
    row["n_upper"] = 0
    return row


def crop_frame(frame_bgr: np.ndarray, calib: Calibration) -> np.ndarray:
    """The full-height crop band ``frame[:, x_left:x_right+1]`` (view, not a copy)."""
    x_left, x_right, _, _ = calib.crop()
    W = frame_bgr.shape[1]
    x_left = max(0, min(x_left, W - 1))
    x_right = max(x_left, min(x_right, W - 1))
    return frame_bgr[:, x_left:x_right + 1]


def row_from_result(result: InterfaceResult, model_fn: ModelFn, *,
                    frame_idx: int = -1, t_sec: float = NAN,
                    tol_ml: float = 7.0) -> dict[str, Any]:
    """Build a CSV row from a detection result (volumes + method uncertainties)."""
    from cylvision.uncertainty.empirical import method_uncertainty  # lazy: keeps the import graph light

    row = nan_row(frame_idx, t_sec)
    V_lower = _safe_volume(model_fn, result.y_lower)
    V_total = _safe_volume(model_fn, result.y_upper)
    row.update({
        "y_lower_px": NAN if result.y_lower is None else float(result.y_lower),
        "std_lower_px": NAN if result.std_lower is None else float(result.std_lower),
        "n_lower": int(result.n_lower),
        "y_upper_px": NAN if result.y_upper is None else float(result.y_upper),
        "n_upper": int(result.n_upper),
        "V_lower_mL": V_lower,
        "V_total_mL": V_total,
        "V_foam_mL": (V_total - V_lower) if (math.isfinite(V_lower) and math.isfinite(V_total)) else NAN,
    })
    u_low = method_uncertainty(result.lower_rows, result.lower_valid, result.y_lower, model_fn, tol_ml=tol_ml)
    u_up = method_uncertainty(result.upper_rows, result.upper_valid, result.y_upper, model_fn, tol_ml=tol_ml)
    row["u_method_lower_mL"] = float(u_low["u_ml"])
    row["u_method_upper_mL"] = float(u_up["u_ml"])
    return row


def process_frame(frame_bgr: np.ndarray, calib: Calibration, params: DetectionParams,
                  model_fn: ModelFn, *, tol_ml: float = 7.0,
                  frame_idx: int = -1, t_sec: float = NAN) -> dict[str, Any]:
    """Crop, detect, convert to volumes. Returns one CSV row (see ``CSV_COLUMNS``)."""
    crop = crop_frame(frame_bgr, calib)
    result, _Gy = detect_from_crop(crop, params)
    return row_from_result(result, model_fn, frame_idx=frame_idx, t_sec=t_sec, tol_ml=tol_ml)


# ---------------------------------------------------------------------------
# Whole video
# ---------------------------------------------------------------------------

def _progress_line(done: int, total: int, row: dict[str, Any]) -> str:
    pct = 100.0 * done / max(1, total)
    V_l, V_t, V_f = row["V_lower_mL"], row["V_total_mL"], row["V_foam_mL"]

    def f(v: float) -> str:
        return f"{v:7.1f}" if math.isfinite(v) else "    n/a"

    return (f"  {pct:5.1f}%  frame {row['frame_idx']:>6d}  t = {row['t_sec']:8.2f} s"
            f"  V_lower ={f(V_l)} mL  V_total ={f(V_t)} mL  V_foam ={f(V_f)} mL")


def process_video(video: VideoSource, calib: Calibration, params: DetectionParams,
                  model_fn: ModelFn, *, start: int = 0, end: int | None = None,
                  frame_step: int = 1, progress: bool = True,
                  on_row: Callable[[dict[str, Any]], None] | None = None,
                  tol_ml: float = 7.0, time_scale: float = 1.0) -> list[dict[str, Any]]:
    """Process frames ``start .. end`` (inclusive), one in ``frame_step``.

    Returns ``end - start + 1`` rows in frame order; non-analysed frames
    (sub-sampling, decode failures) are NaN rows. ``on_row`` is called with
    every ANALYSED row as soon as it is available. ``progress`` prints a
    plain text line roughly every 2 % of the analysed frames.
    """
    n_frames = int(video.n_frames)
    start = max(0, int(start))
    end = n_frames - 1 if end is None else min(int(end), n_frames - 1)
    if end < start:
        raise ValueError(f"empty frame range: start={start} > end={end}")
    frame_step = max(1, int(frame_step))
    fps = float(video.fps) if video.fps and video.fps > 0 else 0.0
    scale = float(time_scale)

    def t_of(idx: int) -> float:
        return (idx - start) * scale / fps if fps > 0 else float(idx - start)

    n_total = end - start + 1
    rows: list[dict[str, Any]] = [nan_row(idx, t_of(idx)) for idx in range(start, end + 1)]
    n_expected = len(range(start, end + 1, frame_step))
    print_every = max(1, int(round(n_expected * 0.02)))
    n_done = 0
    n_ok = 0
    if progress:
        print(f"Batch: frames {start}..{end} step {frame_step} "
              f"({n_expected} frames to analyse out of {n_total})", flush=True)

    for idx, frame in video.frames(start, end + 1, frame_step):
        row = process_frame(frame, calib, params, model_fn, tol_ml=tol_ml,
                            frame_idx=idx, t_sec=t_of(idx))
        rows[idx - start] = row
        n_done += 1
        if math.isfinite(row["V_lower_mL"]):
            n_ok += 1
        if on_row is not None:
            on_row(row)
        if progress and (n_done % print_every == 0 or n_done == n_expected):
            print(_progress_line(n_done, n_expected, row), flush=True)

    if progress:
        n_missing = n_expected - n_done
        msg = f"Batch done: {n_done} frames analysed, {n_ok} with a lower interface"
        if n_missing:
            msg += f", {n_missing} frame(s) could not be decoded (NaN rows)"
        print(msg, flush=True)
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(rows: Sequence[dict[str, Any]], path: Path | str,
              columns: Sequence[str] = CSV_COLUMNS) -> Path:
    """Write ``rows`` as CSV (NaN written as ``nan``). Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt_cell(r.get(c, NAN)) for c in columns})
    return path


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "nan"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if not math.isfinite(f) else repr(f)


def read_csv(path: Path | str) -> list[dict[str, Any]]:
    """Read a CSV written by :func:`write_csv` back into typed rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for k, v in rec.items():
                if v is None or v == "" or v.lower() == "nan":
                    row[k] = 0 if k in _INT_COLUMNS else NAN
                    continue
                try:
                    row[k] = int(v) if k in _INT_COLUMNS else float(v)
                except ValueError:
                    try:
                        row[k] = int(float(v)) if k in _INT_COLUMNS else v
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Run-level summary for ``meta.json`` (counts, ranges, median spreads)."""
    from cylvision.uncertainty.empirical import summarize_method_uncertainty

    def col(key: str) -> np.ndarray:
        return np.array([float(r.get(key, NAN)) for r in rows], dtype=float)

    def rng(a: np.ndarray) -> dict[str, float] | None:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        return {"min": float(a.min()), "max": float(a.max()), "last": float(a[-1])}

    V_l = col("V_lower_mL")
    return {
        "n_rows": int(len(rows)),
        "n_lower_found": int(np.isfinite(V_l).sum()),
        "n_upper_found": int(np.isfinite(col("V_total_mL")).sum()),
        "V_lower_mL": rng(V_l),
        "V_total_mL": rng(col("V_total_mL")),
        "V_foam_mL": rng(col("V_foam_mL")),
        "u_method_lower_mL": summarize_method_uncertainty(rows, "u_method_lower_mL"),
        "u_method_upper_mL": summarize_method_uncertainty(rows, "u_method_upper_mL"),
    }


__all__ = [
    "CSV_COLUMNS", "NAN", "nan_row", "crop_frame", "row_from_result", "process_frame",
    "process_video", "write_csv", "read_csv", "summarize_rows",
]
