#!/usr/bin/env python
"""Run the batch: calibration + parameters -> levels.csv, meta.json, figure.png.

Reads ``calib.json`` and ``params.json`` from the run directory, processes
the frames ``--start .. --end`` one in ``--step`` (default: the
``frame_step`` saved by the tuner) and writes:

* ``levels.csv`` -- one row per frame (NaN rows for non-analysed frames);
* ``meta.json`` -- video, range, step, parameters, timings, summaries;
* ``figure.png`` -- V_lower(t), V_total(t), V_foam(t) with the ±u_total bands of
  the uncertainty budget (skip with ``--no-figure``).

The budget (calibration, pixel, curvature, method) is built when the
calibration carries the cylinder radius (``cylinder.radius_cm``), which the
curvature model needs; otherwise the figure is drawn without bands and a
note says so. The Markdown table of the budget is printed and stored in
``meta.json`` under ``uncertainty_budget``.

Example::

    python scripts/run_batch.py --video pour.mp4 --run-dir runs/pour --start 100 --end 4000 --step 5

``--frame-scale K``: the video is a 1-in-K subsample of the recording that
kept the source fps metadata; ``t_sec`` is multiplied by K so that the time
axis is the source time.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cylvision.calibration import build_model_fn  # noqa: E402
from cylvision.io import VideoSource  # noqa: E402
from cylvision.pipeline import (  # noqa: E402
    CSV_COLUMNS,
    RunDir,
    plot_levels,
    process_video,
    summarize_rows,
    write_csv,
)
from cylvision.uncertainty import (  # noqa: E402
    budget_markdown_table,
    build_budget,
    geometry_from_calibration,
)

BUDGET_VOLUMES: tuple[float, ...] = (100.0, 250.0, 500.0, 750.0, 1000.0)


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="video to process")
    p.add_argument("--run-dir", type=Path, required=True, help="run directory with calib.json + params.json")
    p.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    p.add_argument("--end", type=int, default=None, help="last frame, inclusive (default: last frame)")
    p.add_argument("--step", type=int, default=None,
                   help="analyse one frame in STEP (default: frame_step of params.json)")
    p.add_argument("--tol-ml", type=float, default=7.0,
                   help="outlier guard of the method uncertainty, in mL (default 7)")
    p.add_argument("--frame-scale", type=float, default=1.0,
                   help="source frames per video frame (pre-subsampled video), default 1")
    p.add_argument("--no-figure", action="store_true", help="do not write figure.png")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    return p.parse_args(argv)


def budget_volumes(calib: Any) -> list[float]:
    """Volumes of the budget table: the usual ones inside the calibrated range."""
    grads = [float(v) for v in calib.graduations_ml]
    lo, hi = min(grads), max(grads)
    vals = [v for v in BUDGET_VOLUMES if lo <= v <= hi]
    return vals or grads


def try_build_budget(calib: Any, params: Any, model_fn: Any, rows: list[dict]) -> tuple[Any | None, str | None]:
    """The uncertainty budget of the run, or ``(None, reason)`` when the geometry is unavailable."""
    if getattr(calib.cylinder, "radius_cm", None) is None:
        return None, "the cylinder spec has no radius_cm (curvature model unavailable)"
    try:
        geom = geometry_from_calibration(calib)
        return build_budget(calib, geom, params, model_fn, rows=rows), None
    except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
        return None, str(exc)


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    run = RunDir(args.run_dir)
    if not run.has_calibration():
        sys.exit(f"no calib.json in {run.root} (run scripts/calibrate.py first)")
    if not run.has_params():
        sys.exit(f"no params.json in {run.root} (run scripts/tune.py first)")
    calib = run.load_calibration()
    params, saved_step = run.load_params()
    step = max(1, int(args.step if args.step is not None else saved_step))
    model_fn = build_model_fn(calib)

    t0 = time.perf_counter()
    with VideoSource(args.video) as video:
        print(f"Video: {video}")
        end = video.n_frames - 1 if args.end is None else min(int(args.end), video.n_frames - 1)
        rows = process_video(video, calib, params, model_fn, start=args.start, end=end,
                             frame_step=step, progress=not args.quiet, tol_ml=args.tol_ml,
                             time_scale=args.frame_scale)
        fps = float(video.fps)
        size = (video.width, video.height)
    elapsed = time.perf_counter() - t0

    run.ensure()
    write_csv(rows, run.csv_path)
    summary = summarize_rows(rows)
    n_expected = len(range(int(args.start), end + 1, step))
    budget, budget_skip = try_build_budget(calib, params, model_fn, rows)
    budget_table = budget_markdown_table(budget, budget_volumes(calib)) if budget is not None else None
    meta = {
        "video": str(args.video),
        "video_name": args.video.name,
        "fps": fps,
        "frame_size": list(size),
        "frame_scale": float(args.frame_scale),
        "start": int(args.start),
        "end": int(end),
        "frame_step": step,
        "n_frames_expected": n_expected,
        "tol_ml": float(args.tol_ml),
        "params": params.to_dict(),
        "calibration": {
            "recommended_model": calib.recommended_model,
            "cylinder": calib.cylinder.name,
            "crop": list(calib.crop()),
        },
        "csv_columns": CSV_COLUMNS,
        "summary": summary,
        "uncertainty_budget": budget_table,
        "uncertainty_notes": budget.notes if budget is not None else {"skipped": budget_skip},
        "elapsed_s": round(elapsed, 2),
        "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    run.save_meta(meta)
    print(f"written: {run.csv_path}")
    print(f"written: {run.meta_path}")
    if budget_table is not None:
        print()
        print(budget_table)
        print()
    else:
        print(f"note: uncertainty budget skipped ({budget_skip}); figure.png has no ±u bands")
    if not args.no_figure:
        title = f"{args.video.name}  -  frames {args.start}..{end} step {step}"
        plot_levels(rows, title=title, out_png=run.figure_path, budget=budget)
        print(f"written: {run.figure_path}")
    for key in ("V_lower_mL", "V_total_mL", "V_foam_mL"):
        r = summary.get(key)
        if r:
            print(f"  {key:<11s} min {r['min']:7.1f}   max {r['max']:7.1f}   last {r['last']:7.1f} mL")
    print(f"done in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
