#!/usr/bin/env python
"""Uncertainty report of a run: camera geometry, curvature figure, budget table.

Reads the calibration of a run directory (new ``calib.json`` layout, or a
legacy ``calib.json`` with its sibling ``crop.json`` / ``--legacy-crop``),
rebuilds the camera geometry (:func:`geometry_from_calibration`), assembles
the four-term uncertainty budget (calibration, pixel, curvature, method)
and writes into the run directory:

* ``uncertainty_curvature.png`` -- projected graduation circles on the frame,
  u_curvature(V), top and side views;
* ``uncertainty_schematic.png`` -- standalone cylinder / camera schematic;
* ``uncertainty_budget.png`` -- contributions and quadrature total vs V;
* ``uncertainty.json`` -- geometry, per-volume table, notes, method summary.

The method term comes from the per-frame spread stored in ``levels.csv``
(``--levels-csv`` to point elsewhere); without it the budget carries a
placeholder and says so.

Examples::

    python scripts/uncertainty_report.py --run-dir runs/pour --video pour.mp4 --frame 108
    python scripts/uncertainty_report.py --run-dir legacy_run --legacy-crop legacy_run/crop.json --no-back
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cylvision.calibration import Calibration, build_model_fn, load_legacy_run  # noqa: E402
from cylvision.detection import DetectionParams  # noqa: E402
from cylvision.io import VideoSource, imread_unicode  # noqa: E402
from cylvision.pipeline import RunDir, read_csv  # noqa: E402
from cylvision.uncertainty import (  # noqa: E402
    budget_markdown_table,
    build_budget,
    curvature_uncertainty_ml,
    geometry_from_calibration,
    render_budget_figure,
    render_curvature_figure,
    render_cylinder_schematic,
    summarize_method_uncertainty,
)

DEFAULT_VOLUMES: tuple[float, ...] = (100.0, 250.0, 500.0, 750.0, 1000.0)


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="run directory holding calib.json")
    p.add_argument("--video", type=Path, default=None,
                   help="video (or still image) used to draw the curvature figure on a real frame")
    p.add_argument("--frame", type=int, default=0, help="frame index read from --video (default 0)")
    p.add_argument("--r-px", type=int, default=None,
                   help="band half-width in px (default: r_lower of params.json, else 80)")
    p.add_argument("--alpha-deg", type=float, default=None,
                   help="camera pitch in degrees (default: from the Möbius fit, 0 with back clicks)")
    p.add_argument("--no-back", action="store_true",
                   help="ignore the back-face clicks (physical-radius inversion instead of focal-free)")
    p.add_argument("--levels-csv", type=Path, default=None,
                   help="batch CSV for the method term (default: <run-dir>/levels.csv when present)")
    p.add_argument("--legacy-crop", type=Path, default=None,
                   help="crop.json of a legacy run when it does not sit next to calib.json")
    p.add_argument("--volumes", type=str, default=None,
                   help="comma-separated volumes (mL) for the table, default 100,250,500,750,1000 "
                        "clipped to the calibrated range")
    return p.parse_args(argv)


def load_calibration(run: RunDir, legacy_crop: Path | None) -> Calibration:
    """Calibration of the run, with the explicit ``--legacy-crop`` fallback."""
    if not run.has_calibration():
        sys.exit(f"no calib.json in {run.root}")
    if legacy_crop is not None:
        if not legacy_crop.exists():
            sys.exit(f"legacy crop file not found: {legacy_crop}")
        return load_legacy_run(run.calib_path, legacy_crop)
    return run.load_calibration()


def load_frame(video: Path | None, frame_idx: int) -> np.ndarray | None:
    """One BGR frame of ``video`` (a still image is accepted too), or None."""
    if video is None:
        return None
    if not video.exists():
        sys.exit(f"video not found: {video}")
    if video.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        img = imread_unicode(video)
        if img is None:
            sys.exit(f"cannot read image: {video}")
        return img
    with VideoSource(video) as src:
        fr = src.frame(int(frame_idx))
    if fr is None:
        sys.exit(f"cannot read frame {frame_idx} of {video}")
    return fr


def table_volumes(calib: Calibration, requested: str | None) -> list[float]:
    """Volumes of the budget table: the defaults inside the calibrated range."""
    grads = [float(v) for v in calib.graduations_ml]
    lo, hi = min(grads), max(grads)
    if requested:
        vals = [float(v) for v in requested.replace(";", ",").split(",") if v.strip()]
    else:
        vals = [v for v in DEFAULT_VOLUMES if lo <= v <= hi]
    if not vals:
        vals = grads
    return vals


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    run = RunDir(args.run_dir)
    calib = load_calibration(run, args.legacy_crop)
    model_fn = build_model_fn(calib)

    params = DetectionParams()
    if run.has_params():
        try:
            params, _step = run.load_params()
        except Exception as exc:  # a foreign params.json must not stop the report
            print(f"warning: params.json ignored ({exc})")
    r_px = int(args.r_px) if args.r_px is not None else int(params.r_lower)

    try:
        geom = geometry_from_calibration(calib, alpha_deg=args.alpha_deg,
                                         use_back_clicks=not args.no_back)
    except ValueError as exc:
        sys.exit(f"cannot build the camera geometry: {exc}")

    print(f"Run: {run.root}")
    print(f"Cylinder: {calib.cylinder.name}   crop x [{calib.x_left}, {calib.x_right}]  "
          f"y [{calib.y_top}, {calib.y_bottom}]   model: {calib.recommended_model}")
    print(f"Geometry: {geom.summary()}")
    if geom.v_horizon_ml is not None:
        print(f"  camera horizon at row {geom.cy_px:.1f} px  (V ~ {geom.v_horizon_ml:.0f} mL)")
    else:
        print(f"  camera horizon assumed at row {geom.cy_px:.1f} px (middle of the clicks)")
    print(f"  band half-width r = {r_px} px")

    levels_csv = args.levels_csv if args.levels_csv is not None else (
        run.csv_path if run.csv_path.exists() else None)
    rows: list[dict] = []
    method_summary: dict | None = None
    if levels_csv is not None:
        if not levels_csv.exists():
            sys.exit(f"levels CSV not found: {levels_csv}")
        rows = read_csv(levels_csv)
        method_summary = {
            "lower": summarize_method_uncertainty(rows, "u_method_lower_mL"),
            "upper": summarize_method_uncertainty(rows, "u_method_upper_mL"),
        }
        s = method_summary["lower"]
        print(f"Method uncertainty (lower interface) over {s['n']} frames of {levels_csv.name}: "
              f"median {s['median']:.2f}  mean {s['mean']:.2f}  p95 {s['p95']:.2f} mL")
    else:
        print("Method uncertainty: no levels.csv found, placeholder 0 (run the batch first)")

    budget = build_budget(calib, geom, params, model_fn, rows or None, r_px=r_px)
    volumes = table_volumes(calib, args.volumes)
    print()
    print(budget_markdown_table(budget, volumes))
    print()
    for key, note in budget.notes.items():
        print(f"  {key:<12s} {note}")

    frame = load_frame(args.video, args.frame)
    run.ensure()
    curv_png = run.root / "uncertainty_curvature.png"
    schem_png = run.root / "uncertainty_schematic.png"
    budget_png = run.root / "uncertainty_budget.png"
    render_curvature_figure(frame, calib, geom, r_px, model_fn, curv_png)
    render_cylinder_schematic(geom, calib, r_px, schem_png)
    grads = [float(v) for v in calib.graduations_ml]
    render_budget_figure(budget, (min(grads), max(grads)), budget_png)

    report = {
        "run_dir": str(run.root),
        "calibration": {
            "image_ref": calib.image_ref,
            "cylinder": calib.cylinder.name,
            "recommended_model": calib.recommended_model,
            "crop": list(calib.crop()),
            "has_back_clicks": calib.has_back_clicks(),
        },
        "geometry": asdict(geom),
        "r_px": r_px,
        "curvature_per_graduation": curvature_uncertainty_ml(calib, geom, r_px, model_fn),
        "budget": {
            "u_calibration_ml": budget.u_calibration_ml,
            "u_method_ml": budget.u_method_ml,
            "u_method_available": budget.u_method_available,
            "notes": budget.notes,
            "table": budget.table(volumes),
        },
        "method_summary": method_summary,
        "levels_csv": None if levels_csv is None else str(levels_csv),
        "figures": [str(curv_png), str(schem_png), str(budget_png)],
    }
    out_json = run.root / "uncertainty.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for pth in (curv_png, schem_png, budget_png, out_json):
        print(f"written: {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
