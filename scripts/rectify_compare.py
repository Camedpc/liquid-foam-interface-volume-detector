#!/usr/bin/env python
"""Original vs rectified: does straightening the cylinder reduce the method uncertainty?

For every frame of a window of the video the lower (liquid/foam) interface
is detected twice with the same ``DetectionParams``: on the original crop
and on the crop of the frame rectified by inverse projection
(:mod:`cylvision.rectify`).  The per-frame method uncertainty of both is
computed with the definition of the budget (range / (2 sqrt 3),
``u_*_mL``) and with the one of the original analysis (max half-range /
sqrt 3, ``umax_*_mL``), and written to ``rectify_compare.csv``; the
summary (medians, means, p95, ratios, geometry, residual curvature) goes
to ``rectify_summary.json``.

Two figures are written into ``--out``:

* ``rectify_uncertainty.png`` -- u_method(t) original vs rectified (raw and
  1 s rolling max) with, on the right, one frame before / after rectification
  annotated with the per-column detections and the empirical bounds;
* ``rectify_graduations.png`` -- the graduation circles projected on a frame
  and the same circles, horizontal, after rectification.

The rectification needs the focal-free geometry (front + back graduation
clicks in ``calib.json``); with front clicks only the script warns and uses
the physical-radius geometry, whose horizon is a guess.

Example::

    python scripts/rectify_compare.py --video pour.MOV --run-dir runs/pour --params runs/pour/params.json \\
        --start 2250 --n-frames 1798 --out runs/pour/rectify
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cylvision.calibration import Calibration, build_model_fn, load_legacy_run  # noqa: E402
from cylvision.detection import DetectionParams, annotate_interfaces  # noqa: E402
from cylvision.io import VideoSource  # noqa: E402
from cylvision.pipeline import RunDir  # noqa: E402
from cylvision.rectify import (COMPARE_COLUMNS, compare_frame, compare_video, crop_maps, median_edges,  # noqa: E402
                               rectification_maps, rectify_frame, render_graduations_figure,
                               render_uncertainty_figure, residual_delta_px, summarize_compare, wall_edges)
from cylvision.uncertainty import (curvature_delta_px, geometry_from_calibration,  # noqa: E402
                                   method_uncertainty_from_result, volume_to_row_fn)


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="source video (full frame rate preferred)")
    p.add_argument("--run-dir", type=Path, required=True, help="run directory holding calib.json (+ crop.json)")
    p.add_argument("--params", type=Path, default=None,
                   help="params.json of the detector (default: <run-dir>/params.json)")
    p.add_argument("--legacy-crop", type=Path, default=None, help="crop.json of a legacy run kept elsewhere")
    p.add_argument("--start", type=int, default=0, help="first frame of the window")
    p.add_argument("--n-frames", type=int, default=300, help="number of consecutive frames")
    p.add_argument("--which", choices=("lower", "upper"), default="lower", help="interface compared")
    p.add_argument("--tol-ml", type=float, default=7.0, help="outlier tolerance of the method term (mL)")
    p.add_argument("--walls", choices=("calib", "detect"), default="calib",
                   help="axis column and half-width of the mapping: the calibration crop, or the glass "
                        "walls measured on a few frames")
    p.add_argument("--scatter-frame", type=int, default=None,
                   help="frame of the before/after panels (default: middle of the window)")
    p.add_argument("--graduation-frame", type=int, default=None,
                   help="frame of the graduation figure (default: --start)")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: <run-dir>/rectify)")
    return p.parse_args(argv)


def load_calibration(run: RunDir, legacy_crop: Path | None) -> Calibration:
    if not run.has_calibration():
        sys.exit(f"no calib.json in {run.root}")
    if legacy_crop is not None:
        return load_legacy_run(run.calib_path, legacy_crop)
    return run.load_calibration()


def load_params(path: Path, calib: Calibration) -> DetectionParams:
    if not path.exists():
        sys.exit(f"params file not found: {path}")
    p = DetectionParams.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if p.y_bottom is None:
        p.y_bottom = int(calib.y_bottom)
    if p.cx is None:
        p.cx = int(calib.cx())
    return p


def scatter_panel(crop, res, params, model_fn, *, which: str, tol_ml: float, zoom: float = 3.0, half: int = 40):
    """Zoom on the interface with the per-column dots, the mean line and the dashed bounds."""
    mu = method_uncertainty_from_result(res, model_fn, which=which, tol_ml=tol_ml)
    img = annotate_interfaces(crop, res, params, zoom=zoom, show_extras=False, y_bounds=(mu["y_up"], mu["y_lo"]),
                              line_w=3, show_labels=False, show_masks=False, show_axis=False)
    y = mu["y_mean"] if math.isfinite(mu["y_mean"]) else crop.shape[0] / 2
    r0 = int(max(0, (y - half) * zoom))
    r1 = int(min(img.shape[0], (y + half) * zoom))
    return img[r0:r1], mu


def write_csv(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(COMPARE_COLUMNS) + "\n")
        for r in rows:
            cells = []
            for k in COMPARE_COLUMNS:
                v = r.get(k)
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    cells.append("")
                elif isinstance(v, float):
                    cells.append(f"{v:.6g}")
                else:
                    cells.append(str(v))
            fh.write(",".join(cells) + "\n")


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    run = RunDir(args.run_dir)
    calib = load_calibration(run, args.legacy_crop)
    params = load_params(args.params or run.params_path, calib)
    model_fn = build_model_fn(calib)
    out_dir = args.out or (run.root / "rectify")
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = geometry_from_calibration(calib)
    if geom.source != "focal_free":
        print("warning: no back clicks in calib.json -> physical-radius geometry, the horizon row is a guess")
    W, H = int(calib.image_size[0]), int(calib.image_size[1])
    print(f"Geometry: {geom.summary()}  horizon y = {geom.cy_px:.1f} px"
          + (f" (V ≈ {geom.v_horizon_ml:.0f} mL)" if geom.v_horizon_ml is not None else ""))

    with VideoSource(args.video) as video:
        print(f"Video: {video}")
        fps = float(video.fps) or 30.0
        start, n = int(args.start), int(args.n_frames)

        cx_px, a_px = float(geom.cx_px), float(geom.a_px)
        walls: dict = {"mode": args.walls, "cx_px": cx_px, "a_px": a_px}
        if args.walls == "detect":
            y_band = (int(min(calib.graduations_px_y)) - 20, int(max(calib.graduations_px_y)) + 20)
            probes = [start, start + n // 2, start + n - 1]
            found = [wall_edges(fr, y_band, (calib.x_left, calib.x_right))
                     for fr in (video.frame(i) for i in probes) if fr is not None]
            med = median_edges(found)
            if med is None:
                print("warning: glass walls not found, using the calibration crop")
            else:
                cx_px, a_px = (med[0] + med[1]) / 2.0, (med[1] - med[0]) / 2.0
                walls.update({"cx_px": cx_px, "a_px": a_px, "x_left": med[0], "x_right": med[1],
                              "probes": probes, "found": found})
        print(f"Mapping: cx = {cx_px:.1f} px, a = {a_px:.1f} px ({args.walls}), rho = {geom.rho:.4f}")
        maps = rectification_maps(geom, W, H, cx_px=cx_px, a_px=a_px)

        def progress(k: int, total: int, row: dict) -> None:
            if k == 1 or k == total or k % 100 == 0:
                print(f"  {k:5d}/{total}  frame {row['frame_idx']}  u_orig {row['u_orig_mL']:.2f}  "
                      f"u_rect {row['u_rect_mL']:.2f} mL")

        print(f"Comparing frames {start}..{start + n - 1} ({n / fps:.1f} s)")
        rows = compare_video(video, calib, params, maps, model_fn, start=start, n_frames=n, which=args.which,
                             tol_ml=args.tol_ml, progress=progress)
        if not rows:
            sys.exit("no frame processed")
        write_csv(rows, out_dir / "rectify_compare.csv")
        summary = summarize_compare(rows, fps=fps)

        # --- before / after panels of one frame
        idx_s = int(args.scatter_frame) if args.scatter_frame is not None else start + n // 2
        frame_s = video.frame(idx_s)
        if frame_s is None:
            sys.exit(f"cannot read frame {idx_s}")
        maps_crop = crop_maps(maps, calib.x_left, calib.x_right)
        row_s, res_o, res_r, crop_o, crop_r = compare_frame(frame_s, calib, params, maps_crop, model_fn,
                                                            which=args.which, tol_ml=args.tol_ml)
        pan_o, mu_o = scatter_panel(crop_o, res_o, params, model_fn, which=args.which, tol_ml=args.tol_ml)
        pan_r, mu_r = scatter_panel(crop_r, res_r, params, model_fn, which=args.which, tol_ml=args.tol_ml)
        h = min(pan_o.shape[0], pan_r.shape[0])
        labels = (f"original, frame {idx_s}: {row_s['n_kept_orig']} columns, range {row_s['range_orig_mL']:.2f} mL, "
                  f"u = {row_s['u_orig_mL']:.2f} mL",
                  f"rectified, same frame: {row_s['n_kept_rect']} columns, range {row_s['range_rect_mL']:.2f} mL, "
                  f"u = {row_s['u_rect_mL']:.2f} mL")
        title = (f"{args.which} interface, frames {start}..{start + len(rows) - 1} ({len(rows) / fps:.0f} s), "
                 f"band r = {getattr(params, f'r_{args.which}')} px")
        render_uncertainty_figure(rows, fps, out_dir / "rectify_uncertainty.png", key="u",
                                  scatter=(pan_o[:h], pan_r[:h]), scatter_labels=labels, title=title)
        render_uncertainty_figure(rows, fps, out_dir / "rectify_uncertainty_umax.png", key="umax", title=title)

        # --- graduation figure
        idx_g = int(args.graduation_frame) if args.graduation_frame is not None else start
        frame_g = video.frame(idx_g)
        if frame_g is None:
            sys.exit(f"cannot read frame {idx_g}")
        v2r = volume_to_row_fn(calib, model_fn)
        render_graduations_figure(frame_g, rectify_frame(frame_g, maps), calib, geom,
                                  lambda V: float(v2r(V)), out_dir / "rectify_graduations.png",
                                  cx_px=cx_px, a_px=a_px)

    # --- residual curvature: the model Δ before, and after a rectification with a slightly wrong geometry
    y_med = float(np.nanmedian([r["y_orig_px"] for r in rows]))
    r_px = int(getattr(params, f"r_{args.which}"))
    from dataclasses import replace
    delta_before = curvature_delta_px(y_med, geom, r_px)
    residual = {
        "y_interface_px": y_med, "r_px": r_px, "delta_before_px": delta_before,
        "delta_after_exact_px": residual_delta_px(y_med, geom, geom, r_px, cx_px=cx_px, a_px=a_px),
        "delta_after_rho_plus_0.01_px": residual_delta_px(y_med, geom, replace(geom, rho=geom.rho + 0.01), r_px,
                                                          cx_px=cx_px, a_px=a_px),
        "delta_after_horizon_plus_20px_px": residual_delta_px(y_med, geom, replace(geom, cy_px=geom.cy_px + 20.0),
                                                              r_px, cx_px=cx_px, a_px=a_px),
    }
    summary.update({
        "video": args.video.name, "run_dir": run.root.name, "which": args.which, "start": start,
        "n_frames": len(rows), "fps": fps, "duration_s": len(rows) / fps, "tol_ml": args.tol_ml,
        "params": params.to_dict(), "walls": walls,
        "geometry": {"source": geom.source, "rho": geom.rho, "cy_px": geom.cy_px, "cx_px": geom.cx_px,
                     "a_px": geom.a_px, "v_horizon_ml": geom.v_horizon_ml, "D_cm": geom.D_cm,
                     "f_px": geom.f_px},
        "scatter_frame": {"frame": idx_s, **row_s},
        "graduation_frame": idx_g,
        "residual_curvature": residual,
    })
    (out_dir / "rectify_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                                  encoding="utf-8")

    for key, name in (("u", "range / (2√3)"), ("umax", "max half-range / √3")):
        s = summary[key]
        print(f"u_method [{name}]  original: median {s['orig']['median']:.3f}  mean {s['orig']['mean']:.3f}  "
              f"p95 {s['orig']['p95']:.3f} mL   rectified: median {s['rect']['median']:.3f}  "
              f"mean {s['rect']['mean']:.3f}  p95 {s['rect']['p95']:.3f} mL   "
              f"ratio {s['ratio_median']:.3f} (median) {s['ratio_mean']:.3f} (mean)")
        rm = s["rolling_max_1s"]
        print(f"    1 s rolling max: original mean {rm['orig_mean']:.3f}, rectified mean {rm['rect_mean']:.3f} mL")
    print(f"curvature Δ at y = {y_med:.0f} px, r = {r_px}: {delta_before:.2f} px before; after rectification "
          f"{residual['delta_after_exact_px']:.2f} px (exact geometry), "
          f"{residual['delta_after_rho_plus_0.01_px']:.2f} px (ρ off by 0.01), "
          f"{residual['delta_after_horizon_plus_20px_px']:.2f} px (horizon off by 20 px)")
    print(f"written: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
