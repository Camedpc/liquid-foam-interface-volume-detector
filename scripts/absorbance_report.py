#!/usr/bin/env python
"""Absorbance analysis of a back-lit run: A(t, V) and c(t, V) heat maps + measurement figure.

Needs a run directory with ``calib.json`` (crop band, graduations) and the
gradient parameters of the liquid / foam interface (``params.json`` of the
run, or ``--params``), plus a stretch of the video showing the EMPTY
cylinder under the back-light (``--ref-frames a:b``, frames ``a .. b-1``
averaged into the reference ``I0``).

For every analysed frame: liquid / foam row by the gradient detector,
absorbance map ``A = -ln(I / I0)``, radial profile ``A(z)``, foam top by
the threshold ``A_max / k``, then, once the run is complete, the mass
balance ``gamma = (V_inf - V_beer) / (S * integral A dz)`` and the liquid
content ``c(z) = gamma A(z)`` inside the foam.

Written into ``--out-dir`` (default: the run directory):

* ``absorbance.npz`` -- ``A`` (n_frames x H), ``c``, ``t``, ``frame_idx``,
  ``z``, ``y_liquid``, ``y_foam_top``, ``gamma``, ``V_beer``, ``V_foam``;
* ``absorbance.csv`` -- one row per analysed frame (see
  ``cylvision.absorbance.batch.ABS_CSV_COLUMNS``);
* ``absorbance_meta.json`` -- parameters, reference frames, V_inf, k sweep;
* ``heatmap_A.png``, ``heatmap_c.png``, ``absorbance_measurement.png``.

Example (video pre-subsampled 1 in 60, pour starting at source frame 1347)::

    python scripts/absorbance_report.py --video run09/subsampled_step60.mp4 --run-dir run09 \\
        --params green09_params.json --ref-frames 5:15 --frame-scale 60 --t0-frame 1347 \\
        --k 5 --r-dens 64 --v-inf auto --out-dir out/run09
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cylvision.absorbance import (  # noqa: E402
    ABS_CSV_COLUMNS,
    DEFAULT_KS,
    S_CM2,
    AbsorbanceParams,
    finalize_mass_balance,
    foam_tops_for_ks,
    foam_volume_vs_k,
    plot_heatmap_A,
    plot_heatmap_c,
    plot_measurement_figure,
    prepare_reference,
    process_video_absorbance,
    px_per_cm_from_calibration,
    reference_from_video,
    volume_window,
)
from cylvision.calibration import build_model_fn  # noqa: E402
from cylvision.detection import DetectionParams  # noqa: E402
from cylvision.detection.interfaces import band_columns  # noqa: E402
from cylvision.io import VideoSource  # noqa: E402
from cylvision.pipeline import RunDir, write_csv  # noqa: E402


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_ref_frames(text: str) -> tuple[int, int]:
    """``"a:b"`` -> ``(a, b)`` (frames ``a .. b-1``)."""
    try:
        a, b = (int(v) for v in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--ref-frames expects a:b, got {text!r}") from exc
    if b <= a:
        raise argparse.ArgumentTypeError("--ref-frames: b must be > a")
    return a, b


def parse_v_inf(text: str) -> float | None:
    if text.strip().lower() == "auto":
        return None
    return float(text)


def parse_px_per_cm(text: str) -> float | None:
    if text.strip().lower() == "auto":
        return None
    v = float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError("--px-per-cm must be > 0")
    return v


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="video to process")
    p.add_argument("--run-dir", type=Path, required=True, help="run directory with calib.json (+ params.json)")
    p.add_argument("--params", type=Path, default=None,
                   help="gradient params.json of the liquid/foam interface (default: <run-dir>/params.json)")
    p.add_argument("--out-dir", type=Path, default=None, help="where to write (default: the run directory)")
    p.add_argument("--ref-frames", type=parse_ref_frames, default=(0, 10),
                   help="frames a:b of the empty cylinder averaged into I0 (default 0:10)")
    p.add_argument("--start", type=int, default=0, help="first frame analysed (default 0)")
    p.add_argument("--end", type=int, default=None, help="last frame, inclusive (default: last)")
    p.add_argument("--step", type=int, default=1, help="analyse one frame in STEP (default 1)")
    p.add_argument("--k", type=float, default=5.0, help="foam-top threshold T = A_max / k (default 5)")
    p.add_argument("--r-dens", type=int, default=64, help="half-width of the profile band in px (default 64)")
    p.add_argument("--px-per-cm", type=parse_px_per_cm, default=None,
                   help="vertical scale in px per cm, or 'auto' = S / |dV/dy| from the calibration (default)")
    p.add_argument("--section-cm2", type=float, default=S_CM2,
                   help=f"inner section of the cylinder in cm^2 (default {S_CM2:.2f}, 6.7 cm bore)")
    p.add_argument("--v-inf", type=parse_v_inf, default=None,
                   help="final liquid volume in mL for the mass balance, or 'auto' = max V_beer (default)")
    p.add_argument("--channel", type=str, default="G", help="plane compared with I0: gray|B|G|R (default G)")
    p.add_argument("--light-blur", type=float, default=1.0, help="Gaussian sigma on I and I0 (default 1)")
    p.add_argument("--dead-thresh", type=float, default=178.0,
                   help="reference pixels darker than this are dead and filled (default 178)")
    p.add_argument("--dead-dilate", type=int, default=2, help="dilation of the dead mask in px (default 2)")
    p.add_argument("--t0-frame", type=float, default=None,
                   help="SOURCE frame index taken as t = 0 (start of the pour); default: --start")
    p.add_argument("--frame-scale", type=float, default=1.0,
                   help="source frames per video frame (pre-subsampled video), default 1")
    p.add_argument("--measurement-frame", type=int, default=None,
                   help="frame of the 'how it is measured' figure (default: the analysed frame closest to "
                        "the middle of the run with both interfaces found)")
    p.add_argument("--ks", type=str, default=",".join(str(k) for k in DEFAULT_KS),
                   help="k values overlaid on the A heat map (default 2..10)")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    run = RunDir(args.run_dir)
    if not run.has_calibration():
        sys.exit(f"no calib.json in {run.root} (run scripts/calibrate.py first)")
    calib = run.load_calibration()
    p_path = args.params if args.params is not None else run.params_path
    if not p_path.exists():
        sys.exit(f"params file not found: {p_path}")
    det_params = DetectionParams.from_dict(json.loads(p_path.read_text(encoding="utf-8")))
    if det_params.y_bottom is None:
        det_params.y_bottom = int(calib.y_bottom)
    if det_params.cx is None:
        det_params.cx = int(calib.cx())
    model_fn = build_model_fn(calib)
    out_dir = args.out_dir if args.out_dir is not None else run.root
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = tuple(int(k) for k in args.ks.split(",") if k.strip())
    if int(args.k) not in ks:
        ks = tuple(sorted(set(ks) | {int(args.k)}))

    px_per_cm = (px_per_cm_from_calibration(calib, args.section_cm2) if args.px_per_cm is None
                 else float(args.px_per_cm))
    px_mode = "from the calibration slope and the section" if args.px_per_cm is None else "given"
    print(f"px_per_cm = {px_per_cm:.2f} ({px_mode})")
    abs_params = AbsorbanceParams(
        channel=args.channel, light_blur=args.light_blur, r_dens=args.r_dens, foam_k=args.k,
        dead_thresh=args.dead_thresh, dead_dilate=args.dead_dilate, px_per_cm=px_per_cm,
        section_cm2=args.section_cm2, y_top=int(det_params.y_top), cx=int(det_params.cx),
    )

    t0 = time.perf_counter()
    with VideoSource(args.video) as video:
        print(f"Video: {video}")
        x_left, x_right, _, _ = calib.crop()
        a, b = args.ref_frames
        ref = reference_from_video(video, a, b, crop=(x_left, x_right))
        prep = prepare_reference(ref, abs_params)
        print(f"Reference I0: frames {a}..{b - 1} ({ref.frame_count} averaged), "
              f"median {float(np.median(prep.raw)):.1f}, std across frames {float(np.median(ref.std_bgr)):.2f}, "
              f"dead pixels {prep.n_dead} ({100.0 * prep.n_dead / prep.valid.size:.1f} %)")
        end = video.n_frames - 1 if args.end is None else min(int(args.end), video.n_frames - 1)
        # First pass without maps: find the measurement frame, then keep its map on a second read.
        run_data, _ = process_video_absorbance(
            video, calib, det_params, abs_params, prep, model_fn, start=args.start, end=end,
            frame_step=args.step, t0_frame=args.t0_frame, time_scale=args.frame_scale,
            progress=not args.quiet)
        fps = float(video.fps)
        if args.measurement_frame is None:
            ok = np.isfinite(run_data.V_foam)
            cand = run_data.frame_idx[ok] if ok.any() else run_data.frame_idx
            m_frame = int(cand[np.argmin(np.abs(cand - 0.5 * (cand[0] + cand[-1])))])
        else:
            m_frame = int(args.measurement_frame)
        _, maps = process_video_absorbance(
            video, calib, det_params, abs_params, prep, model_fn, start=m_frame, end=m_frame,
            frame_step=1, t0_frame=args.t0_frame, time_scale=args.frame_scale, progress=False,
            keep_maps=(m_frame,))
    finalize_mass_balance(run_data, args.v_inf, abs_params.section_cm2)
    elapsed = time.perf_counter() - t0

    # -- files ---------------------------------------------------------------
    np.savez_compressed(out_dir / "absorbance.npz", **run_data.to_npz_dict())
    write_csv(run_data.rows(), out_dir / "absorbance.csv", ABS_CSV_COLUMNS)
    foam_tops = foam_tops_for_ks(run_data, ks, y_top=abs_params.y_top)
    i_m = int(np.where(run_data.frame_idx == m_frame)[0][0]) if (run_data.frame_idx == m_frame).any() else None
    k_sweep = foam_volume_vs_k(run_data, i_m, model_fn, ks, y_top=abs_params.y_top) if i_m is not None else {}

    H = run_data.A.shape[1]
    y_range = volume_window(calib, H)
    t_label = "time since the start of the pour (s)" if args.t0_frame is not None else "time (s)"
    title_A = (f"A(t, V) - {args.video.name}, frames {args.start}..{end} step {args.step}, "
               f"reference frames {a}..{b - 1}, band r = {abs_params.r_dens} px")
    plot_heatmap_A(run_data.A, run_data.t_sec, model_fn=model_fn, y_range=y_range,
                   out_png=out_dir / "heatmap_A.png", y_liquid=run_data.y_liquid, foam_tops=foam_tops,
                   k_main=int(args.k), title=title_A, V_inf_ml=run_data.V_inf_ml, time_label=t_label)
    finite_c = np.isfinite(run_data.c)
    c_max = float(np.nanmax(run_data.c)) if finite_c.any() else float("nan")
    plot_heatmap_c(run_data.c, run_data.t_sec, model_fn=model_fn, y_range=y_range,
                   out_png=out_dir / "heatmap_c.png", y_liquid=run_data.y_liquid,
                   y_foam_top=run_data.y_foam_top,
                   title=f"c(t, V) = γ A - liquid volume fraction of the foam, V∞ = {run_data.V_inf_ml:.1f} mL",
                   time_label=t_label)
    res = maps.get(m_frame)
    if res is not None:
        band = band_columns(res.A.shape[1], abs_params.resolved_cx(res.A.shape[1]), abs_params.r_dens)
        t_m = float(run_data.t_sec[i_m]) if i_m is not None else float("nan")
        plot_measurement_figure(prep.raw, res.intensity, res.A, res.A_z, y_liquid=res.y_liquid,
                                y_foam_top=res.y_foam_top, A_max=res.A_max, threshold=res.threshold,
                                band=band, model_fn=model_fn, y_range=y_range,
                                out_png=out_dir / "absorbance_measurement.png", k=abs_params.foam_k,
                                title=f"How the absorbance is measured - frame {m_frame} (t = {t_m:.0f} s)")

    meta: dict[str, Any] = {
        "video": str(args.video), "fps": fps, "frame_scale": float(args.frame_scale),
        "start": int(args.start), "end": int(end), "step": int(args.step),
        "t0_frame": args.t0_frame, "ref_frames": [a, b], "n_dead_pixels": int(prep.n_dead),
        "absorbance_params": abs_params.to_dict(), "detection_params": det_params.to_dict(),
        "V_inf_ml": float(run_data.V_inf_ml), "V_inf_mode": "auto (max V_beer)" if args.v_inf is None else "given",
        "px_per_cm_mode": "auto (S / |dV/dy|)" if args.px_per_cm is None else "given",
        "n_frames": int(run_data.n), "n_foam_top_found": int(np.isfinite(run_data.y_foam_top).sum()),
        "c_max": c_max, "measurement_frame": m_frame,
        "k_sweep": {str(k): {"y_foam_top_px": y, "V_foam_mL": v} for k, (y, v) in k_sweep.items()},
        "heatmap_y_range_px": list(y_range), "csv_columns": ABS_CSV_COLUMNS,
        "elapsed_s": round(elapsed, 2), "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "absorbance_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float),
                                                 encoding="utf-8")
    for name in ("absorbance.npz", "absorbance.csv", "absorbance_meta.json", "heatmap_A.png", "heatmap_c.png",
                 "absorbance_measurement.png"):
        print(f"written: {out_dir / name}")
    print(f"V_inf = {run_data.V_inf_ml:.2f} mL; foam top found on {meta['n_foam_top_found']}/{run_data.n} frames; "
          f"c_max = {c_max:.3f}")
    if k_sweep:
        print(f"V_foam vs k at frame {m_frame}: " + "  ".join(f"k={k}: {v:.0f} mL" for k, (_y, v) in k_sweep.items()))
    print(f"done in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
