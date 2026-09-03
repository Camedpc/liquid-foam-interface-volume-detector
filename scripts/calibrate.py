#!/usr/bin/env python
"""Calibrate a graduated cylinder on a video frame or an image.

Interactive flow:

1. pick the calibration frame (``--frame N`` or the frame picker window);
2. click the crop band (left / right edges), the top / bottom rows and every
   graduation of the chosen cylinder (Shift = precision mode, arrows nudge,
   ``s`` skips a graduation that is out of frame, ``z`` undoes, Enter
   validates, Esc aborts);
3. optionally (``--back``) click the same graduations on the back face of
   the cylinder, used by the focal-free curvature model;
4. fit the three models (PCHIP / poly2 / Möbius), write ``calib.json`` and
   ``calibration_check.png`` into the run directory.

Examples::

    python scripts/calibrate.py --video pour.mp4 --cylinder 1000 --out runs/pour
    python scripts/calibrate.py --image ref.png --cylinder custom --graduations 5,10,20,30,40,50 --radius-cm 1.35
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cylvision.calibration import (  # noqa: E402
    PRESETS,
    Calibration,
    CylinderSpec,
    apply_back_clicks,
    build_back_prompts,
    collect_back_clicks,
    collect_calibration_clicks,
    fit_calibration,
    render_verification_figure,
    residual_summary,
    spec_from_graduations,
)
from cylvision.io import VideoSource, imread_unicode, imwrite_unicode  # noqa: E402
from cylvision.pipeline.run_store import RunDir, find_run  # noqa: E402


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="video to calibrate (a frame is picked)")
    src.add_argument("--image", type=Path, help="still image to calibrate")
    p.add_argument("--cylinder", choices=sorted(PRESETS) + ["custom"], default="1000",
                   help="cylinder preset, or 'custom' with --graduations (default: 1000)")
    p.add_argument("--graduations", type=str, default=None,
                   help="comma-separated graduations to click in mL (custom cylinder), e.g. 100,200,300")
    p.add_argument("--radius-cm", type=float, default=None,
                   help="inner radius of the cylinder in cm (curvature model); overrides the preset")
    p.add_argument("--name", type=str, default=None, help="display name of a custom cylinder")
    p.add_argument("--frame", type=int, default=None,
                   help="frame index to calibrate on (default: interactive frame picker)")
    p.add_argument("--out", type=Path, default=None,
                   help="run directory (default: runs/<video or image stem>)")
    p.add_argument("--back", action="store_true",
                   help="also click the back-face graduations (focal-free curvature model)")
    p.add_argument("--save-frame", action="store_true",
                   help="also save the calibration frame as <out>/calibration_frame.png")
    return p.parse_args(argv)


def make_spec(args: argparse.Namespace) -> CylinderSpec:
    if args.cylinder == "custom":
        if not args.graduations:
            sys.exit("--cylinder custom requires --graduations (e.g. --graduations 100,200,300)")
        grads = [float(v) for v in args.graduations.replace(";", ",").split(",") if v.strip()]
        spec = spec_from_graduations(grads, name=args.name)
    else:
        # Copy: the preset is a module-level object and must not be mutated.
        spec = replace(PRESETS[args.cylinder])
        if args.graduations:
            spec.graduations_ml = [float(v) for v in args.graduations.replace(";", ",").split(",")
                                   if v.strip()]
    if args.radius_cm is not None:
        spec.radius_cm = float(args.radius_cm)
    return spec


def load_source(args: argparse.Namespace) -> tuple[object, str, int | None]:
    """``(image_bgr, image_ref, frame_idx)`` from ``--video`` / ``--image``."""
    if args.image is not None:
        img = imread_unicode(args.image)
        if img is None:
            sys.exit(f"cannot read image: {args.image}")
        return img, args.image.name, None
    with VideoSource(args.video) as video:
        print(f"Video: {video}")
        idx = args.frame
        if idx is None:
            from cylvision.ui.frame_picker import pick_frame

            idx = pick_frame(video.cap, video.n_frames, video.fps,
                             title="Choose the calibration frame  (Enter = validate, Esc = abort)")
            if idx is None:
                sys.exit("frame selection aborted")
        img = video.frame(int(idx))
        if img is None:
            sys.exit(f"cannot read frame {idx}")
        print(f"Calibration frame: #{idx}")
        return img, args.video.name, int(idx)


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    spec = make_spec(args)
    img, image_ref, frame_idx = load_source(args)
    H, W = img.shape[:2]  # type: ignore[attr-defined]
    print(f"Cylinder: {spec.name}  graduations: {[int(v) if float(v).is_integer() else v for v in spec.graduations_ml]}")

    source = args.video if args.video is not None else args.image
    run = RunDir(args.out) if args.out is not None else find_run(REPO_ROOT / "runs", source)

    clicks = collect_calibration_clicks(img, spec.graduations_ml)  # type: ignore[arg-type]
    if clicks is None:
        print("calibration aborted")
        return 1
    calib: Calibration = fit_calibration(clicks, spec, image_ref, (W, H))

    if args.back:
        prompts = build_back_prompts(calib)
        pts = collect_back_clicks(img, prompts)  # type: ignore[arg-type]
        if pts is None:
            print("back clicks skipped (aborted)")
        else:
            try:
                calib = apply_back_clicks(calib, pts, frame_idx)
                print(f"back clicks: {sum(p is not None for p in pts)} graduation(s)")
            except ValueError as exc:
                print(f"back clicks ignored: {exc}")

    run.ensure()
    run.save_calibration(calib)
    render_verification_figure(img, calib, run.check_png)  # type: ignore[arg-type]
    if args.save_frame:
        imwrite_unicode(run.root / "calibration_frame.png", img)  # type: ignore[arg-type]

    print(f"crop: x in [{calib.x_left}, {calib.x_right}], y in [{calib.y_top}, {calib.y_bottom}]")
    for name, s in residual_summary(calib).items():
        flag = "  <- recommended" if name == calib.recommended_model else ""
        print(f"  {name:<7s} rms = {s['rms']:.3f} mL   max = {s['max']:.3f} mL{flag}")
    print(f"written: {run.calib_path}")
    print(f"written: {run.check_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
