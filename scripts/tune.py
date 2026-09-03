#!/usr/bin/env python
"""Open the live tuner on a video and save the detector parameters.

Loads ``calib.json`` from the run directory (a legacy ``calib.json`` +
``crop.json`` pair is accepted), starts from the saved ``params.json`` when
it exists (or from ``--params`` / the defaults), and opens the OpenCV
canvas + the Tk control panel. ``Enter`` or ``Run batch`` saves
``params.json`` (parameters + ``frame_step``) and prints the batch
command; ``Esc`` / ``Abort`` leaves the run directory untouched.

Example::

    python scripts/tune.py --video pour.mp4 --run-dir runs/pour --polarity lower_brighter

``--frame-scale K``: the video is a 1-in-K subsample of the recording (the
frame counter of the info strip is multiplied by K to show the source time).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cylvision.calibration import build_model_fn  # noqa: E402
from cylvision.detection import POLARITIES, DetectionParams  # noqa: E402
from cylvision.io import VideoSource  # noqa: E402
from cylvision.pipeline.run_store import RunDir  # noqa: E402


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="video to tune on")
    p.add_argument("--run-dir", type=Path, required=True, help="run directory holding calib.json")
    p.add_argument("--polarity", choices=POLARITIES, default=None,
                   help="initial polarity (lower_darker = front-lit on black, lower_brighter = back-lit)")
    p.add_argument("--params", type=str, default=None,
                   help="initial parameters: path to a params.json or an inline JSON object")
    p.add_argument("--start-frame", type=int, default=0, help="frame shown first (default 0)")
    p.add_argument("--frame-step", type=int, default=None,
                   help="initial batch frame step (default: saved params.json or 1)")
    p.add_argument("--frame-scale", type=float, default=1.0,
                   help="source frames per video frame (pre-subsampled video), default 1")
    p.add_argument("--two-columns", action="store_true", help="start the control panel in 2 columns")
    return p.parse_args(argv)


def load_initial_params(args: argparse.Namespace, run: RunDir, calib) -> tuple[DetectionParams, int]:
    params: DetectionParams | None = None
    step = 1
    if args.params:
        text = args.params.strip()
        d = json.loads(text) if text.startswith("{") else json.loads(Path(text).read_text(encoding="utf-8"))
        params = DetectionParams.from_dict(d)
        step = int(d.get("frame_step", 1))
    elif run.has_params():
        params, step = run.load_params()
        print(f"loaded {run.params_path}")
    if params is None:
        params = DetectionParams(y_top=int(calib.y_top), y_bottom=int(calib.y_bottom), cx=int(calib.cx()))
    if args.polarity:
        params.polarity = args.polarity
    if args.frame_step is not None:
        step = int(args.frame_step)
    return params, max(1, step)


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    run = RunDir(args.run_dir)
    if not run.has_calibration():
        sys.exit(f"no calib.json in {run.root} (run scripts/calibrate.py first)")
    calib = run.load_calibration()
    model_fn = build_model_fn(calib)
    params, step = load_initial_params(args, run, calib)

    from cylvision.ui.tuner import run_tuner

    with VideoSource(args.video) as video:
        print(f"Video: {video}")
        print("Keys: Enter = run batch, Esc = abort, t = specimen panel, l = inline/stacked, "
              "f = fullscreen, Space = play, arrows/a/d = frame, Home/0 = first frame")
        params, step, accepted = run_tuner(video, calib, params, model_fn=model_fn,
                                           start_idx=args.start_frame, two_columns=args.two_columns,
                                           frame_step=step, time_scale=args.frame_scale)
    if not accepted:
        print("tuner aborted, params.json not written")
        return 1
    run.save_params(params, step)
    print(f"written: {run.params_path}  (frame_step = {step})")
    print(f"next: python scripts/run_batch.py --video \"{args.video}\" --run-dir \"{run.root}\""
          + (f" --frame-scale {args.frame_scale:g}" if args.frame_scale != 1.0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
