#!/usr/bin/env python
"""Regenerate every README image in ``docs/images`` from two real runs.

Two runs are used: a back-lit **green screen** (polarity ``lower_brighter``)
and a front-lit **black background** (polarity ``lower_darker``). Each run
directory holds ``calib.json`` (new or legacy layout with ``crop.json``) and
a ``params.json`` written by ``scripts/tune.py`` (``--green-params`` /
``--black-params`` point elsewhere when the run directory holds a foreign
``params.json``).

Only public package functions are used; nothing interactive is opened
except the Tk control panel when ``--panel-screenshot`` is given (the window
is mapped for a fraction of a second and grabbed).

The batch figures need the videos to be processed (a few minutes for the
black run, which seeks one frame in 60 of a long recording). The rows are
cached as CSV in ``--work-dir``; ``--skip-batch`` reuses the cache.

``--only name,name`` regenerates a subset (names are the PNG stems).

The ``absorbance`` group needs a THIRD run: a back-lit video whose first
frames show the empty cylinder (``--absorbance-run``, ``--absorbance-video``,
``--absorbance-params`` for the liquid/foam gradient parameters,
``--absorbance-ref-frames a:b`` for the reference frames). The green and
black runs are only required for the other groups.

Example::

    python scripts/make_readme_figures.py --green-run runs/green --green-video green.mp4 \\
        --black-run runs/black --black-video black.mp4 --out docs/images --panel-screenshot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cylvision.absorbance import (  # noqa: E402
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
from cylvision.absorbance.batch import DEFAULT_KS  # noqa: E402
from cylvision.absorbance.beer_lambert import S_CM2  # noqa: E402
from cylvision.calibration import (  # noqa: E402
    Calibration,
    build_click_prompts,
    build_model_fn,
    render_click_overlay,
    render_verification_figure,
)
from cylvision.detection import (  # noqa: E402
    DetectionParams,
    InterfaceResult,
    annotate_interfaces,
    detect_from_crop,
    make_gradient_panel,
    make_info_strip,
    make_label_strip,
    make_threshold_panel,
)
from cylvision.detection.interfaces import band_columns  # noqa: E402
from cylvision.detection.panels import (  # noqa: E402
    SEP,
    draw_readout,
    make_specimen_panel,
    readout_rows,
    readout_size,
    volume_labels,
)
from cylvision.ui import text as uitext  # noqa: E402
from cylvision.io import VideoSource, imread_unicode  # noqa: E402
from cylvision.pipeline import (  # noqa: E402
    RunDir,
    crop_frame,
    plot_levels,
    process_video,
    read_csv,
    summarize_rows,
    write_csv,
)
from cylvision.ui import theme  # noqa: E402
from cylvision.ui.tuner import compose_canvas  # noqa: E402
from cylvision.uncertainty import (  # noqa: E402
    build_budget,
    geometry_from_calibration,
    method_uncertainty_from_result,
    render_budget_figure,
    render_curvature_figure,
    render_cylinder_schematic,
    summarize_method_uncertainty,
)

MAX_WIDTH = 1600
MAX_BYTES = 1_500_000
SCREEN = (1920, 1080)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Frames of the two runs used throughout (see the README captions). The
# green run is the "BL evolution" recording, pre-subsampled 1 frame in 60:
# the pour starts at sub frame 22 (source frame 1347; nothing is detected
# before frame ~26, so "pouring" = 30), the foam peaks at 39,
# 108 is the reference frame, a 132 mL foam collapse happens between 380
# and 400, the recording ends at 440.
GREEN_FRAMES = {"empty": 8, "pouring": 30, "peak": 39, "best": 108, "later": 300,
                "before_collapse": 380, "after_collapse": 400, "end": 440}
GREEN_T0_FRAME = 22                 # default --t0-frame: times count from the start of the pour
BLACK_FRAMES = {"early": 1032, "best": 2472, "mid": 6972, "late": 20972}
BLACK_BATCH = (972, 40000, 60)      # start, end, step

# Absorbance run (back-lit, 1 frame = 60 source frames): the reference frames show the
# empty cylinder, the pour starts at source frame t0_frame, the measurement figure uses
# measurement_frame.
ABS_DEFAULTS: dict[str, Any] = {
    "ref_frames": "5:15", "t0_frame": 1347.0, "frame_scale": 60.0, "start": 23, "measurement_frame": 231,
    "k": 5.0, "r_dens": 64, "px_per_cm": None, "dead_thresh": 178.0, "dead_dilate": 2,   # None = from the calibration
}


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------

@dataclass
class RunCtx:
    """Everything the figures need from one run (frames are cached)."""

    name: str
    run: RunDir
    calib: Calibration
    params: DetectionParams
    model_fn: Callable[..., Any]
    video: VideoSource
    time_scale: float = 1.0
    t0_frame: int = 0               # frame whose time is 0 in the captions (start of the pour)
    _cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def frame(self, idx: int) -> np.ndarray:
        if idx not in self._cache:
            fr = self.video.frame(int(idx))
            if fr is None:
                raise OSError(f"cannot read frame {idx} of {self.video.path.name}")
            self._cache[idx] = fr
        return self._cache[idx]

    def t_sec(self, idx: int) -> float:
        """Time of frame ``idx`` in source seconds, counted from ``t0_frame``."""
        return (idx - self.t0_frame) * self.time_scale / self.video.fps

    def t_label(self, idx: int) -> str:
        t = self.t_sec(idx)
        if abs(t) < 600:
            return f"t = {t:.0f} s"
        return f"t = {t / 60:.1f} min"

    def detect(self, idx: int, params: DetectionParams | None = None
               ) -> tuple[np.ndarray, InterfaceResult, np.ndarray]:
        crop = crop_frame(self.frame(idx), self.calib)
        result, Gy = detect_from_crop(crop, params or self.params)
        return crop, result, Gy


def fill_from_calibration(params: DetectionParams, calib: Calibration) -> DetectionParams:
    """``y_bottom`` / ``cx`` from the calibration when the params leave them unset."""
    p = DetectionParams.from_dict(params.to_dict())
    if p.y_bottom is None:
        p.y_bottom = int(calib.y_bottom)
    if p.cx is None:
        p.cx = int(calib.cx())
    return p


def load_ctx(name: str, run_dir: Path, video_path: Path, params_path: Path | None,
             time_scale: float, t0_frame: int = 0) -> RunCtx:
    run = RunDir(run_dir)
    if not run.has_calibration():
        sys.exit(f"{name}: no calib.json in {run.root}")
    calib = run.load_calibration()
    p_path = params_path if params_path is not None else run.params_path
    if not p_path.exists():
        sys.exit(f"{name}: params file not found: {p_path}")
    raw = json.loads(p_path.read_text(encoding="utf-8"))
    params = fill_from_calibration(DetectionParams.from_dict(raw), calib)
    video = VideoSource(video_path)
    print(f"{name}: {video}  crop x [{calib.x_left}, {calib.x_right}]  polarity {params.polarity}")
    return RunCtx(name=name, run=run, calib=calib, params=params, model_fn=build_model_fn(calib),
                  video=video, time_scale=time_scale, t0_frame=int(t0_frame))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    s = width / w
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (int(width), max(1, int(round(h * s)))), interpolation=interp)


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    s = height / h
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, int(round(w * s))), int(height)), interpolation=interp)


def scale_img(img: np.ndarray, s: float) -> np.ndarray:
    """Uniform scale with the same rounding as ``annotate_interfaces(zoom=s)``."""
    h, w = img.shape[:2]
    tw, th = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if tw == w and th == h:
        return img
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (tw, th), interpolation=interp)


def caption_strip(width: int, lines: Sequence[str], colors: Sequence[tuple[int, int, int]] | None = None,
                  *, font_scale: float = 0.55, line_h: int = 24, pad: int = 6) -> np.ndarray:
    """Dark strip with one or more lines of text (theme colours)."""
    strip = np.empty((pad * 2 + line_h * len(lines), int(width), 3), dtype=np.uint8)
    strip[:] = theme.STRIP_BG_BGR
    y = pad + 17
    for i, text in enumerate(lines):
        color = colors[i] if colors is not None and i < len(colors) else theme.STRIP_TEXT_BGR
        cv2.putText(strip, text, (8, y), FONT, font_scale, color, 1, cv2.LINE_AA)
        y += line_h
    return strip


def pad_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] >= h:
        return img
    pad = np.empty((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    pad[:] = theme.BG_BGR
    return np.concatenate([img, pad], axis=0)


def side_by_side(imgs: Sequence[np.ndarray], labels: Sequence[str | Sequence[str]], *,
                 gap: int = 6) -> np.ndarray:
    """Labelled N-up composite: caption strip over each image, dark separators."""
    cols = []
    for img, lab in zip(imgs, labels):
        lines = [lab] if isinstance(lab, str) else list(lab)
        strip = caption_strip(img.shape[1], lines)
        cols.append(np.concatenate([strip, img], axis=0))
    h = max(c.shape[0] for c in cols)
    parts: list[np.ndarray] = []
    for i, c in enumerate(cols):
        if i:
            sep = np.empty((h, gap, 3), dtype=np.uint8)
            sep[:] = theme.SEPARATOR_BGR
            parts.append(sep)
        parts.append(pad_height(c, h))
    return np.concatenate(parts, axis=1)


def stack(imgs: Sequence[np.ndarray], *, gap: int = SEP) -> np.ndarray:
    w = max(i.shape[1] for i in imgs)
    parts: list[np.ndarray] = []
    for i, img in enumerate(imgs):
        if img.shape[1] < w:
            pad = np.empty((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
            pad[:] = theme.BG_BGR
            img = np.concatenate([img, pad], axis=1)
        if i:
            sep = np.empty((gap, w, 3), dtype=np.uint8)
            sep[:] = theme.SEPARATOR_BGR
            parts.append(sep)
        parts.append(img)
    return np.concatenate(parts, axis=0)


def text_panel(width: int, lines: Sequence[tuple[str, tuple[int, int, int]]], *, height: int | None = None,
               font_scale: float = 0.6, line_h: int = 30) -> np.ndarray:
    """Dark panel with coloured text lines (used next to a zoomed image)."""
    h = height if height is not None else 20 + line_h * len(lines)
    panel = np.empty((h, int(width), 3), dtype=np.uint8)
    panel[:] = theme.BG_BGR
    y = 34
    for text, color in lines:
        cv2.putText(panel, text, (16, y), FONT, font_scale, color, 1, cv2.LINE_AA)
        y += line_h
    return panel


def draw_mask_lines(img: np.ndarray, params: DetectionParams, s: float) -> None:
    H, W = img.shape[:2]
    for y in (params.y_top if params.y_top > 0 else None, params.y_bottom):
        if y is None:
            continue
        yy = int(round((y + 0.5) * s))
        if 0 <= yy < H:
            cv2.line(img, (0, yy), (W - 1, yy), theme.MASK_BGR, 1, cv2.LINE_AA)


def analysis_panels(crop: np.ndarray, result: InterfaceResult, params: DetectionParams, *,
                    scale: float = 1.0, rows: tuple[int, int] | None = None,
                    with_labels: bool = True, model_fn: Callable[..., Any] | None = None,
                    **overlay: Any) -> np.ndarray:
    """``[highlight | signed gradient | threshold mask]`` at ``scale``, rows ``[r0, r1)`` of the crop.

    The overlays are drawn after scaling so the lines, the wash and the zone
    labels keep their pixel size; ``model_fn`` prints the volumes in the
    highlight panel; ``overlay`` is forwarded to ``annotate_interfaces``.
    """
    p_high = annotate_interfaces(crop, result, params, zoom=scale, show_extras=False, model_fn=model_fn,
                                 **overlay)
    p_grad = scale_img(make_gradient_panel(result.S), scale)
    p_mask = scale_img(make_threshold_panel(result.S, params), scale)
    draw_mask_lines(p_grad, params, scale)
    draw_mask_lines(p_mask, params, scale)
    panels = [p_high, p_grad, p_mask]
    if rows is not None:
        r0 = max(0, int(round(rows[0] * scale)))
        r1 = min(p_high.shape[0], int(round(rows[1] * scale)))
        panels = [p[r0:r1] for p in panels]
    h, w = panels[0].shape[:2]
    sep = np.empty((h, SEP, 3), dtype=np.uint8)
    sep[:] = theme.SEPARATOR_BGR
    body = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    if not with_labels:
        return body
    return np.concatenate([make_label_strip(body.shape[1], [w, w, w]), body], axis=0)


def fmt(v: float | None, spec: str = "{:.1f}") -> str:
    return "n/a" if v is None or not math.isfinite(v) else spec.format(v)


# ---------------------------------------------------------------------------
# Output + manifest
# ---------------------------------------------------------------------------

class Manifest:
    """Collects ``README_figures.json`` entries and writes the PNGs."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.path = out_dir / "README_figures.json"
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                for e in json.loads(self.path.read_text(encoding="utf-8")).get("figures", []):
                    self.entries[e["file"]] = e
            except (ValueError, KeyError, TypeError):
                pass

    def save_png(self, name: str, img: np.ndarray, description: str, source: dict[str, Any], *,
                 max_w: int = MAX_WIDTH) -> Path:
        out = self.out_dir / f"{name}.png"
        if img.shape[1] > max_w:
            img = resize_to_width(img, max_w)
        buf = _encode(img)
        while buf.nbytes > MAX_BYTES and img.shape[1] > 700:
            img = resize_to_width(img, int(img.shape[1] * 0.9))
            buf = _encode(img)
        out.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(out))
        self.entries[out.name] = {
            "file": out.name,
            "size_px": [int(img.shape[1]), int(img.shape[0])],
            "bytes": int(buf.nbytes),
            "description": description,
            "source": source,
        }
        print(f"  {out.name:<34s} {img.shape[1]:>5d} x {img.shape[0]:<5d} {buf.nbytes / 1e6:5.2f} MB")
        return out

    def save_figure_png(self, name: str, tmp_png: Path, description: str, source: dict[str, Any],
                        *, max_w: int = MAX_WIDTH) -> Path:
        """Re-encode a matplotlib PNG to the README size limits."""
        img = imread_unicode(tmp_png)
        if img is None:
            raise OSError(f"cannot read back {tmp_png}")
        out = self.save_png(name, img, description, source, max_w=max_w)
        if tmp_png.resolve() != out.resolve():
            tmp_png.unlink(missing_ok=True)
        return out

    def write(self) -> None:
        data = {
            "note": "Every image comes from a real run; sources give the run, the frame and the parameters.",
            "figures": [self.entries[k] for k in sorted(self.entries)],
        }
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written: {self.path}")


def _encode(img: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf


def source_of(ctx: RunCtx, frame: int | None, params: DetectionParams | None = None,
              **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"run": ctx.name, "video": ctx.video.path.name, "run_dir": ctx.run.root.name}
    if frame is not None:
        d["frame"] = int(frame)
        d["t_sec"] = round(ctx.t_sec(frame), 2)
    if ctx.t0_frame:
        d["t0_frame"] = int(ctx.t0_frame)
        d["time_origin"] = f"t = 0 at frame {ctx.t0_frame} (start of the pour)"
    d["params"] = (params or ctx.params).to_dict()
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Batches (cached in the work dir)
# ---------------------------------------------------------------------------

def shift_time(rows: list[dict], ctx: RunCtx) -> list[dict]:
    """Move ``t_sec`` of batch rows to the caption time origin (``ctx.t0_frame``)."""
    if not ctx.t0_frame:
        return rows
    dt = ctx.t0_frame * ctx.time_scale / ctx.video.fps
    out = []
    for r in rows:
        r = dict(r)
        if r.get("t_sec") is not None and math.isfinite(float(r["t_sec"])):
            r["t_sec"] = float(r["t_sec"]) - dt
        out.append(r)
    return out


def run_batches(g: RunCtx, b: RunCtx, work_dir: Path, skip: bool,
                reuse: Sequence[str] = ()) -> dict[str, list[dict]]:
    """Batch rows of the three specs; ``skip`` reuses every cache, ``reuse`` only the named runs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    specs = {
        "green_step1": (g, 0, g.video.n_frames - 1, 1),
        "green_step10": (g, 0, g.video.n_frames - 1, 10),
        "black_step60": (b, BLACK_BATCH[0], min(BLACK_BATCH[1], b.video.n_frames - 1), BLACK_BATCH[2]),
    }
    for key, (ctx, start, end, step) in specs.items():
        csv_path = work_dir / f"{key}.csv"
        if skip or ctx.name in reuse:
            if csv_path.exists():
                out[key] = shift_time(read_csv(csv_path), ctx)
                print(f"batch {key}: {len(out[key])} rows read from {csv_path}")
            else:
                print(f"batch {key}: no cache at {csv_path} (skipped)")
            continue
        t0 = time.perf_counter()
        print(f"batch {key}: {ctx.video.path.name} frames {start}..{end} step {step}")
        rows = process_video(ctx.video, ctx.calib, ctx.params, ctx.model_fn, start=start, end=end,
                             frame_step=step, progress=True, time_scale=ctx.time_scale)
        write_csv(rows, csv_path)
        rows = shift_time(rows, ctx)
        out[key] = rows
        summ = summarize_rows(rows)
        print(f"  done in {time.perf_counter() - t0:.0f} s -> {csv_path}")
        for k in ("V_lower_mL", "V_total_mL", "V_foam_mL"):
            r = summ.get(k)
            if r:
                print(f"    {k:<11s} min {r['min']:7.1f}  max {r['max']:7.1f}  last {r['last']:7.1f} mL")
        s = summ["u_method_lower_mL"]
        print(f"    u_method lower: median {s['median']:.2f}  mean {s['mean']:.2f}  p95 {s['p95']:.2f} mL (n={s['n']})")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

HERO_HEIGHT = 820          # height (px) of the four hero panels before the final width limit
TIMELINE_HEIGHT = 520      # height (px) of every timeline panel (7 panels fit MAX_WIDTH without downscaling)
TIMELINE_GREEN = (30, 39, 108, 200, 300, 380, 400)   # frames of the green run (1 frame = 60 source frames)
GIF_FRAMES = (22, 30, 39, 50, 60, 80, 108, 150, 200, 250, 300, 350, 380, 400, 440)
GIF_HEIGHT = 620           # height (px) of every GIF frame
GIF_FRAME_MS = 550         # display time of a GIF frame; the last one stays 4x longer
GIF_MAX_BYTES = 5_000_000


def cylinder_window(ctx: RunCtx, frame_w: int, *, margin_left: int, margin_right: int) -> tuple[int, int]:
    """Frame columns ``[x0, x1)`` around the calibrated crop band."""
    x_left, x_right, _, _ = ctx.calib.crop()
    return max(0, x_left - margin_left), min(frame_w, x_right + 1 + margin_right)


def detected_panel(ctx: RunCtx, idx: int, *, height: int, margin_left: int, margin_right: int,
                   label_scale: float | None = None, pad_right: int | None = None
                   ) -> tuple[np.ndarray, InterfaceResult]:
    """Specimen panel (wash + thick lines + braces + volumes) around the cylinder.

    The frame is cropped to ``[x_left - margin_left, x_right + margin_right)``
    and extended on the right by a dark margin (``pad_right`` output px, sized
    for the longest label by default) so the brace labels never leave the
    picture. Returns the panel at ``height`` px and the detection.
    """
    frame = ctx.frame(idx)
    H, W = frame.shape[:2]
    _, res, _ = ctx.detect(idx)
    z = height / H
    fs = 1.0 if label_scale is None else float(label_scale)
    if pad_right is None:
        bw = readout_size([("liquid", "888 mL", None)], scale=fs)[0]
        pad_right = int(round(14 * fs)) + 6 + int(round(8 * fs)) + 4 + int(round(10 * fs)) + 2 + 5 + bw + 10
    x0, x1 = cylinder_window(ctx, W, margin_left=margin_left, margin_right=margin_right)
    pad_native = int(math.ceil(pad_right / z))
    extra = max(0, x1 + pad_native - W)
    if extra:
        pad = np.empty((H, extra, 3), dtype=np.uint8)
        pad[:] = theme.BG_BGR
        frame = np.concatenate([frame, pad], axis=1)
    panel = make_specimen_panel(frame, ctx.calib.crop(), res, zoom=z, model_fn=ctx.model_fn,
                                label_scale=fs)
    c0 = int(round(x0 * z))
    c1 = int(round((x1 + pad_native) * z))
    return np.ascontiguousarray(panel[:, c0:c1]), res


def raw_panel(ctx: RunCtx, idx: int, *, height: int, margin_left: int, margin_right: int) -> np.ndarray:
    frame = ctx.frame(idx)
    x0, x1 = cylinder_window(ctx, frame.shape[1], margin_left=margin_left, margin_right=margin_right)
    return resize_to_height(np.ascontiguousarray(frame[:, x0:x1]), height)


def zone_lines(res: InterfaceResult, model_fn: Callable[..., Any]) -> list[str]:
    """Caption lines of a detected panel: the three volumes (or what is missing)."""
    lab = volume_labels(res, model_fn)
    if "foam" in lab:
        return [f"{lab['foam']}   {lab['liquid']}", lab["total"]]
    if "liquid" in lab:
        return [lab["liquid"], "no foam / air interface"]
    return ["no interface detected", ""]


def fig_hero(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    """``hero.png``: raw frame -> detected, for both lighting setups; ``tuner_canvas_inline.png``."""
    gi, bi = GREEN_FRAMES["best"], BLACK_FRAMES["best"]
    margins = {"green": (150, 40), "black": (160, 40)}
    imgs, labels, src = [], [], {}
    for ctx, idx in ((g, gi), (b, bi)):
        ml, mr = margins[ctx.name]
        det, res = detected_panel(ctx, idx, height=HERO_HEIGHT, margin_left=ml, margin_right=mr)
        raw = raw_panel(ctx, idx, height=HERO_HEIGHT, margin_left=ml, margin_right=mr)
        setup = "back-lit green screen" if ctx.name == "green" else "front-lit black background"
        imgs += [raw, det]
        labels += [["raw frame", setup, f"frame {idx}, {ctx.t_label(idx)}"],
                   ["detected", *zone_lines(res, ctx.model_fn)]]
        src[ctx.name] = source_of(ctx, idx, y_lower=res.y_lower, y_upper=res.y_upper,
                                  n_lower=res.n_lower, n_upper=res.n_upper)
    m.save_png("hero", side_by_side(imgs, labels, gap=10),
               "Raw frame -> detected, on the green-screen run (left pair) and the black-background run "
               "(right pair): flat red / teal mean lines at the liquid/foam and foam/air interfaces, "
               "peach wash on the foam band, blue wash on the liquid, readout boxes with the volumes.",
               {"runs": src, "layout": "raw | detected | raw | detected, cropped around the cylinder"})

    canvas = compose_canvas(g.frame(gi), g.calib, g.params, g.model_fn, show_specimen=True, inline=True,
                            screen_size=SCREEN, frame_idx=gi, t_sec=g.t_sec(gi), fps=g.video.fps)
    m.save_png("tuner_canvas_inline", canvas,
               "Tuner canvas (specimen + highlight + signed gradient + threshold mask + info strip) on the "
               "green-screen run, inline layout, 1920x1080 screen.",
               source_of(g, gi, layout="inline, specimen on"))


def fig_timeline(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    """``timeline_green.png`` (static strip) and ``detection_timeline.gif`` (animated)."""
    imgs, labels, frames = [], [], []
    for idx in TIMELINE_GREEN:
        if idx >= g.video.n_frames:
            continue
        det, res = detected_panel(g, idx, height=TIMELINE_HEIGHT, margin_left=40, margin_right=8,
                                  label_scale=0.85)
        imgs.append(det)
        lab = volume_labels(res, g.model_fn)
        t = g.t_sec(idx)
        labels.append([f"frame {idx}   {g.t_label(idx)}",
                       lab.get("foam", "pouring: foam top only" if res.y_upper is not None else "no interface"),
                       lab.get("liquid", "")])
        frames.append({"frame": idx, "t_sec": round(t, 2), "y_lower": res.y_lower, "y_upper": res.y_upper,
                       "n_lower": res.n_lower, "n_upper": res.n_upper})
    m.save_png("timeline_green", side_by_side(imgs, labels, gap=8),
               f"{len(imgs)} frames of the green-screen run, each with the detected interfaces, the foam wash "
               "and the readouts: the pour, the foam peak, the band shrinking while the liquid drains out of "
               "it, and the collapse between frames 380 and 400. Times count from the start of the pour.",
               source_of(g, None, frames=frames))
    fig_gif(g, m)


def gif_frame(ctx: RunCtx, idx: int) -> tuple[np.ndarray, InterfaceResult]:
    """One GIF frame: a header with the readout box (t + the three volumes) over the specimen panel."""
    det, res = detected_panel(ctx, idx, height=GIF_HEIGHT, margin_left=54, margin_right=10, label_scale=0.85)
    t = ctx.t_sec(idx)
    t_txt = f"{t:.0f} s" if abs(t) < 600 else f"{t / 60:.1f} min"
    rows = [("time", t_txt, None)]
    vol = readout_rows(res, ctx.model_fn)
    for key in ("total", "foam", "liquid"):
        rows.append(vol.get(key, (key, "--", None)))
    w_box = readout_size([("liquid", "8888 mL", None)], scale=0.85)[0]
    box_h = readout_size(rows, scale=0.85)[1]
    header = np.empty((box_h + 12, det.shape[1], 3), dtype=np.uint8)
    header[:] = theme.BG_BGR
    draw_readout(header, (header.shape[1] - 6, 6), rows, "rt", scale=0.85, min_width=w_box)
    uitext.draw_text(header, f"frame {idx}", (8, 8), theme.LABEL_FONT_ZONE, 12, theme.TEXT_BGR, "lt")
    uitext.draw_text(header, "green screen", (8, 26), theme.LABEL_FONT_ZONE, 11, theme.SUBTEXT_BGR, "lt")
    uitext.draw_text(header, "1 frame = 60 src", (8, 42), theme.LABEL_FONT_ZONE, 11, theme.SUBTEXT_BGR, "lt")
    return np.concatenate([header, det], axis=0), res


def fig_gif(g: RunCtx, m: Manifest) -> None:
    import imageio.v3 as iio

    frames, meta = [], []
    for idx in GIF_FRAMES:
        if idx >= g.video.n_frames:
            continue
        img, res = gif_frame(g, idx)
        frames.append(img)
        meta.append({"frame": idx, "t_sec": round(g.t_sec(idx), 2), "y_lower": res.y_lower,
                     "y_upper": res.y_upper})
    if not frames:
        return
    w = max(f.shape[1] for f in frames)
    h = max(f.shape[0] for f in frames)
    frames = [_pad_to(f, w, h) for f in frames]
    durations = [GIF_FRAME_MS] * len(frames)
    durations[-1] = GIF_FRAME_MS * 4
    out = m.out_dir / "detection_timeline.gif"
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    scale = 1.0
    while True:
        data = iio.imwrite("<bytes>", rgb, extension=".gif", duration=durations, loop=0)
        if len(data) <= GIF_MAX_BYTES or scale < 0.5:
            break
        scale *= 0.85
        rgb = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA) for f in frames]
    out.write_bytes(data)
    m.entries[out.name] = {
        "file": out.name,
        "size_px": [int(w * scale), int(h * scale)],
        "bytes": len(data),
        "description": f"Animated timeline of the green-screen run: {len(frames)} frames from the pour to "
                       "after the foam collapse, each the specimen panel (wash, flat lines, readouts) with a "
                       f"readout box giving t and the three volumes; {GIF_FRAME_MS} ms per frame, the last one "
                       "held longer. Times count from the start of the pour.",
        "source": source_of(g, None, frames=meta, frame_ms=GIF_FRAME_MS),
    }
    print(f"  {out.name:<34s} {int(w * scale):>5d} x {int(h * scale):<5d} {len(data) / 1e6:5.2f} MB  ({len(frames)} frames)")


def _pad_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = theme.BG_BGR
    out[:img.shape[0], :img.shape[1]] = img
    return out


def fig_setup(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    m.save_png("setup_black_background", resize_to_height(b.frame(BLACK_FRAMES["best"]), 700),
               "Raw frame of the front-lit black-background setup (bright foam over dark beer).",
               source_of(b, BLACK_FRAMES["best"], params=None) | {"params": None})
    m.save_png("setup_green_screen", resize_to_height(g.frame(GREEN_FRAMES["best"]), 700),
               "Raw frame of the back-lit green-screen setup (dark foam, translucent beer).",
               source_of(g, GREEN_FRAMES["best"]) | {"params": None})


def polarity_figure(ctx: RunCtx, idx: int, scale: float) -> np.ndarray:
    crop, result, _ = ctx.detect(idx)
    p = ctx.params
    rows = (max(0, p.y_top - 10), min(crop.shape[0], (p.y_bottom or crop.shape[0]) + 10))
    body = analysis_panels(crop, result, p, scale=scale, rows=rows, model_fn=ctx.model_fn)
    info = make_info_strip(body.shape[1], result, p, ctx.model_fn, idx, ctx.t_sec(idx), None)
    return stack([body, info])


def fig_polarity(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    m.save_png("polarity_lower_darker", polarity_figure(b, BLACK_FRAMES["best"], 1.0),
               "Analysis panels + info strip on the black-background run with polarity lower_darker: "
               "the lower (beer/foam) interface is a negative Gy peak, the upper (foam/air) a positive one.",
               source_of(b, BLACK_FRAMES["best"]))
    m.save_png("polarity_lower_brighter", polarity_figure(g, GREEN_FRAMES["best"], 0.65),
               "Analysis panels + info strip on the green-screen run with polarity lower_brighter: "
               "the beer transmits the back light, the foam scatters it, the signs are flipped.",
               source_of(g, GREEN_FRAMES["best"]))


def fig_polarity_wrong(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    idx = GREEN_FRAMES["best"]
    crop, ref, _ = g.detect(idx)
    rows = window(ref, crop.shape[0], 260, 300)
    imgs, labels = [], []
    for pol, tag in (("lower_darker", "WRONG for a back-lit green screen"), ("lower_brighter", "correct")):
        p = replace(g.params, polarity=pol)
        _, res, _ = g.detect(idx, p)
        imgs.append(analysis_panels(crop, res, p, scale=0.62, rows=rows, model_fn=g.model_fn))
        labels.append([f"polarity = {pol}   ({tag})", *detection_lines(res, g.model_fn)])
    m.save_png("polarity_wrong", side_by_side(imgs, labels),
               "Same green-screen frame processed with the wrong polarity (left) and the right one (right): "
               "with lower_darker the detector locks on the wrong sign of the gradient.",
               source_of(g, idx, variants=["lower_darker", "lower_brighter"]))


def window(ref: InterfaceResult, H: int, top_pad: int | None, bot_pad: int | None) -> tuple[int, int]:
    """Rows ``[y_upper - top_pad, y_lower + bot_pad)`` of the reference detection.

    ``None`` extends the window to the top (resp. bottom) of the crop; without
    any detection the whole crop is returned.
    """
    yu = ref.y_upper if ref.y_upper is not None else ref.y_lower
    yl = ref.y_lower if ref.y_lower is not None else ref.y_upper
    if yu is None or yl is None:
        return 0, H
    r0 = 0 if top_pad is None else int(max(0, yu - top_pad))
    r1 = H if bot_pad is None else int(min(H, yl + bot_pad))
    return r0, r1


def detection_lines(res: InterfaceResult, model_fn: Callable[..., Any]) -> list[str]:
    """Two caption lines: the lower and the upper interface (row, columns, volume)."""
    def V(y: float | None) -> str:
        return "n/a" if y is None else f"{float(np.asarray(model_fn(np.array([y]))).ravel()[0]):.0f} mL"
    return [f"lower: y = {fmt(res.y_lower)} px   {res.n_lower} cols   V = {V(res.y_lower)}",
            f"upper: y = {fmt(res.y_upper)} px   {res.n_upper} cols   V = {V(res.y_upper)}"]


def param_composite(ctx: RunCtx, idx: int, name: str, values: Sequence[Any], *, scale: float,
                    rows: tuple[int, int], notes: Sequence[str] | None = None,
                    setter: Callable[[DetectionParams, Any], DetectionParams] | None = None
                    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """N-up composite of the analysis panels for several values of one parameter."""
    crop = crop_frame(ctx.frame(idx), ctx.calib)
    imgs, labels, variants = [], [], []
    for i, val in enumerate(values):
        p = setter(ctx.params, val) if setter else replace(ctx.params, **{name: val})
        _, res, _ = ctx.detect(idx, p)
        imgs.append(analysis_panels(crop, res, p, scale=scale, rows=rows, model_fn=ctx.model_fn))
        note = f"   ({notes[i]})" if notes and i < len(notes) and notes[i] else ""
        labels.append([f"{name} = {val}{note}", *detection_lines(res, ctx.model_fn)])
        variants.append({name: val, "y_lower": res.y_lower, "y_upper": res.y_upper,
                         "n_lower": res.n_lower, "n_upper": res.n_upper})
    return side_by_side(imgs, labels), variants


def fig_params(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    gi, bi = GREEN_FRAMES["best"], BLACK_FRAMES["best"]
    g_crop, g_ref, _ = g.detect(gi)
    b_crop, b_ref, _ = b.detect(bi)
    Hg, Hb = g_crop.shape[0], b_crop.shape[0]
    g_win = window(g_ref, Hg, 230, 230)
    b_win = window(b_ref, Hb, 110, 110)
    yu_g = int(g_ref.y_upper) if g_ref.y_upper is not None else 0
    g_top = (max(0, yu_g - 150), min(Hg, yu_g + 250))      # around the foam top
    W_g = g_crop.shape[1]

    specs: list[tuple[str, RunCtx, int, str, Sequence[Any], float, tuple[int, int], Sequence[str] | None, str]] = [
        ("param_T_lower", g, gi, "T_lower", [4, 40], 0.62, g_win,
         ["too low: weak gradients pass", "clean"],
         "Threshold of the lower (liquid/foam) interface: too low validates weak gradients in every "
         "column (false positives in the mask), the tuned value keeps only the true interface."),
        ("param_T_upper", b, bi, "T_upper", [4, 14], 0.8, b_win,
         ["too low: bottom-up search stops on noise", "clean"],
         "Threshold of the upper (foam/air) interface: the bottom-up search from the lower interface stops "
         "on the first row above threshold, so a low value catches bubble edges right above the beer."),
        ("param_r_lower", g, gi, "r_lower", [10, 64, W_g // 2], 0.5, g_win,
         ["narrow band", "medium band", "tuned: whole crop width"],
         "Half-width of the averaging band of the lower interface: a narrow band uses few columns and "
         "follows the local bend of the interface; the whole width (tuned here) averages every column "
         "that passes the threshold, the walls being excluded by the threshold itself."),
        ("param_blur_sigma", g, gi, "blur_sigma", [0.5, 5.7], 0.62, g_win,
         ["almost no smoothing: bubble edges win", "tuned"],
         "Gaussian smoothing: with a small sigma every bubble edge competes with the interface and the "
         "per-column detections scatter; the tuned value spreads the interface over a few rows."),
        ("param_blur_h", b, bi, "blur_h", [0, 38], 0.8, b_win,
         ["off: foam texture in the gradient", "on: texture washed out"],
         "Horizontal box filter: it averages every row along x, keeping the horizontal interfaces and "
         "washing out the vertical bubble texture of the foam."),
        ("param_min_h_upper", g, gi, "min_h_upper", [0, 15], 0.62, g_top,
         ["off: thin bubble edges accepted", "on: blobs < 15 px tall rejected"],
         "Connected-component height filter of the upper interface: without it the bottom-up search stops "
         "on thin bubble edges inside the foam, below the true foam top; requiring blobs at least 15 rows "
         "tall keeps the foam top (the reference frame, foam full of large bubbles under a bubbly top)."),
        ("param_y_top", b, bi, "y_top", [200, 34], 0.8, window(b_ref, Hb, None, 110),
         ["too low: the foam top itself is masked", "tuned: rows above 34 ignored"],
         "Top gradient mask: rows above y_top are zeroed (magenta line). The glass rim of both runs lies "
         "outside the frame, so the mask is shown the other way round: a y_top below the foam top masks "
         "the true upper interface and the detector reports nothing above threshold."),
        ("param_y_bottom", b, bi, "y_bottom", [Hb - 1, 800, 1031], 0.7, window(b_ref, Hb, 110, None),
         ["no mask", "too high: true interface masked", "tuned"],
         "Bottom gradient mask: rows below y_bottom are zeroed (magenta line). Without it the bright edge of "
         "the base shows up as a teal blob in the threshold mask (an upper-interface candidate, harmless here "
         "but picked whenever the lower interface is lost); a y_bottom above the beer/foam interface masks "
         "the true lower interface."),
        ("param_cx", g, gi, "cx", [40, g.params.cx], 0.62, g_win,
         ["axis shifted to the left wall", "axis on the cylinder centre"],
         "Axis column inside the crop: the band is centred on cx, so a wrong cx samples the interface "
         "near a wall (meniscus, refraction) instead of the centre."),
        ("param_channel", g, gi, "channel", ["gray", "G", "R"], 0.5, g_win,
         ["tuned", "more columns, foam top lower", "almost no signal"],
         "Grey level used by the detector: gray (tuned) and the green channel agree on the lower interface; "
         "G validates more upper columns but stops on bubble edges below the foam top; the red channel "
         "carries almost no signal on a green screen."),
    ]
    for name, ctx, idx, pname, values, scale, rows, notes, desc in specs:
        img, variants = param_composite(ctx, idx, pname, values, scale=scale, rows=rows, notes=notes)
        m.save_png(name, img, desc, source_of(ctx, idx, variants=variants, rows_window=list(rows)))


def fig_calibration_clicks(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    idx = GREEN_FRAMES["empty"]
    frame = g.frame(idx)
    H = frame.shape[0]
    c = g.calib
    prompts = build_click_prompts(c.cylinder.graduations_ml)
    x_mid = (c.x_left + c.x_right) // 2
    order = np.argsort(c.graduations_ml)
    grads = [(float(c.graduations_ml[i]), int(round(c.graduations_px_x[i])), int(round(c.graduations_px_y[i])))
             for i in order]
    pts: list[tuple[int, int] | None] = [(int(c.x_left), H // 2), (int(c.x_right), H // 2),
                                         (x_mid, int(c.y_top)), (x_mid, int(c.y_bottom))]
    pts += [(x, y) for _V, x, y in grads[:3]]
    cursor = (grads[3][1], grads[3][2])
    img = render_click_overlay(frame, prompts, pts, cursor, with_magnifier=True)
    m.save_png("calibration_clicks", resize_to_width(img, min(900, img.shape[1])),
               "Calibration click window after the 4 ROI clicks (left / right edges, y_top, y_bottom) and "
               "the first 3 graduations; the cursor and the x8 loupe sit on the next graduation (400 mL).",
               source_of(g, idx) | {"params": None, "placed": {"roi": pts[:4], "graduations_ml": [v for v, _, _ in grads[:3]]},
                                    "cursor": list(cursor)})


def fig_calibration_check(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    for ctx, idx, name in ((g, GREEN_FRAMES["empty"], "calibration_check_green"),
                           (b, BLACK_FRAMES["early"], "calibration_check_black")):
        tmp = m.out_dir / f"{name}.png"
        render_verification_figure(ctx.frame(idx), ctx.calib, tmp)
        m.save_figure_png(name, tmp,
                          f"Calibration verification figure of the {ctx.name} run: annotated frame with the "
                          "predicted graduation lines, crop band, V(y) of the three models and residuals.",
                          source_of(ctx, idx) | {"params": None, "recommended_model": ctx.calib.recommended_model})


def fig_control_panel(g: RunCtx, b: RunCtx, m: Manifest, panel_screenshot: bool = False, **_: Any) -> None:
    if not panel_screenshot:
        print("  control_panel.png skipped (pass --panel-screenshot)")
        return
    from cylvision.ui.controls_panel import ControlsPanel

    idx = GREEN_FRAMES["best"]
    H = g.frame(idx).shape[0]
    panel = ControlsPanel(params=g.params, H=H, W_crop=g.calib.crop_width(), n_frames=g.video.n_frames,
                          init_frame_idx=idx, include_sampling=True, show_batch_button=True,
                          two_columns=True, frame_step=1)
    img = None
    try:
        try:
            panel.root.attributes("-topmost", True)
            panel.root.lift()
        except Exception:
            pass
        for _ in range(12):
            panel.update()
            time.sleep(0.05)
        img = panel.grab_screenshot()
    finally:
        panel.destroy()
    if img is None:
        print("  control_panel.png: screenshot unavailable (PIL missing or window not mapped)")
        return
    m.save_png("control_panel", img,
               "The Tk control panel (two-column layout) loaded with the green-screen parameters: VIEW, "
               "CYLINDER, GRADIENT and SAMPLING sections, Run batch / Abort buttons, hotkeys and summary.",
               source_of(g, idx, layout="two columns"))


def fig_tuner_stacked(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    idx = BLACK_FRAMES["best"]
    canvas = compose_canvas(b.frame(idx), b.calib, b.params, b.model_fn, show_specimen=True, inline=False,
                            screen_size=SCREEN, frame_idx=idx, t_sec=b.t_sec(idx), fps=b.video.fps)
    m.save_png("tuner_canvas_stacked", canvas,
               "Tuner canvas in the stacked layout (specimen + highlight on the first row, signed gradient + "
               "threshold mask on the second) on the black-background run.",
               source_of(b, idx, layout="stacked, specimen on"))


def fig_frame_step(g: RunCtx, b: RunCtx, m: Manifest, batches: dict[str, list[dict]] | None = None,
                   **_: Any) -> None:
    if not batches or "green_step1" not in batches or "green_step10" not in batches:
        print("  frame_step_illustration.png skipped (no green batch rows)")
        return
    fig = Figure(figsize=(10, 4.2), dpi=150, facecolor="#ffffff")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for key, label, color, style in (("green_step1", "frame_step = 1 (every frame)", "#2a78d6", "-"),
                                     ("green_step10", "frame_step = 10 (1 frame in 10)", "#eb6834", "o")):
        rows = batches[key]
        t = np.array([r["t_sec"] for r in rows], dtype=float)
        V = np.array([r["V_lower_mL"] for r in rows], dtype=float)
        ok = np.isfinite(t) & np.isfinite(V)
        if style == "-":
            ax.plot(t[ok], V[ok], color=color, lw=1.4, label=label)
        else:
            ax.plot(t[ok], V[ok], ls="", marker="o", ms=3.2, mfc="none", mec=color, mew=1.0, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("V_lower  (mL)")
    ax.set_title("Sub-sampling the video does not change the level curve", loc="left", fontsize=11)
    ax.grid(True, color="#e4e4e4")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    tmp = m.out_dir / "frame_step_illustration.png"
    fig.savefig(tmp, dpi=150, facecolor="#ffffff")
    m.save_figure_png("frame_step_illustration", tmp,
                      "V_lower(t) of the green-screen batch at frame_step 1 (line) and 10 (circles): the "
                      "sub-sampled curve sits on the full one.",
                      source_of(g, None, batch="frames 0..%d step 1 and step 10, time_scale %g" %
                                (g.video.n_frames - 1, g.time_scale)))


def budget_of(ctx: RunCtx, rows: list[dict] | None) -> Any:
    geom = geometry_from_calibration(ctx.calib)
    return geom, build_budget(ctx.calib, geom, ctx.params, ctx.model_fn, rows or None, r_px=ctx.params.r_lower)


def fig_batch(g: RunCtx, b: RunCtx, m: Manifest, batches: dict[str, list[dict]] | None = None,
              **_: Any) -> None:
    for ctx, key, name, rng in ((g, "green_step1", "batch_figure_green",
                                 f"frames 0..{g.video.n_frames - 1} step 1 (1 frame = {g.time_scale:g} source frames)"),
                                (b, "black_step60", "batch_figure_black",
                                 f"frames {BLACK_BATCH[0]}..{BLACK_BATCH[1]} step {BLACK_BATCH[2]}")):
        if not batches or key not in batches:
            print(f"  {name}.png skipped (no batch rows)")
            continue
        rows = batches[key]
        geom, budget = budget_of(ctx, rows)
        summ = summarize_rows(rows)
        tmp = m.out_dir / f"{name}.png"
        plot_levels(rows, title=f"{ctx.video.path.name}  -  {rng}", out_png=tmp, budget=budget)
        m.save_figure_png(name, tmp,
                          f"V_lower(t), V_total(t), V_foam(t) of the {ctx.name} batch with the ±u_total bands "
                          "of the uncertainty budget (calibration, pixel, curvature, method); u_total is about "
                          "3 mL, so the bands are barely wider than the lines at this scale.",
                          source_of(ctx, None, batch=rng, summary=summ,
                                    geometry=geom.summary(), budget_notes=budget.notes))


def fig_uncertainty(g: RunCtx, b: RunCtx, m: Manifest, batches: dict[str, list[dict]] | None = None,
                    **_: Any) -> None:
    idx = GREEN_FRAMES["best"]
    rows = batches.get("green_step1") if batches else None
    geom, budget = budget_of(g, rows)
    r_px = int(g.params.r_lower)
    grads = [float(v) for v in g.calib.graduations_ml]
    src = source_of(g, idx, r_px=r_px, geometry=geom.summary(), budget_notes=budget.notes)

    tmp = m.out_dir / "uncertainty_schematic.png"
    render_cylinder_schematic(geom, g.calib, r_px, tmp)
    m.save_figure_png("uncertainty_schematic", tmp,
                      "Why a horizontal graduation circle has a vertical extent in the image: top view, side "
                      "view and image-plane ellipse with the Delta bracket (green-screen geometry).", src)
    tmp = m.out_dir / "uncertainty_curvature.png"
    render_curvature_figure(g.frame(idx), g.calib, geom, r_px, g.model_fn, tmp)
    m.save_figure_png("uncertainty_curvature", tmp,
                      "Projected graduation circles on the green-screen frame, u_curvature(V) with the Delta "
                      "wedge and the camera horizon, top and side views.", src)
    tmp = m.out_dir / "uncertainty_budget.png"
    render_budget_figure(budget, (min(grads), max(grads)), tmp)
    m.save_figure_png("uncertainty_budget", tmp,
                      "Uncertainty budget of the green-screen run: the four contributions and the quadrature "
                      "total vs V, and their share of the variance.",
                      src | {"table": budget.table([100, 250, 500, 750, 1000])})


def fig_uncertainty_method(g: RunCtx, b: RunCtx, m: Manifest, **_: Any) -> None:
    idx = GREEN_FRAMES["best"]
    crop, res, _ = g.detect(idx)
    mu = method_uncertainty_from_result(res, g.model_fn, which="lower")
    zoom = 3.0
    img = annotate_interfaces(crop, res, g.params, zoom=zoom, show_extras=False,
                              y_bounds=(mu["y_up"], mu["y_lo"]), line_w=3, show_labels=False)
    half = 60
    r0 = int(max(0, (mu["y_up"] - half) * zoom))
    r1 = int(min(img.shape[0], (mu["y_lo"] + half) * zoom))
    zoomed = img[r0:r1]
    lines = [
        ("Method uncertainty, lower interface", theme.TEXT_BGR),
        (f"frame {idx}, t = {g.t_sec(idx):.1f} s, band r = {g.params.r_lower} px", theme.SUBTEXT_BGR),
        ("", theme.TEXT_BGR),
        (f"y_mean  = {mu['y_mean']:.2f} px   ->  V = {mu['V_mean']:.1f} mL", theme.LOWER_BGR),
        (f"y_up    = {mu['y_up']:.0f} px   ->  V = {mu['V_up']:.1f} mL   (yellow)", theme.BOUND_UPPER_BGR),
        (f"y_lo    = {mu['y_lo']:.0f} px   ->  V = {mu['V_lo']:.1f} mL   (cyan)", theme.BOUND_LOWER_BGR),
        (f"columns kept: {mu['n_kept']} / {res.n_lower}   (|V - V_mean| <= 7 mL)", theme.SUBTEXT_BGR),
        ("", theme.TEXT_BGR),
        (f"range   = |V_up - V_lo| = {mu['range_ml']:.2f} mL", theme.TEXT_BGR),
        (f"u_method = range / (2 sqrt 3) = {mu['u_ml']:.2f} mL", theme.TEXT_BGR),
        ("", theme.TEXT_BGR),
        ("red dots: per-column detections, red line: mean", theme.SUBTEXT_BGR),
    ]
    panel = text_panel(640, lines, height=max(zoomed.shape[0], 20 + 30 * len(lines)), font_scale=0.55)
    out = side_by_side([zoomed, panel], [f"lower interface, zoom x{zoom:g}  (rows {r0 / zoom:.0f}..{r1 / zoom:.0f})",
                                         "per-frame empirical spread"])
    m.save_png("uncertainty_method", out,
               "Zoom on the lower interface of the green-screen frame with the per-column detections, the mean "
               "line and the dashed empirical bounds, next to the numbers of the method uncertainty.",
               source_of(g, idx, method={k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                         for k, v in mu.items()}))


def fig_absorbance(g: RunCtx | None, b: RunCtx | None, m: Manifest, absorbance: dict[str, Any] | None = None,
                   **_: Any) -> None:
    """``heatmap_A``, ``heatmap_c``, ``absorbance_measurement``: the back-lit run through the absorbance analysis."""
    if not absorbance:
        print("  absorbance figures skipped (no --absorbance-run / --absorbance-video)")
        return
    ctx: RunCtx = absorbance["ctx"]
    o = absorbance["opts"]
    a, b_ref = (int(v) for v in str(o["ref_frames"]).split(":"))
    px_per_cm = (px_per_cm_from_calibration(ctx.calib, S_CM2) if o["px_per_cm"] is None
                 else float(o["px_per_cm"]))
    abs_params = AbsorbanceParams(channel="G", light_blur=1.0, r_dens=int(o["r_dens"]), foam_k=float(o["k"]),
                                  dead_thresh=float(o["dead_thresh"]), dead_dilate=int(o["dead_dilate"]),
                                  px_per_cm=px_per_cm, y_top=int(ctx.params.y_top), cx=int(ctx.params.cx))
    print(f"  px_per_cm = {px_per_cm:.2f}")
    x_left, x_right, _, _ = ctx.calib.crop()
    ref = reference_from_video(ctx.video, a, b_ref, crop=(x_left, x_right))
    prep = prepare_reference(ref, abs_params)
    m_frame = int(o["measurement_frame"])
    t0 = time.perf_counter()
    end = ctx.video.n_frames - 1
    run, maps = process_video_absorbance(ctx.video, ctx.calib, ctx.params, abs_params, prep, ctx.model_fn,
                                         start=int(o["start"]), end=end, frame_step=1,
                                         t0_frame=float(o["t0_frame"]), time_scale=ctx.time_scale,
                                         progress=False, keep_maps=(m_frame,))
    finalize_mass_balance(run, None, abs_params.section_cm2)
    print(f"  absorbance: {run.n} frames in {time.perf_counter() - t0:.0f} s, V_inf = {run.V_inf_ml:.2f} mL, "
          f"foam top on {int(np.isfinite(run.y_foam_top).sum())} frames")
    ks = tuple(sorted(set(DEFAULT_KS) | {int(o["k"])}))
    tops = foam_tops_for_ks(run, ks, y_top=abs_params.y_top)
    y_range = volume_window(ctx.calib, run.A.shape[1])
    i_m = int(np.where(run.frame_idx == m_frame)[0][0])
    sweep = foam_volume_vs_k(run, i_m, ctx.model_fn, ks, y_top=abs_params.y_top)
    src = source_of(ctx, m_frame, absorbance_params=abs_params.to_dict(), ref_frames=[a, b_ref],
                    t0_frame=float(o["t0_frame"]), t_since_pour_s=round(float(run.t_sec[i_m]), 2),
                    V_inf_ml=float(run.V_inf_ml),
                    frames=f"{int(o['start'])}..{end} step 1",
                    k_sweep={str(k): {"y_foam_top_px": y, "V_foam_mL": v} for k, (y, v) in sweep.items()},
                    n_dead_pixels=int(prep.n_dead))
    name = ctx.video.path.name
    tmp = m.out_dir / "heatmap_A.png"
    plot_heatmap_A(run.A, run.t_sec, model_fn=ctx.model_fn, y_range=y_range, out_png=tmp,
                   y_liquid=run.y_liquid, foam_tops=tops, k_main=int(o["k"]), V_inf_ml=run.V_inf_ml,
                   title=f"Effective absorbance A(t, V) - {name}, band |x - cx| <= {abs_params.r_dens} px, "
                         f"reference frames {a}..{b_ref - 1}")
    m.save_figure_png("heatmap_A", tmp,
                      "A(t, V) of the back-lit run: radial mean of the absorbance per row, rows converted to mL "
                      "through the calibration; liquid/foam interface (red), foam top for k = 2..10 with the "
                      "k used in teal, final liquid volume dotted.", src)
    tmp = m.out_dir / "heatmap_c.png"
    plot_heatmap_c(run.c, run.t_sec, model_fn=ctx.model_fn, y_range=y_range, out_png=tmp,
                   y_liquid=run.y_liquid, y_foam_top=run.y_foam_top,
                   title=f"Liquid content of the foam c(t, V) = γ A - {name}, V∞ = {run.V_inf_ml:.1f} mL "
                         f"(mass balance)")
    m.save_figure_png("heatmap_c", tmp,
                      "c(t, V) = gamma A: liquid volume fraction inside the foam from the mass balance "
                      "(V_inf - V_beer = S integral c dz), inverted grey, white outside the foam.", src)
    res = maps[m_frame]
    band = band_columns(res.A.shape[1], abs_params.resolved_cx(res.A.shape[1]), abs_params.r_dens)
    tmp = m.out_dir / "absorbance_measurement.png"
    plot_measurement_figure(prep.raw, res.intensity, res.A, res.A_z, y_liquid=res.y_liquid,
                            y_foam_top=res.y_foam_top, A_max=res.A_max, threshold=res.threshold, band=band,
                            model_fn=ctx.model_fn, y_range=y_range, out_png=tmp, k=abs_params.foam_k,
                            title=f"How the absorbance is measured - frame {m_frame} "
                                  f"(t = {run.t_sec[i_m]:.0f} s after the start of the pour)")
    m.save_figure_png("absorbance_measurement", tmp,
                      "Reference frame of the empty cylinder, live frame, absorbance map A(x, y) and the radial "
                      "profile A(z) with the threshold A_max / k and both interfaces.",
                      src | {"result": res.to_dict()})


FIGURES: dict[str, tuple[Callable[..., None], tuple[str, ...]]] = {
    "hero": (fig_hero, ("hero", "tuner_canvas_inline")),
    "timeline": (fig_timeline, ("timeline_green", "detection_timeline")),
    "setup": (fig_setup, ("setup_black_background", "setup_green_screen")),
    "polarity": (fig_polarity, ("polarity_lower_darker", "polarity_lower_brighter")),
    "polarity_wrong": (fig_polarity_wrong, ("polarity_wrong",)),
    "calibration_clicks": (fig_calibration_clicks, ("calibration_clicks",)),
    "calibration_check": (fig_calibration_check, ("calibration_check_green", "calibration_check_black")),
    "control_panel": (fig_control_panel, ("control_panel",)),
    "tuner_stacked": (fig_tuner_stacked, ("tuner_canvas_stacked",)),
    "params": (fig_params, ("param_T_lower", "param_T_upper", "param_r_lower", "param_blur_sigma",
                            "param_blur_h", "param_min_h_upper", "param_y_top", "param_y_bottom",
                            "param_cx", "param_channel")),
    "frame_step": (fig_frame_step, ("frame_step_illustration",)),
    "batch": (fig_batch, ("batch_figure_green", "batch_figure_black")),
    "uncertainty": (fig_uncertainty, ("uncertainty_schematic", "uncertainty_curvature", "uncertainty_budget")),
    "uncertainty_method": (fig_uncertainty_method, ("uncertainty_method",)),
    "absorbance": (fig_absorbance, ("heatmap_A", "heatmap_c", "absorbance_measurement")),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--green-run", type=Path, default=None, help="run dir of the green-screen video")
    p.add_argument("--green-video", type=Path, default=None, help="green-screen video")
    p.add_argument("--black-run", type=Path, default=None, help="run dir of the black-background video")
    p.add_argument("--black-video", type=Path, default=None, help="black-background video")
    p.add_argument("--green-params", type=Path, default=None,
                   help="params.json of the green run (default: <green-run>/params.json)")
    p.add_argument("--black-params", type=Path, default=None,
                   help="params.json of the black run (default: <black-run>/params.json)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "images", help="output folder")
    p.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "cylvision_readme_runs",
                   help="where the batch CSVs are cached (never inside docs/images)")
    p.add_argument("--frame-scale-green", type=float, default=60.0,
                   help="source frames per frame of the green video (pre-subsampled), default 60")
    p.add_argument("--t0-frame", type=int, default=GREEN_T0_FRAME,
                   help="green frame whose time is 0 in the captions (start of the pour), default %(default)s")
    p.add_argument("--skip-batch", action="store_true", help="reuse the cached batch CSVs instead of processing")
    p.add_argument("--reuse-batch", type=str, default="",
                   help="comma-separated runs (green, black) whose cached batch CSV is reused")
    p.add_argument("--panel-screenshot", action="store_true",
                   help="open the Tk control panel briefly and grab control_panel.png")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated figure groups or PNG stems to regenerate (default: all)")
    a = p.add_argument_group("absorbance group (back-lit run with an empty-cylinder reference)")
    a.add_argument("--absorbance-run", type=Path, default=None, help="run dir of the absorbance video")
    a.add_argument("--absorbance-video", type=Path, default=None, help="back-lit video (pre-subsampled or not)")
    a.add_argument("--absorbance-params", type=Path, default=None,
                   help="params.json of the liquid/foam gradient detector (default: <absorbance-run>/params.json)")
    a.add_argument("--absorbance-ref-frames", type=str, default=ABS_DEFAULTS["ref_frames"],
                   help="frames a:b of the empty cylinder (default %(default)s)")
    a.add_argument("--absorbance-t0-frame", type=float, default=ABS_DEFAULTS["t0_frame"],
                   help="SOURCE frame of the start of the pour, t = 0 (default %(default)s)")
    a.add_argument("--absorbance-frame-scale", type=float, default=ABS_DEFAULTS["frame_scale"],
                   help="source frames per video frame (default %(default)s)")
    a.add_argument("--absorbance-start", type=int, default=ABS_DEFAULTS["start"],
                   help="first analysed frame (default %(default)s)")
    a.add_argument("--absorbance-frame", type=int, default=ABS_DEFAULTS["measurement_frame"],
                   help="frame of the measurement figure (default %(default)s)")
    a.add_argument("--absorbance-k", type=float, default=ABS_DEFAULTS["k"],
                   help="foam-top threshold k (default %(default)s)")
    a.add_argument("--absorbance-r-dens", type=int, default=ABS_DEFAULTS["r_dens"],
                   help="profile band half-width in px (default %(default)s)")
    a.add_argument("--absorbance-px-per-cm", type=float, default=None,
                   help="vertical scale in px per cm (default: from the calibration slope and the section)")
    return p.parse_args(argv)


def selected_groups(only: str | None) -> list[str]:
    if not only:
        return list(FIGURES)
    wanted = {w.strip() for w in only.replace(";", ",").split(",") if w.strip()}
    groups = []
    for name, (_fn, files) in FIGURES.items():
        if name in wanted or any(f in wanted for f in files):
            groups.append(name)
    unknown = wanted - set(FIGURES) - {f for _fn, files in FIGURES.values() for f in files}
    if unknown:
        sys.exit(f"unknown figure name(s): {sorted(unknown)}")
    return groups


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)
    groups = selected_groups(args.only)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir)

    needs_runs = [x for x in groups if x != "absorbance"]
    if needs_runs and not all((args.green_run, args.green_video, args.black_run, args.black_video)):
        sys.exit("--green-run/--green-video and --black-run/--black-video are required for the groups "
                 f"{needs_runs} (only the absorbance group works without them)")
    g = b = None
    absorbance: dict[str, Any] | None = None
    if needs_runs:
        g = load_ctx("green", args.green_run, args.green_video, args.green_params, args.frame_scale_green,
                     t0_frame=args.t0_frame)
        b = load_ctx("black", args.black_run, args.black_video, args.black_params, 1.0)
    if "absorbance" in groups and args.absorbance_run and args.absorbance_video:
        ctx_a = load_ctx("absorbance", args.absorbance_run, args.absorbance_video, args.absorbance_params,
                         args.absorbance_frame_scale)
        absorbance = {"ctx": ctx_a, "opts": {
            **ABS_DEFAULTS, "ref_frames": args.absorbance_ref_frames, "t0_frame": args.absorbance_t0_frame,
            "start": args.absorbance_start, "measurement_frame": args.absorbance_frame, "k": args.absorbance_k,
            "r_dens": args.absorbance_r_dens, "px_per_cm": args.absorbance_px_per_cm}}
    try:
        batches: dict[str, list[dict]] = {}
        if g is not None and b is not None and any(x in groups for x in ("frame_step", "batch", "uncertainty")):
            batches = run_batches(g, b, args.work_dir, args.skip_batch,
                                  reuse=[x.strip() for x in args.reuse_batch.split(",") if x.strip()])
        for name in groups:
            fn, _files = FIGURES[name]
            print(f"[{name}]")
            fn(g, b, manifest, batches=batches, panel_screenshot=args.panel_screenshot, absorbance=absorbance)
    finally:
        for ctx in (g, b, None if absorbance is None else absorbance["ctx"]):
            if ctx is not None:
                ctx.video.close()
    manifest.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
