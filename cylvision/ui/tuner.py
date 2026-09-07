"""Live tuner: video frame + analysis panels + Tk control panel -> DetectionParams.

Two layers:

* :func:`compose_canvas` is PURE (arrays in, array out) and is shared with
  the README figure generator: crop the frame with the calibration band,
  run :func:`detect_from_crop`, build the three analysis panels
  (highlight / signed gradient / threshold mask), optionally the specimen
  panel (full frame with braces) on the left, and the info strip below.
  Panels are scaled UNIFORMLY (up or down) to use the available screen
  area while the text strips keep their native pixel size, exactly like the
  original wide tuner. The result is padded to ``screen_size``.

  Layouts: ``inline`` puts every panel on one row
  ``[specimen] [highlight] [gradient] [mask]``; ``stacked`` uses two rows,
  ``[specimen] [highlight]`` over ``[gradient] [mask]`` (useful when the
  crop is short and wide).

* :func:`run_tuner` owns the windows: an OpenCV window for the canvas and a
  :class:`ControlsPanel` for the knobs, pumped in the same loop. Detection
  runs only when a parameter or the frame changes; the canvas is
  re-rendered only when the detection, the view or the window size changed.

Keys (OpenCV window): ``Enter`` accept + run batch, ``Esc`` abort, ``t``
toggle the specimen panel, ``l`` toggle inline / stacked, ``f``
fullscreen, ``Space`` play/pause, arrows / ``a`` / ``d`` previous / next
frame, ``Home`` / ``0`` first frame.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cylvision.calibration.models import build_model_fn
from cylvision.calibration.store import Calibration
from cylvision.detection.interfaces import DetectionParams, InterfaceResult, detect_from_crop
from cylvision.detection.panels import (
    LABEL_HEIGHT,
    SEP,
    annotate_interfaces,
    make_gradient_panel,
    make_info_strip,
    make_label_strip,
    make_specimen_panel,
    make_threshold_panel,
)
from cylvision.io import (
    KEY_HOME,
    KEY_LEFT,
    KEY_RIGHT,
    VideoSource,
    get_screen_size,
)
from cylvision.pipeline.batch import crop_frame
from cylvision.ui import theme
from cylvision.ui.controls_panel import ControlsPanel

WINDOW_TITLE = "liquid-foam-interface-volume-detector  -  tuner"
PANEL_LABELS_ANALYSIS: tuple[str, str, str] = ("Highlight", "Signed gradient", "Threshold mask")
SPECIMEN_LABEL = "Specimen"
MIN_SCALE = 0.05

TunerHook = Callable[["TunerState"], str | None]


# ---------------------------------------------------------------------------
# Pure composition
# ---------------------------------------------------------------------------

def crop_and_detect(frame_bgr: np.ndarray, calib: Calibration,
                    params: DetectionParams) -> tuple[np.ndarray, InterfaceResult, np.ndarray]:
    """``(crop, result, Gy)`` for a frame (crop = calibration band, full height)."""
    crop = crop_frame(frame_bgr, calib)
    result, Gy = detect_from_crop(crop, params)
    return crop, result, Gy


def _resize(img: np.ndarray, s: float) -> np.ndarray:
    h, w = img.shape[:2]
    tw, th = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if tw == w and th == h:
        return img
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (tw, th), interpolation=interp)


def _mask_lines(img: np.ndarray, params: DetectionParams, s: float) -> None:
    """Magenta ``y_top`` / ``y_bottom`` lines on a scaled panel."""
    H, W = img.shape[:2]
    for y in (params.y_top if params.y_top > 0 else None, params.y_bottom):
        if y is None:
            continue
        yy = int(round((y + 0.5) * s))
        if 0 <= yy < H:
            cv2.line(img, (0, yy), (W - 1, yy), theme.MASK_BGR, 1, cv2.LINE_AA)


_SHORT_LABELS: dict[str, str] = {
    "Signed gradient": "Gradient", "Threshold mask": "Mask", "Highlight": "Highl.", "Specimen": "Spec.",
}


def _fit_labels(labels: Sequence[str], widths: Sequence[int]) -> list[str]:
    """Shorten the labels that would overflow their (narrow) panel."""
    out: list[str] = []
    for lab, w in zip(labels, widths):
        text = lab
        for cand in (lab, _SHORT_LABELS.get(lab, lab)):
            (tw, _), _ = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            text = cand
            if tw + 8 <= w:
                break
        else:
            while len(text) > 2:
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                if tw + 8 <= w:
                    break
                text = text[:-1]
        out.append(text)
    return out


def _row_image(panels: Sequence[np.ndarray], labels: Sequence[str], total_w: int) -> np.ndarray:
    """Label strip + panels side by side, padded on the right to ``total_w``."""
    labels = _fit_labels(labels, [p.shape[1] for p in panels])
    h = max(p.shape[0] for p in panels)
    parts: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if p.shape[0] != h:
            pad = np.empty((h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            pad[:] = theme.BG_BGR
            p = np.concatenate([p, pad], axis=0)
        if i:
            sep = np.empty((h, SEP, 3), dtype=np.uint8)
            sep[:] = theme.SEPARATOR_BGR
            parts.append(sep)
        parts.append(p)
    body = np.concatenate(parts, axis=1)
    if body.shape[1] < total_w:
        pad = np.empty((h, total_w - body.shape[1], 3), dtype=np.uint8)
        pad[:] = theme.BG_BGR
        body = np.concatenate([body, pad], axis=1)
    strip = make_label_strip(body.shape[1], [p.shape[1] for p in panels], labels)
    return np.concatenate([strip, body], axis=0)


def _pad_to(canvas: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Centre ``canvas`` in a ``(W, H)`` background (crops if larger)."""
    ww, wh = int(size[0]), int(size[1])
    ch, cw = canvas.shape[:2]
    if ch == wh and cw == ww:
        return canvas
    out = np.empty((wh, ww, 3), dtype=np.uint8)
    out[:] = theme.BG_BGR
    pad_top = max(0, (wh - ch) // 2)
    pad_left = max(0, (ww - cw) // 2)
    h_copy = min(ch, wh - pad_top)
    w_copy = min(cw, ww - pad_left)
    if h_copy > 0 and w_copy > 0:
        out[pad_top:pad_top + h_copy, pad_left:pad_left + w_copy] = canvas[:h_copy, :w_copy]
    return out


def render_canvas(frame_bgr: np.ndarray, calib: Calibration, params: DetectionParams,
                  crop: np.ndarray, result: InterfaceResult, model_fn: Callable[..., Any] | None, *,
                  show_specimen: bool, inline: bool, screen_size: tuple[int, int],
                  frame_idx: int | None = None, t_sec: float | None = None,
                  fps: float | None = None) -> np.ndarray:
    """Render the tuner canvas from an existing detection (see :func:`compose_canvas`)."""
    ww, wh = max(8, int(screen_size[0])), max(8, int(screen_size[1]))
    H, W = crop.shape[:2]
    W_full = frame_bgr.shape[1]

    # Raw layout (native sizes) -> uniform scale.
    if inline:
        rows_w = [(W_full if show_specimen else 0) + 3 * W + SEP * (3 if show_specimen else 2)]
        rows_h = [H]
    else:
        rows_w = [(W_full if show_specimen else 0) + W + (SEP if show_specimen else 0), 2 * W + SEP]
        rows_h = [H, H]
    raw_w = max(rows_w)
    raw_h = sum(rows_h) + SEP * (len(rows_h) - 1)
    n_rows = len(rows_h)
    # The info strip wraps its text, so its height depends on the final
    # width, which depends on the scale, which depends on the height left
    # by the strip: iterate a few times until the height is stable.
    info = make_info_strip(ww, result, params, model_fn, frame_idx, t_sec, fps)
    s = MIN_SCALE
    for _ in range(4):
        avail_h = max(8, wh - LABEL_HEIGHT * n_rows - info.shape[0] - SEP * n_rows)
        s = max(MIN_SCALE, min(ww / raw_w, avail_h / raw_h))
        probe = make_info_strip(max(1, int(round(raw_w * s))), result, params, model_fn, frame_idx, t_sec, fps)
        if probe.shape[0] == info.shape[0]:
            break
        info = probe

    # Panels at the target scale. The overlays (highlight and specimen) are
    # drawn AFTER scaling so that lines, wash and labels keep their pixel size.
    p_high = annotate_interfaces(crop, result, params, zoom=s, show_extras=True, model_fn=model_fn)
    p_grad = _resize(make_gradient_panel(result.S), s)
    p_mask = _resize(make_threshold_panel(result.S, params), s)
    _mask_lines(p_grad, params, s)
    _mask_lines(p_mask, params, s)
    p_spec = (make_specimen_panel(frame_bgr, calib.crop(), result, zoom=s, model_fn=model_fn)
              if show_specimen else None)

    total_w = max(1, int(round(raw_w * s)))
    if inline:
        panels = ([p_spec] if p_spec is not None else []) + [p_high, p_grad, p_mask]
        labels = ([SPECIMEN_LABEL] if p_spec is not None else []) + list(PANEL_LABELS_ANALYSIS)
        rows = [_row_image(panels, labels, total_w)]
    else:
        top = ([p_spec] if p_spec is not None else []) + [p_high]
        top_labels = ([SPECIMEN_LABEL] if p_spec is not None else []) + [PANEL_LABELS_ANALYSIS[0]]
        rows = [_row_image(top, top_labels, total_w),
                _row_image([p_grad, p_mask], PANEL_LABELS_ANALYSIS[1:], total_w)]
    total_w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == total_w else _pad_to(r, (total_w, r.shape[0])) for r in rows]
    info = make_info_strip(total_w, result, params, model_fn, frame_idx, t_sec, fps)
    sep = np.empty((SEP, total_w, 3), dtype=np.uint8)
    sep[:] = theme.SEPARATOR_BGR
    parts: list[np.ndarray] = []
    for i, r in enumerate(rows):
        if i:
            parts.append(sep)
        parts.append(r)
    parts += [sep, info]
    canvas = np.concatenate(parts, axis=0)
    return _pad_to(canvas, (ww, wh))


def compose_canvas(frame_bgr: np.ndarray, calib: Calibration, params: DetectionParams,
                   model_fn: Callable[..., Any] | None, *, show_specimen: bool = True,
                   inline: bool = True, screen_size: tuple[int, int] = (1920, 1080),
                   frame_idx: int | None = None, t_sec: float | None = None,
                   fps: float | None = None) -> np.ndarray:
    """Crop -> detect -> panels (+ specimen, + info strip), fitted to ``screen_size``.

    Pure function; returns a BGR ``uint8`` image of exactly ``screen_size``
    ``(W, H)``. ``model_fn`` (``y_px -> V_mL``) adds the volumes to the info
    strip; ``frame_idx`` / ``t_sec`` / ``fps`` add the time stamp.
    """
    crop, result, _Gy = crop_and_detect(frame_bgr, calib, params)
    return render_canvas(frame_bgr, calib, params, crop, result, model_fn,
                         show_specimen=show_specimen, inline=inline, screen_size=screen_size,
                         frame_idx=frame_idx, t_sec=t_sec, fps=fps)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

@dataclass
class TunerState:
    """What the loop knows; handed to the optional ``hook`` every iteration."""

    panel: ControlsPanel
    window: str
    iteration: int
    frame_idx: int
    params: DetectionParams
    result: InterfaceResult | None
    canvas: np.ndarray | None


def _initial_params(calib: Calibration, params: DetectionParams) -> DetectionParams:
    """Fill ``y_bottom`` / ``cx`` from the calibration when the params leave them unset."""
    p = DetectionParams.from_dict(params.to_dict())
    if p.y_bottom is None:
        p.y_bottom = int(calib.y_bottom)
    if p.cx is None:
        p.cx = int(calib.cx())
    return p


def run_tuner(video: VideoSource, calib: Calibration, params: DetectionParams, *,
              model_fn: Callable[..., Any] | None = None, start_idx: int = 0,
              two_columns: bool = False, frame_step: int = 1, time_scale: float = 1.0,
              hook: TunerHook | None = None) -> tuple[DetectionParams, int, bool]:
    """Open the tuner on ``video`` and return ``(params, frame_step, batch_requested)``.

    ``batch_requested`` is True after ``Enter`` or the ``Run batch`` button,
    False after ``Esc`` / ``Abort`` / a closed window (the returned params
    are still the last ones shown). ``time_scale`` multiplies the displayed
    time (pre-subsampled videos). ``hook(state)`` is an optional callback
    run every iteration; returning ``"batch"`` or ``"abort"`` ends the loop
    (used by scripted screenshots and tests).
    """
    if model_fn is None:
        model_fn = build_model_fn(calib)
    n_frames = int(video.n_frames)
    start_idx = max(0, min(int(start_idx), n_frames - 1))
    frame = video.frame(start_idx)
    if frame is None:
        raise OSError(f"cannot read frame {start_idx}")
    H = frame.shape[0]
    params = _initial_params(calib, params)

    sw, sh = get_screen_size()
    win = WINDOW_TITLE
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, sw, sh)
    cv2.moveWindow(win, 0, 0)
    panel = ControlsPanel(params=params, H=H, W_crop=calib.crop_width(), n_frames=n_frames,
                          init_frame_idx=start_idx, include_sampling=True, show_batch_button=True,
                          two_columns=two_columns, frame_step=frame_step)

    frame_idx = start_idx
    fps = float(video.fps) if video.fps else None
    last_params: DetectionParams | None = None
    last_view: tuple[bool, bool, int, int, int] | None = None
    crop = result = None
    canvas: np.ndarray | None = None
    playing = False
    fullscreen = False
    outcome: bool | None = None
    iteration = 0

    def t_of(idx: int) -> float | None:
        return idx * float(time_scale) / fps if fps else None

    try:
        while outcome is None:
            panel.update()
            if not panel.is_alive() or panel.is_abort_requested():
                outcome = False
                break
            if panel.is_batch_requested():
                outcome = True
                break
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    outcome = False
                    break
            except cv2.error:
                outcome = False
                break

            # Frame from the panel slider (keys update the slider too).
            want = panel.frame_idx()
            if playing:
                want = min(frame_idx + 1, n_frames - 1)
                if want == frame_idx:
                    playing = False
            if want != frame_idx:
                fr = video.frame(want)
                if fr is not None:
                    frame, frame_idx = fr, want
                    panel.set_frame_idx(frame_idx)
                    last_params = None  # force re-detection
                else:
                    panel.set_frame_idx(frame_idx)

            cur = panel.get_params()
            if last_params is None or cur != last_params:
                crop, result, _Gy = crop_and_detect(frame, calib, cur)
                last_params = cur
                last_view = None

            try:
                _x, _y, ww, wh = cv2.getWindowImageRect(win)
            except cv2.error:
                ww, wh = sw, sh
            if ww <= 0 or wh <= 0:
                ww, wh = sw, sh
            vm = panel.view_mode()
            view_key = (vm["show_specimen"], vm["inline"], ww, wh, frame_idx)
            if view_key != last_view and result is not None and crop is not None:
                canvas = render_canvas(frame, calib, cur, crop, result, model_fn,
                                       show_specimen=vm["show_specimen"], inline=vm["inline"],
                                       screen_size=(ww, wh), frame_idx=frame_idx,
                                       t_sec=t_of(frame_idx), fps=fps)
                last_view = view_key
            if canvas is not None:
                cv2.imshow(win, canvas)

            if hook is not None:
                verdict = hook(TunerState(panel, win, iteration, frame_idx, cur, result, canvas))
                if verdict == "batch":
                    outcome = True
                    break
                if verdict == "abort":
                    outcome = False
                    break
            iteration += 1

            wait_ms = max(1, int(1000.0 / max(fps or 0.0, 5.0))) if playing else 30
            key = cv2.waitKeyEx(wait_ms)
            if key == -1:
                continue
            if key in (13, 10):
                outcome = True
            elif key == 27:
                outcome = False
            elif key == 32:
                playing = not playing
            elif key in KEY_RIGHT or key in (ord("d"), ord("D")):
                playing = False
                panel.set_frame_idx(frame_idx + 1)
            elif key in KEY_LEFT or key in (ord("a"), ord("A")):
                playing = False
                panel.set_frame_idx(frame_idx - 1)
            elif key in KEY_HOME or key == ord("0"):
                playing = False
                panel.set_frame_idx(0)
            elif key in (ord("t"), ord("T")):
                panel.set_view_mode(show_specimen=not vm["show_specimen"])
            elif key in (ord("l"), ord("L")):
                panel.set_view_mode(inline=not vm["inline"])
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if not fullscreen:
                    cv2.resizeWindow(win, sw, sh)
                    cv2.moveWindow(win, 0, 0)
    finally:
        final_params = panel.get_params() if panel.is_alive() else (last_params or params)
        final_step = panel.frame_step() if panel.is_alive() else max(1, int(frame_step))
        panel.destroy()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.waitKey(1)
    return final_params, final_step, bool(outcome)


__all__ = [
    "WINDOW_TITLE", "PANEL_LABELS_ANALYSIS", "SPECIMEN_LABEL", "TunerState",
    "crop_and_detect", "render_canvas", "compose_canvas", "run_tuner",
]
