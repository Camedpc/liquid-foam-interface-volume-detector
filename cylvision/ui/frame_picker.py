"""Interactive frame picker: choose one frame of a video in an OpenCV window.

Used to pick the calibration frame, the first and the last frame of a
batch. The first frame of a recording is rarely representative (lighting
not settled, cylinder not in place yet), hence a scrubber.

Controls: trackbar ``frame`` | ``Space`` play/pause | ``d`` / right arrow
next | ``a`` / left arrow previous | ``0`` / ``Home`` first frame |
``Enter`` / ``v`` validate | ``Esc`` / ``q`` abort (returns ``None``).

The frame is down-scaled to fit the window (never up-scaled) and a status
strip at native text size shows the index, the time and the bounds. Arrow
keys use the ``cv2.waitKeyEx`` codes of Windows, X11 and macOS
(``cylvision.io.KEY_*``); the letter fallbacks work everywhere.
"""
from __future__ import annotations

import cv2
import numpy as np

from cylvision.io import (
    KEY_HOME,
    KEY_LEFT,
    KEY_RIGHT,
    fit_to_screen,
    get_screen_size,
    read_frame,
)

STRIP_H = 36
_KEY_ENTER = (13, 10, ord("v"), ord("V"))
_KEY_ABORT = (27, ord("q"), ord("Q"))


def _read_with_fallback(cap: cv2.VideoCapture, idx: int, min_idx: int) -> tuple[int, np.ndarray | None]:
    """Seek to ``idx``, stepping back to ``min_idx`` when the tail is not decodable.

    ``CAP_PROP_FRAME_COUNT`` is optimistic on some phone recordings: the
    last few indices exist in the header but not in the stream.
    """
    fr = read_frame(cap, idx)
    while fr is None and idx > min_idx:
        idx -= 1
        fr = read_frame(cap, idx)
    return idx, fr


def render_picker_view(frame: np.ndarray, ww: int, wh: int, *, frame_idx: int, n_frames: int,
                       t_sec: float, playing: bool, bounds: tuple[int, int]) -> np.ndarray:
    """Frame fitted in ``(ww, wh)`` + status strip; padded to exactly the window size."""
    avail_h = max(64, wh - STRIP_H)
    disp, _ = fit_to_screen(frame, max(8, ww), avail_h)
    dh, dw = disp.shape[:2]
    bar = np.full((STRIP_H, dw, 3), 25, dtype=np.uint8)
    status = "PLAY " if playing else "PAUSE"
    color = (50, 220, 50) if playing else (60, 60, 220)
    msg = (f"{status}  frame {frame_idx:>6d} / {n_frames - 1}   t = {t_sec:8.2f} s   "
           f"range [{bounds[0]}, {bounds[1]}]   Enter = validate   Esc = abort")
    cv2.putText(bar, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    composite = np.concatenate([disp, bar], axis=0)
    ch, cw = composite.shape[:2]
    if ch == wh and cw == ww:
        return composite
    out = np.zeros((max(wh, 1), max(ww, 1), 3), dtype=np.uint8)
    pad_top = max(0, (wh - ch) // 2)
    pad_left = max(0, (ww - cw) // 2)
    h_copy = min(ch, wh - pad_top)
    w_copy = min(cw, ww - pad_left)
    if h_copy > 0 and w_copy > 0:
        out[pad_top:pad_top + h_copy, pad_left:pad_left + w_copy] = composite[:h_copy, :w_copy]
    return out


def pick_frame(cap: cv2.VideoCapture, n_frames: int, fps: float, *, title: str,
               init_idx: int = 0, min_idx: int = 0, max_idx: int | None = None,
               time_scale: float = 1.0) -> int | None:
    """Open the picker window and return the chosen frame index (``None`` = aborted).

    ``min_idx`` / ``max_idx`` bound the trackbar (e.g. force the end frame
    after the start frame). ``time_scale`` multiplies the displayed time
    (``K`` for a video pre-subsampled by ``K``).
    """
    n_frames = int(n_frames)
    if max_idx is None:
        max_idx = n_frames - 1
    max_idx = max(min_idx, min(int(max_idx), n_frames - 1))
    min_idx = max(0, min(int(min_idx), max_idx))
    init_idx = max(min_idx, min(int(init_idx), max_idx))

    sw, sh = get_screen_size()
    init_w = max(640, int(sw * 0.8))
    init_h = max(480, int(sh * 0.8))
    win = title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, init_w, init_h)
    cv2.moveWindow(win, max(0, (sw - init_w) // 2), max(0, (sh - init_h) // 2))
    span = max(1, max_idx - min_idx)
    cv2.createTrackbar("frame", win, init_idx - min_idx, span, lambda _v: None)

    frame_idx, current = _read_with_fallback(cap, init_idx, min_idx)
    if current is None:
        cv2.destroyWindow(win)
        return None
    if frame_idx != init_idx:
        max_idx = min(max_idx, frame_idx)
        cv2.setTrackbarPos("frame", win, frame_idx - min_idx)
    last_slider = frame_idx - min_idx
    playing = False

    def t_of(idx: int) -> float:
        return idx * float(time_scale) / fps if fps and fps > 0 else 0.0

    def goto(idx: int) -> None:
        nonlocal frame_idx, current, last_slider, playing
        playing = False
        idx = max(min_idx, min(idx, max_idx))
        if idx == frame_idx:
            return
        fr = read_frame(cap, idx)
        if fr is not None:
            current, frame_idx = fr, idx
            cv2.setTrackbarPos("frame", win, frame_idx - min_idx)
            last_slider = frame_idx - min_idx

    while True:
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except cv2.error:
            return None

        slider = cv2.getTrackbarPos("frame", win)
        if slider != last_slider and min_idx + slider != frame_idx:
            goto(min_idx + slider)
            last_slider = slider

        if playing:
            ok, fr = cap.read()
            if ok and fr is not None:
                nxt = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if nxt > max_idx:
                    playing = False
                    read_frame(cap, frame_idx)
                else:
                    current, frame_idx = fr, nxt
                    cv2.setTrackbarPos("frame", win, frame_idx - min_idx)
                    last_slider = frame_idx - min_idx
            else:
                playing = False

        try:
            _x, _y, ww, wh = cv2.getWindowImageRect(win)
        except cv2.error:
            ww, wh = init_w, init_h
        if ww <= 0 or wh <= 0:
            ww, wh = init_w, init_h
        view = render_picker_view(current, ww, wh, frame_idx=frame_idx, n_frames=n_frames,
                                  t_sec=t_of(frame_idx), playing=playing, bounds=(min_idx, max_idx))
        cv2.imshow(win, view)

        wait_ms = max(1, int(1000.0 / max(fps or 0.0, 5.0))) if playing else 30
        key = cv2.waitKeyEx(wait_ms)
        if key == -1:
            continue
        if key in _KEY_ABORT:
            cv2.destroyWindow(win)
            return None
        if key in _KEY_ENTER:
            cv2.destroyWindow(win)
            return int(frame_idx)
        if key == 32:
            playing = not playing
        elif key in KEY_RIGHT or key in (ord("d"), ord("D")):
            goto(frame_idx + 1)
        elif key in KEY_LEFT or key in (ord("a"), ord("A")):
            goto(frame_idx - 1)
        elif key in KEY_HOME or key == ord("0"):
            goto(min_idx)


__all__ = ["STRIP_H", "pick_frame", "render_picker_view"]
