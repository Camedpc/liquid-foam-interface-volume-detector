"""Unicode-safe image IO, video access and screen-fitting helpers.

Why not plain ``cv2.imread`` / ``cv2.imwrite``: on Windows the OpenCV
file functions go through the ANSI code page and silently fail on paths
containing accents or other non-ASCII characters. ``imread_unicode`` and
``imwrite_unicode`` route the bytes through ``numpy.fromfile`` /
``ndarray.tofile`` and ``cv2.imdecode`` / ``cv2.imencode`` instead, which
works with any path Python can open.

Video: ``open_video`` returns ``(cap, n_frames, fps)`` and raises
``OSError`` instead of exiting; ``read_frame`` seeks to an absolute index;
``VideoSource`` wraps both as a context manager and iterates
``(idx, frame)`` pairs, reading sequentially when the step is small
(seeking on every frame is slow on long GOP H.264 files).

Display helpers (``fit_to_screen``, ``fit_to_window``, ``get_screen_size``)
are pure array functions except ``get_screen_size`` which asks Tk once and
falls back to a default on headless machines.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS: tuple[str, ...] = ("mp4", "avi", "mov", "mkv", "MP4", "AVI", "MOV", "MKV")

# Scan codes returned by ``cv2.waitKeyEx`` for the navigation keys. The
# ``*_WIN`` values are the Windows codes; the tuples add the X11 (Linux) and
# Cocoa (macOS) codes so that ``key in KEY_RIGHT`` works on every backend.
# Callers still keep the ``a`` / ``d`` / ``0`` letter fallbacks.
KEY_RIGHT_WIN = 2555904
KEY_LEFT_WIN = 2424832
KEY_UP_WIN = 2490368
KEY_DOWN_WIN = 2621440
KEY_HOME_WIN = 2359296

KEY_RIGHT: tuple[int, ...] = (KEY_RIGHT_WIN, 65363, 63235)
KEY_LEFT: tuple[int, ...] = (KEY_LEFT_WIN, 65361, 63234)
KEY_UP: tuple[int, ...] = (KEY_UP_WIN, 65362, 63232)
KEY_DOWN: tuple[int, ...] = (KEY_DOWN_WIN, 65364, 63233)
KEY_HOME: tuple[int, ...] = (KEY_HOME_WIN, 65360, 63273)

# Above this step, seeking is cheaper than grabbing the skipped frames.
_SEEK_STEP_THRESHOLD = 24


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def imread_unicode(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """``cv2.imread`` that accepts non-ASCII paths. ``None`` on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path | str, img: np.ndarray, params: list[int] | None = None) -> bool:
    """``cv2.imwrite`` that accepts non-ASCII paths. Creates parent dirs."""
    p = Path(path)
    ext = p.suffix.lower() or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(p))
    except OSError:
        return False
    return True


def is_video_path(path: Path | str) -> bool:
    """True when the extension is one of :data:`VIDEO_EXTENSIONS`."""
    return Path(path).suffix.lstrip(".") in VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def open_video(path: Path | str) -> tuple[cv2.VideoCapture, int, float]:
    """Open a video, return ``(cap, n_frames, fps)``.

    Raises ``OSError`` when the file cannot be opened or reports no frame.
    ``cv2.VideoCapture`` may fail on Windows paths with accents; keep the
    videos in ASCII folders if that happens (there is no in-memory decode
    path for videos).
    """
    p = Path(path)
    if not p.exists():
        raise OSError(f"video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {p} (unsupported codec or non-ASCII path?)")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if n_frames <= 0:
        cap.release()
        raise OSError(f"video opened but reports 0 frame: {p}")
    return cap, n_frames, fps


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Seek to frame ``idx`` and decode it. ``None`` when out of range."""
    if idx < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


class VideoSource:
    """Context-manager wrapper around ``cv2.VideoCapture``.

    Attributes: ``path``, ``n_frames``, ``fps``, ``width``, ``height``.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.cap, self.n_frames, self.fps = open_video(self.path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            first = self.frame(0)
            if first is not None:
                self.height, self.width = first.shape[:2]
        self._pos = -1  # index of the next frame a sequential read would return

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None  # type: ignore[assignment]

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_frames

    def __repr__(self) -> str:
        return (f"VideoSource({self.path.name!r}, {self.width}x{self.height}, "
                f"{self.n_frames} frames @ {self.fps:.2f} fps)")

    # -- access ------------------------------------------------------------
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else float("nan")

    def frame(self, idx: int) -> np.ndarray | None:
        """Random access to frame ``idx`` (seek + decode)."""
        if not (0 <= idx < self.n_frames):
            return None
        fr = read_frame(self.cap, idx)
        self._pos = idx + 1 if fr is not None else -1
        return fr

    def _next(self) -> np.ndarray | None:
        ok, fr = self.cap.read()
        if ok and fr is not None:
            self._pos += 1
            return fr
        self._pos = -1
        return None

    def frames(self, start: int = 0, stop: int | None = None,
               step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(idx, frame)`` for ``idx in range(start, stop, step)``.

        Small steps read sequentially (``grab`` skips the intermediate
        frames without decoding them fully); large steps seek. Frames that
        fail to decode are skipped.
        """
        if step <= 0:
            raise ValueError("step must be >= 1")
        stop = self.n_frames if stop is None else min(int(stop), self.n_frames)
        start = max(0, int(start))
        for idx in range(start, stop, step):
            if self._pos == idx:
                fr = self._next()
            elif 0 <= self._pos < idx and idx - self._pos <= _SEEK_STEP_THRESHOLD:
                while self._pos < idx and self.cap.grab():
                    self._pos += 1
                fr = self._next() if self._pos == idx else self.frame(idx)
            else:
                fr = self.frame(idx)
            if fr is not None:
                yield idx, fr


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fit_to_screen(frame: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    """Uniform down-scale to fit ``(max_w, max_h)``; never up-scales.

    Returns ``(frame, scale)``; ``scale == 1.0`` when untouched.
    """
    H, W = frame.shape[:2]
    s_w = max_w / W if W > 0 else 1.0
    s_h = max_h / H if H > 0 else 1.0
    s = min(1.0, s_w, s_h)
    if s >= 0.999:
        return frame, 1.0
    new_w = max(1, int(round(W * s)))
    new_h = max(1, int(round(H * s)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), s


def fit_to_window(canvas: np.ndarray, win: str) -> np.ndarray:
    """Letterbox ``canvas`` into the current size of OpenCV window ``win``.

    The Win32 backend ignores ``WINDOW_KEEPRATIO`` and stretches the image
    in full screen; building an image of exactly the window size makes
    ``imshow`` map it 1:1. Returns ``canvas`` untouched when the window
    size is not available yet.
    """
    try:
        _x, _y, ww, wh = cv2.getWindowImageRect(win)
    except cv2.error:
        return canvas
    if ww <= 0 or wh <= 0:
        return canvas
    H, W = canvas.shape[:2]
    if W == 0 or H == 0:
        return canvas
    scale = min(ww / W, wh / H)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(canvas, (new_w, new_h), interpolation=interp)
    out = np.zeros((wh, ww, 3), dtype=np.uint8)
    pad_top = (wh - new_h) // 2
    pad_left = (ww - new_w) // 2
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return out


def get_screen_size(default: tuple[int, int] = (1600, 900)) -> tuple[int, int]:
    """Screen resolution via Tk; ``default`` when unavailable (headless)."""
    if os.environ.get("CYLVISION_HEADLESS"):
        return default
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if sw > 0 and sh > 0:
            return int(sw), int(sh)
    except Exception:
        pass
    return default


__all__ = [
    "VIDEO_EXTENSIONS", "KEY_RIGHT_WIN", "KEY_LEFT_WIN", "KEY_UP_WIN", "KEY_DOWN_WIN",
    "KEY_HOME_WIN", "KEY_RIGHT", "KEY_LEFT", "KEY_UP", "KEY_DOWN", "KEY_HOME", "imread_unicode", "imwrite_unicode", "is_video_path", "open_video",
    "read_frame", "VideoSource", "fit_to_screen", "fit_to_window", "get_screen_size",
]
