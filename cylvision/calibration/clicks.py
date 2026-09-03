"""Interactive collection of the calibration clicks (OpenCV window).

The user places, in this order, on one still frame of the cylinder:

1. the LEFT edge of the cylinder (vertical green line),
2. the RIGHT edge (vertical green line) -- together they define the crop
   band ``frame[:, x_left:x_right+1]``,
3. the TOP row ``y_top`` just under the glass rim (cyan),
4. the BOTTOM row ``y_bottom`` at the foot of the cylinder (cyan) -- rows
   outside ``[y_top, y_bottom]`` are ignored by the gradient detector,
5. every graduation of the cylinder preset (orange), each one skippable
   with ``s`` when it lies outside the frame.

Pixel-accurate pointing is the whole point of this tool, so the cursor is
a *logical* cursor decoupled from the OS mouse:

* mouse (normal mode) drives the cursor 1:1;
* holding **Shift** turns the mouse into a damped pointer: the cursor moves
  by one logical pixel every ``SHIFT_DAMPEN`` mouse pixels;
* **arrow keys** nudge the cursor by exactly one pixel; for
  ``ARROW_LOCK_MS`` after a nudge the mouse is ignored so that a hand
  tremor does not immediately undo the nudge;
* **Space** or a **left click** places the point at the logical cursor
  (not at the raw mouse position);
* ``f`` toggles a 1:1 fullscreen window (natural pixel mapping when the
  screen and the video share a resolution), ``s`` skips a graduation,
  ``z`` undoes the last point, **Enter** validates once every prompt is
  answered and at least three graduations were placed, **Esc** aborts.

A zoom loupe (``cylvision.magnifier``) follows the cursor.
"""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from cylvision.io import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP

# Prompt kinds -------------------------------------------------------------
PROMPT_KIND_VLINE = "vline_edge"   # left / right edge, green, mandatory
PROMPT_KIND_TOP = "hline_top"      # y_top, cyan, mandatory
PROMPT_KIND_BOT = "hline_bot"      # y_bottom, cyan, mandatory
PROMPT_KIND_GRAD = "hline_grad"    # graduation, orange, skippable

Prompt = tuple[str, str, float | None]      # (kind, label, V_ml or None)
Point = tuple[int, int] | None              # (x, y) or None when skipped

DEFAULT_WINDOW_TITLE = (
    "Calibration  |  mouse = move  |  SPACE/click = place  "
    "|  arrows = nudge 1 px  |  SHIFT = damped mouse  "
    "|  f = fullscreen 1:1  |  s = skip  |  z = undo  "
    "|  Enter = validate  |  Esc = abort"
)
WINDOW_DEFAULT_SIZE = (1600, 900)

# After an arrow key press the mouse is ignored for this long, otherwise
# the slightest hand tremor overwrites the keyboard nudge.
ARROW_LOCK_MS = 400

# Mouse damping factor while Shift is held: 1 logical pixel per
# SHIFT_DAMPEN mouse pixels. Higher = finer but slower.
SHIFT_DAMPEN = 4

MIN_GRADUATIONS = 3

# BGR colours
COLOR_VLINE = (60, 220, 60)      # green: vertical edges
COLOR_TBLINE = (220, 220, 60)    # cyan: top / bottom rows
COLOR_GRAD = (0, 165, 255)       # orange: graduations
COLOR_SHIFT = (200, 0, 200)      # magenta: Shift indicator
COLOR_PROMPT = (0, 255, 255)     # yellow: prompt banner
COLOR_SKIPPED = (200, 200, 200)
COLOR_BANNER_BG = (25, 25, 25)


def build_click_prompts(graduations_ml: list[float]) -> list[Prompt]:
    """Four ROI prompts followed by one prompt per graduation."""
    return [
        (PROMPT_KIND_VLINE, "LEFT edge of the cylinder", None),
        (PROMPT_KIND_VLINE, "RIGHT edge of the cylinder", None),
        (PROMPT_KIND_TOP, "TOP row (y_top, just under the glass rim)", None),
        (PROMPT_KIND_BOT, "BOTTOM row (y_bottom, foot of the cylinder)", None),
    ] + [(PROMPT_KIND_GRAD, f"Graduation {v:g} mL", float(v)) for v in graduations_ml]


def count_placed_graduations(prompts: list[Prompt], pts: list[Point]) -> int:
    """Number of graduation prompts answered with a real point."""
    return sum(1 for p, pt in zip(prompts, pts) if p[0] == PROMPT_KIND_GRAD and pt is not None)


def clicks_from_points(prompts: list[Prompt], pts: list[Point]) -> dict[str, Any]:
    """Turn the answered prompts into the clicks dict (pure, testable).

    Returns ``{"x_left", "x_right", "y_top", "y_bottom", "V_ml", "y_px",
    "x_px", "skipped_ml"}``. Edges and rows are re-ordered if the user
    clicked them in the wrong order.
    """
    if len(pts) != len(prompts):
        raise ValueError("every prompt must be answered before parsing")
    left_pt, right_pt, top_pt, bot_pt = pts[:4]
    if any(p is None for p in (left_pt, right_pt, top_pt, bot_pt)):
        raise ValueError("edges and top/bottom rows cannot be skipped")
    assert left_pt and right_pt and top_pt and bot_pt  # for type checkers
    x_left, x_right = sorted((int(left_pt[0]), int(right_pt[0])))
    y_top, y_bottom = sorted((int(top_pt[1]), int(bot_pt[1])))

    kept = [(p[2], pt) for p, pt in zip(prompts[4:], pts[4:]) if pt is not None]
    skipped = [float(p[2]) for p, pt in zip(prompts[4:], pts[4:])
               if pt is None and p[2] is not None]
    V_ml = np.array([float(V) for V, _ in kept], dtype=float)
    y_px = np.array([float(pt[1]) for _, pt in kept], dtype=float)
    x_px = [int(pt[0]) for _, pt in kept]
    return {
        "x_left": x_left,
        "x_right": x_right,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "V_ml": V_ml,
        "y_px": y_px,
        "x_px": x_px,
        "skipped_ml": skipped,
    }


def render_click_overlay(img: np.ndarray, prompts: list[Prompt], pts: list[Point],
                         cursor: tuple[int, int], *, shift_held: bool = False,
                         with_magnifier: bool = True) -> np.ndarray:
    """Draw the placed points, the prompt banner, the crosshair and the loupe.

    Pure rendering: returns a new BGR array, never shows a window.
    """
    H, W = img.shape[:2]
    display = img.copy()

    for i, pt in enumerate(pts):
        if pt is None:
            continue
        px, py = pt
        kind, label, _ = prompts[i]
        if kind == PROMPT_KIND_VLINE:
            cv2.line(display, (px, 0), (px, H), COLOR_VLINE, 1)
            cv2.putText(display, label, (px + 6, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, COLOR_VLINE, 1, cv2.LINE_AA)
        elif kind in (PROMPT_KIND_TOP, PROMPT_KIND_BOT):
            cv2.line(display, (0, py), (W, py), COLOR_TBLINE, 2)
            cv2.putText(display, label, (12, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, COLOR_TBLINE, 1, cv2.LINE_AA)
        else:
            cv2.line(display, (0, py), (W, py), COLOR_GRAD, 1)
            cv2.circle(display, (px, py), 4, COLOR_GRAD, -1)
            cv2.putText(display, label, (px + 10, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, COLOR_GRAD, 1, cv2.LINE_AA)

    skipped = [prompts[i][1] for i, p in enumerate(pts) if p is None]
    if skipped:
        cv2.putText(display, "Skipped: " + ", ".join(skipped), (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_SKIPPED, 1, cv2.LINE_AA)

    n_grad = count_placed_graduations(prompts, pts)
    if len(pts) < len(prompts):
        _kind, label_next, V_next = prompts[len(pts)]
        skip_hint = "  |  s = skip" if V_next is not None else ""
        text = f"[{len(pts) + 1}/{len(prompts)}] {label_next}  |  SPACE/click = place{skip_hint}"
    elif n_grad < MIN_GRADUATIONS:
        text = (f"Only {n_grad} graduation(s) placed, {MIN_GRADUATIONS} needed: "
                f"z to undo a skip, Esc to abort.")
    else:
        text = "All points placed. Enter to validate, z to undo."
    if shift_held:
        text = f"[SHIFT: damped mouse 1/{SHIFT_DAMPEN}]  " + text
        text_color = COLOR_SHIFT
    else:
        text_color = COLOR_PROMPT
    cv2.rectangle(display, (0, H - 44), (W, H), COLOR_BANNER_BG, -1)
    cv2.putText(display, text, (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                text_color, 2, cv2.LINE_AA)

    if with_magnifier:
        # Imported lazily: only the interactive tools need the loupe.
        from cylvision.magnifier import draw_crosshair, make_magnifier, overlay_magnifier

        draw_crosshair(display, cursor[0], cursor[1])
        magnifier = make_magnifier(img, cursor[0], cursor[1])
        overlay_magnifier(display, magnifier, (cursor[0], cursor[1]))
    return display


class _CursorState:
    """Logical cursor driven by mouse / Shift-damped mouse / arrow keys."""

    def __init__(self, W: int, H: int) -> None:
        self.W, self.H = W, H
        self.xy: list[int] = [W // 2, H // 2]
        self.shift_held = False
        self.arrow_lock_until = 0.0
        self.last_mouse_xy: tuple[int, int] | None = None
        self.shift_accum = [0.0, 0.0]

    def _clamp(self) -> None:
        self.xy[0] = max(0, min(self.W - 1, self.xy[0]))
        self.xy[1] = max(0, min(self.H - 1, self.xy[1]))

    def on_move(self, x: int, y: int, shift: bool) -> None:
        if shift and not self.shift_held:      # Shift OFF -> ON: reset anchor
            self.shift_accum = [0.0, 0.0]
            self.last_mouse_xy = (x, y)
        self.shift_held = shift
        if self.last_mouse_xy == (x, y) and not shift:
            return                              # spurious MOUSEMOVE
        if shift:
            if self.last_mouse_xy is None:
                self.last_mouse_xy = (x, y)
                return
            lx, ly = self.last_mouse_xy
            self.shift_accum[0] += (x - lx) / SHIFT_DAMPEN
            self.shift_accum[1] += (y - ly) / SHIFT_DAMPEN
            step_x, step_y = int(self.shift_accum[0]), int(self.shift_accum[1])
            self.shift_accum[0] -= step_x
            self.shift_accum[1] -= step_y
            self.xy[0] += step_x
            self.xy[1] += step_y
            self._clamp()
            self.last_mouse_xy = (x, y)
        else:
            self.last_mouse_xy = (x, y)
            if time.monotonic() < self.arrow_lock_until:
                return                          # keep the keyboard nudge
            self.xy[0], self.xy[1] = x, y

    def nudge(self, dx: int, dy: int) -> None:
        self.xy[0] += dx
        self.xy[1] += dy
        self._clamp()
        self.arrow_lock_until = time.monotonic() + ARROW_LOCK_MS / 1000.0

    def point(self) -> tuple[int, int]:
        return int(self.xy[0]), int(self.xy[1])


def _arrow_delta(key: int) -> tuple[int, int] | None:
    if key in KEY_RIGHT:
        return 1, 0
    if key in KEY_LEFT:
        return -1, 0
    if key in KEY_UP:
        return 0, -1
    if key in KEY_DOWN:
        return 0, 1
    return None


def collect_calibration_clicks(img: np.ndarray, graduations_ml: list[float], *,
                               window_title: str = DEFAULT_WINDOW_TITLE) -> dict[str, Any] | None:
    """Run the interactive click loop on ``img`` (BGR).

    Returns the clicks dict (see :func:`clicks_from_points`) or ``None``
    when the user aborted with Esc or closed the window.
    """
    prompts = build_click_prompts(graduations_ml)
    H, W = img.shape[:2]
    cur = _CursorState(W, H)
    pts: list[Point] = []
    fullscreen = False

    def on_mouse(event: int, x: int, y: int, flags: int, _param: Any) -> None:
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        if event == cv2.EVENT_LBUTTONDOWN:
            cur.shift_held = shift
            if len(pts) < len(prompts):
                pts.append(cur.point())     # logical cursor, not the OS mouse
            return
        if event == cv2.EVENT_MOUSEMOVE:
            cur.on_move(x, y, shift)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_title, *WINDOW_DEFAULT_SIZE)
    cv2.setMouseCallback(window_title, on_mouse)
    # First paint before the loop so that WND_PROP_VISIBLE is meaningful.
    cv2.imshow(window_title, img)
    cv2.waitKey(50)

    aborted = False
    iteration = 0
    try:
        while True:
            iteration += 1
            # The compositor may report the window as not visible during the
            # first ticks: only trust the flag after a few iterations.
            try:
                visible = cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE)
                if visible < 1 and iteration > 3:
                    aborted = True
                    break
            except cv2.error:
                if iteration > 3:
                    aborted = True
                    break

            display = render_click_overlay(img, prompts, pts, cur.point(),
                                           shift_held=cur.shift_held)
            cv2.imshow(window_title, display)
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            key_low = key & 0xFF
            if key == 27 or key_low == 27:                       # Esc
                aborted = True
                break
            if key_low == 32 and len(pts) < len(prompts):        # Space
                pts.append(cur.point())
                continue
            if key_low in (ord("z"), ord("Z")) and pts:          # undo
                pts.pop()
                continue
            if key_low in (ord("s"), ord("S")) and len(pts) < len(prompts):
                if prompts[len(pts)][2] is not None:             # graduations only
                    pts.append(None)
                continue
            if key in (10, 13) and len(pts) == len(prompts):     # Enter
                if count_placed_graduations(prompts, pts) >= MIN_GRADUATIONS:
                    break
                continue
            if key_low in (ord("f"), ord("F")):                  # fullscreen 1:1
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_title, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if not fullscreen:
                    cv2.resizeWindow(window_title, *WINDOW_DEFAULT_SIZE)
                continue
            if cur.shift_held:          # arrows are disabled in damped mode
                continue
            delta = _arrow_delta(key)
            if delta is not None:
                cur.nudge(*delta)
    finally:
        if _window_exists(window_title):
            cv2.destroyWindow(window_title)

    if aborted:
        return None
    return clicks_from_points(prompts, pts)


def _window_exists(title: str) -> bool:
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 0
    except cv2.error:
        return False
