"""Optional clicks on the *back face* graduations of a transparent cylinder.

Seen from the front, a printed graduation appears twice: on the near wall
(the "front" click of the main calibration) and, through the glass, on the
far wall. Both marks sit on the same horizontal circle of the cylinder; the
perspective offset between them encodes the camera geometry with no need
for the focal length, the radius or the distance:

    rho = (slope_front - slope_back) / (slope_front + slope_back)
    cy  = intersection of the lines y_front(V) and y_back(V)

where ``slope_*`` are the px/mL slopes of the two families of clicks and
``cy`` is the row of the camera horizon (where a graduation circle projects
to a straight line). The curvature model then reads

    u(theta) - cx = a_px * sin(theta) * sqrt(1 - rho^2) / (1 - rho cos theta)
    v(theta, V) - cy = (y_front(V) - cy) * (1 - rho) / (1 - rho cos theta)

(see ``cylvision.uncertainty.curvature``). This module only collects the
clicks: for every front graduation the user clicks the matching back mark
or skips it with ``s`` when the far wall is not visible.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cylvision.calibration.store import Calibration

BackPrompt = tuple[float, str, int, int]      # (V_ml, label, x_hint, y_front)
Point = tuple[int, int] | None

DEFAULT_WINDOW_TITLE = (
    "Back-face clicks  |  click = place  |  s = skip  |  z = undo  |  "
    "Esc = abort  |  Enter = validate"
)
WINDOW_DEFAULT_SIZE = (1600, 900)

COLOR_FRONT = (255, 130, 60)      # light blue: existing front clicks
COLOR_CURRENT = (0, 255, 100)     # green: front line of the current prompt
COLOR_BACK = (0, 200, 255)        # orange: placed back clicks
COLOR_SKIP = (120, 120, 120)
COLOR_PROMPT = (0, 255, 255)
COLOR_BANNER_BG = (25, 25, 25)


def build_back_prompts(calib: Calibration) -> list[BackPrompt]:
    """One prompt per clicked front graduation ``(V, label, x_hint, y_front)``."""
    x_front = calib.graduations_px_x or [0.0] * len(calib.graduations_ml)
    return [
        (float(V), f"{V:g} mL", int(round(x)), int(round(y)))
        for V, x, y in zip(calib.graduations_ml, x_front, calib.graduations_px_y)
    ]


def render_back_overlay(img: np.ndarray, prompts: list[BackPrompt], pts: list[Point],
                        cursor: tuple[int, int], *, with_magnifier: bool = True) -> np.ndarray:
    """Draw front lines, placed back clicks, the current prompt and the loupe."""
    H, W = img.shape[:2]
    display = img.copy()

    for _V, label, _x, y_front in prompts:
        cv2.line(display, (0, y_front), (W, y_front), COLOR_FRONT, 1, cv2.LINE_AA)
        cv2.putText(display, f"front {label}", (W - 130, y_front - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_FRONT, 1, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        _V, label, _x, y_front = prompts[i]
        if pt is None:
            cv2.putText(display, f"SKIP {label}", (8, y_front + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SKIP, 1, cv2.LINE_AA)
            continue
        px, py = pt
        cv2.line(display, (0, py), (W, py), COLOR_BACK, 1, cv2.LINE_AA)
        cv2.circle(display, (px, py), 5, COLOR_BACK, -1)
        cv2.putText(display, f"back {label} -> y={py}", (px + 10, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BACK, 1, cv2.LINE_AA)

    if len(pts) < len(prompts):
        _V, label, _x, y_front = prompts[len(pts)]
        cv2.line(display, (0, y_front), (W, y_front), COLOR_CURRENT, 2, cv2.LINE_AA)
        cv2.putText(display, f"-> front {label}", (W - 230, y_front - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CURRENT, 2, cv2.LINE_AA)
        text = (f"[{len(pts) + 1}/{len(prompts)}] Click the BACK mark (through the glass) "
                f"of {label}  |  front at y={y_front}  |  s = skip")
    else:
        text = "All marks answered. Enter to validate, z to undo."
    cv2.rectangle(display, (0, H - 44), (W, H), COLOR_BANNER_BG, -1)
    cv2.putText(display, text, (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                COLOR_PROMPT, 2, cv2.LINE_AA)

    if with_magnifier:
        from cylvision.magnifier import draw_crosshair, make_magnifier, overlay_magnifier

        draw_crosshair(display, cursor[0], cursor[1])
        magnifier = make_magnifier(img, cursor[0], cursor[1])
        overlay_magnifier(display, magnifier, (cursor[0], cursor[1]))
    return display


def collect_back_clicks(img: np.ndarray, prompts: list[BackPrompt], *,
                        window_title: str = DEFAULT_WINDOW_TITLE) -> list[Point] | None:
    """Interactive loop: one back click (or ``None`` = skipped) per prompt.

    Returns ``None`` when the user aborts (Esc or window closed).
    """
    pts: list[Point] = []
    H, W = img.shape[:2]
    cursor = [W // 2, H // 2]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(prompts):
            pts.append((x, y))

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_title, *WINDOW_DEFAULT_SIZE)
    cv2.setMouseCallback(window_title, on_mouse)
    cv2.imshow(window_title, img)
    cv2.waitKey(50)

    aborted = False
    iteration = 0
    try:
        while True:
            iteration += 1
            try:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1 and iteration > 3:
                    aborted = True
                    break
            except cv2.error:
                if iteration > 3:
                    aborted = True
                    break
            display = render_back_overlay(img, prompts, pts, (cursor[0], cursor[1]))
            cv2.imshow(window_title, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                aborted = True
                break
            if key in (ord("z"), ord("Z")) and pts:
                pts.pop()
            elif key in (ord("s"), ord("S")) and len(pts) < len(prompts):
                pts.append(None)
            elif key in (10, 13) and len(pts) == len(prompts):
                break
    finally:
        try:
            cv2.destroyWindow(window_title)
        except cv2.error:
            pass
    return None if aborted else pts


def back_points_to_lists(pts: list[Point]) -> tuple[list[float | None], list[float | None]]:
    """Split ``[(x, y) | None, ...]`` into ``(back_px_y, back_px_x)`` lists."""
    y_back: list[float | None] = []
    x_back: list[float | None] = []
    for pt in pts:
        if pt is None:
            y_back.append(None)
            x_back.append(None)
        else:
            x_back.append(float(pt[0]))
            y_back.append(float(pt[1]))
    return y_back, x_back


def apply_back_clicks(calib: Calibration, pts: list[Point],
                      frame_idx: int | None, *, min_points: int = 2) -> Calibration:
    """Attach the back clicks to a calibration (copy). Raises if too few."""
    y_back, x_back = back_points_to_lists(pts)
    n = sum(v is not None for v in y_back)
    if n < min_points:
        raise ValueError(f"only {n} back click(s), at least {min_points} are required")
    return calib.with_back_clicks(y_back, x_back, frame_idx)


def back_fit_preview(calib: Calibration) -> dict[str, float] | None:
    """Quick linear preview of the focal-free geometry from front/back clicks.

    Returns ``{"slope_front", "slope_back", "rho", "V_horizon", "cy"}`` in
    px/mL, -, mL and px, or ``None`` when fewer than two back clicks exist
    or the two lines are parallel. The full model lives in
    ``cylvision.uncertainty.curvature``; this is the console summary shown
    right after clicking.
    """
    if not calib.has_back_clicks():
        return None
    V = np.asarray(calib.graduations_ml, dtype=float)
    yf = np.asarray(calib.graduations_px_y, dtype=float)
    yb = np.array([np.nan if v is None else float(v) for v in calib.back_px_y or []], dtype=float)
    mask = ~np.isnan(yb)
    if mask.sum() < 2:
        return None
    slope_f, icpt_f = np.polyfit(V[mask], yf[mask], 1)
    slope_b, icpt_b = np.polyfit(V[mask], yb[mask], 1)
    if abs(slope_f - slope_b) < 1e-9 or abs(slope_f + slope_b) < 1e-12:
        return None
    V_int = (icpt_b - icpt_f) / (slope_f - slope_b)
    return {
        "slope_front": float(slope_f),
        "slope_back": float(slope_b),
        "rho": float((slope_f - slope_b) / (slope_f + slope_b)),
        "V_horizon": float(V_int),
        "cy": float(slope_f * V_int + icpt_f),
    }
