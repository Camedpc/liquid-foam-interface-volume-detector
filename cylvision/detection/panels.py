"""Rendering of the detection: analysis panels, strips and overlays.

Everything here is pure: arrays in, arrays out (BGR ``uint8``), no window.

Panels (``make_panels``), left to right, each topped by a label strip:

1. **Highlight** -- the crop with the detection overlays drawn by
   :func:`annotate_interfaces`: one dot per valid column (red = lower
   interface, teal = upper interface), the two mean lines, the axis
   (mauve), the band markers (orange = lower band, yellow = upper band),
   the gradient masks ``y_top`` / ``y_bottom`` (magenta) and the legacy
   extras of the upper interface (lowest dot, lowest-blob centroid).
2. **Signed gradient** -- heat map of ``S = sign * Gy`` normalised by
   ``max |S|`` AFTER masking (so the contrast is spent on the useful rows,
   not on the glass rim). Red tint = ``S > 0`` = lower-interface
   candidates, teal tint = ``S < 0`` = upper-interface candidates, dark
   grey = 0. The colours match the highlight panel whatever the polarity.
3. **Threshold mask** -- red where ``S > T_lower``, teal where
   ``-S > T_upper``, black elsewhere. What the detector actually sees.

``make_info_strip`` writes the parameters and the numbers under the
panels; ``make_specimen_panel`` draws the full frame with inward arrows
at the cylinder walls and scientific-style braces labelling the foam and
the liquid.

Crisp overlays: ``annotate_interfaces(..., zoom=k)`` and
``make_specimen_panel(..., zoom=k)`` first resize the image and THEN draw
anti-aliased strokes at the output resolution, so the strokes keep their
pixel size instead of sharing the scaling of the photograph. Pixel centres
map to ``(x + 0.5) * zoom``.

"Filled zones + volume labels": the mean interface lines are thick
(``theme.MEAN_LINE_W`` px, dark outline), the foam band between the two
interfaces and the liquid below the lower one receive a semi-transparent
wash (:func:`wash_zones`, ``show_wash``), and the volumes are printed next
to the zones (``foam 440 mL``, ``liquid 213 mL``, ``total 654 mL``) when a
``model_fn`` (``y_px -> V_mL``) is given.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np

from ..ui import theme
from .interfaces import DetectionParams, InterfaceResult

LABEL_HEIGHT = 24
SEP = 2
INFO_LINE_HEIGHT = 20
INFO_PAD = 6
ARROW_W = 7
ARROW_H = 7

FONT = cv2.FONT_HERSHEY_SIMPLEX
BRACE_FONT = cv2.FONT_HERSHEY_DUPLEX
BRACE_OUTLINE_W = 3
BRACE_INNER_W = 1

PANEL_LABELS: tuple[str, str, str] = ("Highlight", "Signed gradient", "Threshold mask")


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------

def dashed_hline(img: np.ndarray, y: int, color: tuple[int, int, int], *,
                 dash: int = 12, gap: int = 8, thickness: int = 1,
                 x_start: int = 0, x_stop: int | None = None) -> None:
    """Anti-aliased horizontal dashed line over ``[x_start, x_stop)``."""
    h, w = img.shape[:2]
    if not (0 <= y < h):
        return
    if x_stop is None:
        x_stop = w
    x = x_start
    while x < x_stop:
        x_end = min(x + dash, x_stop - 1)
        cv2.line(img, (x, y), (x_end, y), color, thickness, cv2.LINE_AA)
        x += dash + gap


def draw_inward_arrow(img: np.ndarray, x_anchor: int, y: int, side: str,
                      color: tuple[int, int, int],
                      arrow_w: int = ARROW_W, arrow_h: int = ARROW_H) -> None:
    """Small filled arrow anchored at column ``x_anchor`` pointing inwards.

    ``side="left"``: arrow on the left edge of a zone (apex to the right);
    ``side="right"``: arrow on the right edge (apex to the left). A 1 px
    black outline keeps it visible on any background.
    """
    H, W = img.shape[:2]
    if not (0 <= y < H):
        return
    if side == "left":
        if x_anchor < 0 or x_anchor + arrow_w >= W:
            return
        pts = np.array([[x_anchor + arrow_w, y],
                        [x_anchor, y - arrow_h // 2],
                        [x_anchor, y + arrow_h // 2]], dtype=np.int32)
    elif side == "right":
        if x_anchor >= W or x_anchor - arrow_w < 0:
            return
        pts = np.array([[x_anchor - arrow_w, y],
                        [x_anchor, y - arrow_h // 2],
                        [x_anchor, y + arrow_h // 2]], dtype=np.int32)
    else:
        raise ValueError("side must be 'left' or 'right'")
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def _stroke_line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
                 color: tuple[int, int, int]) -> None:
    cv2.line(img, p1, p2, theme.BRACE_OUTLINE_BGR, BRACE_OUTLINE_W, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, BRACE_INNER_W, cv2.LINE_AA)


def _stroke_text(img: np.ndarray, text: str, org: tuple[int, int], font: int,
                 scale: float, color: tuple[int, int, int], thick: int = 1) -> None:
    cv2.putText(img, text, org, font, scale, theme.BRACE_OUTLINE_BGR, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def mean_line_width(zoom: float = 1.0) -> int:
    """Stroke width of a mean interface line at the output resolution.

    ``theme.MEAN_LINE_W`` px whatever the down-scaling (the lines are drawn
    after the zoom, so they never thin out in a small figure); when the
    image is zoomed IN the width grows by one pixel per zoom unit so the
    stroke keeps a similar proportion to the magnified crop.
    """
    z = float(zoom)
    return int(theme.MEAN_LINE_W + (max(1, int(round(z))) - 1)) if z > 1.0 else int(theme.MEAN_LINE_W)


def draw_mean_line(img: np.ndarray, x0: int, x1: int, y: int, color: tuple[int, int, int],
                   line_w: int | None = None) -> None:
    """Thick horizontal mean line ``[x0, x1]`` at row ``y`` with a dark outline."""
    H = img.shape[0]
    if not (0 <= y < H) or x1 < x0:
        return
    w = int(theme.MEAN_LINE_W if line_w is None else line_w)
    ow = int(theme.MEAN_LINE_OUTLINE_W)
    if ow > 0:
        cv2.line(img, (x0, y), (x1, y), theme.BRACE_OUTLINE_BGR, w + 2 * ow, cv2.LINE_AA)
    cv2.line(img, (x0, y), (x1, y), color, w, cv2.LINE_AA)


def wash_zones(img: np.ndarray, x0: int, x1: int, y_upper: int | None, y_lower: int | None,
               y_bottom: int | None, *, alpha_foam: float | None = None,
               alpha_liquid: float | None = None) -> None:
    """Semi-transparent colour wash of the foam and liquid zones, in place.

    Foam = rows ``[y_upper, y_lower)`` (peach), liquid = rows
    ``[y_lower, y_bottom)`` (blue), both over columns ``[x0, x1]``. Rows and
    columns are in the coordinates of ``img`` (already zoomed). A zone whose
    bound is unknown (``None``) is skipped; with no lower interface nothing
    is drawn (the foam zone alone would be ambiguous).
    """
    H, W = img.shape[:2]
    x0 = max(0, int(x0))
    x1 = min(W - 1, int(x1))
    if x1 < x0 or y_lower is None:
        return
    a_f = theme.WASH_ALPHA_FOAM if alpha_foam is None else float(alpha_foam)
    a_l = theme.WASH_ALPHA_LIQUID if alpha_liquid is None else float(alpha_liquid)
    zones = []
    if y_upper is not None and y_lower > y_upper:
        zones.append((y_upper, y_lower, theme.FOAM_WASH_BGR, a_f))
    if y_bottom is not None and y_bottom > y_lower:
        zones.append((y_lower, y_bottom, theme.LIQUID_WASH_BGR, a_l))
    for r0, r1, color, alpha in zones:
        r0 = max(0, int(r0))
        r1 = min(H, int(r1))
        if r1 <= r0 or alpha <= 0.0:
            continue
        region = img[r0:r1, x0:x1 + 1]
        tint = np.empty_like(region)
        tint[:] = color
        cv2.addWeighted(region, 1.0 - alpha, tint, alpha, 0.0, dst=region)


def volume_labels(result: InterfaceResult, model_fn: Callable[[Any], Any] | None
                  ) -> dict[str, str]:
    """``{"foam": "foam 440 mL", "liquid": ..., "total": ...}`` for the zones found.

    Volumes come from ``model_fn(y_px) -> V_mL``; without a model the labels
    are the bare zone names. Only the zones whose interfaces were detected
    get an entry.
    """
    def V(y: float | None) -> float | None:
        if y is None or model_fn is None:
            return None
        v = float(np.asarray(model_fn(np.array([float(y)])), dtype=float).ravel()[0])
        return v if np.isfinite(v) else None

    out: dict[str, str] = {}
    V_l, V_t = V(result.y_lower), V(result.y_upper)
    if result.y_lower is not None:
        out["liquid"] = "liquid" if V_l is None else f"liquid {V_l:.0f} mL"
    if result.y_upper is not None and result.y_lower is not None:
        V_f = (V_t - V_l) if (V_l is not None and V_t is not None) else None
        out["foam"] = "foam" if V_f is None else f"foam {V_f:.0f} mL"
        out["total"] = "total" if V_t is None else f"total {V_t:.0f} mL"
    return out


def _label_text(img: np.ndarray, text: str, org: tuple[int, int], *, font_scale: float,
                thick: int, color: tuple[int, int, int] = theme.LABEL_BGR) -> None:
    """Outlined label (white on a black halo) legible on any background."""
    cv2.putText(img, text, org, BRACE_FONT, font_scale, theme.LABEL_OUTLINE_BGR, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, BRACE_FONT, font_scale, color, thick, cv2.LINE_AA)


def _fit_label(text: str, max_w: int, font_scale: float, thick: int) -> tuple[str, int, int] | None:
    """``(text, w, h)`` of the longest of ``text`` / its short form that fits ``max_w``."""
    short = text.split(" ", 1)[1] if " " in text and text.split(" ", 1)[1].endswith("mL") else text
    for cand in (text, short):
        (tw, th), _ = cv2.getTextSize(cand, BRACE_FONT, font_scale, thick)
        if tw <= max_w:
            return cand, tw, th
    return None


def draw_brace_right(img: np.ndarray, y_top: int, y_bot: int, x_anchor: int,
                     label: str, color: tuple[int, int, int] = theme.BRACE_BGR,
                     arm: int = 10, mid_arm: int = 14, font_scale: float = 0.7,
                     font_thick: int = 1) -> None:
    """Vertical ``}`` brace to the right of a zone, from ``y_top`` to ``y_bot``.

    Straight white strokes with a black outline (scientific-illustration
    style); the label is written to the right of the central tip::

        x_anchor ---.
                    |
                    |---- label
                    |
                  --'
    """
    H, W = img.shape[:2]
    if y_bot - y_top < 4:
        return
    y_top = max(0, min(y_top, H - 1))
    y_bot = max(y_top + 1, min(y_bot, H - 1))
    y_mid = (y_top + y_bot) // 2
    x_main = x_anchor + arm
    x_tip = x_main + mid_arm
    if x_tip >= W:
        return
    _stroke_line(img, (x_anchor, y_top), (x_main, y_top), color)
    _stroke_line(img, (x_anchor, y_bot), (x_main, y_bot), color)
    _stroke_line(img, (x_main, y_top), (x_main, y_mid), color)
    _stroke_line(img, (x_main, y_mid), (x_main, y_bot), color)
    _stroke_line(img, (x_main, y_mid), (x_tip, y_mid), color)
    (tw, th), _base = cv2.getTextSize(label, BRACE_FONT, font_scale, font_thick)
    text_x = x_tip + 6
    text_y = y_mid + th // 2
    if text_x + tw < W:
        _stroke_text(img, label, (text_x, text_y), BRACE_FONT, font_scale, color, font_thick)


# ---------------------------------------------------------------------------
# Overlay of a detection on an image (crop or full frame)
# ---------------------------------------------------------------------------

def annotate_interfaces(img_bgr: np.ndarray, result: InterfaceResult,
                        params: DetectionParams, *, zoom: float = 1.0,
                        offset_x: int = 0, show_extras: bool = False,
                        show_bands: bool = True, show_masks: bool = True,
                        show_axis: bool = True,
                        y_bounds: tuple[float | None, float | None] | None = None,
                        show_wash: bool = True, show_labels: bool = True,
                        model_fn: Callable[[Any], Any] | None = None,
                        line_w: int | None = None,
                        label_scale: float | None = None,
                        ) -> np.ndarray:
    """Draw the detection overlays on a copy of ``img_bgr``.

    Parameters
    ----------
    img_bgr
        The crop itself, or any image in which crop column ``0`` sits at
        ``x = offset_x`` (e.g. the full frame with ``offset_x = x_left``).
        Rows are shared (the crop is a full-height band).
    zoom
        Upsampling factor applied BEFORE drawing (``INTER_CUBIC`` when
        ``> 1``, ``INTER_AREA`` when ``< 1``). Strokes are anti-aliased at
        the output resolution; with ``zoom == 1`` the per-column dots are
        single pixels, as in the original tuner.
    show_extras
        Also draw the legacy upper-interface extras (lowest dot line,
        lowest-blob centroid line).
    y_bounds
        Optional ``(y_up, y_lo)`` empirical bounds drawn dashed (yellow /
        cyan) for the uncertainty figures.
    show_wash
        Semi-transparent wash of the foam zone (between the two interfaces)
        and of the liquid zone (lower interface down to ``params.y_bottom``),
        see :func:`wash_zones`. Drawn under every stroke.
    show_labels, model_fn
        Print ``foam ... mL`` / ``liquid ... mL`` inside the zones (volumes
        through ``model_fn``; bare zone names without it). Labels that do
        not fit the crop width fall back to the volume alone, then vanish.
    line_w
        Width of the mean lines at the output resolution; default
        :func:`mean_line_width` of the zoom.
    label_scale
        Font scale of the zone labels (default ``theme.LABEL_FONT_SCALE``).
    """
    if img_bgr.ndim == 2:
        base = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        base = img_bgr
    H0, W0 = base.shape[:2]
    z = float(zoom)
    if abs(z - 1.0) < 1e-9:
        out = base.copy()
    else:
        interp = cv2.INTER_CUBIC if z > 1.0 else cv2.INTER_AREA
        out = cv2.resize(base, (max(1, int(round(W0 * z))), max(1, int(round(H0 * z)))),
                         interpolation=interp)
    H, W = out.shape[:2]
    W_crop = int(result.lower_rows.shape[0])

    def bx(x_col: float) -> int:
        return int(round((x_col + offset_x + 0.5) * z)) if z != 1.0 else int(x_col) + int(offset_x)

    def by(y_row: float) -> int:
        return int(round((y_row + 0.5) * z)) if z != 1.0 else int(round(y_row))

    thin = max(1, int(round(z)))
    x_end = bx(W_crop - 1)
    x_beg = bx(0)
    y_bot_px = (by(params.y_bottom) if params.y_bottom is not None and 0 <= params.y_bottom < H0
                else H - 1)

    # Zone wash (foam / liquid) under everything else.
    if show_wash:
        wash_zones(out, x_beg, x_end,
                   None if result.y_upper is None else by(result.y_upper),
                   None if result.y_lower is None else by(result.y_lower), y_bot_px)

    # Gradient masks (magenta) and axis (mauve).
    if show_masks:
        if params.y_top > 0:
            cv2.line(out, (x_beg, by(params.y_top)), (x_end, by(params.y_top)),
                     theme.MASK_BGR, 1, cv2.LINE_AA)
        if params.y_bottom is not None and 0 <= params.y_bottom < H0 - 1:
            cv2.line(out, (x_beg, by(params.y_bottom)), (x_end, by(params.y_bottom)),
                     theme.MASK_BGR, 1, cv2.LINE_AA)
    if show_axis:
        cx = params.resolved_cx(W_crop)
        cv2.line(out, (bx(cx), 0), (bx(cx), H - 1), theme.AXIS_BGR, 1, cv2.LINE_AA)

    # Band markers: orange = lower band, yellow = upper band. Only when the
    # band does not degenerate to the whole crop width.
    if show_bands:
        for (x_lo, x_hi), r, color in (
            (result.band_lower, params.r_lower, theme.BAND_LOWER_BGR),
            (result.band_upper, params.r_upper, theme.BAND_UPPER_BGR),
        ):
            if r < W_crop // 2:
                for xb in (x_lo, x_hi):
                    cv2.line(out, (bx(xb), 0), (bx(xb), H - 1), color, thin, cv2.LINE_AA)

    # Per-column dots.
    dot_r = max(1, int(round(z)) - 1) if z > 1.0 else 0
    for rows, valid, color in (
        (result.upper_rows, result.upper_valid, theme.UPPER_BGR),
        (result.lower_rows, result.lower_valid, theme.LOWER_BGR),
    ):
        cols = np.flatnonzero(valid)
        if cols.size == 0:
            continue
        if dot_r == 0:
            ys = np.asarray([by(int(r)) for r in rows[cols]])
            xs = np.asarray([bx(int(c)) for c in cols])
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            out[ys[ok], xs[ok]] = color
        else:
            for c in cols:
                cv2.circle(out, (bx(int(c)), by(float(rows[c]))), dot_r, color, -1, cv2.LINE_AA)

    # Legacy extras of the upper interface.
    if show_extras:
        if result.upper_lowest is not None:
            cv2.line(out, (x_beg, by(result.upper_lowest)), (x_end, by(result.upper_lowest)),
                     theme.EXTRA_LOWEST_BGR, 1, cv2.LINE_AA)
        if result.upper_low_blob is not None:
            cv2.line(out, (x_beg, by(result.upper_low_blob)), (x_end, by(result.upper_low_blob)),
                     theme.EXTRA_BLOB_BGR, 1, cv2.LINE_AA)

    # Empirical bounds (dashed) below the solid mean lines.
    if y_bounds is not None:
        y_up, y_lo = y_bounds
        if y_up is not None:
            dashed_hline(out, by(y_up), theme.BOUND_UPPER_BGR, x_start=x_beg, x_stop=x_end + 1)
        if y_lo is not None:
            dashed_hline(out, by(y_lo), theme.BOUND_LOWER_BGR, x_start=x_beg, x_stop=x_end + 1)

    # Mean lines (thick, outlined) + inward arrow at the right edge of the crop.
    lw = mean_line_width(z) if line_w is None else max(1, int(line_w))
    arrow_k = max(1, int(round(lw / 3)))
    for y_mean, color in ((result.y_upper, theme.UPPER_BGR), (result.y_lower, theme.LOWER_BGR)):
        if y_mean is None:
            continue
        yy = by(y_mean)
        draw_mean_line(out, x_beg, x_end - ARROW_W * arrow_k, yy, color, lw)
        draw_inward_arrow(out, x_end, yy, "right", color,
                          arrow_w=ARROW_W * arrow_k, arrow_h=ARROW_H * arrow_k + lw)

    # Zone labels ("foam 440 mL" between the interfaces, "liquid 213 mL" below).
    if show_labels:
        fs = theme.LABEL_FONT_SCALE if label_scale is None else float(label_scale)
        ft = max(1, int(round(theme.LABEL_FONT_THICK * fs / theme.LABEL_FONT_SCALE)))
        labels = volume_labels(result, model_fn)
        max_w = (x_end - x_beg) - 12
        zones = []
        if "foam" in labels and result.y_upper is not None and result.y_lower is not None:
            zones.append((labels["foam"], by(result.y_upper), by(result.y_lower)))
        if "liquid" in labels and result.y_lower is not None:
            zones.append((labels["liquid"], by(result.y_lower), y_bot_px))
        for text, r0, r1 in zones:
            fit = _fit_label(text, max_w, fs, ft)
            if fit is None or r1 - r0 < 4:
                continue
            txt, tw, th = fit
            if r1 - r0 < th + 8:
                continue
            y_txt = (r0 + r1) // 2 + th // 2
            _label_text(out, txt, (x_beg + 6, y_txt), font_scale=fs, thick=ft)
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _draw_mask_lines(img: np.ndarray, params: DetectionParams) -> None:
    H, W = img.shape[:2]
    if params.y_top > 0:
        cv2.line(img, (0, params.y_top), (W - 1, params.y_top), theme.MASK_BGR, 1, cv2.LINE_AA)
    if params.y_bottom is not None and 0 <= params.y_bottom < H - 1:
        cv2.line(img, (0, params.y_bottom), (W - 1, params.y_bottom), theme.MASK_BGR, 1, cv2.LINE_AA)


def make_gradient_panel(S: np.ndarray) -> np.ndarray:
    """Signed-gradient heat map: red tint for ``S > 0``, teal for ``S < 0``."""
    smax = max(float(np.abs(S).max()) if S.size else 0.0, 1.0)
    t = np.clip(S.astype(np.float32) / smax, -1.0, 1.0)
    pos = np.maximum(t, 0.0)[..., None]
    neg = np.maximum(-t, 0.0)[..., None]
    bg = np.asarray(theme.SURFACE_BGR, dtype=np.float32)
    red = np.asarray(theme.LOWER_BGR, dtype=np.float32)
    teal = np.asarray(theme.UPPER_BGR, dtype=np.float32)
    img = bg * (1.0 - pos - neg) + red * pos + teal * neg
    return np.clip(img, 0, 255).astype(np.uint8)


def make_threshold_panel(S: np.ndarray, params: DetectionParams) -> np.ndarray:
    """Binary panel: red where ``S > T_lower``, teal where ``-S > T_upper``."""
    H, W = S.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[S > float(params.T_lower)] = theme.LOWER_BGR
    img[(-S) > float(params.T_upper)] = theme.UPPER_BGR
    return img


def make_highlight_panel(crop_bgr: np.ndarray, result: InterfaceResult,
                         params: DetectionParams) -> np.ndarray:
    """Crop + every overlay (see :func:`annotate_interfaces`)."""
    return annotate_interfaces(crop_bgr, result, params, zoom=1.0, show_extras=True)


def make_label_strip(total_w: int, panel_w: int | Sequence[int],
                     labels: Sequence[str] = PANEL_LABELS) -> np.ndarray:
    """Dark strip with one label per panel (``panel_w`` int or per-panel list)."""
    widths = [int(panel_w)] * len(labels) if isinstance(panel_w, int) else list(panel_w)
    strip = np.empty((LABEL_HEIGHT, total_w, 3), dtype=np.uint8)
    strip[:] = theme.STRIP_BG_BGR
    x_off = 0
    for i, lab in enumerate(labels):
        if i >= len(widths):
            break
        cv2.putText(strip, lab, (x_off + 6, 17), FONT, 0.55, theme.STRIP_TEXT_BGR, 1, cv2.LINE_AA)
        x_off += widths[i] + SEP
    return strip


def make_panels(crop_bgr: np.ndarray, Gy: np.ndarray, result: InterfaceResult,
                params: DetectionParams, *, panel_w: int | None = None,
                with_labels: bool = True) -> np.ndarray:
    """Horizontal stack ``[highlight | signed gradient | threshold mask]``.

    ``Gy`` is only used as a fallback when ``result.S`` does not match the
    crop (the signed gradient ``result.S`` is what the panels show).
    ``panel_w`` resizes every panel to that width (same scale on ``y``).
    A label strip is prepended on top unless ``with_labels`` is False.
    """
    H, W = crop_bgr.shape[:2]
    S = result.S if result.S is not None and result.S.shape == (H, W) else params.sign * Gy
    p_high = make_highlight_panel(crop_bgr, result, params)
    p_grad = make_gradient_panel(S)
    p_mask = make_threshold_panel(S, params)
    _draw_mask_lines(p_grad, params)
    _draw_mask_lines(p_mask, params)
    panels = [p_high, p_grad, p_mask]
    if panel_w is not None and int(panel_w) != W and W > 0:
        scale = int(panel_w) / W
        new_h = max(1, int(round(H * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        panels = [cv2.resize(p, (int(panel_w), new_h), interpolation=interp) for p in panels]
    ph, pw = panels[0].shape[:2]
    sep = np.empty((ph, SEP, 3), dtype=np.uint8)
    sep[:] = theme.SEPARATOR_BGR
    body = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    if not with_labels:
        return body
    strip = make_label_strip(body.shape[1], pw, PANEL_LABELS)
    return np.concatenate([strip, body], axis=0)


# ---------------------------------------------------------------------------
# Info strip
# ---------------------------------------------------------------------------

def _fmt(v: Any, spec: str = "{:.2f}") -> str:
    return spec.format(v) if isinstance(v, (int, float)) and v is not None else "n/a"


def _wrap_tokens(tokens: Sequence[str], width: int, scale: float, sep: str = "   ") -> list[str]:
    """Greedy wrap of ``tokens`` into lines narrower than ``width`` px."""
    lines: list[str] = []
    current = ""
    for tok in tokens:
        cand = tok if not current else current + sep + tok
        (tw, _), _ = cv2.getTextSize(cand, FONT, scale, 1)
        if current and tw + 20 > width:
            lines.append(current)
            current = tok
        else:
            current = cand
    if current:
        lines.append(current)
    return lines


def make_info_strip(width: int, result: InterfaceResult, params: DetectionParams,
                    model_fn: Callable[[Any], Any] | None = None,
                    frame_idx: int | None = None, t_sec: float | None = None,
                    fps: float | None = None, *, font_scale: float = 0.48) -> np.ndarray:
    """Text strip: parameters (line 1), detection numbers, optional volumes.

    The height adapts to the number of wrapped lines so the strip never
    clips on a narrow crop. ``model_fn`` (``y_px -> V_mL``) adds the
    liquid / total / foam volumes; ``frame_idx`` / ``t_sec`` / ``fps`` add a
    time stamp.
    """
    width = max(1, int(width))
    W_band_l = result.band_lower[1] - result.band_lower[0] + 1
    W_band_u = result.band_upper[1] - result.band_upper[0] + 1

    p_tokens = [
        f"polarity={params.polarity}", f"T_lower={params.T_lower:g}", f"T_upper={params.T_upper:g}",
        f"r_lower={params.r_lower}", f"r_upper={params.r_upper}",
        f"sigma={params.blur_sigma:.1f}", f"blur_h={params.blur_h}",
        f"min_h_upper={params.min_h_upper}", f"y_top={params.y_top}",
        f"y_bottom={'n/a' if params.y_bottom is None else params.y_bottom}",
        f"cx={'auto' if params.cx is None else params.cx}", f"channel={params.channel}",
    ]
    std_txt = f" +/- {result.std_lower:.1f} px" if result.std_lower is not None else ""
    lower_txt = (f"lower (liquid/foam): y = {_fmt(result.y_lower)}{std_txt}  "
                 f"({result.n_lower}/{W_band_l} cols)")
    upper_txt = (f"upper (foam/air): y = {_fmt(result.y_upper)}  "
                 f"({result.n_upper}/{W_band_u} cols)  "
                 f"lowest={_fmt(result.upper_lowest, '{:d}') if result.upper_lowest is not None else 'n/a'}  "
                 f"blob={_fmt(result.upper_low_blob)}")

    extra_tokens: list[str] = []
    if model_fn is not None:
        V_l = float(np.asarray(model_fn(result.y_lower), dtype=float)) if result.y_lower is not None else None
        V_t = float(np.asarray(model_fn(result.y_upper), dtype=float)) if result.y_upper is not None else None
        V_f = (V_t - V_l) if (V_l is not None and V_t is not None) else None
        extra_tokens += [f"V_liquid={_fmt(V_l, '{:.1f}')} mL", f"V_total={_fmt(V_t, '{:.1f}')} mL",
                         f"V_foam={_fmt(V_f, '{:.1f}')} mL"]
    if frame_idx is not None:
        extra_tokens.append(f"frame {frame_idx}")
    if t_sec is None and frame_idx is not None and fps:
        t_sec = frame_idx / float(fps)
    if t_sec is not None:
        extra_tokens.append(f"t = {t_sec:.2f} s")

    lines: list[tuple[str, tuple[int, int, int]]] = []
    for ln in _wrap_tokens(p_tokens, width, font_scale):
        lines.append((ln, theme.STRIP_TEXT_BGR))
    for ln in _wrap_tokens([lower_txt], width, font_scale):
        lines.append((ln, theme.LOWER_BGR))
    for ln in _wrap_tokens([upper_txt], width, font_scale):
        lines.append((ln, theme.UPPER_BGR))
    if extra_tokens:
        for ln in _wrap_tokens(extra_tokens, width, font_scale):
            lines.append((ln, theme.SUBTEXT_BGR))

    h = INFO_PAD * 2 + INFO_LINE_HEIGHT * len(lines)
    strip = np.empty((h, width, 3), dtype=np.uint8)
    strip[:] = theme.STRIP_BG_BGR
    y = INFO_PAD + 14
    for text, color in lines:
        cv2.putText(strip, text, (10, y), FONT, font_scale, color, 1, cv2.LINE_AA)
        y += INFO_LINE_HEIGHT
    return strip


# ---------------------------------------------------------------------------
# Specimen panel (full frame)
# ---------------------------------------------------------------------------

def make_specimen_panel(frame_bgr: np.ndarray, calib_crop: Sequence[int],
                        result: InterfaceResult, *,
                        labels: tuple[str, str] = ("foam", "liquid"),
                        draw_braces: bool = True,
                        draw_crop_box: bool = False,
                        zoom: float = 1.0,
                        show_wash: bool = True,
                        model_fn: Callable[[Any], Any] | None = None,
                        line_w: int | None = None,
                        label_scale: float | None = None) -> np.ndarray:
    """Full frame with the interfaces drawn across the cylinder and braces.

    ``calib_crop = (x_left, x_right, y_top, y_bottom)`` in frame
    coordinates. Rows of ``result`` are frame rows (full-height crop), so
    no remapping is needed.

    The frame is first resized by ``zoom`` (``INTER_AREA`` down, cubic up)
    and every stroke is drawn at the output resolution, so the lines, the
    wash and the text keep their size whatever the scale of the panel.
    The two mean lines cross the crop band (thick, outlined, red = lower,
    teal = upper) with inward arrows at the walls; the foam zone (between
    the interfaces) and the liquid zone (lower interface to ``y_bottom``)
    get the semi-transparent wash of :func:`wash_zones`; the braces to the
    right carry ``labels[0]`` / ``labels[1]`` completed with the volumes
    when ``model_fn`` is given (``"foam 440 mL"``), and a ``total`` label
    sits at the upper interface.
    """
    x_left0, x_right0, y_top0, y_bottom0 = (int(v) for v in calib_crop[:4])
    base = frame_bgr if frame_bgr.ndim == 3 else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    H0, W0 = base.shape[:2]
    z = float(zoom)
    if abs(z - 1.0) < 1e-9:
        panel = base.copy()
    else:
        interp = cv2.INTER_CUBIC if z > 1.0 else cv2.INTER_AREA
        panel = cv2.resize(base, (max(1, int(round(W0 * z))), max(1, int(round(H0 * z)))),
                           interpolation=interp)
    H, W = panel.shape[:2]

    def sx(x: float) -> int:
        return int(round((x + 0.5) * z)) if z != 1.0 else int(round(x))

    def sy(y: float) -> int:
        return int(round((y + 0.5) * z)) if z != 1.0 else int(round(y))

    y_top = max(0, min(sy(y_top0), H - 1))
    y_bottom = max(y_top + 1, min(sy(y_bottom0), H - 1))
    x_left = max(0, min(sx(x_left0), W - 1))
    x_right = max(x_left + 1, min(sx(x_right0), W - 1))

    y_up = result.y_upper
    y_lo = result.y_lower
    yu = None if y_up is None else sy(float(y_up))
    yl = None if y_lo is None else sy(float(y_lo))

    if show_wash:
        wash_zones(panel, x_left, x_right, yu, yl, y_bottom)

    if draw_crop_box:
        cv2.rectangle(panel, (x_left, y_top), (x_right, y_bottom), theme.MASK_BGR, 1, cv2.LINE_AA)

    lw = mean_line_width(z) if line_w is None else max(1, int(line_w))
    fs = theme.LABEL_FONT_SCALE if label_scale is None else float(label_scale)
    ft = max(1, int(round(theme.LABEL_FONT_THICK * fs / theme.LABEL_FONT_SCALE)))
    arrow_k = max(1, int(round(lw / 3)))
    aw, ah = ARROW_W * arrow_k, ARROW_H * arrow_k + lw

    vol = volume_labels(result, model_fn)
    brace_x = x_right + 6
    x_txt = brace_x + 10 + 14 + 6
    if draw_braces:
        foam_txt = vol.get("foam", labels[0]) if model_fn is not None else labels[0]
        liq_txt = vol.get("liquid", labels[1]) if model_fn is not None else labels[1]
        # Shrink the font (down to 2/3) then drop the zone name so that the
        # labels fit in the margin right of the cylinder.
        avail = W - x_txt - 4
        fs_b, ft_b = fs, ft
        for cand_fs in (fs, fs * 0.85, fs * 0.67):
            ft_b = max(1, int(round(theme.LABEL_FONT_THICK * cand_fs / theme.LABEL_FONT_SCALE)))
            widths = [cv2.getTextSize(t, BRACE_FONT, cand_fs, ft_b)[0][0] for t in (foam_txt, liq_txt)]
            fs_b = cand_fs
            if max(widths) <= avail:
                break
        else:
            fit_f = _fit_label(foam_txt, avail, fs_b, ft_b)
            fit_l = _fit_label(liq_txt, avail, fs_b, ft_b)
            foam_txt = fit_f[0] if fit_f else foam_txt
            liq_txt = fit_l[0] if fit_l else liq_txt
        if yu is not None and yl is not None and yl - yu >= 4:
            draw_brace_right(panel, yu, yl, brace_x, foam_txt, font_scale=fs_b, font_thick=ft_b)
        if yl is not None and y_bottom - yl >= 4:
            draw_brace_right(panel, yl, y_bottom, brace_x, liq_txt, font_scale=fs_b, font_thick=ft_b)
        fs, ft = fs_b, ft_b

    for yy, color in ((yu, theme.UPPER_BGR), (yl, theme.LOWER_BGR)):
        if yy is None:
            continue
        draw_mean_line(panel, x_left + aw, x_right - aw, yy, color, lw)
        draw_inward_arrow(panel, x_left, yy, "left", color, arrow_w=aw, arrow_h=ah)
        draw_inward_arrow(panel, x_right, yy, "right", color, arrow_w=aw, arrow_h=ah)

    # "total ... mL" at the upper interface, to the right of the brace tip.
    if draw_braces and model_fn is not None and yu is not None and yl is not None and "total" in vol:
        fit = _fit_label(vol["total"], W - x_txt - 4, fs, ft)
        if fit is not None:
            txt, tw, th = fit
            y_txt = yu - 6 if (yl - yu) < 3 * th else yu + th // 2
            if y_txt - th >= 0:
                _label_text(panel, txt, (x_txt, y_txt), font_scale=fs, thick=ft, color=theme.UPPER_BGR)
    return panel


__all__ = [
    "LABEL_HEIGHT", "SEP", "PANEL_LABELS",
    "dashed_hline", "draw_inward_arrow", "draw_brace_right",
    "mean_line_width", "draw_mean_line", "wash_zones", "volume_labels",
    "annotate_interfaces", "make_gradient_panel", "make_threshold_panel",
    "make_highlight_panel", "make_label_strip", "make_panels",
    "make_info_strip", "make_specimen_panel",
]
