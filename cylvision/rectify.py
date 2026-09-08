"""Cylinder rectification by inverse projection.

Every horizontal cross-section of the cylinder (a graduation ring, the
liquid/foam interface) is a circle that the camera sees as an arc of
ellipse; the interface detector averages the per-column rows over a band of
columns, so the vertical span of that arc inflates the per-column spread
(the *method* term of the uncertainty budget) and is the origin of the
*curvature* term.  Rectifying the frame with the camera model turns every
arc into a horizontal line: the columns of the band then agree and the
method term drops.

Maths (pinhole, no tilt; see :mod:`cylvision.uncertainty.curvature`)
---------------------------------------------------------------------
With ``rho = R / D``, the camera horizon row ``cy``, the projected axis
column ``cx`` and the silhouette half-width ``a_px``, the circle whose
front point (``theta = 0``) sits at row ``v_click`` projects to::

    u(theta) - cx = a_px sin(theta) sqrt(1 - rho^2) / (1 - rho cos(theta))
    v(theta) - cy = (v_click - cy) (1 - rho) / (1 - rho cos(theta))

The rectified image is parameterised so that its column is linear in
``sin(theta)`` and its row is the front ("click") row of the circle
through the pixel.  The inverse mapping (output -> input), which is what
``cv2.remap`` needs, is therefore::

    sin(theta) = (x_out - cx) / a_px,        cos(theta) = sqrt(1 - sin^2)
    x_in = cx + a_px sin(theta) sqrt(1 - rho^2) / (1 - rho cos(theta))
    y_in = cy + (y_out - cy) (1 - rho) / (1 - rho cos(theta))

Outside the cylinder (``|sin(theta)| > 1``) the mapping is the identity, so
the background is left untouched.  At ``theta = 0`` both formulas reduce to
the identity: the axis column and every row on it are fixed points, hence
**the front graduation clicks keep their rows and the calibration
``y -> V`` stays valid on the rectified frame** (:func:`rectify_calibration`).

The mapping only needs ``rho``, ``cy``, ``cx`` and ``a_px``.  ``rho`` and
``cy`` come from the focal-free fit of the front + back graduation clicks
(``CameraGeometry.source == "focal_free"``), which is the only branch of
:func:`~cylvision.uncertainty.geometry_from_calibration` that measures the
horizon directly; with front clicks only the horizon is a guess and the
rectification is not recommended.  ``cx`` and ``a_px`` default to the
calibration crop (``(x_left + x_right) / 2`` and half its width); when the
crop was clicked a little wide, :func:`wall_edges` measures the glass walls
on a frame instead (the reference implementation did that; on the run of
the README the two differ by 2-3 px).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from cylvision.detection.interfaces import DetectionParams, InterfaceResult, detect_from_crop
from cylvision.uncertainty.curvature import CameraGeometry, project_graduation
from cylvision.uncertainty.empirical import method_uncertainty_from_result

Maps = tuple[np.ndarray, np.ndarray]
_SQRT3 = float(np.sqrt(3.0))


# --------------------------------------------------------------------------- maps
def rectification_maps(geom: CameraGeometry, W: int, H: int, *,
                       cx_px: float | None = None, a_px: float | None = None) -> Maps:
    """Build the ``(map_x, map_y)`` pair for :func:`rectify_frame`.

    ``W`` x ``H`` is the size of the *full* frame (the maps are expressed in
    full-frame coordinates; slice them to remap a crop).  ``cx_px`` /
    ``a_px`` override the projected axis column and silhouette half-width of
    ``geom`` (for instance the values measured by :func:`wall_edges`).  The
    tilt of ``geom`` is ignored (no-tilt model; the focal-free geometry has
    ``alpha = 0`` by construction).
    """
    rho = float(geom.rho)
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    cx = float(geom.cx_px if cx_px is None else cx_px)
    a = float(geom.a_px if a_px is None else a_px)
    if a <= 0:
        raise ValueError("a_px must be positive")
    cy = float(geom.cy_px)

    x_out, y_out = np.meshgrid(np.arange(int(W), dtype=np.float32), np.arange(int(H), dtype=np.float32))
    s = (x_out - cx) / a                       # sin(theta) of the output column
    inside = np.abs(s) <= 1.0
    s_c = np.clip(s, -1.0, 1.0)
    c = np.sqrt(np.maximum(0.0, 1.0 - s_c * s_c))   # cos(theta) >= 0: front half only
    denom = 1.0 - rho * c
    x_in = cx + a * s_c * float(np.sqrt(1.0 - rho * rho)) / denom
    y_in = cy + (y_out - cy) * (1.0 - rho) / denom
    map_x = np.where(inside, x_in, x_out).astype(np.float32)
    map_y = np.where(inside, y_in, y_out).astype(np.float32)
    return map_x, map_y


def rectify_frame(frame: np.ndarray, maps: Maps, *, interpolation: int = cv2.INTER_LINEAR,
                  border_mode: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Remap ``frame`` (full frame, or any image the maps were built for)."""
    map_x, map_y = maps
    return cv2.remap(frame, map_x, map_y, interpolation, borderMode=border_mode)


def crop_maps(maps: Maps, x_left: int, x_right: int) -> Maps:
    """Slice full-frame maps to the crop band ``[x_left, x_right]`` (inclusive).

    ``cv2.remap`` outputs an image of the shape of the maps, so remapping
    the full frame with the sliced maps yields the rectified crop directly.
    """
    map_x, map_y = maps
    return map_x[:, int(x_left):int(x_right) + 1], map_y[:, int(x_left):int(x_right) + 1]


def forward_rows(v_click_px: float, geom: CameraGeometry, x_out: np.ndarray, *,
                 cx_px: float | None = None, a_px: float | None = None) -> np.ndarray:
    """Row, in the *rectified* image, of the circle ``v_click`` at the output columns ``x_out``.

    By construction it is ``v_click`` for every column inside the cylinder
    (used by the tests and the graduation figure); outside, where the mapping
    is the identity, the circle is not visible and NaN is returned.
    """
    cx = float(geom.cx_px if cx_px is None else cx_px)
    a = float(geom.a_px if a_px is None else a_px)
    s = (np.asarray(x_out, dtype=float) - cx) / a
    out = np.full(s.shape, float(v_click_px))
    out[np.abs(s) > 1.0] = np.nan
    return out


def rectify_calibration(calib: Any) -> Any:
    """The calibration to use on rectified frames.

    The front clicks sit on the axis column, a fixed point of the mapping,
    so their rows, the crop band, the masks and the fitted ``y -> V`` models
    are unchanged: the returned object is a copy with the *back* clicks
    removed (the back face is not part of the front-half parameterisation
    and its rows no longer mean anything) and ``extra["rectified"] = True``.
    """
    extra = dict(getattr(calib, "extra", {}) or {})
    extra["rectified"] = True
    try:
        return replace(calib, back_px_y=None, back_px_x=None, back_frame_idx=None, extra=extra)
    except TypeError:  # not a dataclass (tests use SimpleNamespace)
        import copy
        c = copy.copy(calib)
        c.back_px_y = None
        c.back_px_x = None
        c.back_frame_idx = None
        c.extra = extra
        return c


# --------------------------------------------------------------------------- wall edges
def wall_edges(frame_bgr: np.ndarray, y_band: tuple[int, int], x_hint: tuple[int, int], *,
               tolerance_px: float = 60.0) -> tuple[float, float] | None:
    """Columns of the two glass walls from the peaks of ``|dI/dx|``.

    The horizontal Sobel gradient is averaged over the rows ``y_band`` and
    the two strongest peaks near ``x_hint`` (the clicked crop) are kept.
    Returns ``None`` when the peaks are missing, too far from the hint or
    give an implausible width.  Pass the result of several frames to
    :func:`median_edges`.
    """
    from scipy.signal import find_peaks

    H, W = frame_bgr.shape[:2]
    y0, y1 = max(0, int(y_band[0])), min(H, int(y_band[1]))
    if y1 - y0 < 20:
        return None
    band = frame_bgr[y0:y1]
    gray = (cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    profile = np.convolve(np.abs(grad_x).mean(axis=0), np.ones(7) / 7.0, mode="same")
    margin = max(float(tolerance_px), (x_hint[1] - x_hint[0]) * 0.2)
    lo, hi = max(0, int(x_hint[0] - margin)), min(W, int(x_hint[1] + margin))
    search = np.zeros_like(profile)
    search[lo:hi] = profile[lo:hi]
    if search.max() <= 0:
        return None
    peaks, _ = find_peaks(search, prominence=search.max() * 0.15, distance=30)
    if len(peaks) < 2:
        return None
    x_l, x_r = sorted(sorted(peaks, key=lambda p: -profile[p])[:2])
    hint_w = x_hint[1] - x_hint[0]
    if abs((x_r - x_l) - hint_w) > 0.25 * hint_w:
        return None
    if abs(x_l - x_hint[0]) > tolerance_px or abs(x_r - x_hint[1]) > tolerance_px:
        return None
    return float(x_l), float(x_r)


def median_edges(edges: Iterable[tuple[float, float] | None]) -> tuple[float, float] | None:
    """Median ``(x_left, x_right)`` of several :func:`wall_edges` results."""
    pts = [e for e in edges if e is not None]
    if not pts:
        return None
    return float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))


# --------------------------------------------------------------------------- comparison
def method_uncertainty_max(mu: dict) -> float:
    """The reference definition, ``max(|V_up - V_mean|, |V_lo - V_mean|) / sqrt(3)``.

    ``mu`` is the dict of :func:`~cylvision.uncertainty.method_uncertainty`
    (range / (2 sqrt 3), the definition of the budget).  Both are the uniform
    rule; the reference one uses the larger half-range around the mean, so
    it is at least as large as the budget one (equal for a symmetric spread).
    """
    if not np.isfinite(mu.get("V_up", np.nan)) or not np.isfinite(mu.get("V_lo", np.nan)):
        return float("nan")
    return float(max(abs(mu["V_up"] - mu["V_mean"]), abs(mu["V_lo"] - mu["V_mean"])) / _SQRT3)


def compare_frame(frame_bgr: np.ndarray, calib: Any, params: DetectionParams, maps_crop: Maps,
                  model_fn: Callable[..., Any], *, which: str = "lower", tol_ml: float = 7.0
                  ) -> tuple[dict, InterfaceResult, InterfaceResult, np.ndarray, np.ndarray]:
    """Detect one interface on the original crop and on the rectified crop.

    Returns ``(row, result_orig, result_rect, crop_orig, crop_rect)`` where
    ``row`` holds the mean rows, the volumes, the bounds and both method
    uncertainties (``u_*`` = range / (2 sqrt 3), ``umax_*`` = reference
    max / sqrt 3) for the two images.
    """
    x_left, x_right = int(calib.x_left), int(calib.x_right)
    crop_o = frame_bgr[:, x_left:x_right + 1]
    crop_r = rectify_frame(frame_bgr, maps_crop)
    res_o, _ = detect_from_crop(crop_o, params)
    res_r, _ = detect_from_crop(crop_r, params)
    mu_o = method_uncertainty_from_result(res_o, model_fn, which=which, tol_ml=tol_ml)
    mu_r = method_uncertainty_from_result(res_r, model_fn, which=which, tol_ml=tol_ml)
    row = {}
    for tag, mu, res in (("orig", mu_o, res_o), ("rect", mu_r, res_r)):
        row[f"y_{tag}_px"] = mu["y_mean"]
        row[f"V_{tag}_mL"] = mu["V_mean"]
        row[f"y_up_{tag}_px"] = mu["y_up"]
        row[f"y_lo_{tag}_px"] = mu["y_lo"]
        row[f"range_{tag}_mL"] = mu["range_ml"]
        row[f"n_{tag}"] = int(getattr(res, f"n_{which}"))
        row[f"n_kept_{tag}"] = int(mu["n_kept"])
        row[f"u_{tag}_mL"] = mu["u_ml"]
        row[f"umax_{tag}_mL"] = method_uncertainty_max(mu)
    return row, res_o, res_r, crop_o, crop_r


COMPARE_COLUMNS: tuple[str, ...] = (
    "frame_idx", "t_sec",
    "y_orig_px", "V_orig_mL", "y_up_orig_px", "y_lo_orig_px", "range_orig_mL", "n_orig", "n_kept_orig",
    "u_orig_mL", "umax_orig_mL",
    "y_rect_px", "V_rect_mL", "y_up_rect_px", "y_lo_rect_px", "range_rect_mL", "n_rect", "n_kept_rect",
    "u_rect_mL", "umax_rect_mL",
)


def compare_video(video: Any, calib: Any, params: DetectionParams, maps: Maps, model_fn: Callable[..., Any],
                  *, start: int, n_frames: int, which: str = "lower", tol_ml: float = 7.0,
                  progress: Callable[[int, int, dict], None] | None = None) -> list[dict]:
    """Run :func:`compare_frame` on ``n_frames`` consecutive frames from ``start``.

    ``video`` is a :class:`cylvision.io.VideoSource`; ``t_sec`` counts from
    ``start``.  Frames the video cannot deliver stop the loop.
    """
    maps_crop = crop_maps(maps, calib.x_left, calib.x_right)
    rows: list[dict] = []
    fps = float(getattr(video, "fps", 0.0)) or 30.0
    for idx, frame in video.frames(int(start), int(start) + int(n_frames)):
        row, *_ = compare_frame(frame, calib, params, maps_crop, model_fn, which=which, tol_ml=tol_ml)
        row = {"frame_idx": int(idx), "t_sec": (idx - int(start)) / fps, **row}
        rows.append(row)
        if progress is not None:
            progress(idx - int(start) + 1, int(n_frames), row)
    return rows


def rolling_max(values: Sequence[float], window: int) -> np.ndarray:
    """Trailing rolling maximum over ``window`` samples (NaN-aware), as in the reference video."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    w = max(1, int(window))
    for i in range(v.size):
        seg = v[max(0, i - w + 1):i + 1]
        if np.any(np.isfinite(seg)):
            out[i] = np.nanmax(seg)
    return out


def summarize_compare(rows: Sequence[dict], *, fps: float | None = None) -> dict:
    """Median / mean / p95 of both uncertainty definitions, original vs rectified, and their ratios."""
    def stats(key: str) -> dict:
        v = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            nan = float("nan")
            return {"n": 0, "median": nan, "mean": nan, "p95": nan}
        return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
                "p95": float(np.percentile(v, 95))}

    out: dict[str, Any] = {"n_frames": len(rows)}
    for key in ("u", "umax"):
        o, r = stats(f"{key}_orig_mL"), stats(f"{key}_rect_mL")
        out[key] = {"orig": o, "rect": r,
                    "ratio_median": (r["median"] / o["median"]) if o["median"] else float("nan"),
                    "ratio_mean": (r["mean"] / o["mean"]) if o["mean"] else float("nan")}
        if fps:
            w = max(1, int(round(fps)))
            ro = rolling_max([r_["%s_orig_mL" % key] for r_ in rows], w)
            rr = rolling_max([r_["%s_rect_mL" % key] for r_ in rows], w)
            out[key]["rolling_max_1s"] = {"orig_mean": float(np.nanmean(ro)), "rect_mean": float(np.nanmean(rr)),
                                          "orig_median": float(np.nanmedian(ro)),
                                          "rect_median": float(np.nanmedian(rr))}
    return out


# --------------------------------------------------------------------------- residual curvature
def residual_delta_px(v_click_px: float, geom_true: CameraGeometry, geom_used: CameraGeometry, r_px: float,
                      *, cx_px: float | None = None, a_px: float | None = None) -> float:
    """Vertical span (px), inside the band, of a circle after a rectification built with the wrong geometry.

    The circle is projected with ``geom_true`` and pushed through the
    rectification computed from ``geom_used``; with ``geom_true ==
    geom_used`` the span is 0 (every arc becomes a row), otherwise it is the
    curvature term that survives a mis-estimated ``rho`` or horizon.
    """
    cx = float(geom_used.cx_px if cx_px is None else cx_px)
    a = float(geom_used.a_px if a_px is None else a_px)
    theta = np.linspace(-geom_true.theta_tangent * (1 - 1e-6), geom_true.theta_tangent * (1 - 1e-6), 4001)
    u, v = project_graduation(theta, v_click_px, geom_true)
    # inverse of the *used* mapping: x_out = cx + a sin(theta_used) with sin from x_in
    s = np.clip((u - cx) / a, -1.0, 1.0)
    # solve x_in = cx + a s sqrt(1-rho^2)/(1 - rho sqrt(1-s^2)) for s (monotone) by bisection
    rho = float(geom_used.rho)
    k = float(np.sqrt(1.0 - rho * rho))
    lo, hi = -np.ones_like(s), np.ones_like(s)
    target = (u - cx) / a
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = mid * k / (1.0 - rho * np.sqrt(np.maximum(0.0, 1.0 - mid * mid)))
        lo = np.where(f < target, mid, lo)
        hi = np.where(f < target, hi, mid)
    s_out = 0.5 * (lo + hi)
    c_out = np.sqrt(np.maximum(0.0, 1.0 - s_out * s_out))
    x_out = cx + a * s_out
    y_out = geom_used.cy_px + (v - geom_used.cy_px) * (1.0 - rho * c_out) / (1.0 - rho)
    m = np.abs(x_out - cx) <= float(r_px)
    if not m.any():
        return 0.0
    return float(y_out[m].max() - y_out[m].min())


# --------------------------------------------------------------------------- figures
HEX_ORIG = "#DC3C3C"      # original: red (the lower-interface colour of every README overlay)
HEX_RECT = "#1F77B4"      # rectified: blue
_INK, _MUTED, _GRID, _SPINE, _BG = "#1a1a1a", "#6e6e6e", "#dcdcdc", "#9a9a9a", "#ffffff"


def _figure(width: float, height: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), dpi=100, facecolor=_BG)
    FigureCanvasAgg(fig)
    return fig


def _style(ax, *, grid: str = "both") -> None:
    ax.set_facecolor(_BG)
    for name, sp in ax.spines.items():
        sp.set_color(_SPINE)
        sp.set_linewidth(0.8)
        if name in ("top", "right"):
            sp.set_visible(False)
    ax.tick_params(colors=_INK, labelsize=8.5, length=3, color=_SPINE)
    if grid:
        ax.grid(True, axis=grid, color=_GRID, lw=0.6)
        ax.set_axisbelow(True)


def _save(fig, out_png, dpi: int = 170) -> None:
    from pathlib import Path
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, facecolor=_BG, bbox_inches="tight", pad_inches=0.15)


def render_uncertainty_figure(rows: Sequence[dict], fps: float, out_png, *, key: str = "u",
                              scatter: tuple[np.ndarray, np.ndarray] | None = None,
                              scatter_labels: tuple[str, str] = ("original", "rectified"),
                              title: str | None = None) -> dict:
    """``u_method(t)`` original vs rectified (raw + 1 s rolling max) and, optionally, two image panels.

    ``key`` selects the definition plotted (``"u"`` = range / (2 sqrt 3), the
    budget one; ``"umax"`` = the reference max / sqrt 3).  ``scatter`` is a
    pair of BGR images (one frame before / after, annotated with the
    per-column detections) shown on the right.  Returns the summary dict.
    """
    t = np.array([float(r["t_sec"]) for r in rows])
    uo = np.array([float(r[f"{key}_orig_mL"]) for r in rows])
    ur = np.array([float(r[f"{key}_rect_mL"]) for r in rows])
    w = max(1, int(round(fps)))
    ro, rr = rolling_max(uo, w), rolling_max(ur, w)
    summ = summarize_compare(rows, fps=fps)[key]

    has_img = scatter is not None
    fig = _figure(15.0 if has_img else 10.0, 5.8)
    if has_img:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.28, wspace=0.08, left=0.055, right=0.99,
                              top=0.9, bottom=0.11)
        ax = fig.add_subplot(gs[:, 0])
        ax_o, ax_r = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    else:
        ax = fig.add_axes([0.09, 0.13, 0.88, 0.75])
        ax_o = ax_r = None

    ax.plot(t, uo, color=HEX_ORIG, lw=0.9, alpha=0.28)
    ax.plot(t, ur, color=HEX_RECT, lw=0.9, alpha=0.28)
    ax.plot(t, ro, color=HEX_ORIG, lw=2.2, solid_capstyle="round",
            label=f"original — 1 s rolling max   (median of the raw values {summ['orig']['median']:.2f} mL)")
    ax.plot(t, rr, color=HEX_RECT, lw=2.2, solid_capstyle="round",
            label=f"rectified — 1 s rolling max   (median {summ['rect']['median']:.2f} mL)")
    ax.axhline(summ["orig"]["median"], color=HEX_ORIG, lw=0.9, ls=":")
    ax.axhline(summ["rect"]["median"], color=HEX_RECT, lw=0.9, ls=":")
    ax.set_xlim(0.0, float(t.max()) if t.size else 1.0)
    top = np.nanmax(np.concatenate([ro, rr])) if t.size else 1.0
    ax.set_ylim(0.0, max(0.1, float(top) * 1.18))
    ax.set_xlabel("time  (s)", fontsize=9.5, color=_INK)
    if key == "u":
        label = r"$u_{\mathrm{method}}$  (mL)   =  range / (2$\sqrt{3}$)"
    else:
        label = r"$u_{\mathrm{method}}$  (mL)   =  max half-range / $\sqrt{3}$"
    ax.set_ylabel(label, fontsize=9.5, color=_INK)
    _style(ax)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor=_BG, edgecolor=_SPINE, framealpha=0.92)
    ratio = summ["ratio_median"]
    ax.text(0.015, 0.03, f"rectified / original = {ratio:.2f} (medians), {summ['ratio_mean']:.2f} (means)",
            transform=ax.transAxes, fontsize=9.5, color=_INK, va="bottom", ha="left",
            bbox=dict(facecolor=_BG, edgecolor="none", alpha=0.85, pad=2.0))
    if title:
        ax.set_title(title, fontsize=10.5, color=_INK, loc="left", pad=8)

    if has_img:
        for a, img, lab, col in ((ax_o, scatter[0], scatter_labels[0], HEX_ORIG),
                                 (ax_r, scatter[1], scatter_labels[1], HEX_RECT)):
            a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="nearest", aspect="auto")
            a.set_title(lab, fontsize=9, color=col, loc="left", pad=5)
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color(col)
                sp.set_linewidth(1.4)
    _save(fig, out_png)
    return summ


def render_graduations_figure(frame_bgr: np.ndarray, rect_bgr: np.ndarray, calib: Any, geom: CameraGeometry,
                              volume_to_row: Callable[[float], float], out_png, *, cx_px: float | None = None,
                              a_px: float | None = None, step_ml: float = 50.0, label_every_ml: float = 100.0,
                              margin_frac: float = 0.35, titles: tuple[str, str] = ("original", "rectified")
                              ) -> None:
    """The projected graduation circles on the frame (left) and their rectified rows (right).

    Rings every ``step_ml`` between the first and the last clicked graduation
    are drawn with the plasma ramp; on the right the same circles, pushed
    through the mapping, are horizontal (drawn from :func:`forward_rows`,
    not as ``axhline``, so the figure is a check of the maths).
    """
    import matplotlib
    cmap = matplotlib.colormaps["plasma"]
    cx = float(geom.cx_px if cx_px is None else cx_px)
    a = float(geom.a_px if a_px is None else a_px)
    H, W = frame_bgr.shape[:2]
    grads = np.asarray(calib.graduations_ml, dtype=float)
    V_show = np.arange(grads.min(), grads.max() + 0.5 * step_ml, step_ml)
    x0, x1 = max(0.0, cx - a * (1.0 + margin_frac)), min(float(W), cx + a * (1.0 + margin_frac))
    rows = np.array([float(volume_to_row(V)) for V in V_show])
    y_lo_view, y_hi_view = max(0.0, rows.min() - 40.0), min(float(H), rows.max() + 40.0)

    fig = _figure(7.2, 9.0)
    gs = fig.add_gridspec(1, 2, wspace=0.05, left=0.02, right=0.98, top=0.93, bottom=0.03)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    theta = np.linspace(-geom.theta_tangent * (1 - 1e-6), geom.theta_tangent * (1 - 1e-6), 601)
    xs_out = np.arange(int(np.floor(cx - a)), int(np.ceil(cx + a)) + 1, dtype=float)
    for ax, img in zip(axes, (frame_bgr, rect_bgr)):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="bilinear")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y_hi_view, y_lo_view)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(_SPINE)
    for j, (V, y_click) in enumerate(zip(V_show, rows)):
        color = cmap(0.05 + 0.85 * j / max(1, len(V_show) - 1))
        is_clicked = bool(np.any(np.isclose(grads, V)))
        lw, ls = (1.4, "-") if is_clicked else (0.9, "--")
        u, v = project_graduation(theta, y_click, geom)
        # the projection uses geom.cx/a; shift it to the cx/a of the mapping when they differ
        u = cx + (u - geom.cx_px) * (a / geom.a_px)
        axes[0].plot(u, v, color=color, lw=lw, ls=ls, alpha=0.9)
        axes[1].plot(xs_out, forward_rows(y_click, geom, xs_out, cx_px=cx, a_px=a), color=color, lw=lw, ls=ls,
                     alpha=0.9)
        if float(V) % label_every_ml == 0:
            for ax in axes:
                ax.text(x1 - 4, y_click, f"{V:.0f} mL", color=color, fontsize=8, va="center", ha="right",
                        bbox=dict(facecolor=_BG, edgecolor="none", alpha=0.6, pad=1.0))
    if 0 <= geom.cy_px <= H:
        for ax in axes:
            ax.axhline(geom.cy_px, color="#eb6834", lw=1.0, ls=":")
        axes[0].text(x0 + 6, geom.cy_px - 6, "camera horizon", fontsize=8, color="#eb6834", va="bottom",
                     bbox=dict(facecolor=_BG, edgecolor="none", alpha=0.75, pad=1.2))
    axes[0].set_title(f"{titles[0]}\ngraduation circles, ρ = {geom.rho:.3f}, horizon y = {geom.cy_px:.0f} px",
                      fontsize=9, color=_INK, loc="left", pad=6)
    axes[1].set_title(f"{titles[1]}\nthe same circles through the inverse projection", fontsize=9,
                      color=_INK, loc="left", pad=6)
    _save(fig, out_png)


__all__ = [
    "Maps", "COMPARE_COLUMNS", "rectification_maps", "rectify_frame", "crop_maps", "forward_rows",
    "rectify_calibration", "wall_edges", "median_edges", "method_uncertainty_max", "compare_frame",
    "compare_video", "rolling_max", "summarize_compare", "residual_delta_px",
    "render_uncertainty_figure", "render_graduations_figure", "HEX_ORIG", "HEX_RECT",
]
