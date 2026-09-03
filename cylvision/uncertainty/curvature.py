"""Cylinder-curvature contribution to the reading uncertainty.

A graduation mark on a graduated cylinder is physically a *horizontal circle*
drawn around the glass.  Seen through a camera it does not project to a
straight line but to an arc of ellipse.  The interface detector averages the
per-column interface row over a band of half-width ``r_px`` around the
cylinder axis, so it samples an arc whose vertical span, converted to mL
through the calibration slope, is a reading uncertainty that depends on the
level: it vanishes on the camera horizon and grows linearly away from it.

Camera model (pinhole, optional pitch)
--------------------------------------
World frame: cylinder axis = ``Z`` (up), physical radius ``R`` (cm).  The
camera sits at ``(0, -D, h_cam)``, looks along ``+Y`` and is pitched by an
angle ``alpha`` (positive = camera pointing up).  A point of the cylinder
surface at height ``Z`` and azimuth ``theta`` (measured from the point closest
to the camera) is ``P = (R sin(theta), -R cos(theta), Z)``.

After the translation to the camera and the rotation of ``-alpha`` around the
camera ``X`` axis::

    x_c = R sin(theta)
    y_c = (D - R cos(theta)) cos(alpha) + (Z - h_cam) sin(alpha)     (depth)
    z_c = -(D - R cos(theta)) sin(alpha) + (Z - h_cam) cos(alpha)    (camera "up")

Pinhole projection with the image ``v`` axis pointing DOWN (OpenCV)::

    u - cx = f x_c / y_c = f R sin(theta) / y_c
    v - cy = -f z_c / y_c = f [(D - R cos(theta)) sin(alpha) - (Z - h_cam) cos(alpha)] / y_c

where ``f`` is the focal length in pixels and ``(cx, cy)`` the principal
point; ``cy`` is the row of the *camera horizon* (the row where the optical
axis, once un-tilted, meets the image).

Link with the Möbius calibration model
--------------------------------------
At the front of the cylinder (``theta = 0``) with ``p = D - R`` (camera to
front face) and ``B = Z - h_cam`` (height above the camera)::

    q = v_click - cy = f (p sin(alpha) - B cos(alpha)) / (p cos(alpha) + B sin(alpha))

For a cylinder ``Z`` is affine in the volume ``V`` (``Z = V / (pi R^2) + Z_0``),
hence ``B`` is affine in ``V`` and ``q`` is a ratio of two affine functions of
``V``.  Inverting, ``V = (a v + b) / (c v + 1)``: this is *exactly* the Möbius
model of ``cylvision.calibration.models``, and ``c`` is the signature of the
tilt (``c = 0`` if and only if ``alpha = 0``).  Identifying the coefficients
gives ``tan(alpha) = c f / (1 + c cy)``.

Vertical offset of the arc
--------------------------
Subtracting the front reading from the projection of the same circle at
azimuth ``theta`` factorises remarkably::

    v(theta) - v_click = f R (1 - cos(theta)) B / (D_0 D_theta)
        D_0     = p cos(alpha) + B sin(alpha)                  (depth of the front point)
        D_theta = D_0 + R (1 - cos(theta)) cos(alpha)          (depth at azimuth theta)

``B`` and ``D_0`` are recovered from the clicked row alone::

    B   = p (f sin(alpha) - q cos(alpha)) / (f cos(alpha) + q sin(alpha))
    D_0 = p f / (f cos(alpha) + q sin(alpha))

The horizontal projection ``u - cx = f R sin(theta) / D_theta`` is maximal at
the silhouette, ``cos(theta_t) = rho`` with ``rho = R / D`` (no-tilt case), so
the visible half-width of the cylinder in the image is
``a_px = f rho / sqrt(1 - rho^2)``.

Span inside the averaging band
------------------------------
The detector keeps the columns ``|u - cx| <= r_px``.  The arc inside the band
spans ``Delta = max(v) - min(v)`` pixels (``Delta`` is the *full* span).  For a
band much narrower than the cylinder the small-angle expansion gives
(``theta_r ~ r D_0 / (f R)`` at the band edge)::

    Delta ~ r^2 |B| / (2 f R)  =  |v_click - cy| (r / f)^2 (1 - rho) / (2 rho)     (alpha = 0)

i.e. ``Delta`` is proportional to the distance to the horizon and to the square
of the band half-width.  The code evaluates the exact expression numerically.

The per-column rows are spread over an interval of width ``Delta`` with an
(unknown) sign relative to the clicked front reading; the standard uncertainty
of a quantity known to lie in an interval of width ``Delta`` is that of the
uniform distribution, ``u_curvature = Delta / sqrt(12) = Delta / (2 sqrt(3))``.
Converting to mL uses the local calibration slope ``|dV/dy|``.

Inversion from a calibration file
---------------------------------
Two branches, chosen by :func:`geometry_from_calibration`:

* **Front + back clicks** (``focal_free``, preferred when the optional back
  graduation clicks exist).  At ``theta = 0`` and ``theta = pi`` the rows of
  the same circle are ``v_f - cy = -f B / (D - R)`` and
  ``v_b - cy = -f B / (D + R)`` (no tilt).  Both are affine in ``V`` with slopes
  ``s_f`` and ``s_b`` whose ratio is ``(D + R) / (D - R) = (1 + rho) / (1 - rho)``::

      rho = (s_f - s_b) / (s_f + s_b)

  and the two lines cross at ``B = 0``, i.e. on the horizon: ``cy`` is the
  crossing row and ``V_horizon`` the corresponding volume.  Neither ``R``,
  ``f`` nor ``D`` is needed; with a tilt the crossing row becomes
  ``cy + f tan(alpha)`` which is exactly what the projection needs, so
  ``alpha`` is set to 0 in this branch.  ``D = R / rho`` and
  ``f = a_px sqrt(1 - rho^2) / rho`` are then derived for information.

* **Physical radius only** (``no_tilt`` / ``mobius_tilt``).  With the linear
  slope ``s = |dv/dV|`` of the front clicks and ``a_px = (x_right - x_left) / 2``::

      K = (s pi R^3 / a_px)^2 = (1 + rho) / (1 - rho)
      rho = (K - 1) / (K + 1),   D = R / rho,   f = a_px sqrt(1 - rho^2) / rho

  ``cy`` defaults to the middle of the clicked graduations and ``alpha`` is
  read from the Möbius ``c`` unless given explicitly.  This inversion is
  sensitive (``d rho ~ 0.4 dK`` with ``K ~ 1.2``: a 2 % error on ``a_px`` moves
  ``rho`` by about 0.05), which is why back clicks are worth collecting.

All rendering here uses the Agg canvas directly (no pyplot, no window).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# Single definition of the local calibration slope |dV/dy| (mL/px), shared
# with the pixel term of the budget. ``cylvision.calibration.models`` only
# depends on numpy/scipy, so this import cannot create a cycle.
from cylvision.calibration.models import slope_ml_per_px

GeometrySource = Literal["no_tilt", "mobius_tilt", "focal_free"]

# --------------------------------------------------------------------------- palette
# Light, README-friendly palette shared by every uncertainty figure.
INK = "#1a1a1a"
MUTED = "#6e6e6e"
GRID = "#dcdcdc"
SPINE = "#9a9a9a"
BG = "#ffffff"
GLASS = "#e8eef7"
BAND = "#2a78d6"          # averaging band (blue)
BAND_FILL = "#2a78d6"
HORIZON = "#eb6834"       # camera horizon (orange)
DELTA_FILL = "#9ec5f4"    # Delta wedge (light blue)
DELTA_EDGE = "#1c5cab"
U_LINE = "#1c5cab"        # u_curvature curve (dark blue)
ARC = "#1baf7a"           # visible arc (aqua)
CAMERA = "#eda100"        # camera marker (yellow)
CLICK_FRONT = "#1c5cab"
CLICK_BACK = "#eb6834"
GRAD_CMAP = "plasma"      # ordered graduations: single perceptual ramp, truncated
GRAD_CMAP_RANGE = (0.05, 0.88)


# --------------------------------------------------------------------------- dataclasses
@dataclass
class CameraGeometry:
    """Pinhole geometry of the camera relative to the cylinder.

    Lengths in cm (``R_cm``, ``D_cm``) or pixels (``a_px``, ``f_px``, ``cy_px``,
    ``cx_px``); ``rho = R / D``; ``alpha_rad`` is the pitch (positive = camera
    pointing up); ``source`` tells which inversion produced the numbers.
    ``v_horizon_ml`` is the volume whose graduation lies on the camera horizon
    (known from the focal-free branch only).
    """

    R_cm: float
    a_px: float
    rho: float
    D_cm: float
    f_px: float
    cy_px: float
    alpha_rad: float
    cx_px: float
    source: GeometrySource
    v_horizon_ml: float | None = None

    @property
    def p_cm(self) -> float:
        """Distance from the camera to the front face of the cylinder."""
        return self.D_cm - self.R_cm

    @property
    def theta_tangent(self) -> float:
        """Half-angle of the visible arc (silhouette), ``arccos(rho)``."""
        return float(np.arccos(np.clip(self.rho, -1.0, 1.0)))

    @property
    def alpha_deg(self) -> float:
        return float(np.degrees(self.alpha_rad))

    def summary(self) -> str:
        """One-line human readable summary (used in figure titles)."""
        return (f"R = {self.R_cm:.2f} cm, D = {self.D_cm:.1f} cm, f = {self.f_px:.0f} px, "
                f"rho = {self.rho:.3f}, alpha = {self.alpha_deg:+.2f} deg [{self.source}]")


@dataclass
class FocalFreeFit:
    """Result of the front + back straight-line fits (see module docstring)."""

    rho: float
    cy_px: float
    slope_front: float
    intercept_front: float
    slope_back: float
    intercept_back: float
    v_horizon_ml: float
    n_pairs: int


# --------------------------------------------------------------------------- inversions
def invert_no_tilt(a_px: float, slope_px_per_ml: float, R_cm: float) -> tuple[float, float, float]:
    """Invert ``(a_px, |dv/dV|, R)`` into ``(rho, D_cm, f_px)`` assuming no tilt."""
    if a_px <= 0 or R_cm <= 0:
        raise ValueError("a_px and R_cm must be positive")
    K = (abs(slope_px_per_ml) * np.pi * R_cm ** 3 / a_px) ** 2
    rho = (K - 1.0) / (K + 1.0)
    rho = float(np.clip(rho, 1e-4, 0.999))
    D_cm = R_cm / rho
    f_px = a_px * float(np.sqrt(1.0 - rho ** 2)) / rho
    return rho, float(D_cm), float(f_px)


def alpha_from_mobius(c_mobius: float, f_px: float, cy_px: float) -> float:
    """Pitch (rad) from the Möbius ``c`` coefficient: ``tan(alpha) = c f / (1 + c cy)``."""
    denom = 1.0 + cy_px * c_mobius
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.arctan(c_mobius * f_px / denom))


def fit_focal_free(V_ml: Sequence[float], y_front: Sequence[float],
                   y_back: Sequence[float | None]) -> FocalFreeFit | None:
    """Fit ``rho`` and the horizon row from paired front/back graduation rows.

    ``y_back`` may contain ``None``/NaN for graduations without a back click.
    Returns ``None`` with fewer than two complete pairs or degenerate slopes.
    """
    V = np.asarray(V_ml, dtype=float)
    yf = np.asarray(y_front, dtype=float)
    yb = np.array([np.nan if v is None else float(v) for v in y_back], dtype=float)
    mask = np.isfinite(V) & np.isfinite(yf) & np.isfinite(yb)
    if int(mask.sum()) < 2:
        return None
    sf, bf = np.polyfit(V[mask], yf[mask], 1)
    sb, bb = np.polyfit(V[mask], yb[mask], 1)
    if abs(sf - sb) < 1e-9 or abs(sf + sb) < 1e-9:
        return None
    rho = float((sf - sb) / (sf + sb))
    if not (0.0 < rho < 1.0):
        return None
    V_h = float((bb - bf) / (sf - sb))
    cy = float(sf * V_h + bf)
    return FocalFreeFit(rho=rho, cy_px=cy, slope_front=float(sf), intercept_front=float(bf),
                        slope_back=float(sb), intercept_back=float(bb),
                        v_horizon_ml=V_h, n_pairs=int(mask.sum()))


def geometry_from_calibration(calib: Any, *, alpha_deg: float | None = None,
                              use_back_clicks: bool = True) -> CameraGeometry:
    """Build the :class:`CameraGeometry` of a run from its calibration.

    Uses ``calib.cylinder.radius_cm``, ``calib.x_left``/``x_right``,
    ``calib.graduations_ml``/``graduations_px_y``, ``calib.mobius_popt`` and the
    optional ``calib.back_px_y``.  With back clicks (and ``use_back_clicks``)
    the focal-free branch is used and ``alpha`` defaults to 0; otherwise the
    physical-radius inversion is used and ``alpha`` defaults to the Möbius
    estimate.  ``alpha_deg`` overrides the pitch in both branches.
    """
    spec = getattr(calib, "cylinder", None)
    R_cm = getattr(spec, "radius_cm", None)
    if R_cm is None:
        raise ValueError("the cylinder spec has no radius_cm: the curvature model needs it")
    R_cm = float(R_cm)

    V = np.asarray(calib.graduations_ml, dtype=float)
    y = np.asarray(calib.graduations_px_y, dtype=float)
    if V.size < 2:
        raise ValueError("at least two graduations are required")
    x_left, x_right = float(calib.x_left), float(calib.x_right)
    a_px = (x_right - x_left) / 2.0
    cx_px = (x_left + x_right) / 2.0
    slope_front, _ = np.polyfit(V, y, 1)

    ff: FocalFreeFit | None = None
    back = getattr(calib, "back_px_y", None)
    if use_back_clicks and back is not None and len(back) == V.size:
        ff = fit_focal_free(V, y, back)

    if ff is not None:
        rho = ff.rho
        D_cm = R_cm / rho
        f_px = a_px * float(np.sqrt(1.0 - rho ** 2)) / rho
        cy_px = ff.cy_px
        alpha = float(np.radians(alpha_deg)) if alpha_deg is not None else 0.0
        return CameraGeometry(R_cm=R_cm, a_px=a_px, rho=rho, D_cm=float(D_cm), f_px=f_px,
                              cy_px=cy_px, alpha_rad=alpha, cx_px=cx_px, source="focal_free",
                              v_horizon_ml=ff.v_horizon_ml)

    rho, D_cm, f_px = invert_no_tilt(a_px, float(slope_front), R_cm)
    cy_px = float((y.min() + y.max()) / 2.0)
    source: GeometrySource
    if alpha_deg is not None:
        alpha = float(np.radians(alpha_deg))
        source = "no_tilt"
    else:
        popt = getattr(calib, "mobius_popt", None)
        c = float(popt[2]) if popt is not None and len(popt) >= 3 else 0.0
        alpha = alpha_from_mobius(c, f_px, cy_px)
        source = "mobius_tilt" if abs(alpha) > 0.0 else "no_tilt"
    return CameraGeometry(R_cm=R_cm, a_px=a_px, rho=rho, D_cm=D_cm, f_px=f_px, cy_px=cy_px,
                          alpha_rad=alpha, cx_px=cx_px, source=source, v_horizon_ml=None)


# --------------------------------------------------------------------------- projection
def _front_terms(v_click_px: float, geom: CameraGeometry) -> tuple[float, float]:
    """``(B_cm, D_0_cm)`` of the circle whose front point is at row ``v_click_px``."""
    q = float(v_click_px) - geom.cy_px
    sa, ca = float(np.sin(geom.alpha_rad)), float(np.cos(geom.alpha_rad))
    denom = geom.f_px * ca + q * sa
    if abs(denom) < 1e-9:
        denom = 1e-9 if denom >= 0 else -1e-9
    B = geom.p_cm * (geom.f_px * sa - q * ca) / denom
    D0 = geom.p_cm * geom.f_px / denom
    return float(B), float(D0)


def project_graduation(theta: np.ndarray, v_click_px: float,
                       geom: CameraGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Project the horizontal circle whose front point is at row ``v_click_px``.

    Returns ``(u_px, v_px)`` for every azimuth in ``theta`` (0 = front, pi =
    back).  ``v(0) == v_click_px`` exactly.
    """
    theta = np.asarray(theta, dtype=float)
    B, D0 = _front_terms(v_click_px, geom)
    ca = float(np.cos(geom.alpha_rad))
    one_mcos = 1.0 - np.cos(theta)
    D_theta = D0 + geom.R_cm * one_mcos * ca
    u = geom.cx_px + geom.f_px * geom.R_cm * np.sin(theta) / D_theta
    v = float(v_click_px) + geom.f_px * geom.R_cm * one_mcos * B / (D0 * D_theta)
    return u, v


def _visible_theta(geom: CameraGeometry, n: int = 4001) -> np.ndarray:
    th_t = geom.theta_tangent * (1.0 - 1e-6)
    return np.linspace(-th_t, th_t, n)


def arc_in_band(v_click_px: float, geom: CameraGeometry, r_px: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visible arc ``(u, v, mask)`` where ``mask`` flags the columns inside the band."""
    theta = _visible_theta(geom)
    u, v = project_graduation(theta, v_click_px, geom)
    mask = np.abs(u - geom.cx_px) <= float(r_px)
    return u, v, mask


def curvature_delta_px(v_click_px: float, geom: CameraGeometry, r_px: int) -> float:
    """Full vertical span (px) of the projected circle inside ``|u - cx| <= r_px``."""
    _, v, mask = arc_in_band(v_click_px, geom, r_px)
    if not mask.any():
        return 0.0
    v_in = v[mask]
    return float(v_in.max() - v_in.min())


def band_half_angle(v_click_px: float, geom: CameraGeometry, r_px: float) -> float:
    """Azimuth (rad) at which the arc leaves the averaging band."""
    theta = _visible_theta(geom)
    u, _ = project_graduation(theta, v_click_px, geom)
    inside = np.abs(u - geom.cx_px) <= float(r_px)
    if not inside.any():
        return 0.0
    return float(np.abs(theta[inside]).max())


# --------------------------------------------------------------------------- mL conversion
def _eval_model(model_fn: Callable[..., Any], y: float | np.ndarray) -> np.ndarray:
    return np.asarray(model_fn(np.atleast_1d(np.asarray(y, dtype=float))), dtype=float).ravel()


def volume_to_row_fn(calib: Any, model_fn: Callable[..., Any], *, margin_px: float = 60.0,
                     n: int = 4001) -> Callable[[float | np.ndarray], np.ndarray]:
    """Numerical inverse ``V -> y`` of a monotone calibration model.

    Built by sampling the model on a dense row grid spanning the clicked
    graduations (plus ``margin_px``) and interpolating.  Volumes outside the
    sampled range are clamped.
    """
    y_clicks = np.asarray(calib.graduations_px_y, dtype=float)
    y_lo = max(0.0, y_clicks.min() - margin_px)
    y_hi = y_clicks.max() + margin_px
    size = getattr(calib, "image_size", None)
    if size is not None:
        y_hi = min(y_hi, float(size[1]) - 1.0)
    y_grid = np.linspace(y_lo, y_hi, n)
    V_grid = _eval_model(model_fn, y_grid)
    order = np.argsort(V_grid)
    V_sorted, y_sorted = V_grid[order], y_grid[order]
    good = np.isfinite(V_sorted)
    V_sorted, y_sorted = V_sorted[good], y_sorted[good]

    def fn(V: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(V, dtype=float), V_sorted, y_sorted)

    return fn


def curvature_uncertainty_ml(calib: Any, geom: CameraGeometry, r_px: int,
                             model_fn: Callable[..., Any]) -> list[dict]:
    """Per-graduation curvature uncertainty.

    Each row: ``V_ml``, ``v_click_px``, ``delta_px``, ``slope_ml_per_px``,
    ``delta_ml`` and ``u_ml = delta_ml / (2 sqrt(3))``.
    """
    rows: list[dict] = []
    for V, v_click in zip(calib.graduations_ml, calib.graduations_px_y):
        delta_px = curvature_delta_px(float(v_click), geom, r_px)
        slope = slope_ml_per_px(model_fn, float(v_click))
        delta_ml = delta_px * slope
        rows.append({
            "V_ml": float(V),
            "v_click_px": float(v_click),
            "delta_px": float(delta_px),
            "slope_ml_per_px": float(slope),
            "delta_ml": float(delta_ml),
            "u_ml": float(delta_ml / (2.0 * np.sqrt(3.0))),
        })
    rows.sort(key=lambda r: r["V_ml"])
    return rows


def curvature_uncertainty_fn(calib: Any, geom: CameraGeometry, r_px: int,
                             model_fn: Callable[..., Any]) -> Callable[[float], float]:
    """``V -> u_curvature(V)`` (mL), linear interpolation between graduations."""
    rows = curvature_uncertainty_ml(calib, geom, r_px, model_fn)
    V_arr = np.array([r["V_ml"] for r in rows])
    u_arr = np.array([r["u_ml"] for r in rows])

    def fn(V: float | np.ndarray) -> Any:
        out = np.interp(np.asarray(V, dtype=float), V_arr, u_arr)
        return float(out) if np.ndim(V) == 0 else out

    return fn


# --------------------------------------------------------------------------- figure helpers
def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), dpi=100, facecolor=BG)
    FigureCanvasAgg(fig)
    return fig


def _save(fig: Figure, out_png: Path | str, dpi: int = 170) -> None:
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, facecolor=BG, bbox_inches="tight", pad_inches=0.15)


def _style_axes(ax, *, grid_axis: str = "both") -> None:
    ax.set_facecolor(BG)
    for name, s in ax.spines.items():
        s.set_color(SPINE)
        s.set_linewidth(0.8)
        if name in ("top", "right"):
            s.set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5, length=3, color=SPINE)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def _grad_colors(n: int) -> list:
    import matplotlib
    cmap = matplotlib.colormaps[GRAD_CMAP]
    lo, hi = GRAD_CMAP_RANGE
    if n <= 1:
        return [cmap(lo)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def _arc_rows(calib: Any, geom: CameraGeometry, r_px: float) -> list[dict]:
    """Geometry-only rows (no model): arc, band mask, span, B, per graduation."""
    rows = []
    for V, v_click in zip(calib.graduations_ml, calib.graduations_px_y):
        u, v, mask = arc_in_band(float(v_click), geom, r_px)
        B, _ = _front_terms(float(v_click), geom)
        if mask.any():
            v_min, v_max = float(v[mask].min()), float(v[mask].max())
        else:
            v_min = v_max = float(v_click)
        rows.append({"V_ml": float(V), "v_click_px": float(v_click), "u": u, "v": v,
                     "mask": mask, "v_min": v_min, "v_max": v_max,
                     "delta_px": v_max - v_min, "B_cm": B,
                     "theta_r": band_half_angle(float(v_click), geom, r_px)})
    rows.sort(key=lambda r: r["V_ml"])
    return rows


def _draw_top_view(ax, geom: CameraGeometry, r_px: float, theta_r: float, *, title: str | None) -> None:
    """Top view in units of R: camera on the left, cylinder disk at the origin."""
    dr = geom.D_cm / geom.R_cm
    cam = (-dr, 0.0)
    th_t = geom.theta_tangent

    ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor=GLASS, edgecolor=INK, lw=1.1, zorder=1))
    ax.plot([cam[0], 1.3], [0.0, 0.0], color=MUTED, lw=0.6, ls=":", zorder=1)
    for s in (+1.0, -1.0):
        ax.plot([cam[0], -np.cos(th_t)], [cam[1], s * np.sin(th_t)],
                color=MUTED, lw=0.7, ls="--", zorder=2)
    th = np.linspace(-th_t, th_t, 300)
    ax.plot(-np.cos(th), np.sin(th), color=ARC, lw=2.4, zorder=3, solid_capstyle="round")

    thr = np.linspace(-theta_r, theta_r, 120)
    ax.fill(np.concatenate(([cam[0]], -np.cos(thr), [cam[0]])),
            np.concatenate(([cam[1]], np.sin(thr), [cam[1]])),
            color=BAND_FILL, alpha=0.16, lw=0, zorder=2)
    ax.plot(-np.cos(thr), np.sin(thr), color=BAND, lw=3.2, zorder=4, solid_capstyle="round")

    ax.plot([cam[0]], [cam[1]], marker="s", ms=9, color=CAMERA, mec=INK, mew=0.8, ls="", zorder=5)
    ax.text(cam[0], cam[1] - 0.42, "camera", ha="center", va="top", fontsize=9, color=INK)
    ax.plot([0.0, 0.0], [0.0, 1.0], color=INK, lw=0.8, zorder=2)
    ax.text(0.08, 0.5, "R", fontsize=9, color=INK, va="center")
    ax.text(-1.08, -0.12, "front\n" + r"$\theta = 0$", ha="right", va="top", fontsize=8.5, color=INK)

    ax.annotate(f"visible arc  ±{np.degrees(th_t):.1f}°",
                xy=(-np.cos(th_t * 0.72), np.sin(th_t * 0.72)), xytext=(0.45, 1.62),
                fontsize=9, color=ARC, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ARC, lw=0.8, shrinkB=2))
    ax.annotate(f"averaging band  ±{np.degrees(theta_r):.1f}°  (r = {r_px:.0f} px)",
                xy=(-np.cos(theta_r * 0.9), np.sin(theta_r * 0.9)), xytext=(-dr * 0.62, 1.62),
                fontsize=9, color=BAND, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BAND, lw=0.8, shrinkB=2))
    ax.annotate("", xy=(0.0, -1.45), xytext=(cam[0], -1.45),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax.text(cam[0] / 2.0, -1.55, f"D = {geom.D_cm:.1f} cm   (D / R = {dr:.1f},  ρ = R / D = {geom.rho:.3f})",
            ha="center", va="top", fontsize=9, color=INK)

    ax.set_aspect("equal")
    ax.set_xlim(cam[0] - 0.9, 1.9)
    ax.set_ylim(-2.15, 2.05)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)


def _draw_side_view(ax, geom: CameraGeometry, rows: list[dict], colors: list, *,
                    title: str | None, label_every: int = 1) -> None:
    """Side view in units of R: depth to the right, height up, camera on the left."""
    dr = geom.D_cm / geom.R_cm
    cam = (-dr, 0.0)
    z = np.array([r["B_cm"] / geom.R_cm for r in rows])
    z_lo, z_hi = float(z.min()) - 0.7, float(z.max()) + 0.7
    ax.add_patch(Rectangle((-1.0, z_lo), 2.0, z_hi - z_lo, facecolor=GLASS, edgecolor=INK, lw=1.1, zorder=1))

    # horizon (camera height) and the tilted optical axis
    ax.plot([cam[0], 2.0], [0.0, 0.0], color=HORIZON, lw=1.0, ls=":", zorder=2)
    L = dr + 1.0
    ax.add_patch(FancyArrowPatch(cam, (cam[0] + L * np.cos(geom.alpha_rad), L * np.sin(geom.alpha_rad)),
                                 arrowstyle="-|>", mutation_scale=10, color=CAMERA, lw=1.2, zorder=3))
    # The dotted horizon continues past the tick labels; its own label sits
    # below the line so that it never collides with the "500 mL"-type label
    # of a graduation lying on (or next to) the horizon.
    ax.plot([2.0, 5.2], [0.0, 0.0], color=HORIZON, lw=1.0, ls=":", zorder=2)
    ax.text(3.0, -0.12, "camera horizon (Δ = 0)", fontsize=8.5, color=HORIZON, va="top", ha="left",
            bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85, pad=1.0), zorder=6)

    for r, c in zip(rows, colors):
        zi = r["B_cm"] / geom.R_cm
        ax.plot([cam[0], -1.0], [cam[1], zi], color=c, lw=0.8, alpha=0.85, zorder=2)
        ax.plot([-1.0], [zi], marker="o", ms=4, color=c, mec=BG, mew=0.5, ls="", zorder=4)
        ax.plot([-1.0, 1.0], [zi, zi], color=c, lw=0.6, alpha=0.6, zorder=2)
    for i, (r, c) in enumerate(zip(rows, colors)):
        if i % label_every == 0:
            ax.text(1.1, r["B_cm"] / geom.R_cm, f"{r['V_ml']:.0f} mL", fontsize=7.5, color=INK,
                    va="center", ha="left", zorder=6,
                    bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85, pad=0.8))

    ax.plot([cam[0]], [cam[1]], marker="s", ms=9, color=CAMERA, mec=INK, mew=0.8, ls="", zorder=5)
    ax.text(cam[0], -0.55, "camera", ha="center", va="top", fontsize=9, color=INK)
    arc_r = 1.6
    a = geom.alpha_rad
    if abs(a) > 1e-4:
        t = np.linspace(0.0, a, 40)
        ax.plot(cam[0] + arc_r * np.cos(t), arc_r * np.sin(t), color=CAMERA, lw=1.0)
    ax.text(cam[0], -1.15, f"pitch α = {geom.alpha_deg:+.2f}°", fontsize=8.5,
            color=CAMERA, va="top", ha="center")
    ax.text(0.0, z_hi + 0.15, "cylinder", ha="center", va="bottom", fontsize=8.5, color=INK)

    ax.set_aspect("equal")
    ax.set_xlim(cam[0] - 0.9, 8.5)
    ax.set_ylim(min(z_lo, -1.0) - 0.4, max(z_hi, 1.0) + 0.7)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)


def _draw_image_plane(ax, geom: CameraGeometry, row: dict, r_px: float, *, title: str | None) -> None:
    """Image-plane view of one projected circle with the band and the Delta bracket."""
    v_click = row["v_click_px"]
    theta_full = np.linspace(-np.pi, np.pi, 1441)
    u_f, v_f = project_graduation(theta_full, v_click, geom)
    th_t = geom.theta_tangent
    front = np.abs(theta_full) <= th_t
    ax.fill(u_f, v_f, color=GLASS, alpha=0.9, lw=0, zorder=1)
    ax.plot(u_f[~front & (theta_full > 0)], v_f[~front & (theta_full > 0)], color=MUTED, lw=0.9, ls=":", zorder=2)
    ax.plot(u_f[~front & (theta_full < 0)], v_f[~front & (theta_full < 0)], color=MUTED, lw=0.9, ls=":", zorder=2)
    ax.plot(u_f[front], v_f[front], color=ARC, lw=2.0, zorder=3)

    cx = geom.cx_px
    ax.axvspan(cx - r_px, cx + r_px, color=BAND_FILL, alpha=0.14, lw=0, zorder=0)
    ax.axvline(cx, color=BAND, lw=0.7, ls="--", zorder=1)
    u, v, m = row["u"], row["v"], row["mask"]
    ax.plot(u[m], v[m], color=BAND, lw=3.4, zorder=4, solid_capstyle="round")
    bends_down = row["v_max"] > v_click          # circle above the horizon: arc bends down in the image
    ax.plot([cx], [v_click], marker="o", ms=5, color=INK, mec=BG, ls="", zorder=5)
    ax.text(cx + 2.0 * r_px, v_click, "clicked row  $v_{click}$", fontsize=8.5, color=INK, va="center", ha="left")

    span = float(v_f.max() - v_f.min())
    pad_near, pad_far = 0.25 * span + 4.0, 1.3 * span + 4.0   # room for the zoom inset on the back side
    ax.set_xlim(u_f.min() - 0.12 * geom.a_px, u_f.max() + 0.12 * geom.a_px)
    if bends_down:
        lo, hi = v_f.min() - pad_near, v_f.max() + pad_far
    else:
        lo, hi = v_f.min() - pad_far, v_f.max() + pad_near
    ax.set_ylim(hi, lo)  # image axis: y down
    ax.set_xlabel("u  (px)", fontsize=9, color=INK)
    ax.set_ylabel("v  (px, down)", fontsize=9, color=INK)
    _style_axes(ax, grid_axis="")

    v_back = float(v_f[np.argmin(np.abs(np.abs(theta_full) - np.pi))])
    ax.text(cx, v_back + (3.0 if bends_down else -3.0), "back half (hidden by the glass)", fontsize=8,
            color=MUTED, ha="center", va="top" if bends_down else "bottom")
    # band labels on the front side (opposite to the zoom inset)
    y_lab = lo + 0.02 * (hi - lo) if bends_down else hi - 0.02 * (hi - lo)
    ax.text(cx - r_px, y_lab, "band  −r", fontsize=8, color=BAND, ha="center", va="top" if bends_down else "bottom")
    ax.text(cx + r_px, y_lab, "+r", fontsize=8, color=BAND, ha="center", va="top" if bends_down else "bottom")

    # zoom inset on the arc inside the band, placed on the back side of the ellipse (empty area)
    ins = ax.inset_axes([0.08, 0.05, 0.5, 0.40] if bends_down else [0.08, 0.55, 0.5, 0.40])
    ins.axvspan(cx - r_px, cx + r_px, color=BAND_FILL, alpha=0.14, lw=0)
    ins.plot(u, v, color=ARC, lw=1.2)
    ins.plot(u[m], v[m], color=BAND, lw=3.0, solid_capstyle="round")
    ins.plot([cx], [v_click], marker="o", ms=4, color=INK, ls="")
    d = row["delta_px"]
    x_br = cx + r_px + 0.12 * r_px
    ins.plot([x_br, x_br], [row["v_min"], row["v_max"]], color=DELTA_EDGE, lw=1.4)
    for yy in (row["v_min"], row["v_max"]):
        ins.plot([x_br - 0.05 * r_px, x_br + 0.05 * r_px], [yy, yy], color=DELTA_EDGE, lw=1.4)
    ins.text(x_br + 0.1 * r_px, (row["v_min"] + row["v_max"]) / 2.0, f"Δ = {d:.1f} px",
             fontsize=9, color=DELTA_EDGE, va="center", ha="left", fontweight="bold")
    m_pad = 0.25 * max(d, 1.0) + 0.5
    ins.set_xlim(cx - r_px * 1.25, cx + r_px * 1.9)
    ins.set_ylim(row["v_max"] + m_pad, row["v_min"] - m_pad)
    ins.set_facecolor(BG)
    ins.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ins.spines.values():
        s.set_color(SPINE)
    ins.set_title("zoom on the band", fontsize=8, color=MUTED, pad=2)
    ax.indicate_inset_zoom(ins, edgecolor=SPINE, lw=0.8)

    if title:
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)


def _draw_frame_panel(ax, frame_bgr: np.ndarray | None, calib: Any, geom: CameraGeometry,
                      rows: list[dict], colors: list, r_px: float) -> None:
    """The video frame (cropped around the cylinder) with the projected arcs and the band."""
    W, H = (int(calib.image_size[0]), int(calib.image_size[1])) if getattr(calib, "image_size", None) else (
        frame_bgr.shape[1], frame_bgr.shape[0])
    x_left, x_right = float(calib.x_left), float(calib.x_right)
    pad = max(70.0, r_px + 50.0)
    x0, x1 = max(0.0, x_left - pad), min(float(W), x_right + pad)
    if frame_bgr is not None:
        ax.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), zorder=0)
    else:
        ax.imshow(np.full((H, W, 3), 235, dtype=np.uint8), zorder=0)

    cx = geom.cx_px
    for x in (x_left, x_right):
        ax.axvline(x, color=INK, lw=0.7, alpha=0.5, ls="--", zorder=1)
    ax.axvspan(cx - r_px, cx + r_px, color=BAND_FILL, alpha=0.2, lw=0, zorder=1)
    ax.axvline(cx, color=BAND, lw=0.7, ls="--", alpha=0.9, zorder=1)
    if 0 <= geom.cy_px <= H:
        ax.axhline(geom.cy_px, color=HORIZON, lw=1.2, ls=":", zorder=2)
        ax.text(x0 + 6, geom.cy_px - 6, "camera horizon", fontsize=8, color=HORIZON, va="bottom",
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.75, pad=1.2))

    for r, c in zip(rows, colors):
        u, v, m = r["u"], r["v"], r["mask"]
        ax.plot(u, v, color=c, lw=1.0, alpha=0.8, zorder=3)
        if m.any():
            ax.plot(u[m], v[m], color=c, lw=2.8, zorder=4, solid_capstyle="round")
        ax.text(cx + r_px + 12, r["v_click_px"], f"{r['V_ml']:.0f} mL", fontsize=8, color=c,
                va="center", ha="left", fontweight="bold",
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.75, pad=1.2), zorder=5)

    gx = getattr(calib, "graduations_px_x", None)
    if gx is not None and len(gx) == len(calib.graduations_px_y):
        ax.plot(gx, calib.graduations_px_y, marker="o", ms=3.5, color=CLICK_FRONT, mec=BG, mew=0.5,
                ls="", zorder=6, label="front clicks")
    by, bx = getattr(calib, "back_px_y", None), getattr(calib, "back_px_x", None)
    if by is not None and bx is not None:
        pts = [(x, y) for x, y in zip(bx, by) if x is not None and y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", ms=3.5, color=CLICK_BACK,
                    mec=BG, mew=0.5, ls="", zorder=6, label="back clicks")
    ax.set_xlim(x0, x1)
    ax.set_ylim(H, 0)
    ax.set_xlabel("x  (px)", fontsize=9, color=INK)
    ax.set_ylabel("y  (px)", fontsize=9, color=INK)
    _style_axes(ax, grid_axis="")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor=BG, edgecolor=SPINE, framealpha=0.9)
    ax.set_title("Projected graduation circles", fontsize=10.5, color=INK, loc="left", pad=6)


def _draw_delta_panel(ax, rows_ml: list[dict], geom: CameraGeometry, r_px: float, H: float) -> None:
    """u_curvature(V) with the Delta wedge, y axis shared with the frame panel."""
    y = np.array([r["v_click_px"] for r in rows_ml])
    V = np.array([r["V_ml"] for r in rows_ml])
    d = np.array([r["delta_ml"] for r in rows_ml])
    u = np.array([r["u_ml"] for r in rows_ml])
    order = np.argsort(y)
    ax.fill_betweenx(y[order], 0.0, d[order], color=DELTA_FILL, alpha=0.55, lw=0,
                     label="Δ  full span of the arc")
    ax.plot(d[order], y[order], color=DELTA_EDGE, lw=1.0, alpha=0.8)
    ax.plot(u[order], y[order], color=U_LINE, lw=2.0, zorder=3)
    ax.plot(u, y, marker="s", ms=6, color=U_LINE, mec=BG, mew=0.8, ls="", zorder=4,
            label=r"$u_{\mathrm{curvature}} = \Delta / (2\sqrt{3})$")
    if 0 <= geom.cy_px <= H:
        lab = "camera horizon"
        if geom.v_horizon_ml is not None:
            lab += f"  (V ≈ {geom.v_horizon_ml:.0f} mL)"
        ax.axhline(geom.cy_px, color=HORIZON, lw=1.2, ls=":", label=lab)
    for yi, Vi, ui in zip(y, V, u):
        ax.text(ui + 0.03 * max(d.max(), 0.1), yi, f"{ui:.2f}", fontsize=7.5, color=U_LINE, va="center", ha="left")
    ax.set_xlim(0.0, max(float(d.max()) * 1.25, 0.2))
    ax.set_xlabel("uncertainty  (mL)", fontsize=9, color=INK)
    ax.set_title(f"Curvature uncertainty  (r = {r_px:.0f} px)", fontsize=10.5, color=INK, loc="left", pad=6)
    _style_axes(ax, grid_axis="x")
    ax.tick_params(labelleft=False)
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=BG, edgecolor=SPINE, framealpha=0.92)


# --------------------------------------------------------------------------- public renderers
def render_curvature_figure(frame_bgr: np.ndarray | None, calib: Any, geom: CameraGeometry, r_px: int,
                            model_fn: Callable[..., Any], out_png: Path | str, *,
                            style: str = "light") -> None:
    """Four-panel curvature figure.

    (1) the frame with the projected circle of every graduation and the
    averaging band, (2) ``u_curvature(V)`` with the Δ wedge and the camera
    horizon (rows shared with panel 1), (3) top view, (4) side view.  Only the
    light style is implemented; ``style`` is kept for API compatibility.
    """
    if style != "light":
        raise ValueError("only style='light' is available")
    rows_geo = _arc_rows(calib, geom, r_px)
    rows_ml = curvature_uncertainty_ml(calib, geom, r_px, model_fn)
    colors = _grad_colors(len(rows_geo))
    H = float(calib.image_size[1]) if getattr(calib, "image_size", None) else float(frame_bgr.shape[0])

    fig = _figure(18.0, 9.6)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.78, 0.62, 1.55], height_ratios=[1.0, 1.0],
                          hspace=0.32, wspace=0.16, left=0.04, right=0.99, top=0.9, bottom=0.07)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_u = fig.add_subplot(gs[:, 1], sharey=ax_img)
    ax_top = fig.add_subplot(gs[0, 2])
    ax_side = fig.add_subplot(gs[1, 2])

    _draw_frame_panel(ax_img, frame_bgr, calib, geom, rows_geo, colors, r_px)
    _draw_delta_panel(ax_u, rows_ml, geom, r_px, H)
    ref = max(rows_geo, key=lambda r: r["delta_px"])
    _draw_top_view(ax_top, geom, r_px, ref["theta_r"], title="Top view  (units of R)")
    _draw_side_view(ax_side, geom, rows_geo, colors, title="Side view  (units of R, height relative to the camera)")

    name = getattr(getattr(calib, "cylinder", None), "name", "cylinder")
    fig.suptitle(f"Cylinder curvature — {name}:  {geom.summary()}", fontsize=12, color=INK, y=0.97)
    _save(fig, out_png)


def render_cylinder_schematic(geom: CameraGeometry, calib: Any, r_px: int, out_png: Path | str) -> None:
    """Standalone schematic: top view, side view and the image-plane ellipse with Δ."""
    rows_geo = _arc_rows(calib, geom, r_px)
    colors = _grad_colors(len(rows_geo))
    ref = max(rows_geo, key=lambda r: r["delta_px"])

    fig = _figure(15.0, 7.2)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0],
                          hspace=0.28, wspace=0.12, left=0.02, right=0.98, top=0.9, bottom=0.08)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_img = fig.add_subplot(gs[:, 1])
    _draw_top_view(ax_top, geom, r_px, ref["theta_r"], title="(a) Top view — the band samples an arc of the circle")
    _draw_side_view(ax_side, geom, rows_geo, colors, title="(b) Side view — the circle is seen from above or below the horizon",
                    label_every=1 if len(rows_geo) <= 12 else 2)
    _draw_image_plane(ax_img, geom, ref, r_px,
                      title=f"(c) Image plane — graduation {ref['V_ml']:.0f} mL projected as an ellipse")
    fig.suptitle("Why a horizontal graduation circle has a vertical extent in the image — "
                 f"{geom.summary()}", fontsize=11.5, color=INK, y=0.97)
    _save(fig, out_png)
