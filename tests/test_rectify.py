"""Tests of the cylinder rectification on synthetic geometry (no video, no display)."""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cylvision.detection import DetectionParams
from cylvision.rectify import (compare_frame, crop_maps, forward_rows, median_edges, method_uncertainty_max,
                               rectification_maps, rectify_calibration, rectify_frame, residual_delta_px,
                               rolling_max, summarize_compare, wall_edges)
from cylvision.uncertainty import CameraGeometry, curvature_delta_px, project_graduation

# Synthetic camera: rho = 0.1, a_px = 160 px, horizon at row 960 (same as test_uncertainty).
R_CM, D_CM, A_PX, CY, CX = 3.2, 32.0, 160.0, 960.0, 540.0
RHO = R_CM / D_CM
F_PX = A_PX * np.sqrt(1.0 - RHO ** 2) / RHO
W, H = 1080, 1920


def make_geom(rho: float = RHO, cy: float = CY) -> CameraGeometry:
    return CameraGeometry(R_cm=R_CM, a_px=A_PX, rho=rho, D_cm=R_CM / rho, f_px=F_PX, cy_px=cy,
                          alpha_rad=0.0, cx_px=CX, source="focal_free")


def arc_image(v_click: float, geom: CameraGeometry) -> np.ndarray:
    """White arc of the projected circle ``v_click`` on a black frame (front half only)."""
    img = np.zeros((H, W), dtype=np.uint8)
    theta = np.linspace(-geom.theta_tangent, geom.theta_tangent, 4001)
    u, v = project_graduation(theta, v_click, geom)
    pts = np.round(np.stack([u, v], axis=1)).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, 255, 1)
    return img


def row_of_max(img: np.ndarray, x: int) -> float:
    col = img[:, x].astype(float)
    rows = np.nonzero(col > 127)[0]
    return float(rows.mean()) if rows.size else float("nan")


# --------------------------------------------------------------------------- maps
def test_maps_shape_dtype_and_identity_outside_the_cylinder():
    geom = make_geom()
    map_x, map_y = rectification_maps(geom, W, H)
    assert map_x.shape == (H, W) and map_y.shape == (H, W)
    assert map_x.dtype == np.float32 and map_y.dtype == np.float32
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    outside = np.abs(xs - CX) > A_PX
    assert np.array_equal(map_x[outside], xs[outside])
    assert np.array_equal(map_y[outside], ys[outside])
    # the axis column and the horizon row are fixed points
    assert map_x[:, int(CX)] == pytest.approx(CX, abs=1e-4)
    assert map_y[:, int(CX)] == pytest.approx(np.arange(H), abs=1e-3)
    assert map_y[int(CY), :] == pytest.approx(CY, abs=1e-3)


def test_maps_reject_bad_geometry():
    with pytest.raises(ValueError):
        rectification_maps(make_geom(rho=1.0), W, H)
    with pytest.raises(ValueError):
        rectification_maps(make_geom(), W, H, a_px=0.0)


@pytest.mark.parametrize("v_click", [300.0, 700.0, 1500.0, 1800.0])
def test_projected_arc_becomes_a_horizontal_row(v_click):
    geom = make_geom()
    img = arc_image(v_click, geom)
    maps = rectification_maps(geom, W, H)
    # the arc bends by several pixels inside a 120 px band before rectification ...
    before = [row_of_max(img, x) for x in range(int(CX) - 120, int(CX) + 121, 10)]
    assert max(before) - min(before) >= 2.0 or abs(v_click - CY) < 100
    assert curvature_delta_px(v_click, geom, 120) > 1.0
    # ... and is flat at v_click after (within 1 px), over 95 % of the silhouette
    out = rectify_frame(img, maps, interpolation=cv2.INTER_NEAREST)
    xs = range(int(CX - 0.95 * A_PX), int(CX + 0.95 * A_PX) + 1, 5)
    after = np.array([row_of_max(out, x) for x in xs])
    assert np.all(np.isfinite(after))
    assert np.abs(after - v_click).max() <= 1.0
    assert forward_rows(v_click, geom, np.array(list(xs))) == pytest.approx(v_click)


def test_forward_rows_is_nan_outside():
    fr = forward_rows(500.0, make_geom(), np.array([CX - A_PX - 5, CX, CX + A_PX + 5]))
    assert np.isnan(fr[0]) and fr[1] == 500.0 and np.isnan(fr[2])


def test_crop_maps_give_the_rectified_crop_directly():
    geom = make_geom()
    frame = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    maps = rectification_maps(geom, W, H)
    xl, xr = int(CX - A_PX) - 10, int(CX + A_PX) + 10
    full = rectify_frame(frame, maps)[:, xl:xr + 1]
    direct = rectify_frame(frame, crop_maps(maps, xl, xr))
    assert direct.shape == full.shape == (H, xr - xl + 1, 3)
    assert np.array_equal(direct, full)


def test_residual_delta_vanishes_with_the_right_geometry_and_not_otherwise():
    geom = make_geom()
    assert residual_delta_px(1700.0, geom, geom, 64) == pytest.approx(0.0, abs=1e-6)
    wrong_rho = residual_delta_px(1700.0, geom, make_geom(rho=RHO + 0.02), 64)
    wrong_cy = residual_delta_px(1700.0, geom, make_geom(cy=CY + 80.0), 64)
    assert 0.0 < wrong_rho < curvature_delta_px(1700.0, geom, 64)
    assert 0.0 < wrong_cy < curvature_delta_px(1700.0, geom, 64)


# --------------------------------------------------------------------------- calibration
def test_rectify_calibration_keeps_front_rows_and_drops_back_clicks():
    calib = SimpleNamespace(x_left=380, x_right=700, y_top=0, y_bottom=1919, graduations_ml=[100.0, 200.0],
                            graduations_px_y=[1700.0, 1500.0], graduations_px_x=[540.0, 540.0],
                            back_px_y=[1690.0, 1495.0], back_px_x=[540.0, 540.0], back_frame_idx=3, extra={})
    rc = rectify_calibration(calib)
    assert rc.graduations_px_y == calib.graduations_px_y
    assert rc.x_left == calib.x_left and rc.x_right == calib.x_right
    assert rc.back_px_y is None and rc.back_px_x is None and rc.back_frame_idx is None
    assert rc.extra["rectified"] is True
    assert calib.back_px_y is not None  # the input is untouched


# --------------------------------------------------------------------------- wall edges
def test_wall_edges_find_a_bright_cylinder_on_a_dark_background():
    frame = np.full((400, 600, 3), 20, dtype=np.uint8)
    frame[:, 180:421] = 200
    e = wall_edges(frame, (50, 350), (175, 425))
    assert e is not None
    assert e[0] == pytest.approx(180.0, abs=3.0) and e[1] == pytest.approx(421.0, abs=3.0)
    assert wall_edges(frame, (50, 60), (175, 425)) is None          # band too thin
    assert wall_edges(frame, (50, 350), (175, 300)) is None         # implausible width for the hint
    assert median_edges([None, (180.0, 420.0), (182.0, 422.0)]) == (181.0, 421.0)
    assert median_edges([None]) is None


# --------------------------------------------------------------------------- comparison
def linear_model_fn(y):
    """y -> V (mL), 0.5 mL per px, increasing upwards."""
    return 1000.0 - 0.5 * np.asarray(y, dtype=float)


def two_band_frame(v_click: float, geom: CameraGeometry) -> np.ndarray:
    """Bright liquid below the projected circle ``v_click``, dark foam above (lower_brighter)."""
    theta = np.linspace(-geom.theta_tangent, geom.theta_tangent, 2001)
    u, v = project_graduation(theta, v_click, geom)
    img = np.full((H, W), 60, dtype=np.uint8)
    for x in range(int(CX - A_PX), int(CX + A_PX) + 1):
        y_edge = int(round(float(np.interp(x, u, v))))
        img[y_edge:, x] = 220
    img[:, :int(CX - A_PX)] = 220
    img[:, int(CX + A_PX) + 1:] = 220
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_compare_frame_reduces_the_spread_of_a_curved_interface():
    geom = make_geom()
    v_click = 1700.0
    frame = two_band_frame(v_click, geom)
    calib = SimpleNamespace(x_left=int(CX - A_PX) - 20, x_right=int(CX + A_PX) + 20)
    params = DetectionParams(polarity="lower_brighter", T_lower=20.0, T_upper=20.0, r_lower=120, r_upper=120,
                             blur_sigma=2.0, blur_h=0, min_h_upper=0, y_top=0, y_bottom=H - 1,
                             cx=int(CX) - calib.x_left)
    maps = crop_maps(rectification_maps(geom, W, H), calib.x_left, calib.x_right)
    row, res_o, res_r, crop_o, crop_r = compare_frame(frame, calib, params, maps, linear_model_fn)
    assert crop_o.shape == crop_r.shape
    assert res_o.found_lower and res_r.found_lower
    assert row["range_orig_mL"] > 3 * row["range_rect_mL"]
    assert row["u_rect_mL"] < row["u_orig_mL"]
    assert row["umax_orig_mL"] >= row["u_orig_mL"]
    assert row["y_rect_px"] == pytest.approx(v_click, abs=1.5)


def test_reference_definition_and_summary_helpers():
    mu = {"V_mean": 100.0, "V_up": 102.0, "V_lo": 99.0}
    assert method_uncertainty_max(mu) == pytest.approx(2.0 / np.sqrt(3.0))
    assert np.isnan(method_uncertainty_max({"V_mean": 1.0, "V_up": float("nan"), "V_lo": 1.0}))
    assert rolling_max([1.0, 3.0, np.nan, 2.0], 2).tolist()[:3] == [1.0, 3.0, 3.0]
    rows = [{"u_orig_mL": 1.0, "u_rect_mL": 0.5, "umax_orig_mL": 1.4, "umax_rect_mL": 0.7} for _ in range(4)]
    s = summarize_compare(rows, fps=2.0)
    assert s["u"]["ratio_median"] == pytest.approx(0.5) and s["umax"]["ratio_mean"] == pytest.approx(0.5)
    assert s["u"]["rolling_max_1s"]["orig_mean"] == pytest.approx(1.0)
