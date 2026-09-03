"""Tests of the uncertainty package on synthetic geometry (no video, no display)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cylvision.uncertainty import (CameraGeometry, UncertaintyBudget, build_budget, curvature_delta_px,
                                   curvature_uncertainty_fn, curvature_uncertainty_ml, fit_focal_free,
                                   geometry_from_calibration, method_uncertainty, project_graduation,
                                   summarize_method_uncertainty)

SQRT12 = 2.0 * np.sqrt(3.0)

# Synthetic camera: rho = 0.1 exactly, a_px = 160 px, no tilt.
R_CM, D_CM, A_PX, CY, CX = 3.2, 32.0, 160.0, 960.0, 540.0
RHO = R_CM / D_CM
F_PX = A_PX * np.sqrt(1.0 - RHO ** 2) / RHO
GRADS = [float(v) for v in range(100, 1001, 100)]


def _rows_for(V, *, back: bool = False) -> np.ndarray:
    """Exact pinhole rows of the graduations (front or back), horizon at 550 mL."""
    V = np.asarray(V, dtype=float)
    B = (V - 550.0) / (np.pi * R_CM ** 2)          # height above the camera (cm)
    depth = D_CM + R_CM if back else D_CM - R_CM
    return CY - F_PX * B / depth


def make_calib(*, with_back: bool = False, mobius_c: float = 0.0) -> SimpleNamespace:
    y = _rows_for(GRADS)
    back = list(_rows_for(GRADS, back=True)) if with_back else None
    if with_back:
        back[0] = None  # one missing back click, as in a real file
    return SimpleNamespace(
        cylinder=SimpleNamespace(name="1000 mL", capacity_ml=1000.0, graduations_ml=GRADS, radius_cm=R_CM),
        x_left=CX - A_PX, x_right=CX + A_PX, y_top=0, y_bottom=1919,
        graduations_ml=GRADS, graduations_px_y=[float(v) for v in y],
        graduations_px_x=[CX] * len(GRADS), mobius_popt=[-0.5, 1000.0, mobius_c],
        back_px_y=back, back_px_x=[CX] * len(GRADS) if with_back else None,
        image_size=(1080, 1920), residuals_ml={"pchip": [0.0] * len(GRADS)}, recommended_model="pchip",
    )


def linear_model_fn(y):
    """Exact inverse of the synthetic front rows: y -> V (mL)."""
    y = np.asarray(y, dtype=float)
    return 550.0 + (CY - y) * (D_CM - R_CM) * np.pi * R_CM ** 2 / F_PX


def make_geom(alpha_rad: float = 0.0) -> CameraGeometry:
    return CameraGeometry(R_cm=R_CM, a_px=A_PX, rho=RHO, D_cm=D_CM, f_px=F_PX, cy_px=CY,
                          alpha_rad=alpha_rad, cx_px=CX, source="no_tilt")


# --------------------------------------------------------------------------- projection
def test_project_graduation_reproduces_click_at_front():
    geom = make_geom(alpha_rad=np.radians(1.5))
    for v_click in (300.0, 960.0, 1700.0):
        u, v = project_graduation(np.array([0.0]), v_click, geom)
        assert u[0] == pytest.approx(CX)
        assert v[0] == pytest.approx(v_click)


def test_silhouette_half_width_matches_a_px():
    geom = make_geom()
    u, _ = project_graduation(np.array([geom.theta_tangent]), CY, geom)
    assert u[0] - CX == pytest.approx(A_PX, rel=1e-6)


def test_delta_is_zero_on_the_horizon_and_grows_away_from_it():
    geom = make_geom()
    assert curvature_delta_px(CY, geom, 64) == 0.0
    deltas = [curvature_delta_px(CY + q, geom, 64) for q in (50, 100, 200, 400, 700)]
    assert all(d > 0 for d in deltas)
    assert all(b > a for a, b in zip(deltas, deltas[1:]))
    deltas_up = [curvature_delta_px(CY - q, geom, 64) for q in (50, 100, 200, 400, 700)]
    assert all(b > a for a, b in zip(deltas_up, deltas_up[1:]))


def test_delta_is_symmetric_without_tilt_and_not_with_tilt():
    geom = make_geom()
    for q in (150.0, 600.0):
        assert curvature_delta_px(CY + q, geom, 64) == pytest.approx(curvature_delta_px(CY - q, geom, 64), rel=1e-9)
    tilted = make_geom(alpha_rad=np.radians(3.0))
    assert curvature_delta_px(CY + 600.0, tilted, 64) != pytest.approx(curvature_delta_px(CY - 600.0, tilted, 64), rel=1e-3)


def test_delta_small_angle_closed_form():
    """Delta ~ |q| (r/f)^2 (1 - rho) / (2 rho) for a narrow band and no tilt."""
    geom = make_geom()
    q, r = 500.0, 30
    approx = q * (r / F_PX) ** 2 * (1.0 - RHO) / (2.0 * RHO)
    assert curvature_delta_px(CY + q, geom, r) == pytest.approx(approx, rel=0.05)


def test_delta_grows_with_band_width():
    geom = make_geom()
    d = [curvature_delta_px(CY + 500.0, geom, r) for r in (20, 40, 80)]
    assert d[0] < d[1] < d[2]
    assert d[1] / d[0] == pytest.approx(4.0, rel=0.05)   # quadratic in r


def test_uncertainty_is_delta_over_two_sqrt_three():
    calib = make_calib()
    geom = make_geom()
    rows = curvature_uncertainty_ml(calib, geom, 64, linear_model_fn)
    assert len(rows) == len(GRADS)
    for r in rows:
        assert r["u_ml"] == pytest.approx(r["delta_ml"] / SQRT12)
        assert r["delta_ml"] == pytest.approx(r["delta_px"] * r["slope_ml_per_px"])
    fn = curvature_uncertainty_fn(calib, geom, 64, linear_model_fn)
    assert fn(100.0) == pytest.approx(rows[0]["u_ml"])
    assert fn(550.0) < fn(100.0)


# --------------------------------------------------------------------------- inversions
def test_geometry_from_calibration_no_tilt_recovers_rho_and_f():
    geom = geometry_from_calibration(make_calib(), use_back_clicks=False)
    assert geom.source in ("no_tilt", "mobius_tilt")
    assert geom.rho == pytest.approx(RHO, rel=0.05)
    assert geom.f_px == pytest.approx(F_PX, rel=0.05)
    assert geom.D_cm == pytest.approx(D_CM, rel=0.05)
    assert geom.alpha_rad == pytest.approx(0.0, abs=1e-6)
    assert geom.cx_px == pytest.approx(CX)


def test_geometry_from_calibration_focal_free_recovers_rho_and_horizon():
    calib = make_calib(with_back=True)
    geom = geometry_from_calibration(calib)
    assert geom.source == "focal_free"
    assert geom.rho == pytest.approx(RHO, rel=1e-6)
    assert geom.cy_px == pytest.approx(CY, abs=1e-6)
    assert geom.v_horizon_ml == pytest.approx(550.0, abs=1e-6)
    assert geom.f_px == pytest.approx(F_PX, rel=1e-6)
    # explicit pitch override
    assert geometry_from_calibration(calib, alpha_deg=2.0).alpha_rad == pytest.approx(np.radians(2.0))


def test_fit_focal_free_needs_two_pairs():
    assert fit_focal_free([100, 200, 300], [1, 2, 3], [None, None, 3.0]) is None


def test_alpha_from_mobius_is_used_without_back_clicks():
    geom = geometry_from_calibration(make_calib(mobius_c=-8.0e-6), use_back_clicks=False)
    assert geom.source == "mobius_tilt"
    assert geom.alpha_rad != 0.0
    assert abs(geom.alpha_deg) < 5.0


def test_missing_radius_raises():
    calib = make_calib()
    calib.cylinder.radius_cm = None
    with pytest.raises(ValueError):
        geometry_from_calibration(calib)


# --------------------------------------------------------------------------- empirical
def test_method_uncertainty_range_and_outlier_guard():
    rows = np.array([100, 101, 102, 103, 150, 99], dtype=np.int32)
    valid = np.array([True, True, True, True, True, False])

    def model_fn(y):
        return 2000.0 - np.asarray(y, dtype=float)      # 1 mL per px

    out = method_uncertainty(rows, valid, 101.5, model_fn, tol_ml=7.0)
    assert out["n_kept"] == 4                # 150 is beyond ±7 mL, 99 is not valid
    assert out["y_up"] == 100.0 and out["y_lo"] == 103.0
    assert out["range_ml"] == pytest.approx(3.0)
    assert out["u_ml"] == pytest.approx(3.0 / SQRT12)
    assert out["V_up"] == pytest.approx(1900.0) and out["V_lo"] == pytest.approx(1897.0)

    none = method_uncertainty(rows, np.zeros_like(valid), 101.5, model_fn)
    assert none["n_kept"] == 0 and np.isnan(none["u_ml"])
    assert method_uncertainty(rows, valid, None, model_fn)["n_kept"] == 0


def test_summarize_method_uncertainty():
    rows = [{"u_method_lower_mL": v} for v in (1.0, 2.0, float("nan"), 3.0)] + [{"u_method_lower_mL": None}]
    s = summarize_method_uncertainty(rows)
    assert s["n"] == 3 and s["median"] == 2.0 and s["max"] == 3.0
    assert summarize_method_uncertainty([])["n"] == 0


# --------------------------------------------------------------------------- budget
def test_budget_total_is_quadrature_sum():
    b = UncertaintyBudget(u_calibration_ml=3.0, u_pixel_ml_fn=lambda V: 0.5, u_curvature_ml_fn=lambda V: 1.2,
                          u_method_ml=2.0)
    assert b.total(500.0) == pytest.approx(np.sqrt(9.0 + 0.25 + 1.44 + 4.0))
    row = b.table([500.0])[0]
    assert row["u_total_ml"] == pytest.approx(b.total(500.0))


def test_build_budget_on_synthetic_calibration():
    calib = make_calib(with_back=True)
    geom = geometry_from_calibration(calib)
    params = SimpleNamespace(r_lower=64)
    rows = [{"u_method_lower_mL": v} for v in (0.8, 1.0, 1.4)]
    b = build_budget(calib, geom, params, linear_model_fn, rows)
    assert b.u_calibration_ml == pytest.approx(10.0 / np.sqrt(12.0))     # 10 mL thin step floor
    assert b.u_method_ml == pytest.approx(1.0) and b.u_method_available
    slope = (D_CM - R_CM) * np.pi * R_CM ** 2 / F_PX
    assert b.u_pixel_ml_fn(500.0) == pytest.approx(slope / np.sqrt(12.0), rel=1e-3)
    assert b.u_curvature_ml_fn(550.0) < b.u_curvature_ml_fn(100.0)
    assert b.total(100.0) > b.u_calibration_ml

    placeholder = build_budget(calib, geom, params, linear_model_fn, None, scale_resolution_ml=1.0)
    assert placeholder.u_method_ml == 0.0 and not placeholder.u_method_available
    assert placeholder.u_calibration_ml == pytest.approx(1.0 / np.sqrt(12.0))


def test_pixel_term_uses_the_calibration_slope_definition():
    """The budget and the curvature term must share the calibration package's |dV/dy|."""
    import cylvision.uncertainty as unc
    from cylvision.calibration.models import slope_ml_per_px as ref

    assert unc.slope_ml_per_px is ref
    assert unc.curvature.slope_ml_per_px is ref
    calib = make_calib(with_back=True)
    b = build_budget(calib, geometry_from_calibration(calib), SimpleNamespace(r_lower=64), linear_model_fn, None)
    y = float(np.asarray(calib.graduations_px_y)[4])
    assert b.u_pixel_ml_fn(500.0) == pytest.approx(ref(linear_model_fn, y) / np.sqrt(12.0), rel=1e-6)
