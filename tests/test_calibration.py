"""Tests for cylvision.calibration (synthetic data, no display)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cylvision.calibration.check import predicted_graduation_rows, render_verification_figure
from cylvision.calibration.clicks import (
    PROMPT_KIND_GRAD,
    build_click_prompts,
    clicks_from_points,
    count_placed_graduations,
    render_click_overlay,
)
from cylvision.calibration.back_clicks import (
    apply_back_clicks,
    back_fit_preview,
    build_back_prompts,
    render_back_overlay,
)
from cylvision.calibration.models import (
    best_model,
    build_model_fn,
    evaluate_residuals,
    fit_calibration_models,
    invert_model,
    slope_ml_per_px,
)
from cylvision.calibration.store import (
    PRESETS,
    Calibration,
    CylinderSpec,
    fit_calibration,
    load_legacy_run,
)

DATA = Path(__file__).parent / "data"
LEGACY_JSON = DATA / "legacy_calib.json"


# ---------------------------------------------------------------------------
# Synthetic tilted pinhole projection
# ---------------------------------------------------------------------------

def tilted_pinhole_rows(V_ml: np.ndarray, *, f_px: float = 1500.0, D_cm: float = 60.0,
                        alpha_deg: float = 6.0, cm_per_ml: float = 0.03,
                        cy_px: float = 960.0, z0_cm: float = -15.0) -> np.ndarray:
    """Row of the graduation ``V`` seen by a camera pitched by ``alpha``.

    The mark sits at height ``z = z0 + cm_per_ml * V`` on the axis, at
    distance ``D`` from the camera; the camera is rotated by ``alpha`` about
    its horizontal axis. ``y`` grows downward, so larger ``V`` -> smaller row.
    """
    a = math.radians(alpha_deg)
    z = z0_cm + cm_per_ml * np.asarray(V_ml, dtype=float)
    y_cam = z * math.cos(a) - D_cm * math.sin(a)
    z_cam = z * math.sin(a) + D_cm * math.cos(a)
    return cy_px - f_px * y_cam / z_cam


@pytest.fixture
def synthetic_clicks() -> tuple[np.ndarray, np.ndarray]:
    V = np.array(PRESETS["1000"].graduations_ml, dtype=float)
    y = tilted_pinhole_rows(V)
    return y, V


@pytest.fixture
def synthetic_calibration(synthetic_clicks) -> Calibration:
    y, V = synthetic_clicks
    clicks = {
        "x_left": 360, "x_right": 680, "y_top": 40, "y_bottom": 1880,
        "V_ml": V, "y_px": y, "x_px": [520] * len(V), "skipped_ml": [],
    }
    return fit_calibration(clicks, PRESETS["1000"], "synthetic.mp4", (1080, 1920))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def test_pchip_exact_at_clicks(synthetic_clicks):
    y, V = synthetic_clicks
    models = fit_calibration_models(y, V)
    np.testing.assert_allclose(models["pchip"]["fn"](y), V, atol=1e-9)
    # order independence
    perm = np.random.default_rng(0).permutation(len(y))
    models2 = fit_calibration_models(y[perm], V[perm])
    np.testing.assert_allclose(models2["pchip"]["fn"](y), V, atol=1e-9)


def test_poly2_and_mobius_residuals_small_on_tilted_projection(synthetic_clicks):
    y, V = synthetic_clicks
    models = fit_calibration_models(y, V)
    res = evaluate_residuals(models, y, V)
    assert np.max(np.abs(res["poly2"])) < 0.5
    assert np.max(np.abs(res["mobius"])) < 0.5
    # the projection is exactly Möbius: residual at numerical noise level
    assert np.max(np.abs(res["mobius"])) < 1e-3
    # and it generalises between clicks
    V_mid = np.arange(150.0, 1000.0, 100.0)
    y_mid = tilted_pinhole_rows(V_mid)
    assert np.max(np.abs(models["mobius"]["fn"](y_mid) - V_mid)) < 1e-2
    assert np.max(np.abs(models["poly2"]["fn"](y_mid) - V_mid)) < 0.5


def test_best_model_heuristic():
    res_good = {"pchip": np.zeros(4), "poly2": np.full(4, 0.1), "mobius": np.full(4, 0.1)}
    res_bad = {"pchip": np.zeros(4), "poly2": np.full(4, 0.8), "mobius": np.full(4, 0.8)}
    assert best_model(res_good) == "poly2"
    assert best_model(res_bad) == "pchip"
    assert best_model(res_bad, rms_threshold_ml=1.0) == "poly2"


def test_fit_requires_three_points():
    with pytest.raises(ValueError):
        fit_calibration_models([100.0, 200.0], [10.0, 5.0])


@pytest.mark.parametrize("name", ["pchip", "poly2", "mobius"])
def test_build_model_fn_each_model(synthetic_calibration, name):
    calib = synthetic_calibration
    calib.recommended_model = name
    fn = build_model_fn(calib)
    y = np.asarray(calib.graduations_px_y)
    V = np.asarray(calib.graduations_ml)
    tol = 1e-9 if name == "pchip" else 0.5
    np.testing.assert_allclose(fn(y), V, atol=tol)
    # scalar and array inputs both work
    assert float(np.asarray(fn(y[0]))) == pytest.approx(V[0], abs=tol)
    # explicit override
    fn2 = build_model_fn(calib, model=name)
    np.testing.assert_allclose(fn2(y), fn(y))


def test_build_model_fn_accepts_legacy_dict():
    d = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
    fn = build_model_fn(d)
    np.testing.assert_allclose(fn(d["graduations_px_y"]), d["graduations_ml"], atol=1e-9)


def test_slope_ml_per_px(synthetic_calibration):
    fn = build_model_fn(synthetic_calibration, model="mobius")
    y = np.asarray(synthetic_calibration.graduations_px_y)
    y_mid = float(y.mean())
    s = slope_ml_per_px(fn, y_mid)
    assert s > 0                                        # magnitude, never signed
    # about 100 mL over the pixel span between two neighbouring clicks
    expected = 100.0 / abs(float(y[5] - y[4]))
    assert s == pytest.approx(expected, rel=0.05)
    # the raw derivative is negative (volume goes up, y goes down)
    assert float(fn(y_mid + 1.0) - fn(y_mid - 1.0)) < 0
    assert slope_ml_per_px(fn, y_mid, h=0.5) == pytest.approx(s, rel=1e-3)


def test_invert_model_round_trip(synthetic_calibration):
    fn = build_model_fn(synthetic_calibration, model="mobius")
    V = np.array([150.0, 425.0, 990.0])
    y = invert_model(fn, 0, 1920, V)
    np.testing.assert_allclose(fn(y), V, atol=0.05)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_presets():
    assert set(PRESETS) == {"50", "1000"}
    assert PRESETS["50"].radius_cm == pytest.approx(1.35)
    assert PRESETS["1000"].radius_cm == pytest.approx(3.20)
    assert PRESETS["1000"].graduations_ml == [float(v) for v in range(100, 1001, 100)]
    assert PRESETS["50"].graduations_ml == [5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert PRESETS["1000"].thin_step_ml == 10 and PRESETS["1000"].medium_step_ml == 50
    assert PRESETS["50"].thin_step_ml == 1 and PRESETS["50"].medium_step_ml == 5
    assert PRESETS["1000"].key == "1000"
    spec = CylinderSpec.from_dict(PRESETS["50"].to_dict())
    assert spec == PRESETS["50"]


def test_fit_calibration_from_clicks(synthetic_calibration, synthetic_clicks):
    y, V = synthetic_clicks
    calib = synthetic_calibration
    assert calib.crop() == (360, 680, 40, 1880)
    assert calib.cx() == (680 - 360) // 2
    assert calib.crop_width() == 321
    assert calib.image_size == (1080, 1920)
    assert calib.cylinder.name == "1000 mL"
    assert calib.graduations_ml == V.tolist()
    assert calib.graduations_px_y == y.tolist()
    assert len(calib.poly2_coef) == 3 and len(calib.mobius_popt) == 3
    assert np.asarray(calib.mobius_pcov).shape == (3, 3)
    assert set(calib.residuals_ml) == {"pchip", "poly2", "mobius"}
    assert calib.recommended_model in ("pchip", "poly2")
    assert calib.back_px_y is None and not calib.has_back_clicks()


def test_fit_calibration_reorders_edges_and_rows():
    V = np.array([100.0, 200.0, 300.0, 400.0])
    y = tilted_pinhole_rows(V)
    clicks = {"x_left": 600, "x_right": 400, "y_top": 1800, "y_bottom": 30,
              "V_ml": V, "y_px": y, "x_px": [500] * 4}
    calib = fit_calibration(clicks, PRESETS["1000"], "x", (1080, 1920))
    assert calib.crop() == (400, 600, 30, 1800)
    with pytest.raises(ValueError):
        fit_calibration({**clicks, "V_ml": V[:2], "y_px": y[:2], "x_px": [0, 0]},
                        PRESETS["1000"], "x", (1080, 1920))


def test_json_round_trip(tmp_path, synthetic_calibration):
    calib = synthetic_calibration.with_back_clicks(
        [None, None, 1100.0, 990.0, 850.0, 700.0, 560.0, 410.0, 260.0, 110.0],
        [None, None, 520.0, 518.0, 516.0, 515.0, 514.0, 513.0, 512.0, 510.0], 1234)
    path = tmp_path / "calib.json"
    calib.to_json(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    # legacy aliases are written too
    assert d["roi_x"] == [360, 680]
    assert d["eprouvette"] == "1000"
    assert d["graduations_px_y_back"][0] is None and d["back_calibration_frame"] == 1234
    back = Calibration.from_json(path)
    assert back == calib
    assert back.has_back_clicks()
    fn_a = build_model_fn(calib)
    fn_b = build_model_fn(back)
    yq = np.linspace(100, 1800, 50)
    np.testing.assert_allclose(fn_a(yq), fn_b(yq))


def test_legacy_json_load_without_crop():
    calib = Calibration.from_json(LEGACY_JSON)     # no crop.json next to it
    assert calib.x_left == 361 and calib.x_right == 675
    assert calib.y_top == 0 and calib.y_bottom == 1919
    assert calib.image_size == (1080, 1920)
    assert calib.cylinder.name == "1000 mL" and calib.cylinder.radius_cm == pytest.approx(3.2)
    assert calib.recommended_model == "pchip"
    assert calib.graduations_ml == [float(v) for v in range(100, 1001, 100)]
    assert calib.graduations_px_y[0] == 1686.0
    assert calib.skipped_ml == []
    assert calib.back_px_y is not None and calib.back_px_y[:3] == [None, None, None]
    assert calib.back_px_y[3] == 1124.0 and calib.back_px_x[3] == 520.0
    assert calib.back_frame_idx == 33606
    assert calib.has_back_clicks()
    fn = build_model_fn(calib)
    np.testing.assert_allclose(fn(calib.graduations_px_y), calib.graduations_ml, atol=1e-9)
    assert build_model_fn(calib, model="poly2")(1686.0) == pytest.approx(100.0, abs=1.0)


def test_legacy_run_with_crop_json(tmp_path):
    crop = {"video_ref": "x", "x_left": 361, "x_right": 675, "y_top": 5,
            "y_bottom": 1869, "eprouvette": "1000"}
    (tmp_path / "crop.json").write_text(json.dumps(crop), encoding="utf-8")
    (tmp_path / "calib.json").write_text(LEGACY_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    calib = load_legacy_run(tmp_path / "calib.json", tmp_path / "crop.json")
    assert calib.crop() == (361, 675, 5, 1869)
    # from_json picks the sibling crop.json automatically
    auto = Calibration.from_json(tmp_path / "calib.json")
    assert auto.crop() == (361, 675, 5, 1869)
    # re-saved in the new layout it reloads identically
    auto.to_json(tmp_path / "new.json")
    assert Calibration.from_json(tmp_path / "new.json") == auto


def test_back_fit_preview_on_legacy():
    calib = Calibration.from_json(LEGACY_JSON)
    prev = back_fit_preview(calib)
    assert prev is not None
    assert prev["slope_front"] < 0 and prev["slope_back"] < 0
    assert abs(prev["slope_back"]) < abs(prev["slope_front"])     # far wall = smaller
    assert 0 < prev["rho"] < 1
    assert 500 < prev["V_horizon"] < 3000


# ---------------------------------------------------------------------------
# clicks (pure parts)
# ---------------------------------------------------------------------------

def test_clicks_from_points():
    prompts = build_click_prompts([100.0, 200.0, 300.0, 400.0])
    assert len(prompts) == 8 and prompts[4][0] == PROMPT_KIND_GRAD
    pts = [(650, 10), (350, 20), (30, 1850), (40, 12), (500, 1700), None, (505, 1300), (510, 1100)]
    assert count_placed_graduations(prompts, pts) == 3
    clicks = clicks_from_points(prompts, pts)
    assert (clicks["x_left"], clicks["x_right"]) == (350, 650)
    assert (clicks["y_top"], clicks["y_bottom"]) == (12, 1850)
    np.testing.assert_array_equal(clicks["V_ml"], [100.0, 300.0, 400.0])
    np.testing.assert_array_equal(clicks["y_px"], [1700.0, 1300.0, 1100.0])
    assert clicks["x_px"] == [500, 505, 510]
    assert clicks["skipped_ml"] == [200.0]
    with pytest.raises(ValueError):
        clicks_from_points(prompts, pts[:5])


def test_render_overlays_without_display():
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    prompts = build_click_prompts([10.0, 20.0, 30.0])
    out = render_click_overlay(img, prompts, [(20, 5), (180, 5), (0, 30)], (100, 150),
                               shift_held=True, with_magnifier=False)
    assert out.shape == img.shape and out.any()
    calib = Calibration.from_json(LEGACY_JSON)
    bp = build_back_prompts(calib)
    assert len(bp) == 10 and bp[0][3] == 1686
    big = np.zeros((1920, 1080, 3), dtype=np.uint8)
    out2 = render_back_overlay(big, bp, [(500, 1600), None], (10, 10), with_magnifier=False)
    assert out2.any()
    with pytest.raises(ValueError):
        apply_back_clicks(calib, [None] * 10, 0)
    updated = apply_back_clicks(calib, [(500, 1600), (500, 1420)] + [None] * 8, 7)
    assert updated.back_frame_idx == 7 and updated.back_px_y[1] == 1420.0


# ---------------------------------------------------------------------------
# check figure
# ---------------------------------------------------------------------------

def test_predicted_rows_and_verification_figure(tmp_path, synthetic_calibration):
    calib = synthetic_calibration
    y_medium, y_thin = predicted_graduation_rows(calib)
    assert 150.0 in y_medium and 110.0 in y_thin and 100.0 not in y_medium
    fn = build_model_fn(calib)
    assert fn(y_medium[150.0]) == pytest.approx(150.0, abs=0.1)
    img = np.full((1920, 1080, 3), 40, dtype=np.uint8)
    out = tmp_path / "check.png"
    render_verification_figure(img, calib, out)
    assert out.exists() and out.stat().st_size > 10_000
