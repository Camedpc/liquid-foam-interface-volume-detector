"""Absorbance package: reference / dead pixels, Beer-Lambert map and profile, foam top, mass balance, figures."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cylvision.absorbance import (
    AbsorbanceParams,
    AbsorbanceRun,
    Reference,
    absorbance_map,
    absorbance_profile,
    analyse_crop,
    dead_pixel_masks,
    fill_dead_pixels,
    finalize_mass_balance,
    foam_integral_cm,
    foam_top_from_profile,
    foam_tops_for_ks,
    liquid_content_profile,
    mass_balance_gamma,
    plot_heatmap_A,
    plot_heatmap_c,
    plot_measurement_figure,
    prepare_reference,
    px_per_cm_from_calibration,
    stack_reference,
    volume_ticks,
)

H, W = 120, 60
FOAM_TOP, LIQUID = 30, 80          # foam rows FOAM_TOP .. LIQUID-1, liquid below
I_REF, I_FOAM, I_LIQ = 220.0, 110.0, 200.0


def synthetic_crop(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """BGR crop of a back-lit cylinder: bright air, darker foam band, bright liquid."""
    rng = np.random.default_rng(seed)
    g = np.full((H, W), I_REF, dtype=np.float32)
    g[FOAM_TOP:LIQUID] = I_FOAM
    g[LIQUID:] = I_LIQ
    if noise > 0:
        g += rng.normal(0.0, noise, size=g.shape).astype(np.float32)
    img = np.clip(g, 0, 255).astype(np.uint8)
    return np.dstack([img // 2, img, img // 3])  # green screen: G carries the signal


def synthetic_reference(dark_marks: bool = True) -> Reference:
    frames = []
    for i in range(4):
        g = np.full((H, W), I_REF + (i - 1.5), dtype=np.float32)  # small frame-to-frame flicker
        img = np.clip(g, 0, 255).astype(np.uint8)
        bgr = np.dstack([img // 2, img, img // 3])
        if dark_marks:
            bgr[50, 10:20, :] = 40           # a printed graduation line
            bgr[90:92, 40, :] = 30           # a speck on the glass
        frames.append(bgr)
    return stack_reference(frames, frame_start=5)


def linear_model(y):
    """A calibration-like model: 0.5 mL per px, V = 600 at y = 0."""
    return 600.0 - 0.5 * np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# Reference, dead pixels
# ---------------------------------------------------------------------------

def test_stack_reference_mean_std_and_meta():
    ref = synthetic_reference(dark_marks=False)
    assert ref.mean_bgr.shape == (H, W, 3) and ref.std_bgr.shape == (H, W, 3)
    assert ref.frame_start == 5 and ref.frame_count == 4
    g = ref.intensity("G")
    assert g.dtype == np.float32
    assert abs(float(g.mean()) - I_REF) < 1.0   # the flicker averages out around I_REF
    assert float(ref.std_bgr[:, :, 1].mean()) > 0.5  # and is visible in the std


def test_dead_pixel_masks_threshold_and_dilation():
    ref = synthetic_reference()
    direct, dilated = dead_pixel_masks(ref.intensity("G"), dead_thresh=178, dead_dilate=2)
    assert direct[50, 15] and not direct[50, 5] and not direct[48, 15]
    assert dilated[48, 15] and dilated[52, 15] and dilated[50, 8]   # grown by 2 px
    assert not dilated[47, 15] and not dilated[50, 7]               # but not by 3
    assert dilated.sum() > direct.sum()
    d0, d0d = dead_pixel_masks(ref.intensity("G"), 178, 0)
    assert np.array_equal(d0, d0d)


def test_fill_dead_pixels_keeps_valid_and_fills_from_neighbours():
    img = np.full((H, W), 100.0, dtype=np.float32)
    img[20:40, :] = 150.0
    valid = np.ones((H, W), dtype=bool)
    img[30, 30] = 0.0
    valid[30, 30] = False
    img[3, 3] = 255.0
    valid[3, 3] = False
    out = fill_dead_pixels(img, valid, r_fill=3)
    assert out.dtype == np.float32
    assert np.array_equal(out[valid], img[valid])          # valid pixels untouched
    assert out[30, 30] == pytest.approx(150.0)              # local mean of its (valid) neighbours
    assert out[3, 3] == pytest.approx(100.0)
    assert np.array_equal(fill_dead_pixels(img, np.ones_like(valid), 3), img)


# ---------------------------------------------------------------------------
# Beer-Lambert map and profile
# ---------------------------------------------------------------------------

def test_absorbance_map_recovers_minus_log_ratio():
    I0 = np.full((H, W), I_REF, dtype=np.float32)
    I = np.full((H, W), I_REF, dtype=np.float32)
    I[FOAM_TOP:LIQUID] = I_FOAM
    A = absorbance_map(I, I0)
    assert A.dtype == np.float32
    assert A[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert A[50, 10] == pytest.approx(-math.log(I_FOAM / I_REF), abs=1e-5)
    # scalar reference, and I clamped to 1 (no ln 0)
    A_s = absorbance_map(np.zeros((2, 2), np.float32), 100.0)
    assert np.allclose(A_s, math.log(100.0))
    with pytest.raises(ValueError):
        absorbance_map(I, I0[:10])


def test_absorbance_profile_averages_the_band_only():
    A = np.zeros((H, W), dtype=np.float32)
    A[:, 20:41] = 1.0            # band |x - 30| <= 10
    A[:, 0:5] = 100.0            # dark glass wall outside the band
    A_z = absorbance_profile(A, cx=30, r_dens=10)
    assert A_z.shape == (H,) and np.allclose(A_z, 1.0)
    assert np.allclose(absorbance_profile(A, cx=30, r_dens=15), 21.0 / 31.0)


# ---------------------------------------------------------------------------
# Foam top by threshold
# ---------------------------------------------------------------------------

def test_foam_top_threshold_finds_the_band_top():
    A_z = np.zeros(H, dtype=np.float32)
    A_z[FOAM_TOP:LIQUID] = 1.0
    A_z[45] = 0.5                        # a dip inside the foam that stays above T: must not stop the scan
    y, A_max, T = foam_top_from_profile(A_z, y_liquid=float(LIQUID), k=5.0)
    assert A_max == pytest.approx(1.0) and T == pytest.approx(0.2)
    assert y == FOAM_TOP - 1               # first row above the foam with A < T
    assert foam_top_from_profile(A_z, None, 5.0)[0] is None
    assert foam_top_from_profile(np.ones(H, np.float32), float(LIQUID), 5.0)[0] is None   # never drops below T
    assert foam_top_from_profile(A_z, float(LIQUID), 5.0, y_top=FOAM_TOP + 5)[0] is None  # region all >= T
    A_z[45] = 0.05                       # a hole below T stops the bottom-up scan: it IS an exit of the foam
    assert foam_top_from_profile(A_z, float(LIQUID), 5.0)[0] == 45


def test_foam_top_moves_with_k_on_a_smooth_edge():
    z = np.arange(H, dtype=float)
    A_z = (1.0 / (1.0 + np.exp((FOAM_TOP - z) / 3.0))).astype(np.float32)   # smooth rise at FOAM_TOP
    A_z[LIQUID:] = 0.1
    tops = [foam_top_from_profile(A_z, float(LIQUID), k)[0] for k in (2, 5, 10)]
    assert tops[0] > tops[1] > tops[2]     # a lower threshold climbs higher into the diffuse edge
    assert abs(tops[0] - FOAM_TOP) <= 1    # k = 2 cuts the sigmoid at its midpoint


# ---------------------------------------------------------------------------
# Mass balance
# ---------------------------------------------------------------------------

def test_gamma_and_liquid_content_consistent_with_a_known_integral():
    A_z = np.zeros(H, dtype=np.float32)
    A_z[FOAM_TOP:LIQUID] = 1.0
    px_per_cm, S = 20.0, 10.0
    integ = foam_integral_cm(A_z, FOAM_TOP, float(LIQUID), px_per_cm)
    assert integ == pytest.approx((LIQUID - FOAM_TOP) / px_per_cm)           # 2.5 cm
    gamma = mass_balance_gamma(V_inf_ml=110.0, V_beer_ml=100.0, integral_cm=integ, section_cm2=S)
    assert gamma == pytest.approx(10.0 / (S * integ))
    c = liquid_content_profile(A_z, gamma, FOAM_TOP, float(LIQUID))
    assert np.isnan(c[:FOAM_TOP]).all() and np.isnan(c[LIQUID:]).all()
    assert np.allclose(c[FOAM_TOP:LIQUID], gamma)
    # the liquid content integrates back to the missing beer volume
    assert float(np.nansum(c)) / px_per_cm * S == pytest.approx(10.0)
    assert math.isnan(mass_balance_gamma(float("nan"), 100.0, integ, S))
    assert math.isnan(mass_balance_gamma(110.0, 100.0, 0.0, S))
    assert mass_balance_gamma(90.0, 100.0, integ, S) < 0                      # overshoot flagged, not hidden


# ---------------------------------------------------------------------------
# End to end on a synthetic crop
# ---------------------------------------------------------------------------

def test_analyse_crop_end_to_end():
    ref = synthetic_reference()
    params = AbsorbanceParams(channel="G", light_blur=1.0, r_dens=15, foam_k=5.0, dead_thresh=178,
                              dead_dilate=2, px_per_cm=20.0, section_cm2=10.0, cx=30)
    prep = prepare_reference(ref, params)
    assert prep.n_dead > 0 and prep.I0.shape == (H, W)
    assert prep.I0[50, 15] == pytest.approx(I_REF, abs=2.0)       # the graduation was filled
    crop = synthetic_crop(noise=1.5)
    res = analyse_crop(crop, prep, params, y_liquid=float(LIQUID), V_beer_ml=100.0, V_inf_ml=110.0)
    assert res.A.shape == (H, W) and res.A_z.shape == (H,)
    assert res.A_z[55] == pytest.approx(-math.log(I_FOAM / I_REF), abs=0.03)
    assert res.A_z[100] == pytest.approx(-math.log(I_LIQ / I_REF), abs=0.03)
    assert res.A_z[10] == pytest.approx(0.0, abs=0.03)
    assert res.found_foam_top and abs(res.y_foam_top - (FOAM_TOP - 1)) <= 2
    assert res.integral_cm > 0 and np.isfinite(res.gamma) and res.gamma > 0
    assert np.isnan(res.c_z[5]) and np.isfinite(res.c_z[55])
    d = res.to_dict()
    assert d["y_foam_top"] == res.y_foam_top and d["threshold"] == pytest.approx(res.A_max / 5.0)
    with pytest.raises(ValueError):
        analyse_crop(crop[:50], prep, params, float(LIQUID))


def test_params_round_trip():
    p = AbsorbanceParams(channel="gray", light_blur=2.0, r_dens=10, foam_k=4.0, cx=7)
    q = AbsorbanceParams.from_dict({**p.to_dict(), "unknown_key": 1})
    assert q == p and q.r_fill == 5
    assert AbsorbanceParams(dead_dilate=6).r_fill == 9
    assert p.resolved_cx(W) == 7 and AbsorbanceParams().resolved_cx(W) == W // 2


# ---------------------------------------------------------------------------
# Batch helpers and figures
# ---------------------------------------------------------------------------

def _synthetic_run(n: int = 6) -> AbsorbanceRun:
    A = np.zeros((n, H), dtype=np.float32)
    y_liq = np.linspace(LIQUID, LIQUID + 20, n)
    y_top = np.full(n, float(FOAM_TOP))
    for i in range(n):
        A[i, FOAM_TOP:int(round(y_liq[i]))] = 1.0 - 0.1 * i
    V_beer = linear_model(y_liq)
    return AbsorbanceRun(frame_idx=np.arange(n), t_sec=np.arange(n) * 2.0, A=A, y_liquid=y_liq,
                         y_foam_top=y_top, A_max=A.max(axis=1), threshold=A.max(axis=1) / 5,
                         V_beer=V_beer, V_foam=linear_model(y_top) - V_beer,
                         integral_cm=np.array([foam_integral_cm(A[i], FOAM_TOP, y_liq[i], 20.0) for i in range(n)]),
                         z=np.arange(H, dtype=np.int32))


def test_finalize_mass_balance_auto_v_inf_and_k_sweep():
    run = _synthetic_run()
    finalize_mass_balance(run, None, section_cm2=10.0)
    assert run.V_inf_ml == pytest.approx(float(np.max(run.V_beer)))
    assert run.gamma.shape == (run.n,) and run.c.shape == run.A.shape
    assert run.gamma[np.argmax(run.V_beer)] == pytest.approx(0.0)         # nothing missing on that frame
    i = 0
    expected = (run.V_inf_ml - run.V_beer[i]) / (10.0 * run.integral_cm[i])
    assert run.gamma[i] == pytest.approx(expected)
    assert np.isnan(run.c[i, :FOAM_TOP]).all() and run.c[i, 50] == pytest.approx(expected * 1.0)
    rows = run.rows()
    assert len(rows) == run.n and rows[0]["gamma"] == pytest.approx(expected)
    tops = foam_tops_for_ks(run, ks=(2, 5))
    assert set(tops) == {2, 5} and np.allclose(tops[5], FOAM_TOP - 1)
    npz = run.to_npz_dict()
    assert npz["A"].dtype == np.float32 and npz["V_inf_ml"] == pytest.approx(run.V_inf_ml)


def test_px_per_cm_from_calibration_matches_the_slope():
    from pathlib import Path

    from cylvision.calibration.store import Calibration
    calib = Calibration.from_json(Path(__file__).parent / "data" / "legacy_calib.json")
    y = np.asarray(calib.graduations_px_y, dtype=float)
    V = np.asarray(calib.graduations_ml, dtype=float)
    slope = abs(V[np.argmin(y)] - V[np.argmax(y)]) / np.ptp(y)          # mL / px
    assert px_per_cm_from_calibration(calib, 10.0) == pytest.approx(10.0 / slope)
    # a cylinder of section S: one cm of height holds S mL, i.e. px_per_cm * slope mL
    assert px_per_cm_from_calibration(calib, 10.0) * slope == pytest.approx(10.0)


def test_volume_ticks_are_monotone_inside_the_window():
    y_ticks, labels = volume_ticks(linear_model, 0, H - 1, step_ml=10.0)
    assert labels[0] == "550" and labels[-1] == "600"     # V(119) = 540.5 -> first multiple 550
    assert np.all(np.diff(y_ticks) < 0)                   # volume grows upwards: rows decrease


def test_figures_are_written(tmp_path):
    run = _synthetic_run()
    finalize_mass_balance(run, None, section_cm2=10.0)
    tops = foam_tops_for_ks(run, ks=(2, 5, 10))
    p1 = plot_heatmap_A(run.A, run.t_sec, model_fn=linear_model, y_range=(0, H), out_png=tmp_path / "A.png",
                        y_liquid=run.y_liquid, foam_tops=tops, k_main=5, title="A", V_inf_ml=run.V_inf_ml)
    p2 = plot_heatmap_c(run.c, run.t_sec, model_fn=linear_model, y_range=(0, H), out_png=tmp_path / "c.png",
                        y_liquid=run.y_liquid, y_foam_top=run.y_foam_top, title="c")
    ref = synthetic_reference()
    params = AbsorbanceParams(r_dens=15, cx=30, section_cm2=10.0)
    prep = prepare_reference(ref, params)
    res = analyse_crop(synthetic_crop(), prep, params, float(LIQUID), V_beer_ml=100.0, V_inf_ml=110.0)
    p3 = plot_measurement_figure(prep.raw, res.intensity, res.A, res.A_z, y_liquid=res.y_liquid,
                                 y_foam_top=res.y_foam_top, A_max=res.A_max, threshold=res.threshold,
                                 band=(15, 45), model_fn=linear_model, y_range=(0, H),
                                 out_png=tmp_path / "m.png", k=5.0, title="m")
    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 5_000
