"""IO tests: unicode paths, screen fitting, magnifier (no video needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cylvision import io
from cylvision.magnifier import MAGNIFIED_SIZE, draw_crosshair, make_magnifier, overlay_magnifier


def test_imread_imwrite_unicode_roundtrip(tmp_path: Path) -> None:
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[5:10, 8:12] = (10, 200, 30)
    path = tmp_path / "ünïcödé pâth ø" / "frame_001.png"
    assert io.imwrite_unicode(path, img)
    assert path.exists()
    back = io.imread_unicode(path)
    assert back is not None and back.shape == img.shape
    np.testing.assert_array_equal(back, img)
    gray = io.imread_unicode(path, flags=0)
    assert gray is not None and gray.ndim == 2


def test_imread_missing_or_empty(tmp_path: Path) -> None:
    assert io.imread_unicode(tmp_path / "nope.png") is None
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert io.imread_unicode(empty) is None
    assert io.imread_unicode(tmp_path / "garbage.png") is None


def test_imwrite_bad_extension(tmp_path: Path) -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert not io.imwrite_unicode(tmp_path / "x.notanimage", img)


def test_is_video_path() -> None:
    assert io.is_video_path("run.mp4") and io.is_video_path(Path("a/b.MOV"))
    assert not io.is_video_path("frame.png")


def test_open_video_missing(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        io.open_video(tmp_path / "missing.mp4")


def test_fit_to_screen() -> None:
    frame = np.zeros((1000, 500, 3), dtype=np.uint8)
    same, s = io.fit_to_screen(frame, 2000, 2000)
    assert s == 1.0 and same is frame
    small, s = io.fit_to_screen(frame, 400, 400)
    assert s == pytest.approx(0.4)
    assert small.shape[:2] == (400, 200)


def test_fit_to_window_without_window() -> None:
    canvas = np.zeros((10, 10, 3), dtype=np.uint8)
    assert io.fit_to_window(canvas, "no-such-window-xyz") is canvas


def test_get_screen_size_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYLVISION_HEADLESS", "1")
    assert io.get_screen_size((123, 45)) == (123, 45)


def test_magnifier_shapes_and_overlay() -> None:
    img = np.random.default_rng(0).integers(0, 255, size=(800, 900, 3), dtype=np.uint8)
    mag = make_magnifier(img, 600, 500)
    assert mag.shape == (MAGNIFIED_SIZE, MAGNIFIED_SIZE, 3)
    # Corner: padding must not break the size.
    assert make_magnifier(img, 0, 0).shape == (MAGNIFIED_SIZE, MAGNIFIED_SIZE, 3)
    assert make_magnifier(img, 899, 799).shape == (MAGNIFIED_SIZE, MAGNIFIED_SIZE, 3)
    # Cursor far from the top-left corner: loupe pasted at (12, 12).
    disp = img.copy()
    overlay_magnifier(disp, mag, (600, 500))
    assert np.array_equal(disp[12:12 + MAGNIFIED_SIZE, 12:12 + MAGNIFIED_SIZE], mag)
    # Cursor under the loupe: it jumps to the top-right corner.
    disp = img.copy()
    overlay_magnifier(disp, mag, (50, 50))
    x0 = 900 - MAGNIFIED_SIZE - 12
    assert np.array_equal(disp[12:12 + MAGNIFIED_SIZE, x0:x0 + MAGNIFIED_SIZE], mag)
    # Display smaller than the loupe: cropped paste, no crash.
    tiny = np.zeros((100, 80, 3), dtype=np.uint8)
    overlay_magnifier(tiny, mag, (50, 50))
    draw_crosshair(tiny, 40, 50)
    assert tiny[50, 40].tolist() == [0, 0, 255]
