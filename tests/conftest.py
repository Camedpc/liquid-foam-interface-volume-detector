"""Pytest configuration: the suite never opens a window.

``CYLVISION_HEADLESS`` makes ``cylvision.io.get_screen_size`` return its
default instead of asking Tk, and ``MPLBACKEND=Agg`` keeps matplotlib
window-free even if a test ends up importing ``pyplot``. Both are set only
when the caller did not choose a value.
"""
from __future__ import annotations

import os

os.environ.setdefault("CYLVISION_HEADLESS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
