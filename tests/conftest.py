"""Shared pytest fixtures.

Everything here is headless and torch-free: the suite must run on a CI worker
with no display, no GPU and only the core dependencies installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = REPO_ROOT / "demo_data"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator, so failures are reproducible."""
    return np.random.default_rng(20240617)


@pytest.fixture
def square_grid_points() -> np.ndarray:
    """A 4x4 grid of well-spread control points inside a 100x100 image."""
    xs, ys = np.meshgrid(np.linspace(10, 90, 4), np.linspace(10, 90, 4))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


@pytest.fixture
def checkerboard() -> np.ndarray:
    """A 64x64 checkerboard, useful as a warpable image with sharp features."""
    image = np.zeros((64, 64), dtype=np.uint8)
    image[::16, :] = 255
    image[:, ::16] = 255
    image[8:24, 8:24] = 200
    image[40:56, 40:56] = 120
    return image


@pytest.fixture
def demo_data_dir() -> Path:
    """Path to the bundled demo data, skipping the test when it is absent."""
    if not DEMO_DATA.is_dir():
        pytest.skip("demo_data/ is not present in this checkout")
    return DEMO_DATA


class FakeView:
    """A recording stand-in for the Tk view.

    The presenter talks to its view purely through ``on_*`` callbacks, so a
    plain recorder is enough to drive it end to end without a display.
    """

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.calls: list[str] = []
        self.previews: list[tuple] = []
        self.matched_points: list[tuple] = []
        self.requested_points: list[str] = []

    def on_data_loaded(self) -> None:
        self.calls.append("data_loaded")

    def on_points_changed(self) -> None:
        self.calls.append("points_changed")

    def on_display_update_needed(self) -> None:
        self.calls.append("display_update")

    def on_error(self, message: str) -> None:
        self.calls.append("error")
        self.errors.append(message)

    def on_project_loaded(self) -> None:
        self.calls.append("project_loaded")

    def on_project_reset(self) -> None:
        self.calls.append("project_reset")

    def on_request_corresponding_point(self, target: str) -> None:
        self.calls.append("request_point")
        self.requested_points.append(target)

    def on_show_preview_2d(self, warped, reference) -> None:
        self.calls.append("preview_2d")
        self.previews.append((warped, reference))

    def on_show_preview_3d(self, warped, reference) -> None:
        self.calls.append("preview_3d")
        self.previews.append((warped, reference))

    def on_show_matched_points(self, src_img, dst_img, src_points, dst_points) -> None:
        self.calls.append("matched_points")
        self.matched_points.append((src_img, dst_img, src_points, dst_points))


@pytest.fixture
def fake_view() -> FakeView:
    """A recording view instance."""
    return FakeView()


@pytest.fixture
def presenter(fake_view):
    """An ApplicationPresenter wired to a recording view."""
    from tpsreg.presenter import ApplicationPresenter

    p = ApplicationPresenter()
    p.set_view(fake_view)
    return p
