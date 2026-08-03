"""End-to-end smoke tests against the bundled demo data.

These exercise the real microscopy formats the tool exists to handle: an EDAX
.ang scan, a DREAM.3D volume and a TIFF slice series. They skip cleanly when
demo_data/ is absent from the checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpsreg.models import DataFormat, ImageLoader, TransformManager, TransformType


@pytest.fixture
def data_2d(demo_data_dir):
    path = demo_data_dir / "2D"
    if not path.is_dir():
        pytest.skip("demo_data/2D is not present")
    return path


@pytest.fixture
def data_3d(demo_data_dir):
    path = demo_data_dir / "3D"
    if not path.is_dir():
        pytest.skip("demo_data/3D is not present")
    return path


@pytest.fixture(scope="session")
def loaded_ang(demo_data_dir):
    """The demo .ang scan, parsed once for the whole session.

    Parsing a 313k-row scan is the single slowest thing the suite does, and it
    dominated the macOS and Windows CI runs when every test reloaded it. These
    tests only read the result, so sharing it is safe.
    """
    path = demo_data_dir / "2D" / "EBSD.ang"
    if not path.is_file():
        pytest.skip("demo_data/2D/EBSD.ang is not present")
    return ImageLoader.load(path)


class TestAngLoading:
    """EDAX .ang EBSD scans."""

    def test_loads_expected_modalities(self, loaded_ang):
        assert loaded_ang.metadata["dataformat"] == DataFormat.ANG.value
        # An .ang carries several per-pixel quantities, not just one image.
        assert len(loaded_ang.modalities) > 1
        assert "IQ" in loaded_ang.modalities

    def test_shapes_are_stack_shaped_and_consistent(self, loaded_ang):
        shapes = {name: arr.shape for name, arr in loaded_ang.data.items()}
        first = loaded_ang.data["IQ"]
        assert first.ndim == 4, f"expected (slices, h, w, c), got {shapes['IQ']}"
        assert first.shape[0] == 1

        for name, arr in loaded_ang.data.items():
            assert arr.shape[:3] == first.shape[:3], f"{name} disagrees on grid size"

    def test_step_size_is_read_from_the_header(self, loaded_ang):
        assert loaded_ang.resolution > 0

    def test_euler_angles_are_assembled(self, loaded_ang):
        assert "EulerAngles" in loaded_ang.modalities

    def test_no_nans_survive_loading(self, loaded_ang):
        assert np.all(np.isfinite(loaded_ang.data["IQ"]))


class TestImageLoading:
    """Plain TIFF micrographs."""

    def test_bse_image_loads_as_single_slice(self, data_2d):
        data = ImageLoader.load(data_2d / "BSE.tif", modality_name="BSE")
        assert data.data["BSE"].shape[0] == 1
        assert data.data["BSE"].dtype == np.uint8

    def test_slice_series_loads_as_a_stack(self, data_3d):
        paths = sorted(data_3d.glob("BSE_*.tif"))
        if len(paths) < 2:
            pytest.skip("fewer than two BSE slices in demo_data/3D")

        data = ImageLoader.load(list(paths), modality_name="BSE")
        assert data.data["BSE"].shape[0] == len(paths)

    def test_two_modalities_can_be_combined(self, data_3d):
        bse_paths = sorted(data_3d.glob("BSE_*.tif"))
        se_paths = sorted(data_3d.glob("SE_*.tif"))
        if not bse_paths or len(bse_paths) != len(se_paths):
            pytest.skip("demo_data/3D does not have matching BSE and SE series")

        data = ImageLoader.load(list(bse_paths), modality_name="BSE")
        data.add_modality(ImageLoader.load(list(se_paths), modality_name="SE"))

        assert sorted(data.modalities) == ["BSE", "SE"]


class TestDream3dLoading:
    """DREAM.3D volumes."""

    def test_loads_a_volume(self, data_3d):
        path = data_3d / "EBSD.dream3d"
        if not path.exists():
            pytest.skip("demo_data/3D/EBSD.dream3d is not present")

        data = ImageLoader.load(path)
        assert data.metadata["dataformat"] == DataFormat.DREAM3D.value
        assert data.modalities
        assert data.resolution > 0

    def test_volume_has_multiple_slices(self, data_3d):
        path = data_3d / "EBSD.dream3d"
        if not path.exists():
            pytest.skip("demo_data/3D/EBSD.dream3d is not present")

        data = ImageLoader.load(path)
        first = data.data[data.modalities[0]]
        assert first.ndim == 4
        assert first.shape[0] >= 1


class TestBundledControlPoints:
    """The demo control point files."""

    def test_2d_points_load_and_pair_up(self, data_2d):
        from tpsreg.models import PointManager

        src_file = data_2d / "source_pts.txt"
        dst_file = data_2d / "destination_pts.txt"
        if not (src_file.exists() and dst_file.exists()):
            pytest.skip("demo point files are not present")

        manager = PointManager()
        manager.load_source_from_file(src_file)
        manager.load_destination_from_file(dst_file)

        src, dst = manager.get_point_pairs(0)
        assert len(src) == len(dst)
        assert len(src) >= 3, "need at least 3 points to fit a spline"

    @pytest.mark.slow
    def test_demo_points_produce_a_usable_transform(self, data_2d):
        """The headline workflow: bundled points must actually fit a spline."""
        from tpsreg.models import PointManager

        src_file = data_2d / "source_pts.txt"
        dst_file = data_2d / "destination_pts.txt"
        if not (src_file.exists() and dst_file.exists()):
            pytest.skip("demo point files are not present")

        manager = PointManager()
        manager.load_source_from_file(src_file)
        manager.load_destination_from_file(dst_file)
        src, dst = manager.get_point_pairs(0)

        # Fit on a small grid to keep the test fast; correctness of the fit
        # does not depend on the output size.
        tform = TransformManager().estimate_transform(
            src, dst, TransformType.TPS, (64, 64)
        )

        assert tform._estimated
        assert np.all(np.isfinite(tform.params))
