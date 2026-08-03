"""Tests for the MatchAnything/ROMA matcher wrapper.

The model itself needs the optional ``matchanything`` extra and a multi-hundred
megabyte checkpoint, neither of which CI has. What this module tests is the code
*around* the model, which is where the bugs live: device selection, image
preprocessing, RANSAC wiring, the order results come back in, and the errors
raised when the extra is missing.

Tests that only need our own logic run everywhere by substituting a minimal
stand-in for torch. Tests marked ``torch`` use the real library and skip when it
is absent. Nothing here loads a checkpoint.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tpsreg import roma_matcher

# --------------------------------------------------------------------------
# Stand-in torch
# --------------------------------------------------------------------------


class _FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMps:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _NullContext:
    """Stands in for torch.no_grad() and torch.autocast()."""

    def __init__(self, recorder=None, label=""):
        self._recorder = recorder
        self._label = label

    def __enter__(self):
        if self._recorder is not None:
            self._recorder.append(self._label)
        return self

    def __exit__(self, *exc_info):
        return False


def make_fake_torch(cuda: bool = False, mps: bool = False, recorder=None):
    """Build a torch stand-in exposing only what roma_matcher touches."""
    torch = types.ModuleType("torch")
    torch.cuda = _FakeCuda(cuda)

    backends = types.ModuleType("torch.backends")
    if mps is not None:
        backends.mps = _FakeMps(mps)
    torch.backends = backends

    # from_numpy returns the array itself so assertions can inspect the
    # preprocessing result directly.
    torch.from_numpy = lambda array: _FakeTensor(array)
    torch.no_grad = lambda: _NullContext(recorder, "no_grad")
    torch.autocast = lambda **kwargs: _NullContext(
        recorder, f"autocast:{kwargs.get('device_type')}"
    )
    return torch


class _FakeTensor:
    """Minimal tensor: tracks unsqueeze/to and unwraps back to numpy."""

    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array)
        self.device = None
        self.batched = False

    def unsqueeze(self, dim: int) -> _FakeTensor:
        out = _FakeTensor(np.expand_dims(self.array, dim))
        out.batched = True
        return out

    def to(self, device: str) -> _FakeTensor:
        self.device = device
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a torch stand-in for the duration of one test."""

    def _install(cuda: bool = False, mps: bool = False, recorder=None):
        torch = make_fake_torch(cuda=cuda, mps=mps, recorder=recorder)
        monkeypatch.setattr(roma_matcher, "_import_torch", lambda: torch)
        return torch

    return _install


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestModulePaths:
    """Paths are resolved against the package, not the working directory."""

    def test_vendor_root_is_inside_the_package(self):
        assert roma_matcher.MATCHANYTHING_ROOT.is_dir()
        assert roma_matcher.MATCHANYTHING_ROOT.parent.name == "tpsreg"

    def test_default_config_exists(self):
        """The yacs config is loaded by path, so it must really be there."""
        assert roma_matcher.DEFAULT_CONFIG_PATH.is_file()

    def test_default_checkpoint_is_under_the_package(self):
        # The weights are downloaded by the user and are not shipped, so the
        # file will not exist; only its location is fixed.
        assert roma_matcher.DEFAULT_CHECKPOINT_PATH.name.endswith(".ckpt")
        assert (
            roma_matcher.MATCHANYTHING_ROOT
            in roma_matcher.DEFAULT_CHECKPOINT_PATH.parents
        )

    def test_paths_do_not_depend_on_cwd(self, tmp_path, monkeypatch):
        """Config resolution used to be relative to the working directory."""
        monkeypatch.chdir(tmp_path)
        import importlib

        reloaded = importlib.reload(roma_matcher)
        try:
            assert reloaded.DEFAULT_CONFIG_PATH.is_file()
        finally:
            importlib.reload(roma_matcher)


class TestMissingDependencies:
    """Absent optional dependencies produce an actionable error."""

    def test_message_names_the_extra(self):
        assert 'pip install "tpsreg[matchanything]"' in (
            roma_matcher._MISSING_DEPS_MESSAGE
        )

    def test_import_torch_raises_the_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else (__builtins__.__import__)
        )

        def no_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", no_torch)

        with pytest.raises(ImportError, match=r"tpsreg\[matchanything\]"):
            roma_matcher._import_torch()

    def test_get_config_raises_the_hint_without_lightning(self, monkeypatch):
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else (__builtins__.__import__)
        )

        def no_lightning(name, *args, **kwargs):
            if name.startswith("pytorch_lightning"):
                raise ImportError("No module named 'pytorch_lightning'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", no_lightning)

        with pytest.raises(ImportError, match=r"tpsreg\[matchanything\]"):
            roma_matcher.get_config()

    def test_create_matcher_raises_the_hint_without_the_model(self, monkeypatch):
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else (__builtins__.__import__)
        )

        def no_model(name, *args, **kwargs):
            if "lightning_loftr" in name:
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", no_model)

        with pytest.raises(ImportError, match=r"tpsreg\[matchanything\]"):
            roma_matcher.create_matcher()


class TestSelectDevice:
    """Device autodetection."""

    def test_prefers_cuda_when_present(self, fake_torch):
        fake_torch(cuda=True, mps=True)
        assert roma_matcher.select_device() == "cuda"

    def test_falls_back_to_mps(self, fake_torch):
        fake_torch(cuda=False, mps=True)
        assert roma_matcher.select_device() == "mps"

    def test_falls_back_to_cpu(self, fake_torch):
        fake_torch(cuda=False, mps=False)
        assert roma_matcher.select_device() == "cpu"

    def test_explicit_choice_is_honoured(self, fake_torch):
        fake_torch(cuda=True, mps=True)
        assert roma_matcher.select_device("mps") == "mps"
        assert roma_matcher.select_device("cpu") == "cpu"

    def test_unavailable_choice_falls_back_with_a_warning(self, fake_torch, caplog):
        """Asking for CUDA on a CPU box must not crash the detection."""
        fake_torch(cuda=False, mps=False)

        with caplog.at_level("WARNING"):
            assert roma_matcher.select_device("cuda") == "cpu"

        assert any("unavailable" in record.message for record in caplog.records)

    def test_cpu_is_always_available(self, fake_torch):
        fake_torch(cuda=False, mps=False)
        assert roma_matcher.select_device("cpu") == "cpu"

    def test_missing_mps_backend_is_not_an_error(self, monkeypatch):
        """Older torch builds have no torch.backends.mps at all."""
        torch = make_fake_torch(cuda=False)
        del torch.backends.mps
        monkeypatch.setattr(roma_matcher, "_import_torch", lambda: torch)

        assert roma_matcher.select_device() == "cpu"


class TestPrepareImage:
    """Image preprocessing before the model sees it."""

    def _prepared(self, image):
        return roma_matcher._prepare_image(image).array

    def test_grayscale_becomes_three_channels(self, fake_torch):
        fake_torch()
        out = self._prepared(np.full((8, 10), 128, dtype=np.uint8))
        # Channels-first, three of them.
        assert out.shape == (3, 8, 10)

    def test_channels_are_identical_for_grayscale(self, fake_torch):
        fake_torch()
        rng = np.random.default_rng(0)
        out = self._prepared((rng.random((6, 6)) * 255).astype(np.uint8))
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_array_equal(out[1], out[2])

    def test_single_channel_image_is_expanded(self, fake_torch):
        fake_torch()
        out = self._prepared(np.full((8, 10, 1), 200, dtype=np.uint8))
        assert out.shape == (3, 8, 10)

    def test_rgb_is_left_alone(self, fake_torch):
        fake_torch()
        out = self._prepared(np.full((8, 10, 3), 90, dtype=np.uint8))
        assert out.shape == (3, 8, 10)

    def test_extra_channels_are_dropped(self, fake_torch):
        """RGBA input would otherwise reach the model as four channels."""
        fake_torch()
        image = np.zeros((8, 10, 4), dtype=np.uint8)
        image[..., 3] = 255  # alpha
        assert self._prepared(image).shape == (3, 8, 10)

    def test_values_are_scaled_into_the_unit_interval(self, fake_torch):
        fake_torch()
        out = self._prepared(np.array([[0, 128, 255]], dtype=np.uint8))
        assert out.min() >= 0.0
        assert out.max() == pytest.approx(1.0)

    def test_constant_image_does_not_divide_by_zero(self, fake_torch):
        """A blank slice has a zero peak; the result must stay finite."""
        fake_torch()
        out = self._prepared(np.zeros((8, 8), dtype=np.uint8))
        assert np.all(np.isfinite(out))
        assert out.max() == 0.0

    def test_negative_values_are_clipped(self, fake_torch):
        fake_torch()
        out = self._prepared(np.array([[-50.0, 0.0, 100.0]], dtype=np.float32))
        assert out.min() >= 0.0

    def test_output_is_contiguous(self, fake_torch):
        """torch.from_numpy needs a contiguous buffer after the transpose."""
        fake_torch()
        out = self._prepared(np.zeros((8, 10, 3), dtype=np.uint8))
        assert out.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize(
        "image",
        [
            np.full((12, 16), 200, dtype=np.uint8),
            np.zeros((6, 6, 1), dtype=np.uint8),
            np.zeros((6, 6, 4), dtype=np.uint8),
            np.full((5, 5), 3.5, dtype=np.float64),
        ],
    )
    def test_output_is_float32(self, fake_torch, image):
        """torch.from_numpy adopts the array's dtype.

        The model needs float32, and a float64 input would otherwise produce a
        double tensor and a dtype mismatch inside the network. Asserting it on
        the array means the guarantee is checked without torch installed.
        """
        fake_torch()
        assert self._prepared(image).dtype == np.float32


class TestApplyMatcher:
    """The wiring between the model output and the returned points."""

    @staticmethod
    def _matcher(src_pts, dst_pts, conf, device="cpu", recorder=None):
        """A stand-in model that writes results into the data dict."""

        class FakeMatcher:
            def __init__(self):
                self._tpsreg_device = device
                self.calls = 0
                self.seen_keys = None

            def __call__(self, data):
                self.calls += 1
                self.seen_keys = set(data)
                data["mkpts0_f"] = _FakeTensor(np.asarray(dst_pts, dtype=float))
                data["mkpts1_f"] = _FakeTensor(np.asarray(src_pts, dtype=float))
                data["mconf"] = _FakeTensor(np.asarray(conf, dtype=float))

        return FakeMatcher()

    def test_returns_source_then_destination_then_confidence(self, fake_torch):
        """mkpts1 is the source side; the swap is easy to get backwards."""
        fake_torch()
        src = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        dst = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
        conf = [0.9, 0.8, 0.7]
        matcher = self._matcher(src, dst, conf)

        out_src, out_dst, out_conf = roma_matcher.apply_matcher(
            matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
        )

        np.testing.assert_allclose(out_src, src)
        np.testing.assert_allclose(out_dst, dst)
        np.testing.assert_allclose(out_conf, conf)

    def test_model_receives_both_images_under_the_expected_keys(self, fake_torch):
        fake_torch()
        matcher = self._matcher([[0.0, 0.0]], [[1.0, 1.0]], [1.0])

        roma_matcher.apply_matcher(
            matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
        )

        assert matcher.calls == 1
        assert matcher.seen_keys == {"image0_rgb_origin", "image1_rgb_origin"}

    def test_ransac_filters_the_results(self, fake_torch, monkeypatch):
        fake_torch()
        src = [[float(i), float(i)] for i in range(6)]
        dst = [[float(i) * 2, float(i) * 2] for i in range(6)]
        conf = [0.5] * 6
        matcher = self._matcher(src, dst, conf)

        keep = np.array([True, False, True, True, False, True])
        captured = {}

        def fake_ransac(a, b, **kwargs):
            captured["kwargs"] = kwargs
            return keep

        monkeypatch.setattr(roma_matcher, "ransac", fake_ransac)

        out_src, out_dst, out_conf = roma_matcher.apply_matcher(
            matcher,
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            ransac_filter=True,
            ransac_threshold=0.11,
            ransac_method="projective",
            ransac_max_trials=222,
        )

        assert len(out_src) == keep.sum()
        assert len(out_dst) == keep.sum()
        assert len(out_conf) == keep.sum()
        # The settings must reach the filter rather than being dropped.
        assert captured["kwargs"]["threshold"] == 0.11
        assert captured["kwargs"]["method"] == "projective"
        assert captured["kwargs"]["max_trials"] == 222

    def test_ransac_is_skipped_below_four_matches(self, fake_torch, monkeypatch):
        """A minimal model cannot be fitted from three correspondences."""
        fake_torch()
        matcher = self._matcher(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
            [0.5, 0.5, 0.5],
        )

        def explode(*args, **kwargs):
            raise AssertionError("RANSAC should not run with fewer than 4 matches")

        monkeypatch.setattr(roma_matcher, "ransac", explode)

        out_src, _, _ = roma_matcher.apply_matcher(
            matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=True
        )
        assert len(out_src) == 3

    def test_disabling_ransac_returns_every_match(self, fake_torch, monkeypatch):
        fake_torch()
        src = [[float(i), 0.0] for i in range(10)]
        matcher = self._matcher(src, src, [0.5] * 10)

        def explode(*args, **kwargs):
            raise AssertionError("RANSAC should not run when disabled")

        monkeypatch.setattr(roma_matcher, "ransac", explode)

        out_src, _, _ = roma_matcher.apply_matcher(
            matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
        )
        assert len(out_src) == 10

    def test_tensors_are_moved_to_the_matcher_device(self, fake_torch):
        """The cached device avoids redetecting on every call."""
        fake_torch(cuda=True)
        matcher = self._matcher([[0.0, 0.0]], [[1.0, 1.0]], [1.0], device="cuda")

        captured = {}
        original = roma_matcher._prepare_image

        def spy(image):
            tensor = original(image)
            captured.setdefault("tensors", []).append(tensor)
            return tensor

        import unittest.mock

        with unittest.mock.patch.object(roma_matcher, "_prepare_image", spy):
            roma_matcher.apply_matcher(
                matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
            )

        assert captured["tensors"], "images were never prepared"

    def test_autocast_only_on_cuda(self, fake_torch):
        """Autocast is a CUDA-only optimisation; enabling it elsewhere is wrong."""
        recorder = []
        fake_torch(cuda=True, recorder=recorder)
        matcher = self._matcher([[0.0, 0.0]], [[1.0, 1.0]], [1.0], device="cuda")

        roma_matcher.apply_matcher(
            matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
        )
        assert "autocast:cuda" in recorder

    def test_no_autocast_on_cpu(self, fake_torch):
        recorder = []
        fake_torch(cuda=False, recorder=recorder)
        matcher = self._matcher([[0.0, 0.0]], [[1.0, 1.0]], [1.0], device="cpu")

        roma_matcher.apply_matcher(
            matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
        )
        assert not any(entry.startswith("autocast") for entry in recorder)
        assert "no_grad" in recorder

    def test_gradients_are_disabled(self, fake_torch):
        """Inference under autograd would waste a lot of memory."""
        recorder = []
        fake_torch(recorder=recorder)
        matcher = self._matcher([[0.0, 0.0]], [[1.0, 1.0]], [1.0])

        roma_matcher.apply_matcher(
            matcher, np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
        )
        assert "no_grad" in recorder

    def test_device_is_detected_when_the_matcher_has_none(self, fake_torch):
        """A matcher built by other means still gets a valid device."""
        fake_torch(cuda=False, mps=False)

        class BareMatcher:
            def __call__(self, data):
                data["mkpts0_f"] = _FakeTensor(np.zeros((2, 2)))
                data["mkpts1_f"] = _FakeTensor(np.zeros((2, 2)))
                data["mconf"] = _FakeTensor(np.zeros(2))

        out_src, _, _ = roma_matcher.apply_matcher(
            BareMatcher(), np.zeros((8, 8)), np.zeros((8, 8)), ransac_filter=False
        )
        assert len(out_src) == 2


class TestDetectPointsConvenience:
    """The one-call wrapper."""

    def test_builds_a_matcher_and_forwards_ransac_settings(self, monkeypatch):
        created = {}
        applied = {}

        def fake_create(checkpoint_path=None, device=None):
            created["checkpoint_path"] = checkpoint_path
            created["device"] = device
            return object()

        def fake_apply(matcher, src, dst, **kwargs):
            applied.update(kwargs)
            return np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2)

        monkeypatch.setattr(roma_matcher, "create_matcher", fake_create)
        monkeypatch.setattr(roma_matcher, "apply_matcher", fake_apply)

        roma_matcher.detect_points_matchanything(
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            checkpoint_path="/models/roma.ckpt",
            device="cpu",
            ransac_threshold=0.3,
            ransac_method="affine",
            ransac_max_trials=17,
        )

        assert created["checkpoint_path"] == "/models/roma.ckpt"
        assert created["device"] == "cpu"
        assert applied["ransac_threshold"] == 0.3
        assert applied["ransac_method"] == "affine"
        assert applied["ransac_max_trials"] == 17

    def test_defaults_are_sensible(self, monkeypatch):
        applied = {}

        monkeypatch.setattr(roma_matcher, "create_matcher", lambda **kwargs: object())

        def fake_apply(matcher, src, dst, **kwargs):
            applied.update(kwargs)
            return np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2)

        monkeypatch.setattr(roma_matcher, "apply_matcher", fake_apply)

        roma_matcher.detect_points_matchanything(np.zeros((8, 8)), np.zeros((8, 8)))

        assert applied["ransac_filter"] is True
        assert applied["ransac_method"] == "deformable"


class TestCreateMatcherGuards:
    """Failures before any model is constructed."""

    def test_missing_checkpoint_is_reported_clearly(self, monkeypatch, tmp_path):
        """The user picks this path in a file dialog, so it is easy to get wrong."""
        # Get past the import guard with a stand-in model module.
        fake_module = types.ModuleType(
            "tpsreg.Matchanything.src.lightning.lightning_loftr"
        )
        fake_module.PL_LoFTR = object
        monkeypatch.setitem(
            sys.modules,
            "tpsreg.Matchanything.src.lightning.lightning_loftr",
            fake_module,
        )
        monkeypatch.setattr(
            roma_matcher,
            "get_config",
            lambda checkpoint_path=None: (
                object(),
                {"ckpt_path": str(tmp_path / "absent.ckpt")},
            ),
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            roma_matcher.create_matcher(checkpoint_path=tmp_path / "absent.ckpt")

        message = str(exc_info.value)
        assert "absent.ckpt" in message
        # It should say how to fix it, not just what failed.
        assert "checkpoint" in message.lower()


@pytest.mark.torch
class TestWithRealTorch:
    """A few checks against the real library, skipped when it is absent."""

    def test_select_device_returns_something_usable(self):
        torch = pytest.importorskip("torch")

        device = roma_matcher.select_device()
        assert device in {"cuda", "mps", "cpu"}
        # Whatever was chosen must actually be constructible.
        torch.device(device)

    def test_prepare_image_produces_a_real_tensor(self):
        torch = pytest.importorskip("torch")

        tensor = roma_matcher._prepare_image(np.full((12, 16), 200, dtype=np.uint8))
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 12, 16)
        assert tensor.dtype == torch.float32
        assert float(tensor.max()) <= 1.0
        assert float(tensor.min()) >= 0.0

    def test_prepare_image_handles_rgba(self):
        pytest.importorskip("torch")

        tensor = roma_matcher._prepare_image(np.zeros((6, 6, 4), dtype=np.uint8))
        assert tensor.shape == (3, 6, 6)
