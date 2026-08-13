"""Tests for slice-by-slice stack registration.

The matching model is injected, so these run without torch and without a
checkpoint. What is tested is everything around the model: fitting, composing,
warping, failure handling, and the report. The model itself is exercised in
test_roma_matcher.py.

Several tests build a stack by shifting one image by known amounts, so the
answer is known exactly and the assertions can be tight rather than
impressionistic.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from skimage import transform as sktransform

from tpsreg.stack_registration import (
    COMFORTABLE_MATCHES,
    MINIMUM_MATCHES,
    REFERENCE_MODES,
    TRANSFORM_TYPES,
    ChainedTransform,
    IdentityTransform,
    PairResult,
    StackResult,
    TranslationTransform,
    apply_transforms,
    estimate_pair_transform,
    find_images,
    flag_outlying_slices,
    match_figure,
    pair_residuals,
    register_stack,
    write_report,
)

#: A grid of correspondences dense enough for every model, including the
#: spline's leave-one-out residuals.
GRID = np.stack(
    np.meshgrid(np.linspace(20.0, 120.0, 5), np.linspace(20.0, 100.0, 5)), -1
).reshape(-1, 2)


@pytest.fixture
def shifted_stack(rng):
    """A stack of known relative shifts, with a matcher that knows them.

    Returns ``(images, cumulative_shifts, match_fn)``. Because the shifts are
    known, the recovered transforms can be checked against the truth rather
    than merely against themselves.
    """
    base = (rng.random((120, 140)) * 255).astype(np.uint8)
    base[30:50, 40:70] = 255

    steps = [np.zeros(2)] + [rng.uniform(-4, 4, 2) for _ in range(5)]
    cumulative = np.cumsum(steps, axis=0)

    images = [
        sktransform.warp(
            base,
            sktransform.EuclideanTransform(translation=-shift).inverse,
            order=1,
            preserve_range=True,
        ).astype(np.uint8)
        for shift in cumulative
    ]

    def match_fn(moving, reference):
        moving_index = next(i for i, im in enumerate(images) if im is moving)
        reference_index = next(i for i, im in enumerate(images) if im is reference)
        offset = cumulative[moving_index] - cumulative[reference_index]
        return GRID + offset, GRID, np.ones(len(GRID))

    return images, cumulative, match_fn


class TestTranslationTransform:
    """The shift-only model scikit-image does not provide."""

    def test_it_recovers_a_known_shift(self):
        src = GRID
        dst = GRID + np.array([3.0, -7.0])
        transform = TranslationTransform.estimate(src, dst)
        np.testing.assert_allclose(transform.offset, [3.0, -7.0])

    def test_it_maps_coordinates(self):
        transform = TranslationTransform(np.array([2.0, 5.0]))
        np.testing.assert_allclose(transform(np.array([[1.0, 1.0]])), [[3.0, 6.0]])

    def test_the_median_ignores_a_bad_match(self):
        """RANSAC does not catch everything; the median covers the remainder."""
        src = GRID.copy()
        dst = GRID + np.array([3.0, -7.0])
        dst[0] += np.array([200.0, 200.0])

        transform = TranslationTransform.estimate(src, dst)
        np.testing.assert_allclose(transform.offset, [3.0, -7.0], atol=1e-9)

    def test_a_mean_would_have_been_fooled(self):
        """Establishes that the previous test is actually testing something."""
        src = GRID.copy()
        dst = GRID + np.array([3.0, -7.0])
        dst[0] += np.array([200.0, 200.0])

        mean_offset = np.mean(dst - src, axis=0)
        assert not np.allclose(mean_offset, [3.0, -7.0], atol=0.5)

    def test_it_describes_itself_for_the_report(self):
        assert TranslationTransform(np.array([1.5, 2.5])).describe() == {
            "dx": 1.5,
            "dy": 2.5,
        }


class TestChainedTransform:
    """Composition, which is what keeps the stack from being resampled twice."""

    def test_it_applies_in_order(self):
        chain = ChainedTransform(
            [
                TranslationTransform(np.array([1.0, 0.0])),
                TranslationTransform(np.array([0.0, 10.0])),
            ]
        )
        np.testing.assert_allclose(chain(np.array([[0.0, 0.0]])), [[1.0, 10.0]])

    def test_an_empty_chain_is_the_identity(self):
        chain = ChainedTransform([])
        query = np.array([[3.0, 4.0]])
        np.testing.assert_allclose(chain(query), query)

    def test_it_composes_a_spline_too(self, rng):
        """Matrix transforms compose by multiplication; splines do not.

        Composing them as functions is what makes TPS work in sequential mode
        at all, and it means one resampling however long the chain.
        """
        from tpsreg.tps import ThinPlateSplineTransform

        spline = ThinPlateSplineTransform()
        spline.estimate(GRID + np.array([2.0, 3.0]), GRID, (120, 140))

        chain = ChainedTransform([spline, TranslationTransform(np.array([5.0, 0.0]))])
        query = np.array([[60.0, 60.0]])

        np.testing.assert_allclose(
            chain(query), spline(query) + np.array([5.0, 0.0]), atol=1e-6
        )


class TestEstimatePairTransform:
    """Fitting one pair."""

    @pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
    def test_every_model_recovers_a_pure_shift(self, transform_type):
        """A shift is inside every one of these models."""
        shift = np.array([4.0, -6.0])
        transform = estimate_pair_transform(
            GRID + shift, GRID, transform_type, shape=(120, 140)
        )

        query = np.array([[50.0, 50.0], [90.0, 70.0]])
        np.testing.assert_allclose(transform(query), query + shift, atol=1e-6)

    def test_the_direction_is_reference_to_moving(self, rng):
        """Warping fills an output pixel by reading from the moving image.

        Getting this backwards produces a result that looks plausible and is
        wrong by twice the offset, so it is worth pinning explicitly.
        """
        shift = np.array([7.0, 0.0])
        transform = estimate_pair_transform(GRID + shift, GRID, "translation")

        reference_point = np.array([[40.0, 40.0]])
        np.testing.assert_allclose(transform(reference_point), reference_point + shift)

    def test_affine_recovers_a_scale(self):
        moving = GRID * 1.25
        transform = estimate_pair_transform(moving, GRID, "affine")
        np.testing.assert_allclose(
            transform(np.array([[40.0, 80.0]])), [[50.0, 100.0]], atol=1e-6
        )

    def test_rigid_recovers_a_rotation(self):
        angle = np.deg2rad(10.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        moving = GRID @ rotation.T

        transform = estimate_pair_transform(moving, GRID, "rigid")
        np.testing.assert_allclose(transform(GRID), moving, atol=1e-6)

    def test_a_spline_absorbs_local_deformation(self, rng):
        """The reason to reach for it: something the rigid models cannot fit."""
        moving = GRID.copy()
        moving[12] += np.array([6.0, -5.0])

        spline = estimate_pair_transform(moving, GRID, "tps", shape=(120, 140))
        rigid = estimate_pair_transform(moving, GRID, "rigid")

        spline_error = np.linalg.norm(spline(GRID) - moving, axis=1).max()
        rigid_error = np.linalg.norm(rigid(GRID) - moving, axis=1).max()
        assert spline_error < rigid_error

    @pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
    def test_too_few_matches_is_refused_by_name(self, transform_type):
        needed = MINIMUM_MATCHES[transform_type]
        with pytest.raises(ValueError, match=f"at least {needed}"):
            estimate_pair_transform(
                GRID[: needed - 1], GRID[: needed - 1], transform_type
            )

    def test_an_unknown_model_lists_the_options(self):
        with pytest.raises(ValueError, match="Unknown transform type"):
            estimate_pair_transform(GRID, GRID, "elastic")


class TestPairResiduals:
    """How far the matches sit from the fit."""

    def test_a_constrained_model_has_real_residuals(self):
        moving = GRID.copy()
        moving[3] += np.array([9.0, 9.0])

        transform = estimate_pair_transform(moving, GRID, "rigid")
        residuals = pair_residuals(transform, moving, GRID, "rigid")

        assert residuals.shape == (len(GRID),)
        assert residuals.max() > 1.0

    def test_a_spline_is_measured_by_leave_one_out(self):
        """Its ordinary residual is zero whatever the matches look like."""
        moving = GRID.copy()
        moving[12] += np.array([9.0, 9.0])

        transform = estimate_pair_transform(moving, GRID, "tps", shape=(120, 140))

        ordinary = np.linalg.norm(transform(GRID) - moving, axis=1)
        assert ordinary.max() < 1e-6, "the spline interpolates, as expected"

        residuals = pair_residuals(transform, moving, GRID, "tps")
        assert int(np.nanargmax(residuals)) == 12

    def test_a_spline_with_too_few_matches_reports_nothing(self):
        """Below the threshold leave-one-out misleads rather than informs."""
        # Spread, not GRID[:5] -- that is the grid's first row, which is
        # collinear and cannot be fitted at all.
        few = np.array(
            [[20.0, 20.0], [110.0, 25.0], [25.0, 95.0], [105.0, 90.0], [60.0, 55.0]]
        )
        transform = estimate_pair_transform(few, few, "tps", shape=(120, 140))
        assert np.all(np.isnan(pair_residuals(transform, few, few, "tps")))


class TestRegisterStack:
    """The whole pass over a stack."""

    def test_it_recovers_the_true_shifts(self, shifted_stack):
        images, cumulative, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn, transform_type="rigid")

        query = np.array([[70.0, 60.0]])
        for index, transform in enumerate(transforms):
            expected = query + (cumulative[index] - cumulative[0])
            np.testing.assert_allclose(transform(query), expected, atol=1e-6)

    @pytest.mark.parametrize("transform_type", TRANSFORM_TYPES)
    def test_every_model_recovers_them(self, shifted_stack, transform_type):
        images, cumulative, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn, transform_type=transform_type)

        query = np.array([[70.0, 60.0]])
        errors = [
            np.linalg.norm(transform(query) - (query + (cumulative[i] - cumulative[0])))
            for i, transform in enumerate(transforms)
        ]
        assert max(errors) < 1e-5

    def test_registration_actually_aligns_the_images(self, shifted_stack):
        """The end-to-end claim, measured on pixels rather than parameters."""
        images, _, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn, transform_type="rigid")
        registered = apply_transforms(images, transforms, order=1)

        interior = (slice(20, 100), slice(20, 120))
        before = np.std([im[interior] for im in images], axis=0).mean()
        after = np.std([im[interior] for im in registered], axis=0).mean()

        assert after < before / 10

    def test_the_anchor_slice_is_left_alone(self, shifted_stack):
        images, _, match_fn = shifted_stack
        transforms, result = register_stack(images, match_fn)

        query = np.array([[50.0, 50.0]])
        np.testing.assert_allclose(transforms[0](query), query)
        assert result.pairs[0].is_reference

    @pytest.mark.parametrize("mode", REFERENCE_MODES)
    def test_every_reference_mode_runs(self, shifted_stack, mode):
        images, _cumulative, match_fn = shifted_stack
        transforms, result = register_stack(
            images, match_fn, reference_mode=mode, transform_type="rigid"
        )

        assert len(transforms) == len(images)
        assert result.reference_mode == mode
        assert not result.failed

    def test_middle_anchors_on_the_middle_slice(self, shifted_stack):
        images, _, match_fn = shifted_stack
        _, result = register_stack(images, match_fn, reference_mode="middle")

        anchors = [pair.index for pair in result.pairs if pair.is_reference]
        assert anchors == [len(images) // 2]

    def test_previous_mode_chains(self, shifted_stack):
        """Each slice composes every step back to the anchor."""
        images, _, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn, reference_mode="previous")

        assert isinstance(transforms[-1], ChainedTransform)
        assert len(transforms[-1]) == len(images) - 1

    def test_fixed_reference_mode_does_not_chain(self, shifted_stack):
        images, _, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn, reference_mode="first")
        assert not isinstance(transforms[-1], ChainedTransform)

    def test_drift_is_reported(self, shifted_stack):
        images, cumulative, match_fn = shifted_stack
        _, result = register_stack(images, match_fn, transform_type="rigid")

        expected = float(np.linalg.norm(cumulative[-1] - cumulative[0]))
        assert result.cumulative_displacement[-1] == pytest.approx(expected, abs=1e-4)

    def test_progress_is_reported_for_every_slice(self, shifted_stack):
        images, _, match_fn = shifted_stack
        seen = []
        register_stack(
            images,
            match_fn,
            on_progress=lambda done, total, pair: seen.append((done, total)),
        )

        assert [done for done, _ in seen] == list(range(1, len(images) + 1))
        assert all(total == len(images) for _, total in seen)

    def test_an_empty_stack_is_refused(self):
        with pytest.raises(ValueError, match="No images"):
            register_stack([], lambda a, b: (GRID, GRID, None))

    def test_an_unknown_model_is_refused(self, shifted_stack):
        images, _, match_fn = shifted_stack
        with pytest.raises(ValueError, match="Unknown transform type"):
            register_stack(images, match_fn, transform_type="elastic")

    def test_an_unknown_reference_mode_is_refused(self, shifted_stack):
        images, _, match_fn = shifted_stack
        with pytest.raises(ValueError, match="Unknown reference mode"):
            register_stack(images, match_fn, reference_mode="last")


class TestFailureHandling:
    """One bad pair must not cost the rest of the stack."""

    def test_a_matching_failure_is_recorded_not_raised(self, shifted_stack):
        images, _, match_fn = shifted_stack

        def flaky(moving, reference):
            if moving is images[3]:
                raise RuntimeError("the model returned nothing")
            return match_fn(moving, reference)

        transforms, result = register_stack(images, flaky)

        assert len(transforms) == len(images)
        assert [pair.index for pair in result.failed] == [3]
        assert "the model returned nothing" in result.failed[0].warnings[0]

    def test_a_failed_slice_gets_the_identity(self, shifted_stack):
        images, _, match_fn = shifted_stack

        def flaky(moving, reference):
            if moving is images[2]:
                raise RuntimeError("no matches")
            return match_fn(moving, reference)

        _, result = register_stack(images, flaky, reference_mode="first")
        assert result.failed

        transforms, _ = register_stack(images, flaky, reference_mode="first")
        query = np.array([[10.0, 10.0]])
        np.testing.assert_allclose(transforms[2](query), query)

    def test_a_fit_failure_is_recorded(self, shifted_stack):
        """Too few matches for the chosen model."""
        images, _, match_fn = shifted_stack

        def sparse(moving, reference):
            moving_points, reference_points, confidence = match_fn(moving, reference)
            return moving_points[:1], reference_points[:1], confidence[:1]

        _, result = register_stack(images, sparse, transform_type="affine")

        assert len(result.failed) == len(images) - 1
        assert "at least 3" in result.failed[0].warnings[0]

    def test_few_matches_are_flagged_even_when_the_fit_succeeds(self, shifted_stack):
        images, _, match_fn = shifted_stack

        def sparse(moving, reference):
            moving_points, reference_points, confidence = match_fn(moving, reference)
            keep = COMFORTABLE_MATCHES - 1
            return moving_points[:keep], reference_points[:keep], confidence[:keep]

        _, result = register_stack(images, sparse, transform_type="translation")

        assert not result.failed
        assert result.flagged
        assert "little redundancy" in result.flagged[0].warnings[0]


class TestOutlierSlices:
    """Finding the slice that moved unlike its neighbours."""

    @staticmethod
    def _result(displacements):
        result = StackResult(transform_type="rigid", reference_mode="previous")
        result.pairs = [PairResult(index=0, reference_index=0, displacement=0.0)]
        for index, value in enumerate(displacements, start=1):
            result.pairs.append(
                PairResult(index=index, reference_index=index - 1, displacement=value)
            )
        return result

    def test_a_consistent_stack_flags_nothing(self):
        assert flag_outlying_slices(self._result([2.0, 2.1, 1.9, 2.0, 2.05])) == []

    def test_a_jump_is_flagged(self):
        assert flag_outlying_slices(self._result([2.0, 2.1, 1.9, 40.0, 2.05])) == [4]

    def test_an_identical_stack_with_one_jump_is_flagged(self):
        """The spread is zero, so the usual score is undefined.

        Returning "nothing stands out" here would be exactly wrong: when every
        other slice agrees perfectly, the one that does not is the whole point.
        """
        assert flag_outlying_slices(self._result([2.0, 2.0, 2.0, 2.0, 30.0])) == [5]

    def test_a_perfectly_uniform_stack_flags_nothing(self):
        assert flag_outlying_slices(self._result([2.0, 2.0, 2.0, 2.0, 2.0])) == []

    def test_the_median_is_used_rather_than_the_mean(self):
        """One bad slice inflates the standard deviation enough to hide."""
        displacements = [2.0, 2.0, 2.0, 2.0, 2.0, 30.0]
        values = np.array(displacements)
        classic = (values.max() - values.mean()) / values.std()
        assert classic < 3, "the setup must actually defeat a plain z-score"

        assert flag_outlying_slices(self._result(displacements))

    def test_too_short_a_stack_flags_nothing(self):
        assert flag_outlying_slices(self._result([2.0])) == []


class TestReport:
    """What gets written to disk."""

    def test_it_writes_both_files(self, shifted_stack, tmp_path):
        images, _, match_fn = shifted_stack
        _, result = register_stack(images, match_fn)
        write_report(result, tmp_path)

        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "summary.csv").exists()

    def test_the_json_records_every_slice(self, shifted_stack, tmp_path):
        images, _, match_fn = shifted_stack
        _, result = register_stack(images, match_fn)
        write_report(result, tmp_path)

        payload = json.loads((tmp_path / "report.json").read_text())
        assert len(payload["pairs"]) == len(images)
        assert payload["transform_type"] == "rigid"
        assert "cumulative_displacement" in payload

    def test_the_csv_has_one_row_per_slice(self, shifted_stack, tmp_path):
        images, _, match_fn = shifted_stack
        _, result = register_stack(images, match_fn)
        write_report(result, tmp_path)

        with (tmp_path / "summary.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == len(images)
        assert rows[0]["index"] == "0"
        assert all(row["ok"] == "yes" for row in rows)

    def test_a_failure_is_visible_in_the_csv(self, shifted_stack, tmp_path):
        images, _, match_fn = shifted_stack

        def flaky(moving, reference):
            if moving is images[2]:
                raise RuntimeError("nope")
            return match_fn(moving, reference)

        _, result = register_stack(images, flaky)
        write_report(result, tmp_path)

        with (tmp_path / "summary.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert rows[2]["ok"] == "NO"
        assert "nope" in rows[2]["warnings"]

    def test_extra_settings_are_recorded(self, shifted_stack, tmp_path):
        """So a result folder says how it was produced."""
        images, _, match_fn = shifted_stack
        _, result = register_stack(images, match_fn)
        write_report(result, tmp_path, extra={"settings": {"transform": "rigid"}})

        payload = json.loads((tmp_path / "report.json").read_text())
        assert payload["settings"]["transform"] == "rigid"

    def test_the_summary_reads_as_prose(self, shifted_stack):
        images, _, match_fn = shifted_stack
        _, result = register_stack(images, match_fn)

        summary = result.summary()
        assert "slice(s)" in summary
        assert "drift" in summary


class TestMatchFigure:
    """The picture that makes a bad pair obvious."""

    def test_it_places_both_images_side_by_side(self, rng):
        left = (rng.random((40, 50)) * 255).astype(np.uint8)
        right = (rng.random((40, 60)) * 255).astype(np.uint8)

        figure = match_figure(right, left, GRID[:5], GRID[:5])
        assert figure.shape == (40, 110, 3)

    def test_differently_sized_slices_are_survivable(self, rng):
        left = (rng.random((40, 50)) * 255).astype(np.uint8)
        right = (rng.random((70, 30)) * 255).astype(np.uint8)

        figure = match_figure(right, left, GRID[:3], GRID[:3])
        assert figure.shape == (70, 80, 3)

    def test_no_matches_still_produces_an_image(self, rng):
        image = (rng.random((30, 30)) * 255).astype(np.uint8)
        figure = match_figure(image, image, np.empty((0, 2)), np.empty((0, 2)))
        assert figure.shape == (30, 60, 3)

    def test_it_draws_something(self, rng):
        """A blank figure would be a silent failure."""
        image = np.zeros((120, 140), dtype=np.uint8)
        figure = match_figure(image, image, GRID, GRID)
        assert figure.any(), "no lines were drawn"

    def test_it_caps_how_many_lines_it_draws(self, rng):
        """Hundreds of matches make an unreadable tangle."""
        image = np.zeros((120, 140), dtype=np.uint8)
        many = np.repeat(GRID, 20, axis=0)

        sparse = match_figure(image, image, many, many, max_lines=5)
        dense = match_figure(image, image, many, many, max_lines=60)
        assert sparse.astype(bool).sum() < dense.astype(bool).sum()


class TestFindImages:
    """Listing a folder."""

    def test_it_sorts_numerically_not_lexically(self, tmp_path):
        """slice_10 must come after slice_2, or the stack is silently reordered."""
        for name in ("slice_1.tif", "slice_2.tif", "slice_10.tif", "slice_20.tif"):
            (tmp_path / name).touch()

        names = [path.name for path in find_images(tmp_path)]
        assert names == ["slice_1.tif", "slice_2.tif", "slice_10.tif", "slice_20.tif"]

    def test_lexical_sorting_would_have_been_wrong(self, tmp_path):
        """Establishes that the previous test is testing something."""
        names = ["slice_1.tif", "slice_2.tif", "slice_10.tif"]
        assert sorted(names) != names

    def test_it_ignores_other_files(self, tmp_path):
        (tmp_path / "image.tif").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "report.json").touch()

        assert [p.name for p in find_images(tmp_path)] == ["image.tif"]

    def test_it_is_case_insensitive(self, tmp_path):
        (tmp_path / "a.TIF").touch()
        (tmp_path / "b.PNG").touch()
        assert len(find_images(tmp_path)) == 2

    def test_a_missing_folder_says_so(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            find_images(tmp_path / "nope")

    def test_an_empty_folder_returns_nothing(self, tmp_path):
        assert find_images(tmp_path) == []


class TestApplyTransforms:
    """Warping the stack."""

    def test_one_output_per_input(self, shifted_stack):
        images, _, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn)
        assert len(apply_transforms(images, transforms)) == len(images)

    def test_the_output_shape_is_honoured(self, shifted_stack):
        images, _, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn)

        warped = apply_transforms(images, transforms, output_shape=(60, 70))
        assert all(image.shape[:2] == (60, 70) for image in warped)

    def test_order_zero_preserves_the_dtype(self, shifted_stack):
        """Label images must not be interpolated into meaningless values."""
        images, _, match_fn = shifted_stack
        transforms, _ = register_stack(images, match_fn)

        warped = apply_transforms(images, transforms, order=0)
        assert warped[0].dtype == images[0].dtype

    def test_an_identity_transform_returns_the_image(self, rng):
        image = (rng.random((40, 50)) * 255).astype(np.uint8)
        warped = apply_transforms([image], [IdentityTransform()], order=0)
        np.testing.assert_array_equal(warped[0], image)
