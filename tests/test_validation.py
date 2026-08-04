"""Tests for the pre-estimation control point checks.

No Tk and no image data: these are pure geometry, which is the whole point of
keeping them out of the view.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from tpsreg.models import TransformType
from tpsreg.validation import (
    COMFORTABLE_POINTS,
    Issue,
    Severity,
    check_control_points,
    convex_hull_area,
    format_issues,
    has_errors,
    minimum_points,
)


def codes(issues: list[Issue]) -> set[str]:
    return {issue.code for issue in issues}


def issue_with(issues: list[Issue], code: str) -> Issue:
    matching = [issue for issue in issues if issue.code == code]
    assert matching, f"expected a {code!r} issue, got {sorted(codes(issues))}"
    return matching[0]


def spread_points(n: int, size: int = 100) -> np.ndarray:
    """``n`` well-separated, non-collinear points inside a square."""
    rng = np.random.default_rng(0)
    return rng.uniform(5, size - 5, size=(n, 2))


class TestMinimumPoints:
    """How many pairs a transform needs."""

    def test_accepts_a_transform_type_member(self):
        assert minimum_points(TransformType.TPS) == 3

    def test_accepts_the_bare_value(self):
        assert minimum_points("tps") == 3

    def test_is_case_insensitive(self):
        assert minimum_points("TPS") == 3

    def test_unknown_transform_falls_back(self):
        """A new transform type must not make the checks crash."""
        assert minimum_points("some_future_transform") == 3

    def test_a_non_string_falls_back(self):
        assert minimum_points(object()) == 3


class TestStructuralProblems:
    """Problems that stop any geometry check from being meaningful."""

    def test_no_points_at_all(self):
        issues = check_control_points([], [])
        assert codes(issues) == {"no_points"}
        assert has_errors(issues)

    def test_no_points_on_one_side_only(self):
        issues = check_control_points(spread_points(5), [])
        assert codes(issues) == {"no_points"}

    def test_mismatched_counts(self):
        issues = check_control_points(spread_points(5), spread_points(4))
        assert codes(issues) == {"count_mismatch"}
        assert "5 source points but 4" in issue_with(issues, "count_mismatch").message

    def test_a_mismatch_suppresses_the_geometry_checks(self):
        """One clear message beats a pile of consequential ones."""
        collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        issues = check_control_points(collinear, collinear[:2])
        assert codes(issues) == {"count_mismatch"}

    def test_too_few_points(self):
        two = np.array([[0.0, 0.0], [50.0, 80.0]])
        issues = check_control_points(two, two)
        assert "too_few_points" in codes(issues)
        assert has_errors(issues)

    def test_exactly_the_minimum_is_not_an_error(self):
        three = np.array([[0.0, 0.0], [90.0, 5.0], [10.0, 90.0]])
        issues = check_control_points(three, three)
        assert "too_few_points" not in codes(issues)
        assert not has_errors(issues)

    def test_ragged_input_is_reported_as_empty_not_crashed(self):
        """A malformed array must not raise out of a validity check."""
        issues = check_control_points([[1.0]], [[1.0]])
        assert codes(issues) == {"no_points"}


class TestDuplicatePoints:
    """Coincident points, which is what makes the spline singular."""

    def test_duplicate_source_points_are_an_error(self):
        points = spread_points(6)
        points[3] = points[1]
        issues = check_control_points(points, spread_points(6))
        assert "duplicate_source_points" in codes(issues)
        assert issue_with(issues, "duplicate_source_points").is_error

    def test_duplicate_destination_points_are_an_error(self):
        """Two identical rows in the system matrix; it cannot be solved."""
        points = spread_points(6)
        points[2] = points[0]
        issues = check_control_points(spread_points(6), points)
        assert issue_with(issues, "duplicate_destination_points").is_error

    def test_near_duplicates_within_the_tolerance_count(self):
        points = spread_points(6)
        points[4] = points[2] + [0.4, 0.3]
        issues = check_control_points(points, spread_points(6))
        assert "duplicate_source_points" in codes(issues)

    def test_points_just_outside_the_tolerance_do_not(self):
        points = spread_points(6)
        points[4] = points[2] + [3.0, 3.0]
        issues = check_control_points(points, spread_points(6))
        assert "duplicate_source_points" not in codes(issues)

    def test_the_tolerance_is_configurable(self):
        points = spread_points(6)
        points[4] = points[2] + [3.0, 0.0]
        assert "duplicate_source_points" not in codes(
            check_control_points(points, spread_points(6))
        )
        assert "duplicate_source_points" in codes(
            check_control_points(points, spread_points(6), duplicate_tolerance=5.0)
        )

    def test_the_message_names_the_offending_indices(self):
        points = spread_points(6)
        points[3] = points[1]
        message = issue_with(
            check_control_points(points, spread_points(6)), "duplicate_source_points"
        ).message
        assert "1" in message and "3" in message


class TestCollinearPoints:
    """A line says nothing about the direction across it."""

    def test_collinear_source_points_are_an_error(self):
        line = np.array([[float(i), float(i) * 2] for i in range(8)])
        issues = check_control_points(line, spread_points(8))
        assert "collinear_source_points" in codes(issues)
        assert has_errors(issues)

    def test_collinear_destination_points_are_reported_first(self):
        """Both collinear: name the one that makes the system singular.

        The solver checks the destination first, so this keeps the two
        messages saying the same thing about the same point set.
        """
        line = np.array([[float(i) * 10, float(i) * 10] for i in range(8)])
        issues = check_control_points(line, line + 1.0)
        assert "collinear_destination_points" in codes(issues)

    def test_a_horizontal_line_counts(self):
        line = np.array([[float(i) * 10, 5.0] for i in range(8)])
        issues = check_control_points(line, spread_points(8))
        assert "collinear_source_points" in codes(issues)

    def test_a_vertical_line_counts(self):
        line = np.array([[5.0, float(i) * 10] for i in range(8)])
        issues = check_control_points(line, spread_points(8))
        assert "collinear_source_points" in codes(issues)

    def test_collinear_destination_points_are_an_error(self):
        line = np.array([[float(i) * 10, float(i) * 10] for i in range(8)])
        issues = check_control_points(spread_points(8), line)
        assert "collinear_destination_points" in codes(issues)

    def test_a_slight_bend_is_accepted(self):
        """Real clicks are never exactly on a line; don't reject them."""
        points = np.array([[float(i) * 10, float(i) * 10] for i in range(8)])
        points[3, 1] += 6.0
        issues = check_control_points(points, points)
        assert "collinear_source_points" not in codes(issues)

    def test_spread_points_are_not_collinear(self):
        points = spread_points(10)
        assert "collinear_source_points" not in codes(
            check_control_points(points, points)
        )

    def test_all_points_identical_reports_duplicates_not_collinearity(self):
        """ "They're on a line" is true but useless when they're one point."""
        same = np.tile([20.0, 30.0], (5, 1))
        issues = check_control_points(same, same)
        assert "duplicate_source_points" in codes(issues)
        assert "collinear_source_points" not in codes(issues)


class TestAdvisoryWarnings:
    """Things that will work but probably disappoint."""

    def test_few_points_warns_about_an_affine_like_fit(self):
        points = spread_points(4)
        issues = check_control_points(points, points)
        assert "sparse_points" in codes(issues)
        assert not has_errors(issues)

    def test_plenty_of_points_does_not_warn(self):
        points = spread_points(COMFORTABLE_POINTS + 2)
        assert "sparse_points" not in codes(check_control_points(points, points))

    def test_clustered_points_warn_about_extrapolation(self):
        cluster = np.array(
            [
                [10.0, 10.0],
                [30.0, 12.0],
                [12.0, 34.0],
                [28.0, 30.0],
                [20.0, 20.0],
                [15.0, 28.0],
            ]
        )
        issues = check_control_points(cluster, cluster, image_shape=(1000, 1000))
        assert "poor_coverage" in codes(issues)
        assert not has_errors(issues)

    def test_points_spanning_the_image_do_not_warn(self):
        spread = np.array(
            [
                [10.0, 10.0],
                [900.0, 20.0],
                [20.0, 900.0],
                [880.0, 890.0],
                [400.0, 500.0],
                [600.0, 200.0],
            ]
        )
        issues = check_control_points(spread, spread, image_shape=(1000, 1000))
        assert "poor_coverage" not in codes(issues)

    def test_coverage_is_skipped_without_an_image_shape(self):
        cluster = spread_points(8, size=20)
        assert "poor_coverage" not in codes(check_control_points(cluster, cluster))

    def test_a_degenerate_image_shape_is_ignored(self):
        points = spread_points(8, size=20)
        assert "poor_coverage" not in codes(
            check_control_points(points, points, image_shape=(0, 0))
        )

    def test_the_coverage_threshold_is_configurable(self):
        points = np.array(
            [
                [10.0, 10.0],
                [300.0, 20.0],
                [20.0, 300.0],
                [280.0, 290.0],
                [150.0, 150.0],
                [200.0, 90.0],
            ]
        )
        lenient = check_control_points(
            points, points, image_shape=(1000, 1000), coverage_fraction=0.01
        )
        strict = check_control_points(
            points, points, image_shape=(1000, 1000), coverage_fraction=0.5
        )
        assert "poor_coverage" not in codes(lenient)
        assert "poor_coverage" in codes(strict)


class TestConvexHullArea:
    """The coverage measure, on its own."""

    def test_a_unit_square(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert convex_hull_area(square) == pytest.approx(1.0)

    def test_interior_points_do_not_change_the_area(self):
        square = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0], [3.0, 7.0]]
        )
        assert convex_hull_area(square) == pytest.approx(100.0)

    def test_a_triangle(self):
        triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
        assert convex_hull_area(triangle) == pytest.approx(6.0)

    def test_collinear_points_enclose_nothing(self):
        line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert convex_hull_area(line) == 0.0

    def test_too_few_points_enclose_nothing(self):
        assert convex_hull_area(np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0
        assert convex_hull_area(np.empty((0, 2))) == 0.0

    def test_repeated_points_are_not_counted_twice(self):
        square = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0], [2.0, 0.0]]
        )
        assert convex_hull_area(square) == pytest.approx(4.0)

    def test_the_area_does_not_depend_on_point_order(self):
        rng = np.random.default_rng(1)
        points = rng.uniform(0, 50, size=(12, 2))
        shuffled = points[rng.permutation(len(points))]
        assert convex_hull_area(points) == pytest.approx(convex_hull_area(shuffled))


class TestOrderingAndFormatting:
    """What the dialog ends up showing."""

    def test_errors_are_listed_before_warnings(self):
        points = spread_points(4)
        points[3] = points[0]  # duplicate -> error, and 4 points -> warning
        issues = check_control_points(points, points, image_shape=(1000, 1000))

        severities = [issue.severity for issue in issues]
        assert Severity.ERROR in severities
        assert Severity.WARNING in severities
        first_warning = severities.index(Severity.WARNING)
        assert all(s is Severity.ERROR for s in severities[:first_warning])

    def test_has_errors_is_false_for_warnings_only(self):
        points = spread_points(4)
        issues = check_control_points(points, points)
        assert issues
        assert not has_errors(issues)

    def test_has_errors_is_false_for_an_empty_list(self):
        assert not has_errors([])

    def test_format_lists_every_message(self):
        issues = [
            Issue(Severity.ERROR, "a", "first problem"),
            Issue(Severity.WARNING, "b", "second problem"),
        ]
        text = format_issues(issues)
        assert "first problem" in text
        assert "second problem" in text

    def test_format_of_nothing_is_empty(self):
        assert format_issues([]) == ""

    def test_good_points_produce_no_issues_at_all(self):
        points = np.array(
            [
                [50.0, 50.0],
                [450.0, 60.0],
                [60.0, 440.0],
                [440.0, 450.0],
                [250.0, 120.0],
                [120.0, 260.0],
                [380.0, 300.0],
            ]
        )
        assert check_control_points(points, points, image_shape=(500, 500)) == []


class TestAgainstRealEstimation:
    """The checks have to agree with what the solver actually does.

    A warning the solver turns out to reject, or an error it would have
    accepted, is worse than no check at all: it teaches people to distrust the
    messages. These pin the two together.

    This is not hypothetical. The solver used to rely on ``np.linalg.solve``
    raising ``LinAlgError`` for a singular system, and whether it does depends
    on the LAPACK build -- macOS raised, the Linux and Windows CI runners
    silently returned a garbage transform for the very same collinear points.
    """

    #: (name, source points, destination points, expected issue code).
    #: Every one of these must be an error here *and* raise in the solver.
    DEGENERATE: ClassVar[list[tuple[str, str, str, str]]] = [
        ("duplicate source", "dup_src", "good", "duplicate_source_points"),
        ("duplicate destination", "good", "dup_dst", "duplicate_destination_points"),
        ("collinear destination", "good", "line", "collinear_destination_points"),
        ("collinear source", "line", "good", "collinear_source_points"),
        ("both collinear", "line", "line2", "collinear_destination_points"),
    ]

    @staticmethod
    def _points(name):
        good = np.array(
            [
                [20.0, 20.0],
                [80.0, 25.0],
                [25.0, 80.0],
                [78.0, 82.0],
                [50.0, 40.0],
                [40.0, 60.0],
            ]
        )
        line = np.array([[float(i) * 10, float(i) * 10] for i in range(6)])

        if name == "good":
            return good
        if name == "line":
            return line
        if name == "line2":
            return line + 1.0
        duplicated = good.copy()
        duplicated[3] = duplicated[1]
        return duplicated

    @pytest.mark.parametrize(
        ("name", "src_name", "dst_name", "code"),
        DEGENERATE,
        ids=[case[0] for case in DEGENERATE],
    )
    def test_degenerate_points_are_an_error_and_really_do_break_the_fit(
        self, name, src_name, dst_name, code
    ):
        from tpsreg.tps import ThinPlateSplineTransform

        src = self._points(src_name)
        dst = self._points(dst_name)

        assert issue_with(check_control_points(src, dst), code).is_error

        with pytest.raises(ValueError):
            ThinPlateSplineTransform().estimate(src, dst, (100, 100))

    def test_the_solver_rejects_collinear_points_without_help_from_lapack(self):
        """The rejection must not depend on the LAPACK build.

        Proven by checking the points are refused before any solve happens: a
        solve that never runs cannot depend on how it would have behaved.
        """
        from tpsreg.tps import ThinPlateSplineTransform

        line = self._points("line")

        def explode(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the solver reached np.linalg.solve")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(np.linalg, "solve", explode)
            with pytest.raises(ValueError, match="collinear"):
                ThinPlateSplineTransform().estimate(line, line + 1.0, (100, 100))

    def test_a_silently_bad_solve_is_caught(self):
        """The backstop for ill-conditioning the geometry checks do not name.

        If some LAPACK returns a garbage answer instead of raising, the
        substituted-back residual catches it.
        """
        from tpsreg.tps import ThinPlateSplineTransform

        good = self._points("good")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                np.linalg, "solve", lambda a, b: np.zeros((a.shape[0], b.shape[1]))
            )
            with pytest.raises(ValueError, match=r"residual|not finite"):
                ThinPlateSplineTransform().estimate(good, good + 2.0, (50, 50))

    def test_a_non_finite_solve_is_caught(self):
        from tpsreg.tps import ThinPlateSplineTransform

        good = self._points("good")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                np.linalg,
                "solve",
                lambda a, b: np.full((a.shape[0], b.shape[1]), np.nan),
            )
            with pytest.raises(ValueError, match="not finite"):
                ThinPlateSplineTransform().estimate(good, good + 2.0, (50, 50))

    def test_the_residual_check_does_not_fire_on_a_large_honest_problem(self):
        """It must not reject the big stitched images this tool is for.

        The system matrix gets badly conditioned as points and coordinates
        grow, but LU still produces a small backward error, which is exactly
        why the residual is the right thing to measure.
        """
        from tpsreg.tps import ThinPlateSplineTransform

        rng = np.random.default_rng(0)
        src = rng.uniform(0, 20000, size=(200, 2))
        dst = src + rng.normal(0, 40, size=src.shape)

        transform = ThinPlateSplineTransform()
        transform.estimate(src, dst, (64, 64))
        assert np.isfinite(transform.params).all()

    def test_points_the_checks_accept_really_do_estimate(self):
        from tpsreg.tps import ThinPlateSplineTransform

        src = np.array(
            [
                [20.0, 20.0],
                [80.0, 25.0],
                [25.0, 80.0],
                [78.0, 82.0],
                [50.0, 40.0],
                [40.0, 60.0],
            ]
        )
        dst = src + 2.0

        assert check_control_points(src, dst, image_shape=(100, 100)) == []

        transform = ThinPlateSplineTransform()
        transform.estimate(src, dst, (100, 100))
        assert transform.params is not None
