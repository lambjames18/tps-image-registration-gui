"""Tests for the pre-estimation control point checks.

No Tk and no image data: these are pure geometry, which is the whole point of
keeping them out of the view.
"""

from __future__ import annotations

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

    def test_duplicate_destination_points_are_only_a_warning(self):
        """Solvable, just contradictory: two features sent to one spot."""
        points = spread_points(6)
        points[2] = points[0]
        issues = check_control_points(spread_points(6), points)
        issue = issue_with(issues, "duplicate_destination_points")
        assert issue.severity is Severity.WARNING
        assert not has_errors(issues)

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
    """The checks have to agree with what the solver actually does."""

    def test_points_flagged_as_duplicates_really_do_break_the_fit(self):
        """If this stops being true, the error is misleading and must change."""
        from tpsreg.tps import ThinPlateSplineTransform

        points = spread_points(6)
        points[3] = points[1]

        issues = check_control_points(points, spread_points(6))
        assert issue_with(issues, "duplicate_source_points").is_error

        with pytest.raises(ValueError, match="duplicate"):
            transform = ThinPlateSplineTransform()
            transform.estimate(points, spread_points(6), (100, 100))

    def test_points_flagged_as_collinear_really_do_break_the_fit(self):
        from tpsreg.tps import ThinPlateSplineTransform

        line = np.array([[float(i) * 10, float(i) * 10] for i in range(8)])

        issues = check_control_points(line, line + 1.0)
        assert issue_with(issues, "collinear_source_points").is_error

        with pytest.raises(ValueError, match=r"collinear|solve"):
            transform = ThinPlateSplineTransform()
            transform.estimate(line, line + 1.0, (100, 100))

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
