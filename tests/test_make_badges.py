"""Tests for the badge generator.

The badges are what people see first on the README, so a silently wrong number
is worse than no badge. These cover the parsing and the colour thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_badges import (
    build_coverage_badge,
    build_tests_badge,
    coverage_colour,
    main,
    read_coverage,
    read_test_counts,
)

COBERTURA = """<?xml version="1.0" ?>
<coverage line-rate="{rate}" lines-covered="{covered}" lines-valid="{valid}">
  <packages/>
</coverage>
"""

JUNIT_BARE = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="{tests}" failures="{failures}"
           errors="{errors}" skipped="{skipped}"/>
"""

JUNIT_WRAPPED = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="a" tests="10" failures="1" errors="0" skipped="2"/>
  <testsuite name="b" tests="5" failures="0" errors="1" skipped="0"/>
</testsuites>
"""


def write_coverage(tmp_path: Path, covered: int, valid: int) -> Path:
    path = tmp_path / "coverage.xml"
    rate = covered / valid if valid else 0
    path.write_text(COBERTURA.format(rate=rate, covered=covered, valid=valid))
    return path


def write_junit(tmp_path: Path, tests=10, failures=0, errors=0, skipped=0) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(
        JUNIT_BARE.format(
            tests=tests, failures=failures, errors=errors, skipped=skipped
        )
    )
    return path


class TestReadCoverage:
    """Parsing the Cobertura report."""

    def test_uses_exact_line_counts(self, tmp_path):
        path = write_coverage(tmp_path, covered=1385, valid=2037)
        assert read_coverage(path) == pytest.approx(67.99, abs=0.01)

    def test_full_coverage(self, tmp_path):
        assert read_coverage(write_coverage(tmp_path, 50, 50)) == pytest.approx(100.0)

    def test_zero_coverage(self, tmp_path):
        assert read_coverage(write_coverage(tmp_path, 0, 50)) == pytest.approx(0.0)

    def test_prefers_counts_over_the_rounded_rate(self, tmp_path):
        """line-rate is rounded to 4dp; the counts are exact."""
        path = tmp_path / "coverage.xml"
        # A deliberately wrong rate: the counts must win.
        path.write_text(COBERTURA.format(rate=0.1, covered=3, valid=4))
        assert read_coverage(path) == pytest.approx(75.0)

    def test_falls_back_to_line_rate(self, tmp_path):
        path = tmp_path / "coverage.xml"
        path.write_text('<coverage line-rate="0.8235"><packages/></coverage>')
        assert read_coverage(path) == pytest.approx(82.35)

    def test_zero_valid_lines_falls_back(self, tmp_path):
        """An empty report must not divide by zero."""
        path = tmp_path / "coverage.xml"
        path.write_text('<coverage line-rate="0" lines-covered="0" lines-valid="0"/>')
        assert read_coverage(path) == pytest.approx(0.0)

    def test_unusable_report_raises(self, tmp_path):
        path = tmp_path / "coverage.xml"
        path.write_text("<coverage/>")
        with pytest.raises(ValueError, match="line-rate"):
            read_coverage(path)


class TestReadTestCounts:
    """Parsing the JUnit report."""

    def test_bare_testsuite(self, tmp_path):
        path = write_junit(tmp_path, tests=100, failures=2, errors=1, skipped=5)
        counts = read_test_counts(path)
        assert counts["total"] == 100
        assert counts["passed"] == 92
        assert counts["failures"] == 2
        assert counts["errors"] == 1
        assert counts["skipped"] == 5

    def test_wrapped_testsuites_are_summed(self, tmp_path):
        path = tmp_path / "junit.xml"
        path.write_text(JUNIT_WRAPPED)
        counts = read_test_counts(path)
        assert counts["total"] == 15
        assert counts["failures"] == 1
        assert counts["errors"] == 1
        assert counts["skipped"] == 2
        assert counts["passed"] == 11

    def test_missing_attributes_default_to_zero(self, tmp_path):
        path = tmp_path / "junit.xml"
        path.write_text('<testsuite name="pytest" tests="7"/>')
        counts = read_test_counts(path)
        assert counts["total"] == 7
        assert counts["passed"] == 7

    def test_report_without_a_suite_raises(self, tmp_path):
        path = tmp_path / "junit.xml"
        path.write_text("<testsuites/>")
        with pytest.raises(ValueError, match="no <testsuite>"):
            read_test_counts(path)


class TestColours:
    """Colour thresholds."""

    @pytest.mark.parametrize(
        ("percent", "expected"),
        [
            (100.0, "brightgreen"),
            (90.0, "brightgreen"),
            (89.9, "green"),
            (80.0, "green"),
            (79.9, "yellowgreen"),
            (70.0, "yellowgreen"),
            (69.9, "yellow"),
            (60.0, "yellow"),
            (59.9, "orange"),
            (50.0, "orange"),
            (49.9, "red"),
            (0.0, "red"),
        ],
    )
    def test_thresholds(self, percent, expected):
        assert coverage_colour(percent) == expected

    def test_thresholds_are_monotonic(self):
        """Higher coverage must never produce a worse-looking colour."""
        order = ["red", "orange", "yellow", "yellowgreen", "green", "brightgreen"]
        ranks = [coverage_colour(p / 10) for p in range(0, 1001)]
        indices = [order.index(name) for name in ranks]
        assert indices == sorted(indices)


class TestBadgePayloads:
    """The JSON shields.io consumes."""

    def test_coverage_payload_shape(self):
        payload = build_coverage_badge(67.99)
        assert payload["schemaVersion"] == 1
        assert payload["label"] == "coverage"
        assert payload["message"] == "68.0%"
        assert payload["color"] == "yellow"

    def test_coverage_message_has_one_decimal(self):
        assert build_coverage_badge(100.0)["message"] == "100.0%"
        assert build_coverage_badge(7.25)["message"] == "7.2%"

    def test_tests_payload_when_green(self):
        payload = build_tests_badge(
            {"total": 50, "passed": 48, "failures": 0, "errors": 0, "skipped": 2}
        )
        assert payload["message"] == "48 passed"
        assert payload["color"] == "brightgreen"

    def test_tests_payload_counts_failures(self):
        payload = build_tests_badge(
            {"total": 50, "passed": 45, "failures": 3, "errors": 2, "skipped": 0}
        )
        assert payload["message"] == "45 passed, 5 failed"
        assert payload["color"] == "red"

    def test_errors_alone_still_read_as_failing(self):
        payload = build_tests_badge(
            {"total": 10, "passed": 9, "failures": 0, "errors": 1, "skipped": 0}
        )
        assert payload["color"] == "red"


class TestCommandLine:
    """End to end through main()."""

    def test_writes_both_badges(self, tmp_path, capsys):
        coverage = write_coverage(tmp_path, covered=680, valid=1000)
        junit = write_junit(tmp_path, tests=100, skipped=4)
        out_dir = tmp_path / "badges"

        exit_code = main(
            [
                "--coverage",
                str(coverage),
                "--junit",
                str(junit),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 0

        coverage_payload = json.loads((out_dir / "coverage.json").read_text())
        tests_payload = json.loads((out_dir / "tests.json").read_text())

        assert coverage_payload["message"] == "68.0%"
        assert tests_payload["message"] == "96 passed"

        # The summary is what a maintainer reads in the workflow log.
        assert "coverage: 68.0%" in capsys.readouterr().out

    def test_creates_the_output_directory(self, tmp_path):
        coverage = write_coverage(tmp_path, 1, 2)
        junit = write_junit(tmp_path)
        out_dir = tmp_path / "deeply" / "nested"

        assert (
            main(
                [
                    "--coverage",
                    str(coverage),
                    "--junit",
                    str(junit),
                    "--output-dir",
                    str(out_dir),
                ]
            )
            == 0
        )
        assert (out_dir / "coverage.json").is_file()

    def test_missing_input_reports_an_error(self, tmp_path, capsys):
        junit = write_junit(tmp_path)
        exit_code = main(
            ["--coverage", str(tmp_path / "absent.xml"), "--junit", str(junit)]
        )
        assert exit_code == 1
        assert "does not exist" in capsys.readouterr().err

    def test_payloads_are_valid_shields_endpoints(self, tmp_path):
        """shields.io rejects a payload missing schemaVersion/label/message."""
        coverage = write_coverage(tmp_path, 900, 1000)
        junit = write_junit(tmp_path, tests=10)
        out_dir = tmp_path / "badges"

        main(
            [
                "--coverage",
                str(coverage),
                "--junit",
                str(junit),
                "--output-dir",
                str(out_dir),
            ]
        )

        for name in ("coverage", "tests"):
            payload = json.loads((out_dir / f"{name}.json").read_text())
            assert payload["schemaVersion"] == 1
            assert isinstance(payload["label"], str) and payload["label"]
            assert isinstance(payload["message"], str) and payload["message"]
            assert isinstance(payload["color"], str) and payload["color"]
