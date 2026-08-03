"""Generate shields.io endpoint JSON for the README badges.

The badges are self-hosted: CI runs this over the coverage and JUnit reports
produced by the test job, then commits the resulting JSON to a dedicated
``badges`` branch. shields.io reads it from raw.githubusercontent.com. No
external account or token is involved beyond the workflow's own GITHUB_TOKEN.

Usage::

    python scripts/make_badges.py --coverage coverage.xml --junit junit.xml \\
        --output-dir badges
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

#: shields.io endpoint schema version. Fixed by shields, not by us.
SCHEMA_VERSION = 1

#: Coverage thresholds, highest first, mapped to shields colour names.
COVERAGE_COLOURS: tuple[tuple[float, str], ...] = (
    (90.0, "brightgreen"),
    (80.0, "green"),
    (70.0, "yellowgreen"),
    (60.0, "yellow"),
    (50.0, "orange"),
    (0.0, "red"),
)


def coverage_colour(percent: float) -> str:
    """Pick a badge colour for a coverage percentage."""
    for threshold, colour in COVERAGE_COLOURS:
        if percent >= threshold:
            return colour
    return "red"


def read_coverage(path: Path) -> float:
    """Return the line-coverage percentage from a Cobertura XML report.

    Prefers the explicit covered/valid line counts over the ``line-rate``
    attribute, which is rounded to four decimal places.
    """
    root = ET.parse(path).getroot()

    covered = root.get("lines-covered")
    valid = root.get("lines-valid")
    if covered is not None and valid is not None and int(valid) > 0:
        return 100.0 * int(covered) / int(valid)

    line_rate = root.get("line-rate")
    if line_rate is None:
        raise ValueError(f"{path} has neither line counts nor a line-rate attribute")
    return 100.0 * float(line_rate)


def read_test_counts(path: Path) -> dict[str, int]:
    """Return test counts from a JUnit XML report.

    pytest emits either a bare ``<testsuite>`` or a ``<testsuites>`` wrapper
    depending on version and plugins, so handle both and sum the suites.
    """
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise ValueError(f"{path} contains no <testsuite> element")

    def total(attr: str) -> int:
        return sum(int(suite.get(attr) or 0) for suite in suites)

    tests = total("tests")
    failures = total("failures")
    errors = total("errors")
    skipped = total("skipped")

    return {
        "total": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": tests - failures - errors - skipped,
    }


def build_coverage_badge(percent: float) -> dict:
    """Build the shields endpoint payload for the coverage badge."""
    return {
        "schemaVersion": SCHEMA_VERSION,
        "label": "coverage",
        # One decimal place: enough to see movement, not so much it looks noisy.
        "message": f"{percent:.1f}%",
        "color": coverage_colour(percent),
    }


def build_tests_badge(counts: dict[str, int]) -> dict:
    """Build the shields endpoint payload for the tests badge."""
    broken = counts["failures"] + counts["errors"]
    if broken:
        message = f"{counts['passed']} passed, {broken} failed"
        colour = "red"
    else:
        message = f"{counts['passed']} passed"
        colour = "brightgreen"

    return {
        "schemaVersion": SCHEMA_VERSION,
        "label": "tests",
        "message": message,
        "color": colour,
    }


def write_badge(directory: Path, name: str, payload: dict) -> Path:
    """Write one badge payload, returning the path written."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage", type=Path, required=True, help="Cobertura coverage.xml"
    )
    parser.add_argument("--junit", type=Path, required=True, help="JUnit junit.xml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("badges"),
        help="directory to write the badge JSON into",
    )
    args = parser.parse_args(argv)

    for path in (args.coverage, args.junit):
        if not path.is_file():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 1

    percent = read_coverage(args.coverage)
    counts = read_test_counts(args.junit)

    written = [
        write_badge(args.output_dir, "coverage", build_coverage_badge(percent)),
        write_badge(args.output_dir, "tests", build_tests_badge(counts)),
    ]

    print(f"coverage: {percent:.1f}% ({coverage_colour(percent)})")
    print(
        f"tests: {counts['passed']} passed, {counts['failures']} failed, "
        f"{counts['errors']} errors, {counts['skipped']} skipped"
    )
    for path in written:
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
