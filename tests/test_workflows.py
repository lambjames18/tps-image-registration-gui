"""Structural checks on the GitHub Actions workflows.

A bad merge silently broke CI once: it dropped ``--junitxml`` from the test
command and left two copies of the GUI test steps. Neither is a Python problem,
so nothing in the suite noticed, and the failure only surfaced minutes later in
a different job as "junit.xml does not exist". These tests catch that shape of
damage locally, before it is pushed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is needed to parse workflows")

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"

pytestmark = pytest.mark.skipif(
    not WORKFLOW_DIR.is_dir(), reason="no .github/workflows in this checkout"
)


def workflow_files() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def all_workflows() -> list[tuple[str, dict]]:
    return [(path.name, load(path)) for path in workflow_files()]


def run_commands(job: dict) -> list[str]:
    """Every shell command in a job, joined for substring searching."""
    return [step["run"] for step in job.get("steps", []) if "run" in step]


@pytest.fixture(scope="module")
def ci() -> dict:
    path = WORKFLOW_DIR / "ci.yml"
    if not path.is_file():
        pytest.skip("ci.yml is not present")
    return load(path)


class TestWorkflowsAreWellFormed:
    """Basic structure of every workflow file."""

    def test_at_least_one_workflow_exists(self):
        assert workflow_files()

    @pytest.mark.parametrize("name,workflow", all_workflows())
    def test_parses_and_has_jobs(self, name, workflow):
        assert isinstance(workflow, dict), f"{name} is not a mapping"
        assert workflow.get("jobs"), f"{name} declares no jobs"

    @pytest.mark.parametrize("name,workflow", all_workflows())
    def test_step_names_are_unique_within_a_job(self, name, workflow):
        """Duplicate step names are the fingerprint of a bad merge.

        They also make a job silently do the same work twice.
        """
        for job_id, job in workflow["jobs"].items():
            names = [step["name"] for step in job.get("steps", []) if step.get("name")]
            duplicates = sorted({n for n in names if names.count(n) > 1})
            assert not duplicates, (
                f"{name}: job '{job_id}' has duplicated steps {duplicates}; "
                "this is usually left behind by a merge"
            )

    @pytest.mark.parametrize("name,workflow", all_workflows())
    def test_every_job_declares_a_runner(self, name, workflow):
        for job_id, job in workflow["jobs"].items():
            assert job.get("runs-on"), f"{name}: job '{job_id}' has no runs-on"

    @pytest.mark.parametrize("name,workflow", all_workflows())
    def test_job_dependencies_exist(self, name, workflow):
        jobs = workflow["jobs"]
        for job_id, job in jobs.items():
            needs = job.get("needs") or []
            needs = [needs] if isinstance(needs, str) else needs
            for dependency in needs:
                assert dependency in jobs, (
                    f"{name}: job '{job_id}' needs '{dependency}', which is not defined"
                )


class TestBadgePipeline:
    """The reports the badge job consumes must actually be produced.

    These three steps live in two different jobs, so nothing but a check like
    this ties them together.
    """

    def test_tests_are_run_with_a_junit_report(self, ci):
        commands = run_commands(ci["jobs"]["test"])
        assert any("--junitxml=junit.xml" in command for command in commands), (
            "no test step writes junit.xml; the badges job reads it to count "
            "tests and will fail with 'junit.xml does not exist'"
        )

    def test_tests_are_run_with_an_xml_coverage_report(self, ci):
        commands = run_commands(ci["jobs"]["test"])
        assert any("--cov-report=xml" in command for command in commands), (
            "no test step writes coverage.xml, which the badges job reads"
        )

    def test_every_platform_writes_both_reports(self, ci):
        """The Linux and non-Linux commands are separate; both need the flags."""
        commands = [
            command
            for command in run_commands(ci["jobs"]["test"])
            if '-m "not gui"' in command
        ]
        assert commands, "no non-GUI test command found"
        for command in commands:
            assert "--junitxml=junit.xml" in command, f"missing --junitxml: {command}"
            assert "--cov-report=xml" in command, f"missing --cov-report: {command}"

    def test_upload_step_publishes_both_reports(self, ci):
        uploads = [
            step
            for step in ci["jobs"]["test"]["steps"]
            if str(step.get("uses", "")).startswith("actions/upload-artifact")
        ]
        assert uploads, "the test job uploads nothing"

        paths = "\n".join(str(step.get("with", {}).get("path", "")) for step in uploads)
        assert "coverage.xml" in paths
        assert "junit.xml" in paths

    def test_upload_fails_loudly_on_a_missing_report(self, ci):
        """'warn' uploads a partial artifact and defers the error to another job."""
        for step in ci["jobs"]["test"]["steps"]:
            if not str(step.get("uses", "")).startswith("actions/upload-artifact"):
                continue
            if "junit.xml" not in str(step.get("with", {}).get("path", "")):
                continue
            assert step["with"].get("if-no-files-found") == "error", (
                "the reports upload should fail when a report is missing, so "
                "the error points at the step that produced it"
            )

    def test_badge_job_reads_what_the_test_job_wrote(self, ci):
        badge_commands = "\n".join(run_commands(ci["jobs"]["badges"]))
        assert "make_badges.py" in badge_commands
        assert "coverage.xml" in badge_commands
        assert "junit.xml" in badge_commands

    def test_badge_generator_script_exists(self, ci):
        """The workflow references it by path, so a rename would break CI."""
        badge_commands = "\n".join(run_commands(ci["jobs"]["badges"]))
        assert "scripts/make_badges.py" in badge_commands

        script = Path(__file__).resolve().parents[1] / "scripts" / "make_badges.py"
        assert script.is_file(), f"{script} is referenced by CI but missing"

    def test_badge_job_waits_for_the_tests(self, ci):
        needs = ci["jobs"]["badges"].get("needs")
        needs = [needs] if isinstance(needs, str) else (needs or [])
        assert "test" in needs, "badges must run after test to get its artifact"

    def test_badge_job_can_write(self, ci):
        """Pushing the badges branch needs contents: write."""
        permissions = ci["jobs"]["badges"].get("permissions", {})
        assert permissions.get("contents") == "write"

    def test_badge_job_is_limited_to_the_default_branch(self, ci):
        """A pull request must not be able to rewrite the published badges."""
        condition = ci["jobs"]["badges"].get("if", "")
        assert "default_branch" in condition
        assert "push" in condition


class TestGuiTestIsolation:
    """The GUI tests run in their own process; a Tk segfault must not take
    the rest of the suite down with it."""

    def test_non_gui_and_gui_runs_are_separate(self, ci):
        commands = run_commands(ci["jobs"]["test"])
        assert any('-m "not gui"' in command for command in commands)
        assert any("-m gui" in command for command in commands)

    def test_gui_run_is_not_duplicated(self, ci):
        """The merge that broke CI also left two copies of these steps."""
        gui_commands = [
            command
            for command in run_commands(ci["jobs"]["test"])
            if "-m gui" in command and "not gui" not in command
        ]
        # One for Linux (under Xvfb) and one for everything else.
        assert len(gui_commands) == 2, (
            f"expected exactly 2 GUI test commands, found {len(gui_commands)}: "
            f"{gui_commands}"
        )
