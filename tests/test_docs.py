"""Tests that keep the documentation honest.

Prose rots more quietly than code: a renamed function or a moved section
leaves a link that still looks fine in a diff and is only discovered by a
reader. These check the parts of the docs that make checkable claims -- links,
images, CLI flags, and the numbers copied out of the source.

What they deliberately do not check is whether the prose is any good.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = [
    REPO / "README.md",
    REPO / "docs" / "user-guide.md",
    REPO / "docs" / "stack-registration.md",
    REPO / "docs" / "api.md",
    REPO / "CONTRIBUTING.md",
]

LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def heading_anchors(path: Path) -> set[str]:
    """GitHub's anchor for each heading: lowercased, punctuation dropped."""
    anchors = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            text = re.sub(r"[^\w\s-]", "", match.group(2).strip().lower())
            anchors.add(re.sub(r"\s+", "-", text))
    return anchors


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
class TestLinks:
    def test_internal_links_resolve(self, doc):
        """A link to a file that does not exist is a broken promise."""
        broken = []
        for _label, target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, _fragment = target.partition("#")
            if path_part and not (doc.parent / path_part).exists():
                broken.append(target)

        assert not broken, f"{doc.name} links to missing files: {broken}"

    def test_section_links_resolve(self, doc):
        """Anchors are what break when a section is renamed or moved."""
        broken = []
        for _label, target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, fragment = target.partition("#")
            if not fragment:
                continue
            destination = (doc.parent / path_part) if path_part else doc
            if (
                destination.suffix == ".md"
                and destination.exists()
                and fragment not in heading_anchors(destination)
            ):
                broken.append(target)

        assert not broken, f"{doc.name} links to missing sections: {broken}"

    def test_images_exist(self, doc):
        missing = [
            target
            for _alt, target in IMAGE.findall(doc.read_text(encoding="utf-8"))
            if not target.startswith("http") and not (doc.parent / target).exists()
        ]
        assert not missing, f"{doc.name} references missing images: {missing}"


class TestStackRegistrationDoc:
    """The CLI guide quotes the parser, so it can disagree with it."""

    @pytest.fixture
    def doc(self):
        return (REPO / "docs" / "stack-registration.md").read_text(encoding="utf-8")

    @pytest.fixture
    def parser(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "register_stack", REPO / "scripts" / "register_stack.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_parser()

    def test_every_flag_is_documented(self, doc, parser):
        """A flag nobody wrote down may as well not exist."""
        documented = set(re.findall(r"`(--[a-z][a-z-]+)", doc))
        real = {
            action.option_strings[0]
            for action in parser._actions
            if action.option_strings and action.dest != "help"
        }
        assert not (real - documented), (
            f"undocumented flags: {sorted(real - documented)}"
        )

    def test_no_flags_are_invented(self, doc, parser):
        """The worse direction: documenting something that does not work."""
        documented = set(re.findall(r"`(--[a-z][a-z-]+)", doc))
        real = {
            option
            for action in parser._actions
            for option in action.option_strings
            if action.dest != "help"
        }
        assert not (documented - real), (
            f"documented but not real: {sorted(documented - real)}"
        )

    def test_the_minimum_match_table_matches_the_code(self, doc):
        """The table is a copy of MINIMUM_MATCHES and can drift from it."""
        from tpsreg.stack_registration import MINIMUM_MATCHES

        for model, minimum in MINIMUM_MATCHES.items():
            assert re.search(rf"\|\s*`{model}`\s*\|\s*{minimum}\s*\|", doc), (
                f"the table does not say {model} needs {minimum} matches"
            )

    def test_every_transform_and_reference_mode_is_mentioned(self, doc):
        from tpsreg.stack_registration import REFERENCE_MODES, TRANSFORM_TYPES

        for name in (*TRANSFORM_TYPES, *REFERENCE_MODES):
            assert f"`{name}`" in doc, f"{name} is not documented"


class TestReadme:
    """The front page makes a few claims worth holding it to."""

    @pytest.fixture
    def readme(self):
        return (REPO / "README.md").read_text(encoding="utf-8")

    def test_it_links_to_every_guide(self, readme):
        for name in ("user-guide.md", "stack-registration.md", "api.md"):
            assert name in readme, f"the README does not link to {name}"

    def test_the_install_command_is_not_version_pinned(self, readme):
        """A hardcoded wheel version goes stale on the next release."""
        assert not re.search(r"pip install tpsreg-\d+\.\d+\.\d+", readme)

    def test_the_entry_point_is_real(self, readme):
        """The README tells people to run `tpsreg`; something must provide it."""
        pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert "tpsreg = " in pyproject
        assert (REPO / "src" / "tpsreg" / "__main__.py").exists(), (
            "the troubleshooting section offers `python -m tpsreg` as a fallback"
        )
