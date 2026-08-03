"""Checks on the vendored MatchAnything/RoMa tree.

The tree was pruned to the inference path only. These tests re-derive the
reachable set from the import graph, so if someone re-vendors from upstream or
prunes further, a mismatch shows up here rather than as an ImportError the
first time a user clicks "Auto detect".
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePath, PureWindowsPath

import pytest

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "src/tpsreg/Matchanything"
ROMA_ROOT = VENDOR_ROOT / "third_party" / "ROMA"

#: The modules tpsreg.roma_matcher loads, directly or via yacs config merge.
ENTRY_POINTS = [
    VENDOR_ROOT / "__init__.py",
    VENDOR_ROOT / "src/lightning/lightning_loftr.py",
    VENDOR_ROOT / "src/config/default.py",
    VENDOR_ROOT / "configs/models/roma_model.py",
]

pytestmark = pytest.mark.skipif(
    not VENDOR_ROOT.is_dir(), reason="vendored tree is not present"
)


def _candidates(dotted: str) -> list[Path]:
    """Map an absolute dotted import onto files in the vendored tree."""
    parts = dotted.split(".")
    out: list[Path] = []

    if parts[:2] == ["tpsreg", "Matchanything"]:
        rel = parts[2:]
        out += [
            VENDOR_ROOT.joinpath(*rel).with_suffix(".py"),
            VENDOR_ROOT.joinpath(*rel, "__init__.py"),
        ]
    if parts[0] in {"roma", "experiments", "third_party"}:
        out += [
            ROMA_ROOT.joinpath(*parts).with_suffix(".py"),
            ROMA_ROOT.joinpath(*parts, "__init__.py"),
        ]
    if parts[0] in {"src", "configs"}:
        out += [
            VENDOR_ROOT.joinpath(*parts).with_suffix(".py"),
            VENDOR_ROOT.joinpath(*parts, "__init__.py"),
        ]

    return [p for p in out if p.is_file()]


def _relative(path: Path, node: ast.ImportFrom) -> list[Path]:
    """Resolve a relative import against the importing file's location."""
    base = path.parent
    for _ in range(node.level - 1):
        base = base.parent

    out: list[Path] = []
    prefix = node.module.split(".") if node.module else []
    if prefix:
        target = base.joinpath(*prefix)
        out += [target.with_suffix(".py"), target / "__init__.py"]
    for alias in node.names:
        target = base.joinpath(*prefix, alias.name)
        out += [target.with_suffix(".py"), target / "__init__.py"]

    return [p for p in out if p.is_file()]


def _import_key(relative_path: PurePath, module: str) -> str:
    """Build the stable identifier for an unresolved import.

    Uses ``as_posix()`` so the key is identical on every platform. Plain
    ``str(Path)`` renders backslashes on Windows, which silently broke the
    comparison against the expected set below.
    """
    return f"{relative_path.as_posix()}: {module}"


def _targets(path: Path, node: ast.AST) -> list[Path]:
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return _relative(path, node)
        if not node.module:
            return []
        found = _candidates(node.module)
        # `from pkg import submodule`
        for alias in node.names:
            found += _candidates(f"{node.module}.{alias.name}")
        return found
    if isinstance(node, ast.Import):
        found = []
        for alias in node.names:
            found += _candidates(alias.name)
        return found
    return []


def reachable_modules() -> set[Path]:
    """Every vendored file reachable from the entry points."""
    seen: set[Path] = set()
    queue = [p.resolve() for p in ENTRY_POINTS if p.is_file()]

    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)

        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            queue += [t.resolve() for t in _targets(path, node)]

        # Importing pkg.sub also executes every ancestor pkg/__init__.py.
        parent = path.parent
        while parent == VENDOR_ROOT or VENDOR_ROOT in parent.parents:
            init = parent / "__init__.py"
            if init.is_file():
                queue.append(init.resolve())
            parent = parent.parent

    return seen


class TestImportKey:
    """The identifier used to compare unresolved imports."""

    def test_posix_and_windows_paths_agree(self):
        """The key must not depend on the host separator.

        This is the bug that failed every Windows CI job: the key was built
        with str(Path), so Windows produced backslashes and never matched the
        expected set. PureWindowsPath lets us prove the fix from any platform.
        """
        posix = _import_key(
            PurePath("src/loftr/utils/coarse_matching.py"), ".superglue"
        )
        windows = _import_key(
            PureWindowsPath(r"src\loftr\utils\coarse_matching.py"), ".superglue"
        )

        assert posix == windows
        assert posix == "src/loftr/utils/coarse_matching.py: .superglue"

    def test_key_never_contains_a_backslash(self):
        key = _import_key(PureWindowsPath(r"a\b\c.py"), ".mod")
        assert "\\" not in key


class TestReachability:
    """The tree contains exactly what the inference path needs."""

    def test_entry_points_exist(self):
        missing = [p for p in ENTRY_POINTS if not p.is_file()]
        assert not missing, f"vendored entry points are missing: {missing}"

    def test_every_import_resolves(self):
        """A pruned-away module would surface as an unresolvable import."""
        unresolved = []

        for path in reachable_modules():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.level:
                    continue
                # Relative imports can only point inside the vendored tree, so
                # any that fails to resolve names a file that is really gone.
                if not _relative(path, node):
                    module = "." * node.level + (node.module or "")
                    unresolved.append(
                        _import_key(path.relative_to(VENDOR_ROOT), module)
                    )

        # Upstream ships this one dangling import behind a try/except for a
        # match type this project does not use.
        expected = {"src/loftr/utils/coarse_matching.py: .superglue"}
        assert set(unresolved) <= expected, (
            f"unresolvable relative imports in the vendored tree: "
            f"{sorted(set(unresolved) - expected)}"
        )

    def test_no_unreachable_python_files_remain(self):
        """The tree should stay pruned; dead code invites confusion."""
        reachable = reachable_modules()
        all_py = {p.resolve() for p in VENDOR_ROOT.rglob("*.py")}
        orphans = sorted(p.relative_to(VENDOR_ROOT) for p in all_py - reachable)
        assert not orphans, (
            "vendored files unreachable from the inference path; prune them or "
            f"add an entry point: {orphans}"
        )

    def test_every_vendored_file_compiles(self):
        """Catches a truncated or half-deleted file."""
        broken = []
        for path in VENDOR_ROOT.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError as exc:
                broken.append(f"{path.relative_to(VENDOR_ROOT)}: {exc}")
        assert not broken, broken


class TestAttribution:
    """Upstream license and attribution files stay put."""

    @pytest.mark.parametrize(
        "relative_path",
        ["LICENSE", "README.md", "third_party/ROMA/LICENSE"],
    )
    def test_upstream_file_is_retained(self, relative_path):
        path = VENDOR_ROOT / relative_path
        assert path.is_file(), f"upstream attribution file removed: {relative_path}"
        assert path.read_text(encoding="utf-8", errors="ignore").strip()
