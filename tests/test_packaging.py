"""Tests that the package is correctly installed and importable.

These guard the class of breakage that made the project un-installable in the
first place: flat imports that only resolve when the interpreter happens to be
run from inside the source directory, and resources located by walking up from
``__file__``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


class TestImportability:
    """The package imports cleanly, from anywhere."""

    def test_top_level_import(self):
        import tpsreg

        assert tpsreg.__version__

    def test_public_api_is_exported(self):
        import tpsreg

        for name in tpsreg.__all__:
            assert hasattr(tpsreg, name), f"{name} is in __all__ but not defined"

    @pytest.mark.parametrize(
        "module",
        [
            "tpsreg.models",
            "tpsreg.presenter",
            "tpsreg.ransac",
            "tpsreg.resources_util",
            "tpsreg.roma_matcher",
            "tpsreg.tps",
            "tpsreg.warping",
        ],
    )
    def test_module_imports_without_torch_or_tk(self, module):
        """Only tpsreg.GUI may require Tk; nothing may require torch."""
        importlib.import_module(module)

    def test_import_works_from_an_unrelated_directory(self, tmp_path):
        """Flat imports used to make this fail outside src/tpsreg/."""
        result = subprocess.run(
            [sys.executable, "-c", "import tpsreg; print(tpsreg.__version__)"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()

    def test_roma_matcher_imports_without_torch_installed(self):
        """The optional extra must not be needed just to import the module."""
        from tpsreg import roma_matcher

        assert hasattr(roma_matcher, "create_matcher")
        assert hasattr(roma_matcher, "select_device")


class TestPackagedResources:
    """Theme files and the icon ship inside the package."""

    def test_resources_directory_is_inside_the_package(self):
        from tpsreg import resources_util

        assert resources_util.RESOURCES_PATH.is_dir()
        # It must live under the package, not two levels above it.
        assert resources_util.RESOURCES_PATH.parent.name == "tpsreg"

    @pytest.mark.parametrize("style", ["dark", "light"])
    def test_theme_files_are_present(self, style):
        from tpsreg.resources_util import theme_path

        path = theme_path(style)
        assert path.exists()
        assert path.read_text(encoding="utf-8").lstrip().startswith("#")

    def test_theme_image_directories_are_present(self):
        """The .tcl files glob for PNGs next to themselves."""
        from tpsreg.resources_util import RESOURCES_PATH

        for style in ("dark", "light"):
            image_dir = RESOURCES_PATH / "theme" / style
            assert image_dir.is_dir()
            assert list(image_dir.glob("*.png")), f"no images for {style} theme"

    def test_unknown_theme_rejected(self):
        from tpsreg.resources_util import theme_path

        with pytest.raises(ValueError, match="Unknown theme style"):
            theme_path("solarized")

    def test_icon_is_packaged(self):
        from tpsreg.resources_util import ICON_PATH

        assert ICON_PATH.exists()


class TestEntryPoint:
    """The console script is registered and responds."""

    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "tpsreg.GUI", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Importing tpsreg.GUI needs Tk; skip where it is unavailable.
        if "No module named" in result.stderr and "tkinter" in result.stderr:
            pytest.skip("tkinter is not available in this environment")

        assert result.returncode == 0, result.stderr
        assert "tpsreg" in result.stdout

    def test_console_script_is_declared(self):
        from importlib.metadata import entry_points

        scripts = entry_points(group="console_scripts")
        names = {ep.name: ep.value for ep in scripts}
        assert names.get("tpsreg") == "tpsreg.cli:main"

    def test_launcher_imports_without_tk(self):
        """The launcher must be importable so it can report a missing Tk."""
        from tpsreg import cli

        assert callable(cli.main)

    def test_missing_tk_gives_installation_advice(self, monkeypatch, capsys):
        """A missing Tk should be actionable, not an import traceback."""
        import builtins

        from tpsreg import cli

        real_import = builtins.__import__

        def no_tkinter(name, *args, **kwargs):
            if name == "tkinter":
                raise ImportError("No module named 'tkinter'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_tkinter)

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        message = capsys.readouterr().err
        assert "Tk" in message
        # It must name a concrete install command, not just state the problem.
        assert "install" in message.lower()

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_platform_advice_is_available_everywhere(self, monkeypatch, platform):
        from tpsreg import cli

        monkeypatch.setattr(sys, "platform", platform)
        assert cli._tk_platform_help().strip()


class TestLogFileLocation:
    """Logs go to a user data directory, not the working directory."""

    def test_log_path_is_not_in_the_cwd(self):
        pytest.importorskip("tkinter")
        from tpsreg.GUI import log_file_path

        path = log_file_path()
        assert path.is_absolute()
        assert path.name.endswith(".log")
        assert path.parent.name == "tpsreg"
