# Contributing

Thanks for your interest in improving this tool. Bug reports from people using
it on real microscopy data are especially valuable.

## Reporting bugs

Please [open an issue](https://github.com/lambjames18/tps-image-registration-gui/issues)
and include:

- What you were doing and what happened instead
- Your OS and the output of `python --version` and `pip show tpsreg`
- The data format involved (`.ang`, `.dream3d`, TIFF stack, ...) and whether
  it was 2D or 3D
- The traceback, if there was one, and the relevant part of the log file

The log file lives at:

| Platform | Location |
|---|---|
| Linux | `~/.local/state/tpsreg/tpsreg.log` |
| macOS | `~/Library/Logs/tpsreg/tpsreg.log` |
| Windows | `%LOCALAPPDATA%\tpsreg\tpsreg.log` |

Please do not attach proprietary or unpublished data. A cropped or synthetic
example that reproduces the problem is ideal.

## Development setup

```bash
git clone https://github.com/lambjames18/tps-image-registration-gui.git
cd tps-image-registration-gui
pip install -e ".[dev]"
pre-commit install
```

On Linux you also need Tk for the GUI: `sudo apt install python3-tk`.

## Running the checks

These are the same checks CI runs:

```bash
pytest                                  # the test suite
ruff check src/ tests/ scripts/         # lint
ruff format src/ tests/ scripts/        # format
```

Useful subsets:

```bash
pytest tests/test_tps.py -v             # one module
pytest -m "not slow"                    # skip the slower tests
pytest -m "not gui"                     # skip tests that need a display
pytest --cov=tpsreg --cov-report=term   # with coverage
```

## Badges

The coverage and tests badges are self-hosted, with no external service. On a
push to the default branch, CI runs `scripts/make_badges.py` over the coverage
and JUnit reports and commits two small JSON files to an orphan `badges`
branch; the README points shields.io at them. The badge branch never appears in
the source history.

`src/tpsreg/GUI.py` is excluded from the coverage measurement. Its behaviour is
covered by the smoke tests in `tests/test_gui.py`, but line coverage of widget
construction measures very little, and at ~1400 statements it would dominate
the figure for the logic actually under test. Everything else counts.

## Guidelines

**Keep the core importable without torch.** Loading data, placing points,
estimating transforms and exporting must all work on a plain
`pip install tpsreg`. Torch, kornia and PyTorch Lightning are optional extras;
import them lazily inside the function that needs them and provide a
scikit-image fallback or a clear error. There is a CI job that enforces this.

**Tests must run headless.** The suite runs on CI workers with no display and no
GPU. Drive the presenter through a fake view (see `tests/conftest.py`) rather
than instantiating Tk widgets. Mark anything that genuinely needs a display with
`@pytest.mark.gui`.

**Use lazy logging.** `logger.info("Loaded %s", path)`, not
`logger.info(f"Loaded {path}")`. Ruff enforces this.

**Do not modify `src/tpsreg/Matchanything/`** beyond what is needed to keep its
imports working. It is vendored upstream code and is excluded from linting and
formatting so it stays diffable against upstream.

The tree has been reduced to the inference path: upstream's training scripts,
benchmarks, dataset builders, demos and notebooks were removed because nothing
the project loads reaches them. If you re-vendor from upstream, see the
"Re-vendoring" section of [NOTICE.md](NOTICE.md) for how the keep-set is
determined. If you later add point-assisted fine-tuning, expect to bring back
or write training code deliberately rather than relying on what upstream
happened to ship.

**Prefer real assertions over smoke tests.** A test that only checks something
did not raise catches very little. Assert on shapes, values and error messages.

## Pull requests

1. Branch off `dev`.
2. Make your change, with tests covering it.
3. Confirm `pytest` and `ruff check` pass locally.
4. Open the PR against `dev` and describe what changed and why.

Small, focused pull requests get reviewed faster than large ones.

**Do not re-add the CroCo or DUSt3R subtrees.** Upstream RoMa ships
`roma/models/croco/` and `roma/models/dust3r/` under CC BY-NC-SA 4.0
(non-commercial use only). Neither is reachable from the code path this project
uses, so both were removed; keeping them would prevent the project from being
redistributed under its MIT license. If you re-vendor RoMa from upstream, drop
them again. The release workflow fails the build if a `CC BY-NC-SA` file
reappears under `src/`.

## Releasing

The project is distributed through GitHub Releases, not PyPI. Maintainers only:

1. Update `version` in `pyproject.toml` and add a `CHANGELOG.md` entry.
2. Merge to the default branch and confirm CI is green.
3. Tag and push: `git tag v0.3.0 && git push origin v0.3.0`.

The release workflow then verifies the tag matches the packaged version, builds
the wheel and sdist, installs the wheel into a clean environment and smoke tests
it, checks no non-commercially-licensed code is bundled, and attaches the
artifacts to a GitHub release with installation instructions.

You can run the workflow manually (`workflow_dispatch`) to build and check the
artifacts without tagging.
