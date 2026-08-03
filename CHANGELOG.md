# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-03

The project is now an installable Python package with a test suite and CI.

### Added

- `pip install`-able package with a `tpsreg` console script and
  `python -m tpsreg` entry point.
- Test suite of 209 tests covering the transform, RANSAC, data models, the
  presenter, packaging, and end-to-end loading of the bundled demo data.
- GitHub Actions CI: lint, a test matrix over Ubuntu/macOS/Windows and Python
  3.11–3.13, a job asserting the core install stays torch-free, and a build job
  that verifies packaged resources are present in the wheel.
- Release workflow that builds the wheel and sdist on a version tag, verifies
  the tag matches the packaged version, installs the wheel into a clean
  environment and smoke tests it, and attaches the artifacts to a GitHub
  release with installation instructions. Distribution is via GitHub Releases,
  not PyPI.
- Ruff lint and format configuration, plus pre-commit hooks.
- Optional dependency extras: `accelerated` (kornia/torchvision) and
  `matchanything` (the pretrained detection model).
- Automatic device selection for MatchAnything: CUDA, Apple Silicon MPS, or CPU.
- `progress_callback` on `ThinPlateSplineTransform.estimate`, so a GUI can
  report progress during the bending computation.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `NOTICE.md`, `CITATION.cff`, and issue and
  pull request templates.

### Changed

- **Breaking:** `Point.to_array()` now takes an explicit `include_slice` flag
  instead of inferring the output shape from `slice_idx`.
- Core dependencies no longer include torch. Loading data, placing points,
  estimating transforms, previewing and exporting all work without it; CLAHE and
  resizing fall back to scikit-image implementations.
- Torch is no longer pinned to a CUDA-only package index, so installation works
  on macOS and CPU-only machines.
- The Tk theme and application icon ship inside the package instead of being
  located by walking up from `__file__`.
- Logs are written to a per-user state directory rather than the current working
  directory, and rotate at 2 MB.
- A missing Tk installation now produces platform-specific installation advice
  instead of an import traceback.
- Automatic detection that finds no matches reports this to the user rather than
  returning silently.
- Control points outside the image are rejected when placed, rather than failing
  later during transform estimation.
- Minimum supported Python is 3.11.

### Fixed

- Undo and redo did nothing for manually placed control points: the presenter
  wrote directly to the point set, bypassing the undo history entirely.
- `PointManager.undo()` stepped back two states instead of one, and `redo()`
  could never return to the most recent state.
- `Point.to_array()` returned a two-element array for points on slice 0 and a
  three-element array otherwise, corrupting 3D stack registration.
- `PointSet.remove_point()` reported success when nothing was removed.
- **File → Export transform** always failed, due to a mismatched keyword
  argument.
- `PointAutoIdentifier.detect_points()` raised `TypeError: got multiple values
  for keyword argument 'checkpoint_path'` whenever a caller supplied one.
- `estimate_transform_stack()` crashed with `TypeError: 'NoneType' object cannot
  be interpreted as an integer` when `n_slices` was omitted.
- Constant-valued images produced NaN, and then garbage uint8, in image loading,
  normalization, and SIFT preprocessing.
- Stack warping assigned the wrong parameters to the slice preceding each keyed
  slice, due to an off-by-one in the interpolation interval.
- `deformable_ransac_filter()` reseeded the caller's global NumPy random state.
- The scikit-image RANSAC wrappers returned `None` instead of reporting failure
  when no model could be fitted.
- `ThinPlateSplineTransform` raised a bare `LinAlgError` on collinear control
  points, and `IndexError` when queried outside the estimated grid.
- MatchAnything required an NVIDIA GPU: `.cuda()` and CUDA autocast were
  unconditional, and the model config was resolved relative to the current
  working directory.

### Removed

- The vendored CroCo and DUSt3R subtrees under
  `third_party/ROMA/roma/models/`, 47 files licensed CC BY-NC-SA 4.0
  (non-commercial use only). Neither is reachable from the code path this
  project uses, and their presence would have prevented redistribution under
  the project's MIT license. The whole repository is now permissively licensed.
  A release check fails the build if such files reappear.
- The rest of the unreachable vendored code: training scripts, benchmarks,
  dataset builders, COLMAP helpers, evaluation tools, demos, notebooks, the
  alternate ELoFTR config, and upstream requirements files. The vendored tree
  is now the inference path only — 64 Python modules, down from 175, and
  2.8 MB down to 820 KB. Determined by walking the import graph from the entry
  points the project actually loads; the reachable set is byte-for-byte
  identical before and after.
- `src/tpsreg/image_texture.py`, which opened matplotlib windows at import time
  and referenced hardcoded local dataset paths. The reusable function moved to
  `scripts/texture_analysis.py`.
- The `__main__` demo blocks in `tps.py` and `ransac.py`, one of which performed
  arithmetic on a boolean return value.
- `uv.lock`, which was stale and pinned torch to a CUDA-only index. Regenerate
  it with `uv lock` if you want one.

## [0.1.0]

Initial version.
