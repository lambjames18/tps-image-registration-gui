# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Slice-by-slice registration of an image stack, as a standalone script:
  `scripts/register_stack.py`. Takes a folder of images, a transform type
  (`translation`, `rigid`, `affine`, `tps`) and an output folder, and does the
  whole stack with the MatchAnything model. Nothing in the GUI is involved and
  no display is needed.

  Each slice can be registered against the previous one, the first, or the
  middle. Sequential is easiest to match — adjacent slices look alike — but its
  transforms compose, so per-pair error accumulates as drift along the stack;
  the fixed-reference modes cannot drift but ask the matcher to relate slices
  that may no longer resemble each other. Composition is done on the coordinate
  mapping rather than by resampling once per link, so every slice is
  interpolated exactly once however long the chain.

  The output folder is a debugging trail, not just images: the registered
  stack, each pair's matches drawn as lines between the two images, each slice
  checkerboarded against its reference, the transforms as `.npy`, a
  `report.json`, a `summary.csv`, and the log. Per pair it records the match
  count, median and maximum residual, displacement, and warnings — too few
  matches for the chosen model, degenerate point geometry, a residual out of
  line with the rest. Slices that moved unlike their neighbours are flagged by
  a median/MAD z-score.

  A pair that fails to match or fit takes the identity and is recorded as
  failed rather than aborting; one bad slice in a hundred should not cost the
  run. Filenames sort numerically, so `slice_10` follows `slice_9` — a plain
  lexicographic sort silently reorders a stack.

  New: `tpsreg.stack_registration`, which takes the matcher as an argument and
  so is usable with any matcher, and testable without torch.

- Fit quality is shown live while placing points, rather than only on demand.
  Each marker carries its leave-one-out residual, is coloured on a scale from
  good to bad, and the status bar shows the median and the worst point.
  Recomputed when a pair is added or removed and when a drag lands -- not
  during the drag, where the numbers would be noise.

  This is affordable because of the closed form added with regularization:
  a single solve rather than one refit per point, measured at 0.2 ms for 25
  control points and 1.5 ms for 100, against 15 ms and 62 ms for the
  refitting version.

  Nothing is shown below nine control points. Leave-one-out asks the
  remaining points to predict the held-out one, and with fewer than that they
  cannot: every residual comes out large and a genuinely bad point does not
  stand out. Showing those numbers while someone places their first few
  points would be alarming and wrong.
- Optional smoothing of the spline (regularization), off by default. Set it
  from the **Smoothing** selector in the top bar: *Off*, *Automatic*, or a
  number.

  Exact interpolation is not the same as an accurate transform. A thin-plate
  spline passes through every control point, which means it also reproduces
  every click error exactly, contorting itself to honour mistakes. Smoothing
  relaxes that, buying a better fit to the real deformation with the slack.

  *Automatic* chooses the strength by leave-one-out cross-validation, which
  matters because the number is in units nobody has an intuition for and the
  right value depends on how noisy the clicks are relative to the deformation.
  On synthetic data with a known answer and 1.5 px of click noise it roughly
  halved the error against the true deformation; with clean points it selects
  zero and changes nothing.

  The cross-validation uses a closed form rather than refitting once per
  point: for a linear smoother the leave-one-out residual is `w_i / M_ii`
  from a single fit. Verified against brute-force refitting to 1e-13, at a
  fourteenth of the cost for 16 control points and better above that.

  The strength is normalised by the kernel magnitude so the same number means
  roughly the same thing at any image size — within about twofold across a
  200-fold change in coordinates, since `r²log(r²)` is not scale-homogeneous.

  New: `tpsreg.tps.loocv_residuals`, `tpsreg.tps.select_regularization`,
  `ThinPlateSplineTransform(regularization=...)`, and
  `ApplicationPresenter.set_regularization`.
- Registration quality metrics (`tpsreg.metrics`), reachable from
  **View → Check registration quality**.

  The obvious check — how far the fit lands from each control point — says
  nothing about a thin-plate spline. It interpolates, so residuals come out
  around 1e-12 whether the correspondences are good or catastrophically
  wrong; a point clicked 40 px from its true partner is indistinguishable
  from a perfect one. What the report gives instead:

  - **Leave-one-out residuals.** Refit without each point and see how far the
    fit misses it, which is where a bad correspondence does show up. Points
    that disagree with the rest are flagged and drawn with a larger
    warning-coloured ring, so a number in a report can be found on a canvas
    holding several dozen points. Needs roughly nine well-spread points to be
    meaningful — below that the sparsity swamps the signal, which is
    documented and tested rather than left to be discovered.
  - **Jacobian determinant.** Negative where the mapping folds over itself,
    producing mirrored patches in the warp. This failure is invisible to any
    per-point measure, because the control points on either side of a fold
    are still matched exactly.
  - **Bending energy**, coverage, and a one-line summary for the status bar.

  The report is discarded whenever the points change, since a report about a
  different point set is worse than none.
- `ThinPlateSplineTransform.map` evaluates the spline directly at arbitrary
  coordinates, in double precision, with no grid involved. Mapping 40 points
  went from 66 s and 134 MB to 1 ms and 0.07 MB, because `transform_coords`
  used to build the whole dense field and then index it.
- `ThinPlateSplineTransform.build_field`, `set_field`, `clear_field`, `field`,
  and `field_step`: the dense field as a cache, at a chosen resolution. A
  quarter-resolution field costs a few hundredths of a pixel for a sixteenth
  of the memory, and the error falls away quadratically as it gets finer.
- `tpsreg.warping.warp`, which warps large outputs a tile at a time.
  `skimage.transform.warp` builds one coordinate array for the entire output,
  16 bytes per pixel — 6.4 GB for a 400 Mpx image before any pixels are read.
  Tiling bounds that by the tile, and turns out to be faster as well.
- `tpsreg.warping.interpolate_fields`, for blending two displacement fields.
- Drag a control point to adjust it. Correcting a click that landed slightly
  off previously meant deleting the pair and re-placing both halves. The whole
  drag is one undo step, and only the marker being dragged moves.
- Control points are checked before a transform is estimated. Problems that
  make the fit impossible — too few points, coincident points, points all on
  one line, a half-finished pair — are reported by name and block estimation;
  problems that only degrade the result, such as points clustered in one part
  of the image, warn and can be dismissed. New `tpsreg.validation` module,
  with tests pinning what it calls an error to what the solver actually
  refuses.
- "Link views" checkbox: zooming or scrolling either panel does the same to
  the other, so a feature stays in the same place in both.
- Checkerboard and difference comparison modes in the 2D preview, alongside the
  existing wipe. New `tpsreg.overlays` module holds the compositing, so it can
  be tested without a display.
- File dialogs remember the folder the last one used, instead of starting from
  the working directory every time.
- A `warning` colour in both palettes, for drawing attention to a flagged
  control point.
- `PointManager.move_point`, `PointSet.move_point`, `PointManager.can_undo`,
  `PointManager.can_redo`, and the presenter methods that wrap them
  (`move_point`, `commit_point_move`, `find_point_near`, `check_points`,
  `assess_transform`, `can_undo`, `can_redo`).

### Removed

- **The "TPS affine" transform type.** It dropped the bending term at
  evaluation time, which is a strictly worse affine than fitting one
  directly. With it gone the transform-type dialog offered a choice of one,
  so that has gone too: applying, exporting and checking quality no longer
  interrupt with a modal. `TransformType.TPS` remains, so project files and
  the API are unaffected.

### Changed

- **Breaking:** a `ThinPlateSplineTransform` is now its fitted coefficients
  rather than a dense displacement field, and `.params` returns those
  coefficients. The field became a resolution-configurable cache, built only
  when something asks for it (`build_field`, or `estimate(build_field=True)`).

  Fitting used to evaluate the spline over the entire destination grid, so the
  cost of a transform scaled with the image rather than with the control
  points: a 400 Mpx stitched image meant a 3.2 GB transform, and 1600 Mpx
  meant 12.8 GB, which simply would not run. The coefficients are 0.7 KB
  whatever the grid.

  Exported transforms are correspondingly small, and CSV and TXT export now
  work at all — `np.savetxt` refuses a 3D array, so those two formats had
  always failed for a TPS despite being offered in the export dialog.

  The one place the dense form is still the working representation is
  interpolating between slices of a 3D stack: neighbouring slices are fitted
  to different control points, so their coefficients cannot be blended, but
  their fields share a grid and can. That path now says so explicitly and
  takes a `downsample` argument.
- Undo and Redo are greyed out when there is nothing to undo or redo, rather
  than being permanently enabled and silently doing nothing.
- Control points are placed on mouse release rather than press, so that a press
  landing on an existing marker can start a drag instead. A press that grabs a
  marker and releases without moving leaves it alone, so a slightly misjudged
  click near a point no longer stacks a second one on top of it.

### Fixed

- Warping a large image was between 4 and 100 times slower than it needed to
  be. Three separate causes, measured at 8192x8192 with order 1:

  | | before | after |
  |---|---|---|
  | rigid or affine, one transform | 101.5 s | 4.1 s |
  | rigid or affine, slice 300 of a sequential stack | 116.8 s | 1.2 s |
  | spline, one transform | 132.1 s | 35.1 s |

  *Clipping was inside the tile loop.* skimage clips every warp call to the
  input image's range, and finding that range means scanning the whole input —
  the same cost whether the call is warping a 256-pixel tile or the entire
  image. Tiled, that was a full pass over the source for every tile: 100 ms
  each, 1024 tiles, 85% of the runtime. It is now done once over the assembled
  output, which is also exactly what an untiled call does.

  *Matrix transforms were being hidden behind a callable.* skimage has a Cython
  path for a homogeneous matrix that computes each source coordinate as it
  goes, so it needs no coordinate array and therefore no tiling. Wrapping the
  transform in a Python callable, as tiling did, made it unreachable. A
  translation, rigid or affine transform is now handed over as a matrix.

  *Sequential chains were composed as functions.* Every link was called for
  every output pixel, so slice 300 cost three hundred times slice 1 — the
  stack got slower the further into it you were. Where every link is a matrix
  the chain now collapses into a single matrix when it is built, and depth
  stops mattering.

  A spline chain still cannot collapse, so `tps` with `reference_mode`
  `previous` remains linear in depth and is the one combination that stays
  expensive; `register_stack` now warns when a run asks for it, since `first`
  and `middle` build no chain at all.

- A matrix transform is now interpolated the same way at every image size. It
  was not before: outputs under the 4 Mpx tiling threshold went to skimage
  whole and took its Cython path, while larger ones were tiled through a
  callable and took scipy's. Those two disagree at order 3 — a cubic
  convolution against a prefiltered B-spline — so the same transform on the
  same data gave different pixels either side of the threshold. Order 3 output
  for translation, rigid and affine transforms above 4 Mpx therefore changes
  slightly with this release, to match what the same transform already produced
  below it.

- Sequential runs of `register_stack` exported no transforms at all. The CLI
  saved `transform.params`, which a composed chain did not have, so
  `transforms/` came out empty in the default reference mode. A collapsed chain
  now reports its matrix.

- The busy indicator never animated. It was driven by a Tk timer, which only
  runs while the event loop does — and during a synchronous operation the loop
  never runs, so the bar sat frozen. It also stepped every 1 ms, asking for a
  thousand redraws a second, and any redraw during an operation stopped it
  outright because `update_display` unconditionally cleared it.

  The bar is now stepped by hand wherever the application reports progress,
  chiefly on a status change, and flushed with `update_idletasks`. It no
  longer depends on the event loop at all. The indicator nests, so an inner
  operation cannot stop an outer one, and unbalanced stops cannot drive it
  negative and swallow the next start.

  Note that a single long call — a large warp, model inference — still blocks
  between reported points; the bar advances *between* steps, not within one.
- Every TPS warp was offset by one pixel in both axes. The dense field was
  sampled on a 1-based grid (`linspace(1, width, width)`) while control points
  and queries are 0-based, so a transform fitted to identical point sets moved
  the image by (1, 1) instead of leaving it alone. scikit-image's own affine
  on the same points was exact, which is what the comparison should always
  have been against. Two existing tests had encoded the offset as expected
  behaviour rather than questioning it.
- Coordinates were truncated to whole pixels before lookup, so sub-pixel
  warping was impossible no matter what interpolation order was requested.
- Collinear control points produced a garbage transform instead of an error on
  Linux and Windows. The solver relied on `np.linalg.solve` raising
  `LinAlgError` for a singular system, and whether it does depends on the
  LAPACK build: macOS Accelerate raised, the OpenBLAS builds on the CI runners
  returned a nonsense result for the identical points. `estimate` now tests the
  control point geometry directly — costing microseconds, and giving the same
  answer everywhere — and verifies the solution by substituting it back, so an
  inaccurate solve is caught whatever the cause.
- Clicks were recorded up to one image pixel away from where they were aimed
  when zoomed past 100%. The canvas-to-image conversion divided the event
  position and the canvas origin separately and truncated each, so the last
  screen pixel of every image cell reported the next cell along. The cursor
  readout shared the bug and now uses the same conversion as the click, so the
  two can no longer disagree.

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
