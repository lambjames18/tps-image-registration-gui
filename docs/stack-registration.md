# Registering a stack of images

`scripts/register_stack.py` aligns a folder of images slice by slice using the
MatchAnything model, without the GUI and without a display. It is meant for
serial-section data — FIB/SEM slices, serial optical sections — where clicking
points on every pair is not realistic.

```bash
python scripts/register_stack.py \
    --input  path/to/slices \
    --output path/to/results \
    --transform rigid
```

Add `--dry-run` first on a large stack: it prints what would be processed, in
the order it would be processed, and stops before loading the model. Filenames
are sorted numerically, so `slice_10` lands after `slice_9` rather than after
`slice_1`.

Requires the `matchanything` extra and a checkpoint — see
[installation](../README.md#optional-extras).

---

## Choosing a transform

`--transform` is one of `translation`, `rigid`, `affine`, or `tps`.

Prefer the most constrained model the physical situation allows. A constrained
model needs fewer matches and cannot invent deformation that is not there. A
thin-plate spline will happily absorb a handful of bad matches into a warp that
looks plausible slice by slice and is wrong.

| Model | Minimum matches | Use when |
|---|---|---|
| `translation` | 1 | The stage shifts between slices and nothing else changes. |
| `rigid` | 2 | Shift and rotation — the usual choice for serial sections. |
| `affine` | 3 | Scale or shear as well, e.g. drift in a scanned modality. |
| `tps` | 3 | Genuine local deformation. Read the [warning below](#the-one-slow-combination) before using it sequentially. |

---

## Choosing a reference

`--reference` decides what each slice is registered against, and the choice is
a real trade-off rather than a default worth trusting blindly:

- **`previous`** (default) matches adjacent slices, which look alike, so
  matching is easy. Each slice's transform is the composition of every one
  before it, so small per-pair errors accumulate into drift along the stack.
- **`first`** and **`middle`** register everything against one fixed slice.
  Nothing accumulates, but the matcher is asked to relate slices that may be
  far apart and no longer resemble each other.

The report gives you the number to judge this by: `cumulative_displacement`
tracks how far each slice has moved from the first, and in `previous` mode that
is where drift shows up.

### Composition and cost

Sequential mode composes coordinate mappings rather than resampling once per
link, so a slice at the end of the stack is interpolated exactly once no matter
how long the chain is. For `translation`, `rigid` and `affine` the composition
is a matrix product, so warping slice 300 costs the same as slice 1.

### The one slow combination

A spline chain cannot reduce to a matrix — warping slice N evaluates N splines
— so `--transform tps --reference previous` gets slower the further into the
stack it goes. On a long stack of large images it is impractical: measured at
8192×8192 with 300 slices, the late slices take hours each.

Use `--reference first` or `--reference middle` with `tps`, where each slice has
exactly one spline and nothing accumulates. The script warns when a run asks
for the expensive combination.

---

## Output

The output folder holds the aligned images plus the debugging trail:

```
registered/     the aligned images
matches/        each pair's matches, drawn as lines between the two images
overlays/       each registered slice checkerboarded against its reference
transforms/     the fitted transform per slice, as .npy
report.json     every number the run produced
summary.csv     one row per slice, for scanning or plotting
register.log    the full log
```

`matches/` is the one to look at first when a slice comes out wrong: bad
correspondences are the lines that cross the others, which no summary number
conveys as directly.

Per pair the report records the match count, the median and maximum residual,
how far the slice moved, and any warnings — too few matches for the chosen
model, degenerate point geometry, a residual out of line with the rest.
Residuals are ordinary for the constrained models and leave-one-out for `tps`,
since a spline interpolates and its ordinary residuals are always near zero.

Slices whose displacement is unlike their neighbours' are flagged by a
median/MAD z-score, on the reasoning that consecutive sections move by similar
amounts and a jump is usually a mis-registration. That list is in
`report.json` as `outlying_slices` and is printed at the end of a run.

A pair that fails to match or fit gets the identity transform and is recorded
as failed rather than aborting the run, so one bad slice does not cost you the
whole stack. In `previous` mode a failure breaks the chain: every later slice
is composed through the identity where that link should have been, so they are
all shifted by however much that pair actually moved. This is exactly why the
failure is reported.

---

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `-i` | — | Folder of input images. |
| `--output`, `-o` | — | Folder to write results to. |
| `--transform`, `-t` | `rigid` | Transform model. |
| `--reference` | `previous` | What each slice registers against. |
| `--checkpoint` | packaged default | MatchAnything checkpoint. |
| `--device` | autodetected | torch device: cuda, then mps, then cpu. |
| `--confidence` | `0.1` | Drop matches below this confidence. |
| `--ransac-threshold` | `0.05` | Inlier threshold. Normalised (~0.05) for `deformable`, pixels (~5) for `affine`/`projective`. |
| `--ransac-method` | `deformable` | RANSAC model used to reject outlier matches. |
| `--no-ransac` | off | Keep every match the model returns. Rarely a good idea. |
| `--order` | `1` | Interpolation order. 0 preserves the input dtype and range exactly, which matters for label images. |
| `--limit` | none | Process only the first N slices. Useful for a trial run. |
| `--no-figures` | off | Skip the match and overlay images, which dominate the runtime on a large stack once the model is warm. |
| `--dry-run` | off | List what would be processed and stop, without loading the model. |
| `--verbose`, `-v` | off | Debug-level logging. |

---

## Using it from Python

The registration logic is in `tpsreg.stack_registration` and takes the matcher
as an argument, so it can be used with a different matcher — or tested without
one:

```python
from tpsreg.stack_registration import register_stack, apply_transforms

transforms, result = register_stack(images, match_fn, transform_type="rigid")
registered = apply_transforms(images, transforms)
print(result.summary())
```

`match_fn(moving, reference)` returns `(moving_points, reference_points,
confidences)`. Anything with that signature works — phase correlation, SIFT,
or your own matcher.
