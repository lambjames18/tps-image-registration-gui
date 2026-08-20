# Using tpsreg as a library

The registration core is importable and does not require a display, so it can
be used from a script, a notebook, or a batch pipeline.

```python
import numpy as np
from tpsreg import ThinPlateSplineTransform, transform_image

src = np.array([[10, 10], [10, 90], [90, 10], [90, 90], [50, 50]], dtype=float)
dst = np.array([[12, 11], [11, 88], [91, 13], [89, 92], [51, 49]], dtype=float)

# Fit a spline. The cost is set by the control points, not the image, and
# size is only recorded so exports are self-describing.
tform = ThinPlateSplineTransform()
tform.estimate(src, dst, size=(100, 100))

# Map coordinates directly. Exact, including at fractional positions.
where_they_land = tform.map(dst)

# Or warp an image in one call.
warped = transform_image(image, src, dst, output_shape=(100, 100), order=1)
```

Points are `(N, 2)` arrays in `(x, y)`.

---

## What is where

| Module | For |
|---|---|
| `tpsreg.tps` | `ThinPlateSplineTransform` — fitting, mapping, regularization, field caching. |
| `tpsreg.warping` | `warp`, `transform_image`, `transform_image_stack`, and the transform estimators behind one interface. |
| `tpsreg.metrics` | Leave-one-out residuals, Jacobian determinant, bending energy, coverage. |
| `tpsreg.ransac` | `ransac_filter`, for rejecting outlier correspondences. |
| `tpsreg.stack_registration` | Slice-by-slice registration of a stack; see [the stack guide](stack-registration.md). |
| `tpsreg.roma_matcher` | The MatchAnything matcher. Needs the `matchanything` extra. |

---

## Smoothing

A spline interpolates: it passes through every control point, reproducing click
error exactly along with the real deformation. `regularization` relaxes that.

```python
tform = ThinPlateSplineTransform(regularization=0.01)
tform.estimate(src, dst, size=(100, 100))
```

To pick the strength from the data rather than by guessing:

```python
from tpsreg.tps import select_regularization

# Leave-one-out cross-validation over a grid of candidates.
alpha, candidates, scores = select_regularization(src, dst)

tform = ThinPlateSplineTransform(regularization=alpha)
```

`candidates` and `scores` are the whole curve, so you can plot it and see how
sharp the choice was.

The value is normalised by the kernel magnitude, so the same number means
roughly the same thing at any image size — within about twofold across a
200-fold change in coordinates, since `r²log(r²)` is not scale-homogeneous.
With clean points the selection returns zero.

---

## Checking a fit

```python
from tpsreg import metrics

quality = metrics.assess(tform, src, dst, image_shape=(100, 100))
print(quality.summary())
print(quality.median_residual, quality.folded_fraction, quality.coverage)
```

`TransformQuality` carries the per-point `leave_one_out` residuals, an
`outliers` mask, the `worst_point` index, `min_jacobian`, `folded_fraction`,
`bending_energy` and `coverage`, with `median_residual` and `has_folds` as
conveniences.

Ordinary residuals are useless for a spline — it interpolates, so they come out
around 1e-12 whether the correspondences are good or catastrophic.
`metrics.leave_one_out_residuals` refits without each point and measures how far
the fit misses it, which is where a bad correspondence actually shows up. It
needs roughly nine well-spread points to mean anything.

---

## Large images

A fitted transform is its coefficients — under a kilobyte, whatever the size
of the image. Warping is done a tile at a time once the output is large
enough that one coordinate array for the whole thing would be the limiting
factor, so a stitched optical mosaic is bounded by the tile rather than by
the image.

A transform that is really a matrix — translation, rigid, affine, projective —
skips tiling entirely and is handed to skimage as a matrix, which lets it
compute source coordinates in Cython without building a coordinate array at
all. At 8192×8192 that is 4 s rather than 101 s. Splines take the tiled path,
where the cost is dominated by evaluating the kernel.

If you are going to query most of a grid repeatedly, a dense displacement
field can be cached, at whatever resolution is worth the memory:

```python
tform.build_field(size=(20000, 20000), downsample=4)   # a sixteenth the memory
...
tform.clear_field()                                    # give it back
```

The field is a cache. Clearing it changes speed, not results, beyond the
interpolation error a coarse field introduces — which falls away
quadratically as the field gets finer, and at quarter resolution is a few
hundredths of a pixel for typical control point spacings.
