# User guide

The desktop application, menu by menu. If you are starting from nothing, the
[quickstart in the README](../README.md#quickstart) gets you to a first result;
this is the reference to come back to with the app open.

- [Main window](#main-window)
- [File menu](#file-menu)
- [Edit menu](#edit-menu)
- [View menu](#view-menu)
- [Smoothing](#smoothing)
- [Auto menu](#auto-menu)
- [Export formats and cropping](#export-formats-and-cropping)

---

## Main window

![The main window](images/GUI-main.jpg)

The left and right panels show the source and destination images. The top bar
controls the slice (for 3D data), the displayed modality, CLAHE, zoom, view
linking, resolution matching, and smoothing. The bottom bar shows the cursor
position, the point count, the live fit quality, a status message, and a busy
indicator during long operations.

Left-click places a control point. Right-clicking near one removes the pair.
Dragging a point moves it, which is the quick way to correct a click that
landed slightly off; the whole drag is a single undo step. **Link views** ties
the two panels together so zooming or scrolling one does the same to the other.

Once there are nine or more pairs, each marker shows its leave-one-out
residual and is coloured from green to orange by how large it is, and the
status bar shows the median and the worst point. These update as you place,
delete and drag points, so a bad correspondence is visible while you are still
looking at it. Below nine points nothing is shown — see
[*Check registration quality*](#check-registration-quality) for why.

---

## File menu

![File menu](images/GUI-file-menubar.jpg)

Create, open and save projects; import and export data; import and export
control points.

**Open source image** and **Open destination image** load the two images. The
*source* is the one that gets warped, so load the more distorted image there.
Plain images (`.tif`, `.png`, `.jpg`) prompt for a modality name; EBSD files
(`.ang`, `.h5`, `.dream3d`) load all their modalities automatically. Calling
*Open ... image* again with a different name adds another modality on the same
grid.

**Save project** writes all data, points and settings to a single JSON file, so
a session can be picked up later or handed to someone else.

---

## Edit menu

![Edit menu](images/GUI-edit-menubar.jpg)

Undo and redo, clear points, and set the pixel size of each image. Each click is
one undoable step, as is each drag. Undo and redo are greyed out when there is
nothing to undo or redo.

**Set resolution** takes microns per pixel for both images. It is needed for
*Match resolutions* and for source-cropped export. If the resolution is not set,
both images are assumed to have the same pixel size.

---

## View menu

![View menu](images/GUI-view-menubar.jpg)

Toggle point visibility and open the preview windows.

### View corrected image

Overlays the warped source on the destination, with sliders for the blend and —
in 3D — for the slice and slicing axis. The 2D preview offers three ways to
compare:

| Mode | Shows |
| --- | --- |
| `wipe` | Drag the sliders to sweep one image over the other. Best for checking a single edge. |
| `checkerboard` | Alternating squares from each image. Misalignment shows as features stepping sideways at every tile boundary. |
| `difference` | Absolute difference. A good alignment is black; whatever still glows did not line up. |

![Correction preview](images/GUI-preview.jpg)

### Check registration quality

Fits a transform and reports how good the correspondences look. Worth knowing
about what it does **not** do: it does not report how far the fit lands from
each control point, because a thin-plate spline interpolates — it passes
exactly through every point it was given, so that number is around 1e-12
whether the points are good or badly wrong.

| Measure | What it tells you |
| --- | --- |
| Leave-one-out residual | Refits without each point and measures how far the fit misses it. This is where a mistyped correspondence shows up. Flagged points are drawn with a larger warning-coloured ring so you can find them. |
| Jacobian determinant | Goes negative where the mapping folds over itself, producing mirrored patches. Invisible to any per-point measure, since the points on either side of a fold are still matched exactly. |
| Bending energy | How far the warp is from a plain affine. Zero for a pure affine, growing as the deformation gets more local. |
| Coverage | Fraction of the image the points enclose. Everything outside is extrapolated. |

Leave-one-out needs roughly nine well-spread points to mean anything: it asks
the remaining points to predict the held-out one, and with only a handful that
question has no good answer for any of them. That is why the live per-point
numbers stay hidden below nine points — showing them while you place your first
few would be alarming and wrong.

The menu version refits once per control point, so it is on a menu rather than
running as you click — about 15 ms at 50 points, 0.3 s at 200.

### View matched points

Shows both images side by side with lines joining matched points, which makes
bad correspondences obvious: they are the lines that cross the others.

![Matched points](images/GUI-points.jpg)

---

## Smoothing

Off by default. The top bar has a **Smoothing** selector with *Off*,
*Automatic*, and a manual number.

Exact interpolation is not the same as an accurate transform. The spline
passes through every control point you place, which means it also reproduces
every click error exactly — it contorts itself to honour your mistakes.
Smoothing lets the fit miss the points a little in exchange for a better match
to the real deformation.

*Automatic* picks the strength by leave-one-out cross-validation, which is
worth using in preference to the manual number: the strength has no units
anyone has an intuition for, and the right value depends on how noisy your
clicks are relative to the distortion you are correcting. On synthetic data
with a known answer and 1.5 px of click noise it roughly halved the error
against the true deformation. With clean points it selects zero and changes
nothing, so it costs you nothing to leave on.

The manual number is normalised, so the same value means roughly the same
thing whatever the image size. Useful values are typically between 0.001 and
1.

---

## Auto menu

![Auto menu](images/GUI-auto-menubar.jpg)

Detect control points automatically with either SIFT or the pretrained
MatchAnything model. MatchAnything needs the `matchanything` extra and a
checkpoint — see [installation](../README.md#optional-extras).

![MatchAnything settings](images/GUI-auto-options.jpg)

| Setting | Meaning |
|---|---|
| Num samples | Maximum number of matched points to return. |
| Enable RANSAC filtering | Reject correspondences inconsistent with a global transform. Leave on. |
| RANSAC method | `deformable` (default), `affine`, or `projective`. `deformable` and `projective` give broadly similar results; use `affine` only when you know the distortion is affine. |
| RANSAC threshold | 0.01–0.1 for `deformable` (normalized units), around 5.0 for `affine`/`projective` (pixels). |
| RANSAC max trials | Iterations. At least 100; raise it when the outlier rate is high. |

Automatic detection is a starting point, not an answer: check the result in
*View matched points* and delete or drag anything that looks wrong before
exporting.

---

## Export formats and cropping

Exporting asks for a format and a cropping mode:

- **Destination cropping** produces output with the destination image's
  dimensions. Prefer this in most cases.
- **Source cropping** preserves the source grid. Use this for EBSD data,
  particularly DREAM.3D files, where the grid must be preserved. Set the
  resolution first so features end up at the same scale.

An `.ang` file can only be exported when the source data was loaded from one.

---

## Before the fit runs

Control points are checked before a transform is estimated. Anything that will
make the fit fail — too few points, points on top of each other, points all on
one line — is reported with the problem named, rather than surfacing as a
solver error a few seconds later. Points that will merely give a poor result,
such as a cluster in one corner of the image, produce a warning you can
dismiss.

---

## See also

- [Registering a stack of images](stack-registration.md) — the command-line
  script for serial-section data, where clicking points on every pair is not
  realistic.
- [Using tpsreg as a library](api.md) — the registration core, importable and
  display-free.
