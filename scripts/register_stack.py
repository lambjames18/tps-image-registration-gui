#!/usr/bin/env python3
"""Register a folder of images slice by slice with MatchAnything.

Standalone: nothing in the GUI knows about this, and it does not need a
display. The registration logic lives in :mod:`tpsreg.stack_registration` so
it can be tested without torch; this file is the command line around it.

Example
-------
    python scripts/register_stack.py \\
        --input  path/to/slices \\
        --output path/to/results \\
        --transform rigid

The output folder gets:

    registered/     the aligned images
    matches/        each pair's matches, drawn as lines between the images
    overlays/       each registered slice checkerboarded against its reference
    transforms/     the fitted transform per slice, as .npy
    report.json     every number the run produced
    summary.csv     one row per slice, for scanning or plotting
    register.log    the full log

Run with --dry-run first on a large stack: it lists what would be processed,
in the order it would be processed, without loading the model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpsreg import overlays
from tpsreg.stack_registration import (
    REFERENCE_MODES,
    TRANSFORM_TYPES,
    apply_transforms,
    find_images,
    flag_outlying_slices,
    match_figure,
    register_stack,
    write_report,
)

logger = logging.getLogger("register_stack")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Folder of input images."
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Folder to write results to."
    )
    parser.add_argument(
        "--transform",
        "-t",
        choices=TRANSFORM_TYPES,
        default="rigid",
        help=(
            "Transform model. More constrained models need fewer matches and "
            "cannot invent deformation that is not there, so prefer the most "
            "constrained one the physical situation allows. Default: rigid."
        ),
    )
    parser.add_argument(
        "--reference",
        choices=REFERENCE_MODES,
        default="previous",
        help=(
            "What each slice registers against. 'previous' matches easily but "
            "accumulates drift along the stack; 'first' and 'middle' cannot "
            "drift but ask the matcher to relate distant slices. "
            "Default: previous."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MatchAnything checkpoint. Uses the packaged default when omitted.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device. Autodetected (cuda, then mps, then cpu) when omitted.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="Drop matches below this confidence. Default: 0.1.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=0.05,
        help=(
            "RANSAC inlier threshold. The deformable method works in "
            "normalised coordinates (~0.05); affine and projective work in "
            "pixels (~5). Default: 0.05."
        ),
    )
    parser.add_argument(
        "--ransac-method",
        choices=("deformable", "affine", "projective"),
        default="deformable",
        help="RANSAC model used to reject outlier matches. Default: deformable.",
    )
    parser.add_argument(
        "--no-ransac",
        action="store_true",
        help="Keep every match the model returns. Rarely a good idea.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        choices=(0, 1, 3),
        help=(
            "Interpolation order for the output images. 0 preserves the input "
            "dtype and range exactly, which matters for label images; 1 and "
            "above convert integer images to float. Default: 1."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N slices. Useful for a trial run.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the match and overlay images, which dominate the runtime "
        "on a large stack once the model is warm.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be processed and stop, without loading the model.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Debug-level logging."
    )
    return parser


def setup_logging(output: Path, verbose: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output / "register.log", mode="w", encoding="utf-8"),
        ],
    )


def load_images(paths: list[Path]) -> list[np.ndarray]:
    from skimage import io

    images = []
    for path in paths:
        image = io.imread(path)
        # Registration works on intensity; a colour stack is matched on its
        # luminance rather than one arbitrary channel.
        if image.ndim == 3 and image.shape[2] in (3, 4):
            from skimage.color import rgb2gray

            image = (rgb2gray(image[:, :, :3]) * 255).astype(np.uint8)
        images.append(image)
        logger.debug("Loaded %s %s %s", path.name, image.shape, image.dtype)
    return images


def make_match_fn(args: argparse.Namespace):
    """Build the matching callable, loading the model once.

    Loading is by far the slowest step, so the matcher is created here and
    closed over rather than rebuilt per pair.
    """
    from tpsreg.roma_matcher import apply_matcher, create_matcher

    logger.info("Loading MatchAnything (this is the slow part)...")
    matcher = create_matcher(checkpoint_path=args.checkpoint, device=args.device)
    logger.info("Model ready.")

    def match(moving: np.ndarray, reference: np.ndarray):
        moving_points, reference_points, confidence = apply_matcher(
            matcher,
            moving,
            reference,
            ransac_filter=not args.no_ransac,
            ransac_threshold=args.ransac_threshold,
            ransac_method=args.ransac_method,
        )

        if args.confidence > 0 and len(confidence):
            keep = np.asarray(confidence) >= args.confidence
            if keep.any():
                moving_points = moving_points[keep]
                reference_points = reference_points[keep]
                confidence = np.asarray(confidence)[keep]
            else:
                logger.warning(
                    "Every match fell below the confidence threshold; keeping "
                    "them all so the pair can still be fitted."
                )

        return moving_points, reference_points, confidence

    return match, matcher


def save_debug_figures(
    output: Path,
    index: int,
    name: str,
    moving: np.ndarray,
    reference: np.ndarray,
    registered: np.ndarray,
    match_fn_output,
) -> None:
    """Write the two pictures that make a bad pair obvious."""
    from skimage import io

    moving_points, reference_points, _ = match_fn_output

    matches_dir = output / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(
        matches_dir / f"{index:04d}_{name}.png",
        match_figure(moving, reference, moving_points, reference_points),
        check_contrast=False,
    )

    overlays_dir = output / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(
        overlays_dir / f"{index:04d}_{name}.png",
        overlays.checkerboard(registered, reference),
        check_contrast=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    paths = find_images(args.input)
    if args.limit:
        paths = paths[: args.limit]

    if not paths:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"{len(paths)} image(s) would be registered, in this order:")
        for i, path in enumerate(paths):
            print(f"  {i:4d}  {path.name}")
        print(f"\ntransform={args.transform} reference={args.reference}")
        return 0

    setup_logging(args.output, args.verbose)
    logger.info("Registering %d slice(s) from %s", len(paths), args.input)

    images = load_images(paths)
    names = [path.stem for path in paths]

    match_fn, _matcher = make_match_fn(args)

    # Keep each pair's matches so the figures can be drawn afterwards without
    # running the model twice. register_stack visits slices in order and calls
    # this once per non-reference slice, so the Nth call belongs to the Nth
    # non-reference pair; that mapping is rebuilt below rather than assumed to
    # be index - 1, which is only true when the anchor happens to be slice 0.
    recorded: list[tuple] = []

    def recording_match_fn(moving, reference):
        output = match_fn(moving, reference)
        recorded.append(output)
        return output

    def on_progress(done: int, total: int, pair) -> None:
        status = "ok" if pair.ok else "FAILED"
        logger.info(
            "[%d/%d] %s: %d matches, residual %.2f px, moved %.1f px  %s",
            done,
            total,
            pair.name,
            pair.n_matches,
            pair.residual_median,
            pair.displacement,
            status,
        )
        for warning in pair.warnings:
            logger.warning("  %s: %s", pair.name, warning)

    transforms, result = register_stack(
        images,
        recording_match_fn,
        transform_type=args.transform,
        reference_mode=args.reference,
        names=names,
        on_progress=on_progress,
    )

    logger.info("Warping...")
    registered = apply_transforms(images, transforms, order=args.order)

    from skimage import io

    registered_dir = args.output / "registered"
    registered_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, image) in enumerate(zip(names, registered, strict=True)):
        io.imsave(
            registered_dir / f"{index:04d}_{name}.tif", image, check_contrast=False
        )

    transforms_dir = args.output / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    for index, transform in enumerate(transforms):
        params = getattr(transform, "params", None)
        if params is not None:
            np.save(transforms_dir / f"{index:04d}.npy", np.asarray(params))

    if not args.no_figures:
        logger.info("Writing debug figures...")
        matched_pairs = [pair for pair in result.pairs if not pair.is_reference]
        for call, pair in enumerate(matched_pairs):
            if call >= len(recorded):
                break
            save_debug_figures(
                args.output,
                pair.index,
                pair.name,
                images[pair.index],
                images[pair.reference_index],
                registered[pair.index],
                recorded[call],
            )

    write_report(
        result,
        args.output,
        extra={
            "input": str(args.input),
            "files": [path.name for path in paths],
            "settings": {
                "transform": args.transform,
                "reference": args.reference,
                "ransac_method": None if args.no_ransac else args.ransac_method,
                "ransac_threshold": args.ransac_threshold,
                "confidence": args.confidence,
                "order": args.order,
            },
        },
    )

    print()
    print(result.summary())

    outliers = flag_outlying_slices(result)
    if outliers:
        print(f"  outliers:  slices {outliers} moved unlike their neighbours")
    print(f"\nWrote results to {args.output}")

    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
