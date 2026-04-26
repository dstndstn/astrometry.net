"""Benchmark blur_score over a directory or list of astronomical images.

Outputs CSV: image,fwhm,ellipticity,n_sources,score.

Usage:
    python -m astrometry.util.blur_score_benchmark \
        --images-dir /src/Astrometry-testing-data/data
    python -m astrometry.util.blur_score_benchmark \
        --paths img1.png img2.fits ...
"""
import argparse
import os
import sys

from .blur_score import (
    DEFAULT_TARGET_FWHM,
    InsufficientSources,
    compute_blur_score,
    measure_psf,
)

DEFAULT_IMAGES_DIR = "/src/Astrometry-testing-data/data"
IMG_EXTS = (".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _walk_images(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="blur_score_benchmark")
    p.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    p.add_argument("--paths", nargs="*",
                   help="Specific image paths (overrides --images-dir walk)")
    p.add_argument("--target-fwhm", type=float, default=DEFAULT_TARGET_FWHM)
    args = p.parse_args(argv)

    if args.paths:
        files = args.paths
    else:
        if not os.path.isdir(args.images_dir):
            print(f"images-dir not found: {args.images_dir}", file=sys.stderr)
            return 2
        files = _walk_images(args.images_dir)

    print("image,fwhm,ellipticity,n_sources,score")
    for path in files:
        try:
            f, e, n = measure_psf(path)
            s = compute_blur_score(path, target_fwhm=args.target_fwhm)
            print(f"{path},{f:.4f},{e:.4f},{n},{s:.4f}")
        except InsufficientSources:
            print(f"{path},nan,nan,0,nan")
        except Exception as ex:
            print(f"# error {path}: {ex}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
