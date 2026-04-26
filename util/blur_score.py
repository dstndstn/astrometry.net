"""Blur score for astronomical images via stellar PSF measurement.

Runs source-extractor to detect stars, takes the median FWHM and
ellipticity over flag-clean sources, and maps to a [0, 1] score
(1 = sharp pinpoint stars, 0 = severely blurred).

The score is FWHM-driven (penalizing wide PSFs from defocus / poor
seeing) and weighted down by ellipticity (penalizing tracking-error
streaks). Satellite trails do not affect the score: the metric is local
to detected stars, so a single bright trail on an otherwise sharp frame
still scores high.
"""
import os
import subprocess
import tempfile
import numpy as np

DEFAULT_TARGET_FWHM = 3.0
DETECT_THRESH = 5.0
DETECT_MINAREA = 5
MIN_SOURCES = 10
MIN_BRIGHT = 4
BRIGHT_QUANTILE = 0.7
COUNT_SATURATION = 10
FWHM_RANGE = (1.0, 30.0)
SOURCE_EXTRACTOR = "source-extractor"
FILTER_NAME = "/usr/share/source-extractor/gauss_3.0_5x5.conv"


class InsufficientSources(RuntimeError):
    pass


def _to_2d_float(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        c = arr.shape[2]
        if c >= 3:
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        else:
            arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _load_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data
        return _to_2d_float(data)
    from PIL import Image
    img = Image.open(path)
    if img.mode in ("L", "F") or img.mode.startswith("I"):
        return _to_2d_float(np.asarray(img))
    return _to_2d_float(np.asarray(img.convert("F")))


def to_grayscale(image):
    """Accept a path or ndarray. Return float32 2D array (no resampling)."""
    if isinstance(image, str):
        return _load_path(image)
    return _to_2d_float(np.asarray(image))


def _ensure_fits(image, workdir):
    if isinstance(image, str):
        ext = os.path.splitext(image)[1].lower()
        if ext in (".fits", ".fit", ".fts"):
            return image
        arr = _load_path(image)
    else:
        arr = to_grayscale(image)
    from astropy.io import fits
    out = os.path.join(workdir, "in.fits")
    fits.PrimaryHDU(arr.astype(np.float32)).writeto(out, overwrite=True)
    return out


def detect_sources(image,
                   detect_thresh=DETECT_THRESH,
                   detect_minarea=DETECT_MINAREA):
    """Return a list of dicts {fwhm, ellipticity, flux} for FLAGS==0 sources
    whose FWHM falls inside FWHM_RANGE."""
    with tempfile.TemporaryDirectory() as td:
        fits_path = _ensure_fits(image, td)
        param_file = os.path.join(td, "sex.param")
        with open(param_file, "w") as f:
            f.write("FWHM_IMAGE\nELLIPTICITY\nFLAGS\nFLUX_AUTO\nFLUX_RADIUS\n")
        cat_file = os.path.join(td, "sex.cat")
        cmd = [
            SOURCE_EXTRACTOR, fits_path,
            "-CATALOG_NAME", cat_file,
            "-CATALOG_TYPE", "ASCII_HEAD",
            "-PARAMETERS_NAME", param_file,
            "-DETECT_THRESH", str(detect_thresh),
            "-ANALYSIS_THRESH", str(detect_thresh),
            "-DETECT_MINAREA", str(detect_minarea),
            "-FILTER", "Y",
            "-FILTER_NAME", FILTER_NAME,
            "-PHOT_FLUXFRAC", "0.5",
            "-VERBOSE_TYPE", "QUIET",
        ]
        subprocess.run(cmd, check=True, capture_output=True, cwd=td)
        rows = []
        with open(cat_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    fwhm = float(parts[0])
                    ecc = float(parts[1])
                    flags = int(parts[2])
                    flux = float(parts[3])
                    r50 = float(parts[4]) if len(parts) > 4 else float("nan")
                except ValueError:
                    continue
                if flags != 0:
                    continue
                if not (FWHM_RANGE[0] < fwhm < FWHM_RANGE[1]):
                    continue
                rows.append({"fwhm": fwhm, "ellipticity": ecc,
                             "flux": flux, "r50": r50})
        return rows


def measure_psf(image, bright_quantile=BRIGHT_QUANTILE, **kwargs):
    """Return (median_fwhm, median_ellipticity, n_bright_sources).

    Restricts the median to the brightest `bright_quantile` of detected
    sources (top 30% by FLUX_AUTO by default), so faint detections and
    leftover noise blobs don't wash out the PSF signal coming from real
    reference stars. Falls back to all sources when too few bright ones
    pass the cut. Raises InsufficientSources if total detections
    < MIN_SOURCES."""
    sources = detect_sources(image, **kwargs)
    n = len(sources)
    if n < MIN_SOURCES:
        raise InsufficientSources(
            f"only {n} clean sources detected (need >= {MIN_SOURCES})"
        )
    flux = np.array([s["flux"] for s in sources])
    cut = np.quantile(flux, bright_quantile)
    bright = [s for s in sources if s["flux"] >= cut]
    if len(bright) < MIN_BRIGHT:
        raise InsufficientSources(
            f"only {len(bright)} bright sources above quantile "
            f"{bright_quantile} (need >= {MIN_BRIGHT})"
        )
    fwhms = np.array([s["fwhm"] for s in bright])
    eccs = np.array([s["ellipticity"] for s in bright])
    return float(np.median(fwhms)), float(np.median(eccs)), len(bright)


def compute_blur_score(image, target_fwhm=DEFAULT_TARGET_FWHM, **kwargs):
    """Return blur score in [0, 1]. 1 = sharp, 0 = severely blurred.

    score = fwhm_score * shape_weight * count_weight, where
        fwhm_score   = clip(target_fwhm / median_bright_fwhm, 0, 1)
        shape_weight = clip(1 - median_bright_ellipticity, 0, 1)
        count_weight = clip(n_bright / COUNT_SATURATION, 0, 1)

    Severely blurred frames lose faint stars below the detection
    threshold, so n_bright drops; count_weight folds that signal in.
    """
    fwhm, ecc, n_bright = measure_psf(image, **kwargs)
    fwhm_score = min(1.0, target_fwhm / fwhm) if fwhm > 0 else 0.0
    shape_weight = max(0.0, 1.0 - ecc)
    count_weight = min(1.0, n_bright / COUNT_SATURATION)
    return float(fwhm_score * shape_weight * count_weight)


def _main(argv=None):
    import argparse, sys
    p = argparse.ArgumentParser(
        prog="blur_score",
        description="Astro blur score (1 = sharp, 0 = blurred) via stellar PSF.",
    )
    p.add_argument("image")
    p.add_argument("--target-fwhm", type=float, default=DEFAULT_TARGET_FWHM)
    p.add_argument("--detect-thresh", type=float, default=DETECT_THRESH)
    p.add_argument("--raw", action="store_true",
                   help="Print 'fwhm=... ellipticity=... n_sources=...' instead of the score.")
    args = p.parse_args(argv)
    try:
        if args.raw:
            f, e, n = measure_psf(args.image, detect_thresh=args.detect_thresh)
            print(f"fwhm={f:.3f} ellipticity={e:.3f} n_sources={n}")
        else:
            s = compute_blur_score(
                args.image,
                target_fwhm=args.target_fwhm,
                detect_thresh=args.detect_thresh,
            )
            print(s)
    except InsufficientSources as ex:
        print(f"insufficient_sources: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
