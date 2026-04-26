# `blur_score` — astro image sharpness score

A `[0, 1]` score where 1 = sharp pinpoint stars, 0 = severely blurred.
Designed for astrophotography frames: motion blur (mount wobble),
defocus, and bad seeing all push the score down. Satellite trails
through an otherwise-sharp frame do **not** push it down — by design,
that's a separate defect.

## Why Laplacian variance was the wrong tool

The first iteration used Laplacian variance — a standard photographic
sharpness metric (variance of the discrete Laplacian; sharp images
have more high-frequency edge content, so higher variance). On the
labeled testing-data frames, it ranked the wrong direction:

| Image (label) | Laplacian variance | Score |
|---|---:|---:|
| `normal.png` (sharp reference) | 3.39 | 0.007 |
| `kleiner Wackler.png` (motion blur) | 5.10 | 0.010 |
| `mit Satellit.png` (sharp + satellite) | 5.66 | 0.011 |
| `Wackler mit Satellit.png` (blur + satellite) | 6.00 | 0.012 |

The reference `normal` has the *lowest* variance and the streaked /
contaminated frames have *higher* variance. No threshold tuning fixes
this — the metric is monotonic in the wrong direction.

### Why it inverts on astro images

Laplacian variance assumes a natural image — texture and edges
distributed across the whole frame. Astro frames violate every
assumption:

1. **The frame is mostly empty sky.** 99% of pixels are featureless
   background. Edge content comes almost entirely from however many
   stars are detected, not from surface texture. The "sharper image
   has more edges" intuition doesn't hold when there *is* no surface.
2. **Motion-blurred stars have *more* edge pixels, not fewer.** A
   sharp star is a 1–3 px Gaussian spot. A motion-blurred star is a
   long thin streak. The streak has many more pixels along its long
   edge than the spot has perimeter. Laplacian variance literally
   rewards streaking.
3. **Satellite trails dominate the metric.** A single bright
   high-contrast line across the frame pumps variance through the
   roof while the rest of the frame may be perfectly sharp. The
   metric flags the wrong defect.
4. **Score saturation.** All four labeled frames had variance in
   3–6, against the configured threshold of 500 (`v / (v + T)`
   normalization). Scores all collapsed to ~0.01 with essentially
   no resolution between them.

The fix is not a better threshold — it's a different metric, one
grounded in what "sharp" means for stars specifically.

## What's implemented now

**Stellar PSF measurement via `source-extractor`.** This is the
metric every astronomy pipeline (SDSS, Pan-STARRS, LSST, observatory
autofocus) uses for image quality, because the only meaningful sense
of "sharp" in an astro frame is the width of the point-spread
function imprinted on every star.

### Pipeline

1. **Load the image** (FITS via `astropy.io.fits`, anything else via
   PIL → grayscale).
2. **If non-FITS, write a temporary FITS** (`source-extractor`
   reliably reads only FITS).
3. **Run `source-extractor`** with:
   - `DETECT_THRESH = 5.0` (σ above background — high enough to reject
     noise, low enough to keep the bright reference stars)
   - `DETECT_MINAREA = 5`
   - `FILTER = Y` with `gauss_3.0_5x5.conv` (matched roughly to the
     expected PSF width, sharpens detection without inventing peaks)
   - `PHOT_FLUXFRAC = 0.5` so `FLUX_RADIUS` reports the half-light
     radius
   - Output catalog columns: `FWHM_IMAGE`, `ELLIPTICITY`, `FLAGS`,
     `FLUX_AUTO`, `FLUX_RADIUS`
4. **Filter detections** to `FLAGS == 0` and `1 < FWHM < 30 px` (drops
   single-pixel cosmics, deblending failures, and absurd extents).
5. **Brightness cut.** Take only the brightest 30% of detections
   (`FLUX_AUTO ≥ 70th percentile`). Faint detections near the noise
   floor are unreliable; the bright top end is where the PSF is
   actually well-measured. This was the single biggest accuracy fix
   over taking medians across all detections.
6. **Robust statistics.** Median FWHM and median ellipticity over
   the bright subset. Median (not mean) because a few bad detections
   shouldn't move the answer.
7. **Score the frame** (formula below).

### Score formula

```
score = fwhm_score × shape_weight × count_weight
```

where

| term | definition | what it captures |
|---|---|---|
| `fwhm_score`   | `clip(target_fwhm / median_bright_fwhm, 0, 1)` | isotropic blur (defocus, seeing) |
| `shape_weight` | `clip(1 - median_bright_ellipticity, 0, 1)` | anisotropic blur (tracking error / Wackler) |
| `count_weight` | `clip(n_bright / COUNT_SATURATION, 0, 1)` | severe blur drops faint stars below the detection threshold; n_bright itself is signal |

Defaults: `target_fwhm = 3.0 px`, `COUNT_SATURATION = 10`. The product
keeps the result bounded in `[0, 1]` and means any single dimension
of degradation can drag the score down independently — exactly what
we want.

### Why three terms instead of one

Each captures a *different* failure mode:

- **`fwhm_score` alone** misses tracking errors. A motion-blurred
  star can have moderate FWHM (the geometric mean of the major and
  minor axes barely moves) but its shape is clearly elongated.
- **`shape_weight` alone** misses defocus. A defocused star is wide
  but still round, so ellipticity stays low.
- **`count_weight` alone** is too crude — a partly-cloudy frame and a
  perfectly-sharp short exposure both have few stars.

Together, the product fires on whichever defect is present and stays
near 1 only when *all three* signals say the frame is sharp.

### Behavior on the 4 labeled testing-data frames

| Image | label | FWHM | ε | n_bright | score |
|---|---|---:|---:|---:|---:|
| `normal.png`              | sharp                    | 5.35  | 0.14 | 43 | **0.48** |
| `mit Satellit.png`        | sharp + satellite trail  | 6.26  | 0.25 | 25 | **0.36** |
| `Wackler mit Satellit.png`| motion blur + satellite  | 8.27  | 0.35 |  4 | **0.09** |
| `kleiner Wackler.png`     | motion blur              | 14.06 | 0.56 | 13 | **0.09** |

Order matches expectation. Satellite trail correctly does not
collapse the score (compare `mit Satellit` 0.36 vs blurred 0.09 — a
clean 4× separation).

## Calibration

`target_fwhm = 3.0 px` is a baseline "ideal seeing" target. It is
the FWHM at which `fwhm_score` saturates at 1.0 — sharper than this
doesn't earn additional credit. The choice is sensor-dependent: at
the ~1900×1080 RGB resolution of this dataset, `normal` measures FWHM
5.35, so 3.0 puts a clean unattainable ceiling above the real frames
(score caps at ~0.55) while preserving full dynamic range below.

If used on a different sensor / pixel scale, retune via `--target-fwhm`
on the CLI. The score remains internally consistent across frames
captured on the same setup as long as the target stays fixed.

## Limitations

1. **Needs ≥10 total detections and ≥4 in the bright top 30%.**
   Below that, `measure_psf` raises `InsufficientSources`. Frames
   that are *so* blurred no star peaks above 5σ produce no score
   rather than a misleading low one. The CLI reports this on stderr
   with exit code 2.
2. **No ground-truth Pearson correlation.** The dataset
   (`ilretho/Astrometry-testing-data`) ships no `blur_ratings.csv`,
   so the plan's `Pearson ≥ 0.85` acceptance gate could not be
   numerically evaluated. Validation here is qualitative on the four
   labeled frames plus five unlabeled session frames spanning
   FWHM 3.7–17.0 with score range 0.10–0.71.
3. **Satellite trail still slightly affects score.** `mit Satellit`
   (sharp + satellite) scored 0.36 vs `normal` 0.48. The trail
   fragments through the bright-source filter and inflates the median
   FWHM. The metric still cleanly separates it from blurred frames
   (0.36 vs 0.09), but if a satellite-aware mask is desired later, it
   would close the remaining 0.12 gap.
4. **Per-frame cost.** `source-extractor` subprocess overhead is on
   the order of a few hundred ms per frame; the original "<100 ms"
   plan target is not achievable while keeping a real PSF metric.
   For real-time use, batching frames into one `source-extractor`
   invocation, or switching to `sep` (in-process Python source
   extraction) would help.

## Usage

```bash
# Single image, normalized score
python -m astrometry.util.blur_score IMAGE
# 0.4794...

# Raw PSF measurement
python -m astrometry.util.blur_score IMAGE --raw
# fwhm=5.350 ellipticity=0.145 n_sources=43

# Override target FWHM (different sensor)
python -m astrometry.util.blur_score IMAGE --target-fwhm 4.0

# Benchmark over a directory
python -m astrometry.util.blur_score_benchmark --images-dir /path/to/frames

# Benchmark on specific paths
python -m astrometry.util.blur_score_benchmark --paths a.fits b.png c.png
```

## Files

- `util/blur_score.py` — module + CLI
- `util/test_blur_score.py` — synthetic-star unit tests
  (`python -m unittest astrometry.util.test_blur_score -v`)
- `util/blur_score_benchmark.py` — CSV benchmark CLI

## Dependencies

`numpy`, `scipy`, `astropy`, `Pillow`, and the `source-extractor`
binary. All are present in `astrometrynet/solver:test` (see
`docker/solver/common.dockerfile`); no new packages were added.
