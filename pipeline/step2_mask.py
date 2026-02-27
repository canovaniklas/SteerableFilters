"""
Step 2 – Mask prediction & two-pass cleanup.

Takes the ridge magnitude / orientation from Step 1, thresholds, then applies
a light cleanup pass, directional closing, and a heavy cleanup pass with
shape-based filtering.

Standalone usage:
    python -m pipeline.step2_mask --ridge ridge.npz --output mask.npz
"""
from __future__ import annotations

import argparse

import numpy as np
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)
from skimage.filters import threshold_otsu, threshold_li
from skimage.measure import label, regionprops

from pipeline.io_utils import PredictionResult


# ── Helpers ──────────────────────────────────────────────────────────────

def _line_se(length, width, angle):
    """Create a line structuring element at given angle."""
    size = max(length, 3)
    if size % 2 == 0:
        size += 1
    half = size // 2
    se = np.zeros((size, size), dtype=bool)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for t in np.linspace(-half, half, length * 3):
        x = int(round(half + t * cos_a))
        y = int(round(half - t * sin_a))
        for w in range(-width, width + 1):
            wx = int(round(x + w * sin_a))
            wy = int(round(y + w * cos_a))
            if 0 <= wx < size and 0 <= wy < size:
                se[wy, wx] = True
    return se


def _directional_closing(mask, orientation, length=15, width=1):
    """
    Anisotropic closing: close gaps along the local fiber direction.

    Discretizes orientations into N bins and applies closing with a
    rotated line SE for each bin, only affecting pixels in that bin.
    """
    n_bins = 12
    angle_bins = np.linspace(0, np.pi, n_bins, endpoint=False)
    result = mask.copy()
    for theta in angle_bins:
        se = _line_se(length, width, theta)
        angle_diff = np.abs(orientation - theta)
        angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
        bin_mask = angle_diff < (np.pi / n_bins)
        closed = binary_closing(mask, se)
        new_pixels = closed & ~mask & bin_mask
        result = result | new_pixels
    return result


# ── Public entry point ───────────────────────────────────────────────────

def run_mask_prediction(
    image,
    ridge_magnitude,
    ridge_orientation,
    threshold_method='otsu',
    min_object_size_light=100,
    min_object_size_heavy=1000,
    min_hole_size=50,
    morph_radius=2,
    min_eccentricity=0.9,
    min_fiber_length=250,
    min_elongation=3.0,
):
    """
    Threshold ridge magnitude and clean up with two-pass noise removal.

    Returns a PredictionResult with raw_mask, clean_mask, removed_mask, etc.
    """
    mag_norm = ridge_magnitude

    # Threshold
    if threshold_method == 'otsu':
        try:
            thresh = threshold_otsu(mag_norm)
        except ValueError:
            thresh = 0.5
    elif threshold_method == 'li':
        try:
            thresh = threshold_li(mag_norm)
        except ValueError:
            thresh = 0.5
    elif threshold_method == 'percentile':
        thresh = np.percentile(mag_norm, 85)
    else:
        thresh = 0.5

    raw_mask = mag_norm > thresh

    # ── Pass 1: light cleanup ────────────────────────────────────────────
    light = remove_small_objects(raw_mask, min_size=min_object_size_light)

    # ── Directional closing along fiber orientation ──────────────────────
    dir_closed = _directional_closing(
        light, ridge_orientation,
        length=morph_radius * 6 + 3, width=1,
    )

    # Small isotropic closing to smooth remaining tiny gaps
    merged = binary_closing(dir_closed, disk(morph_radius))

    # ── Pass 2: aggressive cleanup ───────────────────────────────────────
    merged = remove_small_objects(merged, min_size=min_object_size_heavy)
    merged = binary_opening(merged, disk(morph_radius))
    merged = remove_small_objects(merged, min_size=min_object_size_heavy)

    # Shape-based filtering
    large_object_thresh = min_fiber_length * min_fiber_length
    labeled = label(merged)
    props = regionprops(labeled)

    fiber_mask = np.zeros_like(merged)
    for prop in props:
        if prop.area < min_object_size_heavy:
            continue
        if prop.area >= large_object_thresh:
            fiber_mask[labeled == prop.label] = True
            continue
        eccentricity = prop.eccentricity
        major_axis = prop.major_axis_length
        minor_axis = prop.minor_axis_length + 1e-10
        elongation = major_axis / minor_axis
        is_fiber = (
            (eccentricity > min_eccentricity and elongation > min_elongation)
            or (major_axis > min_fiber_length)
        )
        if is_fiber:
            fiber_mask[labeled == prop.label] = True

    clean = remove_small_holes(fiber_mask.astype(bool), area_threshold=min_hole_size)
    removed = raw_mask & ~clean

    return PredictionResult(
        raw_mask=raw_mask,
        clean_mask=clean,
        removed_mask=removed,
        ridge_magnitude=mag_norm,
        ridge_orientation=ridge_orientation,
        threshold=thresh,
    )


# ── Standalone CLI ───────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description='Step 2: Mask prediction & cleanup')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--ridge', required=True, help='ridge.npz from step 1')
    parser.add_argument('--threshold', choices=['otsu', 'li', 'percentile'], default='otsu')
    parser.add_argument('--min-object-light', type=int, default=100)
    parser.add_argument('--min-object-heavy', type=int, default=1000)
    parser.add_argument('--min-hole', type=int, default=50)
    parser.add_argument('--morph-radius', type=int, default=2)
    parser.add_argument('--min-eccentricity', type=float, default=0.9)
    parser.add_argument('--min-fiber-length', type=float, default=250)
    parser.add_argument('--min-elongation', type=float, default=3.0)
    parser.add_argument('--output', '-o', default='mask_output.npz')
    args = parser.parse_args()

    from pipeline.io_utils import load_image
    image = load_image(args.image)
    data = np.load(args.ridge)
    magnitude, orientation = data['magnitude'], data['orientation']

    print("Running mask prediction...")
    result = run_mask_prediction(
        image, magnitude, orientation,
        threshold_method=args.threshold,
        min_object_size_light=args.min_object_light,
        min_object_size_heavy=args.min_object_heavy,
        min_hole_size=args.min_hole,
        morph_radius=args.morph_radius,
        min_eccentricity=args.min_eccentricity,
        min_fiber_length=args.min_fiber_length,
        min_elongation=args.min_elongation,
    )
    print(f"Threshold: {result.threshold:.4f}")
    print(f"Clean mask pixels: {int(result.clean_mask.sum())}")

    np.savez_compressed(
        args.output,
        raw_mask=result.raw_mask,
        clean_mask=result.clean_mask,
        removed_mask=result.removed_mask,
        threshold=result.threshold,
    )
    print(f"Saved -> {args.output}")


if __name__ == '__main__':
    _cli()
