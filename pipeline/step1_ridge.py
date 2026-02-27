"""
Step 1 – Multi-scale ridge filter detection.

Applies steerable 4th-order (or Hessian) ridge filters at multiple scales
and returns the maximum-response magnitude + orientation maps.

Standalone usage:
    python -m pipeline.step1_ridge --image img.tif --sigma 2.0 4.0 --output ridge.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage


# ── Gaussian derivative kernels ──────────────────────────────────────────

def _hermite_factor(x, sigma, n):
    t = x / sigma
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return -t / sigma
    elif n == 2:
        return (t**2 - 1) / sigma**2
    elif n == 3:
        return (-t**3 + 3*t) / sigma**3
    elif n == 4:
        return (t**4 - 6*t**2 + 3) / sigma**4
    else:
        raise ValueError(f"Order {n} not implemented")


def gaussian_kernel_2d(sigma, order, size=None):
    if size is None:
        size = int(4 * np.ceil(sigma)) * 2 + 1
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, x)
    g = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kx = _hermite_factor(X, sigma, order[0])
    ky = _hermite_factor(Y, sigma, order[1])
    return kx * ky * g


# ── Ridge filters ────────────────────────────────────────────────────────

def hessian_ridge_filter(image, sigma):
    k_xx = gaussian_kernel_2d(sigma, (2, 0))
    k_xy = gaussian_kernel_2d(sigma, (1, 1))
    k_yy = gaussian_kernel_2d(sigma, (0, 2))
    r_xx = ndimage.convolve(image, k_xx)
    r_xy = ndimage.convolve(image, k_xy)
    r_yy = ndimage.convolve(image, k_yy)
    orientation = 0.5 * np.arctan2(2 * r_xy, r_xx - r_yy) % np.pi
    discriminant = np.sqrt((r_xx - r_yy)**2 + 4 * r_xy**2)
    lambda1 = 0.5 * (r_xx + r_yy + discriminant)
    lambda2 = 0.5 * (r_xx + r_yy - discriminant)
    magnitude = np.maximum(np.abs(lambda1), np.abs(lambda2))
    return magnitude, orientation


def steerable_ridge_4th(image, sigma):
    orders = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    kernels = [gaussian_kernel_2d(sigma, o) for o in orders]
    basis = [ndimage.convolve(image, k) for k in kernels]
    n_angles = 36
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    magnitude = np.zeros_like(image, dtype=np.float64)
    orientation = np.zeros_like(image, dtype=np.float64)
    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        coeffs = [c**4, 4*c**3*s, 6*c**2*s**2, 4*c*s**3, s**4]
        response = sum(ci * bi for ci, bi in zip(coeffs, basis))
        mask = np.abs(response) > magnitude
        magnitude = np.where(mask, np.abs(response), magnitude)
        orientation = np.where(mask, (theta + np.pi/2) % np.pi, orientation)
    return magnitude, orientation


def multiscale_ridge(image, sigmas, method='ridge4'):
    orientation = np.zeros_like(image, dtype=np.float64)
    magnitude = np.zeros_like(image, dtype=np.float64)
    for sigma in sigmas:
        if method == 'ridge4':
            mag, ori = steerable_ridge_4th(image, sigma)
        else:
            mag, ori = hessian_ridge_filter(image, sigma)
        normalized = mag * sigma**2
        better = normalized > magnitude
        magnitude = np.where(better, normalized, magnitude)
        orientation = np.where(better, ori, orientation)
    return magnitude, orientation


# ── Public entry point ───────────────────────────────────────────────────

def run_ridge_filter(image, sigmas=(2.0, 4.0), method='ridge4'):
    """
    Run multi-scale ridge detection on *image*.

    Returns
    -------
    magnitude : ndarray   – normalised ridge magnitude in [0, 1]
    orientation : ndarray  – ridge orientation in [0, pi)
    """
    magnitude, orientation = multiscale_ridge(image, sigmas, method)
    mag_norm = magnitude / (magnitude.max() + 1e-10)
    return mag_norm, orientation


# ── Standalone CLI ───────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description='Step 1: Ridge filter detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--sigma', nargs='+', type=float, default=[2.0, 4.0])
    parser.add_argument('--method', choices=['ridge4', 'ridge'], default='ridge4')
    parser.add_argument('--output', '-o', default='ridge_output.npz',
                        help='Output .npz file')
    args = parser.parse_args()

    from pipeline.io_utils import load_image
    image = load_image(args.image)
    print(f"Image shape: {image.shape}")

    print("Running ridge filter...")
    magnitude, orientation = run_ridge_filter(image, args.sigma, args.method)
    print(f"Magnitude range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")

    np.savez_compressed(args.output, magnitude=magnitude, orientation=orientation)
    print(f"Saved -> {args.output}")


if __name__ == '__main__':
    _cli()
