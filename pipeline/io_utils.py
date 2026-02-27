"""
Shared I/O helpers and data classes used across pipeline steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    removed_mask: np.ndarray
    ridge_magnitude: np.ndarray
    ridge_orientation: np.ndarray
    threshold: float


@dataclass
class SkeletonGraph:
    nodes: dict          # node_id -> {'pos': (y,x), 'degree': int, 'type': str}
    edges: list          # [(n1, n2, {'path': [...], 'length': int})]
    junction_pixels: np.ndarray
    endpoint_pixels: np.ndarray


@dataclass
class Fork:
    position: tuple
    node_id: int
    degree: int
    fork_type: str       # 'normal' or 'reversed'
    branch_angles: list
    branch_lengths: list


# ── Image / mask loading ─────────────────────────────────────────────────

def load_image(path):
    """Load an image file and normalise to [0, 1] float64 grayscale."""
    p = Path(path)
    if p.suffix.lower() in ('.tif', '.tiff'):
        try:
            import tifffile
            img = tifffile.imread(str(p))
        except ImportError:
            from PIL import Image
            img = np.array(Image.open(p))
    else:
        from PIL import Image
        img = np.array(Image.open(p))
    if img.ndim == 3:
        img = np.mean(img[..., :3], axis=-1)
    if img.ndim > 2:
        img = img[img.shape[0] // 2]
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img


def load_mask(path):
    """Load a mask file and return a boolean array."""
    p = Path(path)
    if p.suffix.lower() in ('.tif', '.tiff'):
        try:
            import tifffile
            mask = tifffile.imread(str(p))
        except ImportError:
            from PIL import Image
            mask = np.array(Image.open(p))
    else:
        from PIL import Image
        mask = np.array(Image.open(p))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim > 2:
        mask = mask[0]
    return mask.astype(bool)


def match_images_masks(image_dir, mask_dir):
    """Match image files to mask files by stem name."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    img_exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in img_exts)
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in img_exts}
    pairs, unmatched = [], []
    for img_path in images:
        stem = img_path.stem
        if stem in masks:
            pairs.append((img_path, masks[stem]))
            continue
        found = False
        for mask_stem, mask_path in masks.items():
            if mask_stem.startswith(stem) or stem.startswith(mask_stem):
                pairs.append((img_path, mask_path))
                found = True
                break
        if not found:
            unmatched.append(img_path)
    return pairs, unmatched
