"""
Central configuration for all pipeline hyperparameters.

Edit the defaults here or override them from run_pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    """All tuneable hyperparameters for the pipeline, in one place."""

    # ── I/O ──────────────────────────────────────────────────────────────
    images_dir: str = ""
    masks_dir: str = ""
    output_dir: str = "./steerable_results"
    max_images: int | None = None

    # ── Step 1: Ridge filter ─────────────────────────────────────────────
    sigmas: List[float] = field(default_factory=lambda: [2.0, 4.0])
    ridge_method: str = "ridge4"          # 'ridge4' or 'ridge'

    # ── Step 2: Mask prediction & cleanup ────────────────────────────────
    threshold_method: str = "otsu"        # 'otsu', 'li', or 'percentile'
    min_object_size_light: int = 100      # pass-1 small-object removal
    min_object_size_heavy: int = 1000     # pass-2 small-object removal
    min_hole_size: int = 50               # hole-fill threshold
    morph_radius: int = 2                 # structuring-element radius
    min_eccentricity: float = 0.9         # shape filter
    min_fiber_length: float = 250         # minimum major-axis length
    min_elongation: float = 3.0           # major / minor axis ratio

    # ── Step 3: Skeletonization ──────────────────────────────────────────
    prune_length: int = 25                # min spur-branch length to keep

    # ── Step 4: Graph extraction ─────────────────────────────────────────
    min_edge_length: int = 50             # min graph-edge length to keep

    # ── Step 5: Fork detection ───────────────────────────────────────────
    min_arm_length: int = 50              # min arm length at a fork
    max_cycle_length: int = 300           # max total length for cycle search
