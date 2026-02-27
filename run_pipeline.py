#!/usr/bin/env python
"""
Run the full TEM replication-fork analysis pipeline.

All hyperparameters live in the PipelineConfig at the top of this file.
Edit them there, or override via command-line flags.

Usage:
    python run_pipeline.py \
        --images /path/to/images_4096 \
        --masks  /path/to/masks_4096 \
        --output ./results

Each pipeline step can also be run independently – see the individual
modules under ``pipeline/``.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.io_utils import load_image, load_mask, match_images_masks
from pipeline.step1_ridge import run_ridge_filter
from pipeline.step2_mask import run_mask_prediction
from pipeline.step3_skeleton import run_skeletonize
from pipeline.step4_graph import run_graph_extraction
from pipeline.step5_forks import run_fork_detection
from pipeline.step6_visualize import (
    compute_metrics,
    run_visualization,
    plot_summary,
)


# =====================================================================
#  HYPERPARAMETERS — edit defaults here or override via CLI
# =====================================================================
DEFAULT_CONFIG = PipelineConfig(
    # I/O
    images_dir="",
    masks_dir="",
    output_dir="./steerable_results",
    max_images=None,
    # Step 1: Ridge filter
    sigmas=[2.0, 4.0],
    ridge_method="ridge4",
    # Step 2: Mask prediction
    threshold_method="otsu",
    min_object_size_light=100,
    min_object_size_heavy=1000,
    min_hole_size=50,
    morph_radius=2,
    min_eccentricity=0.9,
    min_fiber_length=250,
    min_elongation=3.0,
    # Step 3: Skeletonization
    prune_length=25,
    # Step 4: Graph
    min_edge_length=50,
    # Step 5: Fork detection
    min_arm_length=50,
    max_cycle_length=300,
)


# =====================================================================
#  CLI — command-line overrides for any config field
# =====================================================================
def _parse_args() -> PipelineConfig:
    """Build config from DEFAULT_CONFIG + CLI overrides."""
    p = argparse.ArgumentParser(
        description="Steerable filter -> skeleton -> graph -> fork detection")

    # I/O
    p.add_argument("--images", "-i", type=str, required=True)
    p.add_argument("--masks", "-m", type=str, required=True)
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--max-images", type=int, default=None)

    # Step 1
    p.add_argument("--sigma", "-s", nargs="+", type=float, default=None)
    p.add_argument("--method", choices=["ridge4", "ridge"], default=None)

    # Step 2
    p.add_argument("--threshold", choices=["otsu", "li", "percentile"], default=None)
    p.add_argument("--min-object-light", type=int, default=None)
    p.add_argument("--min-object-heavy", type=int, default=None)
    p.add_argument("--min-hole", type=int, default=None)
    p.add_argument("--morph-radius", type=int, default=None)
    p.add_argument("--min-eccentricity", type=float, default=None)
    p.add_argument("--min-fiber-length", type=float, default=None)
    p.add_argument("--min-elongation", type=float, default=None)

    # Step 3
    p.add_argument("--prune-length", type=int, default=None)

    # Step 4
    p.add_argument("--min-edge-length", type=int, default=None)

    # Step 5
    p.add_argument("--min-arm-length", type=int, default=None)
    p.add_argument("--max-cycle-length", type=int, default=None)

    args = p.parse_args()

    # Start from DEFAULT_CONFIG, override only what the user passed
    cfg = PipelineConfig(
        images_dir=args.images,
        masks_dir=args.masks,
        output_dir=args.output or DEFAULT_CONFIG.output_dir,
        max_images=args.max_images if args.max_images is not None else DEFAULT_CONFIG.max_images,
        sigmas=args.sigma if args.sigma is not None else DEFAULT_CONFIG.sigmas,
        ridge_method=args.method if args.method is not None else DEFAULT_CONFIG.ridge_method,
        threshold_method=args.threshold if args.threshold is not None else DEFAULT_CONFIG.threshold_method,
        min_object_size_light=args.min_object_light if args.min_object_light is not None else DEFAULT_CONFIG.min_object_size_light,
        min_object_size_heavy=args.min_object_heavy if args.min_object_heavy is not None else DEFAULT_CONFIG.min_object_size_heavy,
        min_hole_size=args.min_hole if args.min_hole is not None else DEFAULT_CONFIG.min_hole_size,
        morph_radius=args.morph_radius if args.morph_radius is not None else DEFAULT_CONFIG.morph_radius,
        min_eccentricity=args.min_eccentricity if args.min_eccentricity is not None else DEFAULT_CONFIG.min_eccentricity,
        min_fiber_length=args.min_fiber_length if args.min_fiber_length is not None else DEFAULT_CONFIG.min_fiber_length,
        min_elongation=args.min_elongation if args.min_elongation is not None else DEFAULT_CONFIG.min_elongation,
        prune_length=args.prune_length if args.prune_length is not None else DEFAULT_CONFIG.prune_length,
        min_edge_length=args.min_edge_length if args.min_edge_length is not None else DEFAULT_CONFIG.min_edge_length,
        min_arm_length=args.min_arm_length if args.min_arm_length is not None else DEFAULT_CONFIG.min_arm_length,
        max_cycle_length=args.max_cycle_length if args.max_cycle_length is not None else DEFAULT_CONFIG.max_cycle_length,
    )
    return cfg


# =====================================================================
#  Pipeline runner
# =====================================================================
def run(cfg: PipelineConfig):
    """Execute the full pipeline for all image/mask pairs."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Match images to masks ────────────────────────────────────────
    print("Matching images to masks...")
    pairs, unmatched = match_images_masks(cfg.images_dir, cfg.masks_dir)
    if not pairs:
        print("ERROR: No matched pairs!")
        sys.exit(1)
    print(f"Found {len(pairs)} pairs")
    if cfg.max_images:
        pairs = pairs[: cfg.max_images]

    all_metrics = []
    all_fork_counts: dict[str, int] = defaultdict(int)

    print(
        f"\nSettings: sigma={cfg.sigmas}, "
        f"min_object={cfg.min_object_size_light}/{cfg.min_object_size_heavy} (light/heavy), "
        f"min_fiber_length={cfg.min_fiber_length}, prune={cfg.prune_length}"
    )
    print("=" * 70)

    for idx, (img_path, mask_path) in enumerate(pairs):
        name = img_path.stem
        print(f"\n[{idx + 1}/{len(pairs)}] {name}")

        # ── Load ─────────────────────────────────────────────────────
        image = load_image(img_path)
        gt_mask = load_mask(mask_path)
        print(f"  Shape: {image.shape}, GT: {gt_mask.mean() * 100:.1f}%")

        if image.shape != gt_mask.shape:
            from scipy.ndimage import zoom
            factors = (
                image.shape[0] / gt_mask.shape[0],
                image.shape[1] / gt_mask.shape[1],
            )
            gt_mask = zoom(gt_mask.astype(float), factors, order=0) > 0.5

        # ── Step 1: Ridge filter ─────────────────────────────────────
        print("  Step 1: Ridge filter...")
        magnitude, orientation = run_ridge_filter(
            image, sigmas=cfg.sigmas, method=cfg.ridge_method,
        )

        # ── Step 2: Mask prediction ──────────────────────────────────
        print("  Step 2: Mask prediction...")
        result = run_mask_prediction(
            image, magnitude, orientation,
            threshold_method=cfg.threshold_method,
            min_object_size_light=cfg.min_object_size_light,
            min_object_size_heavy=cfg.min_object_size_heavy,
            min_hole_size=cfg.min_hole_size,
            morph_radius=cfg.morph_radius,
            min_eccentricity=cfg.min_eccentricity,
            min_fiber_length=cfg.min_fiber_length,
            min_elongation=cfg.min_elongation,
        )
        metrics_clean = compute_metrics(result.clean_mask, gt_mask)
        print(
            f"  F1={metrics_clean['f1']:.3f} "
            f"P={metrics_clean['precision']:.3f} "
            f"R={metrics_clean['recall']:.3f}"
        )

        # ── Step 3: Skeletonize ──────────────────────────────────────
        print("  Step 3: Skeletonization...")
        skel, skel_pruned = run_skeletonize(
            result.clean_mask, min_branch_length=cfg.prune_length,
        )
        print(f"  Skeleton: {int(skel.sum())} -> {int(skel_pruned.sum())} px")

        # ── Step 4: Graph extraction ─────────────────────────────────
        print("  Step 4: Graph extraction...")
        graph_raw, graph = run_graph_extraction(
            skel_pruned, min_edge_length=cfg.min_edge_length,
        )
        n_junctions = sum(1 for n in graph.nodes.values() if n['degree'] >= 3)
        print(
            f"  Graph: {len(graph.nodes)} nodes ({n_junctions} junctions), "
            f"{len(graph.edges)} edges"
        )

        # ── Step 5: Fork detection ───────────────────────────────────
        print("  Step 5: Fork detection...")
        forks, cycles = run_fork_detection(
            graph, skel_pruned,
            min_arm_length=cfg.min_arm_length,
            max_cycle_length=cfg.max_cycle_length,
        )
        fc: dict[str, int] = defaultdict(int)
        for f in forks:
            fc[f.fork_type] += 1
            all_fork_counts[f.fork_type] += 1
        print(f"  Forks: {dict(fc)}, Cycles: {len(cycles)}")

        metrics_raw = compute_metrics(result.raw_mask, gt_mask)
        all_metrics.append({
            'name': name[:30], 'raw': metrics_raw, 'clean': metrics_clean,
        })

        # ── Step 6: Visualization ────────────────────────────────────
        save_path = str(output_dir / f"{name}_analysis.png")
        print("  Step 6: Visualization...")
        run_visualization(
            image, result, gt_mask, metrics_clean,
            skel, skel_pruned, graph, forks, cycles,
            name=name, save_path=save_path,
        )
        print(f"  -> {save_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    clean_f1s = [m['clean']['f1'] for m in all_metrics]
    print(f"F1: {np.mean(clean_f1s):.3f} +/- {np.std(clean_f1s):.3f}")
    print(f"Forks: {dict(all_fork_counts)}")

    plot_summary(all_metrics, str(output_dir / "summary.png"))

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        f.write("image,raw_f1,clean_f1,clean_iou,clean_p,clean_r\n")
        for m in all_metrics:
            f.write(
                f"{m['name']},{m['raw']['f1']:.4f},{m['clean']['f1']:.4f},"
                f"{m['clean']['iou']:.4f},{m['clean']['precision']:.4f},"
                f"{m['clean']['recall']:.4f}\n"
            )

    print(f"\nResults in: {output_dir}/")


# =====================================================================
#  Entry point
# =====================================================================
if __name__ == "__main__":
    cfg = _parse_args()
    run(cfg)
