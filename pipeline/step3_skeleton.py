"""
Step 3 – Skeletonization with iterative branch pruning.

Takes the cleaned binary mask from Step 2 and produces a thinned skeleton,
then iteratively removes short spur branches (endpoint-to-junction only).

Standalone usage:
    python -m pipeline.step3_skeleton --mask mask.npz --prune-length 25 --output skel.npz
"""
from __future__ import annotations

import argparse

import numpy as np
from skimage.morphology import skeletonize


# ── Helpers ──────────────────────────────────────────────────────────────

def _neighbor_count(skel):
    """Count 8-connected neighbors for each skeleton pixel."""
    count = np.zeros_like(skel, dtype=np.int32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            count += np.roll(np.roll(skel.astype(np.int32), di, axis=0), dj, axis=1)
    return count * skel


def prune_skeleton(skel, min_branch_length=25, iterations=10):
    """
    Iteratively remove short spur branches from skeleton.

    ONLY removes branches that go from an endpoint to a junction
    (actual spurs/wiggles). Never removes endpoint-to-endpoint paths,
    which are real fiber segments.
    """
    skel = skel.copy().astype(bool)
    h, w = skel.shape

    for _ in range(iterations):
        changed = False
        neighbors = _neighbor_count(skel)
        endpoints = skel & (neighbors == 1)
        ep_coords = np.argwhere(endpoints)

        for ey, ex in ep_coords:
            if not skel[ey, ex]:
                continue

            path = [(ey, ex)]
            cy, cx = ey, ex
            prev_y, prev_x = -1, -1
            reached_junction = False

            max_trace = min_branch_length + 5
            for _ in range(max_trace):
                found_next = False
                for di in (-1, 0, 1):
                    if found_next or reached_junction:
                        break
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ny, nx = cy + di, cx + dj
                        if (ny, nx) == (prev_y, prev_x):
                            continue
                        if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                            n_count = neighbors[ny, nx]
                            if n_count >= 3:
                                reached_junction = True
                                break
                            path.append((ny, nx))
                            prev_y, prev_x = cy, cx
                            cy, cx = ny, nx
                            found_next = True
                            break
                if not found_next or reached_junction:
                    break

            if reached_junction and len(path) < min_branch_length:
                for py, px in path:
                    skel[py, px] = False
                changed = True

        if not changed:
            break
        neighbors = _neighbor_count(skel)

    return skel


# ── Public entry point ───────────────────────────────────────────────────

def run_skeletonize(clean_mask, min_branch_length=25):
    """
    Skeletonize *clean_mask* and prune short spur branches.

    Returns
    -------
    skel : ndarray        – raw skeleton (bool)
    skel_pruned : ndarray – pruned skeleton (bool)
    """
    skel = skeletonize(clean_mask > 0)
    skel_pruned = prune_skeleton(skel, min_branch_length=min_branch_length)
    return skel, skel_pruned


# ── Standalone CLI ───────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description='Step 3: Skeletonization & pruning')
    parser.add_argument('--mask', required=True, help='mask.npz from step 2')
    parser.add_argument('--prune-length', type=int, default=25)
    parser.add_argument('--output', '-o', default='skel_output.npz')
    args = parser.parse_args()

    data = np.load(args.mask)
    clean_mask = data['clean_mask']

    print("Running skeletonization...")
    skel, skel_pruned = run_skeletonize(clean_mask, min_branch_length=args.prune_length)
    print(f"Skeleton: {int(skel.sum())} -> {int(skel_pruned.sum())} px (pruned)")

    np.savez_compressed(args.output, skel=skel, skel_pruned=skel_pruned)
    print(f"Saved -> {args.output}")


if __name__ == '__main__':
    _cli()
