from __future__ import annotations
r"""
Steerable Filter -> Skeleton -> Graph -> Fork Detection
========================================================

Full pipeline for TEM replication fork analysis:
  1. Steerable ridge filter (Jacob & Unser 4th-order)
  2. Aggressive noise removal (pre-close small object removal + shape filtering)
  3. Skeletonization with branch pruning
  4. Graph extraction from skeleton
  5. Fork detection: degree-3/4 nodes classified as normal vs reversed forks

Usage:
  python steerable_predict.py \
    --images /path/to/images_4096 \
    --masks /path/to/masks_4096 \
    --sigma 2.0 4.0 \
    --output ./results
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    skeletonize,
    disk,
)
from skimage.filters import threshold_otsu, threshold_li
from skimage.measure import label, regionprops


# =============================================================================
# Gaussian Derivative Kernels
# =============================================================================

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


# =============================================================================
# Steerable Ridge Filters
# =============================================================================

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


# =============================================================================
# Mask Prediction Pipeline
# =============================================================================

@dataclass
class PredictionResult:
    raw_mask: np.ndarray
    clean_mask: np.ndarray
    removed_mask: np.ndarray
    ridge_magnitude: np.ndarray
    ridge_orientation: np.ndarray
    threshold: float


def _directional_closing(mask, orientation, length=15, width=1):
    """
    Anisotropic closing: close gaps along the local fiber direction.

    For each pixel, we create a line structuring element aligned with the
    local ridge orientation, then apply closing. This bridges gaps along
    fibers without connecting orthogonal structures.

    We discretize orientations into N bins and apply closing with a
    rotated line SE for each bin, only affecting pixels in that bin.
    """
    n_bins = 12
    angle_bins = np.linspace(0, np.pi, n_bins, endpoint=False)
    result = mask.copy()

    for theta in angle_bins:
        # Create rotated line structuring element
        se = _line_se(length, width, theta)

        # Mask of pixels whose orientation is close to this bin
        angle_diff = np.abs(orientation - theta)
        # Wrap around pi
        angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
        bin_mask = angle_diff < (np.pi / n_bins)

        # Apply closing only where orientation matches
        closed = binary_closing(mask, se)

        # Only accept new pixels near existing fiber-oriented regions
        new_pixels = closed & ~mask & bin_mask
        result = result | new_pixels

    return result


def _line_se(length, width, angle):
    """Create a line structuring element at given angle."""
    # Size must be odd
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


def predict_mask(
    image,
    sigmas=(2.0,),
    method='ridge4',
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
    Full pipeline: steerable filter -> threshold -> two-pass cleanup.

    Pass 1 (light): remove only tiny isolated speckles.
    Close with large disk to merge fiber fragments.
    Pass 2 (heavy): aggressive size + shape filtering on merged components.
    """
    # Step 1: Multi-scale ridge detection
    magnitude, orientation = multiscale_ridge(image, sigmas, method)
    mag_norm = magnitude / (magnitude.max() + 1e-10)

    # Step 2: Threshold
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

    # ---- Pass 1: Light cleanup (preserve fiber fragments) ----
    light = remove_small_objects(raw_mask, min_size=min_object_size_light)

    # ---- Directional closing: bridge gaps along fiber direction ----
    # This uses the ridge orientation to close only along the local
    # fiber direction, avoiding orthogonal merging
    dir_closed = _directional_closing(light, orientation, length=morph_radius * 6 + 3,
                                       width=1)

    # Small isotropic closing to smooth out remaining tiny gaps
    merged = binary_closing(dir_closed, disk(morph_radius))

    # ---- Pass 2: Aggressive cleanup on merged mask ----
    # Remove small objects that didn't merge into fibers
    merged = remove_small_objects(merged, min_size=min_object_size_heavy)
    # Opening to break remaining thin noise bridges
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
            (eccentricity > min_eccentricity and elongation > min_elongation) or
            (major_axis > min_fiber_length)
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
        ridge_orientation=orientation,
        threshold=thresh,
    )


# =============================================================================
# Skeletonization with Branch Pruning
# =============================================================================

def _neighbor_count(skel):
    """Count 8-connected neighbors for each pixel in skeleton."""
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

            # Trace from endpoint until we hit a junction, another endpoint,
            # or exhaust max trace length
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

            # Only prune if: short AND connects to a junction (actual spur)
            # Never prune endpoint-to-endpoint paths (real fiber segments)
            if reached_junction and len(path) < min_branch_length:
                for py, px in path:
                    skel[py, px] = False
                changed = True

        if not changed:
            break
        neighbors = _neighbor_count(skel)

    return skel


def skeletonize_mask(mask, min_branch_length=25):
    """Skeletonize binary mask and prune short branches."""
    skel = skeletonize(mask > 0)
    skel_pruned = prune_skeleton(skel, min_branch_length=min_branch_length)
    return skel, skel_pruned


# =============================================================================
# Graph Extraction from Skeleton
# =============================================================================

@dataclass
class SkeletonGraph:
    nodes: dict       # node_id -> {'pos': (y,x), 'degree': int, 'type': str}
    edges: list       # [(n1, n2, {'path': [...], 'length': int})]
    junction_pixels: np.ndarray
    endpoint_pixels: np.ndarray


def extract_graph(skel):
    """
    Extract graph from skeleton.
    Nodes = junctions (degree>=3) + endpoints (degree=1).
    Edges = paths connecting nodes.
    """
    skel = skel.astype(bool)
    h, w = skel.shape
    neighbors = _neighbor_count(skel)

    junctions = skel & (neighbors >= 3)
    endpoints = skel & (neighbors == 1)

    # Cluster nearby junction pixels into single nodes
    junction_labeled = label(ndimage.binary_dilation(junctions, disk(2)) & skel)
    nodes = {}
    node_id = 0
    pixel_to_node = {}

    for region in regionprops(junction_labeled):
        coords = region.coords
        center_y = int(np.mean(coords[:, 0]))
        center_x = int(np.mean(coords[:, 1]))
        nodes[node_id] = {
            'pos': (center_y, center_x),
            'degree': 0,
            'type': 'junction',
            'coords': set(map(tuple, coords)),
        }
        for y, x in coords:
            pixel_to_node[(y, x)] = node_id
        node_id += 1

    ep_coords = np.argwhere(endpoints)
    for ey, ex in ep_coords:
        if (ey, ex) not in pixel_to_node:
            nodes[node_id] = {
                'pos': (ey, ex),
                'degree': 0,
                'type': 'endpoint',
                'coords': {(ey, ex)},
            }
            pixel_to_node[(ey, ex)] = node_id
            node_id += 1

    # Trace edges
    edges = []
    visited_edges = set()

    for nid, node_info in nodes.items():
        for sy, sx in node_info['coords']:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ny, nx = sy + di, sx + dj
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if not skel[ny, nx]:
                        continue
                    if (ny, nx) in node_info['coords']:
                        continue

                    path = [(ny, nx)]
                    cy, cx = ny, nx
                    prev = (sy, sx)
                    target_node = None

                    for _ in range(max(h, w)):
                        if (cy, cx) in pixel_to_node:
                            target_node = pixel_to_node[(cy, cx)]
                            break
                        found = False
                        for di2 in (-1, 0, 1):
                            for dj2 in (-1, 0, 1):
                                if di2 == 0 and dj2 == 0:
                                    continue
                                ny2, nx2 = cy + di2, cx + dj2
                                if (ny2, nx2) == prev:
                                    continue
                                if 0 <= ny2 < h and 0 <= nx2 < w and skel[ny2, nx2]:
                                    path.append((ny2, nx2))
                                    prev = (cy, cx)
                                    cy, cx = ny2, nx2
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            break

                    if target_node is not None and target_node != nid:
                        edge_key = (min(nid, target_node), max(nid, target_node))
                        if edge_key not in visited_edges:
                            visited_edges.add(edge_key)
                            edges.append((nid, target_node, {
                                'path': path,
                                'length': len(path),
                            }))

    degree_count = defaultdict(int)
    for n1, n2, _ in edges:
        degree_count[n1] += 1
        degree_count[n2] += 1
    for nid in nodes:
        nodes[nid]['degree'] = degree_count.get(nid, 0)

    return SkeletonGraph(
        nodes=nodes, edges=edges,
        junction_pixels=junctions, endpoint_pixels=endpoints,
    )


def prune_graph(graph, min_edge_length=50):
    """
    Remove short edges and isolated small components from the graph.

    - Edges shorter than min_edge_length between two endpoints (not junctions)
      are removed (tiny disconnected skeleton fragments).
    - Edges from endpoint to junction shorter than min_edge_length are removed
      (short spurs that survived skeleton pruning).
    - After edge removal, orphaned nodes (degree 0) are cleaned up.
    """
    kept_edges = []
    removed_node_ids = set()

    for n1, n2, edata in graph.edges:
        length = edata['length']
        d1 = graph.nodes[n1]['degree']
        d2 = graph.nodes[n2]['degree']

        # Both endpoints and short: isolated tiny fragment -> remove
        if d1 <= 1 and d2 <= 1 and length < min_edge_length:
            removed_node_ids.add(n1)
            removed_node_ids.add(n2)
            continue

        # Endpoint to junction and short: residual spur -> remove
        if length < min_edge_length // 2:
            if (d1 <= 1 and d2 >= 3) or (d1 >= 3 and d2 <= 1):
                ep = n1 if d1 <= 1 else n2
                removed_node_ids.add(ep)
                continue

        kept_edges.append((n1, n2, edata))

    # Rebuild degree counts
    degree_count = defaultdict(int)
    for n1, n2, _ in kept_edges:
        degree_count[n1] += 1
        degree_count[n2] += 1

    # Clean up orphaned nodes
    kept_nodes = {}
    for nid, node in graph.nodes.items():
        if nid in removed_node_ids and degree_count.get(nid, 0) == 0:
            continue
        node_copy = dict(node)
        node_copy['degree'] = degree_count.get(nid, 0)
        kept_nodes[nid] = node_copy

    return SkeletonGraph(
        nodes=kept_nodes, edges=kept_edges,
        junction_pixels=graph.junction_pixels,
        endpoint_pixels=graph.endpoint_pixels,
    )


# =============================================================================
# Fork Detection & Classification
# =============================================================================

@dataclass
class Fork:
    position: tuple
    node_id: int
    degree: int
    fork_type: str     # 'normal' or 'reversed'
    branch_angles: list
    branch_lengths: list


def _angle_diff(a1, a2):
    d = a1 - a2
    return (d + np.pi) % (2 * np.pi) - np.pi


def classify_forks(graph, skel, min_branch_for_fork=30, min_arm_length=50):
    """
    Classify junction nodes as fork types based on branch angles.

    Only classifies junctions where ALL connected arms have true length
    >= min_arm_length. True length = edge path length + Euclidean distance
    from junction center to start of edge path.
    """
    forks = []

    for nid, node in graph.nodes.items():
        if node['degree'] < 3:
            continue

        jy, jx = node['pos']
        branch_info = []

        for n1, n2, edata in graph.edges:
            if n1 != nid and n2 != nid:
                continue
            path = edata['path']
            edge_length = edata['length']
            if edge_length < 3:
                continue

            # True arm length: distance from junction center through
            # the edge path to the other node
            other = n2 if n1 == nid else n1
            oy, ox = graph.nodes[other]['pos']
            # Include distance from junction center to edge start
            if path:
                start_y, start_x = path[0]
                junction_offset = np.sqrt((jy - start_y)**2 + (jx - start_x)**2)
            else:
                junction_offset = 0
            true_length = edge_length + junction_offset

            sample_dist = min(20, len(path) - 1)
            py, px = path[sample_dist]
            angle = np.arctan2(py - jy, px - jx)
            branch_info.append({
                'angle': angle,
                'length': edge_length,
                'true_length': true_length,
                'other_node': other,
            })

        if len(branch_info) < 3:
            continue

        # All arms must be at least min_arm_length (using true length)
        shortest_arm = min(b['true_length'] for b in branch_info)
        if shortest_arm < min_arm_length:
            continue

        angles = [b['angle'] for b in branch_info]
        lengths = [b['true_length'] for b in branch_info]

        fork_type = _classify_junction(angles, lengths, node['degree'])

        forks.append(Fork(
            position=(jy, jx), node_id=nid, degree=node['degree'],
            fork_type=fork_type, branch_angles=angles, branch_lengths=lengths,
        ))

    return forks


def _classify_junction(angles, lengths, degree):
    """Classify as normal fork or reversed fork based on branch angles."""
    n = len(angles)
    if n < 3:
        return 'normal'  # default

    sorted_idx = np.argsort(lengths)[::-1]
    parent_angle = angles[sorted_idx[0]]
    child_angles = [angles[sorted_idx[i]] for i in range(1, min(3, n))]
    child_diffs = [abs(_angle_diff(ca, parent_angle)) for ca in child_angles]

    # Check for two arms nearly parallel (reversed fork signature)
    min_pair_diff = np.pi
    for i in range(n):
        for j in range(i+1, n):
            pd = abs(_angle_diff(angles[i], angles[j]))
            min_pair_diff = min(min_pair_diff, pd)

    # Reversed: two arms point in nearly the same direction
    if min_pair_diff < 0.5:
        return 'reversed'

    return 'normal'


# =============================================================================
# Cycle Detection
# =============================================================================

def find_cycles(graph, max_cycle_length=500):
    adjacency = defaultdict(list)
    for n1, n2, edata in graph.edges:
        adjacency[n1].append((n2, edata['length']))
        adjacency[n2].append((n1, edata['length']))

    cycles = []
    found_cycles = set()

    for start in graph.nodes:
        stack = [(start, [start], 0)]
        visited_local = {start}

        while stack:
            current, path, total_len = stack.pop()
            for neighbor, edge_len in adjacency[current]:
                new_len = total_len + edge_len
                if new_len > max_cycle_length:
                    continue
                if neighbor == start and len(path) >= 3:
                    cycle_key = tuple(sorted(path))
                    if cycle_key not in found_cycles:
                        cycles.append({'nodes': list(path), 'length': new_len})
                        found_cycles.add(cycle_key)
                elif neighbor not in visited_local:
                    visited_local.add(neighbor)
                    stack.append((neighbor, path + [neighbor], new_len))

    return cycles


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    return {
        'precision': precision, 'recall': recall,
        'f1': f1, 'iou': iou,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
    }


# =============================================================================
# Visualization
# =============================================================================

def create_overlay(image, pred_mask, gt_mask, removed_mask=None):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    if image.ndim == 2:
        rgb = np.stack([image]*3, axis=-1)
    else:
        rgb = image[..., :3].copy()
    rgb = rgb.astype(np.float64)
    if rgb.max() > 1:
        rgb /= rgb.max()
    overlay = rgb.copy()
    a = 0.45
    overlay[pred & gt] = overlay[pred & gt] * (1-a) + np.array([0,1,0]) * a
    overlay[pred & ~gt] = overlay[pred & ~gt] * (1-a) + np.array([1,0,0]) * a
    overlay[~pred & gt] = overlay[~pred & gt] * (1-a) + np.array([0.2,0.4,1.0]) * a
    if removed_mask is not None:
        rm = removed_mask.astype(bool)
        overlay[rm] = overlay[rm] * (1-a) + np.array([1,1,0]) * a
    return np.clip(overlay, 0, 1)


def _fork_bounding_box(fork, graph, skel, padding=40):
    """
    Compute bounding box around a fork by collecting all skeleton pixels
    in the immediate neighborhood of the junction + its branches.
    """
    jy, jx = fork.position
    # Collect pixels from all edges connected to this fork node
    pixels = [(jy, jx)]
    for n1, n2, edata in graph.edges:
        if n1 != fork.node_id and n2 != fork.node_id:
            continue
        # Take up to 60px along each branch for the box
        for py, px in edata['path'][:60]:
            pixels.append((py, px))

    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    y_min = max(0, min(ys) - padding)
    y_max = min(skel.shape[0], max(ys) + padding)
    x_min = max(0, min(xs) - padding)
    x_max = min(skel.shape[1], max(xs) + padding)
    return y_min, y_max, x_min, x_max


FORK_COLORS = {
    'normal': '#00FF00', 'reversed': '#FF0000',
}


def plot_full_analysis(
    image, result, gt_mask, metrics_clean,
    skel, skel_pruned, graph, forks, cycles,
    title="", save_path=None,
):
    """
    8-panel pipeline figure — each step shown on the input image:
      Row 1: [1. Input]           [2. Ridge magnitude] [3. Raw mask on image]  [4. Cleaned mask on image]
      Row 2: [5. GT comparison]   [6. Skeleton on img] [7. Graph on image]     [8. Fork detections]
    """
    fig = plt.figure(figsize=(28, 14))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.15)

    def img_rgb():
        rgb = np.stack([image]*3, axis=-1).copy()
        if rgb.max() > 1:
            rgb /= rgb.max()
        return rgb

    # --- 1. Input ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap='gray')
    ax.set_title('1. Input TEM', fontweight='bold', fontsize=10)
    ax.axis('off')

    # --- 2. Ridge magnitude ---
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(image, cmap='gray', alpha=0.4)
    ax.imshow(result.ridge_magnitude, cmap='inferno', alpha=0.6)
    ax.set_title(f'2. Ridge Magnitude\nthresh={result.threshold:.3f}',
                 fontweight='bold', fontsize=10)
    ax.axis('off')

    # --- 3. Raw mask on image ---
    ax = fig.add_subplot(gs[0, 2])
    rgb = img_rgb()
    rgb[result.raw_mask] = [1, 0.3, 0.3]
    ax.imshow(rgb)
    ax.set_title(f'3. Raw Thresholded Mask\n{int(result.raw_mask.sum())} pixels',
                 fontweight='bold', fontsize=10)
    ax.axis('off')

    # --- 4. Cleaned mask on image ---
    ax = fig.add_subplot(gs[0, 3])
    rgb = img_rgb()
    rgb[result.clean_mask] = [0.2, 0.8, 0.2]
    rgb[result.removed_mask] = [1, 1, 0]
    ax.imshow(rgb)
    ax.set_title(f'4. Cleaned Mask (green)\n'
                 f'removed {int(result.removed_mask.sum())} px (yellow)',
                 fontweight='bold', fontsize=10)
    ax.axis('off')

    # --- 5. GT comparison ---
    ax = fig.add_subplot(gs[1, 0])
    overlay = create_overlay(image, result.clean_mask, gt_mask)
    ax.imshow(overlay)
    ax.set_title(
        f'5. vs Ground Truth\n'
        f'F1={metrics_clean["f1"]:.3f}  IoU={metrics_clean["iou"]:.3f}  '
        f'P={metrics_clean["precision"]:.3f}  R={metrics_clean["recall"]:.3f}',
        fontweight='bold', fontsize=10)
    ax.axis('off')

    # --- 6. Skeleton on image ---
    ax = fig.add_subplot(gs[1, 1])
    rgb = img_rgb()
    rgb[skel & ~skel_pruned] = [0, 0.8, 0.8]
    rgb[skel_pruned] = [1, 0, 1]
    ax.imshow(rgb)
    ax.set_title(
        f'6. Skeleton\n'
        f'{int(skel.sum())} -> {int(skel_pruned.sum())} px (pruned cyan)',
        fontweight='bold', fontsize=10)
    ax.axis('off')

    # --- 7. Graph on image ---
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(image, cmap='gray')
    for n1, n2, edata in graph.edges:
        path = edata['path']
        if len(path) > 1:
            py = [p[0] for p in path]
            px = [p[1] for p in path]
            ax.plot(px, py, '-', color='lime', linewidth=1.2, alpha=0.8)
    for nid, node in graph.nodes.items():
        y, x = node['pos']
        if node['degree'] >= 3:
            ax.plot(x, y, 'o', color='yellow', markersize=7,
                    markeredgecolor='black', markeredgewidth=1)
    n_junctions = sum(1 for n in graph.nodes.values() if n['degree'] >= 3)
    ax.set_title(
        f'7. Graph Topology\n'
        f'{len(graph.nodes)} nodes ({n_junctions} junctions), '
        f'{len(graph.edges)} edges',
        fontweight='bold', fontsize=10)
    ax.axis('off')

    # --- 8. Fork detections with bounding boxes ---
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(image, cmap='gray')

    n_normal = 0
    n_reversed = 0
    for fork in forks:
        y0, y1, x0, x1 = _fork_bounding_box(fork, graph, skel_pruned)
        color = FORK_COLORS.get(fork.fork_type, '#00FF00')
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=2.5, edgecolor=color,
                              facecolor='none', linestyle='-')
        ax.add_patch(rect)
        label = 'N' if fork.fork_type == 'normal' else 'R'
        ax.text(x0 + 3, y0 + 15, label,
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7))
        if fork.fork_type == 'normal':
            n_normal += 1
        else:
            n_reversed += 1

    for cycle in cycles:
        coords = [graph.nodes[n]['pos'] for n in cycle['nodes'] if n in graph.nodes]
        if coords:
            cy_c = np.mean([c[0] for c in coords])
            cx_c = np.mean([c[1] for c in coords])
            circle = plt.Circle((cx_c, cy_c), 35, fill=False, color='orange',
                                linewidth=2, linestyle='--')
            ax.add_patch(circle)

    ax.set_title(
        f'8. Fork Detection\n'
        f'{n_normal} normal (green), {n_reversed} reversed (red), '
        f'{len(cycles)} cycles',
        fontweight='bold', fontsize=10)
    ax.axis('off')

    # Legend as text below
    legend_patches = [
        mpatches.Patch(facecolor='none', edgecolor='#00FF00', linewidth=2,
                       label='Normal fork (Y-shape)'),
        mpatches.Patch(facecolor='none', edgecolor='#FF0000', linewidth=2,
                       label='Reversed fork (regression)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markeredgecolor='black', markersize=8, label='Junction node'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
                   label='Cycle'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=11, frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_summary(all_metrics, save_path=None):
    if not all_metrics:
        return
    names = [m['name'] for m in all_metrics]
    clean_f1 = [m['clean']['f1'] for m in all_metrics]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(12, len(names)*0.8), 6))
    ax.bar(x, clean_f1, color='lightgreen', edgecolor='darkgreen')
    ax.set_ylabel('F1 Score')
    ax.set_title('Clean F1 per Image', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.axhline(np.mean(clean_f1), color='green', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


# =============================================================================
# Image / Mask Loading
# =============================================================================

def load_image(path):
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Steerable filter -> skeleton -> graph -> fork detection')
    parser.add_argument('--images', '-i', type=str, required=True)
    parser.add_argument('--masks', '-m', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default='./steerable_results')
    parser.add_argument('--sigma', '-s', nargs='+', type=float, default=[2.0, 4.0])
    parser.add_argument('--method', choices=['ridge4', 'ridge'], default='ridge4')
    parser.add_argument('--threshold', choices=['otsu', 'li', 'percentile'], default='otsu')
    parser.add_argument('--min-object-light', type=int, default=100,
                        help='Light pass: min object size (default: 100)')
    parser.add_argument('--min-object-heavy', type=int, default=1000,
                        help='Heavy pass: min object size (default: 1000)')
    parser.add_argument('--min-hole', type=int, default=50)
    parser.add_argument('--morph-radius', type=int, default=2)
    parser.add_argument('--min-eccentricity', type=float, default=0.9)
    parser.add_argument('--min-fiber-length', type=float, default=250)
    parser.add_argument('--min-elongation', type=float, default=3.0)
    parser.add_argument('--prune-length', type=int, default=25,
                        help='Min spur branch length to prune (default: 25)')
    parser.add_argument('--min-edge-length', type=int, default=50,
                        help='Min graph edge length to keep (default: 50)')
    parser.add_argument('--min-arm-length', type=int, default=50,
                        help='Min length of shortest arm at a fork (default: 50)')
    parser.add_argument('--max-images', type=int, default=None)

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Matching images to masks...")
    pairs, unmatched = match_images_masks(args.images, args.masks)
    if not pairs:
        print("ERROR: No matched pairs!"); sys.exit(1)
    print(f"Found {len(pairs)} pairs")
    if args.max_images:
        pairs = pairs[:args.max_images]

    all_metrics = []
    all_fork_counts = defaultdict(int)

    print(f"\nSettings: sigma={args.sigma}, "
          f"min_object={args.min_object_light}/{args.min_object_heavy} (light/heavy), "
          f"min_fiber_length={args.min_fiber_length}, prune={args.prune_length}")
    print("=" * 70)

    for idx, (img_path, mask_path) in enumerate(pairs):
        name = img_path.stem
        print(f"\n[{idx+1}/{len(pairs)}] {name}")

        image = load_image(img_path)
        gt_mask = load_mask(mask_path)
        print(f"  Shape: {image.shape}, GT: {gt_mask.mean()*100:.1f}%")

        if image.shape != gt_mask.shape:
            from scipy.ndimage import zoom
            factors = (image.shape[0]/gt_mask.shape[0], image.shape[1]/gt_mask.shape[1])
            gt_mask = zoom(gt_mask.astype(float), factors, order=0) > 0.5

        # 1. Predict
        print("  Ridge filter...")
        result = predict_mask(
            image, sigmas=args.sigma, method=args.method,
            threshold_method=args.threshold,
            min_object_size_light=args.min_object_light,
            min_object_size_heavy=args.min_object_heavy,
            min_hole_size=args.min_hole,
            morph_radius=args.morph_radius,
            min_eccentricity=args.min_eccentricity,
            min_fiber_length=args.min_fiber_length,
            min_elongation=args.min_elongation,
        )
        metrics_raw = compute_metrics(result.raw_mask, gt_mask)
        metrics_clean = compute_metrics(result.clean_mask, gt_mask)
        print(f"  F1={metrics_clean['f1']:.3f} P={metrics_clean['precision']:.3f} "
              f"R={metrics_clean['recall']:.3f}")

        # 2. Skeletonize
        print("  Skeletonizing...")
        skel, skel_pruned = skeletonize_mask(result.clean_mask,
                                              min_branch_length=args.prune_length)
        print(f"  Skeleton: {int(skel.sum())} -> {int(skel_pruned.sum())} px")

        # 3. Graph
        print("  Graph extraction...")
        graph = extract_graph(skel_pruned)
        n_junctions = sum(1 for n in graph.nodes.values() if n['degree'] >= 3)
        print(f"  Raw graph: {len(graph.nodes)} nodes ({n_junctions} junctions), "
              f"{len(graph.edges)} edges")

        # Prune short edges
        graph = prune_graph(graph, min_edge_length=args.min_edge_length)
        n_junctions = sum(1 for n in graph.nodes.values() if n['degree'] >= 3)
        print(f"  Pruned graph: {len(graph.nodes)} nodes ({n_junctions} junctions), "
              f"{len(graph.edges)} edges")

        # 4. Forks
        print("  Fork classification...")
        forks = classify_forks(graph, skel_pruned, min_arm_length=args.min_arm_length)
        cycles = find_cycles(graph, max_cycle_length=300)

        fc = defaultdict(int)
        for f in forks:
            fc[f.fork_type] += 1
            all_fork_counts[f.fork_type] += 1
        print(f"  Forks: {dict(fc)}, Cycles: {len(cycles)}")

        all_metrics.append({'name': name[:30], 'raw': metrics_raw, 'clean': metrics_clean})

        # 5. Plot
        save_path = str(output_dir / f'{name}_analysis.png')
        plot_full_analysis(
            image, result, gt_mask, metrics_clean,
            skel, skel_pruned, graph, forks, cycles,
            title=name, save_path=save_path,
        )
        print(f"  -> {save_path}")

    # Summary
    print("\n" + "=" * 70)
    clean_f1s = [m['clean']['f1'] for m in all_metrics]
    print(f"F1: {np.mean(clean_f1s):.3f} +/- {np.std(clean_f1s):.3f}")
    print(f"Forks: {dict(all_fork_counts)}")

    plot_summary(all_metrics, str(output_dir / 'summary.png'))

    csv_path = output_dir / 'metrics.csv'
    with open(csv_path, 'w') as f:
        f.write('image,raw_f1,clean_f1,clean_iou,clean_p,clean_r\n')
        for m in all_metrics:
            f.write(f"{m['name']},{m['raw']['f1']:.4f},{m['clean']['f1']:.4f},"
                    f"{m['clean']['iou']:.4f},{m['clean']['precision']:.4f},"
                    f"{m['clean']['recall']:.4f}\n")

    print(f"\nResults in: {output_dir}/")


if __name__ == '__main__':
    main()