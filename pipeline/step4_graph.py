"""
Step 4 – Graph extraction & pruning.

Converts a pruned skeleton into a topology graph where nodes are junctions
(degree >= 3) and endpoints (degree 1), and edges are the skeleton paths
connecting them.  Short edges and isolated fragments are then pruned.

Standalone usage:
    python -m pipeline.step4_graph --skel skel.npz --min-edge-length 50 --output graph.npz
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import numpy as np
from scipy import ndimage
from skimage.morphology import disk
from skimage.measure import label, regionprops

from pipeline.io_utils import SkeletonGraph
from pipeline.step3_skeleton import _neighbor_count


# ── Graph extraction ─────────────────────────────────────────────────────

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


# ── Graph pruning ────────────────────────────────────────────────────────

def prune_graph(graph, min_edge_length=50):
    """
    Remove short edges and isolated small components from the graph.
    """
    kept_edges = []
    removed_node_ids = set()

    for n1, n2, edata in graph.edges:
        length = edata['length']
        d1 = graph.nodes[n1]['degree']
        d2 = graph.nodes[n2]['degree']

        # Both endpoints and short: isolated tiny fragment
        if d1 <= 1 and d2 <= 1 and length < min_edge_length:
            removed_node_ids.add(n1)
            removed_node_ids.add(n2)
            continue

        # Endpoint to junction and short: residual spur
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


# ── Public entry point ───────────────────────────────────────────────────

def run_graph_extraction(skel_pruned, min_edge_length=50):
    """
    Build a graph from *skel_pruned*, then prune short edges.

    Returns
    -------
    graph_raw : SkeletonGraph   – before edge pruning
    graph     : SkeletonGraph   – after edge pruning
    """
    graph_raw = extract_graph(skel_pruned)
    graph = prune_graph(graph_raw, min_edge_length=min_edge_length)
    return graph_raw, graph


# ── Standalone CLI ───────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description='Step 4: Graph extraction & pruning')
    parser.add_argument('--skel', required=True, help='skel.npz from step 3')
    parser.add_argument('--min-edge-length', type=int, default=50)
    parser.add_argument('--output', '-o', default='graph_output.pkl')
    args = parser.parse_args()

    data = np.load(args.skel)
    skel_pruned = data['skel_pruned']

    print("Running graph extraction...")
    graph_raw, graph = run_graph_extraction(skel_pruned, args.min_edge_length)
    n_j_raw = sum(1 for n in graph_raw.nodes.values() if n['degree'] >= 3)
    n_j = sum(1 for n in graph.nodes.values() if n['degree'] >= 3)
    print(f"Raw graph: {len(graph_raw.nodes)} nodes ({n_j_raw} junctions), "
          f"{len(graph_raw.edges)} edges")
    print(f"Pruned:    {len(graph.nodes)} nodes ({n_j} junctions), "
          f"{len(graph.edges)} edges")

    with open(args.output, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Saved -> {args.output}")


if __name__ == '__main__':
    _cli()
