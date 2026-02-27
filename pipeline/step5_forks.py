"""
Step 5 – Fork detection, classification & cycle detection.

Identifies junction nodes with degree >= 3 as candidate forks, classifies
them as "normal" (Y-shape) or "reversed" (regression), and detects cycles
in the skeleton graph.

Standalone usage:
    python -m pipeline.step5_forks --graph graph.pkl --skel skel.npz --output forks.pkl
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import numpy as np

from pipeline.io_utils import Fork


# ── Angle helpers ────────────────────────────────────────────────────────

def _angle_diff(a1, a2):
    d = a1 - a2
    return (d + np.pi) % (2 * np.pi) - np.pi


def _classify_junction(angles, lengths, degree):
    """Classify as normal fork or reversed fork based on branch angles."""
    n = len(angles)
    if n < 3:
        return 'normal'

    sorted_idx = np.argsort(lengths)[::-1]
    parent_angle = angles[sorted_idx[0]]
    child_angles = [angles[sorted_idx[i]] for i in range(1, min(3, n))]
    child_diffs = [abs(_angle_diff(ca, parent_angle)) for ca in child_angles]

    # Check for two arms nearly parallel (reversed fork signature)
    min_pair_diff = np.pi
    for i in range(n):
        for j in range(i + 1, n):
            pd = abs(_angle_diff(angles[i], angles[j]))
            min_pair_diff = min(min_pair_diff, pd)

    if min_pair_diff < 0.5:
        return 'reversed'

    return 'normal'


# ── Fork classification ──────────────────────────────────────────────────

def classify_forks(graph, skel, min_branch_for_fork=30, min_arm_length=50):
    """
    Classify junction nodes as fork types based on branch angles.

    Only classifies junctions where ALL connected arms have true length
    >= min_arm_length.
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

            other = n2 if n1 == nid else n1
            oy, ox = graph.nodes[other]['pos']
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


# ── Cycle detection ──────────────────────────────────────────────────────

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


# ── Public entry point ───────────────────────────────────────────────────

def run_fork_detection(graph, skel_pruned, min_arm_length=50, max_cycle_length=300):
    """
    Detect and classify forks, and find cycles.

    Returns
    -------
    forks  : list[Fork]
    cycles : list[dict]
    """
    forks = classify_forks(graph, skel_pruned, min_arm_length=min_arm_length)
    cycles = find_cycles(graph, max_cycle_length=max_cycle_length)
    return forks, cycles


# ── Standalone CLI ───────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description='Step 5: Fork detection & classification')
    parser.add_argument('--graph', required=True, help='graph.pkl from step 4')
    parser.add_argument('--skel', required=True, help='skel.npz from step 3')
    parser.add_argument('--min-arm-length', type=int, default=50)
    parser.add_argument('--max-cycle-length', type=int, default=300)
    parser.add_argument('--output', '-o', default='forks_output.pkl')
    args = parser.parse_args()

    with open(args.graph, 'rb') as f:
        graph = pickle.load(f)
    data = np.load(args.skel)
    skel_pruned = data['skel_pruned']

    print("Running fork detection...")
    forks, cycles = run_fork_detection(
        graph, skel_pruned,
        min_arm_length=args.min_arm_length,
        max_cycle_length=args.max_cycle_length,
    )

    fc = defaultdict(int)
    for fork in forks:
        fc[fork.fork_type] += 1
    print(f"Forks: {dict(fc)}, Cycles: {len(cycles)}")

    with open(args.output, 'wb') as f:
        pickle.dump({'forks': forks, 'cycles': cycles}, f)
    print(f"Saved -> {args.output}")


if __name__ == '__main__':
    _cli()
