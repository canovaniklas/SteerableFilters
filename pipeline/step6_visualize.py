"""
Step 6 – Evaluation metrics, per-image 8-panel figure, and summary plots.

Standalone usage:
    python -m pipeline.step6_visualize --help
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    return {
        'precision': precision, 'recall': recall,
        'f1': f1, 'iou': iou,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
    }


# ── Overlay helper ───────────────────────────────────────────────────────

def create_overlay(image, pred_mask, gt_mask, removed_mask=None):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image[..., :3].copy()
    rgb = rgb.astype(np.float64)
    if rgb.max() > 1:
        rgb /= rgb.max()
    overlay = rgb.copy()
    a = 0.45
    overlay[pred & gt] = overlay[pred & gt] * (1 - a) + np.array([0, 1, 0]) * a
    overlay[pred & ~gt] = overlay[pred & ~gt] * (1 - a) + np.array([1, 0, 0]) * a
    overlay[~pred & gt] = overlay[~pred & gt] * (1 - a) + np.array([0.2, 0.4, 1.0]) * a
    if removed_mask is not None:
        rm = removed_mask.astype(bool)
        overlay[rm] = overlay[rm] * (1 - a) + np.array([1, 1, 0]) * a
    return np.clip(overlay, 0, 1)


# ── Fork bounding box ───────────────────────────────────────────────────

FORK_COLORS = {'normal': '#00FF00', 'reversed': '#FF0000'}


def _fork_bounding_box(fork, graph, skel, padding=40):
    jy, jx = fork.position
    pixels = [(jy, jx)]
    for n1, n2, edata in graph.edges:
        if n1 != fork.node_id and n2 != fork.node_id:
            continue
        for py, px in edata['path'][:60]:
            pixels.append((py, px))
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    y_min = max(0, min(ys) - padding)
    y_max = min(skel.shape[0], max(ys) + padding)
    x_min = max(0, min(xs) - padding)
    x_max = min(skel.shape[1], max(xs) + padding)
    return y_min, y_max, x_min, x_max


# ── 8-panel analysis figure ─────────────────────────────────────────────

def plot_full_analysis(
    image, result, gt_mask, metrics_clean,
    skel, skel_pruned, graph, forks, cycles,
    title="", save_path=None,
):
    """
    8-panel pipeline figure:
      Row 1: Input | Ridge magnitude | Raw mask | Cleaned mask
      Row 2: GT comparison | Skeleton | Graph | Fork detections
    """
    fig = plt.figure(figsize=(28, 14))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.15)

    def img_rgb():
        rgb = np.stack([image] * 3, axis=-1).copy()
        if rgb.max() > 1:
            rgb /= rgb.max()
        return rgb

    # 1. Input
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, cmap='gray')
    ax.set_title('1. Input TEM', fontweight='bold', fontsize=10)
    ax.axis('off')

    # 2. Ridge magnitude
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(image, cmap='gray', alpha=0.4)
    ax.imshow(result.ridge_magnitude, cmap='inferno', alpha=0.6)
    ax.set_title(f'2. Ridge Magnitude\nthresh={result.threshold:.3f}',
                 fontweight='bold', fontsize=10)
    ax.axis('off')

    # 3. Raw mask
    ax = fig.add_subplot(gs[0, 2])
    rgb = img_rgb()
    rgb[result.raw_mask] = [1, 0.3, 0.3]
    ax.imshow(rgb)
    ax.set_title(f'3. Raw Thresholded Mask\n{int(result.raw_mask.sum())} pixels',
                 fontweight='bold', fontsize=10)
    ax.axis('off')

    # 4. Cleaned mask
    ax = fig.add_subplot(gs[0, 3])
    rgb = img_rgb()
    rgb[result.clean_mask] = [0.2, 0.8, 0.2]
    rgb[result.removed_mask] = [1, 1, 0]
    ax.imshow(rgb)
    ax.set_title(f'4. Cleaned Mask (green)\n'
                 f'removed {int(result.removed_mask.sum())} px (yellow)',
                 fontweight='bold', fontsize=10)
    ax.axis('off')

    # 5. GT comparison
    ax = fig.add_subplot(gs[1, 0])
    overlay = create_overlay(image, result.clean_mask, gt_mask)
    ax.imshow(overlay)
    ax.set_title(
        f'5. vs Ground Truth\n'
        f'F1={metrics_clean["f1"]:.3f}  IoU={metrics_clean["iou"]:.3f}  '
        f'P={metrics_clean["precision"]:.3f}  R={metrics_clean["recall"]:.3f}',
        fontweight='bold', fontsize=10)
    ax.axis('off')

    # 6. Skeleton
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

    # 7. Graph
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

    # 8. Fork detections
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(image, cmap='gray')
    n_normal = n_reversed = 0
    for fork in forks:
        y0, y1, x0, x1 = _fork_bounding_box(fork, graph, skel_pruned)
        color = FORK_COLORS.get(fork.fork_type, '#00FF00')
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=2.5, edgecolor=color,
                              facecolor='none', linestyle='-')
        ax.add_patch(rect)
        lbl = 'N' if fork.fork_type == 'normal' else 'R'
        ax.text(x0 + 3, y0 + 15, lbl,
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

    # Legend
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


# ── Summary bar chart ────────────────────────────────────────────────────

def plot_summary(all_metrics, save_path=None):
    if not all_metrics:
        return
    names = [m['name'] for m in all_metrics]
    clean_f1 = [m['clean']['f1'] for m in all_metrics]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.8), 6))
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


# ── Public entry point ───────────────────────────────────────────────────

def run_visualization(
    image, result, gt_mask, metrics_clean,
    skel, skel_pruned, graph, forks, cycles,
    name, save_path,
):
    """Generate the 8-panel analysis figure and return metrics dict."""
    plot_full_analysis(
        image, result, gt_mask, metrics_clean,
        skel, skel_pruned, graph, forks, cycles,
        title=name, save_path=save_path,
    )
