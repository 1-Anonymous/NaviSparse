#!/usr/bin/env python
"""Diagnostic: analyze map_head deformable attention sampling behavior.
Hooks into a trained model's map_head DeformableFeatureAggregation layers
during inference and captures 3D key points, 2D projections, and attention
weights.  Produces three diagnostic outputs:
1. Key-point projection heatmap on camera images (distance-colored).
2. Sampling spread vs BEV distance scatter plot.
3. FPN level attention weight distribution vs distance.
Usage
-----
    python tools/visualization/diagnose_map_deformable.py \
        projects/configs/sparsedrive_small_stage1.py \
        work_dirs/<exp>/latest.pth \
        --frame-indices 0 50 100 200 \
        --top-k 15 \
        --out-dir work_dirs/diag_map
"""
from __future__ import annotations
import argparse
import importlib
import os
import sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
# Embed TrueType fonts in vector PDF output instead of Type 3 glyphs.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
# Needed for custom dataloader that handles sequential sampling correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
# Camera labels (NuScenes standard order in the info pickle)
CAM_NAMES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
IMG_H, IMG_W = 256, 704  # after resize, from config input_shape
# ---------------------------------------------------------------------------
#  Hook infrastructure
# ---------------------------------------------------------------------------
class DeformableCapture:
    """Captures intermediate data from DeformableFeatureAggregation.forward."""
    def __init__(self):
        self.layer_data: list[dict] = []
        self._hooks: list = []
    def _hook_fn(self, module, inputs, _output):
        instance_feature, anchor, anchor_embed, _fmaps, metas = inputs[:5]
        bs, num_anchor = instance_feature.shape[:2]
        with torch.no_grad():
            key_points = module.kps_generator(anchor, instance_feature)
            if isinstance(key_points, tuple):
                key_points = key_points[0]
            pts_2d_raw = module.project_points(
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            # pts_2d_raw: [bs, num_cams, num_anchor, num_pts, 2] (normalized 0-1)
            points_2d = (
                pts_2d_raw.permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, module.num_pts, module.num_cams, 2)
            )
            weights_raw = module._get_weights(
                instance_feature, anchor_embed, metas
            )
            weights = (
                weights_raw.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs, num_anchor, module.num_pts,
                    module.num_cams, module.num_levels, module.num_groups,
                )
            )
        self.layer_data.append(dict(
            anchor=anchor.detach().cpu(),            # [bs, 100, 40]
            key_points_3d=key_points.detach().cpu(), # [bs, 100, 300, 3]
            points_2d=points_2d.detach().cpu(),      # [bs, 100, 300, 6, 2]
            weights=weights.detach().cpu(),          # [bs, 100, 300, 6, 4, 8]
        ))
    def register(self, model):
        from projects.mmdet3d_plugin.models.blocks import (
            DeformableFeatureAggregation,
        )
        head = model.head if not hasattr(model, "module") else model.module.head
        map_head = head.map_head
        for _name, mod in map_head.named_modules():
            if isinstance(mod, DeformableFeatureAggregation):
                h = mod.register_forward_hook(self._hook_fn)
                self._hooks.append(h)
        print(f"[diag] Registered {len(self._hooks)} hooks on map_head "
              f"DeformableFeatureAggregation layers")
    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
    def clear(self):
        self.layer_data.clear()
    @property
    def last_layer(self) -> dict | None:
        return self.layer_data[-1] if self.layer_data else None
# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------
def _bev_distance(anchor: np.ndarray) -> np.ndarray:
    """Compute mean BEV distance from ego for each anchor.
    anchor: [num_anchor, 40]  -> 20 points x 2 coords (x, y in BEV).
    Returns: [num_anchor] mean distance.
    """
    pts = anchor.reshape(-1, 20, 2)  # [num_anchor, 20, 2]
    dist = np.linalg.norm(pts, axis=-1)  # [num_anchor, 20]
    return dist.mean(axis=-1)  # [num_anchor]
def _per_point_bev_distance(anchor: np.ndarray) -> np.ndarray:
    """BEV distance for each of the 20 sample points.
    Returns: [num_anchor, 20]
    """
    pts = anchor.reshape(-1, 20, 2)
    return np.linalg.norm(pts, axis=-1)
def plot_keypoint_heatmap(
    cap: dict,
    out_path: str,
    top_k: int = 15,
    frame_idx: int = 0,
):
    """Diagnostic 1: scatter projected key points on camera images, colored
    by the BEV distance of the originating sample point."""
    anchor = cap["anchor"][0].numpy()       # [100, 40]
    pts_2d = cap["points_2d"][0].numpy()    # [100, 300, 6, 2] normalized
    num_anchor = anchor.shape[0]
    num_sample = 20
    # Each anchor has 300 key points = 20 sample_pts x 5 heights x 3 offsets.
    # Map each key point back to its originating sample index.
    per_pt_dist = _per_point_bev_distance(anchor)  # [100, 20]
    # Expand: each sample point -> 5 heights * 3 offsets = 15 key points
    kp_dist = np.repeat(per_pt_dist, 15, axis=1)  # [100, 300]
    # Pick top-K anchors by mean distance spread (use all for now; will
    # color by distance).  Just use all anchors.
    fig, axes = plt.subplots(2, 3, figsize=(21, 8))
    axes = axes.flatten()
    dist_flat = kp_dist.flatten()
    vmin, vmax = dist_flat.min(), dist_flat.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("plasma")
    for cam_idx in range(6):
        ax = axes[cam_idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # image coords: y increases downward
        ax.set_aspect(IMG_H / IMG_W)
        ax.set_title(f"Cam {cam_idx} ({CAM_NAMES[cam_idx]})", fontsize=9)
        ax.set_facecolor("black")
        for a_idx in range(num_anchor):
            px = pts_2d[a_idx, :, cam_idx, 0]  # [300]
            py = pts_2d[a_idx, :, cam_idx, 1]
            d = kp_dist[a_idx]  # [300]
            # Only plot points inside the image
            mask = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
            if mask.sum() == 0:
                continue
            ax.scatter(
                px[mask], py[mask],
                c=d[mask], cmap=cmap, norm=norm,
                s=1.5, alpha=0.4, edgecolors="none",
            )
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes, label="BEV distance (m)", shrink=0.6,
    )
    fig.suptitle(
        f"Diagnostic 1: Key Point Projections (frame {frame_idx}, "
        f"{num_anchor} anchors × 300 kps)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [D1] Saved key-point heatmap -> {out_path}")
def plot_spread_vs_distance(
    cap: dict,
    out_path: str,
    frame_idx: int = 0,
):
    """Diagnostic 2: pixel spread (std) of projected key points vs BEV
    distance, for front camera only."""
    anchor = cap["anchor"][0].numpy()       # [100, 40]
    pts_2d = cap["points_2d"][0].numpy()    # [100, 300, 6, 2]
    num_anchor = anchor.shape[0]
    front_cam = 0  # CAM_FRONT
    mean_dist = _bev_distance(anchor)  # [100]
    spreads_x = []
    spreads_y = []
    valid_dist = []
    for a_idx in range(num_anchor):
        px = pts_2d[a_idx, :, front_cam, 0] * IMG_W
        py = pts_2d[a_idx, :, front_cam, 1] * IMG_H
        mask = (px >= 0) & (px <= IMG_W) & (py >= 0) & (py <= IMG_H)
        if mask.sum() < 10:
            continue
        spreads_x.append(np.std(px[mask]))
        spreads_y.append(np.std(py[mask]))
        valid_dist.append(mean_dist[a_idx])
    valid_dist = np.array(valid_dist)
    spreads_x = np.array(spreads_x)
    spreads_y = np.array(spreads_y)
    spread_total = np.sqrt(spreads_x ** 2 + spreads_y ** 2)
    # Paper figure size: 8.8 cm wide and 2.44 cm high.
    width_in = 8.8 / 2.54
    fig, axes = plt.subplots(1, 3, figsize=(width_in, 2.44 / 2.54))
    for ax, sp, ylabel in zip(
        axes,
        [spreads_x, spreads_y, spread_total],
        [
            "Horizontal(px)",
            "Vertical(px)",
            "Combined(px)",
        ],
    ):
        ax.scatter(
            valid_dist, sp, s=4, alpha=0.6, edgecolors="k", linewidths=0.2
        )
        ax.set_ylabel(ylabel, fontname="Arial", fontsize=8, labelpad=-1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.tick_params(
            axis="both", which="major", direction="in", length=2, width=0.5,
            pad=0, labelsize=6,
        )
        ax.tick_params(axis="x", pad=1)
        ax.tick_params(axis="y", labelrotation=90)
        for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            tick_label.set_fontname("Arial")
        ax.grid(True, alpha=0.3, linewidth=0.3)
        # Fit a trend line
        if len(valid_dist) > 3:
            z = np.polyfit(valid_dist, sp, 2)
            xs = np.linspace(valid_dist.min(), valid_dist.max(), 50)
            ax.plot(
                xs,
                np.polyval(z, xs),
                "r--",
                linewidth=0.7,
            )
    shared_xlabel = fig.supxlabel(
        "Mean Distance of Each Map Anchor to Ego (m)",
        x=0.525,
        y=0.038,
        fontname="Arial",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.052, right=0.998, bottom=0.254, top=0.972, wspace=0.21,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel_bottom_in = shared_xlabel.get_window_extent(renderer).y0 / fig.dpi
    figure_width_in, figure_height_in = fig.get_size_inches()
    crop_box = Bbox.from_bounds(
        0,
        xlabel_bottom_in,
        figure_width_in,
        figure_height_in - xlabel_bottom_in,
    )
    fig.savefig(out_path, format="pdf", bbox_inches=crop_box, pad_inches=0)
    plt.close(fig)
    print(f"  [D2] Saved spread-vs-distance -> {out_path}")
def plot_level_attention(
    cap: dict,
    out_path: str,
    frame_idx: int = 0,
):
    """Diagnostic 3: average attention weight per FPN level, sorted by BEV
    distance."""
    anchor = cap["anchor"][0].numpy()     # [100, 40]
    weights = cap["weights"][0].numpy()   # [100, 300, 6, 4, 8]
    num_levels = weights.shape[3]
    mean_dist = _bev_distance(anchor)     # [100]
    # Average over key-points, cameras, and groups -> per-anchor per-level
    level_weights = weights.mean(axis=(1, 2, 4))  # [100, 4]
    sort_idx = np.argsort(mean_dist)
    sorted_dist = mean_dist[sort_idx]
    sorted_w = level_weights[sort_idx]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    stride_labels = ["stride 4", "stride 8", "stride 16", "stride 32"]
    # Left: stacked bar chart
    bottom = np.zeros(len(sorted_dist))
    colors = cm.get_cmap("Set2")(np.linspace(0, 1, num_levels))
    for lvl in range(num_levels):
        ax1.bar(
            np.arange(len(sorted_dist)), sorted_w[:, lvl],
            bottom=bottom, width=1.0, color=colors[lvl],
            label=stride_labels[lvl],
        )
        bottom += sorted_w[:, lvl]
    ax1.set_xlabel("Anchor (sorted by BEV distance)")
    ax1.set_ylabel("Mean attention weight")
    ax1.set_title("Attention weight per FPN level (stacked)")
    ax1.legend(fontsize=8)
    # Right: line plot of level weight vs distance
    for lvl in range(num_levels):
        window = max(3, len(sorted_dist) // 10)
        smoothed = np.convolve(
            sorted_w[:, lvl],
            np.ones(window) / window, mode="valid",
        )
        dist_smooth = np.convolve(
            sorted_dist,
            np.ones(window) / window, mode="valid",
        )
        ax2.plot(dist_smooth, smoothed, linewidth=2, label=stride_labels[lvl])
    ax2.set_xlabel("BEV distance (m)")
    ax2.set_ylabel("Mean attention weight (smoothed)")
    ax2.set_title("FPN level weight vs distance")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.suptitle(
        f"Diagnostic 3: FPN Level Attention Distribution (frame {frame_idx})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [D3] Saved level-attention -> {out_path}")
def plot_per_layer_spread(
    all_layer_data: list[dict],
    out_path: str,
    frame_idx: int = 0,
):
    """Bonus: show how sampling spread changes across the 6 decoder layers."""
    num_layers = len(all_layer_data)
    front_cam = 0
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer_idx, cap in enumerate(all_layer_data):
        anchor = cap["anchor"][0].numpy()
        pts_2d = cap["points_2d"][0].numpy()
        num_anchor = anchor.shape[0]
        mean_dist = _bev_distance(anchor)
        dists, spreads = [], []
        for a_idx in range(num_anchor):
            px = pts_2d[a_idx, :, front_cam, 0] * IMG_W
            py = pts_2d[a_idx, :, front_cam, 1] * IMG_H
            mask = (px >= 0) & (px <= IMG_W) & (py >= 0) & (py <= IMG_H)
            if mask.sum() < 10:
                continue
            sp = np.sqrt(np.std(px[mask])**2 + np.std(py[mask])**2)
            dists.append(mean_dist[a_idx])
            spreads.append(sp)
        if len(dists) > 3:
            dists = np.array(dists)
            spreads = np.array(spreads)
            z = np.polyfit(dists, spreads, 2)
            xs = np.linspace(dists.min(), dists.max(), 50)
            ax.plot(xs, np.polyval(z, xs), linewidth=2,
                    label=f"Layer {layer_idx}", alpha=0.8)
    ax.set_xlabel("Mean Distance of Each Map Anchor to Ego (m)")
    ax.set_ylabel(
        r"Quadratic-Fit 2D Projection Spread "
        # r"$\sqrt{\sigma_x^2+\sigma_y^2}$ (px)"
    )
    ax.set_title(
        "Per-Layer Trend of Per-Anchor 300-Key-Point "
        "Projection Spread in CAM_FRONT\n"
        f"Frame {frame_idx}; each line is a quadratic fit using anchors with "
        "at least 10 projected key points inside the image"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [D4] Saved per-layer spread -> {out_path}")
# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="Config file path")
    p.add_argument("checkpoint", help="Checkpoint file path")
    p.add_argument(
        "--frame-indices", type=int, nargs="+", default=[0, 50, 100, 200],
        help="Validation set frame indices to analyze",
    )
    p.add_argument("--top-k", type=int, default=15,
                   help="Top-K anchors to highlight (for heatmap)")
    p.add_argument("--out-dir", default="work_dirs/diag_map",
                   help="Output directory for plots")
    return p.parse_args()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # ---- Build model ----
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin") and cfg.plugin:
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_dir = _module_dir.split("/")
        _module_dir = [p for p in _module_dir if p != ""]
        _module_dir = ".".join(_module_dir)
        plg_lib = importlib.import_module(_module_dir)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    # ---- Build dataset & dataloader ----
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    # ---- Register hooks ----
    capture = DeformableCapture()
    capture.register(model.module)
    # ---- Process selected frames ----
    requested = set(args.frame_indices)
    max_idx = max(args.frame_indices)
    print(f"[diag] Processing frames: {sorted(requested)}")
    print(f"[diag] Output directory: {args.out_dir}")
    for i, data in enumerate(data_loader):
        if i > max_idx:
            break
        if i not in requested:
            continue
        print(f"\n--- Frame {i} ---")
        capture.clear()
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)
        # capture.layer_data now has one entry per deformable layer in
        # map_head (should be 6 for the default config)
        num_layers = len(capture.layer_data)
        print(f"  Captured {num_layers} deformable layers")
        if num_layers == 0:
            print("  WARNING: no data captured, skipping frame")
            continue
        last = capture.last_layer
        # # D1: key-point projection heatmap (last layer)
        # plot_keypoint_heatmap(
        #     last,
        #     os.path.join(args.out_dir, f"d1_kp_heatmap_frame{i:04d}.png"),
        #     top_k=args.top_k,
        #     frame_idx=i,
        # )
        # D2: sampling spread vs distance (last layer)
        plot_spread_vs_distance(
            last,
            os.path.join(args.out_dir, f"d2_spread_vs_dist_frame{i:04d}.pdf"),
            frame_idx=i,
        )
        # # D3: FPN level attention (last layer)
        # plot_level_attention(
        #     last,
        #     os.path.join(args.out_dir, f"d3_level_attention_frame{i:04d}.png"),
        #     frame_idx=i,
        # )
        # # D4 (bonus): spread trend across decoder layers
        # if num_layers >= 2:
        #     plot_per_layer_spread(
        #         capture.layer_data,
        #         os.path.join(args.out_dir, f"d4_layer_spread_frame{i:04d}.png"),
        #         frame_idx=i,
        #     )
    capture.remove()
    print(f"\n[diag] Done. All plots saved to {args.out_dir}")
if __name__ == "__main__":
    main()
