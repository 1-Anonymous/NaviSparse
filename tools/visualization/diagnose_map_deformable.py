#!/usr/bin/env python
"""Diagnostic: analyze map_head deformable attention sampling behavior.

Hooks into a trained model's map_head DeformableFeatureAggregation layers
during inference and captures 3D key points, 2D projections, and attention
weights. Produces five diagnostic outputs:

1. Key-point projection heatmap on camera images (distance-colored).
2. Controlled unscaled-vs-adaptive sampling comparison.
3. Sampling spread vs BEV distance scatter plot.
4. FPN level attention weight distribution vs distance.
5. Sampling spread trend across decoder layers.

Usage
-----
    python tools/visualization/diagnose_map_deformable.py \
        projects/configs/sparsedrive_small_stage1.py \
        work_dirs/<exp>/latest.pth \
        --frame-indices 0 50 100 200 \
        --top-k 15 \
        --out-dir work_dirs/diag_map-0519-0716
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import cv2
import numpy as np
import torch
from PIL import Image

from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

# Needed for custom dataloader that handles sequential sampling correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

# Camera order used by the dataset tensors / projection matrices.
CAM_NAMES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]

# Display order matching tools/visualization/cam_render.py.
CAM_DISPLAY_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
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

            # Controlled comparison: keep anchors, query features, and learned
            # offsets fixed, and change only distance-adaptive scaling.
            generator = module.kps_generator
            if hasattr(generator, "adaptive_offset"):
                original_adaptive = generator.adaptive_offset
                try:
                    generator.adaptive_offset = False
                    key_points_unscaled = generator(anchor, instance_feature)
                    generator.adaptive_offset = True
                    key_points_adaptive = generator(anchor, instance_feature)
                finally:
                    generator.adaptive_offset = original_adaptive
                if isinstance(key_points_unscaled, tuple):
                    key_points_unscaled = key_points_unscaled[0]
                if isinstance(key_points_adaptive, tuple):
                    key_points_adaptive = key_points_adaptive[0]
                adaptive_params = (
                    generator.ref_distance,
                    generator.scale_min,
                    generator.scale_max,
                )
            else:
                key_points_unscaled = key_points
                key_points_adaptive = key_points
                adaptive_params = None

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
            key_points_3d_unscaled=key_points_unscaled.detach().cpu(),
            key_points_3d_adaptive=key_points_adaptive.detach().cpu(),
            adaptive_params=adaptive_params,
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


def _project_keypoints(
    key_points: torch.Tensor,
    lidar2img: np.ndarray,
) -> np.ndarray:
    """Project captured 3D key points to original-image pixel coordinates."""
    key_points = key_points[0].numpy()  # [num_anchor, num_pts, 3]
    points_h = np.concatenate(
        [key_points, np.ones_like(key_points[..., :1])], axis=-1
    )
    projected = np.einsum("cij,apj->capi", lidar2img, points_h)
    depth = projected[..., 2]
    points_2d = projected[..., :2] / np.maximum(depth[..., None], 1e-5)
    points_2d[depth <= 1e-5] = np.nan
    return points_2d.transpose(1, 2, 0, 3)


def _load_original_camera_data(
    dataset,
    frame_idx: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load full-resolution images and original lidar2img matrices.

    The model projects points onto resized/cropped inputs. For a sharp D1
    visualization matching CamRender, project the same 3D key points with the
    original lidar2img matrices and draw them on the original 1600x900 images.

    Returns:
        images: Camera images in dataset order, RGB uint8.
        lidar2img: Original projection matrices [num_cams, 4, 4].
    """
    raw_data = dataset.get_data_info(frame_idx)
    images = [
        np.asarray(Image.open(path).convert("RGB"))
        for path in raw_data["img_filename"]
    ]
    lidar2img = np.asarray(raw_data["lidar2img"], dtype=np.float32)
    return images, lidar2img


def plot_keypoint_heatmap(
    cap: dict,
    images: list[np.ndarray],
    points_2d: np.ndarray,
    out_path: str,
    top_k: int = 15,
    frame_idx: int = 0,
):
    """Diagnostic 1: scatter projected key points on camera images, colored
    by the BEV distance of the originating sample point."""
    anchor = cap["anchor"][0].numpy()       # [100, 40]
    num_anchor = anchor.shape[0]
    # Each anchor has 300 key points = 20 sample_pts x 5 heights x 3 offsets.
    # Map each key point back to its originating sample index.
    per_pt_dist = _per_point_bev_distance(anchor)  # [100, 20]
    # Expand: each sample point -> 5 heights * 3 offsets = 15 key points
    kp_dist = np.repeat(per_pt_dist, 15, axis=1)  # [100, 300]

    # Pick top-K anchors by mean distance spread (use all for now; will
    # color by distance).  Just use all anchors.
    fig, axes = plt.subplots(
        2, 3, figsize=(24, 9.5), constrained_layout=True
    )
    axes = axes.flatten()

    dist_flat = kp_dist.flatten()
    vmin, vmax = dist_flat.min(), dist_flat.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("plasma")

    for display_idx, cam_name in enumerate(CAM_DISPLAY_ORDER):
        cam_idx = CAM_NAMES.index(cam_name)
        ax = axes[display_idx]
        image = cv2.putText(
            images[cam_idx].copy(),
            cam_name,
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        img_h, img_w = image.shape[:2]
        ax.imshow(image)
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)  # image coordinates: y increases downward
        ax.set_aspect("equal")
        ax.axis("off")

        for a_idx in range(num_anchor):
            px = points_2d[a_idx, :, cam_idx, 0]  # [300]
            py = points_2d[a_idx, :, cam_idx, 1]
            d = kp_dist[a_idx]  # [300]

            # Only plot points inside the image
            mask = (
                np.isfinite(px) & np.isfinite(py)
                & (px >= 0) & (px < img_w)
                & (py >= 0) & (py < img_h)
            )
            if mask.sum() == 0:
                continue
            ax.scatter(
                px[mask],
                py[mask],
                c=d[mask], cmap=cmap, norm=norm,
                s=3.0, alpha=0.6, edgecolors="none",
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [D1] Saved key-point heatmap -> {out_path}")


def plot_controlled_sampling_comparison(
    cap: dict,
    images: list[np.ndarray],
    points_unscaled: np.ndarray,
    points_adaptive: np.ndarray,
    out_path: str,
    top_k: int = 5,
    min_distance: float = 20.0,
    max_points_per_camera: int = 120,
    frame_idx: int = 0,
):
    """Controlled D1 comparison with only distance scaling changed.

    Selects the farthest anchors and their highest-attention visible points.
    Blue/red point pairs share identical anchors, query features, and learned
    offsets; their displacement is caused solely by adaptive offset scaling.
    """
    anchor = cap["anchor"][0].numpy()
    mean_dist = _bev_distance(anchor)
    per_pt_dist = _per_point_bev_distance(anchor)
    kp_dist = np.repeat(per_pt_dist, 15, axis=1)
    weights = cap["weights"][0].numpy()
    point_camera_weight = weights.mean(axis=(3, 4))  # [anchor, point, camera]

    candidates = np.flatnonzero(mean_dist >= min_distance)
    if candidates.size == 0:
        candidates = np.arange(mean_dist.shape[0])
    order = np.argsort(mean_dist[candidates])[::-1]
    selected = candidates[order[:max(1, min(top_k, candidates.size))]]

    fig, axes = plt.subplots(
        2, 3, figsize=(24, 9.5), constrained_layout=True
    )
    axes = axes.flatten()
    displacement_values = []

    for display_idx, cam_name in enumerate(CAM_DISPLAY_ORDER):
        cam_idx = CAM_NAMES.index(cam_name)
        ax = axes[display_idx]
        image = cv2.putText(
            images[cam_idx].copy(),
            cam_name,
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        img_h, img_w = image.shape[:2]
        ax.imshow(image)
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        p0 = points_unscaled[selected, :, cam_idx].reshape(-1, 2)
        p1 = points_adaptive[selected, :, cam_idx].reshape(-1, 2)
        dist = kp_dist[selected].reshape(-1)
        importance = point_camera_weight[
            selected, :, cam_idx
        ].reshape(-1)

        valid = (
            np.isfinite(p0).all(axis=-1)
            & np.isfinite(p1).all(axis=-1)
            & (p0[:, 0] >= 0) & (p0[:, 0] < img_w)
            & (p0[:, 1] >= 0) & (p0[:, 1] < img_h)
            & (p1[:, 0] >= 0) & (p1[:, 0] < img_w)
            & (p1[:, 1] >= 0) & (p1[:, 1] < img_h)
            & (dist >= min_distance)
        )
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        keep_count = min(max_points_per_camera, valid_idx.size)
        keep = valid_idx[
            np.argsort(importance[valid_idx])[-keep_count:]
        ]
        before = p0[keep]
        after = p1[keep]
        displacement = np.linalg.norm(after - before, axis=-1)
        displacement_values.append(displacement)

        segments = np.stack([before, after], axis=1)
        ax.add_collection(LineCollection(
            segments,
            colors="white",
            linewidths=0.7,
            alpha=0.45,
            zorder=3,
        ))
        ax.scatter(
            before[:, 0], before[:, 1],
            s=13, c="#00BFFF", marker="o",
            edgecolors="black", linewidths=0.2,
            alpha=0.9, zorder=4,
        )
        ax.scatter(
            after[:, 0], after[:, 1],
            s=17, c="#FF3030", marker="x",
            linewidths=0.8, alpha=0.95, zorder=5,
        )
        ax.text(
            0.015, 0.04,
            f"median displacement: {np.median(displacement):.1f} px",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=2),
            zorder=6,
        )

    params = cap.get("adaptive_params")
    if params is None:
        param_text = "adaptive parameters unavailable"
    else:
        param_text = (
            f"d_ref={params[0]:g}m, "
            f"s_min={params[1]:g}, s_max={params[2]:g}"
        )
    if displacement_values:
        overall_median = np.median(np.concatenate(displacement_values))
        displacement_text = f", median displacement={overall_median:.1f}px"
    else:
        displacement_text = ""

    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="#00BFFF", markeredgecolor="black",
                label="Unscaled offset",
            ),
            Line2D(
                [0], [0], marker="x", color="#FF3030",
                linestyle="none", label="Distance-adaptive offset",
            ),
        ],
        loc="lower center", ncol=2, fontsize=10,
    )
    fig.suptitle(
        f"Controlled Sampling Comparison (frame {frame_idx}, "
        f"top-{len(selected)} far anchors, distance >= {min_distance:g}m)\n"
        f"{param_text}{displacement_text}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [D1-C] Saved controlled comparison -> {out_path}")


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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, sp, label in zip(
        axes,
        [spreads_x, spreads_y, spread_total],
        ["X-spread (px)", "Y-spread (px)", "Total spread (px)"],
    ):
        ax.scatter(valid_dist, sp, s=20, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel("Mean BEV distance (m)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        # Fit a trend line
        if len(valid_dist) > 3:
            z = np.polyfit(valid_dist, sp, 2)
            xs = np.linspace(valid_dist.min(), valid_dist.max(), 50)
            ax.plot(xs, np.polyval(z, xs), "r--", linewidth=1.5, label="poly2 fit")
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Diagnostic 2: Sampling Spread vs Distance "
        f"(frame {frame_idx}, CAM_FRONT, {len(valid_dist)} anchors)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
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

    ax.set_xlabel("Mean BEV distance (m)")
    ax.set_ylabel("Total spread (px)")
    ax.set_title(f"Sampling Spread Trend Across Decoder Layers (frame {frame_idx})")
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
    p.add_argument(
        "--compare-top-k", type=int, default=5,
        help="Farthest anchors shown in the controlled comparison",
    )
    p.add_argument(
        "--compare-min-distance", type=float, default=20.0,
        help="Minimum per-point BEV distance in the controlled comparison",
    )
    p.add_argument(
        "--compare-max-points", type=int, default=120,
        help="Maximum high-attention point pairs per camera",
    )
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
        original_images, original_lidar2img = (
            _load_original_camera_data(dataset, i)
        )
        original_points_2d = _project_keypoints(
            last["key_points_3d"], original_lidar2img
        )
        unscaled_points_2d = _project_keypoints(
            last["key_points_3d_unscaled"], original_lidar2img
        )
        adaptive_points_2d = _project_keypoints(
            last["key_points_3d_adaptive"], original_lidar2img
        )

        # D1: key-point projection heatmap (last layer)
        plot_keypoint_heatmap(
            last,
            original_images,
            original_points_2d,
            os.path.join(args.out_dir, f"d1_kp_heatmap_frame{i:04d}.png"),
            top_k=args.top_k,
            frame_idx=i,
        )

        # D1-C: controlled unscaled-vs-adaptive comparison (last layer)
        plot_controlled_sampling_comparison(
            last,
            original_images,
            unscaled_points_2d,
            adaptive_points_2d,
            os.path.join(
                args.out_dir,
                f"d1_controlled_compare_frame{i:04d}.png",
            ),
            top_k=args.compare_top_k,
            min_distance=args.compare_min_distance,
            max_points_per_camera=args.compare_max_points,
            frame_idx=i,
        )

        # D2: sampling spread vs distance (last layer)
        plot_spread_vs_distance(
            last,
            os.path.join(args.out_dir, f"d2_spread_vs_dist_frame{i:04d}.png"),
            frame_idx=i,
        )

        # D3: FPN level attention (last layer)
        plot_level_attention(
            last,
            os.path.join(args.out_dir, f"d3_level_attention_frame{i:04d}.png"),
            frame_idx=i,
        )

        # D4 (bonus): spread trend across decoder layers
        if num_layers >= 2:
            plot_per_layer_spread(
                capture.layer_data,
                os.path.join(args.out_dir, f"d4_layer_spread_frame{i:04d}.png"),
                frame_idx=i,
            )

    capture.remove()
    print(f"\n[diag] Done. All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
