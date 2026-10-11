#!/usr/bin/env python3
"""Debug ego prev_plan_traj cross-frame transform.

Compare current InstanceQueue._transform_points_ego (treat cumsum as absolute)
vs geometry-correct transform for ego-centric cumulative displacements.

Usage (from repo root):
  PYTHONPATH=. python tools/debug/debug_ego_plan_transform.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from projects.mmdet3d_plugin.models.motion.instance_queue import _transform_points_ego


def make_se2_transform(dx, dy, dyaw):
    c, s = np.cos(dyaw), np.sin(dyaw)
    T = np.eye(4, dtype=np.float32)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3] = dx
    T[1, 3] = dy
    return T


def transform_cumsum_correct(T_temp2cur, plan_cum, ego_prev, ego_cur):
    """Transform ego-centric cumsum displacements across frames."""
    # plan_cum: (T, 2) displacement from prev-frame ego origin (lidar axes)
    # ego_prev / ego_cur: (2,) xy of ego in respective frames' lidar coords
    T = torch.tensor(T_temp2cur, dtype=torch.float32)
    ego_prev = torch.tensor(ego_prev, dtype=torch.float32)
    ego_cur = torch.tensor(ego_cur, dtype=torch.float32)
    plan_cum = torch.tensor(plan_cum, dtype=torch.float32)

    abs_prev = plan_cum + ego_prev.unsqueeze(0)
    zeros = abs_prev.new_zeros(abs_prev.shape[0], 1)
    ones = abs_prev.new_ones(abs_prev.shape[0], 1)
    homo = torch.cat([abs_prev, zeros, ones], dim=-1)
    abs_cur = torch.einsum('ij,tj->ti', T, homo)[..., :2]
    return abs_cur - ego_cur.unsqueeze(0)


def main():
    np.random.seed(0)
    T = 6
    dyaw = np.deg2rad(3.0)
    dx, dy = 2.5, 0.1  # ~0.5s ego motion in lidar frame
    ego_prev = np.array([0.0, 0.0], dtype=np.float32)
    ego_cur = np.array([dx, dy], dtype=np.float32)

    # Synthetic GT-like incremental plan then cumsum (same as kmeans_plan / loss cache)
    deltas = np.stack([np.array([0.8, 0.02])] * T, axis=0).astype(np.float32)
    plan_cum = deltas.cumsum(axis=0)

    T_temp2cur = make_se2_transform(dx, dy, dyaw)

    plan_cum_t = torch.tensor(plan_cum[None], dtype=torch.float32)  # (1, T, 2)
    T_t = torch.tensor(T_temp2cur[None], dtype=torch.float32)

    current_impl = _transform_points_ego(T_t, plan_cum_t)[0].numpy()
    correct = transform_cumsum_correct(
        T_temp2cur, plan_cum, ego_prev, ego_cur,
    ).numpy()

    err = np.linalg.norm(current_impl - correct, axis=-1)
    endpoint_err = np.linalg.norm(current_impl[-1] - correct[-1])

    print("=== ego prev_plan_traj T_temp2cur debug ===")
    print(f"inter-frame motion: dx={dx:.3f}m dy={dy:.3f}m dyaw={np.rad2deg(dyaw):.2f}deg")
    print(f"plan_cum last step (prev frame): {plan_cum[-1]}")
    print(f"current _transform_points_ego last: {current_impl[-1]}")
    print(f"geometry-correct last:            {correct[-1]}")
    print(f"per-step L2 error: {err}")
    print(f"mean per-step error: {err.mean():.4f} m")
    print(f"endpoint error: {endpoint_err:.4f} m")
    print()
    if endpoint_err > 0.05:
        print(
            "WARNING: current transform deviates from geometry-correct transform.\n"
            "Cause: cumsum displacements were treated as absolute points (T@q),\n"
            "but should be T@(ego_prev + q) - ego_cur."
        )
    else:
        print("Current transform matches correct transform on this synthetic case.")

    # Second case: pure translation, current impl is especially wrong on endpoint
    T2 = make_se2_transform(3.0, 0.0, 0.0)
    cur2 = _transform_points_ego(
        torch.tensor(T2[None], dtype=torch.float32),
        plan_cum_t,
    )[0].numpy()
    cor2 = transform_cumsum_correct(T2, plan_cum, ego_prev, np.array([3.0, 0.0])).numpy()
    print("\n=== pure translation +3m case ===")
    print(f"current last: {cur2[-1]}, correct last: {cor2[-1]}")
    print(f"endpoint error: {np.linalg.norm(cur2[-1] - cor2[-1]):.4f} m")


if __name__ == "__main__":
    main()
