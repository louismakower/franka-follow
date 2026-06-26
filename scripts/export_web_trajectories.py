"""Export trajectory traces to a compact JSON for the project webpage's 3-D viewer.

For each evaluated trajectory this writes three world-frame, recentred point lists:

  * ``waypoints`` — the discrete input waypoints (markers)
  * ``desired``   — one clean loop of the interpolated target path (faint reference line)
  * ``achieved``  — the path the end-effector actually followed (thin line)

The trace arrays (``des_pos``/``act_pos`` from ``eval.py``) live in the simulator world frame,
including a per-environment grid offset, whereas the raw bank waypoints are in the robot root
frame. Rather than reconstruct that transform, we exploit the fact that every waypoint is a knot
of the cubic spline: ``des_pos`` already passes through it, so we just sample ``des_pos`` at the
knot's time index. Waypoints, desired and achieved therefore share one frame automatically. Each
trajectory is finally recentred on its desired-loop centroid so the viewer's axes sit around zero.

Usage (one or more runs -> one JSON; splits become key prefixes, e.g. ``train-0``, ``eval-2``):

    python scripts/export_web_trajectories.py \
        --traces logs/louis_rl/franka_follow_sac/<run_train>/eval/traces.npz \
        --traces logs/louis_rl/franka_follow_sac/<run_eval>/eval/traces.npz \
        --out webpage-data/trajectories.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TRAJ_MODULE = os.path.join(
    REPO,
    "source/followtrajectory/followtrajectory/tasks/manager_based/follow/mdp/trajectories.py",
)

# Human-readable labels mirroring the banks defined in trajectories.py. The period is appended
# programmatically from the data, so only the shape description lives here.
SHAPE_LABELS = {
    "train": [
        "Circle — horizontal", "Circle — vertical", "Circle — tilted",
        "Figure-8 — horizontal", "Figure-8 — vertical",
        "Random loop #0", "Random loop #1", "Random loop #2", "Random loop #3",
    ],
    "eval": [
        "Circle — horizontal", "Circle — vertical", "Figure-8 — tilted",
        "Random loop #0", "Random loop #1",
    ],
}


def _load_traj_module():
    """Import trajectories.py directly (it is numpy-only) to avoid pulling in Isaac Sim."""
    spec = importlib.util.spec_from_file_location("_ft_trajectories", TRAJ_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _downsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly subsample rows of ``arr`` (T, 3) down to at most ``max_points``, keeping the ends."""
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).round().astype(int)
    return arr[idx]


def _round(arr: np.ndarray, dp: int = 4) -> list:
    return np.round(arr, dp).tolist()


def export_run(traces_path: str, split: str | None, dt: float | None, max_points: int,
               achieved_mode: str, prefix: str | None = None) -> dict:
    """Build the per-trajectory dict for a single traces.npz file.

    ``prefix`` namespaces the keys (``<prefix>-0``, ``<prefix>-1``, ...); it defaults to the split,
    which is fine for a single run but collides if several runs share a split — pass distinct
    prefixes (e.g. ``noise0.0``, ``noise0.2``) to keep them apart.
    """
    # metrics sit beside the traces with a matching name: traces.npz -> metrics.json, and tagged
    # study files traces_<tag>.npz -> metrics_<tag>.json.
    metrics_name = os.path.basename(traces_path).replace("traces", "metrics", 1).replace(".npz", ".json")
    metrics_path = os.path.join(os.path.dirname(traces_path), metrics_name)
    meta = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            meta = json.load(f)
    split = split or meta.get("split")
    dt = dt or meta.get("dt")
    if split is None or dt is None:
        raise SystemExit(f"need --split and --dt (no usable metrics.json beside {traces_path})")

    data = np.load(traces_path)
    des_pos, act_pos = data["des_pos"], data["act_pos"]  # (T, N, 3) world frame
    horizon, n_envs = des_pos.shape[0], des_pos.shape[1]

    bank = _load_traj_module().make_bank(split)
    n_traj = min(len(bank), n_envs)  # deterministic_eval maps env i -> trajectory i

    labels = SHAPE_LABELS.get(split, [])
    key_prefix = prefix or split
    out: dict[str, dict] = {}
    for i in range(n_traj):
        wp = np.asarray(bank[i], dtype=float)          # (num_wp, 4) -> [x, y, z, t], root frame
        period = float(wp[-1, 3])
        length = int(round(period / dt))               # samples in one loop of the spline
        length = min(length, horizon)

        des = des_pos[:length, i, :]                   # one clean loop of the target
        # "last-loop" shows the final, converged loop (drops the fly-in from the home pose);
        # "full" shows the whole rollout (convergence transient + tracking).
        act = act_pos[horizon - length:, i, :] if achieved_mode == "last-loop" else act_pos[:, i, :]

        # The traces are world-frame (with a per-environment grid offset); the bank waypoints are
        # root-frame. Root->world here is a pure translation, so recover the offset by matching the
        # world-frame spline knots to their known root-frame waypoints, then subtract it. Everything
        # ends up in the robot root frame so all trajectories share the fixed WORKSPACE_BOUNDS axes.
        knot_t = wp[:-1, 3]                              # drop the duplicated closing knot (t==period)
        root_wp = wp[:-1, :3]
        knot_idx = np.clip((knot_t / period * (length - 1)).round().astype(int), 0, length - 1)
        offset = (des_pos[knot_idx, i, :] - root_wp).mean(axis=0)

        shape = labels[i] if i < len(labels) else f"Trajectory {i}"
        out[f"{key_prefix}-{i}"] = {
            "name": f"{shape} · T={period:g}s",
            "split": split,
            "period_s": period,
            "n_waypoints": int(len(knot_t)),
            "waypoints": _round(root_wp),                       # exact root-frame waypoints
            "desired": _round(_downsample(des, max_points) - offset),
            "achieved": _round(_downsample(act, max_points) - offset),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", action="append", required=True, metavar="[PREFIX=]PATH",
                    help="Path to a traces.npz, optionally as PREFIX=PATH to namespace its keys "
                         "(e.g. noise0.2=logs/.../traces.npz -> noise0.2-0). Repeatable.")
    ap.add_argument("--out", default=os.path.join(REPO, "webpage-data", "trajectories.json"))
    ap.add_argument("--merge", action="store_true",
                    help="Merge into an existing --out file instead of overwriting it.")
    ap.add_argument("--split", default=None, help="Override split (else read from sibling metrics.json).")
    ap.add_argument("--dt", type=float, default=None, help="Override control dt (else from metrics.json).")
    ap.add_argument("--max-points", type=int, default=250, help="Cap points per line (downsampled).")
    ap.add_argument("--achieved", choices=["last-loop", "full"], default="last-loop",
                    help="'last-loop' (converged loop only) or 'full' (whole rollout incl. fly-in).")
    args = ap.parse_args()

    trajectories: dict[str, dict] = {}
    if args.merge and os.path.exists(args.out):
        with open(args.out) as f:
            trajectories = json.load(f).get("trajectories", {})

    for entry in args.traces:
        prefix, _, path = entry.partition("=") if "=" in entry else ("", "", entry)
        run = export_run(path, args.split, args.dt, args.max_points, args.achieved, prefix or None)
        for key in run:
            if key in trajectories:
                print(f"[warn] overwriting existing key {key!r}.")
        trajectories.update(run)

    # Fixed Franka workspace bounds (root frame) so the viewer's axes stay constant across
    # trajectories. The achieved path can overshoot slightly, so the viewer pads these a touch.
    bounds = [list(b) for b in _load_traj_module().WORKSPACE_BOUNDS]
    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bounds": bounds,
        },
        "trajectories": trajectories,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    print(f"[export] wrote {len(trajectories)} trajectories to {args.out} ({size_kb:.0f} KB)")
    for key, t in trajectories.items():
        print(f"  {key:>10}  {t['name']:<34}  wp={t['n_waypoints']:>2}  "
              f"des={len(t['desired']):>3}  act={len(t['achieved']):>3}")


if __name__ == "__main__":
    main()
