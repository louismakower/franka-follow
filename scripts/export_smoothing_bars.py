"""Export the action-smoothing comparison (relative+EMA vs absolute) to a compact JSON for the page.

The Approach -> Actions section claims that relative joint actions with EMA smoothing are *drastically*
smoother and less jerky than predicting absolute joint angles. This builds the data behind that
claim's interactive figure: a trajectory dropdown plus a paired bar chart comparing the two
controllers on two metrics for the selected held-out trajectory:

  * ``rmse``   — end-effector position RMSE (mm). The two controllers track almost equally well, so
                 this bar shows the smoothing comes at *no tracking-accuracy cost*.
  * ``accel``  — end-effector acceleration, the mean per-step change in the *achieved* EE velocity
                 (``mean |v_t - v_{t-1}| / dt``, m/s^2). Unlike ``eval.py``'s ``action_jitter`` proxy
                 this is computed on the robot's real motion, so it is **parameterisation-independent**
                 and fair: the relative action is a 0.2-scaled delta while the absolute action is the
                 joint target itself, so the raw actions are on different scales and not comparable.
                 This is where the two controllers differ ~15x — the headline of the section.

Both metrics are scored over the **converged last loop only** (the rollout starts with the hand
flying in from its reset pose; including that transient inflates and distorts both numbers), exactly
as the robustness viewers do. Values are averaged over the available seeds.

The two controllers come from two different studies (see ``run_study.py``):

  * relative + 0.2 EMA  — the *default* controller; reuse the ``noise`` study's noise=0 run, which is
                          that identical controller at zero actuation noise.
  * absolute            — the ``action_smoothing`` study's ``follow-absolute`` run (the action is
                          rescaled directly onto the joint limits each step).

Usage:
    python scripts/export_smoothing_bars.py            # seeds 0 1, default run locations
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


def _load_exporter():
    """Import the sibling exporter so we can reuse its trajectory bank/labels (numpy-only)."""
    path = os.path.join(HERE, "export_web_trajectories.py")
    spec = importlib.util.spec_from_file_location("_export_web_trajectories", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics_path(traces_path: str) -> str:
    name = os.path.basename(traces_path).replace("traces", "metrics", 1).replace(".npz", ".json")
    return os.path.join(os.path.dirname(traces_path), name)


def _last_loop(traces_path, exp):
    """Per-trajectory last-loop RMSE (mm) and EE acceleration (m/s^2) for one rollout file.

    The hand spawns off-trajectory and flies in over the first ~1-2 s, so the full-rollout numbers are
    dominated by that one-off transient. We score only the final period of the rollout — the same
    converged window the robustness viewers use — so the figure describes the steady-state behaviour.

    ``accel`` is the mean magnitude of the step-to-step change in the *achieved* end-effector velocity
    (divided by dt to give m/s^2). It is measured on the robot's real motion rather than on the policy
    output, so it compares the two action parameterisations fairly (the raw actions are on different
    scales — see the module docstring).

    Returns ``(rmse_mm[N], accel[N])`` arrays, one entry per trajectory.
    """
    meta = json.load(open(_metrics_path(traces_path)))
    dt, split = meta["dt"], meta["split"]
    periods = [float(np.asarray(w)[-1, 3]) for w in exp._load_traj_module().make_bank(split)]

    data = np.load(traces_path)
    des, act, act_vel = data["des_pos"], data["act_pos"], data["act_vel"]  # (T,N,3) each, world frame
    horizon, n_envs = des.shape[0], des.shape[1]

    rmse, accel = [], []
    for i in range(min(len(periods), n_envs)):
        length = min(int(round(periods[i] / dt)), horizon)
        seg = slice(horizon - length, horizon)
        err = np.linalg.norm(act[seg, i] - des[seg, i], axis=-1)
        rmse.append(float(np.sqrt((err ** 2).mean()) * 1e3))
        # EE acceleration: mean |Δv|/dt of the achieved end-effector velocity over the loop (m/s^2)
        a = np.linalg.norm(np.diff(act_vel[seg, i], axis=0), axis=-1) / dt
        accel.append(float(a.mean()))
    return np.array(rmse), np.array(accel)


def _controller(traces, exp):
    """Average last-loop RMSE / EE acceleration across the seed rollouts listed in ``traces``."""
    rmses, accels = [], []
    for path in traces:
        if not os.path.exists(path):
            print(f"[warn] missing {path}")
            continue
        r, a = _last_loop(path, exp)
        rmses.append(r); accels.append(a)
    if not rmses:
        raise SystemExit("no trace files found for a controller")
    return np.mean(rmses, axis=0), np.mean(accels, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="Seeds to average over.")
    ap.add_argument("--noise-eval", default=os.path.join(REPO, "logs", "studies", "noise", "eval"),
                    help="Dir with the noise study's traces (noise=0 run = relative+0.2-EMA baseline).")
    ap.add_argument("--smooth-eval", default=os.path.join(REPO, "logs", "studies", "action_smoothing", "eval"),
                    help="Dir with the action_smoothing study's traces (follow-absolute run).")
    ap.add_argument("--out", default=os.path.join(REPO, "webpage-data", "smoothing-bars.json"))
    ap.add_argument("--max-points", type=int, default=200,
                    help="Cap points per 3-D line (downsampled), matching the noise/delay viewers.")
    args = ap.parse_args()

    exp = _load_exporter()

    rel_traces = [os.path.join(args.noise_eval, f"traces_noise__noise_std-0.0__seed{s}.npz") for s in args.seeds]
    abs_traces = [os.path.join(args.smooth_eval,
                  f"traces_action_smoothing__task-follow-absolute__seed{s}.npz") for s in args.seeds]

    rel_rmse, rel_acc = _controller(rel_traces, exp)
    abs_rmse, abs_acc = _controller(abs_traces, exp)

    # 3-D achieved paths (+ the shared noise-free desired/waypoints) for the click-through viewer,
    # reusing export_web_trajectories.export_run so the frame-alignment/recentring matches every other
    # 3-D plot on the page. The bar values are averaged over all seeds, but the plotted path is a single
    # representative seed (the first), exactly as the noise/delay viewers do ("paths shown for seed 0").
    path_seed = args.seeds[0]
    rel_path = os.path.join(args.noise_eval, f"traces_noise__noise_std-0.0__seed{path_seed}.npz")
    abs_path = os.path.join(args.smooth_eval,
                            f"traces_action_smoothing__task-follow-absolute__seed{path_seed}.npz")
    rel_run = exp.export_run(rel_path, split=None, dt=None, max_points=args.max_points,
                             achieved_mode="last-loop", prefix="rel")
    abs_run = exp.export_run(abs_path, split=None, dt=None, max_points=args.max_points,
                             achieved_mode="last-loop", prefix="abs")

    n = min(len(rel_rmse), len(abs_rmse), len(rel_run), len(abs_run))

    trajectories: dict[str, dict] = {}      # traj index -> name/waypoints/desired + per-controller bars
    achieved: dict[str, dict[str, list]] = {}  # traj index -> controller key -> achieved path (seed 0)
    order = []
    for i in range(n):
        rt = rel_run[f"rel-{i}"]                       # desired/waypoints are the same target for both
        trajectories[str(i)] = {
            "name": rt["name"],
            "n_waypoints": rt["n_waypoints"],
            "waypoints": rt["waypoints"],
            "desired": rt["desired"],
            "rmse": {"relative": round(float(rel_rmse[i]), 2), "absolute": round(float(abs_rmse[i]), 2)},
            "accel": {"relative": round(float(rel_acc[i]), 3), "absolute": round(float(abs_acc[i]), 3)},
        }
        achieved[str(i)] = {
            "relative": rt["achieved"],
            "absolute": abs_run[f"abs-{i}"]["achieved"],
        }
        order.append(str(i))

    bounds = [list(b) for b in exp._load_traj_module().WORKSPACE_BOUNDS]
    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seeds": args.seeds,
            "path_seed": path_seed,                         # seed whose achieved 3-D path is shown
            "bounds": bounds,                               # fixed workspace axes for the 3-D viewer
            "order": order,                                 # dropdown order (no average; 0..n-1)
            "controllers": [                                # bar series; colours assigned by the page
                {"key": "relative", "label": "Relative + 0.2 EMA"},
                {"key": "absolute", "label": "Absolute joint angles"},
            ],
            # Two metrics on very different scales -> the page renders them on a dual y-axis ("side").
            "metrics": [
                {"key": "rmse", "label": "RMSE", "side": "left",
                 "unit": "mm", "title": "Tracking RMSE (mm) · lower is better"},
                {"key": "accel", "label": "Jitter", "side": "right",
                 "unit": "m/s²", "title": "Jitter — mean EE acceleration (m/s²) · lower is smoother"},
            ],
        },
        "trajectories": trajectories,
        "achieved": achieved,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    print(f"[export] wrote {n} trajectories (bars: seeds {args.seeds}; paths: seed {path_seed}) "
          f"to {args.out} ({size_kb:.1f} KB)")
    for i in range(n):
        t = trajectories[str(i)]
        print(f"  traj {i}: {t['name']:<38} RMSE rel={t['rmse']['relative']:>5.1f} abs={t['rmse']['absolute']:>5.1f} "
              f"| jitter rel={t['accel']['relative']:.3f} abs={t['accel']['absolute']:.3f}")


if __name__ == "__main__":
    main()
