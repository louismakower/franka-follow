"""Export the train-noise x eval-noise eval grid to a compact JSON for the webpage's noise viewer.

The Action-noise section of the project page has an interactive plot with a trajectory dropdown and
two sliders: the actuation noise the policy was *trained* under, and the actuation noise injected at
*eval* time. For every (train_noise, eval_noise) cell, ``eval.py`` was run on the seed-0 checkpoint
trained at ``train_noise`` with ``env.actions.arm_action.noise_std=eval_noise``, writing
``traces_t{tn}_e{en}.npz`` into ``webpage-data/noise-grid-traces``.

The per-trajectory target (``desired``) and ``waypoints`` are noise-free, so they are identical
across every cell and stored once per trajectory. Only the ``achieved`` path varies per cell, so it
is nested ``achieved[traj][train_noise][eval_noise]``. All point lists are recentred and frame-
aligned exactly as in ``export_web_trajectories.py`` (whose ``export_run`` we reuse for the heavy
lifting), so they share the fixed workspace axes.

Usage:
    python scripts/export_noise_grid.py            # defaults to the 4x4 grid below
    python scripts/export_noise_grid.py --levels 0.0 0.05 0.1 0.2 --max-points 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load_exporter():
    """Import the sibling exporter so we can reuse its frame-aligned ``export_run`` + bounds."""
    path = os.path.join(HERE, "export_web_trajectories.py")
    spec = importlib.util.spec_from_file_location("_export_web_trajectories", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rmse_grid(traces_dir, levels, exp):
    """Per-trajectory tracking RMSE (mm) for every (train, test) noise cell, over the *last loop*.

    The rollout starts with the hand off-trajectory (it spawns at the reset pose and flies in), so the
    full-episode RMSE is dominated by that one-off transient — which is also what made the matrix look
    noisy. We instead score only the final period of the rollout, the exact window the 3-D viewer shows
    as the achieved path, so the matrix and the plot describe the same converged loop.

    Returns ``{traj_key: {train_noise: {test_noise: rmse_mm}}}`` mirroring the ``achieved`` structure.
    """
    import numpy as np

    # dt + split are identical across cells; read them from the first metrics file we find.
    dt = split = None
    for tn in levels:
        for en in levels:
            p = os.path.join(traces_dir, f"metrics_t{tn}_e{en}.json")
            if os.path.exists(p):
                m = json.load(open(p))
                dt, split = m["dt"], m["split"]
                break
        if dt is not None:
            break
    if dt is None:
        return {}

    # one loop = period / dt samples; the spline closes on its first knot so the period is the last t.
    periods = [float(np.asarray(w)[-1, 3]) for w in exp._load_traj_module().make_bank(split)]

    rmse: dict[str, dict[str, dict[str, float]]] = {}
    for tn in levels:
        for en in levels:
            path = os.path.join(traces_dir, f"traces_t{tn}_e{en}.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path)
            des, act = data["des_pos"], data["act_pos"]      # (T, N, 3) world frame (error is frame-free)
            horizon, n_envs = des.shape[0], des.shape[1]
            for i in range(min(len(periods), n_envs)):
                length = min(int(round(periods[i] / dt)), horizon)
                err = np.linalg.norm(act[horizon - length:, i] - des[horizon - length:, i], axis=-1)
                val = float(np.sqrt((err ** 2).mean()) * 1e3)
                rmse.setdefault(str(i), {}).setdefault(tn, {})[en] = round(val, 1)
    return rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-dir", default=os.path.join(REPO, "webpage-data", "noise-grid-traces"),
                    help="Dir holding traces_t{tn}_e{en}.npz from the eval grid.")
    ap.add_argument("--levels", nargs="+", default=["0.0", "0.05", "0.1", "0.2"],
                    help="Noise levels (strings, matching the file tags) used for both sliders.")
    ap.add_argument("--out", default=os.path.join(REPO, "webpage-data", "noise-grid.json"))
    ap.add_argument("--max-points", type=int, default=200, help="Cap points per line (downsampled).")
    ap.add_argument("--achieved", choices=["last-loop", "full"], default="last-loop",
                    help="'last-loop' (converged loop only) or 'full' (whole rollout incl. fly-in).")
    args = ap.parse_args()

    exp = _load_exporter()

    trajectories: dict[str, dict] = {}                 # traj index -> name/waypoints/desired (noise-free)
    achieved: dict[str, dict[str, dict[str, list]]] = {}  # traj -> train_noise -> eval_noise -> achieved path
    missing = []

    for tn in args.levels:
        for en in args.levels:
            tag = f"t{tn}_e{en}"
            traces_path = os.path.join(args.traces_dir, f"traces_{tag}.npz")
            if not os.path.exists(traces_path):
                missing.append(tag)
                continue
            # export_run reads dt/split from the sibling metrics_<tag>.json and returns
            # {f"{prefix}-{i}": {name, waypoints, desired, achieved, ...}} per trajectory.
            run = exp.export_run(traces_path, split=None, dt=None, max_points=args.max_points,
                                 achieved_mode=args.achieved, prefix=tag)
            for key, t in run.items():
                i = key.rsplit("-", 1)[-1]
                if i not in trajectories:
                    trajectories[i] = {k: t[k] for k in ("name", "period_s", "n_waypoints",
                                                          "waypoints", "desired")}
                achieved.setdefault(i, {}).setdefault(tn, {})[en] = t["achieved"]

    if missing:
        print(f"[warn] {len(missing)} missing cells: {', '.join(missing)}")
    if not trajectories:
        raise SystemExit(f"no traces found under {args.traces_dir}")

    # Per-trajectory last-loop RMSE for the live matrix beside the 3-D plot (excludes the fly-in).
    rmse = _rmse_grid(args.traces_dir, args.levels, exp)
    for i, cells in rmse.items():
        if i in trajectories:
            trajectories[i]["rmse"] = cells

    bounds = [list(b) for b in exp._load_traj_module().WORKSPACE_BOUNDS]
    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bounds": bounds,
            "train_noise": [float(x) for x in args.levels],
            "eval_noise": [float(x) for x in args.levels],
        },
        "trajectories": trajectories,
        "achieved": achieved,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    n_cells = sum(len(tn) for tj in achieved.values() for tn in tj.values())
    print(f"[export] wrote {len(trajectories)} trajectories x {len(args.levels)}x{len(args.levels)} "
          f"grid ({n_cells} achieved cells) to {args.out} ({size_kb:.0f} KB)")
    for i, t in sorted(trajectories.items()):
        vals = [v for tn in t.get("rmse", {}).values() for v in tn.values()]
        rng = f"rmse {min(vals):.0f}-{max(vals):.0f} mm" if vals else "rmse n/a"
        print(f"  traj {i}: {t['name']:<34} wp={t['n_waypoints']:>2}  des={len(t['desired']):>3}  {rng}")


if __name__ == "__main__":
    main()
