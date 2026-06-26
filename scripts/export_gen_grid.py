"""Export the data-scaling (generalisation) eval sweep to a compact JSON for the webpage viewer.

The Generalisation section of the project page reuses the same interactive widget as the Action-noise
and Action-delay sections, but the sweep is **one-dimensional**: the only axis is the number of
trajectories the policy was *trained* on (``num_trajectories`` in {1,2,3,5,7,9}). Every checkpoint is
evaluated on the *same* fixed held-out ``eval`` split, so the desired paths are identical across cells
and the achieved path is what changes. The eval-time ``num_trajectories`` override has no effect (the
held-out split is forced internally) — a cell is identified purely by *which checkpoint* was loaded.

Because there is a single axis, the viewer renders a horizontal RMSE strip (one row, one cell per
training-set size) instead of a 2-D matrix and shows a single slider. We keep the JSON shape identical
to the noise/delay grids (``achieved[traj][row][col]``) by nesting under a single dummy row key, and we
flag ``meta.single1d`` so the widget hides the (unused) row slider + y-axis decorations.

This mirrors ``export_delay_grid.py`` / ``export_noise_grid.py`` exactly (the ``export_run`` reuse and
last-loop RMSE are copied verbatim) — only the swept axis and the ``meta.axes`` labels differ.

Usage:
    python scripts/export_gen_grid.py            # seed-0, num_trajectories 1/2/3/5/7/9
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Single dummy row key: the sweep is 1-D, but we keep the achieved[traj][row][col] nesting so the
# webpage renderer (written for the 2-D noise/delay grids) needs no special-casing on the data side.
ROW = "_"


def _load_exporter():
    """Import the sibling exporter so we can reuse its frame-aligned ``export_run`` + bounds."""
    path = os.path.join(HERE, "export_web_trajectories.py")
    spec = importlib.util.spec_from_file_location("_export_web_trajectories", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_name(num, seed):
    return f"traces_data_scaling__num_trajectories-{num}__seed{seed}.npz"


def _rmse_grid(traces_dir, nums, seed, exp):
    """Per-trajectory tracking RMSE (mm) for every training-set size, over the *last loop*.

    Identical reasoning to ``export_noise_grid._rmse_grid``: the rollout begins with the hand flying
    in from its reset pose, so the full-episode RMSE is dominated by that one-off transient. We score
    only the final period — the exact window the 3-D viewer shows — so the strip and plot agree.

    Returns ``{traj_key: {ROW: {num: rmse_mm}}}`` mirroring the ``achieved`` structure.
    """
    import numpy as np

    # dt + split are identical across cells; read them from the first metrics file we find.
    dt = split = None
    for n in nums:
        p = os.path.join(traces_dir, _trace_name(n, seed).replace("traces", "metrics", 1)
                         .replace(".npz", ".json"))
        if os.path.exists(p):
            m = json.load(open(p))
            dt, split = m["dt"], m["split"]
            break
    if dt is None:
        return {}

    # one loop = period / dt samples; the spline closes on its first knot, so the period is the last t.
    periods = [float(np.asarray(w)[-1, 3]) for w in exp._load_traj_module().make_bank(split)]

    rmse: dict[str, dict[str, dict[str, float]]] = {}
    for n in nums:
        path = os.path.join(traces_dir, _trace_name(n, seed))
        if not os.path.exists(path):
            continue
        data = np.load(path)
        des, act = data["des_pos"], data["act_pos"]      # (T, N, 3) world frame (error is frame-free)
        horizon, n_envs = des.shape[0], des.shape[1]
        for i in range(min(len(periods), n_envs)):
            length = min(int(round(periods[i] / dt)), horizon)
            err = np.linalg.norm(act[horizon - length:, i] - des[horizon - length:, i], axis=-1)
            val = float(np.sqrt((err ** 2).mean()) * 1e3)
            rmse.setdefault(str(i), {}).setdefault(ROW, {})[str(n)] = round(val, 1)
    return rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-dir", default=os.path.join(REPO, "logs", "studies", "data_scaling", "eval"),
                    help="Dir holding traces_data_scaling__num_trajectories-{n}__seed{s}.npz.")
    ap.add_argument("--nums", nargs="+", type=int, default=[1, 2, 3, 5, 7, 9],
                    help="Training-set sizes (number of trajectories) to include.")
    ap.add_argument("--seed", type=int, default=0, help="Which seed's checkpoint to show (3-D path).")
    ap.add_argument("--out", default=os.path.join(REPO, "webpage-data", "gen-grid.json"))
    ap.add_argument("--max-points", type=int, default=200, help="Cap points per line (downsampled).")
    ap.add_argument("--achieved", choices=["last-loop", "full"], default="last-loop",
                    help="'last-loop' (converged loop only) or 'full' (whole rollout incl. fly-in).")
    args = ap.parse_args()

    exp = _load_exporter()

    trajectories: dict[str, dict] = {}                # traj index -> name/waypoints/desired (cell-invariant)
    achieved: dict[str, dict[str, dict[str, list]]] = {}  # traj -> ROW -> num -> achieved path
    missing = []

    for n in args.nums:
        tag = f"n{n}"
        traces_path = os.path.join(args.traces_dir, _trace_name(n, args.seed))
        if not os.path.exists(traces_path):
            missing.append(tag)
            continue
        # export_run reads dt/split from the sibling metrics_<...>.json and returns
        # {f"{prefix}-{i}": {name, waypoints, desired, achieved, ...}} per trajectory.
        run = exp.export_run(traces_path, split=None, dt=None, max_points=args.max_points,
                             achieved_mode=args.achieved, prefix=tag)
        for key, t in run.items():
            i = key.rsplit("-", 1)[-1]
            if i not in trajectories:
                trajectories[i] = {k: t[k] for k in ("name", "period_s", "n_waypoints",
                                                      "waypoints", "desired")}
            achieved.setdefault(i, {}).setdefault(ROW, {})[str(n)] = t["achieved"]

    if missing:
        print(f"[warn] {len(missing)} missing cells: {', '.join(missing)}")
    if not trajectories:
        raise SystemExit(f"no traces found under {args.traces_dir}")

    # Per-trajectory last-loop RMSE for the live strip beside the 3-D plot (excludes the fly-in).
    rmse = _rmse_grid(args.traces_dir, args.nums, args.seed, exp)
    for i, cells in rmse.items():
        if i in trajectories:
            trajectories[i]["rmse"] = cells

    bounds = [list(b) for b in exp._load_traj_module().WORKSPACE_BOUNDS]
    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bounds": bounds,
            "seed": args.seed,
            # 1-D sweep: tell the widget to hide the row slider + y-axis decorations and just sweep
            # the single (column) axis below. "col" labels the slider, the strip ticks and the title.
            "single1d": True,
            "axes": {
                "col": {
                    "label": "Training trajectories", "title": "number of training trajectories",
                    "valLabels": {str(n): str(n) for n in args.nums},
                },
            },
        },
        "trajectories": trajectories,
        "achieved": achieved,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    n_cells = sum(len(c) for tj in achieved.values() for c in tj.values())
    print(f"[export] wrote {len(trajectories)} trajectories x {len(args.nums)} training-set sizes "
          f"({n_cells} achieved cells, seed {args.seed}) to {args.out} ({size_kb:.0f} KB)")
    for i, t in sorted(trajectories.items()):
        vals = [v for c in t.get("rmse", {}).values() for v in c.values()]
        rng = f"rmse {min(vals):.0f}-{max(vals):.0f} mm" if vals else "rmse n/a"
        print(f"  traj {i}: {t['name']:<34} wp={t['n_waypoints']:>2}  des={len(t['desired']):>3}  {rng}")


if __name__ == "__main__":
    main()
