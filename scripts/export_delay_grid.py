"""Export the control-delay x action-history eval grid to a compact JSON for the webpage viewer.

The Action-delay section of the project page reuses the same interactive widget as the Action-noise
section (a trajectory dropdown, two sliders, a live RMSE matrix and a 3-D path plot), only the two
swept axes differ: the control delay applied to the policy's actions, and the action-history length
the policy observes. For every (delay_steps, history_length) cell, ``eval.py`` was run on the seed-0
checkpoint and wrote ``traces_delay__delay_steps-{d}_history_length-{h}__seed0.npz`` into
``logs/studies/delay/eval``.

This mirrors ``export_noise_grid.py`` exactly (whose ``export_run`` reuse and last-loop RMSE we copy)
— the only real differences are the axis levels/tags and the ``meta.axes`` block that tells the now
axis-agnostic viewer how to label the delay/history sliders and matrix.

The per-trajectory ``desired``/``waypoints`` do not depend on delay or history, so they are stored
once per trajectory; only ``achieved`` varies per cell and is nested
``achieved[traj][delay][history]``, with a matching last-loop ``rmse[traj][delay][history]`` (mm).

Usage:
    python scripts/export_delay_grid.py            # seed-0, delays 0/2/4 x history 1/5
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


def _trace_name(delay, hist, seed):
    return f"traces_delay__delay_steps-{delay}_history_length-{hist}__seed{seed}.npz"


def _rmse_grid(traces_dir, delays, hists, seed, exp):
    """Per-trajectory tracking RMSE (mm) for every (delay, history) cell, over the *last loop*.

    Identical reasoning to ``export_noise_grid._rmse_grid``: the rollout begins with the hand flying
    in from its reset pose, so the full-episode RMSE is dominated by that one-off transient. We score
    only the final period — the exact window the 3-D viewer shows — so the matrix and plot agree.

    Returns ``{traj_key: {delay: {history: rmse_mm}}}`` mirroring the ``achieved`` structure.
    """
    import numpy as np

    # dt + split are identical across cells; read them from the first metrics file we find.
    dt = split = None
    for d in delays:
        for h in hists:
            p = os.path.join(traces_dir, _trace_name(d, h, seed).replace("traces", "metrics", 1)
                             .replace(".npz", ".json"))
            if os.path.exists(p):
                m = json.load(open(p))
                dt, split = m["dt"], m["split"]
                break
        if dt is not None:
            break
    if dt is None:
        return {}

    # one loop = period / dt samples; the spline closes on its first knot, so the period is the last t.
    periods = [float(np.asarray(w)[-1, 3]) for w in exp._load_traj_module().make_bank(split)]

    rmse: dict[str, dict[str, dict[str, float]]] = {}
    for d in delays:
        for h in hists:
            path = os.path.join(traces_dir, _trace_name(d, h, seed))
            if not os.path.exists(path):
                continue
            data = np.load(path)
            des, act = data["des_pos"], data["act_pos"]      # (T, N, 3) world frame (error is frame-free)
            horizon, n_envs = des.shape[0], des.shape[1]
            for i in range(min(len(periods), n_envs)):
                length = min(int(round(periods[i] / dt)), horizon)
                err = np.linalg.norm(act[horizon - length:, i] - des[horizon - length:, i], axis=-1)
                val = float(np.sqrt((err ** 2).mean()) * 1e3)
                rmse.setdefault(str(i), {}).setdefault(str(d), {})[str(h)] = round(val, 1)
    return rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-dir", default=os.path.join(REPO, "logs", "studies", "delay", "eval"),
                    help="Dir holding traces_delay__delay_steps-{d}_history_length-{h}__seed{s}.npz.")
    ap.add_argument("--delays", nargs="+", type=int, default=[0, 2, 4], help="Control-delay steps.")
    ap.add_argument("--hists", nargs="+", type=int, default=[1, 5], help="Action-history lengths.")
    ap.add_argument("--seed", type=int, default=0, help="Which seed's checkpoint to show (3-D path).")
    ap.add_argument("--out", default=os.path.join(REPO, "webpage-data", "delay-grid.json"))
    ap.add_argument("--max-points", type=int, default=200, help="Cap points per line (downsampled).")
    ap.add_argument("--achieved", choices=["last-loop", "full"], default="last-loop",
                    help="'last-loop' (converged loop only) or 'full' (whole rollout incl. fly-in).")
    args = ap.parse_args()

    exp = _load_exporter()

    trajectories: dict[str, dict] = {}                    # traj index -> name/waypoints/desired
    achieved: dict[str, dict[str, dict[str, list]]] = {}  # traj -> delay -> history -> achieved path
    missing = []

    for d in args.delays:
        for h in args.hists:
            tag = f"d{d}_h{h}"
            traces_path = os.path.join(args.traces_dir, _trace_name(d, h, args.seed))
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
                achieved.setdefault(i, {}).setdefault(str(d), {})[str(h)] = t["achieved"]

    if missing:
        print(f"[warn] {len(missing)} missing cells: {', '.join(missing)}")
    if not trajectories:
        raise SystemExit(f"no traces found under {args.traces_dir}")

    # Per-trajectory last-loop RMSE for the live matrix beside the 3-D plot (excludes the fly-in).
    rmse = _rmse_grid(args.traces_dir, args.delays, args.hists, args.seed, exp)
    for i, cells in rmse.items():
        if i in trajectories:
            trajectories[i]["rmse"] = cells

    bounds = [list(b) for b in exp._load_traj_module().WORKSPACE_BOUNDS]
    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bounds": bounds,
            "seed": args.seed,
            # tells the axis-agnostic grid viewer how to label the two sliders + matrix axes.
            "axes": {
                "row": {
                    "label": "Control delay", "title": "control delay (steps)",
                    "valLabels": {str(d): (f"{d} steps (none)" if d == 0 else f"{d} steps")
                                  for d in args.delays},
                },
                "col": {
                    "label": "Action history", "title": "action-history length",
                    "valLabels": {str(h): f"length {h}" for h in args.hists},
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
    n_cells = sum(len(d) for tj in achieved.values() for d in tj.values())
    print(f"[export] wrote {len(trajectories)} trajectories x {len(args.delays)}x{len(args.hists)} "
          f"grid ({n_cells} achieved cells, seed {args.seed}) to {args.out} ({size_kb:.0f} KB)")
    for i, t in sorted(trajectories.items()):
        vals = [v for d in t.get("rmse", {}).values() for v in d.values()]
        rng = f"rmse {min(vals):.0f}-{max(vals):.0f} mm" if vals else "rmse n/a"
        print(f"  traj {i}: {t['name']:<34} wp={t['n_waypoints']:>2}  des={len(t['desired']):>3}  {rng}")


if __name__ == "__main__":
    main()