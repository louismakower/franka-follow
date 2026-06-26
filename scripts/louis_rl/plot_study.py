"""Aggregate and plot the results of a study produced by ``run_study.py``.

Reads ``logs/studies/<study>/results.json`` (one record per train+eval run) and produces a summary
figure with mean +/- std over seeds:

  * ``data_scaling`` — held-out position RMSE vs number of training trajectories.
  * ``noise``        — held-out position RMSE and action jitter vs actuation-noise std.
  * ``delay``        — grouped bars of last-loop position RMSE by (control delay, action-history
                       length). Computed straight from the eval traces (not results.json) so it
                       excludes the fly-in transient and picks up the history=5 runs.
  * ``action_smoothing`` — grouped bars of held-out position RMSE and action jitter for relative
                       (EMA) vs absolute joint-position control.

Usage:
    python scripts/louis_rl/plot_study.py --study data_scaling
"""

import argparse
import glob
import importlib.util
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(study):
    path = os.path.join("logs", "studies", study, "results.json")
    with open(path) as f:
        return json.load(f), os.path.dirname(path)


def _group(results, key_fn, metric):
    """Map cell-key -> list of metric values (one per seed)."""
    groups = defaultdict(list)
    for r in results:
        groups[key_fn(r["cell"])].append(r["metrics"][metric] * (1e3 if metric.endswith("_m") else 1.0))
    xs = sorted(groups)
    mean = np.array([np.mean(groups[x]) for x in xs])
    std = np.array([np.std(groups[x]) for x in xs])
    return xs, mean, std


def plot_data_scaling(results, out):
    xs, mean, std = _group(results, lambda c: c["env.commands.trajectory.num_trajectories"], "pos_rmse_m")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, mean, yerr=std, marker="o", capsize=4, lw=1.5)
    ax.set_xlabel("number of training trajectories")
    ax.set_ylabel("held-out position RMSE (mm)")
    ax.set_title("Generalisation vs training-set size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_noise(results, out):
    xr, mr, sr = _group(results, lambda c: c["env.actions.arm_action.noise_std"], "pos_rmse_m")
    xj, mj, sj = _group(results, lambda c: c["env.actions.arm_action.noise_std"], "action_jitter")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.errorbar(xr, mr, yerr=sr, marker="o", capsize=4, color="C0", lw=1.5, label="position RMSE")
    ax1.set_xlabel("actuation noise std")
    ax1.set_ylabel("held-out position RMSE (mm)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.errorbar(xj, mj, yerr=sj, marker="s", capsize=4, color="C1", lw=1.5, label="action jitter")
    ax2.set_ylabel("action jitter", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title("Robustness to actuation noise")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def _smoothing_label(cell):
    """Human-readable control scheme for an ``action_smoothing`` cell."""
    if cell.get("_task") == "follow-absolute":
        return "absolute"
    ma = cell.get("env.actions.arm_action.moving_average")
    return f"relative (MA={ma})" if ma is not None else "relative"


def plot_action_smoothing(results, out):
    rmse, jit = defaultdict(list), defaultdict(list)
    for r in results:
        lab = _smoothing_label(r["cell"])
        rmse[lab].append(r["metrics"]["pos_rmse_m"] * 1e3)
        jit[lab].append(r["metrics"]["action_jitter"])
    # The relative MA=0.2 baseline isn't retrained for this study; borrow the ``noise`` study's
    # noise=0 runs (the same default relative controller at zero actuation noise) if present.
    try:
        noise_results, _ = _load("noise")
    except FileNotFoundError:
        noise_results = []
    for r in noise_results:
        if r["cell"].get("env.actions.arm_action.noise_std") == 0.0:
            rmse["relative (MA=0.2)"].append(r["metrics"]["pos_rmse_m"] * 1e3)
            jit["relative (MA=0.2)"].append(r["metrics"]["action_jitter"])
    # relative first, absolute second; fall back to alphabetical for anything unexpected
    order = ["relative (MA=0.2)", "absolute"]
    labels = [l for l in order if l in rmse] + sorted(set(rmse) - set(order))
    x = np.arange(len(labels))
    w = 0.35
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(x - w / 2, [np.mean(rmse[l]) for l in labels], w,
            yerr=[np.std(rmse[l]) for l in labels], capsize=4, color="C0", label="position RMSE")
    ax1.set_ylabel("held-out position RMSE (mm)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, [np.mean(jit[l]) for l in labels], w,
            yerr=[np.std(jit[l]) for l in labels], capsize=4, color="C1", label="action jitter")
    ax2.set_ylabel("action jitter", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Relative (EMA) vs absolute joint-position control")
    ax1.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def _exporter():
    """Reuse export_web_trajectories for its (Isaac-free) trajectory-bank loader."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "export_web_trajectories.py")
    spec = importlib.util.spec_from_file_location("_export_web_trajectories", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _delay_lastloop_rmse(eval_dir):
    """Per-cell held-out position RMSE (mm) over the converged last loop, straight from the traces.

    results.json only holds the history=1 runs and its ``pos_rmse_m`` scores the *whole* rollout —
    but the hand spawns off-trajectory and flies in over the first ~1-2 s, so that one-off transient
    dominates (it inflates the working cells ~5x and buries the action-history effect). We instead
    score only the final period — the same window the 3-D viewer shows — and discover cells from the
    trace filenames, which also picks up the history=5 runs missing from results.json.

    Returns ``{(delay, history): [rmse_per_seed_mm]}``.
    """
    metrics = sorted(glob.glob(os.path.join(eval_dir, "metrics_*.json")))
    if not metrics:
        return {}
    meta = json.load(open(metrics[0]))           # dt + split are identical across cells
    dt, split = meta["dt"], meta["split"]
    # one loop = period / dt samples; the spline closes on its first knot, so period is the last t.
    periods = [float(np.asarray(w)[-1, 3]) for w in _exporter()._load_traj_module().make_bank(split)]

    cells = defaultdict(list)
    pat = re.compile(r"delay_steps-(\d+)_history_length-(\d+)__seed(\d+)")
    for path in sorted(glob.glob(os.path.join(eval_dir, "traces_*.npz"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        delay, hist = int(m.group(1)), int(m.group(2))
        data = np.load(path)
        des, act = data["des_pos"], data["act_pos"]      # (T, N, 3); the error is frame-independent
        horizon, n_envs = des.shape[0], des.shape[1]
        per_traj = []
        for i in range(min(len(periods), n_envs)):
            length = min(int(round(periods[i] / dt)), horizon)
            err = np.linalg.norm(act[horizon - length:, i] - des[horizon - length:, i], axis=-1)
            per_traj.append(np.sqrt((err ** 2).mean()) * 1e3)
        cells[(delay, hist)].append(float(np.mean(per_traj)))   # mean over the held-out trajectories
    return cells


def plot_delay(results, out):
    cells = _delay_lastloop_rmse(os.path.join(os.path.dirname(out), "eval"))
    if not cells:
        raise SystemExit("no eval traces found beside results.json")
    delays = sorted({k[0] for k in cells})
    hists = sorted({k[1] for k in cells})
    width = 0.8 / len(hists)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, h in enumerate(hists):
        means = np.array([np.mean(cells[(d, h)]) if (d, h) in cells else np.nan for d in delays])
        stds = np.array([np.std(cells[(d, h)]) if (d, h) in cells else 0.0 for d in delays])
        x = np.arange(len(delays)) + (i - (len(hists) - 1) / 2) * width
        ax.bar(x, means, width, yerr=stds, capsize=3, label=f"action history = {h}")
        for xi, mi, si in zip(x, means, stds):
            if np.isfinite(mi):
                ax.annotate(f"{mi:.0f}", (xi, mi + si), textcoords="offset points", xytext=(0, 3),
                            ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(3, 900)                                  # headroom above the ~400 mm failed cells
    ax.set_xticks(np.arange(len(delays)))
    ax.set_xticklabels(delays)
    ax.set_xlabel("control delay (steps)")
    ax.set_ylabel("held-out position RMSE (mm, log scale)")
    ax.set_title("Control-delay ablation: does action history help?")
    ax.legend()
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


PLOTTERS = {"data_scaling": plot_data_scaling, "noise": plot_noise, "delay": plot_delay,
            "action_smoothing": plot_action_smoothing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, choices=list(PLOTTERS))
    args = parser.parse_args()

    results, study_dir = _load(args.study)
    if not results:
        print("No results found.")
        return
    out = os.path.join(study_dir, f"{args.study}.png")
    PLOTTERS[args.study](results, out)
    print(f"[plot] {len(results)} runs -> {out}")


if __name__ == "__main__":
    main()
