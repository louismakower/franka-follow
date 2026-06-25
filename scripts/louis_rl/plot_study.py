"""Aggregate and plot the results of a study produced by ``run_study.py``.

Reads ``logs/studies/<study>/results.json`` (one record per train+eval run) and produces a summary
figure with mean +/- std over seeds:

  * ``data_scaling`` — held-out position RMSE vs number of training trajectories.
  * ``noise``        — held-out position RMSE and action jitter vs actuation-noise std.
  * ``delay``        — grouped bars of position RMSE by (control delay, action-history length).

Usage:
    python scripts/louis_rl/plot_study.py --study data_scaling
"""

import argparse
import json
import os
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


def plot_delay(results, out):
    # group by (delay, history) -> pos rmse
    cells = defaultdict(list)
    for r in results:
        c = r["cell"]
        key = (c["env.actions.arm_action.delay_steps"], c["env.observations.policy_action_obs.history_length"])
        cells[key].append(r["metrics"]["pos_rmse_m"] * 1e3)
    delays = sorted({k[0] for k in cells})
    hists = sorted({k[1] for k in cells}, reverse=True)
    width = 0.8 / len(hists)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, h in enumerate(hists):
        means = [np.mean(cells[(d, h)]) if (d, h) in cells else np.nan for d in delays]
        stds = [np.std(cells[(d, h)]) if (d, h) in cells else 0.0 for d in delays]
        x = np.arange(len(delays)) + (i - (len(hists) - 1) / 2) * width
        ax.bar(x, means, width, yerr=stds, capsize=3, label=f"action history = {h}")
    ax.set_xticks(np.arange(len(delays)))
    ax.set_xticklabels(delays)
    ax.set_xlabel("control delay (steps)")
    ax.set_ylabel("held-out position RMSE (mm)")
    ax.set_title("Control-delay ablation: does action history help?")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


PLOTTERS = {"data_scaling": plot_data_scaling, "noise": plot_noise, "delay": plot_delay}


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
