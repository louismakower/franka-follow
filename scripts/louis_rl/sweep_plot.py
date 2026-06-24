"""Visualise hyperparameter sweep results.

Produces three plots:
  1. Parallel coordinates — all parameters at once, coloured by reward
  2. Marginal effects    — reward vs each parameter individually
  3. Importance          — Spearman rank correlation per parameter

Usage:
    python scripts/louis_rl/sweep_plot.py logs/louis_rl_sac/sweeps/sweep_*.json
"""

import argparse
import glob
import json
import os
import struct

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event


# ── data loading ──────────────────────────────────────────────────────────────

def read_tag_from_events(path, tag):
    tag_bytes = tag.encode()
    vals = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            length = struct.unpack("<Q", hdr[:8])[0]
            data = f.read(length)
            f.read(4)
            if tag_bytes not in data:
                continue
            ev = Event()
            ev.ParseFromString(data)
            for v in ev.summary.value:
                if v.tag == tag:
                    vals.append(v.simple_value)
    return vals


METRIC_LABEL = {
    "mean":       "mean reward",
    "iqm":        "IQM reward",
    "p25":        "p25 reward",
    "median":     "median reward",
    "sharpe":     "reward Sharpe (μ/σ)",
    "final_mean": "final-20% mean reward",
}


def compute_metric(vals, metric):
    vals = np.array(vals)
    if metric == "mean":
        return float(np.mean(vals))
    if metric == "iqm":
        lo, hi = np.percentile(vals, 25), np.percentile(vals, 75)
        mask = (vals >= lo) & (vals <= hi)
        return float(np.mean(vals[mask])) if mask.any() else float(np.mean(vals))
    if metric == "p25":
        return float(np.percentile(vals, 25))
    if metric == "median":
        return float(np.median(vals))
    if metric == "sharpe":
        return float(np.mean(vals) / (np.std(vals) + 1e-8))
    if metric == "final_mean":
        tail = vals[int(0.8 * len(vals)):]
        return float(np.mean(tail)) if len(tail) else float(np.mean(vals))
    raise ValueError(f"Unknown metric: {metric}")


def get_avg_reward(log_dir, tag, metric="mean"):
    subdirs = sorted(glob.glob(os.path.join(log_dir, "*")))
    if not subdirs:
        return None
    event_files = sorted(glob.glob(os.path.join(subdirs[-1], "events.out.tfevents.*")))
    if not event_files:
        return None
    vals = read_tag_from_events(event_files[-1], tag)
    return compute_metric(vals, metric) if vals else None


# ── plot 1: parallel coordinates ──────────────────────────────────────────────

def plot_parallel(param_vals, rewards, names, metric_label="mean reward"):
    _, n_params = param_vals.shape
    r_norm = (rewards - rewards.min()) / max(rewards.max() - rewards.min(), 1e-9)
    cmap = plt.cm.viridis

    # Rank-normalise each column so discrete values are evenly spaced vertically
    normed = np.zeros_like(param_vals, dtype=float)
    axis_ticks = []
    for j in range(n_params):
        col = param_vals[:, j]
        unique = np.unique(col)
        rank = {v: i / max(len(unique) - 1, 1) for i, v in enumerate(unique)}
        normed[:, j] = [rank[v] for v in col]
        axis_ticks.append(([rank[v] for v in unique], [str(v) for v in unique]))

    fig, ax = plt.subplots(figsize=(n_params * 1.8 + 1, 5))

    for i in np.argsort(r_norm):  # draw high-reward lines on top
        ax.plot(range(n_params), normed[i], color=cmap(r_norm[i]),
                alpha=0.15 + 0.65 * r_norm[i], linewidth=0.8)

    for j in range(n_params):
        ax.axvline(j, color="gray", linewidth=0.8, zorder=0)
        for ty, tl in zip(*axis_ticks[j]):
            ax.text(j, ty, tl, ha="center", va="bottom", fontsize=6,
                    bbox=dict(fc="white", ec="none", pad=0.5))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(rewards.min(), rewards.max()))
    plt.colorbar(sm, ax=ax, label=metric_label)
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.5, n_params - 0.5)
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Parallel Coordinates")
    fig.tight_layout()
    return fig


# ── plot 2: marginal effects ───────────────────────────────────────────────────

def plot_marginal(param_vals, rewards, names, col_encodings=None):
    col_encodings = col_encodings or {}
    n_params = len(names)
    ncols = 4
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))

    rng = np.random.default_rng(0)
    for j, ax in enumerate(axes.flat):
        if j >= n_params:
            ax.set_visible(False)
            continue

        col = param_vals[:, j]
        unique = np.unique(col)
        is_categorical = j in col_encodings
        log_x = not is_categorical and all(v > 0 for v in unique)

        # Jitter in log- or linear-space
        if log_x and len(unique) > 1:
            min_gap_log = np.min(np.diff(np.log(unique)))
            jitter = col * np.exp(rng.uniform(-min_gap_log * 0.15, min_gap_log * 0.15, len(col)))
        elif not is_categorical and len(unique) > 1:
            min_gap = np.min(np.diff(unique))
            jitter = col + rng.uniform(-min_gap * 0.15, min_gap * 0.15, len(col))
        else:
            jitter = col

        ax.scatter(jitter, rewards, s=18, alpha=0.5, c=rewards, cmap="viridis")

        # Mean reward per unique value (red line)
        means = [rewards[col == v].mean() for v in unique]
        ax.plot(unique, means, "r.-", linewidth=1.5, markersize=7, zorder=3)

        if log_x:
            ax.set_xscale("log")
        if is_categorical:
            inv = {code: label for label, code in col_encodings[j].items()}
            ax.set_xticks(unique)
            ax.set_xticklabels([str(inv[v]) for v in unique], fontsize=7)
        ax.set_xlabel(names[j], fontsize=8)
        ax.set_ylabel("reward" if j % ncols == 0 else "")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Marginal Effects", fontsize=11)
    fig.tight_layout()
    return fig


# ── plot 3: importance ────────────────────────────────────────────────────────

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_importance(param_vals, rewards, names, metric_label="mean reward"):
    corrs = np.array([spearman(param_vals[:, j], rewards) for j in range(len(names))])
    order = np.argsort(np.abs(corrs))

    fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.45)))
    colors = ["steelblue" if c >= 0 else "tomato" for c in corrs[order]]
    ax.barh(range(len(names)), corrs[order], color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Spearman ρ with {metric_label}")
    ax.set_title("Parameter Importance")
    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", help="Path to sweep JSON")
    parser.add_argument("--log-dir", default="logs/louis_rl_sac")
    parser.add_argument("--tag", default="Episode_Reward/obj_height")
    parser.add_argument(
        "--reward-metric",
        default="mean",
        choices=list(METRIC_LABEL),
        help="How to summarise per-run rewards. Choices: " + ", ".join(METRIC_LABEL),
    )
    args = parser.parse_args()

    with open(args.sweep) as f:
        sweep = json.load(f)

    param_names = list(sweep["runs"][0].keys())
    short_names = [p.split(".")[-1] for p in param_names]

    rows, rewards = [], []
    for i, params in enumerate(sweep["runs"]):
        print(f"  loading run {i:03d}...", end="\r", flush=True)
        reward = get_avg_reward(os.path.join(args.log_dir, f"lift_sweep_{i:03d}"), args.tag, args.reward_metric)
        if reward is None:
            continue
        rows.append([params[k] for k in param_names])
        rewards.append(reward)

    print(f"Loaded {len(rewards)} runs.          ")
    if not rewards:
        print("No data found.")
        return

    # Encode categorical columns (e.g. target_entropy='auto') as integer codes.
    rows_encoded = []
    for row in rows:
        encoded = []
        for v in row:
            try:
                encoded.append(float(v))
            except (ValueError, TypeError):
                encoded.append(v)
        rows_encoded.append(encoded)

    col_encodings = {}  # col_index -> {original_val: float}
    for j in range(len(param_names)):
        col = [r[j] for r in rows_encoded]
        if any(isinstance(v, str) for v in col):
            unique = sorted(set(col), key=str)
            mapping = {v: float(i) for i, v in enumerate(unique)}
            col_encodings[j] = {v: i for i, v in enumerate(unique)}
            for row in rows_encoded:
                row[j] = mapping[row[j]]

    param_vals = np.array(rows_encoded, dtype=float)
    rewards = np.array(rewards)

    metric_label = METRIC_LABEL[args.reward_metric]
    stem = args.sweep.replace(".json", "")
    plot_parallel(param_vals, rewards, short_names, metric_label)
    plt.savefig(f"{stem}_parallel.png", dpi=150, bbox_inches="tight")

    plot_marginal(param_vals, rewards, short_names, col_encodings)
    plt.savefig(f"{stem}_marginal.png", dpi=150, bbox_inches="tight")

    plot_importance(param_vals, rewards, short_names, metric_label)
    plt.savefig(f"{stem}_importance.png", dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
