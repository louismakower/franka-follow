"""Run a multi-seed experiment study: train a grid of configs, evaluate each on the held-out bank.

Three studies, matching the report's robustness/generalisation analyses:

  * ``data_scaling`` — vary the number of *training* trajectories; eval on the fixed held-out bank.
                       Answers "how many demo shapes does the policy need to generalise?".
  * ``noise``        — vary actuation noise (``arm_action.noise_std``); train *and* eval under it.
                       Answers "how does tracking degrade as disturbance grows?".
  * ``delay``        — grid of control delay x action-history length. Tests whether observing the
                       action history lets the policy compensate for control latency.

For every (config, seed) it launches ``train.py`` then ``eval.py`` and appends the eval metrics to
``logs/studies/<study>/results.json`` (written incrementally so a crash is resumable). Plot the
result with ``plot_study.py``.

Usage:
    python scripts/louis_rl/run_study.py --study data_scaling --seeds 0 1 2 \
        --max_steps 500000 --num_envs 256 -- --headless

Everything after ``--`` is forwarded verbatim to both train.py and eval.py (e.g. ``--headless``).
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime

TASK = "follow"
AGENT = "sac"
LOG_ROOT = os.path.join("logs", "louis_rl")  # train.py logs here under <experiment_name>/<timestamp>

# Per-study grids. Each entry is a dict of Hydra overrides describing one cell of the grid; the
# values are applied to BOTH training and evaluation (eval forces the held-out split internally, so
# the num_trajectories override is harmless there).
STUDIES = {
    "data_scaling": [
        {"env.commands.trajectory.num_trajectories": n} for n in (1, 2, 3, 5, 7, 9)
    ],
    "noise": [
        {"env.actions.arm_action.noise_std": s} for s in (0.0, 0.02, 0.05, 0.1, 0.2)
    ],
    "delay": [
        {"env.actions.arm_action.delay_steps": d, "env.observations.policy_action_obs.history_length": h}
        for d, h in itertools.product((0, 2, 4), (5, 1))
    ],
}


def _run_name(study: str, cell: dict, seed: int) -> str:
    parts = [f"{k.split('.')[-1]}={v}" for k, v in cell.items()]
    return f"{study}__{'_'.join(parts)}__seed{seed}"


def _overrides(cell: dict) -> list[str]:
    return [f"{k}={v}" for k, v in cell.items()]


def _latest_checkpoint(experiment_name: str) -> str | None:
    exp_dir = os.path.join(LOG_ROOT, experiment_name)
    runs = sorted(d for d in os.listdir(exp_dir)) if os.path.isdir(exp_dir) else []
    for run in reversed(runs):  # newest first
        ckpt_dir = os.path.join(exp_dir, run, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        if ckpts:
            newest = max(ckpts, key=lambda f: int(f.split("_")[-1].split(".")[0]))
            return os.path.join(ckpt_dir, newest)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run a multi-seed train+eval study.")
    parser.add_argument("--study", required=True, choices=list(STUDIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # SAC max_steps counts learn-iterations (each steps all envs once + does num_train_updates
    # gradient steps), so this is far smaller than total env-steps. Tune to your compute budget.
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs already in results.json.")
    args, passthrough = parser.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]

    study_dir = os.path.join("logs", "studies", args.study)
    os.makedirs(study_dir, exist_ok=True)
    results_path = os.path.join(study_dir, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else []
    done = {(r["run"]) for r in results}

    grid = STUDIES[args.study]
    jobs = [(cell, seed) for cell in grid for seed in args.seeds]
    print(f"[study] {args.study}: {len(grid)} cells x {len(args.seeds)} seeds = {len(jobs)} runs")

    for i, (cell, seed) in enumerate(jobs):
        run = _run_name(args.study, cell, seed)
        if args.skip_existing and run in done:
            print(f"[study] ({i+1}/{len(jobs)}) skip existing {run}")
            continue
        experiment_name = f"study_{run}"
        print(f"\n{'='*70}\n[study] ({i+1}/{len(jobs)}) {run}\n{'='*70}", flush=True)

        # ---- train -------------------------------------------------------------------------
        train_cmd = [
            sys.executable, "scripts/louis_rl/train.py", "--task", TASK, "--agent", AGENT,
            "--num_envs", str(args.num_envs), "--seed", str(seed), *passthrough,
            f"agent.experiment_name={experiment_name}", f"agent.max_steps={args.max_steps}",
            *_overrides(cell),
        ]
        if subprocess.run(train_cmd).returncode != 0:
            print(f"[study] WARNING train failed for {run}; skipping eval")
            continue

        ckpt = _latest_checkpoint(experiment_name)
        if ckpt is None:
            print(f"[study] WARNING no checkpoint found for {run}; skipping eval")
            continue

        # ---- eval (same system params; eval.py forces the held-out split internally) --------
        eval_cmd = [
            sys.executable, "scripts/louis_rl/eval.py", "--task", TASK, "--agent", AGENT,
            "--checkpoint", ckpt, "--out_dir", os.path.join(study_dir, "eval"), "--tag", run,
            *passthrough, *_overrides(cell),
        ]
        if subprocess.run(eval_cmd).returncode != 0:
            print(f"[study] WARNING eval failed for {run}")
            continue

        with open(os.path.join(study_dir, "eval", f"metrics_{run}.json")) as f:
            metrics = json.load(f)
        results = [r for r in results if r["run"] != run]  # replace any stale entry
        results.append({"run": run, "study": args.study, "seed": seed, "cell": cell,
                        "checkpoint": ckpt, "metrics": metrics["aggregate"]})
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[study] recorded {run}: pos_rmse={metrics['aggregate']['pos_rmse_m']*1e3:.2f} mm")

    print(f"\n[study] complete. {len(results)} results in {results_path}")
    print(f"[study] plot with: python scripts/louis_rl/plot_study.py --study {args.study}")


if __name__ == "__main__":
    main()
