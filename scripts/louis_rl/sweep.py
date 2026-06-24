"""Random hyperparameter sweep for SAC on the lift task.

Usage:
    python scripts/louis_rl/sweep.py [--num_runs N] [--seed S] [--start_run R] -- --headless --num_envs 256

Arguments after '--' are forwarded verbatim to train.py (e.g. --headless, --num_envs).
--start_run lets you resume a sweep after a crash (re-use same --seed to reproduce the same configs).
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

# UTD = batch_size * num_train_updates / (num_envs * steps_per_iter)
# With the defaults (4096 * 1) / (256 * 16) = 1. Don't vary these independently.
FIXED = {
    # don't sweep: gamma, network sizes, max_steps, experiment_name
}

PARAM_RANGES = {
    "agent.alpha_init":             [0.01, 0.05, 0.1, 0.2],
    "agent.alpha_lr":               [0.1, 0.01],
    "agent.target_entropy":         ["auto", "-12.0", "-3.0"],
    "agent.q_learning_rate":        [0.0003, 0.001, 0.003],
    "agent.policy_learning_rate":   [0.0003, 0.001, 0.003],
    "agent.q_tau":                  [0.005, 0.01, 0.05],
    "agent.q_grad_clip_norm":       [0.5, 1.0, 5.0],
    "agent.warmup_steps":           [1000, 5000],
    "agent.logstd_max":             [0.5, 1.0, 2.0],
    "agent.logstd_min":             [-0.5, -1.0, -2.0],
    "agent.steps_per_iter":         [4, 16, 32, 64, 128],
    "agent.num_train_updates":      [1, 8, 32, 64],
    "agent.batch_size":             [256, 1024, 4096],
    "agent.reward_scaling":         [True, False],
    "agent.reward_G_max":           [1.0, 5.0, 10.0],
    "agent.reward_clip":            [0.0, 1.0, 5.0, 10.0],
}


def sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_RANGES.items()}


def run_trial(run_id: int, params: dict, passthrough_args: list[str]) -> int:
    experiment_name = f"lift_sweep_{run_id:03d}"
    overrides = [f"agent.experiment_name={experiment_name}"]
    for k, v in {**FIXED, **params}.items():
        overrides.append(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}")

    cmd = [sys.executable, "scripts/louis_rl/train.py", "--task", "lift"] + passthrough_args + overrides

    print(f"\n{'='*60}")
    print(f"  Run {run_id:03d} — {experiment_name}")
    for k, v in params.items():
        print(f"    {k.split('.')[-1]:<30} {v}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Random SAC hyperparameter sweep.")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of random configs to sample.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--start_run", type=int, default=0, help="Skip runs before this index (for resuming).")
    args, passthrough_args = parser.parse_known_args()

    rng = random.Random(args.seed)
    configs = [sample_params(rng) for _ in range(args.num_runs)]

    sweep_dir = os.path.join("logs", "louis_rl_sac", "sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_path = os.path.join(sweep_dir, f"sweep_{sweep_id}.json")
    with open(config_path, "w") as f:
        json.dump({"seed": args.seed, "num_runs": args.num_runs, "runs": configs}, f, indent=2)
    print(f"Sweep config written to {config_path}")
    print(f"Running {args.num_runs} trials (starting from run {args.start_run})\n")

    failed = []
    for i, params in enumerate(configs):
        if i < args.start_run:
            continue
        returncode = run_trial(i, params, passthrough_args)
        if returncode != 0:
            print(f"[WARNING] Run {i:03d} exited with code {returncode}")
            failed.append(i)

    print(f"\nSweep complete. {len(failed)} failed runs: {failed}")


if __name__ == "__main__":
    main()
