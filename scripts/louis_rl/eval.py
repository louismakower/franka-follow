"""Evaluate a trained louis_rl policy on the held-out trajectory bank.

Rolls the deterministic policy out on every trajectory of a split (default: the held-out ``eval``
bank), one trajectory per environment, recording the desired and achieved end-effector position and
velocity at every step. From these it computes per-trajectory tracking metrics — RMSE, mean error,
time-integrated error (the "area under the error curve") and an action-jitter proxy — then writes:

  * ``eval/metrics.json``   — per-trajectory + aggregate metrics (machine-readable, used by sweeps)
  * ``eval/traces.npz``     — raw (T, N, 3) desired/actual position & velocity arrays
  * ``eval/tracking_error.png`` — position/velocity error vs time, one line per trajectory
  * ``eval/paths_3d.png``   — desired vs achieved 3-D path overlay per trajectory

Usage:
    python scripts/louis_rl/eval.py --task follow --agent sac \
        --checkpoint logs/louis_rl/franka_follow_sac/<run>/checkpoints/model_<n>.pth --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained louis_rl policy on tracking metrics.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pth file).")
parser.add_argument("--task", type=str, default="follow", help="Name of the task.")
parser.add_argument("--agent", type=str, help="Name of the RL agent configuration entry point.")
parser.add_argument("--split", type=str, default="eval", choices=["train", "eval"], help="Trajectory split.")
parser.add_argument("--repeats", type=int, default=1, help="Environments per trajectory (averages over noise).")
parser.add_argument("--steps", type=int, default=None, help="Rollout steps (default: one full episode).")
parser.add_argument("--noise", action="store_true", default=False, help="Keep observation corruption on during eval.")
parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <checkpoint_dir>/../eval).")
parser.add_argument("--tag", type=str, default=None, help="Optional suffix for output filenames.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from louis_rl.rl_runner import RLRunner
from louis_rl.isaac.env_wrapper import IsaacEnvWrapper
from cli_args import _add_agent

import followtrajectory.tasks  # noqa: F401
from followtrajectory.tasks.manager_based.follow.mdp import make_bank

# fix louis_rl agent cli arg
args_cli = _add_agent(args_cli)

# the tracked end-effector body (must match the reward asset_cfg in the env config)
TRACKED_BODY = "panda_hand"


def _metrics(des, act, dt):
    """Per-environment tracking metrics from desired/actual arrays of shape (T, N, 3)."""
    err = np.linalg.norm(act - des, axis=-1)  # (T, N)
    return {
        "rmse": np.sqrt((err**2).mean(axis=0)),       # (N,)
        "mae": err.mean(axis=0),                       # mean error over time
        "integral": err.sum(axis=0) * dt,              # time-integrated error (area under curve)
        "max": err.max(axis=0),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    # ---- configure the env for reproducible held-out evaluation -------------------------------
    traj_cfg = env_cfg.commands.trajectory
    traj_cfg.split = args_cli.split
    traj_cfg.num_trajectories = 0            # 0 = use the whole split
    traj_cfg.deterministic_eval = True       # env i -> trajectory i (mod num_traj) at phase 0
    traj_cfg.debug_vis = False
    env_cfg.observations.policy.enable_corruption = args_cli.noise

    n_traj = len(make_bank(args_cli.split))
    env_cfg.scene.num_envs = n_traj * args_cli.repeats

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = IsaacEnvWrapper(env)
    runner = RLRunner(env, agent_cfg, log_dir="/tmp/louis_rl_eval", inference_only=True)
    runner.load_checkpoint(args_cli.checkpoint)

    base = env.unwrapped
    robot = base.scene["robot"]
    body_id = robot.find_bodies(TRACKED_BODY)[0][0]
    dt = base.step_dt
    horizon = args_cli.steps if args_cli.steps is not None else int(base.max_episode_length) - 1

    obs, _ = env.reset()

    des_p, act_p, des_v, act_v, actions = [], [], [], [], []
    with torch.inference_mode():
        for _ in range(horizon):
            action = runner.get_deterministic_action(obs)

            cmd = base.command_manager.get_command("trajectory")
            des_pos_b = cmd[:, :3]
            des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
            vel_start = cmd.shape[1] // 2
            des_vel_w = quat_apply(robot.data.root_quat_w, cmd[:, vel_start : vel_start + 3])

            des_p.append(des_pos_w.cpu().numpy())
            act_p.append(robot.data.body_pos_w[:, body_id].cpu().numpy())
            des_v.append(des_vel_w.cpu().numpy())
            act_v.append(robot.data.body_lin_vel_w[:, body_id].cpu().numpy())
            actions.append(action.cpu().numpy())

            obs, _, _, _, _ = env.step(action)

    env.close()

    des_p = np.stack(des_p); act_p = np.stack(act_p)        # (T, N, 3)
    des_v = np.stack(des_v); act_v = np.stack(act_v)
    actions = np.stack(actions)                              # (T, N, A)

    # ---- metrics (averaged across the `repeats` envs assigned to each trajectory) --------------
    pos = _metrics(des_p, act_p, dt)
    vel = _metrics(des_v, act_v, dt)
    # jitter proxy: mean per-step action change magnitude (smaller = smoother)
    jitter = np.linalg.norm(np.diff(actions, axis=0), axis=-1).mean(axis=0)  # (N,)

    def per_traj(arr):
        return arr.reshape(args_cli.repeats, n_traj).mean(axis=0) if args_cli.repeats > 1 else arr[:n_traj]

    per = {
        "pos_rmse_m": per_traj(pos["rmse"]),
        "pos_mae_m": per_traj(pos["mae"]),
        "pos_integral_ms": per_traj(pos["integral"]),
        "pos_max_m": per_traj(pos["max"]),
        "vel_rmse_mps": per_traj(vel["rmse"]),
        "vel_integral_mps_s": per_traj(vel["integral"]),
        "action_jitter": per_traj(jitter),
    }
    summary = {
        "checkpoint": args_cli.checkpoint,
        "split": args_cli.split,
        "n_trajectories": n_traj,
        "horizon_steps": horizon,
        "dt": dt,
        "noise": args_cli.noise,
        "aggregate": {k: float(np.mean(v)) for k, v in per.items()},
        "per_trajectory": {k: v.tolist() for k, v in per.items()},
    }

    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(args_cli.checkpoint), os.pardir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{args_cli.tag}" if args_cli.tag else ""

    with open(os.path.join(out_dir, f"metrics{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, f"traces{suffix}.npz"),
        des_pos=des_p, act_pos=act_p, des_vel=des_v, act_vel=act_v, actions=actions,
    )
    _plot_tracking_error(des_p, act_p, des_v, act_v, dt, n_traj, os.path.join(out_dir, f"tracking_error{suffix}.png"))
    _plot_paths_3d(des_p, act_p, n_traj, os.path.join(out_dir, f"paths_3d{suffix}.png"))

    # ---- console summary -----------------------------------------------------------------------
    print(f"\n[eval] split={args_cli.split}  trajectories={n_traj}  noise={args_cli.noise}")
    print(f"{'traj':>4} | {'pos_rmse(mm)':>12} | {'pos_int(mm·s)':>13} | {'vel_rmse':>9} | {'jitter':>8}")
    for i in range(n_traj):
        print(f"{i:>4} | {1e3*per['pos_rmse_m'][i]:>12.2f} | {1e3*per['pos_integral_ms'][i]:>13.2f} | "
              f"{per['vel_rmse_mps'][i]:>9.4f} | {per['action_jitter'][i]:>8.4f}")
    agg = summary["aggregate"]
    print(f"{'MEAN':>4} | {1e3*agg['pos_rmse_m']:>12.2f} | {1e3*agg['pos_integral_ms']:>13.2f} | "
          f"{agg['vel_rmse_mps']:>9.4f} | {agg['action_jitter']:>8.4f}")
    print(f"[eval] wrote results to {os.path.abspath(out_dir)}")


def _plot_tracking_error(des_p, act_p, des_v, act_v, dt, n_traj, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(des_p.shape[0]) * dt
    pos_err = np.linalg.norm(act_p[:, :n_traj] - des_p[:, :n_traj], axis=-1) * 1e3  # mm
    vel_err = np.linalg.norm(act_v[:, :n_traj] - des_v[:, :n_traj], axis=-1)        # m/s
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i in range(n_traj):
        ax1.plot(t, pos_err[:, i], lw=1.0, label=f"traj {i}")
        ax2.plot(t, vel_err[:, i], lw=1.0)
    ax1.set_ylabel("position error (mm)")
    ax1.set_title("End-effector tracking error over time (held-out trajectories)")
    ax1.legend(fontsize=7, ncol=max(1, n_traj // 2))
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("velocity error (m/s)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_paths_3d(des_p, act_p, n_traj, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = min(n_traj, 3)
    nrows = (n_traj + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 3.6, nrows * 3.4))
    for i in range(n_traj):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        d, a = des_p[:, i], act_p[:, i]
        ax.plot(d[:, 0], d[:, 1], d[:, 2], "g-", lw=1.5, label="desired")
        ax.plot(a[:, 0], a[:, 1], a[:, 2], "b-", lw=1.0, alpha=0.8, label="achieved")
        ax.set_title(f"traj {i}", fontsize=9)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Desired vs achieved end-effector path", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
    simulation_app.close()
