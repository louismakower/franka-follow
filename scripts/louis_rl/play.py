"""Script to play a trained louis_rl policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained louis_rl policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pth file).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, help="Name of the RL agent configuration entry point.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the play run.")
parser.add_argument("--video_length", type=int, default=200, help="Number of steps to record.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from louis_rl.rl_runner import RLRunner
from louis_rl.isaac.env_wrapper import IsaacEnvWrapper
from cli_args import _add_agent

import followtrajectory.tasks  # noqa: F401

# fix louis_rl agent cli arg
args_cli = _add_agent(args_cli)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(args_cli.checkpoint), "videos")
        checkpoint_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        print("[INFO] Recording video to:", video_folder)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
            name_prefix=checkpoint_name,
        )

    env = IsaacEnvWrapper(env)
    runner = RLRunner(env, agent_cfg, log_dir="/tmp/louis_rl_play", inference_only=True)
    runner.load_checkpoint(args_cli.checkpoint)

    obs, _ = env.reset()

    timestep = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            action = runner.get_deterministic_action(obs)
            obs, _, _, _, _ = env.step(action)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
