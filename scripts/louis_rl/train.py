"""Script to train RL agent with Louis-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# local imports
from cli_args import _add_agent

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Louis-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, help="Name of the RL agent configuration entry point.", required=True,
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--visualise", action="store_true", default=False, help="Show live value-vs-distance plot (requires a display).")
parser.add_argument("--max_steps", type=int, default=None, help="Total environment steps to train for.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="Track experiment with Weights & Biases.",
)
parser.add_argument("--wandb-project-name", type=str, default=None, help="wandb project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity (team)")
parser.add_argument("--wandb-name", type=str, default=None, help="wandb run name (overrides experiment_name)")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)

# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# fix louis_rl agent cli arg
args_cli = _add_agent(args_cli)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from louis_rl.rl_runner import RLRunner
from louis_rl.isaac.env_wrapper import IsaacEnvWrapper

# import logger
logger = logging.getLogger(__name__)

import followtrajectory.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    log_root_path = os.path.abspath(os.path.join("logs", "louis_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    if args_cli.track:
        import dataclasses
        import wandb
        wandb.init(
            project=agent_cfg.experiment_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name,
            entity=args_cli.wandb_entity,
            name=agent_cfg.experiment_name if args_cli.wandb_name is None else args_cli.wandb_name,
            sync_tensorboard=True,
            save_code=True,
            config=dataclasses.asdict(agent_cfg),
        )

    env = IsaacEnvWrapper(env, add_terminal_obs=True)
    runner = RLRunner(env, agent_cfg, log_dir=log_dir)

    if args_cli.visualise:
        if args_cli.task == "reach":
            from grasp2grasp.tasks.manager_based.franka_reach.visualise import (
                PPOValueVisualiser,
                SACValueVisualiser,
            )
            # visualiser_cls = {"ppo": PPOValueVisualiser, "sac": SACValueVisualiser}.get(
            #     agent_cfg.algo_name.lower()
            # )
            # env._visualisers.append(visualiser_cls(runner, env))
        else:
            print(f"[WARN] --visualise: no value visualiser for task={args_cli.task}")

    runner.learn()
    env.close()


if __name__ == "__main__":
    main()
