<h2 align="center">
  📺 For methodology, results &amp; videos, see the
  <a href="https://louismakower.github.io/franka-follow/">PROJECT PAGE →</a>
</h2>

# HUMANOID Challenge

Built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and my own
[SAC](https://github.com/louismakower/louis_rl) agent.


## Install

1. **Install IsaacLab** (and IsaacSim) as per the
   [guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
   This repo assumes IsaacLab's Python (a conda/venv with `isaaclab` importable).

2. **Clone this repo with submodules** (the [`louis_rl`](https://github.com/louismakower/louis_rl)
  SAC/PPO implementation):

    ```bash
    git clone --recursive https://github.com/louismakower/franka-follow.git
    cd franka-follow
    # if you already cloned without --recursive:
    git submodule update --init --recursive
    ```

3. **Install the extension and the RL library** (editable) into Isaac Lab's Python:

   ```bash
   python -m pip install -e source/followtrajectory
   python -m pip install -e louis_rl
   ```

## Run

Train the SAC agent:

`python scripts/louis_rl/train.py --agent sac --task follow --num_envs 256 --headless`

See the trained agent run live:

`python scripts/louis_rl/play.py --agent sac --task follow --num_envs 8 --checkpoint /path/to/checkpoint`

See a trained agent run live:

`python scripts/louis_rl/play.py --agent sac --task follow --num_envs 8 --checkpoint TODO`

Create a custom trajectory, and see the robot trace it out

1. First, define your trajectory as a sequence of $(x_i, y_i, z_i, t_ i)$ points in [custom_trajectory.py](TODO/path/to/custom_trajectory.py).
2. Then run a pretrained agent on your trajectory:

     `python scripts/louis_rl/play.py --agent sac --task follow --num_envs 8 --checkpoint /path/to/checkpoint --custom_trajectory`

## Repository layout

```bash
source/followtrajectory/.../tasks/manager_based/follow/
  follow_env_cfg.py  # scene, observations, rewards, terminations, env settings
  config/franka/  # Franka robot binding + SAC/PPO agent configs + gym registration
  mdp/
    trajectories.py  # trajectory generators + TRAIN/EVAL banks + make_bank() factory
    commands.py  # TrajectoryCommand: spline fit, look-ahead, deterministic eval
    actions.py  # SmoothedJointPositionAction: EMA smoothing, noise, control delay
    rewards.py  # position/velocity tracking reward terms
scripts/louis_rl/
  train.py  play.py  eval.py  # train / visualise / evaluate
  run_study.py  plot_study.py  # multi-seed studies + aggregation plots
louis_rl/  # git submodule: SAC/PPO implementation
```