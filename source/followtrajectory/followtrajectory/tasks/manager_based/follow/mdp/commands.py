from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.envs import ManagerBasedRLEnv

from scipy.interpolate import CubicSpline
import torch

class TrajectoryCommand(CommandTerm):
    cfg: TrajectoryCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        self._init_trajectory()
        self.traj_id = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.traj_progress = self._random_phase(self.traj_id)

    def _init_trajectory(self):
        # fit each trajectory's cubic spline 
        # pad every trajectory to a common length so the bank is a single (num_traj, max_len, 3)
        pos_list, vel_list, lengths = [], [], []
        for traj in self.cfg.trajectories:
            points = torch.tensor(traj)  # (num_waypoints, 4), keep on cpu for scipy
            t = points[:, 3]
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            cs_x = CubicSpline(t, x, bc_type=self.cfg.bc_type)
            cs_y = CubicSpline(t, y, bc_type=self.cfg.bc_type)
            cs_z = CubicSpline(t, z, bc_type=self.cfg.bc_type)

            length = int((t[-1] - t[0]) / self._env.step_dt)
            t_fine = torch.linspace(float(t[0]), float(t[-1]), length)
            pos_list.append(torch.stack([
                torch.from_numpy(cs_x(t_fine)).to(dtype=torch.float32),
                torch.from_numpy(cs_y(t_fine)).to(dtype=torch.float32),
                torch.from_numpy(cs_z(t_fine)).to(dtype=torch.float32),
            ], dim=1))  # (length, 3)
            vel_list.append(torch.stack([
                torch.from_numpy(cs_x(t_fine, 1)).to(dtype=torch.float32),
                torch.from_numpy(cs_y(t_fine, 1)).to(dtype=torch.float32),
                torch.from_numpy(cs_z(t_fine, 1)).to(dtype=torch.float32),
            ], dim=1))  # (length, 3)
            lengths.append(length)

        self.num_traj = len(pos_list)
        self.traj_lengths = torch.tensor(lengths, device=self.device, dtype=torch.long)  # (num_traj,)
        max_len = int(self.traj_lengths.max().item())
        self.positions = torch.zeros((self.num_traj, max_len, 3), device=self.device)
        self.velocities = torch.zeros((self.num_traj, max_len, 3), device=self.device)
        for i in range(self.num_traj):
            self.positions[i, : lengths[i]] = pos_list[i].to(self.device)
            self.velocities[i, : lengths[i]] = vel_list[i].to(self.device)

        self._command = torch.zeros((self.num_envs, self.cfg.future_length * 6), device=self.device)  # 3 pos, 3 vel

    def _random_phase(self, traj_id: torch.Tensor) -> torch.Tensor:
        # phase in [0, length) for each env's assigned trajectory
        return (torch.rand(traj_id.shape[0], device=self.device) * self.traj_lengths[traj_id]).long()

    def _update_command(self):
        self.traj_progress = (self.traj_progress + 1) % self.traj_lengths[self.traj_id]
        self._command = self._calculate_command()

    def _resample_command(self, env_ids):
        # trajectory assignment handled in reset
        pass

    def _calculate_command(self):
        lengths = self.traj_lengths[self.traj_id]  # (num_envs,)
        offsets = torch.arange(self.cfg.future_length, device=self.device)
        # look-ahead modulo each env's trajectory length
        idx = (self.traj_progress[:, None] + offsets[None, :]) % lengths[:, None]  # (num_envs, future_length)
        env_idx = self.traj_id[:, None]  # (num_envs, 1) broadcasts against idx -> (num_envs, future_length)
        targ_pos = self.positions[env_idx, idx]  # (num_envs, future_length, 3)
        targ_vel = self.velocities[env_idx, idx]  # (num_envs, future_length, 3)
        return torch.cat([
            targ_pos.reshape(self.num_envs, -1),
            targ_vel.reshape(self.num_envs, -1)
        ], dim=-1)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        # pick a new trajectory and a random starting phase along it
        self.traj_id[ids] = torch.randint(0, self.num_traj, (n,), device=self.device)
        self.traj_progress[ids] = self._random_phase(self.traj_id[ids])
        extras = super().reset(env_ids)
        self._command = self._calculate_command()  # refresh so first obs of the episode isn't stale
        return extras
    
    @property
    def command(self):
        return self._command
    
    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create the marker the first time visualization is enabled
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/trajectory_goal")
                marker_cfg.markers["sphere"].radius = 0.025
                marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self.goal_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot may be de-initialized, in which case its data is unavailable
        if not self.robot.is_initialized:
            return
        # current tracking point (step-0 target) is in the root frame -> transform to world
        goal_pos_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w, self._command[:, :3]
        )
        self.goal_visualizer.visualize(translations=goal_pos_w)


@configclass
class TrajectoryCommandCfg(CommandTermCfg):
    class_type: type = TrajectoryCommand
    # bank of closed loops; each is a list of [x, y, z, t] waypoints, e.g. [[x1,y1,z1,t1], ...]
    trajectories: list[list[list[float]]] = MISSING
    bc_type: str = MISSING
    future_length: int = MISSING