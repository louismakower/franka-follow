from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, normalize, quat_from_matrix
from isaaclab.envs import ManagerBasedRLEnv

from scipy.interpolate import CubicSpline
import torch


def look_at_quat(forward: torch.Tensor, up_ref: torch.Tensor | None = None) -> torch.Tensor:
    """Quaternion (w, x, y, z) whose local +z points along ``forward`` with local +y kept near ``up_ref``.

    ``forward`` is ``(N, 3)`` (need not be unit). ``up_ref`` is the world reference for "up" used to pin
    roll about the pointing axis (default world +z); where ``forward`` is near-parallel to it (gimbal lock)
    a fallback reference is used so the basis stays well-defined. Returns ``(N, 4)``.
    """
    f = normalize(forward)
    ref = torch.tensor([0.0, 0.0, 1.0], device=f.device, dtype=f.dtype) if up_ref is None else up_ref
    up = ref.reshape(1, 3).expand_as(f)
    # near-vertical pointing -> up_ref ~parallel to f -> pick a different reference to avoid a degenerate cross
    parallel = (f * up).sum(-1, keepdim=True).abs() > 0.99
    alt = torch.tensor([1.0, 0.0, 0.0], device=f.device, dtype=f.dtype).reshape(1, 3).expand_as(f)
    up = torch.where(parallel, alt, up)
    right = normalize(torch.cross(up, f, dim=-1))  # local +x
    new_up = torch.cross(f, right, dim=-1)  # local +y
    rot = torch.stack([right, new_up, f], dim=-1)  # columns (x, y, z) -> (N, 3, 3), right-handed
    return quat_from_matrix(rot)


class TrajectoryCommand(CommandTerm):
    cfg: TrajectoryCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: TrajectoryCommandCfg, env):
        # the base __init__ calls set_debug_vis() -> _set_debug_vis_impl() before _init_trajectory runs,
        # so this flag must exist beforehand
        self.has_look_at = cfg.look_at_targets is not None
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        self._init_trajectory()
        self.traj_id = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.traj_progress = self._random_phase(self.traj_id)
        # body the aim-ray is drawn from (vis only)
        self._ee_body_idx = int(self.robot.find_bodies("panda_hand")[0][0])

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

        # optional gimbal targets: one (x, y, z) point per trajectory the EE should aim at (root frame)
        targets_cfg = self.cfg.look_at_targets
        self.has_look_at = targets_cfg is not None
        if targets_cfg is not None:
            assert len(targets_cfg) == self.num_traj, (
                f"look_at_targets ({len(targets_cfg)}) must match number of trajectories ({self.num_traj})"
            )
            targets = torch.tensor(targets_cfg, dtype=torch.float32, device=self.device)
            assert targets.shape == (self.num_traj, 3), "each look_at_target must be an [x, y, z] point"
            self.look_at_targets = targets  # (num_traj, 3)

        # command layout: [pos (future_length*3) | vel (future_length*3) | target (3, only if look-at)]
        target_dim = 3 if self.has_look_at else 0
        self._command = torch.zeros((self.num_envs, self.cfg.future_length * 6 + target_dim), device=self.device)

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
        blocks = [targ_pos.reshape(self.num_envs, -1), targ_vel.reshape(self.num_envs, -1)]
        if self.has_look_at:
            # constant per trajectory -> append once per env (not per future step)
            blocks.append(self.look_at_targets[self.traj_id])  # (num_envs, 3)
        return torch.cat(blocks, dim=-1)

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
        # create the markers the first time visualization is enabled
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/trajectory_goal")
                marker_cfg.markers["sphere"].radius = 0.025
                marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self.goal_visualizer = VisualizationMarkers(marker_cfg)
            # gimbal target point (red sphere) + aim ray (thin red cylinder, local +z along its length)
            if self.has_look_at and not hasattr(self, "target_visualizer"):
                target_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/look_at_target")
                target_cfg.markers["sphere"].radius = 0.025
                target_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
                self.target_visualizer = VisualizationMarkers(target_cfg)
                ray_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/aim_ray",
                    markers={
                        "ray": sim_utils.CylinderCfg(
                            radius=0.004,
                            height=1.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        )
                    },
                )
                self.ray_visualizer = VisualizationMarkers(ray_cfg)
            self.goal_visualizer.set_visibility(True)
            if self.has_look_at:
                self.target_visualizer.set_visibility(True)
                self.ray_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot may be de-initialized, in which case its data is unavailable
        if not self.robot.is_initialized:
            return
        root_pos_w, root_quat_w = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        # current tracking point (step-0 target) is in the root frame -> transform to world
        goal_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, self._command[:, :3])
        self.goal_visualizer.visualize(translations=goal_pos_w)
        if not self.has_look_at:
            return
        # gimbal target -> world, and the aim ray from the actual hand to that target
        target_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, self.look_at_targets[self.traj_id])
        self.target_visualizer.visualize(translations=target_w)
        hand_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ray = target_w - hand_pos_w
        dist = torch.norm(ray, dim=-1, keepdim=True)  # (num_envs, 1)
        scales = torch.cat([torch.ones_like(dist), torch.ones_like(dist), dist], dim=-1)  # stretch local z
        self.ray_visualizer.visualize(
            translations=0.5 * (hand_pos_w + target_w),  # cylinder origin is its centre
            orientations=look_at_quat(ray),
            scales=scales,
        )


@configclass
class TrajectoryCommandCfg(CommandTermCfg):
    class_type: type = TrajectoryCommand
    # bank of closed loops; each is a list of [x, y, z, t] waypoints, e.g. [[x1,y1,z1,t1], ...]
    trajectories: list[list[list[float]]] = MISSING
    bc_type: str = MISSING
    future_length: int = MISSING
    # optional gimbal targets: one [x, y, z] point (root frame) per trajectory that the EE should aim at
    # while tracing it (NOT an orientation). Same length as ``trajectories``. None -> no look-at target is
    # appended to the command and the look-at reward/visualisation must stay disabled.
    look_at_targets: list[list[float]] | None = None