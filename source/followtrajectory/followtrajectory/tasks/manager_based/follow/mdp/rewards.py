# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude

from .commands import look_at_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the desired position using L2-norm.

    The command stores ``future_length`` future targets as ``[positions | velocities]``; the first
    block (``command[:, :3]``) is the desired position for the current step, expressed in the asset's
    root frame. It is transformed into the world frame and compared against the current position of the
    tracked body. The error is the L2-norm of the difference.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the desired position using the tanh kernel.

    Same desired/current positions as :func:`position_command_error`, mapped through a tanh kernel so
    the reward saturates to 1 as the body reaches the target.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def velocity_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the desired linear velocity using L2-norm.

    The command stores ``future_length`` future targets as ``[positions | velocities]``; the first
    velocity block is the desired linear velocity for the current step, expressed in the asset's root
    frame. It is rotated into the world frame and compared against the current linear velocity of the
    tracked body. The error is the L2-norm of the difference.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # command layout is [positions | velocities | (optional look-at target)]; the velocity block starts
    # after the future_length position triples (can't assume "halfway" once the target is appended)
    vel_start = 3 * env.command_manager.get_term(command_name).cfg.future_length  # type: ignore[attr-defined]
    des_vel_b = command[:, vel_start : vel_start + 3]
    # velocity is a free vector: rotate (no translation) from the root frame into the world frame
    des_vel_w = quat_apply(asset.data.root_quat_w, des_vel_b)
    curr_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_vel_w - des_vel_w, dim=1)


def velocity_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the desired linear velocity using the tanh kernel.

    Same desired/current velocities as :func:`velocity_command_error`, mapped through a tanh kernel so
    the reward saturates to 1 as the body matches the target velocity.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # command layout is [positions | velocities | (optional look-at target)]; the velocity block starts
    # after the future_length position triples (can't assume "halfway" once the target is appended)
    vel_start = 3 * env.command_manager.get_term(command_name).cfg.future_length  # type: ignore[attr-defined]
    des_vel_b = command[:, vel_start : vel_start + 3]
    # velocity is a free vector: rotate (no translation) from the root frame into the world frame
    des_vel_w = quat_apply(asset.data.root_quat_w, des_vel_b)
    curr_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]  # type: ignore
    error = torch.norm(curr_vel_w - des_vel_w, dim=1)
    return 1 - torch.tanh(error / std)


def look_at_orientation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize how far the end-effector is from pointing its +z axis at the command's gimbal target.

    The command's trailing three entries are an ``[x, y, z]`` target point in the asset's root frame (set
    via ``TrajectoryCommandCfg.look_at_targets``). The target is transformed to the world frame and a desired
    "look-at" orientation is built from the **actual** body position: local +z aimed at the target, with roll
    pinned by keeping local +y near world up. The error is the shortest-path rotation between the current and
    desired orientations (so aim and roll are penalized together).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # guard the optional contract: this reward reads the trailing target block, which only exists when
    # look_at_targets is set. Without it command[:, -3:] would silently be the last velocity triple.
    if not getattr(env.command_manager.get_term(command_name), "has_look_at", False):
        raise ValueError(
            f"look_at_orientation_error needs command '{command_name}' to have look_at_targets set "
            "(TrajectoryCommandCfg.look_at_targets); disable this reward if you run without targets."
        )
    command = env.command_manager.get_command(command_name)
    # gimbal target (last 3 entries of the command) -> world frame
    target_b = command[:, -3:]
    target_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    des_quat_w = look_at_quat(target_w - curr_pos_w)
    return quat_error_magnitude(curr_quat_w, des_quat_w)
