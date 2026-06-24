# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply

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
    # the velocity block starts halfway through the command ([positions | velocities])
    vel_start = command.shape[1] // 2
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
    # the velocity block starts halfway through the command ([positions | velocities])
    vel_start = command.shape[1] // 2
    des_vel_b = command[:, vel_start : vel_start + 3]
    # velocity is a free vector: rotate (no translation) from the root frame into the world frame
    des_vel_w = quat_apply(asset.data.root_quat_w, des_vel_b)
    curr_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]  # type: ignore
    error = torch.norm(curr_vel_w - des_vel_w, dim=1)
    return 1 - torch.tanh(error / std)
