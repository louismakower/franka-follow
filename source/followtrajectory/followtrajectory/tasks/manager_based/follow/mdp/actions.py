from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import saturate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SmoothedJointPositionAction(ActionTerm):
    """Incremental joint position action with exponential smoothing.

    The network outputs a delta in [-1, 1] (scaled by ``scale``).  Each step:

        target = clamp(prev_target + moving_average * scale * action, lower, upper)

    ``prev_target`` is reset to the actual joint positions after each episode
    reset, so the first action in a new episode is anchored to the reset pose.
    """

    cfg: SmoothedJointPositionActionCfg

    def __init__(self, cfg: SmoothedJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Initialise prev_targets to default joint positions
        self._prev_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()

        targets = self._prev_targets + self.cfg.moving_average * actions

        lower = self._asset.data.joint_pos_limits[:, self._joint_ids, 0]
        upper = self._asset.data.joint_pos_limits[:, self._joint_ids, 1]
        self._processed_actions = saturate(
            targets,
            lower,
            upper,
        )
        self._prev_targets = self._processed_actions.clone()

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor):
        # Anchor prev_targets to the actual post-reset joint positions so the
        # first delta in the new episode is relative to where the hand landed.
        self._prev_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class SmoothedJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`SmoothedJointPositionAction`."""

    class_type: type = SmoothedJointPositionAction

    joint_names: list[str] = MISSING
    """Regex patterns or exact names of the joints to control."""

    moving_average: float = MISSING
    """Fraction of the scaled delta applied each step (0 = frozen, 1 = full step)."""
