# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import followtrajectory.tasks.manager_based.follow.mdp as mdp
from followtrajectory.tasks.manager_based.follow.follow_env_cfg import FollowEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class FrankaFollowEnvCfg(FollowEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards: track with the panda hand
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_velocity_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_velocity_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]

        # relative joint-position control with EMA smoothing
        self.actions.arm_action = mdp.SmoothedJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], moving_average=0.2
        )


@configclass
class FrankaFollowAbsoluteEnvCfg(FrankaFollowEnvCfg):
    """Absolute joint-position control baseline.

    Identical to :class:`FrankaFollowEnvCfg` except the action parameterisation: the policy
    commands target joint positions *directly* — each action in [-1, 1] is rescaled onto the
    joint's limits, with no incremental accumulation or EMA smoothing. Used as the absolute-control
    arm of the ``action_smoothing`` study, against the relative ``SmoothedJointPositionAction``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, rescale_to_limits=True
        )


@configclass
class FrankaFollowEnvCfg_PLAY(FrankaFollowEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
