from louis_rl.algos.sac import SACRunnerCfg
from isaaclab.utils import configclass
# from ..mdp.her_cfg import FrankaReachHerCfg


@configclass
class FrankaReachSACCfg(SACRunnerCfg):
    experiment_name = "franka_follow_sac"

    gamma = 0.99
    alpha_init = 0.1
    alpha_lr = 3e-4
    target_entropy = "auto"

    replay_buffer_size = 5_000_000
    warmup_transitions = 20_000

    q_hidden_dims = [256, 256]
    q_learning_rate = 3e-4
    q_tau = 0.005
    q_grad_clip_norm = 10.0

    policy_hidden_dims = [256, 256]
    logstd_min = -5.0
    logstd_max = 2.0
    policy_learning_rate = 3e-4

    reward_scaling = False
    reward_G_max = 5.0
    reward_clip = 0.0

    max_steps = 7500
    steps_per_iter = 1
    num_train_updates = 16
    batch_size = 1024

    save_interval = 600

    # her_cfg = FrankaReachHerCfg(mode="future", success_threshold=0.03)

    # intrinsic rewards
    use_intrinsic = False
    intrinsic_cfg = None
    intrinsic_critic_hidden_layers = None
    intrinsic_critic_lr: float = None
    intrinsic_rew_weight: float = None
    intrinsic_rew_clip: float = None  # 0.0 = disabled
    intrinsic_critic_tau: float = None
    intrinsic_gamma: float = None
    intrinsic_G_max: float = None
