from louis_rl.algos.ppo import PPORunnerCfg
from isaaclab.utils import configclass

@configclass
class FrankaReachPPOCfg(PPORunnerCfg):
    experiment_name: str = "franka_follow_ppo"
    
    num_iterations: int = 1000
    steps_per_rollout: int = 32

    num_policy_grad_steps: int = 8
    num_v_grad_steps: int = 8

    policy_lr: float = 0.001
    v_lr: float = 0.001

    policy_hidden_dims = [256, 256]
    v_hidden_dims = [256, 256]

    gamma: float = 0.99
    eps: float = 0.2

    save_interval: int = 50

    intrinsic = None
    intrinsic_V_hidden_layers = None
    intrinsic_gamma = None
    intrinsic_v_grad_steps = None
    intrinsic_V_lr = None
    intrinsic_weight = None