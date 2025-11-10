from dataclasses import dataclass
from typing import Optional

@dataclass
class RLConfig:
    learning_rate: float = 1e-6
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    log_with: str = "wandb"
    tracker_project_name: str = "phi4-rl-code"
    num_train_epochs: int = 1
    max_steps: int = 10000
    save_freq: int = 500
    eval_freq: int = 100
    output_dir: str = "checkpoints/phi4_rl"
    logging_steps: int = 10
    warmup_steps: int = 100
    max_new_tokens: int = 2000
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    response_length: int = 2000
    kl_penalty: str = "kl"
    init_kl_coef: float = 0.1
    adap_kl_ctrl: bool = True
    target_kl: float = 6.0

@dataclass
class PPOConfig(RLConfig):
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    whiten_rewards: bool = True
    gamma: float = 1.0
    lam: float = 0.95

@dataclass
class GRPOConfig(RLConfig):
    num_sample_generations: int = 8
    group_size: int = 8
    beta: float = 0.01
    grpo_epochs: int = 1
    whiten_rewards: bool = True

def get_config(algorithm: str = "ppo"):
    if algorithm == "ppo":
        return PPOConfig()
    elif algorithm == "grpo":
        return GRPOConfig()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'ppo' or 'grpo'")

