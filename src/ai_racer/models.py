from __future__ import annotations

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


def algorithm_class(config):
    return RecurrentPPO if config["train"].get("algorithm", "ppo") == "recurrent_ppo" else PPO


def policy_name(config):
    return "MultiInputLstmPolicy" if config["train"].get("algorithm") == "recurrent_ppo" else "MultiInputPolicy"


def load_model(path, config, **kwargs):
    return algorithm_class(config).load(path, **kwargs)
