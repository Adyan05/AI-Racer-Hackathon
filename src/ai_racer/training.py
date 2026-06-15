from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from .callbacks import FixedSeedEvalCallback, LiveHUDCallback
from .config import save_config
from .envs import make_vec_env


def linear_schedule(initial_value: float):
    return lambda progress_remaining: progress_remaining * initial_value


def create_run_dir(config: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["run"]["output_dir"]) / f"{config['run']['name']}_{timestamp}"
    for child in ("checkpoints", "evaluations", "metrics", "videos", "tensorboard"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def train(
    config: dict[str, Any],
    resume: str | None = None,
    run_dir: str | Path | None = None,
    visualize: bool = False,
) -> Path:
    run_dir = Path(run_dir) if run_dir else create_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    (run_dir / "seeds.json").write_text(json.dumps({"training": config["run"]["seed"], "evaluation": config["eval"]["seeds"]}, indent=2), encoding="utf-8")
    set_random_seed(config["run"]["seed"])
    device = config["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --set train.device=cpu or install a CUDA-enabled PyTorch build.")
    if visualize and config["env"]["n_envs"] != 1:
        config = {**config, "env": {**config["env"], "n_envs": 1}}
        save_config(config, run_dir / "config.yaml")
        print("Visualization enabled: using one training environment in a live window.")
    env = make_vec_env(
        config,
        training=True,
        run_dir=run_dir,
        render_mode="human" if visualize else "rgb_array",
    )
    params = config["train"]
    if resume:
        model = PPO.load(resume, env=env, device=device, tensorboard_log=str(run_dir / "tensorboard"))
        reset_num_timesteps = False
    else:
        model = PPO(
            "CnnPolicy", env, device=device, seed=config["run"]["seed"], verbose=1,
            learning_rate=linear_schedule(float(params["learning_rate"])), n_steps=params["n_steps"],
            batch_size=params["batch_size"], n_epochs=params["n_epochs"], gamma=params["gamma"],
            gae_lambda=params["gae_lambda"], clip_range=params["clip_range"], ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"], max_grad_norm=params["max_grad_norm"],
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        reset_num_timesteps = True
    callback_items = [
        CheckpointCallback(save_freq=max(1, params["checkpoint_freq"] // config["env"]["n_envs"]), save_path=str(run_dir / "checkpoints"), name_prefix="ppo_carracing"),
        FixedSeedEvalCallback(config, run_dir),
    ]
    if visualize:
        callback_items.append(LiveHUDCallback())
    callbacks = CallbackList(callback_items)
    try:
        model.learn(total_timesteps=params["total_timesteps"], callback=callbacks, reset_num_timesteps=reset_num_timesteps, progress_bar=False)
        model.save(run_dir / "final_model")
    finally:
        env.close()
    return run_dir
