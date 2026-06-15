from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .envs import make_vec_env


def evaluate_model(model, config: dict[str, Any], seeds: list[int] | None = None) -> dict[str, Any]:
    seeds = list(seeds or config["eval"]["seeds"])
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env = make_vec_env(config, training=False, seed=seed)
        observation = env.reset()
        total_reward = 0.0
        length = 0
        complete = False
        progress = 0.0
        done = np.array([False])
        while not done[0]:
            action, _ = model.predict(observation, deterministic=config["eval"].get("deterministic", True))
            observation, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            length += 1
            progress = float(infos[0].get("lap_progress", progress))
            complete = bool(infos[0].get("lap_complete", complete))
        env.close()
        episodes.append({"seed": seed, "reward": total_reward, "length": length, "completed": complete, "lap_progress": progress})
    rewards = [episode["reward"] for episode in episodes]
    lengths = [episode["length"] for episode in episodes]
    return {
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(median(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "completion_rate": float(np.mean([episode["completed"] for episode in episodes])),
        "episodes": episodes,
    }


def save_evaluation(result: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

