from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .envs import make_vec_env


def record_episode(model, config: dict[str, Any], output: str | Path, seed: int, fps: int = 50) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(config, training=False, seed=seed)
    observation = env.reset()
    frame = env.get_images()[0]
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total_reward, length, completed = 0.0, 0, False
    done = np.array([False])
    state = None
    episode_start = np.ones((1,), dtype=bool)
    try:
        while not done[0]:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            action, state = model.predict(observation, state=state, episode_start=episode_start, deterministic=True)
            observation, reward, done, infos = env.step(action)
            episode_start = done
            total_reward += float(reward[0])
            length += 1
            completed = bool(infos[0].get("lap_complete", completed))
            frame = env.get_images()[0]
    finally:
        writer.release()
        env.close()
    return {"seed": seed, "reward": total_reward, "length": length, "completed": completed, "video": str(output)}
