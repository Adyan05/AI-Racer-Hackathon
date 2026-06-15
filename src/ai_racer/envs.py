from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage


class GrayResizeObservation(gym.ObservationWrapper):
    """Convert RGB observations to 84x84-style grayscale channel-last images."""

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(0, 255, shape=(size, size, 1), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized[..., None].astype(np.uint8)


class LapInfoWrapper(gym.Wrapper):
    """Expose ordered track progress and enforce configurable road boundaries."""

    def __init__(
        self,
        env: gym.Env,
        terminate_off_track: bool = True,
        off_track_grace_steps: int = 10,
        off_track_penalty: float = -100.0,
    ):
        super().__init__(env)
        self.terminate_off_track = terminate_off_track
        self.off_track_grace_steps = off_track_grace_steps
        self.off_track_penalty = off_track_penalty
        self.off_track_steps = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.off_track_steps = 0
        base = self.env.unwrapped
        centerline = [
            {"index": index, "x": float(tile[2]), "y": float(tile[3])}
            for index, tile in enumerate(base.track)
        ]
        info = dict(info)
        info.update(
            {
                "track_tiles": len(centerline),
                "track_start": centerline[0],
                "track_finish": centerline[-1],
                "track_centerline": centerline,
            }
        )
        return observation, info

    @staticmethod
    def _is_on_track(base: Any) -> bool:
        return any(bool(wheel.tiles) for wheel in base.car.wheels)

    @staticmethod
    def _nearest_track_tile(base: Any) -> int:
        x, y = base.car.hull.position
        points = np.asarray([(tile[2], tile[3]) for tile in base.track], dtype=np.float32)
        return int(np.argmin(np.square(points[:, 0] - x) + np.square(points[:, 1] - y)))

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        visited = int(getattr(base, "tile_visited_count", 0))
        total = len(getattr(base, "track", []))
        progress = visited / total if total else 0.0
        on_track = self._is_on_track(base)
        self.off_track_steps = 0 if on_track else self.off_track_steps + 1
        nearest_tile = self._nearest_track_tile(base) if total else 0
        velocity = base.car.hull.linearVelocity
        speed = float(np.hypot(velocity[0], velocity[1]))
        heading_degrees = float(np.degrees(base.car.hull.angle) % 360.0)
        info = dict(info)
        info.update(
            {
                "tiles_visited": visited,
                "track_tiles": total,
                "lap_progress": progress,
                "current_track_tile": nearest_tile,
                "track_position_percent": nearest_tile / max(1, total - 1),
                "on_track": on_track,
                "off_track_steps": self.off_track_steps,
                "speed": speed,
                "heading_degrees": heading_degrees,
            }
        )
        info["lap_complete"] = bool(terminated and progress >= getattr(base, "lap_complete_percent", 0.95))
        if self.terminate_off_track and self.off_track_steps >= self.off_track_grace_steps:
            reward = self.off_track_penalty
            terminated = True
            info["lap_complete"] = False
            info["reset_reason"] = "off_track"
        elif terminated:
            info["reset_reason"] = "lap_complete" if info["lap_complete"] else "out_of_bounds"
        elif truncated:
            info["reset_reason"] = "time_limit"
        return observation, reward, terminated, truncated, info


class TrainingHUDWrapper(gym.Wrapper):
    """Draw live training telemetry without changing observations or rewards."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.learning_rate = 0.0
        self.episode_reward = 0.0

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        return self.env.reset(**kwargs)

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = float(learning_rate)

    @staticmethod
    def _direction(heading: float) -> str:
        labels = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
        return labels[int((heading + 22.5) // 45) % 8]

    def _draw_hud(self, reward: float, info: dict[str, Any]) -> None:
        base = self.env.unwrapped
        if base.screen is None or base.surf is None:
            return
        risk = "SAFE"
        risk_color = (80, 220, 120)
        if not info.get("on_track", True):
            risk = f"OFF TRACK {info.get('off_track_steps', 0)}"
            risk_color = (255, 90, 80)
        elif info.get("off_track_steps", 0):
            risk = "EDGE"
            risk_color = (255, 200, 70)
        font = pygame.font.Font(pygame.font.get_default_font(), 24)
        heading = float(info.get("heading_degrees", 0.0))
        lines = [
            (f"Speed: {info.get('speed', 0.0):6.2f}", (255, 255, 255)),
            (f"Direction: {self._direction(heading)}  {heading:6.1f} deg", (255, 255, 255)),
            (f"Risk: {risk}", risk_color),
            (f"Reward: {reward:+7.2f}  Episode: {self.episode_reward:+8.2f}", (120, 220, 255)),
            (f"Learning rate: {self.learning_rate:.7f}", (210, 170, 255)),
            (f"Lap progress: {100.0 * info.get('lap_progress', 0.0):5.1f}%", (255, 255, 255)),
        ]
        panel = pygame.Surface((430, 178), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 185))
        base.surf.blit(panel, (12, 12))
        for index, (text, color) in enumerate(lines):
            base.surf.blit(font.render(text, True, color), (24, 22 + index * 25))
        base.screen.fill(0)
        base.screen.blit(base.surf, (0, 0))
        pygame.display.flip()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        self._draw_hud(float(reward), info)
        return observation, reward, terminated, truncated, info


def make_single_env(
    env_config: dict[str, Any], seed: int, monitor_path: str | Path | None = None, render_mode: str = "rgb_array"
) -> gym.Env:
    env = gym.make(
        env_config["id"],
        render_mode=render_mode,
        continuous=True,
        lap_complete_percent=env_config["lap_complete_percent"],
        domain_randomize=env_config["domain_randomize"],
        max_episode_steps=env_config.get("max_episode_steps", 1000),
    )
    env = LapInfoWrapper(
        env,
        terminate_off_track=env_config.get("terminate_off_track", True),
        off_track_grace_steps=env_config.get("off_track_grace_steps", 10),
        off_track_penalty=env_config.get("off_track_penalty", -100.0),
    )
    if render_mode == "human":
        env = TrainingHUDWrapper(env)
    env = GrayResizeObservation(env, env_config["image_size"])
    env = Monitor(
        env,
        filename=str(monitor_path) if monitor_path else None,
        info_keywords=("lap_complete", "lap_progress", "reset_reason"),
    )
    env.action_space.seed(seed)
    return env


def _factory(
    env_config: dict[str, Any], seed: int, monitor_path: Path | None, render_mode: str
) -> Callable[[], gym.Env]:
    def build() -> gym.Env:
        return make_single_env(env_config, seed, monitor_path, render_mode=render_mode)
    return build


def make_vec_env(
    config: dict[str, Any],
    training: bool,
    run_dir: str | Path | None = None,
    seed: int | None = None,
    render_mode: str = "rgb_array",
):
    env_config = config["env"]
    count = int(env_config["n_envs"] if training else 1)
    base_seed = config["run"]["seed"] if seed is None else seed
    monitor_dir = Path(run_dir) / "monitor" if run_dir else None
    if monitor_dir:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    factories = [
        _factory(
            env_config,
            base_seed + rank,
            monitor_dir / f"env_{rank}" if monitor_dir else None,
            render_mode,
        )
        for rank in range(count)
    ]
    if training and count > 1:
        vec_env = SubprocVecEnv(factories, start_method="spawn")
    else:
        vec_env = DummyVecEnv(factories)
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=env_config["frame_stack"], channels_order="first")
    vec_env.seed(base_seed)
    return vec_env
