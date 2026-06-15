from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage


class FixedTrackWrapper(gym.Wrapper):
    """Regenerate the same circuit on every reset for circuit learning."""

    def __init__(self, env: gym.Env, track_seed: int):
        super().__init__(env)
        self.track_seed = track_seed

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=self.track_seed, options=options)


class CircuitCurriculumWrapper(gym.Wrapper):
    """Persist circuit mastery and focus training near the learning frontier."""

    def __init__(
        self,
        env: gym.Env,
        state_path: str | Path,
        segments: int = 20,
        mastery_visits: int = 3,
        review_interval: int = 4,
        frontier_unlock_segment: int | None = None,
    ):
        super().__init__(env)
        self.state_path = Path(state_path)
        self.segments = segments
        self.mastery_visits = mastery_visits
        self.review_interval = review_interval
        self.frontier_unlock_segment = (
            segments // 2 if frontier_unlock_segment is None else int(frontier_unlock_segment)
        )
        self.start_segment = 0
        self.episode_max_segment = 0
        self.current_segment = 0

    def _default_state(self) -> dict[str, Any]:
        return {
            "episodes": 0,
            "mastered_segment": 0,
            "segment_visits": {},
            "segment_failures": {},
            "segment_actions": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        return {**self._default_state(), **json.loads(self.state_path.read_text(encoding="utf-8"))}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _segment_for_tile(self, tile: int, total_tiles: int) -> int:
        return min(self.segments - 1, int(tile * self.segments / max(total_tiles, 1)))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        state = self._load_state()
        state["episodes"] += 1
        mastered = int(state["mastered_segment"])
        review_episode = state["episodes"] % self.review_interval == 0
        if mastered < self.frontier_unlock_segment:
            self.start_segment = 0
        else:
            self.start_segment = 0 if review_episode else max(0, mastered - 1)
        self.episode_max_segment = self.start_segment
        self.current_segment = self.start_segment
        base = self.env.unwrapped
        if self.start_segment > 0:
            start_tile = min(
                len(base.track) - 1,
                int(self.start_segment * len(base.track) / self.segments),
            )
            base.car.destroy()
            base.car = Car(base.world, *base.track[start_tile][1:4])
            base.state = base._render("state_pixels")
            observation = base.state
        self._save_state(state)
        info = dict(info)
        info.update(
            {
                "curriculum_start_segment": self.start_segment,
                "mastered_segment": mastered,
                "curriculum_review_episode": review_episode,
                "frontier_unlock_segment": self.frontier_unlock_segment,
            }
        )
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        state = self._load_state()
        total_tiles = int(info.get("track_tiles", 1))
        segment = self._segment_for_tile(int(info.get("current_track_tile", 0)), total_tiles)
        self.current_segment = segment
        if segment > self.episode_max_segment:
            visits = state["segment_visits"]
            for reached in range(self.episode_max_segment + 1, segment + 1):
                key = str(reached)
                visits[key] = int(visits.get(key, 0)) + 1
            self.episode_max_segment = segment
            next_segment = int(state["mastered_segment"]) + 1
            if next_segment < self.segments and int(visits.get(str(next_segment), 0)) >= self.mastery_visits:
                state["mastered_segment"] = next_segment
            self._save_state(state)
        if (terminated or truncated) and info.get("reset_reason") == "off_track":
            failures = state["segment_failures"]
            key = str(segment)
            failures[key] = int(failures.get(key, 0)) + 1
            self._save_state(state)
        info = dict(info)
        info.update(
            {
                "current_segment": segment,
                "mastered_segment": int(state["mastered_segment"]),
                "curriculum_segments": self.segments,
                "curriculum_start_segment": self.start_segment,
            }
        )
        return observation, reward, terminated, truncated, info

    def action_success_rate(self, segment: int, action_name: str) -> tuple[float, int]:
        state = self._load_state()
        stats = state["segment_actions"].get(str(segment), {}).get(action_name, {})
        successes = int(stats.get("successes", 0))
        failures = int(stats.get("failures", 0))
        attempts = successes + failures
        return (successes + 1.0) / (attempts + 2.0), attempts

    def record_action_outcome(self, segment: int, action_name: str, success: bool) -> None:
        state = self._load_state()
        segment_actions = state["segment_actions"].setdefault(str(segment), {})
        stats = segment_actions.setdefault(action_name, {"successes": 0, "failures": 0})
        key = "successes" if success else "failures"
        stats[key] = int(stats.get(key, 0)) + 1
        self._save_state(state)


class GuidedObservation(gym.ObservationWrapper):
    """Combine the camera image with explainable track guidance values."""

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(size, size, 1), dtype=np.uint8),
                "guidance": spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
            }
        )

    def observation(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return {
            "image": resized[..., None].astype(np.uint8),
            "guidance": np.asarray(self.env.get_wrapper_attr("guidance_vector"), dtype=np.float32),
        }


class LapInfoWrapper(gym.Wrapper):
    """Expose ordered track progress and enforce configurable road boundaries."""

    def __init__(
        self,
        env: gym.Env,
        terminate_off_track: bool = True,
        max_vehicle_off_track_fraction: float = 0.5,
        off_track_penalty: float = -100.0,
        probe_dead_zone: float = 0.02,
        probe_turn_threshold: float = 0.05,
        probe_max_distance: float = 16.0,
    ):
        super().__init__(env)
        self.terminate_off_track = terminate_off_track
        self.max_vehicle_off_track_fraction = max_vehicle_off_track_fraction
        self.off_track_penalty = off_track_penalty
        self.probe_dead_zone = probe_dead_zone
        self.probe_turn_threshold = probe_turn_threshold
        self.off_track_steps = 0
        self._hull_sample_points = np.empty((0, 2), dtype=np.float32)
        self._road_polygons: list[np.ndarray] = []
        self.guidance_vector = np.zeros(7, dtype=np.float32)
        self.probe_values = (0.0, 0.0)
        self.probe_distances = (0.0, 0.0)
        self.probe_max_distance = probe_max_distance

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.off_track_steps = 0
        base = self.env.unwrapped
        self._hull_sample_points = self._sample_hull(base)
        self._road_polygons = [
            np.asarray(body.fixtures[0].shape.vertices, dtype=np.float32)
            for body in base.road
        ]
        self._update_guidance(base, 0)
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
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    @classmethod
    def _sample_hull(cls, base: Any, spacing: float = 0.22) -> np.ndarray:
        polygons = [
            np.asarray(fixture.shape.vertices, dtype=np.float32)
            for fixture in base.car.hull.fixtures
        ]
        vertices = np.concatenate(polygons)
        xs = np.arange(vertices[:, 0].min(), vertices[:, 0].max() + spacing, spacing)
        ys = np.arange(vertices[:, 1].min(), vertices[:, 1].max() + spacing, spacing)
        samples = [
            (x, y)
            for x in xs
            for y in ys
            if any(cls._point_in_polygon(np.asarray((x, y)), polygon) for polygon in polygons)
        ]
        return np.asarray(samples, dtype=np.float32)

    def _vehicle_off_track_fraction(self, base: Any, nearest_tile: int) -> float:
        if not len(self._hull_sample_points) or not self._road_polygons:
            return 1.0
        angle = float(base.car.hull.angle)
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
            dtype=np.float32,
        )
        position = np.asarray(base.car.hull.position, dtype=np.float32)
        world_points = self._hull_sample_points @ rotation.T + position
        total_tiles = len(self._road_polygons)
        candidate_indices = [(nearest_tile + offset) % total_tiles for offset in range(-8, 9)]
        candidate_polygons = [self._road_polygons[index] for index in candidate_indices]
        inside_count = sum(
            any(self._point_in_polygon(point, polygon) for polygon in candidate_polygons)
            for point in world_points
        )
        return 1.0 - inside_count / len(world_points)

    def _road_probe(self, base: Any, angle_offset: float, nearest_tile: int) -> tuple[float, float]:
        angle = float(base.car.hull.angle)
        forward = np.asarray((-np.sin(angle), np.cos(angle)), dtype=np.float32)
        side = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float32)
        direction = forward * np.cos(angle_offset) + side * np.sin(angle_offset)
        lateral_offset = -0.75 if angle_offset < 0.0 else 0.75
        origin = np.asarray(base.car.hull.position, dtype=np.float32) + forward * 1.1 + side * lateral_offset
        polygons = self._road_polygons

        def on_road(distance: float) -> bool:
            point = origin + direction * distance
            return any(self._point_in_polygon(point, polygon) for polygon in polygons)

        if not on_road(0.0):
            return 0.0, 0.0
        step = 0.2
        last_inside = 0.0
        first_outside = self.probe_max_distance
        for distance in np.arange(step, self.probe_max_distance + step, step):
            if not on_road(float(distance)):
                first_outside = float(distance)
                break
            last_inside = min(float(distance), self.probe_max_distance)
        else:
            return 1.0, self.probe_max_distance

        for _ in range(8):
            midpoint = 0.5 * (last_inside + first_outside)
            if on_road(midpoint):
                last_inside = midpoint
            else:
                first_outside = midpoint
        distance = min(last_inside, self.probe_max_distance)
        return distance / self.probe_max_distance, distance

    def _update_guidance(self, base: Any, nearest_tile: int) -> None:
        left, left_distance = self._road_probe(base, -np.deg2rad(28.0), nearest_tile)
        right, right_distance = self._road_probe(base, np.deg2rad(28.0), nearest_tile)
        current = np.asarray(base.car.hull.position, dtype=np.float32)
        lookahead = base.track[(nearest_tile + 10) % len(base.track)]
        target = np.asarray((lookahead[2], lookahead[3]), dtype=np.float32) - current
        target /= max(float(np.linalg.norm(target)), 1e-6)
        angle = float(base.car.hull.angle)
        forward = np.asarray((-np.sin(angle), np.cos(angle)), dtype=np.float32)
        heading_error = np.arctan2(forward[0] * target[1] - forward[1] * target[0], np.dot(forward, target)) / np.pi
        center = np.asarray((base.track[nearest_tile][2], base.track[nearest_tile][3]), dtype=np.float32)
        tangent_tile = base.track[(nearest_tile + 1) % len(base.track)]
        tangent = np.asarray((tangent_tile[2], tangent_tile[3]), dtype=np.float32) - center
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        lateral_error = np.clip((tangent[0] * (current - center)[1] - tangent[1] * (current - center)[0]) / 7.0, -1.0, 1.0)
        speed = np.clip(float(np.linalg.norm(base.car.hull.linearVelocity)) / 50.0, 0.0, 1.0)
        position = nearest_tile / max(1, len(base.track) - 1)
        self.probe_values = (float(left), float(right))
        self.probe_distances = (float(left_distance), float(right_distance))
        self.guidance_vector = np.asarray(
            (left, right, left - right, heading_error, lateral_error, speed, 2.0 * position - 1.0),
            dtype=np.float32,
        )

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
        nearest_tile = self._nearest_track_tile(base) if total else 0
        self._update_guidance(base, nearest_tile)
        left_probe, right_probe = self.probe_values
        left_distance, right_distance = self.probe_distances
        off_track_fraction = self._vehicle_off_track_fraction(base, nearest_tile)
        beyond_tolerance = off_track_fraction >= self.max_vehicle_off_track_fraction
        on_track = not beyond_tolerance
        self.off_track_steps = self.off_track_steps + 1 if beyond_tolerance else 0
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
                "vehicle_off_track_fraction": off_track_fraction,
                "vehicle_off_track_percent": 100.0 * off_track_fraction,
                "off_track_tolerance_percent": 100.0 * self.max_vehicle_off_track_fraction,
                "left_probe": left_probe,
                "right_probe": right_probe,
                "left_probe_distance": left_distance,
                "right_probe_distance": right_distance,
                "probe_max_distance": self.probe_max_distance,
                "probe_decision": self._probe_decision(left_distance, right_distance),
                "heading_error": float(self.guidance_vector[3]),
                "lateral_error": float(self.guidance_vector[4]),
                "speed": speed,
                "heading_degrees": heading_degrees,
            }
        )
        info["lap_complete"] = bool(terminated and progress >= getattr(base, "lap_complete_percent", 0.95))
        if self.terminate_off_track and beyond_tolerance:
            reward = self.off_track_penalty
            terminated = True
            info["lap_complete"] = False
            info["reset_reason"] = "off_track"
        elif terminated:
            info["reset_reason"] = "lap_complete" if info["lap_complete"] else "out_of_bounds"
        elif truncated:
            info["reset_reason"] = "time_limit"
        return observation, reward, terminated, truncated, info

    def _probe_decision(self, left: float, right: float) -> str:
        relative_difference = abs(left - right) / max(left, right, 1e-6)
        if relative_difference <= self.probe_dead_zone:
            return "straight"
        if relative_difference < self.probe_turn_threshold:
            return "correct"
        return "left" if left > right else "right"


class ClearanceSteeringWrapper(gym.Wrapper):
    """Use learned straight, soft-turn, and hard-turn actions per segment."""

    def __init__(
        self,
        env: gym.Env,
        strength: float = 0.7,
        straight_threshold: float = 0.02,
        soft_turn_threshold: float = 0.10,
        hard_turn_threshold: float = 0.20,
        straight_gas: float = 0.8,
        soft_steering: float = 0.22,
        hard_steering: float = 0.65,
        exploration_rate: float = 0.1,
        straight_target_speed: float = 12.0,
        soft_turn_target_speed: float = 9.0,
        hard_turn_target_speed: float = 6.5,
        max_gas: float = 0.35,
        max_brake: float = 0.6,
        absolute_max_speed: float = 20.0,
    ):
        super().__init__(env)
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.straight_threshold = max(float(straight_threshold), 0.0)
        self.soft_turn_threshold = max(float(soft_turn_threshold), self.straight_threshold)
        self.hard_turn_threshold = max(float(hard_turn_threshold), self.soft_turn_threshold)
        self.straight_gas = float(np.clip(straight_gas, 0.0, 1.0))
        self.soft_steering = float(np.clip(soft_steering, 0.0, 1.0))
        self.hard_steering = float(np.clip(hard_steering, self.soft_steering, 1.0))
        self.exploration_rate = float(np.clip(exploration_rate, 0.0, 1.0))
        self.straight_target_speed = straight_target_speed
        self.soft_turn_target_speed = soft_turn_target_speed
        self.hard_turn_target_speed = hard_turn_target_speed
        self.max_gas = float(np.clip(max_gas, 0.0, 1.0))
        self.max_brake = float(np.clip(max_brake, 0.0, 1.0))
        self.absolute_max_speed = max(float(absolute_max_speed), 0.0)
        self.manual_max_speed = self.absolute_max_speed
        self.previous_segment = 0
        self.previous_action_name = "straight"

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_segment = int(info.get("curriculum_start_segment", 0))
        self.previous_action_name = "straight"
        return observation, info

    def set_manual_max_speed(self, value: float) -> None:
        self.manual_max_speed = float(np.clip(value, 0.0, 100.0))

    def get_manual_max_speed(self) -> float:
        return self.manual_max_speed

    def _learned_turn(self, segment: int, direction: float, difference: float) -> tuple[float, str]:
        side = "right" if direction > 0.0 else "left"
        soft_name = f"soft_{side}"
        hard_name = f"hard_{side}"
        try:
            success_rate = self.env.get_wrapper_attr("action_success_rate")
            soft_rate, soft_attempts = success_rate(segment, soft_name)
            hard_rate, hard_attempts = success_rate(segment, hard_name)
        except AttributeError:
            soft_rate, soft_attempts = 0.5, 0
            hard_rate, hard_attempts = 0.5, 0
        use_hard = difference > self.hard_turn_threshold
        if not use_hard and soft_attempts + hard_attempts >= 4 and np.random.random() < self.exploration_rate:
            use_hard = hard_rate > soft_rate and difference >= self.hard_turn_threshold
        magnitude = self.hard_steering if use_hard else self.soft_steering
        return direction * magnitude, hard_name if use_hard else soft_name

    def _clearance_target(self) -> tuple[float, str, float]:
        left, right = self.env.get_wrapper_attr("probe_distances")
        largest = max(float(left), float(right), 1e-6)
        difference = float(right - left)
        relative_difference = abs(difference) / largest
        if relative_difference <= self.straight_threshold:
            return 0.0, "straight", relative_difference
        direction = 1.0 if difference > 0.0 else -1.0
        if relative_difference < self.soft_turn_threshold:
            correction = relative_difference / max(self.soft_turn_threshold, 1e-6)
            return direction * self.soft_steering * 0.5 * correction, f"soft_{'right' if direction > 0.0 else 'left'}", relative_difference
        try:
            segment = int(self.env.get_wrapper_attr("current_segment"))
        except AttributeError:
            segment = 0
        target, action_name = self._learned_turn(segment, direction, relative_difference)
        return target, action_name, relative_difference

    def _speed_control(self, action_name: str) -> tuple[float, float, float, float]:
        speed = float(np.linalg.norm(self.env.unwrapped.car.hull.linearVelocity))
        active_max_speed = min(self.absolute_max_speed, self.manual_max_speed)
        if speed >= active_max_speed:
            overspeed = speed - active_max_speed
            brake = min(self.max_brake, 0.25 + overspeed / max(active_max_speed, 1e-6))
            return 0.0, brake, speed, min(active_max_speed, self.straight_target_speed)
        if action_name.startswith("hard_"):
            target_speed = self.hard_turn_target_speed
        elif action_name.startswith("soft_"):
            target_speed = self.soft_turn_target_speed
        else:
            target_speed = self.straight_target_speed
        target_speed = min(target_speed, active_max_speed)
        error = target_speed - speed
        if error > 0.0:
            gas = min(self.max_gas, max(0.08, self.max_gas * error / max(target_speed, 1e-6)))
            brake = 0.0
        else:
            gas = 0.0
            brake = min(self.max_brake, self.max_brake * (-error) / max(target_speed, 1e-6))
        return gas, brake, speed, target_speed

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32).copy()
        applied = requested.copy()
        target, decision, relative_difference = self._clearance_target()
        applied[0] = np.clip(
            (1.0 - self.strength) * requested[0] + self.strength * target,
            -1.0,
            1.0,
        )
        applied[1], applied[2], speed_before_action, target_speed = self._speed_control(decision)
        observation, reward, terminated, truncated, info = self.env.step(applied)
        current_segment = int(info.get("current_segment", self.previous_segment))
        try:
            record_outcome = self.env.get_wrapper_attr("record_action_outcome")
            success_rate_for = self.env.get_wrapper_attr("action_success_rate")
        except AttributeError:
            record_outcome = None
            success_rate_for = None
        if record_outcome is not None and current_segment > self.previous_segment:
            record_outcome(self.previous_segment, self.previous_action_name, True)
        if record_outcome is not None and (terminated or truncated) and info.get("reset_reason") == "off_track":
            record_outcome(current_segment, decision, False)
        self.previous_segment = current_segment
        self.previous_action_name = decision
        success_rate, attempts = success_rate_for(current_segment, decision) if success_rate_for else (0.5, 0)
        info = dict(info)
        info.update(
            {
                "clearance_decision": decision,
                "clearance_steering_target": target,
                "policy_steering": float(requested[0]),
                "applied_steering": float(applied[0]),
                "clearance_steering_strength": self.strength,
                "clearance_difference_percent": 100.0 * relative_difference,
                "policy_gas": float(requested[1]),
                "applied_gas": float(applied[1]),
                "applied_brake": float(applied[2]),
                "speed_before_action": speed_before_action,
                "target_speed": target_speed,
                "goal_priority": "finish_track",
                "absolute_max_speed": self.absolute_max_speed,
                "manual_max_speed": self.manual_max_speed,
                "steering_action": decision,
                "segment_action_success_rate": success_rate,
                "segment_action_attempts": attempts,
            }
        )
        return observation, reward, terminated, truncated, info


class TrainingHUDWrapper(gym.Wrapper):
    """Draw live training telemetry without changing observations or rewards."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.learning_rate = 0.0
        self.episode_reward = 0.0
        self.slider_dragging = False
        self.slider_rect = pygame.Rect(0, 0, 240, 14)

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        observation, info = self.env.reset(**kwargs)
        self._draw_hud(0.0, info)
        return observation, info

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = float(learning_rate)

    @staticmethod
    def _slider_value_from_mouse(mouse_x: int, slider_rect: pygame.Rect) -> float:
        relative = np.clip((mouse_x - slider_rect.left) / max(slider_rect.width, 1), 0.0, 1.0)
        return float(relative * 100.0)

    @staticmethod
    def _slider_knob_x(value: float, slider_rect: pygame.Rect) -> int:
        normalized = np.clip(value / 100.0, 0.0, 1.0)
        return int(round(slider_rect.left + normalized * slider_rect.width))

    def _process_slider_events(self) -> None:
        clearance = self.env.get_wrapper_attr("set_manual_max_speed")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                continue
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.slider_rect.collidepoint(event.pos):
                    self.slider_dragging = True
                    clearance(self._slider_value_from_mouse(event.pos[0], self.slider_rect))
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.slider_dragging = False
            elif event.type == pygame.MOUSEMOTION and self.slider_dragging:
                clearance(self._slider_value_from_mouse(event.pos[0], self.slider_rect))

    @staticmethod
    def _direction(heading: float) -> str:
        labels = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
        return labels[int((heading + 22.5) // 45) % 8]

    def _draw_hud(self, reward: float, info: dict[str, Any]) -> None:
        base = self.env.unwrapped
        if base.screen is None or base.surf is None:
            return
        self._process_slider_events()
        risk = "SAFE"
        risk_color = (80, 220, 120)
        if not info.get("on_track", True):
            risk = f"FAIL {info.get('vehicle_off_track_percent', 0.0):.0f}% OUT"
            risk_color = (255, 90, 80)
        elif info.get("vehicle_off_track_fraction", 0.0) > 0.0:
            risk = f"EDGE {info.get('vehicle_off_track_percent', 0.0):.0f}% OUT"
            risk_color = (255, 200, 70)
        font = pygame.font.Font(pygame.font.get_default_font(), 24)
        heading = float(info.get("heading_degrees", 0.0))
        lines = [
            ("GOAL 1: FINISH TRACK", (100, 255, 120)),
            (f"Speed: {info.get('speed', 0.0):6.2f}", (255, 255, 255)),
            (f"Target speed: {info.get('target_speed', 0.0):5.1f}", (255, 255, 255)),
            (f"Max speed slider: {info.get('manual_max_speed', 0.0):5.1f} / 100", (255, 255, 255)),
            (f"Direction: {self._direction(heading)}  {heading:6.1f} deg", (255, 255, 255)),
            (f"Risk: {risk}", risk_color),
            (f"Reward: {reward:+7.2f}  Episode: {self.episode_reward:+8.2f}", (120, 220, 255)),
            (f"Learning rate: {self.learning_rate:.7f}", (210, 170, 255)),
            (f"Lap progress: {100.0 * info.get('lap_progress', 0.0):5.1f}%", (255, 255, 255)),
            (f"Rays L:{info.get('left_probe_distance', 0.0):.1f}m R:{info.get('right_probe_distance', 0.0):.1f}m -> {info.get('probe_decision', 'straight').upper()}", (255, 230, 80)),
            (f"Steer policy:{info.get('policy_steering', 0.0):+.2f} applied:{info.get('applied_steering', 0.0):+.2f}", (120, 255, 180)),
            (f"Action:{info.get('steering_action', 'straight').upper()} diff:{info.get('clearance_difference_percent', 0.0):.1f}%", (120, 255, 180)),
            (f"Gas:{info.get('applied_gas', 0.0):.2f} brake:{info.get('applied_brake', 0.0):.2f}", (120, 255, 180)),
            (f"Curriculum:{info.get('current_segment', 0) + 1}/{info.get('curriculum_segments', 1)} mastered:{info.get('mastered_segment', 0) + 1}", (255, 180, 120)),
            (f"Segment action score:{100.0 * info.get('segment_action_success_rate', 0.5):.0f}% ({info.get('segment_action_attempts', 0)} tries)", (255, 180, 120)),
        ]
        panel = pygame.Surface((570, 380), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 245))
        base.surf.blit(panel, (12, 12))
        for index, (text, color) in enumerate(lines):
            base.surf.blit(font.render(text, True, color), (24, 22 + index * 25))
        self.slider_rect = pygame.Rect(base.surf.get_width() - 290, 28, 240, 14)
        pygame.draw.rect(base.surf, (230, 230, 230), self.slider_rect, border_radius=7)
        fill_width = max(1, int(self.slider_rect.width * info.get("manual_max_speed", 0.0) / 100.0))
        pygame.draw.rect(
            base.surf,
            (100, 220, 255),
            pygame.Rect(self.slider_rect.left, self.slider_rect.top, fill_width, self.slider_rect.height),
            border_radius=7,
        )
        knob_x = self._slider_knob_x(float(info.get("manual_max_speed", 0.0)), self.slider_rect)
        pygame.draw.circle(base.surf, (255, 180, 80), (knob_x, self.slider_rect.centery), 11)
        base.surf.blit(font.render("Max speed", True, (255, 255, 255)), (self.slider_rect.left, self.slider_rect.top - 28))
        car_anchor = np.asarray((base.surf.get_width() // 2, 3 * base.surf.get_height() // 4 - 28), dtype=np.float32)
        zoom = 0.1 * 6.0 * max(1.0 - base.t, 0.0) + 2.7 * 6.0 * min(base.t, 1.0)
        probe_angle = np.deg2rad(28.0)
        for side_sign, key in ((-1.0, "left_probe_distance"), (1.0, "right_probe_distance")):
            start = car_anchor + np.asarray((side_sign * 12.0, 0.0), dtype=np.float32)
            direction = np.asarray((side_sign * np.sin(probe_angle), -np.cos(probe_angle)), dtype=np.float32)
            endpoint = start + direction * float(info.get(key, 0.0)) * zoom
            pygame.draw.line(base.surf, (0, 0, 0), start.astype(int), endpoint.astype(int), 7)
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
    if env_config.get("fixed_track_seed") is not None:
        env = FixedTrackWrapper(env, int(env_config["fixed_track_seed"]))
    env = LapInfoWrapper(
        env,
        terminate_off_track=env_config.get("terminate_off_track", True),
        max_vehicle_off_track_fraction=env_config.get("max_vehicle_off_track_fraction", 0.5),
        off_track_penalty=env_config.get("off_track_penalty", -100.0),
        probe_dead_zone=env_config.get("clearance_dead_zone", 0.02),
        probe_turn_threshold=env_config.get("clearance_turn_threshold", 0.05),
        probe_max_distance=env_config.get("probe_max_distance", 16.0),
    )
    if env_config.get("curriculum_enabled", False):
        env = CircuitCurriculumWrapper(
            env,
            state_path=env_config["curriculum_state_path"],
            segments=env_config.get("curriculum_segments", 20),
            mastery_visits=env_config.get("curriculum_mastery_visits", 3),
            review_interval=env_config.get("curriculum_review_interval", 4),
            frontier_unlock_segment=env_config.get("curriculum_frontier_unlock_segment"),
        )
    if env_config.get("clearance_steering", False):
        env = ClearanceSteeringWrapper(
            env,
            strength=env_config.get("clearance_steering_strength", 0.7),
            straight_threshold=env_config.get("clearance_dead_zone", 0.02),
            soft_turn_threshold=env_config.get("soft_turn_threshold", 0.10),
            hard_turn_threshold=env_config.get("hard_turn_threshold", 0.20),
            straight_gas=env_config.get("straight_gas", 0.8),
            soft_steering=env_config.get("soft_steering", 0.22),
            hard_steering=env_config.get("hard_steering", 0.65),
            exploration_rate=env_config.get("segment_action_exploration", 0.1),
            straight_target_speed=env_config.get("straight_target_speed", 12.0),
            soft_turn_target_speed=env_config.get("soft_turn_target_speed", 9.0),
            hard_turn_target_speed=env_config.get("hard_turn_target_speed", 6.5),
            max_gas=env_config.get("max_gas", 0.35),
            max_brake=env_config.get("max_brake", 0.6),
            absolute_max_speed=env_config.get("absolute_max_speed", 20.0),
        )
    if render_mode == "human":
        env = TrainingHUDWrapper(env)
    env = GuidedObservation(env, env_config["image_size"])
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
    env_config = dict(config["env"])
    if not training:
        env_config["curriculum_enabled"] = False
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
