import numpy as np
import gymnasium as gym
import pygame

from ai_racer.config import load_config
from ai_racer.envs import CircuitCurriculumWrapper, ClearanceSteeringWrapper, TrainingHUDWrapper, make_single_env, make_vec_env


def test_single_env_preprocessing_and_action_space():
    config = load_config("configs/smoke.yaml")
    env = make_single_env(config["env"], seed=11)
    observation, _ = env.reset(seed=11)
    assert observation["image"].shape == (84, 84, 1)
    assert observation["image"].dtype == np.uint8
    assert observation["guidance"].shape == (7,)
    assert np.all(np.abs(observation["guidance"]) <= 1.0)
    assert env.action_space.shape == (3,)
    np.testing.assert_allclose(env.action_space.low, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(env.action_space.high, [1.0, 1.0, 1.0])
    env.close()


def test_vector_env_transpose_and_frame_stack():
    config = load_config("configs/smoke.yaml")
    env = make_vec_env(config, training=False, seed=12)
    observation = env.reset()
    assert observation["image"].shape == (1, 4, 84, 84)
    assert observation["guidance"].shape == (1, 28)
    env.close()


def test_seeded_first_observation_is_reproducible():
    config = load_config("configs/smoke.yaml")
    first = make_single_env(config["env"], seed=99)
    second = make_single_env(config["env"], seed=99)
    obs_a, _ = first.reset(seed=99)
    obs_b, _ = second.reset(seed=99)
    np.testing.assert_array_equal(obs_a["image"], obs_b["image"])
    np.testing.assert_array_equal(obs_a["guidance"], obs_b["guidance"])
    first.close()
    second.close()


def test_reset_identifies_ordered_track_from_start_to_finish():
    config = load_config("configs/smoke.yaml")
    env = make_single_env(config["env"], seed=77)
    _, info = env.reset(seed=77)
    assert info["track_tiles"] == len(info["track_centerline"])
    assert info["track_start"] == info["track_centerline"][0]
    assert info["track_finish"] == info["track_centerline"][-1]
    assert [point["index"] for point in info["track_centerline"]] == list(range(info["track_tiles"]))
    env.close()


def test_off_track_terminates_episode_with_reset_reason():
    config = load_config("configs/smoke.yaml")
    env = make_single_env(config["env"], seed=88)
    env.reset(seed=88)
    base = env.unwrapped
    base.car.hull.position = (0.0, 0.0)
    for wheel in base.car.wheels:
        wheel.position = (0.0, 0.0)
    _, reward, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated is True
    assert reward == -100.0
    assert info["on_track"] is False
    assert info["vehicle_off_track_fraction"] >= 0.5
    assert info["reset_reason"] == "off_track"
    env.close()


def test_vehicle_remains_valid_below_half_off_track():
    config = load_config("configs/smoke.yaml")
    env = make_single_env(config["env"], seed=89)
    env.reset(seed=89)
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated is False
    assert info["vehicle_off_track_fraction"] < 0.5
    assert info["on_track"] is True
    env.close()


def test_training_hud_direction_labels():
    assert TrainingHUDWrapper._direction(0.0) == "E"
    assert TrainingHUDWrapper._direction(90.0) == "N"
    assert TrainingHUDWrapper._direction(180.0) == "W"
    assert TrainingHUDWrapper._direction(270.0) == "S"


def test_slider_value_maps_to_zero_and_hundred():
    rect = pygame.Rect(100, 20, 240, 14)
    assert TrainingHUDWrapper._slider_value_from_mouse(100, rect) == 0.0
    assert TrainingHUDWrapper._slider_value_from_mouse(340, rect) == 100.0


def test_fixed_track_seed_recreates_same_circuit():
    config = load_config("configs/circuit.yaml")
    env = make_single_env(config["env"], seed=1)
    _, first_info = env.reset()
    _, second_info = env.reset()
    assert first_info["track_centerline"] == second_info["track_centerline"]
    env.close()


def test_directional_rays_are_boundary_clipped():
    config = load_config("configs/circuit.yaml")
    env = make_single_env(config["env"], seed=2)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert 0.0 <= info["left_probe_distance"] <= info["probe_max_distance"]
    assert 0.0 <= info["right_probe_distance"] <= info["probe_max_distance"]
    assert np.isclose(info["left_probe"], info["left_probe_distance"] / info["probe_max_distance"])
    assert np.isclose(info["right_probe"], info["right_probe_distance"] / info["probe_max_distance"])
    assert info["probe_max_distance"] == 16.0
    env.close()


def test_clearance_steering_chooses_largest_boundary_distance():
    config = load_config("configs/circuit.yaml")
    env = make_single_env(config["env"], seed=3)
    env.reset()
    _, _, _, _, info = env.step(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    left = info["left_probe_distance"]
    right = info["right_probe_distance"]
    relative_difference = abs(left - right) / max(left, right, 1e-6)
    if relative_difference <= config["env"]["clearance_dead_zone"]:
        assert info["clearance_decision"] == "straight"
    elif relative_difference < config["env"]["soft_turn_threshold"]:
        assert info["clearance_decision"] in ("soft_left", "soft_right")
    elif left > right:
        assert info["clearance_decision"] in ("soft_left", "hard_left")
        assert info["applied_steering"] < info["policy_steering"]
    else:
        assert info["clearance_decision"] in ("soft_right", "hard_right")
    env.close()


def test_clearance_target_uses_straight_dead_zone():
    wrapper = object.__new__(ClearanceSteeringWrapper)
    wrapper.straight_threshold = 0.02
    wrapper.soft_turn_threshold = 0.10
    wrapper.hard_turn_threshold = 0.20
    wrapper.soft_steering = 0.22
    wrapper.env = type("ProbeEnv", (), {"get_wrapper_attr": lambda self, name: (100.0, 101.9)})()
    target, decision, difference = wrapper._clearance_target()
    assert target == 0.0
    assert decision == "straight"
    assert difference < 0.02


def test_clearance_target_uses_soft_turn_between_ten_and_twenty_percent():
    class ProbeEnv:
        def get_wrapper_attr(self, name):
            if name == "probe_distances":
                return (10.0, 11.5)
            if name == "current_segment":
                return 3
            if name == "action_success_rate":
                return lambda segment, action: (0.5, 0)
            raise AttributeError(name)

    wrapper = object.__new__(ClearanceSteeringWrapper)
    wrapper.straight_threshold = 0.02
    wrapper.soft_turn_threshold = 0.10
    wrapper.hard_turn_threshold = 0.20
    wrapper.soft_steering = 0.22
    wrapper.hard_steering = 0.65
    wrapper.exploration_rate = 0.0
    wrapper.env = ProbeEnv()
    target, decision, difference = wrapper._clearance_target()
    assert difference >= 0.05
    assert target > 0.0
    assert decision == "soft_right"


def test_clearance_target_uses_hard_turn_above_twenty_percent():
    class ProbeEnv:
        def get_wrapper_attr(self, name):
            if name == "probe_distances":
                return (10.0, 13.0)
            if name == "current_segment":
                return 3
            if name == "action_success_rate":
                return lambda segment, action: (0.5, 0)
            raise AttributeError(name)

    wrapper = object.__new__(ClearanceSteeringWrapper)
    wrapper.straight_threshold = 0.02
    wrapper.soft_turn_threshold = 0.10
    wrapper.hard_turn_threshold = 0.20
    wrapper.soft_steering = 0.22
    wrapper.hard_steering = 0.65
    wrapper.exploration_rate = 0.0
    wrapper.env = ProbeEnv()
    target, decision, difference = wrapper._clearance_target()
    assert difference > 0.20
    assert target == 0.65
    assert decision == "hard_right"


def test_evaluation_disables_curriculum_frontier(tmp_path):
    config = load_config(
        "configs/circuit.yaml",
        [f"env.curriculum_state_path={tmp_path.as_posix()}/curriculum.json"],
    )
    env = make_vec_env(config, training=False, seed=2026)
    env.reset()
    assert not env.env_is_wrapped(CircuitCurriculumWrapper)[0]
    env.close()


def test_completion_first_speed_controller_throttles_and_brakes():
    config = load_config("configs/circuit.yaml")
    env = make_single_env(config["env"], seed=4)
    env.reset()
    base = env.unwrapped
    base.car.hull.linearVelocity = (0.0, 0.0)
    _, _, _, _, low_speed = env.step(np.zeros(3, dtype=np.float32))
    assert 0.0 < low_speed["applied_gas"] <= config["env"]["max_gas"]
    assert low_speed["applied_brake"] == 0.0

    base.car.hull.linearVelocity = (30.0, 0.0)
    _, _, _, _, high_speed = env.step(np.zeros(3, dtype=np.float32))
    assert high_speed["applied_gas"] == 0.0
    assert 0.0 < high_speed["applied_brake"] <= config["env"]["max_brake"] + 1e-6
    env.close()


def test_manual_slider_max_speed_caps_target_speed():
    config = load_config("configs/circuit.yaml")
    env = make_single_env(config["env"], seed=5)
    env.reset()
    env.get_wrapper_attr("set_manual_max_speed")(7.5)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["manual_max_speed"] == 7.5
    assert info["target_speed"] <= 7.5 + 1e-6
    env.close()


def test_circuit_config_targets_half_track_completion():
    config = load_config("configs/circuit.yaml")
    assert config["env"]["lap_complete_percent"] == 0.50
    assert config["env"]["max_episode_steps"] == 5000


def test_curriculum_stays_at_start_until_half_track_mastered(tmp_path):
    state_path = tmp_path / "curriculum.json"
    class StubEnv(gym.Env):
        metadata = {}

        def __init__(self):
            self.track = [(0.0, 0.0, 0.0, 0.0)] * 20
            self.car = None

        @property
        def unwrapped(self):
            return self

        def reset(self, *, seed=None, options=None):
            return "obs", {}

    wrapper = CircuitCurriculumWrapper(
        env=StubEnv(),
        state_path=state_path,
        segments=20,
        frontier_unlock_segment=10,
    )
    state = wrapper._default_state()
    state["mastered_segment"] = 9
    wrapper._save_state(state)
    _, info = wrapper.reset()
    assert info["curriculum_start_segment"] == 0
    assert info["frontier_unlock_segment"] == 10
