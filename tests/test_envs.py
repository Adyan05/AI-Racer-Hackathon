import numpy as np

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
    elif relative_difference < config["env"]["clearance_turn_threshold"]:
        assert info["clearance_decision"] == "correct"
    elif left > right:
        assert info["clearance_decision"] == "left"
        assert info["applied_steering"] < info["policy_steering"]
    else:
        assert info["clearance_decision"] == "right"
    env.close()


def test_clearance_target_uses_straight_dead_zone():
    wrapper = object.__new__(ClearanceSteeringWrapper)
    wrapper.straight_threshold = 0.02
    wrapper.turn_threshold = 0.05
    wrapper.env = type("ProbeEnv", (), {"get_wrapper_attr": lambda self, name: (100.0, 101.9)})()
    target, decision, difference = wrapper._clearance_target()
    assert target == 0.0
    assert decision == "straight"
    assert difference < 0.02


def test_clearance_target_turns_at_five_percent():
    wrapper = object.__new__(ClearanceSteeringWrapper)
    wrapper.straight_threshold = 0.02
    wrapper.turn_threshold = 0.05
    wrapper.env = type("ProbeEnv", (), {"get_wrapper_attr": lambda self, name: (10.0, 10.6)})()
    target, decision, difference = wrapper._clearance_target()
    assert difference >= 0.05
    assert target > 0.0
    assert decision == "right"


def test_evaluation_disables_curriculum_frontier(tmp_path):
    config = load_config(
        "configs/circuit.yaml",
        [f"env.curriculum_state_path={tmp_path.as_posix()}/curriculum.json"],
    )
    env = make_vec_env(config, training=False, seed=2026)
    env.reset()
    assert not env.env_is_wrapped(CircuitCurriculumWrapper)[0]
    env.close()
