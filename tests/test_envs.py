import numpy as np

from ai_racer.config import load_config
from ai_racer.envs import TrainingHUDWrapper, make_single_env, make_vec_env


def test_single_env_preprocessing_and_action_space():
    config = load_config("configs/smoke.yaml")
    env = make_single_env(config["env"], seed=11)
    observation, _ = env.reset(seed=11)
    assert observation.shape == (84, 84, 1)
    assert observation.dtype == np.uint8
    assert env.action_space.shape == (3,)
    np.testing.assert_allclose(env.action_space.low, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(env.action_space.high, [1.0, 1.0, 1.0])
    env.close()


def test_vector_env_transpose_and_frame_stack():
    config = load_config("configs/smoke.yaml")
    env = make_vec_env(config, training=False, seed=12)
    observation = env.reset()
    assert observation.shape == (1, 4, 84, 84)
    env.close()


def test_seeded_first_observation_is_reproducible():
    config = load_config("configs/smoke.yaml")
    first = make_single_env(config["env"], seed=99)
    second = make_single_env(config["env"], seed=99)
    obs_a, _ = first.reset(seed=99)
    obs_b, _ = second.reset(seed=99)
    np.testing.assert_array_equal(obs_a, obs_b)
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
    config["env"]["off_track_grace_steps"] = 1
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
    assert info["reset_reason"] == "off_track"
    env.close()


def test_training_hud_direction_labels():
    assert TrainingHUDWrapper._direction(0.0) == "E"
    assert TrainingHUDWrapper._direction(90.0) == "N"
    assert TrainingHUDWrapper._direction(180.0) == "W"
    assert TrainingHUDWrapper._direction(270.0) == "S"
