from pathlib import Path

from stable_baselines3 import PPO

from ai_racer.config import load_config
from ai_racer.evaluation import evaluate_model
from ai_racer.recording import record_episode
from ai_racer.training import train


def test_training_checkpoint_resume_evaluation_and_video(tmp_path: Path):
    config = load_config("configs/smoke.yaml", [f"run.output_dir={tmp_path.as_posix()}"])
    first_run = train(config)
    model_path = first_run / "final_model.zip"
    assert model_path.exists()
    first_model = PPO.load(model_path)
    first_steps = first_model.num_timesteps

    resumed_config = load_config("configs/smoke.yaml", ["train.total_timesteps=64"])
    train(resumed_config, resume=str(model_path), run_dir=first_run)
    resumed_model = PPO.load(model_path.parent / "final_model.zip")
    assert resumed_model.num_timesteps > first_steps

    result = evaluate_model(resumed_model, resumed_config, seeds=[123])
    assert len(result["episodes"]) == 1
    video = tmp_path / "episode.mp4"
    recording = record_episode(resumed_model, resumed_config, video, seed=123)
    assert video.exists() and video.stat().st_size > 0
    assert recording["length"] > 0

