from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from .evaluation import evaluate_model, save_evaluation


class FixedSeedEvalCallback(BaseCallback):
    def __init__(self, config, run_dir: str | Path, verbose: int = 1):
        super().__init__(verbose)
        self.config = config
        self.run_dir = Path(run_dir)
        self.frequency = int(config["eval"]["frequency"])
        self.next_evaluation = self.frequency
        self.best_reward = float("-inf")
        self.target_streak = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_evaluation:
            return True
        result = evaluate_model(self.model, self.config)
        result["timesteps"] = self.num_timesteps
        metrics_path = self.run_dir / "metrics" / "evaluations.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        save_evaluation(result, self.run_dir / "evaluations" / f"step_{self.num_timesteps}.json")
        if result["mean_reward"] > self.best_reward:
            self.best_reward = result["mean_reward"]
            self.model.save(self.run_dir / "best_model")
        target = float(self.config["eval"]["target_mean_reward"])
        self.target_streak = self.target_streak + 1 if result["mean_reward"] >= target else 0
        self.logger.record("eval/mean_reward", result["mean_reward"])
        self.logger.record("eval/completion_rate", result["completion_rate"])
        if self.verbose:
            print(f"Evaluation at {self.num_timesteps}: mean={result['mean_reward']:.2f}, completion={result['completion_rate']:.0%}")
        self.next_evaluation += self.frequency
        return self.target_streak < int(self.config["eval"]["target_streak"])


class LiveHUDCallback(BaseCallback):
    """Keep the visualization HUD synchronized with PPO's LR schedule."""

    def _on_training_start(self) -> None:
        self._update_learning_rate()

    def _on_step(self) -> bool:
        self._update_learning_rate()
        return True

    def _update_learning_rate(self) -> None:
        learning_rate = float(self.model.lr_schedule(self.model._current_progress_remaining))
        self.training_env.env_method("set_learning_rate", learning_rate)


class ContinuousSaveCallback(BaseCallback):
    """Maintain one resumable model across repeated training sessions."""

    def __init__(self, path: str | Path, frequency: int, verbose: int = 1):
        super().__init__(verbose)
        self.path = Path(path)
        self.frequency = max(1, frequency)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.frequency == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.path)
            if self.verbose:
                print(f"Continuous model saved at {self.num_timesteps} steps: {self.path}.zip")
        return True
