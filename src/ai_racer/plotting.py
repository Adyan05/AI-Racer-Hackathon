from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_evaluations(metrics_file: str | Path, output: str | Path) -> Path:
    metrics_file, output = Path(metrics_file), Path(output)
    records = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No evaluation records found in {metrics_file}")
    steps = [record["timesteps"] for record in records]
    rewards = [record["mean_reward"] for record in records]
    completion = [100 * record["completion_rate"] for record in records]
    fig, reward_axis = plt.subplots(figsize=(9, 5))
    reward_axis.plot(steps, rewards, marker="o", label="Mean reward")
    reward_axis.axhline(800, color="tab:green", linestyle="--", label="Target reward")
    reward_axis.set(xlabel="Environment steps", ylabel="Mean reward", title="AI Racer evaluation")
    completion_axis = reward_axis.twinx()
    completion_axis.plot(steps, completion, color="tab:orange", marker="s", label="Completion rate")
    completion_axis.set_ylabel("Completion rate (%)")
    lines = reward_axis.lines + completion_axis.lines
    reward_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output

