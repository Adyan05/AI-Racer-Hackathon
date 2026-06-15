from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluation import evaluate_model, save_evaluation
from .plotting import plot_evaluations
from .recording import record_episode
from .training import train
from .models import load_model


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ai-racer")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "record"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/gpu.yaml")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name != "train":
            command.add_argument("--model", required=True)
    commands.choices["train"].add_argument("--resume")
    commands.choices["train"].add_argument("--run-dir")
    commands.choices["train"].add_argument(
        "--visualize",
        action="store_true",
        help="Open a live window showing the car while training (forces one environment).",
    )
    commands.choices["evaluate"].add_argument("--output", default="evaluation.json")
    commands.choices["record"].add_argument("--output", default="agent.mp4")
    commands.choices["record"].add_argument("--seed", type=int, default=2026)
    plot = commands.add_parser("plot")
    plot.add_argument("--metrics", required=True)
    plot.add_argument("--output", default="learning_curve.png")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "plot":
        print(plot_evaluations(args.metrics, args.output))
        return
    config = load_config(args.config, args.set)
    if args.command == "train":
        print(train(config, resume=args.resume, run_dir=args.run_dir, visualize=args.visualize))
        return
    model = load_model(args.model, config, device="cpu")
    if args.command == "evaluate":
        result = evaluate_model(model, config)
        save_evaluation(result, args.output)
    else:
        result = record_episode(model, config, args.output, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
