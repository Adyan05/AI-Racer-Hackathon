from pathlib import Path

from ai_racer.config import load_config
from ai_racer.cli import parser


def test_nested_config_override():
    config = load_config(Path("configs/smoke.yaml"), ["train.total_timesteps=128", "env.domain_randomize=true"])
    assert config["train"]["total_timesteps"] == 128
    assert config["env"]["domain_randomize"] is True


def test_train_visualize_flag():
    args = parser().parse_args(["train", "--config", "configs/smoke.yaml", "--visualize"])
    assert args.visualize is True
