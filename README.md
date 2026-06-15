# AI Racer

AI Racer trains a continuous-action PPO policy directly from pixels in Gymnasium's `CarRacing-v3`. Frames are converted to 84x84 grayscale, stacked four deep, and passed to Stable-Baselines3's CNN policy. Environment rewards are never reshaped.

## Track boundary rules

Every generated circuit is identified as an ordered list of road tiles from `track_start` through `track_finish`. At each step the environment reports the nearest tile, positional percentage, visited-tile lap progress, and whether any wheel is touching a Box2D road tile.

The car may lose road contact for 10 frames by default to tolerate brief edge contact or jumps. If all wheels remain off the road beyond that grace period, the environment assigns `-100`, ends the episode with `reset_reason=off_track`, and the vectorized trainer automatically resets onto a new generated track. Configure this with `env.terminate_off_track`, `env.off_track_grace_steps`, and `env.off_track_penalty`.

## Setup

Python 3.11 or 3.12 is supported. Create a virtual environment and install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For NVIDIA training, install the PyTorch build matching your CUDA version first, following the PyTorch installation selector. Confirm it with `python -c "import torch; print(torch.cuda.is_available())"`.

## Commands

Run the lightweight CPU profile first:

```powershell
ai-racer train --config configs/smoke.yaml
```

Watch the vehicle learn in a live window:

```powershell
ai-racer train --config configs/smoke.yaml --visualize --set train.total_timesteps=10000 --set env.max_episode_steps=1000
```

The live HUD shows vehicle speed, compass direction and heading, road-contact risk, native step reward, cumulative episode reward, PPO's current learning rate, and lap progress. `OFF TRACK` changes to red while the grace-period counter advances toward an automatic reset.

The visualization mode uses one environment and is therefore slower than headless parallel training. Closing the race window may stop rendering; use `Ctrl+C` in PowerShell to stop training cleanly. Models and metrics produced before normal completion are available only at configured checkpoint intervals.

Start the full eight-environment, five-million-step GPU run:

```powershell
ai-racer train --config configs/gpu.yaml
```

Resume a checkpoint without resetting its timestep counter:

```powershell
ai-racer train --config configs/gpu.yaml --resume runs/RUN/checkpoints/ppo_carracing_100000_steps.zip --run-dir runs/RUN
```

Evaluate, record, and plot:

```powershell
ai-racer evaluate --config configs/gpu.yaml --model runs/RUN/best_model.zip --output runs/RUN/evaluations/final_standard.json
ai-racer record --config configs/gpu.yaml --model runs/RUN/best_model.zip --output runs/RUN/videos/best.mp4 --seed 2026
ai-racer plot --metrics runs/RUN/metrics/evaluations.jsonl --output runs/RUN/metrics/learning_curve.png
```

Any YAML value can be overridden repeatedly with `--set`, for example:

```powershell
ai-racer train --config configs/gpu.yaml --set train.device=cpu --set env.n_envs=2
```

## Domain-randomized fine-tuning

Resume the standard-color best model into a new run while changing only the environment colors:

```powershell
ai-racer train --config configs/gpu.yaml --resume runs/STANDARD/best_model.zip --set env.domain_randomize=true --set run.name=ppo_carracing_randomized
```

Evaluate the resulting model once with `env.domain_randomize=false` and once with `true`, writing each result to a distinct JSON file. Track geometry is randomized on every episode in both modes; domain randomization additionally changes colors.

## Outputs

Each timestamped run stores the resolved `config.yaml`, training/evaluation seeds, TensorBoard events, monitor CSVs, periodic checkpoints, per-evaluation JSON, JSONL metrics, `best_model.zip`, and `final_model.zip`. Videos and plots can be written into the run's pre-created `videos` and `metrics` directories.

The default evaluator uses 20 fixed unseen seeds. Training stops early only after three consecutive evaluations reach a mean reward of at least 800. Otherwise, it completes the configured budget and leaves the measured scores in the run directory.

## Tests

```powershell
pytest
```

The suite checks preprocessing, action/observation spaces, deterministic resets, frame stacking, configuration overrides, training, checkpoint resume, evaluation, and MP4 creation.
