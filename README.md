# AI Racer

AI Racer trains a continuous-action PPO policy directly from pixels in Gymnasium's `CarRacing-v3`. Frames are converted to 84x84 grayscale, stacked four deep, and passed to Stable-Baselines3's CNN policy. Environment rewards are never reshaped.

## Circuit learning with directional probes

The circuit profile regenerates the same track on every reset and uses Recurrent PPO with a CNN plus LSTM memory. Its observation combines the stacked camera frames with seven explainable values: left and right forward-road probe scores, their difference, upcoming heading error, lateral centerline error, speed, and circuit position.

The black lines shown over the car are boundary-clipped rays with a fixed 16-unit measurement range. Distances within 2% select straight mode and raise gas to at least 0.8. Differences of 5% or more select a turn toward the longer-clearance ray. The 2-5% band applies a gentle correction so steering does not oscillate at the thresholds.

In the circuit profile, clearance steering is also enabled as a transparent safety controller. The larger ray distance creates a proportional steering target, while differences within `clearance_dead_zone` target straight driving. `clearance_steering_strength` controls how strongly that target is blended with the recurrent policy's steering; gas and brake remain fully learned. The HUD displays both policy-requested and actually applied steering.

Run circuit learning on CUDA with the live probe display:

```powershell
ai-racer train --config configs/circuit.yaml --visualize
```

This command now performs continuous learning. It reuses `runs/continuous_circuit/latest_model.zip` automatically across launches and saves it every 5,000 steps plus at normal shutdown. The fixed track is divided into 20 segments in `curriculum.json`: three successful reaches master the next segment. In the current phase, episodes start from the beginning until segment 10 is mastered, and an episode only ends when the car crashes or reaches the 50% lap-completion target.

Each segment also remembers outcomes for `straight`, `soft_left`, `soft_right`, `hard_left`, and `hard_right`. Crossing into the next segment records the preceding action as successful; leaving the track records failure. Later episodes reuse the better-performing soft or hard turn for that segment, with 10% exploration so the controller can still discover improvements. Soft steering defaults to 0.22 and hard steering to 0.65.

The current priority is track completion, not speed. The controller targets speed 12 on straights, 9 during soft turns, and 6.5 during hard turns, with gas capped at 0.35 and automatic braking above target. These values remain fixed; no speed curriculum is enabled yet. The opaque metrics panel is redrawn immediately on reset and every frame so it remains visible throughout training.

Steering priority is deterministic: up to 2% ray difference stays straight; 2-10% applies a small soft correction; 10-20% uses soft left/right; above 20% uses hard left/right. Segment outcome memory may score these actions but cannot choose a hard turn below the 20% threshold. An absolute speed ceiling of 20 forces gas to zero and braking whenever exceeded.

The default fixed circuit is controlled by `env.fixed_track_seed`. Change that seed to learn a different circuit. Remove it or set it to `null` to return to randomly generated tracks.

## Track boundary rules

Every generated circuit is identified as an ordered list of road tiles from `track_start` through `track_finish`. At each step the environment reports the nearest tile, positional percentage, visited-tile lap progress, and whether any wheel is touching a Box2D road tile.

The environment samples the actual Box2D vehicle footprint against nearby road polygons every frame. Partial edge contact is allowed while less than 50% of the vehicle is outside the road. At 50% or more, the environment assigns `-100`, ends the episode with `reset_reason=off_track`, and the vectorized trainer automatically resets onto a new generated track. Configure this with `env.terminate_off_track`, `env.max_vehicle_off_track_fraction`, and `env.off_track_penalty`.

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

The live HUD shows vehicle speed, compass direction and heading, the percentage of the vehicle outside the road, native step reward, cumulative episode reward, PPO's current learning rate, and lap progress. Edge risk is amber; reaching the 50% tolerance displays a red failure before the automatic reset.

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
