## Work so far in steps

Here’s the learning pipeline you’ve built up, in order, with the active pieces first and the later experimental requests after that.
Active in the code now
Train a continuous-action agent in Gymnasium CarRacing-v3 from raw pixels, with native rewards unchanged.
Preprocess each observation into 84x84 grayscale, stack 4 frames, and feed it into a Stable-Baselines3 CNN policy.
Use a recurrent policy for the circuit profile so the agent can remember earlier parts of the track.
Force a fixed circuit seed for the circuit profile so the same track comes back on reset.
Add curriculum memory over the track, split into 20 segments, so training can start near the currently learned frontier and revisit earlier segments on schedule.
Treat the road as an ordered route from track_start to track_finish, and record progress segment by segment.
Detect when the vehicle leaves the road by measuring the actual Box2D car footprint against the road polygons.
End the episode with -100 when off-track tolerance is exceeded.
Keep the off-track tolerance at 50% of the vehicle being outside the road.
Use directional probe rays to look ahead left and right with a fixed 16-unit maximum range.
Use the probe difference to decide steering: under 2% means straight, around 2-5% means gentle correction, and 5% or more means turn toward the side with more clearance.
Blend the learned steering policy with the clearance controller so the model still learns, but the path-following logic helps it stay on track.
Cap gas when the car is going too fast, and brake automatically above the target speed.
Show the live HUD during visualization with speed, heading, risk, reward, learning rate, lap progress, probe distances, and curriculum status.
Save and resume training from latest_model.zip, with checkpoints and evaluation summaries.
Evaluate on fixed unseen seeds and stop early if the model hits the reward target for 3 consecutive evaluations.
Keep training and evaluation output separated by run directory.
Later requests you made that were part of the learning process, but are now rolled back
Add soft_left and soft_right actions before hard turns.
Change thresholds so small differences stay straight, medium differences use soft turns, and larger differences use hard turns.
Limit max speed to 20 during training.
Add a manual speed slider in the HUD.
Extend the completion target to 50% in a later experiment.
Use the same learned path memory so the agent would not keep re-learning already mastered sections.
Keep the metrics window always visible during training.
Let the agent continue learning across sessions instead of starting from scratch each run.