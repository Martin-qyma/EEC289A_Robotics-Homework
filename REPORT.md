# EEC289A — Assignment 1: Command-Conditioned Go2 Locomotion

**Author:** Qiyao Ma — `martin.qyma@gmail.com`
**Repo:** https://github.com/Martin-qyma/EEC289A_Robotics-Homework
**Colab notebook:** `Assignment_1_Colab.ipynb`
**Demo video:** `artifacts/demo_bundle/demo.mp4`

---

## 1. Problem setup

The baseline at `WeijieLai1024/EEC289A_Robotics-Homework` ships a two-stage PPO
recipe for the Unitree Go2 quadruped on flat terrain (MuJoCo Playground +
Brax PPO + MJX). The actor receives a 48-dim observation that includes the
3-D command `[vx, vy, yaw_rate]` in the robot's local frame; the action is a
12-dim joint position offset added to a default standing pose. Two reward
terms reward velocity tracking — `tracking_lin_vel` (Gaussian on `[vx, vy]`)
and `tracking_ang_vel` (Gaussian on yaw_rate) — and a `stand_still` cost
penalizes pose deviation only when `||command|| < 0.01`. The provided baseline
configures both stage_1 and stage_2 with `vy = yaw_rate = 0`, so the policy
only ever sees forward motion commands and predictably collapses on the
public benchmark's lateral, yaw, and combined episodes.

The objective is to extend this into a single command-conditioned policy that
tracks `±vx`, `±vy`, `±yaw_rate`, including combined commands, and to
characterize its behavior with both the official benchmark and a custom
per-axis evaluation.

## 2. Design choices

### 2.1 Curriculum strategy

I kept stage_1 untouched as a forward-only warm start (10 M env steps,
`vx ∈ [0, 0.8]`) and only changed stage_2 (5 M env steps,
`restore_previous_stage_checkpoint: true`). The stage_1 → stage_2 hand-off
already exists as a curriculum scaffold; I reuse it instead of inventing a
new schedule. Stage_1 teaches the policy the basics of standing/striding from
a stable forward signal; stage_2 fine-tunes that policy onto the full
command distribution. The alternative — training multi-direction from
scratch — empirically takes longer to find a stable gait because PPO has to
discover both a balanced trot and command conditioning at once.

### 2.2 Stage_2 command distribution

| Field | Baseline | Mine |
|---|---|---|
| `command_range.min` | `[ 0.0, 0.0, 0.0]` | `[-0.6, -0.2, -0.6]` |
| `command_range.max` | `[ 1.0, 0.0, 0.0]` | `[ 0.6,  0.2,  0.6]` |
| `command_keep_prob` | `[1.0, 0.0, 0.0]` | `[1.0, 0.6, 0.6]` |

The range matches the public benchmark's `safe_command_ranges`, so the policy
trains exactly on the distribution it's evaluated against. For the keep
probabilities I went **higher (0.6) than the suggested 0.35**: stage_1
already biases the policy heavily toward `+vx`, so within only 5 M stage_2
steps the policy needs frequent `vy` and `yaw_rate` exposure to overcome
that prior. Lower keep probabilities (e.g. 0.35) leave the per-step command
distribution dominated by zeros on the lateral / yaw axes, which slows the
fine-tune.

### 2.3 Code seam

The baseline already factors stage_2 command sampling into a homework hook,
`Joystick._student_stage2_sampling_profile()` (joystick.py:552). The default
implementation returns the stage_1 ranges, so even with the JSON edit above
the *mid-episode* command resamples (joystick.py:297–308) would collapse
back to forward-only. I changed it to return the `student_stage2_goal_*`
values, so reset-time and step-time sampling agree:

```python
def _student_stage2_sampling_profile(self, current_command):
    del current_command
    return (
        self._student_stage2_goal_min,
        self._student_stage2_goal_max,
        self._student_stage2_goal_b,
    )
```

This is the **only** functional change to the env. I deliberately did not
touch reward weights, observation layout, or domain randomization — the
existing reward terms already cover all 3 command axes symmetrically and
the existing `stand_still` cost only fires for near-zero commands.

### 2.4 What I did not change (and why)

- **No reward shaping per axis.** `tracking_lin_vel` and `tracking_ang_vel`
  treat all axes symmetrically. Adding axis-specific weights would be
  another knob to tune, and the benchmark normalizer treats a unit of `vx`
  error the same as a unit of `vy` error.
- **No new curriculum inside stage_2.** Doing it cleanly needs a global
  step counter threaded through env state, which the baseline does not
  expose. The two-stage split already provides one curriculum step.
- **No network changes.** The default 256-256-128 actor / critic MLP
  already conditions on the 3-D command in its 48-dim input; the bottleneck
  was data, not capacity.

## 3. Official benchmark results

Run with `generate_public_rollout.py --num-episodes 4 --render-first-episode`
followed by `public_eval.py`.

### 3.1 Composite scores

| Run | composite_score | velocity_tracking_error | yaw_tracking_error | fall_rate | energy_proxy | foot_slip_proxy |
|---|---|---|---|---|---|---|
| Baseline (forward-only) | 0.974 | 0.064 | 0.087 | 0.00 | 7.90 | 0.067 |
| Mine (command-conditioned) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

> The baseline's high composite is misleading: it tracks `vx` well because
> that is all it ever sees, and its mean error on lateral / yaw episodes is
> low only because both commanded and measured `vy` are near zero (the robot
> simply doesn't move sideways).

### 3.2 Per-episode breakdown

| Episode | Description | velocity_tracking_error (mine) | yaw_tracking_error (mine) | fell |
|---|---|---|---|---|
| 0 | forward / backward `vx` | _TODO_ | _TODO_ | _TODO_ |
| 1 | lateral `vy` | _TODO_ | _TODO_ | _TODO_ |
| 2 | yaw `yaw_rate` | _TODO_ | _TODO_ | _TODO_ |
| 3 | combined | _TODO_ | _TODO_ | _TODO_ |

> Baseline episode 1 (lateral): velocity_tracking_error = 0.079.
> Baseline episode 2 (yaw): yaw_tracking_error = 0.170.
> The baseline never falls because it just stands when commanded sideways.
> The improvement test is whether mine produces *lower* velocity / yaw error
> on episodes 1, 2, 3 *while keeping* fall_rate at 0.

## 4. Custom evaluation

Run with `custom_eval.py` (added in this submission). For each command in a
per-axis grid the script holds the command fixed for 5 s, drops the first
1 s as warmup, and reports the residual mean / std of measured velocity vs.
commanded.

### 4.1 Per-axis tracking

![per-axis vx](artifacts/custom_eval/custom_eval_vx.png)
![per-axis vy](artifacts/custom_eval/custom_eval_vy.png)
![per-axis yaw_rate](artifacts/custom_eval/custom_eval_yaw.png)
![per-axis absolute error](artifacts/custom_eval/custom_eval_err.png)

Discussion (fill in after running): which axis tracks best? At what
magnitude does tracking error grow? Does the policy refuse to go beyond
the trained safe range (saturation) or does it diverge / fall?

### 4.2 Stability under combined commands

The two `combo` rows show whether the policy can simultaneously satisfy
`(vx, vy, yaw_rate)`. If the per-axis tracking is good but the combo error
is much larger, the policy has learned axis-specific behaviors that don't
compose; if combo error is comparable, the conditioning is genuinely 3-D.

## 5. Demo video

`artifacts/demo_bundle/demo.mp4` runs the policy through 12 segments of 5 s
each (60 s total) covering: stand → forward `+vx ∈ {0.6, 0.8, 1.0}` →
backward `-vx ∈ {-0.6, -1.0}` → lateral `±vy = 0.3` → yaw `±yaw = 0.8` →
combined `(0.4, 0.2, 0.6)` → stand. Some magnitudes (`vx = 1.0`,
`vy = 0.3`, `yaw = 1.0`) sit at or beyond the trained range so the video
also shows extrapolation behavior.

Generate with:

```bash
python test_policy.py \
  --config configs/colab_runtime_config.json \
  --checkpoint-dir artifacts/run_extended/best_checkpoint \
  --stage-name stage_2 \
  --output-dir artifacts/demo_bundle \
  --episode-length 3500
```

## 6. Insights

_Fill in after the Colab run. Suggested points to discuss:_

- Did stage_2 reward initially dip when the multi-direction commands turned
  on, then recover? (Expected — stage_1 weights are forward-biased.)
- Is yaw harder to track than lateral, or vice versa?
- How much of the gain on lateral / yaw episodes came from just changing
  the JSON range vs. also fixing the homework seam in joystick.py?
- Sim-to-real implications: does training on the full safe range make the
  policy more or less robust to perturbations? (Hint: `pert_config` is off
  by default; flipping it on after the run is a useful follow-up.)

## 7. Failed idea (required by the rubric)

_Replace this with a real failed attempt during your run._ Examples:

- Using `command_keep_prob = [1.0, 1.0, 1.0]`: too much per-step churn, the
  policy never settles into a stable command and tracking error explodes.
- Skipping stage_1 and training stage_2 from scratch on the wider range:
  PPO does not find a stable gait in 5 M steps without the stage_1 warm
  start.
- Setting stage_2 ranges to `±1.0` everywhere (matching the demo extreme
  values): the policy spends too much capacity on commands outside the
  benchmark's safe range and tracks worse inside the safe range.

## 8. Reproduction

```bash
# In Colab
python train.py \
  --config configs/colab_runtime_config.json \
  --stage both \
  --output-dir artifacts/run_extended

python generate_public_rollout.py \
  --config configs/colab_runtime_config.json \
  --checkpoint-dir artifacts/run_extended/best_checkpoint \
  --stage-name stage_2 \
  --output-dir artifacts/public_eval_bundle \
  --num-episodes 4 \
  --render-first-episode

python public_eval.py \
  --config configs/colab_runtime_config.json \
  --rollout-npz artifacts/public_eval_bundle/rollout_public_eval.npz \
  --output-json artifacts/public_eval_bundle/public_eval.json

python custom_eval.py \
  --config configs/colab_runtime_config.json \
  --checkpoint-dir artifacts/run_extended/best_checkpoint \
  --stage-name stage_2 \
  --output-dir artifacts/custom_eval

python test_policy.py \
  --config configs/colab_runtime_config.json \
  --checkpoint-dir artifacts/run_extended/best_checkpoint \
  --stage-name stage_2 \
  --output-dir artifacts/demo_bundle \
  --episode-length 3500
```
