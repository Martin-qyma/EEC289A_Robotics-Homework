# Training & evaluation logs

Note on environment: the assignment runs in Colab, but for this submission
the policy was trained locally on a single RTX 4090 (CUDA 12, conda env,
identical Brax PPO + MJX setup as the Colab notebook). The files below are
the equivalent of the Colab cell outputs.

## Per-stage training output (Brax PPO writes these)

```
stage_1/             stage_1: 13.1 M env steps, forward-only (vx ∈ [0, 0.8])
stage_2_v1/          stage_2 first pass: 6.5 M steps, multi-direction
                     command range, restored from stage_1 best ckpt
stage_2_v2/          stage_2 continuation: +13.1 M steps, restored from
                     stage_2_v1 best ckpt; this is the policy that was
                     evaluated and exported as best_checkpoint/
```

Each per-stage directory contains:

| File | What it is |
|---|---|
| `progress.json` | One entry per evaluation pass. `num_steps` is the env-step count; `metrics["eval/episode_reward"]` is the eval mean reward; per-term reward breakdowns and PPO training diagnostics (KL, entropy loss, sps) are also under `metrics`. |
| `summary.json` | Final training summary (final eval reward, selected checkpoint dir, wall time). |
| `resolved_config.json` | Exact config used for this stage (command_range, command_keep_prob, reward scales, PPO hyperparams). |
| `latest_metrics.json` | The final eval pass alone, for quick grep. |

## Reward trajectories (extracted)

| stage | (steps, eval_reward) |
|---|---|
| stage_1 | (0, 0.005), (3.3 M, 2.518), (6.6 M, 21.72), (9.8 M, 26.85), (13.1 M, 29.58) |
| stage_2_v1 | (0, 0.000), (1.6 M, 4.73), (3.3 M, 11.67), (4.9 M, 19.07), (6.6 M, 21.65) |
| stage_2_v2 | (0, 21.87), (3.3 M, 25.36), (6.6 M, 27.75), (9.8 M, 29.03), (13.1 M, 29.28) |

Stage_2_v1's reward at 6.6 M (21.65) was *not* convergence — the slope
was still steep. That's why the v2 continuation was needed (see report
§2.4 and §6).

## Top-level metadata

`run_metadata_phase1.json` covers `train.py --stage both` (stage_1 +
stage_2_v1).
`run_metadata_phase2.json` covers the second `train.py --stage stage_2
--restore-checkpoint-dir ... --stage2-steps 10000000` invocation that
produced `best_checkpoint/`.

## Raw stdout

`stdout/` contains the full piped stdout of each invocation (training,
public rollout, custom eval). `train_stage1_and_stage2_v1.log` and
`train_stage2_v2.log` show the real-time eval prints; the rollout / eval
logs print the final JSON to stdout.
