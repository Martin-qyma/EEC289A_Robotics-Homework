# Colab output

The "Colab output (training logs or evaluation results)" deliverable.

**Note on environment:** the assignment runs in Colab, but for this
submission the policy was trained locally on a single RTX 4090 (CUDA 12,
Python 3.11 conda env, identical Brax PPO + MJX setup as the Colab
notebook). The files below are the equivalent of the Colab cell outputs
— same scripts (`train.py`, `generate_public_rollout.py`,
`public_eval.py`, `custom_eval.py`), same JSON file format Brax PPO
writes, just on different hardware.

## Layout

```
colab_output/
├── README.md                                  this file
├── stage_1/                                   per-stage training output
│   ├── progress.json                          5 eval entries: reward + per-term breakdown + PPO diagnostics
│   ├── summary.json                           final eval reward, selected ckpt dir, wall time
│   ├── resolved_config.json                   exact env + PPO config used
│   └── latest_metrics.json                    final eval pass alone
├── stage_2_v1/                                same 4-file set, first 6.5 M-step stage_2 pass
├── stage_2_v2/                                same 4-file set, 13.1 M-step stage_2 continuation
├── run_metadata_phase1.json                   train.py --stage both invocation (stage_1 + stage_2_v1)
├── run_metadata_phase2.json                   train.py --stage stage_2 --restore-checkpoint-dir … invocation (stage_2_v2)
├── stdout/                                    raw piped stdout (eval prints, exit codes, etc.)
│   ├── train_phase1_stage1_and_stage2_v1.log
│   ├── train_phase2_stage2_v2.log
│   ├── public_rollout.log
│   └── custom_eval.log
└── eval_results/                              evaluation script outputs
    ├── public_eval.json                       composite 0.980 + per-episode breakdown (also at submission/public_eval_bundle/)
    ├── rollout_summary.json                   episode lengths, video paths, npz path
    └── custom_eval.json                       29-case per-axis sweep
```

The trained policy itself (`best_checkpoint/`) is the *output* of these
training runs; it lives at `../best_checkpoint/` in the submission root
because `course_config.json:submission.required_files` mandates that
exact path.

## Reward trajectories (extracted from progress.json files)

| stage | (steps, eval_reward) at each eval pass |
|---|---|
| stage_1 | (0, 0.005), (3.3 M, 2.518), (6.6 M, 21.72), (9.8 M, 26.85), (13.1 M, 29.58) |
| stage_2_v1 | (0, 0.000), (1.6 M, 4.73), (3.3 M, 11.67), (4.9 M, 19.07), (6.6 M, 21.65) |
| stage_2_v2 | (0, 21.87), (3.3 M, 25.36), (6.6 M, 27.75), (9.8 M, 29.03), (13.1 M, 29.28) |

Stage_2_v1 ended at 21.65 with the slope still steep — that was the
data-driven reason to continue training (see report §6).

## Headline evaluation numbers

From `eval_results/public_eval.json`:

| metric | mine (v2) | baseline |
|---|---|---|
| course_composite_score | 0.980 | 0.974 |
| velocity_tracking_error | 0.055 | 0.064 |
| yaw_tracking_error | 0.077 | 0.087 |
| fall_rate | 0.00 | 0.00 |
| energy_proxy | 5.79 | 7.90 |
| foot_slip_proxy | 0.056 | 0.067 |

From `eval_results/custom_eval.json` mean absolute errors across the
29-case per-axis grid: vx 0.150, vy 0.094, yaw 0.123, fall_rate 0.00.
