#!/usr/bin/env python3
"""Per-direction evaluation of a Go2 locomotion checkpoint.

Companion to `public_eval.py`. The official benchmark mixes axes inside each
episode, which makes it hard to attribute error to a single command direction.
This script runs short fixed-command episodes for an explicit grid of single-
axis and combined commands so the report can show per-axis tracking, per-
magnitude behavior, and per-command stability.

Outputs:
  <output-dir>/custom_eval.json     numeric summary per command
  <output-dir>/custom_eval_lin.png  bar chart: vx / vy commanded vs. measured
  <output-dir>/custom_eval_yaw.png  bar chart: yaw_rate commanded vs. measured
  <output-dir>/custom_eval_err.png  bar chart: per-axis absolute tracking error
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent


@dataclass
class CommandCase:
    label: str
    group: str
    command: tuple[float, float, float]


def default_command_grid() -> list[CommandCase]:
    """Per-axis sweep + a couple of combined commands.

    Magnitudes match the assignment's suggested values. Some are intentionally
    at the edge or just outside the trained safe range so the report can show
    where the policy starts to break down.
    """
    cases: list[CommandCase] = [CommandCase("stand", "stand", (0.0, 0.0, 0.0))]

    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        cases.append(CommandCase(f"+vx={v:.1f}", "vx_forward", (v, 0.0, 0.0)))
    for v in (-0.2, -0.4, -0.6, -0.8, -1.0):
        cases.append(CommandCase(f"vx={v:.1f}", "vx_backward", (v, 0.0, 0.0)))

    for v in (0.1, 0.2, 0.3, 0.4):
        cases.append(CommandCase(f"+vy={v:.1f}", "vy_left", (0.0, v, 0.0)))
    for v in (-0.1, -0.2, -0.3, -0.4):
        cases.append(CommandCase(f"vy={v:.1f}", "vy_right", (0.0, v, 0.0)))

    for v in (0.3, 0.6, 0.8, 1.0):
        cases.append(CommandCase(f"+yaw={v:.1f}", "yaw_left", (0.0, 0.0, v)))
    for v in (-0.3, -0.6, -0.8, -1.0):
        cases.append(CommandCase(f"yaw={v:.1f}", "yaw_right", (0.0, 0.0, v)))

    cases.append(CommandCase("combo+", "combined", (0.4, 0.2, 0.6)))
    cases.append(CommandCase("combo-", "combined", (-0.4, -0.2, -0.6)))

    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument(
        "--seconds-per-case",
        type=float,
        default=5.0,
        help="How long to hold each command. The first warmup_seconds are dropped from the metric.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.0,
        help="Skip this many seconds at the start of each case so the policy can ramp up to commanded velocity.",
    )
    parser.add_argument("--episode-length", type=int, default=2000)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _force_command(state, command: np.ndarray, jax) -> object:
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def _build_stack(config: dict, force_cpu: bool):
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)
    return lazy_import_stack()


def _make_env(stack, config: dict, stage_name: str, episode_length: int):
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)
    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, stage_name)
    env_cfg.episode_length = int(episode_length)
    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    return env, env_cfg


def run_case(
    env,
    state,
    policy,
    jax,
    command: np.ndarray,
    n_steps: int,
    rng,
    step_fn,
) -> dict:
    state = _force_command(state, command, jax)
    measured_xy = np.zeros((n_steps, 2), dtype=np.float32)
    measured_yaw = np.zeros((n_steps,), dtype=np.float32)
    torques = np.zeros((n_steps, 12), dtype=np.float32)
    joint_vels = np.zeros((n_steps, 12), dtype=np.float32)
    feet_slip = np.zeros((n_steps, 4), dtype=np.float32)
    base_height = np.zeros((n_steps,), dtype=np.float32)
    fell_step = -1

    for t in range(n_steps):
        rng, act_key = jax.random.split(rng)
        action, _ = policy(state.obs, act_key)
        state = step_fn(state, action)
        state = _force_command(state, command, jax)
        measured_xy[t] = np.asarray(env.get_local_linvel(state.data)[:2])
        measured_yaw[t] = float(env.get_gyro(state.data)[2])
        torques[t] = np.asarray(state.data.actuator_force)
        joint_vels[t] = np.asarray(state.data.qvel[6:])
        feet_vel = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr])
        feet_slip[t] = np.linalg.norm(feet_vel[:, :2], axis=-1)
        base_height[t] = float(state.data.qpos[2])
        if bool(np.asarray(state.done)) and fell_step < 0:
            fell_step = t
            break

    return {
        "rng": rng,
        "state": state,
        "measured_xy": measured_xy,
        "measured_yaw": measured_yaw,
        "torques": torques,
        "joint_vels": joint_vels,
        "feet_slip": feet_slip,
        "base_height": base_height,
        "fell_step": fell_step,
    }


def summarize_case(case: CommandCase, run: dict, ctrl_dt: float, warmup_steps: int) -> dict:
    cmd = np.asarray(case.command, dtype=np.float32)
    measured_xy = run["measured_xy"]
    measured_yaw = run["measured_yaw"]
    fell = run["fell_step"] >= 0
    valid_steps = run["fell_step"] if fell else len(measured_xy)
    if valid_steps <= warmup_steps:
        warmup = 0
    else:
        warmup = warmup_steps
    sl = slice(warmup, valid_steps)

    if valid_steps > warmup:
        mx = measured_xy[sl, 0]
        my = measured_xy[sl, 1]
        myaw = measured_yaw[sl]
        err_vx = mx - cmd[0]
        err_vy = my - cmd[1]
        err_yaw = myaw - cmd[2]
        torque = run["torques"][sl]
        jvel = run["joint_vels"][sl]
        slip = run["feet_slip"][sl]
        bh = run["base_height"][sl]
        energy = float(np.mean(np.sum(np.abs(torque * jvel), axis=-1)))
        slip_proxy = float(np.mean(slip))
        base_h_mean = float(np.mean(bh))
    else:
        mx = my = myaw = err_vx = err_vy = err_yaw = np.zeros(0)
        energy = 0.0
        slip_proxy = 0.0
        base_h_mean = 0.0

    return {
        "label": case.label,
        "group": case.group,
        "command": [float(x) for x in case.command],
        "fell": fell,
        "fall_time_s": (run["fell_step"] * ctrl_dt) if fell else None,
        "measured_vx_mean": float(np.mean(mx)) if mx.size else None,
        "measured_vy_mean": float(np.mean(my)) if my.size else None,
        "measured_yaw_mean": float(np.mean(myaw)) if myaw.size else None,
        "measured_vx_std": float(np.std(mx)) if mx.size else None,
        "measured_vy_std": float(np.std(my)) if my.size else None,
        "measured_yaw_std": float(np.std(myaw)) if myaw.size else None,
        "abs_err_vx": float(np.mean(np.abs(err_vx))) if err_vx.size else None,
        "abs_err_vy": float(np.mean(np.abs(err_vy))) if err_vy.size else None,
        "abs_err_yaw": float(np.mean(np.abs(err_yaw))) if err_yaw.size else None,
        "lin_err_norm": float(np.mean(np.sqrt(err_vx ** 2 + err_vy ** 2))) if err_vx.size else None,
        "energy_proxy": energy,
        "foot_slip_proxy": slip_proxy,
        "base_height_mean_m": base_h_mean,
        "valid_steps": int(valid_steps),
    }


def maybe_plot(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[custom_eval] matplotlib unavailable, skipping plots: {exc}")
        return

    def _bar(metric: str, ylabel: str, fname: str, *, signed: bool = False) -> None:
        labels = [r["label"] for r in rows]
        cmds = np.array([r["command"] for r in rows])
        idx_axis = {"vx": 0, "vy": 1, "yaw": 2}[metric]
        commanded = cmds[:, idx_axis]
        measured_key = {"vx": "measured_vx_mean", "vy": "measured_vy_mean", "yaw": "measured_yaw_mean"}[metric]
        measured = np.array([(r[measured_key] if r[measured_key] is not None else np.nan) for r in rows])

        x = np.arange(len(labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.5))
        ax.bar(x - width / 2, commanded, width, label="commanded")
        ax.bar(x + width / 2, measured, width, label="measured")
        ax.axhline(0.0, color="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} tracking, commanded vs. measured")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=140)
        plt.close(fig)

    _bar("vx", "vx (m/s)", "custom_eval_vx.png")
    _bar("vy", "vy (m/s)", "custom_eval_vy.png")
    _bar("yaw", "yaw_rate (rad/s)", "custom_eval_yaw.png")

    labels = [r["label"] for r in rows]
    err_vx = [(r["abs_err_vx"] or 0.0) for r in rows]
    err_vy = [(r["abs_err_vy"] or 0.0) for r in rows]
    err_yaw = [(r["abs_err_yaw"] or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.5))
    x = np.arange(len(labels))
    width = 0.27
    ax.bar(x - width, err_vx, width, label="|err vx|")
    ax.bar(x, err_vy, width, label="|err vy|")
    ax.bar(x + width, err_yaw, width, label="|err yaw|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, fontsize=8)
    ax.set_ylabel("absolute tracking error")
    ax.set_title("Per-axis absolute tracking error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "custom_eval_err.png", dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True
    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = _build_stack(config, force_cpu)
    jax = stack["jax"]

    env, env_cfg = _make_env(stack, config, args.stage_name, args.episode_length)
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    ctrl_dt = float(env_cfg.ctrl_dt)
    n_steps = max(1, int(round(args.seconds_per_case / ctrl_dt)))
    warmup_steps = max(0, int(round(args.warmup_seconds / ctrl_dt)))

    cases = default_command_grid()
    seed = int(args.seed if args.seed is not None else config.get("seed", 0)) + 7
    rng = jax.random.PRNGKey(seed)

    rows: list[dict] = []
    for case_idx, case in enumerate(cases):
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)
        cmd = np.asarray(case.command, dtype=np.float32)
        run = run_case(env, state, policy, jax, cmd, n_steps, rng, step_fn)
        rng = run["rng"]
        row = summarize_case(case, run, ctrl_dt, warmup_steps)
        rows.append(row)
        print(
            f"[{case_idx + 1:2d}/{len(cases)}] {case.label:>12s}  "
            f"|err vx|={(row['abs_err_vx'] or 0):.3f}  "
            f"|err vy|={(row['abs_err_vy'] or 0):.3f}  "
            f"|err yaw|={(row['abs_err_yaw'] or 0):.3f}  "
            f"fell={row['fell']}"
        )

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "seconds_per_case": float(args.seconds_per_case),
        "warmup_seconds": float(args.warmup_seconds),
        "ctrl_dt": ctrl_dt,
        "num_cases": len(cases),
        "fall_rate": float(np.mean([1.0 if r["fell"] else 0.0 for r in rows])),
        "mean_abs_err_vx": float(np.mean([r["abs_err_vx"] for r in rows if r["abs_err_vx"] is not None])),
        "mean_abs_err_vy": float(np.mean([r["abs_err_vy"] for r in rows if r["abs_err_vy"] is not None])),
        "mean_abs_err_yaw": float(np.mean([r["abs_err_yaw"] for r in rows if r["abs_err_yaw"] is not None])),
        "rows": rows,
    }
    save_json(output_dir / "custom_eval.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))

    if not args.no_plots:
        maybe_plot(rows, output_dir)


if __name__ == "__main__":
    main()
