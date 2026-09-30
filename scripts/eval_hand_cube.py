"""Evaluate trained hand-cube policies under fixed conditions.

All policies run in the same environment config with the same seed. So the
first episode of each env starts from the same state for every policy. Only the
first episode of each env counts. The play config is used: observation noise is
off and the episode has no time limit, so the script stops after one training
episode length. The policy uses its mean action. Contact friction can be fixed
to one value. All other domain randomization stays on.

This copy is for mjlab v1.6.0 and rsl_rl 5. It loads only the actor weights.
A checkpoint from rsl_rl 4 (mjlab v1.1.1) also works: its noise parameter
"std" is renamed to the rsl_rl 5 name "distribution.std_param".

Usage, inside the container from the repo root:
  uv run python scripts/eval_hand_cube.py \
    --run baseline=logs/rsl_rl/<experiment>/<run>/model_4999.pt \
    --run softgate=logs/rsl_rl/<experiment>/<run>/model_4999.pt \
    --friction default 0.3 0.5 1.0 1.6 2.0 \
    --out-dir /workspace/robot-hand/eval

"default" keeps the training friction randomization of the task.

Outputs in --out-dir:
  summary.csv, summary.md   one row per run and condition
  <run>_<condition>.npz     per-env values
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi
from mjlab.utils.torch import configure_torch_backends

METRICS = ("rotation_progress", "position_error", "tilt_error", "fingertip_contact_fraction")
# rsl_rl 4 actor keys and their rsl_rl 5 names.
RSL_RL4_ACTOR_KEYS = {"std": "distribution.std_param", "log_std": "distribution.log_std_param"}


def load_actor(runner, checkpoint: str, device: str) -> bool:
  """Load only the actor weights. Return True if rsl_rl 4 keys were renamed."""
  ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
  actor = ckpt["actor_state_dict"]
  converted = False
  for old_key, new_key in RSL_RL4_ACTOR_KEYS.items():
    if old_key in actor and new_key not in actor:
      actor[new_key] = actor.pop(old_key)
      converted = True
  runner.alg.load(ckpt, {"actor": True}, strict=True)
  return converted


def prepare_agent_cfg(agent_cfg) -> dict:
  """Same as scripts/play.py: drop CNN fields for MLP models."""
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict) and model_cfg.get("class_name", "MLPModel") != "CNNModel":
      model_cfg.pop("cnn_cfg", None)
  return cfg_dict


def episode_steps(task: str) -> int:
  """Number of control steps in one training episode."""
  cfg = load_env_cfg(task, play=False)
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  return int(round(cfg.episode_length_s / step_dt))


def make_env_cfg(args, condition: str):
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  if condition != "default":
    if args.friction_event not in cfg.events:
      raise SystemExit(f"error: event '{args.friction_event}' not found in {args.task}")
    value = float(condition)
    cfg.events[args.friction_event].params[args.friction_param] = (value, value)
  return cfg


def evaluate(args, name: str, checkpoint: str, condition: str, steps: int) -> dict:
  device = args.device
  env_cfg = make_env_cfg(args, condition)
  rot_params = env_cfg.rewards[args.rotation_term].params
  sign = -1.0 if rot_params.get("negate_yaw_rate", False) else 1.0
  object_name = rot_params["object_name"]

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  agent_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, prepare_agent_cfg(agent_cfg), device=device)
  converted = load_actor(runner, checkpoint, device)
  policy = runner.get_inference_policy(device=device)

  n = env.num_envs
  cube = env.scene[object_name]
  robot = env.scene[args.robot]
  term_names = list(env.termination_manager.active_terms)
  metric_names = list(env.metrics_manager.active_terms)
  metric_idx = [metric_names.index(m) for m in METRICS]
  clip = float(agent_cfg.clip_actions)

  with torch.inference_mode():
    obs, _ = wrapped.reset()
    init_state = obs["critic"].clone()
    prev_yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)[2]

    alive = torch.ones(n, dtype=torch.bool, device=device)
    reason = torch.full((n,), -1, dtype=torch.long, device=device)
    ep_len = torch.full((n,), steps, dtype=torch.long, device=device)
    metric_sum = torch.zeros((n, len(METRICS)), device=device)
    metric_steps = torch.zeros(n, device=device)
    yaw_sum = torch.zeros(n, device=device)
    state_steps = torch.zeros(n, device=device)
    torque_sum = torch.zeros(n, device=device)
    power_sum = torch.zeros(n, device=device)
    sat_sum = torch.zeros(n, device=device)

    for t in range(steps):
      actions = policy(obs)
      sat = (actions.abs() > clip).float().mean(dim=1)
      obs, _, dones, _ = wrapped.step(actions)
      done = dones.bool()
      live = alive.float()

      # Metrics and terminations of this step are computed before any reset.
      metric_sum += env.metrics_manager._step_values[:, metric_idx] * live[:, None]
      metric_steps += live
      sat_sum += sat * live

      # Envs that terminated now were already reset, so their cube pose and
      # joint state belong to a new episode. Skip their state for this step.
      yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)[2]
      full = (alive & ~done).float()
      yaw_sum += sign * wrap_to_pi(yaw - prev_yaw) * full
      prev_yaw = yaw
      tau = robot.data.actuator_force
      qd = robot.data.joint_vel
      torque_sum += tau.abs().mean(dim=1) * full
      power_sum += (tau * qd).abs().sum(dim=1) * full
      state_steps += full

      new_done = alive & done
      if bool(new_done.any()):
        for k, term in enumerate(term_names):
          hit = new_done & env.termination_manager.get_term(term) & (reason < 0)
          reason[hit] = k
        ep_len[new_done] = t + 1
      alive &= ~done

  step_dt = env.step_dt
  safe_state = state_steps.clamp(min=1.0)
  per_env = {
    "survived": (reason < 0).cpu().numpy(),
    "reason": reason.cpu().numpy(),
    "episode_steps": ep_len.cpu().numpy(),
    "yaw_rad": yaw_sum.cpu().numpy(),
    "yaw_rate": (yaw_sum / (safe_state * step_dt)).cpu().numpy(),
    "mean_abs_torque": (torque_sum / safe_state).cpu().numpy(),
    "mean_power": (power_sum / safe_state).cpu().numpy(),
    "action_saturation": (sat_sum / metric_steps.clamp(min=1.0)).cpu().numpy(),
    "init_state": init_state.cpu().numpy(),
  }
  metric_mean = (metric_sum / metric_steps.clamp(min=1.0)[:, None]).cpu().numpy()
  for i, m in enumerate(METRICS):
    per_env[m] = metric_mean[:, i]

  out = Path(args.out_dir)
  np.savez(out / f"{name}_{condition}.npz", term_names=np.array(term_names), **per_env)

  env.close()
  del policy, runner, wrapped, env
  gc.collect()
  torch.cuda.empty_cache()
  return {
    "per_env": per_env,
    "term_names": term_names,
    "step_dt": step_dt,
    "rsl_rl4_checkpoint": converted,
  }


def summarize(name: str, condition: str, result: dict, reference) -> dict:
  e = result["per_env"]
  n = len(e["survived"])
  survived = e["survived"]
  row = {
    "run": name,
    "condition": condition,
    "envs": n,
    "survival_rate": survived.mean(),
    "mean_episode_steps": e["episode_steps"].mean(),
    "yaw_rate_rad_s": e["yaw_rate"].mean(),
    "yaw_rate_se": e["yaw_rate"].std(ddof=1) / math.sqrt(n),
    "turns_survivors": (e["yaw_rad"][survived] / (2 * math.pi)).mean() if survived.any() else float("nan"),
  }
  for k, term in enumerate(result["term_names"]):
    row[f"fail_{term}"] = (e["reason"] == k).mean()
  for m in METRICS:
    row[m] = e[m].mean()
  row["mean_abs_torque_nm"] = e["mean_abs_torque"].mean()
  row["mean_power_w"] = e["mean_power"].mean()
  row["action_saturation"] = e["action_saturation"].mean()
  row["rsl_rl4_checkpoint"] = result["rsl_rl4_checkpoint"]
  row["init_state_max_diff"] = (
    float(np.abs(e["init_state"] - reference).max()) if reference is not None else 0.0
  )
  return row


def write_outputs(rows: list[dict], out: Path) -> None:
  keys = list(rows[0].keys())
  for row in rows:
    keys += [k for k in row if k not in keys]
  with open(out / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
  cols = [
    ("run", "run", "{}"),
    ("condition", "friction", "{}"),
    ("yaw_rate_rad_s", "yaw rate rad/s", "{:.3f}"),
    ("survival_rate", "survival", "{:.3f}"),
    ("rotation_progress", "rotation_progress", "{:.3f}"),
    ("position_error", "pos err m", "{:.4f}"),
    ("tilt_error", "tilt err rad", "{:.3f}"),
    ("fingertip_contact_fraction", "contact", "{:.3f}"),
    ("mean_abs_torque_nm", "torque Nm", "{:.3f}"),
    ("action_saturation", "action sat", "{:.3f}"),
    ("init_state_max_diff", "init diff", "{:.2e}"),
  ]
  lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
  for row in rows:
    lines.append("| " + " | ".join(fmt.format(row[key]) for key, _, fmt in cols) + " |")
  (out / "summary.md").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--run", action="append", required=True, metavar="NAME=CHECKPOINT")
  parser.add_argument("--task", default="Mjlab-Leap-Left-HandCube-Rotate")
  parser.add_argument("--friction", nargs="+", default=["default"])
  parser.add_argument("--friction-event", default="dr_shared_contact_friction")
  parser.add_argument("--friction-param", default="friction_range")
  parser.add_argument("--rotation-term", default="rotate_finite_diff")
  parser.add_argument("--robot", default="robot")
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--steps", type=int, default=None, help="default: one training episode")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--out-dir", required=True)
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  configure_torch_backends()
  runs = [r.split("=", 1) for r in args.run]
  steps = args.steps or episode_steps(args.task)
  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  print(f"[eval] {len(runs)} runs x {len(args.friction)} conditions, "
        f"{args.num_envs} envs, {steps} steps, seed {args.seed}")

  rows = []
  for condition in args.friction:
    reference = None
    for name, checkpoint in runs:
      print(f"[eval] run={name} condition={condition}")
      result = evaluate(args, name, checkpoint, condition, steps)
      row = summarize(name, condition, result, reference)
      if reference is None:
        reference = result["per_env"]["init_state"]
      rows.append(row)
      write_outputs(rows, out)


if __name__ == "__main__":
  main()
