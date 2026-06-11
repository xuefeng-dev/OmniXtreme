#!/usr/bin/env python3
"""诊断增重后失败原因：力矩/速度限幅 vs 策略跟踪。"""

import importlib.util
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy_mujoco import (  # noqa: E402
    INV_PERM,
    JOINT_NAMES,
    PERM,
    Y1_list,
    Y2_list,
    X1_list,
    X2_list,
    pd_control_friction,
)
_spec = importlib.util.spec_from_file_location(
  "test_weight_limit",
  ROOT / "scripts/test_weight_limit.py",
)
_tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tw)
HeadlessDeployNode = _tw.HeadlessDeployNode


def _envelope_tau_limits(joint_vel, env):
  eps = 1e-6
  abs_dq = joint_vel.abs()
  over = torch.clamp(abs_dq - env.X1, min=0.0)
  base_pos = torch.where(
    abs_dq <= env.v_eps,
    env.Y2,
    torch.where(joint_vel >= 0.0, env.Y1, env.Y2),
  )
  slope_pos = base_pos / torch.clamp(env.X2 - env.X1, min=eps)
  tau_high = torch.clamp(base_pos - slope_pos * over, min=0.0)
  base_neg = torch.where(
    abs_dq <= env.v_eps,
    -env.Y2,
    torch.where(joint_vel >= 0.0, -env.Y2, -env.Y1),
  )
  slope_neg = (-base_neg) / torch.clamp(env.X2 - env.X1, min=eps)
  tau_low = torch.clamp(base_neg + slope_neg * over, max=0.0)
  return tau_low, tau_high


def run_diagnosis(xml_path: str, label: str, end_frame: int = 550):
  os.environ["JOINT_LOG_CSV"] = ""
  node = HeadlessDeployNode()
  for key in node.config.noise_scales:
    node.config.noise_scales[key] = 0.0

  node.env.init_mujoco_viewer(robot_xml=xml_path)
  mass = float(node.env.mj_model.body_mass.sum())

  motion0 = node.motion_res_buf[0]
  root_pos = motion0["rg_pos_t"][0][0].detach().cpu().numpy()
  root_rot_xyzw = motion0["rg_rot_t"][0][0].detach().cpu().numpy()
  node.init_motion_rot_quat = torch.tensor(root_rot_xyzw)
  root_rot_wxyz = np.array(
    [root_rot_xyzw[3], root_rot_xyzw[0], root_rot_xyzw[1], root_rot_xyzw[2]],
    dtype=np.float32,
  )
  ref_dof_pos = motion0["dof_pos"][0].detach().cpu().numpy()

  node.rootpos_offset = root_pos.copy()
  node.rootpos_offset[2] = 0
  node.env.mj_data.qpos[:3] = root_pos - node.rootpos_offset
  node.env.mj_data.qpos[2] = 0.79
  node.env.mj_data.qpos[3:7] = root_rot_wxyz
  node.env.mj_data.qpos[7:] = ref_dof_pos
  mujoco.mj_forward(node.env.mj_model, node.env.mj_data)
  node.joint_pos = torch.from_numpy(node.env.mj_data.qpos[7:].copy()).to(node.device)

  node.episode_length_buf = torch.zeros(1, device=node.device, dtype=torch.long)
  node.update_initial_yaw = True
  node.angles_last = node.angles.copy()

  # 统计窗口：后空翻前 400-550 帧
  win_start, win_end = 400, end_frame
  stats = {
    "env_clip_steps": 0,
    "tau_sat_steps": 0,
    "vel_over_steps": 0,
    "steps": 0,
    "max_tau_ratio": 0.0,
    "max_vel_ratio": 0.0,
    "max_root_err": 0.0,
    "fail_frame": None,
    "clip_joint_count": np.zeros(29),
    "tau_sat_joint_count": np.zeros(29),
  }

  mj_tau_hi = node.env.mj_model.actuator_ctrlrange[:, 1]
  mj_tau_lo = node.env.mj_model.actuator_ctrlrange[:, 0]

  with torch.no_grad():
    for step in range(end_frame):
      node.lowlevel_state_mujoco()
      if node.update_initial_yaw:
        diff = (
          R.from_quat(node.init_motion_rot_quat.cpu().numpy())
          * R.from_quat(node.quat_xyzw_r).inv()
        ).as_euler("xyz", degrees=False)
        node.initial_yaw = -diff[2]
        node.update_initial_yaw = False

      node.compute_observations()
      node.episode_length_buf += 1
      frame = int(node.episode_length_buf.item())

      a_base = node.base_policy(node.obs_dict)
      node._last_base_action_env = a_base.detach().clone()
      node.obs_dict["residual_obs"][:, -29:] = node._last_base_action_env[:, INV_PERM]
      a_res = node.residual_policy(node.obs_dict["residual_obs"].detach())[:, PERM]
      raw_actions = a_base + float(getattr(node, "residual_gain", 0.0)) * a_res

      actions_scaled = raw_actions.squeeze(0) * node.env.action_scale + node.env.default_dof_pos
      clipped = False

      if node.clip_mode == "envelope":
        eps = 1e-6
        abs_dq = node.joint_vel.abs()
        over = torch.clamp(abs_dq - node.env.X1, min=0.0)
        base_pos = torch.where(
          abs_dq <= node.env.v_eps,
          node.env.Y2,
          torch.where(node.joint_vel >= 0.0, node.env.Y1, node.env.Y2),
        )
        slope_pos = base_pos / torch.clamp(node.env.X2 - node.env.X1, min=eps)
        tau_high = torch.clamp(base_pos - slope_pos * over, min=0.0)
        base_neg = torch.where(
          abs_dq <= node.env.v_eps,
          -node.env.Y2,
          torch.where(node.joint_vel >= 0.0, -node.env.Y2, -node.env.Y1),
        )
        slope_neg = (-base_neg) / torch.clamp(node.env.X2 - node.env.X1, min=eps)
        tau_low = torch.clamp(base_neg + slope_neg * over, max=0.0)
        p_limits_low = tau_low + node.env.d_gains_tensor * node.joint_vel
        p_limits_high = tau_high + node.env.d_gains_tensor * node.joint_vel
        actions_low = (p_limits_low / node.env.p_gains_tensor) + node.joint_pos
        actions_high = (p_limits_high / node.env.p_gains_tensor) + node.joint_pos

        _as = actions_scaled.view(-1)
        clipped_mask = _as != torch.clip(_as, actions_low.view(-1), actions_high.view(-1))
        if clipped_mask.any():
          clipped = True
        angles = torch.clip(actions_scaled, actions_low, actions_high).view(-1)
      else:
        angles = actions_scaled.view(-1)

      node.angles = angles.cpu().numpy()
      node.angles_last = node.angles.copy()

      tau_cmd = pd_control_friction(
        node.angles_last,
        node.env.mj_data.qpos[7:],
        node.env.p_gains,
        np.zeros(node.env.num_actions),
        node.env.mj_data.qvel[6:],
        node.env.d_gains,
      )
      tau_env_lo, tau_env_hi = _envelope_tau_limits(node.joint_vel, node.env)
      tau_lo = np.maximum(tau_env_lo.cpu().numpy(), mj_tau_lo)
      tau_hi = np.minimum(tau_env_hi.cpu().numpy(), mj_tau_hi)

      tau_ratio = np.maximum(
        np.abs(tau_cmd) / np.maximum(np.where(tau_cmd >= 0, tau_hi, -tau_lo), 1e-6),
        0.0,
      )
      vel_ratio = (np.abs(node.env.mj_data.qvel[6:]) / np.array(X2_list)).max()

      node.env.mj_data.ctrl[:] = np.clip(tau_cmd, mj_tau_lo, mj_tau_hi)
      mujoco.mj_step(node.env.mj_model, node.env.mj_data)

      root_z = float(node.env.mj_data.qpos[2])
      ref_z = float(node.motion_res_buf[frame]["rg_pos_t"][0][0][2].item())
      root_err = abs(root_z - ref_z)
      stats["max_root_err"] = max(stats["max_root_err"], root_err)

      if win_start <= frame <= win_end:
        stats["steps"] += 1
        if clipped:
          stats["env_clip_steps"] += 1
          stats["clip_joint_count"] += clipped_mask.cpu().numpy().astype(float)
        if tau_ratio.max() > 0.95:
          stats["tau_sat_steps"] += 1
          stats["tau_sat_joint_count"] += (tau_ratio > 0.95).astype(float)
        if vel_ratio > 1.0:
          stats["vel_over_steps"] += 1
        stats["max_tau_ratio"] = max(stats["max_tau_ratio"], float(tau_ratio.max()))
        stats["max_vel_ratio"] = max(stats["max_vel_ratio"], float(vel_ratio))

      if root_err > 0.35 and stats["fail_frame"] is None:
        stats["fail_frame"] = frame

  stats["mass"] = mass
  stats["label"] = label
  return stats


def _top_joints(counts: np.ndarray, n: int = 5):
  idx = np.argsort(-counts)[:n]
  return [(JOINT_NAMES[i], int(counts[i])) for i in idx if counts[i] > 0]


def main():
  cases = [
    ("robots/g1/no_hand.xml", "baseline_33kg"),
    ("robots/g1/weight_variants/no_hand_36kg.xml", "heavy_36kg"),
  ]
  print("Diagnosis window: policy frames 400-550 (backflip phase ~462)\n")
  all_stats = []
  for xml, label in cases:
    print(f"Running {label} ...")
    all_stats.append(run_diagnosis(xml, label))

  for s in all_stats:
    steps = max(s["steps"], 1)
    print(f"\n=== {s['label']} | mass={s['mass']:.2f} kg ===")
    print(f"  fail_frame (root_err>0.35m): {s['fail_frame']}")
    print(f"  max_root_err in run: {s['max_root_err']:.3f} m")
    print(
      f"  envelope position-clip: {s['env_clip_steps']}/{steps} "
      f"({100*s['env_clip_steps']/steps:.1f}%)"
    )
    print(
      f"  PD torque near envelope (>95%): {s['tau_sat_steps']}/{steps} "
      f"({100*s['tau_sat_steps']/steps:.1f}%)"
    )
    print(
      f"  joint speed over X2: {s['vel_over_steps']}/{steps} "
      f"({100*s['vel_over_steps']/steps:.1f}%)"
    )
    print(f"  peak torque/envelope ratio: {s['max_tau_ratio']:.2f}")
    print(f"  peak |dq|/X2 ratio: {s['max_vel_ratio']:.2f}")
    print(f"  top clipped joints: {_top_joints(s['clip_joint_count'])}")
    print(f"  top torque-sat joints: {_top_joints(s['tau_sat_joint_count'])}")


if __name__ == "__main__":
  main()
