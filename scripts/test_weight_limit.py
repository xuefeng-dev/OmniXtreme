#!/usr/bin/env python3
"""无头批量测试不同体重模型能否完成参考动作。"""

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import deploy_mujoco
from deploy_mujoco import (  # noqa: E402
    DeployNode,
    G1,
    INV_PERM,
    PERM,
    pd_control_friction,
)


@dataclass
class WeightTestResult:
    xml_path: str
    target_mass_kg: float
    actual_mass_kg: float
    completed: bool
    sim_frames: int
    total_frames: int
    completion_ratio: float
    min_root_z: float
    max_root_z_err: float
    mean_joint_pos_err: float
    max_joint_pos_err: float
    torque_clip_steps: int
    torque_clip_ratio: float
    fallen: bool
    success: bool
    fail_reason: str


class _DummyViewer:
    def sync(self):
        return None


class HeadlessG1(G1):
    """不启动 MuJoCo viewer 的 G1 环境。"""

    def init_mujoco_viewer(self, robot_xml):
        self.mj_model = mujoco.MjModel.from_xml_path(robot_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.02 / self.decimation
        self.viewer = _DummyViewer()


class HeadlessDeployNode(DeployNode):
    def init_policy(self):
        original_g1 = deploy_mujoco.G1
        deploy_mujoco.G1 = HeadlessG1
        try:
            super().init_policy()
        finally:
            deploy_mujoco.G1 = original_g1

    def __init__(self):
        super().__init__()
        # 测试模式不需要启动等待
        self.update_initial_yaw = True


def _model_mass_kg(xml_path: str) -> float:
    model = mujoco.MjModel.from_xml_path(xml_path)
    return float(model.body_mass.sum())


def _parse_target_kg(stem_suffix: str) -> float:
    raw = stem_suffix.replace("kg", "")
    return float(raw.replace("p", "."))


def _discover_models(models_dir: Path) -> List[Tuple[str, float]]:
    items = []
    for path in sorted(models_dir.glob("no_hand_*kg.xml")):
        kg = _parse_target_kg(path.stem.split("_")[-1])
        items.append((str(path), kg))
    return items


def run_single_test(
    xml_path: str,
    target_mass_kg: float,
    max_frames: Optional[int],
    max_root_tracking_err: float,
    max_clip_ratio: float,
    max_mean_joint_err: float,
) -> WeightTestResult:
    os.environ["JOINT_LOG_CSV"] = ""
    node = HeadlessDeployNode()
    node.config["xml_path"] = xml_path
    # 关闭观测噪声，保证体重扫参结果可复现
    for key in node.config.noise_scales:
        node.config.noise_scales[key] = 0.0

    # 重新加载指定模型（init_policy 已用默认 config 初始化过一次）
    node.env.init_mujoco_viewer(robot_xml=xml_path)

    motion_res_cur = node.motion_res_buf[0]
    root_pos = motion_res_cur["rg_pos_t"][0][0].detach().cpu().numpy()
    root_rot_xyzw = motion_res_cur["rg_rot_t"][0][0].detach().cpu().numpy()
    node.init_motion_rot_quat = torch.tensor(root_rot_xyzw)
    root_rot_wxyz = np.array(
        [root_rot_xyzw[3], root_rot_xyzw[0], root_rot_xyzw[1], root_rot_xyzw[2]],
        dtype=np.float32,
    )
    ref_dof_pos = motion_res_cur["dof_pos"][0].detach().cpu().numpy()

    node.rootpos_offset = root_pos[:]
    node.rootpos_offset[2] = 0
    node.env.mj_data.qpos[:3] = root_pos
    node.env.mj_data.qpos[:3] -= node.rootpos_offset
    node.env.mj_data.qpos[2] = 0.79
    node.env.mj_data.qpos[3:7] = root_rot_wxyz
    node.env.mj_data.qpos[7:] = ref_dof_pos
    mujoco.mj_forward(node.env.mj_model, node.env.mj_data)
    node.joint_pos = torch.from_numpy(node.env.mj_data.qpos[7:].copy()).to(node.device)

    total_frames = node._ref_motion_length
    # compute_observations 使用 episode_length_buf + 1，需保留 1 帧余量
    playable_frames = max(total_frames - 1, 1)
    test_frames = playable_frames if max_frames is None else min(max_frames, playable_frames)

    min_root_z = float("inf")
    max_root_z_err = 0.0
    joint_err_sum = 0.0
    max_joint_pos_err = 0.0
    torque_clip_steps = 0
    fail_reason = ""
    tracking_fail_steps = 0

    node.episode_length_buf = torch.zeros(1, device=node.device, dtype=torch.long)
    node.update_initial_yaw = True
    node.angles_last = node.angles.copy()

    with torch.no_grad():
        try:
            _ = node.base_policy(node.obs_dict)
        except Exception:
            pass
        try:
            _ = node.residual_policy(node.obs_dict["residual_obs"])
        except Exception:
            pass

        for step in range(test_frames):
            node.lowlevel_state_mujoco()
            if node.update_initial_yaw:
                from scipy.spatial.transform import Rotation as R

                diff_root_rot = (
                    R.from_quat(node.init_motion_rot_quat.cpu().numpy())
                    * R.from_quat(node.quat_xyzw_r).inv()
                ).as_euler("xyz", degrees=False)
                node.initial_yaw = -diff_root_rot[2]
                node.update_initial_yaw = False

            node.compute_observations()
            node.episode_length_buf += 1

            a_base = node.base_policy(node.obs_dict)
            node._last_base_action_env = a_base.detach().clone()
            node.obs_dict["residual_obs"][:, -29:] = node._last_base_action_env[:, INV_PERM]

            a_res = node.residual_policy(node.obs_dict["residual_obs"].detach())
            a_res = a_res[:, PERM]
            raw_actions = a_base + float(getattr(node, "residual_gain", 0.0)) * a_res

            if torch.any(torch.isnan(raw_actions)):
                fail_reason = "nan_action"
                break

            node.prev_prev_action = node.prev_action.clone()
            node.prev_action = raw_actions.squeeze(0)
            node.prev_base_action = a_base.squeeze(0)
            whole_body_action = raw_actions.squeeze(0)

            actions_scaled = whole_body_action * node.env.action_scale + node.env.default_dof_pos
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
                _al = actions_low.view(-1)
                _ah = actions_high.view(-1)
                clipped_mask = _as != torch.clip(_as, _al, _ah)
                if clipped_mask.any():
                    torque_clip_steps += 1
                angles = torch.clip(actions_scaled, actions_low, actions_high).view(-1)
            else:
                angles = actions_scaled.view(-1)

            node.angles = angles.cpu().numpy()
            action_delay_decimation = 0

            for i in range(node.env.decimation):
                if i == action_delay_decimation:
                    node.angles_last = node.angles.copy()
                tau = pd_control_friction(
                    node.angles_last,
                    node.env.mj_data.qpos[7:],
                    node.env.p_gains,
                    np.zeros(node.env.num_actions),
                    node.env.mj_data.qvel[6:],
                    node.env.d_gains,
                )
                node.env.mj_data.ctrl[:] = tau
                mujoco.mj_step(node.env.mj_model, node.env.mj_data)

            root_z = float(node.env.mj_data.qpos[2])
            min_root_z = min(min_root_z, root_z)

            motion_res_cur = node.motion_res_buf[int(node.episode_length_buf.item())]
            ref_root_z = float(motion_res_cur["rg_pos_t"][0][0][2].item())
            root_z_err = abs(root_z - ref_root_z)
            max_root_z_err = max(max_root_z_err, root_z_err)
            if root_z_err > max_root_tracking_err:
                tracking_fail_steps += 1

            ref_joint = motion_res_cur["dof_pos"][0]
            joint_err = (node.joint_pos - ref_joint).abs()
            mean_err = float(joint_err.mean().item())
            max_err = float(joint_err.max().item())
            joint_err_sum += mean_err
            max_joint_pos_err = max(max_joint_pos_err, max_err)

            if tracking_fail_steps >= 30:
                fail_reason = "root_tracking_lost"
                break

    sim_frames = int(node.episode_length_buf.item())
    completed = sim_frames >= test_frames and fail_reason == ""
    completion_ratio = sim_frames / total_frames
    mean_joint_pos_err = joint_err_sum / max(sim_frames, 1)
    torque_clip_ratio = torque_clip_steps / max(sim_frames, 1)
    fallen = tracking_fail_steps > 0 and fail_reason == "root_tracking_lost"

    if fail_reason == "":
        if not completed:
            fail_reason = "incomplete"
        elif torque_clip_ratio > max_clip_ratio:
            fail_reason = "torque_saturated"
        elif mean_joint_pos_err > max_mean_joint_err:
            fail_reason = "tracking_error"
        elif max_root_z_err > max_root_tracking_err:
            fail_reason = "root_tracking_error"

    success = completed and fail_reason == ""
    actual_mass = _model_mass_kg(xml_path)

    return WeightTestResult(
        xml_path=xml_path,
        target_mass_kg=target_mass_kg,
        actual_mass_kg=actual_mass,
        completed=completed,
        sim_frames=sim_frames,
        total_frames=total_frames,
        completion_ratio=completion_ratio,
        min_root_z=min_root_z if min_root_z != float("inf") else 0.0,
        max_root_z_err=max_root_z_err,
        mean_joint_pos_err=mean_joint_pos_err,
        max_joint_pos_err=max_joint_pos_err,
        torque_clip_steps=torque_clip_steps,
        torque_clip_ratio=torque_clip_ratio,
        fallen=fallen,
        success=success,
        fail_reason=fail_reason if fail_reason else "ok",
    )


def main():
    parser = argparse.ArgumentParser(description="Batch test weight-scaled G1 models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "robots/g1/weight_variants",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "robots/g1/no_hand.xml",
        help="Also test baseline model",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit policy steps (default: full motion)",
    )
    parser.add_argument(
        "--max-root-err",
        type=float,
        default=0.30,
        help="Max allowed |sim_root_z - ref_root_z| (m)",
    )
    parser.add_argument("--max-clip-ratio", type=float, default=0.30)
    parser.add_argument("--max-mean-joint-err", type=float, default=0.35)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "logs/weight_limit_test.csv",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Only test selected target weights (kg)",
    )
    args = parser.parse_args()

    models: List[Tuple[str, float]] = []
    if args.baseline.exists():
        models.append((str(args.baseline), 33.34))
    models.extend(_discover_models(args.models_dir))

    if args.weights is not None:
        wanted = {round(w, 1) for w in args.weights}
        models = [
            (p, kg) for p, kg in models
            if round(kg, 1) in wanted or (kg < 34 and 33.3 in wanted)
        ]

    if not models:
        print("No models found to test.")
        sys.exit(1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results: List[WeightTestResult] = []

    print(f"Testing {len(models)} models...")
    for xml_path, target_kg in models:
        print(f"\n=== {Path(xml_path).name} (target ~{target_kg:.1f} kg) ===")
        result = run_single_test(
            xml_path=xml_path,
            target_mass_kg=target_kg,
            max_frames=args.max_frames,
            max_root_tracking_err=args.max_root_err,
            max_clip_ratio=args.max_clip_ratio,
            max_mean_joint_err=args.max_mean_joint_err,
        )
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(
            f"[{status}] mass={result.actual_mass_kg:.2f}kg | "
            f"frames={result.sim_frames}/{result.total_frames} | "
            f"min_z={result.min_root_z:.3f} | "
            f"clip={result.torque_clip_ratio:.1%} | "
            f"joint_err={result.mean_joint_pos_err:.3f} | "
            f"reason={result.fail_reason}"
        )

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    passed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n========== Summary ==========")
    for r in results:
        mark = "OK" if r.success else "XX"
        print(
            f"  [{mark}] {r.actual_mass_kg:6.2f} kg -> {r.fail_reason} "
            f"(clip {r.torque_clip_ratio:.1%}, err {r.mean_joint_pos_err:.3f})"
        )

    if passed:
        print(f"\nMax passing weight: {max(r.actual_mass_kg for r in passed):.2f} kg")
    if failed:
        print(f"Min failing weight: {min(r.actual_mass_kg for r in failed):.2f} kg")


if __name__ == "__main__":
    main()
