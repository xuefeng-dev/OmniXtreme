#!/usr/bin/env python3
"""统计各关节峰值力矩/速度，及对应时刻的另一变量。"""

import argparse
import csv
import sys
from pathlib import Path

# 关节英文名 -> 中文备注
JOINT_NAME_CN = {
    "left_hip_pitch_joint": "左髋俯仰",
    "left_hip_roll_joint": "左髋侧摆",
    "left_hip_yaw_joint": "左髋偏航",
    "left_knee_joint": "左膝",
    "left_ankle_pitch_joint": "左踝俯仰",
    "left_ankle_roll_joint": "左踝侧摆",
    "right_hip_pitch_joint": "右髋俯仰",
    "right_hip_roll_joint": "右髋侧摆",
    "right_hip_yaw_joint": "右髋偏航",
    "right_knee_joint": "右膝",
    "right_ankle_pitch_joint": "右踝俯仰",
    "right_ankle_roll_joint": "右踝侧摆",
    "waist_yaw_joint": "腰部偏航",
    "waist_roll_joint": "腰部侧摆",
    "waist_pitch_joint": "腰部俯仰",
    "left_shoulder_pitch_joint": "左肩俯仰",
    "left_shoulder_roll_joint": "左肩侧摆",
    "left_shoulder_yaw_joint": "左肩偏航",
    "left_elbow_joint": "左肘",
    "left_wrist_roll_joint": "左腕滚转",
    "left_wrist_pitch_joint": "左腕俯仰",
    "left_wrist_yaw_joint": "左腕偏航",
    "right_shoulder_pitch_joint": "右肩俯仰",
    "right_shoulder_roll_joint": "右肩侧摆",
    "right_shoulder_yaw_joint": "右肩偏航",
    "right_elbow_joint": "右肘",
    "right_wrist_roll_joint": "右腕滚转",
    "right_wrist_pitch_joint": "右腕俯仰",
    "right_wrist_yaw_joint": "右腕偏航",
}

# 输出 CSV 列名（中文）
CSV_COLUMNS = [
    ("joint", "关节"),
    ("peak_torque", "峰值力矩(N·m)"),
    ("peak_torque_time", "峰值力矩时刻(s)"),
    ("vel_at_peak_torque", "峰值力矩时速度(rad/s)"),
    ("peak_vel", "峰值速度(rad/s)"),
    ("peak_vel_time", "峰值速度时刻(s)"),
    ("torque_at_peak_vel", "峰值速度时力矩(N·m)"),
]


def _joint_label(name: str) -> str:
    cn = JOINT_NAME_CN.get(name, name)
    return f"{name}({cn})"


def _fmt3(value: float) -> float:
    """保留 3 位小数。"""
    return round(value, 3)


def _parse_joint_columns(fieldnames):
    """从 CSV 表头解析关节名与力矩/速度列。"""
    joints = []
    for name in fieldnames:
        if name.endswith("_torque"):
            joint = name[: -len("_torque")]
            vel_col = f"{joint}_vel"
            if vel_col in fieldnames:
                joints.append((joint, name, vel_col))
    return joints


def _peak_by_abs(values):
    """返回绝对值最大元素的索引。"""
    return max(range(len(values)), key=lambda i: abs(values[i]))


def analyze(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        joints = _parse_joint_columns(reader.fieldnames)
        if not joints:
            raise ValueError(
                "No joint columns found. Expected columns like "
                "<joint_name>_torque and <joint_name>_vel"
            )

        series = {
            joint: {"torque": [], "vel": [], "time": []}
            for joint, _, _ in joints
        }
        col_map = {joint: (t_col, v_col) for joint, t_col, v_col in joints}

        for row in reader:
            t = float(row["time"])
            for joint, (t_col, v_col) in col_map.items():
                series[joint]["time"].append(t)
                series[joint]["torque"].append(float(row[t_col]))
                series[joint]["vel"].append(float(row[v_col]))

    rows = []
    for joint, _, _ in joints:
        data = series[joint]
        torques = data["torque"]
        vels = data["vel"]
        times = data["time"]
        if not torques:
            continue

        peak_t_idx = _peak_by_abs(torques)
        peak_v_idx = _peak_by_abs(vels)

        rows.append({
            "joint": joint,
            "peak_torque": torques[peak_t_idx],
            "peak_torque_time": times[peak_t_idx],
            "vel_at_peak_torque": vels[peak_t_idx],
            "peak_vel": vels[peak_v_idx],
            "peak_vel_time": times[peak_v_idx],
            "torque_at_peak_vel": torques[peak_v_idx],
        })
    return rows


def _print_table(rows):
    header = (
        f"{'关节':<40} "
        f"{'峰值力矩':>10} {'力矩时速度':>10} "
        f"{'峰值速度':>10} {'速度时力矩':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        label = _joint_label(r["joint"])
        print(
            f"{label:<40} "
            f"{r['peak_torque']:10.3f} {r['vel_at_peak_torque']:10.3f} "
            f"{r['peak_vel']:10.3f} {r['torque_at_peak_vel']:10.3f}"
        )


def _write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _ in CSV_COLUMNS]
    headers = {k: cn for k, cn in CSV_COLUMNS}
    out_rows = []
    for r in rows:
        out_rows.append({
            headers["joint"]: _joint_label(r["joint"]),
            headers["peak_torque"]: _fmt3(r["peak_torque"]),
            headers["peak_torque_time"]: _fmt3(r["peak_torque_time"]),
            headers["vel_at_peak_torque"]: _fmt3(r["vel_at_peak_torque"]),
            headers["peak_vel"]: _fmt3(r["peak_vel"]),
            headers["peak_vel_time"]: _fmt3(r["peak_vel_time"]),
            headers["torque_at_peak_vel"]: _fmt3(r["torque_at_peak_vel"]),
        })
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[headers[k] for k in keys])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Summary written to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-joint peak torque/velocity from joint telemetry CSV."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Input CSV from deploy_mujoco.py (JOINT_LOG_CSV)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Optional summary CSV output path",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    rows = analyze(args.csv)
    _print_table(rows)

    out = args.output
    if out is None:
        out = args.csv.with_name(args.csv.stem + "_peaks.csv")
    _write_csv(rows, out)


if __name__ == "__main__":
    main()
