#!/usr/bin/env python3
"""按目标总重生成 G1 MuJoCo 模型变体（等比例缩放质量与惯性）。"""

import argparse
import re
from pathlib import Path

BASE_MASS_KG = 33.3411
SRC_XML = Path("robots/g1/no_hand.xml")
OUT_DIR = Path("robots/g1/weight_variants")


def _scale_diaginertia(match: re.Match, scale: float) -> str:
    vals = [float(v) * scale for v in match.group(1).split()]
    return f'diaginertia="{" ".join(f"{v:.9g}" for v in vals)}"'


def scale_ctrlrange(text: str, factor: float) -> str:
    """缩放 actuator ctrlrange（用于特定体重额外放宽力矩硬限）。"""
    def repl(match: re.Match) -> str:
        lo = float(match.group(1)) * factor
        hi = float(match.group(2)) * factor
        return f'ctrlrange="{lo:g} {hi:g}"'

    return re.sub(r'ctrlrange="([-\d.]+) ([-\d.]+)"', repl, text)


def scale_model_xml(text: str, scale: float, target_kg: float) -> str:
    """等比例缩放所有刚体质量与转动惯量。"""
    # 变体文件放在子目录，mesh 路径需指向原始资源目录
    text = text.replace(
        'meshdir="./meshes/g1"',
        'meshdir="../meshes/g1"',
        1,
    )
    text = re.sub(
        r'mass="([0-9.eE+-]+)"',
        lambda m: f'mass="{float(m.group(1)) * scale:.6f}"',
        text,
    )
    text = re.sub(
        r'diaginertia="([0-9.eE+\-.\s]+)"',
        lambda m: _scale_diaginertia(m, scale),
        text,
    )
    header = (
        f"<!-- Auto-generated weight variant: target={target_kg:.1f} kg, "
        f"scale={scale:.4f} from base {BASE_MASS_KG:.4f} kg -->\n"
    )
    if text.startswith("<mujoco"):
        text = text.replace("<mujoco", header + "<mujoco", 1)
    return text


def verify_mass(text: str) -> float:
    masses = [float(m) for m in re.findall(r'mass="([0-9.eE+-]+)"', text)]
    return sum(masses)


def main():
    parser = argparse.ArgumentParser(description="Generate G1 weight variant XML files")
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[34, 35, 36, 37, 38, 39, 40, 45, 50, 55, 60, 65, 70],
        help="Target total mass in kg",
    )
    args = parser.parse_args()

    src_text = SRC_XML.read_text(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for target in sorted(args.targets):
        scale = target / BASE_MASS_KG
        out_text = scale_model_xml(src_text, scale, target)
        extra_ctrl_scale = 1.0
        if abs(target - 40.0) < 0.05:
            # 40kg 在全局 2x 基础上再 x2 => 相对原始 4x
            out_text = scale_ctrlrange(out_text, 2.0)
            extra_ctrl_scale = 2.0
        actual = verify_mass(out_text)
        tag = f"{target:.1f}".replace(".", "p")
        out_path = OUT_DIR / f"no_hand_{tag}kg.xml"
        out_path.write_text(out_text, encoding="utf-8")
        print(
            f"Wrote {out_path} | target={target:.1f} kg | "
            f"actual={actual:.4f} kg | mass_scale={scale:.4f} | "
            f"ctrl_extra={extra_ctrl_scale:.1f}x"
        )


if __name__ == "__main__":
    main()
