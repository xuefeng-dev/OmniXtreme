
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DISPLAY=:2

# 关节力矩/速度 CSV 输出路径（设为空则关闭记录）
export JOINT_LOG_CSV="${JOINT_LOG_CSV:-logs/joint_telemetry.csv}"

# 可选：切换体重变体模型，例如 40kg（自动 4x 力矩包络 + 4x ctrlrange）
# export XML_PATH="robots/g1/weight_variants/no_hand_40p0kg.xml"

# 力矩包络缩放（默认随模型：基准 2x，40kg 4x；可手动覆盖）
# export TORQUE_LIMIT_SCALE=4.0

python deploy_mujoco.py
