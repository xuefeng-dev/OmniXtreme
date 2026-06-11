
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DISPLAY=:2

# 关节力矩/速度 CSV 输出路径（设为空则关闭记录）
export JOINT_LOG_CSV="${JOINT_LOG_CSV:-logs/joint_telemetry.csv}"

python deploy_mujoco.py
