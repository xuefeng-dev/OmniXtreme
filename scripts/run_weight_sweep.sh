#!/usr/bin/env bash
# 生成体重变体模型并批量测试策略跟踪能力
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  conda activate omnixtreme
fi

export JOINT_LOG_CSV=""

echo "[1/2] Generating weight variant models..."
python scripts/generate_weight_models.py

echo "[2/2] Running headless weight limit tests (full motion)..."
python scripts/test_weight_limit.py --out-csv logs/weight_limit_test.csv "$@"
