#!/bin/bash
# ============================================================
# 快速冒烟测试：验证代码框架能否正常运行
# 使用小批量数据和少量步骤，快速检查所有模块
# ============================================================

set -e

echo "=========================================="
echo "UniFusion 冒烟测试"
echo "=========================================="

# 切到项目根目录（脚本可能在 scripts/ 下被调用）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 大模型和小模型都放在同一张卡上（冒烟测试用的是同一个 1.5B 模型）
export CUDA_VISIBLE_DEVICES=0
# UniFusionModel 包需要其父目录在 PYTHONPATH 上
export PYTHONPATH="$(dirname "$PROJECT_ROOT"):$PYTHONPATH"

# 用内置小样本（不需要下载大数据集）
echo "[1/3] 生成少量校准数据..."
python -c "
from UniFusionModel.data.prepare_data import get_builtin_code_samples, save_jsonl
save_jsonl(get_builtin_code_samples(20), 'data/smoke_test.jsonl')
"

echo "[2/3] 快速校准（10步）..."
python main.py calibrate \
    --large_model "Qwen/Qwen2-1.5B-Instruct" \
    --small_model "Qwen/Qwen2-1.5B-Instruct" \
    --large_device cuda:0 \
    --small_device cuda:0 \
    --train_data data/smoke_test.jsonl \
    --output_dir checkpoints/smoke_test \
    --max_steps 10

echo "[3/3] 快速评测（5样本）..."
python main.py evaluate \
    --large_model "Qwen/Qwen2-1.5B-Instruct" \
    --small_model "Qwen/Qwen2-1.5B-Instruct" \
    --large_device cuda:0 \
    --small_device cuda:0 \
    --adapter checkpoints/smoke_test/adapter_final.pt \
    --tasks gsm8k \
    --modes native,call \
    --num_samples 5 \
    --output_path logs/smoke_test_results.json

echo ""
echo "冒烟测试通过！"
