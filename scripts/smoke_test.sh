#!/bin/bash
# ============================================================
# 快速冒烟测试：验证代码框架能否正常运行
# 使用小批量数据和少量步骤，快速检查所有模块
# ============================================================

set -e

echo "=========================================="
echo "UniFusion 冒烟测试"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0
WORK_DIR=$(dirname "$0")
cd "$WORK_DIR"

# 用内置小样本（不需要下载大数据集）
echo "[1/3] 生成少量校准数据..."
python -c "
from unifusion.data.prepare_data import get_builtin_code_samples, save_jsonl
samples = get_builtin_code_samples(20)
save_jsonl(samples, 'data/smoke_test.jsonl')
"

echo "[2/3] 快速校准（10步）..."
python main.py calibrate \
    --large_model "Qwen/Qwen2-1.5B-Instruct" \
    --small_model "Qwen/Qwen2-1.5B-Instruct" \
    --train_data data/smoke_test.jsonl \
    --output_dir checkpoints/smoke_test \
    --max_steps 10

echo "[3/3] 快速评测（5样本）..."
python main.py evaluate \
    --large_model "Qwen/Qwen2-1.5B-Instruct" \
    --small_model "Qwen/Qwen2-1.5B-Instruct" \
    --adapter checkpoints/smoke_test/adapter_final.pt \
    --tasks gsm8k \
    --modes native,call \
    --num_samples 5 \
    --output_path logs/smoke_test_results.json

echo ""
echo "冒烟测试通过！"
