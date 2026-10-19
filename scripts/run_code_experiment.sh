#!/bin/bash
# ============================================================
# UniFusion 验证实验完整流程
# 适用环境：8卡H800，CUDA 12.1+，PyTorch 2.1+
# ============================================================

set -e

echo "=========================================="
echo "UniFusion 验证实验 - 代码专业模型融合"
echo "=========================================="

# 配置
export CUDA_VISIBLE_DEVICES=0,1
LARGE_MODEL="Qwen/Qwen2-7B-Instruct"
SMALL_MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
WORK_DIR=$(dirname "$0")
cd "$WORK_DIR"

# Step 1: 准备校准数据
echo ""
echo "[Step 1/3] 准备代码领域校准数据..."
python main.py prepare --domain code --num_samples 200 --output_dir data

# Step 2: 适配层校准训练
echo ""
echo "[Step 2/3] 适配层校准训练..."
python main.py calibrate \
    --large_model "$LARGE_MODEL" \
    --small_model "$SMALL_MODEL" \
    --train_data data/calibration_code.jsonl \
    --output_dir checkpoints/code_adapter \
    --max_steps 500

# Step 3: 评测对比
echo ""
echo "[Step 3/3] 评测对比（native vs call）..."
python main.py evaluate \
    --large_model "$LARGE_MODEL" \
    --small_model "$SMALL_MODEL" \
    --adapter checkpoints/code_adapter/adapter_final.pt \
    --tasks humaneval,gsm8k,mmlu \
    --modes native,call \
    --num_samples 50 \
    --output_path logs/eval_results.json

echo ""
echo "=========================================="
echo "实验完成！"
echo "结果查看: logs/eval_results.json"
echo "HumanEval代码: logs/humaneval_call.jsonl"
echo "=========================================="
