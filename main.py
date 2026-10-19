"""
UniFusion 主入口

用法：
    # 1. 准备校准数据
    python main.py prepare --domain code --num_samples 200
    
    # 2. 校准训练
    python main.py calibrate --large_model Qwen/Qwen2.5-7B-Instruct \
        --small_model Qwen/Qwen2.5-Coder-7B-Instruct \
        --train_data data/calibration_code.jsonl --max_steps 500
    
    # 3. 评测
    python main.py evaluate --adapter checkpoints/adapter_final.pt --modes native call
"""
import argparse
import sys
import os

# 把项目根目录的父目录加入path，使 UniFusionModel 包（即本目录）可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniFusionModel.config import get_default_config
from UniFusionModel.trainers import run_calibration
from UniFusionModel.eval import run_evaluation
from UniFusionModel.data.prepare_data import (
    generate_code_calibration_data,
    generate_math_calibration_data,
    generate_medical_calibration_data,
)


def cmd_prepare(args):
    """准备校准数据"""
    output_dir = args.output_dir
    num_samples = args.num_samples
    
    if args.domain in ["code", "all"]:
        generate_code_calibration_data(
            os.path.join(output_dir, "calibration_code.jsonl"), num_samples
        )
    if args.domain in ["math", "all"]:
        generate_math_calibration_data(
            os.path.join(output_dir, "calibration_math.jsonl"), num_samples
        )
    if args.domain in ["medical", "all"]:
        generate_medical_calibration_data(
            os.path.join(output_dir, "calibration_medical.jsonl"), num_samples
        )


def cmd_calibrate(args):
    """运行校准训练"""
    config = get_default_config()
    
    # 覆盖配置
    if args.large_model:
        config.model.large_model_name = args.large_model
    if args.small_model:
        config.model.small_model_name = args.small_model
    if args.train_data:
        config.train.train_data_path = args.train_data
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.max_steps:
        config.train.max_steps = args.max_steps
    if args.large_device:
        config.model.large_model_device = args.large_device
    if args.small_device:
        config.model.small_model_device = args.small_device
    
    run_calibration(config)


def cmd_evaluate(args):
    """运行评测"""
    config = get_default_config()
    
    if args.large_model:
        config.model.large_model_name = args.large_model
    if args.small_model:
        config.model.small_model_name = args.small_model
    if args.tasks:
        config.eval.tasks = args.tasks.split(",")
    if args.num_samples:
        config.eval.num_samples = args.num_samples
    if args.output_path:
        config.eval.output_path = args.output_path
    if args.large_device:
        config.model.large_model_device = args.large_device
    if args.small_device:
        config.model.small_model_device = args.small_device
    
    modes = args.modes.split(",") if args.modes else ["native", "call"]
    
    run_evaluation(config, adapter_path=args.adapter, modes=modes)


def main():
    parser = argparse.ArgumentParser(description="UniFusion: 大模型兼容融合小模型")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # prepare
    p_prepare = subparsers.add_parser("prepare", help="准备校准数据")
    p_prepare.add_argument("--domain", type=str, default="code",
                          choices=["code", "math", "medical", "all"])
    p_prepare.add_argument("--num_samples", type=int, default=200)
    p_prepare.add_argument("--output_dir", type=str, default="data")
    
    # calibrate
    p_calibrate = subparsers.add_parser("calibrate", help="适配层校准训练")
    p_calibrate.add_argument("--large_model", type=str, default=None)
    p_calibrate.add_argument("--small_model", type=str, default=None)
    p_calibrate.add_argument("--train_data", type=str, default=None)
    p_calibrate.add_argument("--output_dir", type=str, default=None)
    p_calibrate.add_argument("--max_steps", type=int, default=None)
    p_calibrate.add_argument("--large_device", type=str, default=None, help="大模型设备，如 cuda:0")
    p_calibrate.add_argument("--small_device", type=str, default=None, help="小模型设备，如 cuda:1")
    
    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="评测")
    p_eval.add_argument("--adapter", type=str, default=None, help="适配层权重路径")
    p_eval.add_argument("--large_model", type=str, default=None)
    p_eval.add_argument("--small_model", type=str, default=None)
    p_eval.add_argument("--tasks", type=str, default="humaneval,gsm8k,mmlu",
                       help="逗号分隔的任务列表")
    p_eval.add_argument("--modes", type=str, default="native,call",
                       help="逗号分隔的模式列表")
    p_eval.add_argument("--num_samples", type=int, default=None)
    p_eval.add_argument("--output_path", type=str, default=None)
    p_eval.add_argument("--large_device", type=str, default=None, help="大模型设备，如 cuda:0")
    p_eval.add_argument("--small_device", type=str, default=None, help="小模型设备，如 cuda:1")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
