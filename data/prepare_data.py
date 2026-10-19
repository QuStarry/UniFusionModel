"""
校准数据准备脚本

生成专业领域的无标注文本，用于适配层校准。
支持从HuggingFace数据集加载，或使用内置样本。
"""
import os
import json
import argparse
from typing import List, Dict


def generate_code_calibration_data(output_path: str, num_samples: int = 200):
    """生成代码领域校准数据（从The Stack或类似数据集）"""
    print(f"生成代码校准数据: {num_samples}条 -> {output_path}")
    
    samples = []
    try:
        from datasets import load_dataset
        # 尝试加载代码数据集
        ds = load_dataset("codeparrot/github-code", split="train", streaming=True)
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            code = item.get("code", "")
            if len(code) > 100:  # 过滤太短的
                samples.append({"text": code[:4000]})
    except Exception as e:
        print(f"  无法加载在线数据集 ({e})，使用内置代码样本")
        samples = get_builtin_code_samples(num_samples)
    
    save_jsonl(samples, output_path)
    return samples


def generate_math_calibration_data(output_path: str, num_samples: int = 200):
    """生成数学领域校准数据"""
    print(f"生成数学校准数据: {num_samples}条 -> {output_path}")
    
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("math_dataset", "algebra__linear_1d", split="train", streaming=True)
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = f"问题：{item['question']}\n解答：{item['answer']}"
            samples.append({"text": text})
    except Exception as e:
        print(f"  无法加载在线数据集 ({e})，使用内置数学样本")
        samples = get_builtin_math_samples(num_samples)
    
    save_jsonl(samples, output_path)
    return samples


def generate_medical_calibration_data(output_path: str, num_samples: int = 200):
    """生成医学领域校准数据"""
    print(f"生成医学校准数据: {num_samples}条 -> {output_path}")
    
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", streaming=True)
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = f"问题：{item['question']}\n上下文：{item['context']['contexts'][0]}\n答案：{item['long_answer']}"
            samples.append({"text": text[:4000]})
    except Exception as e:
        print(f"  无法加载在线数据集 ({e})，使用内置医学样本")
        samples = get_builtin_medical_samples(num_samples)
    
    save_jsonl(samples, output_path)
    return samples


def save_jsonl(samples: List[Dict], output_path: str):
    """保存为jsonl格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"  保存了 {len(samples)} 条样本到 {output_path}")


def get_builtin_code_samples(n: int) -> List[Dict]:
    """内置代码样本"""
    templates = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n    \n    def insert(self, value):\n        if value < self.value:\n            if self.left is None:\n                self.left = BinaryTree(value)\n            else:\n                self.left.insert(value)\n        else:\n            if self.right is None:\n                self.right = BinaryTree(value)\n            else:\n                self.right.insert(value)\n",
        "import numpy as np\n\ndef matrix_multiply(A, B):\n    return np.dot(A, B)\n\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))\n",
    ]
    return [{"text": templates[i % len(templates)]} for i in range(n)]


def get_builtin_math_samples(n: int) -> List[Dict]:
    """内置数学样本"""
    templates = [
        "问题：求解方程 2x + 5 = 15\n解答：2x = 10, x = 5",
        "问题：计算 15 * 23\n解答：15 * 23 = 345",
        "问题：一个长方形的长是8，宽是5，求面积\n解答：面积 = 长 × 宽 = 8 × 5 = 40",
        "问题：求导数 f(x) = x^3 + 2x^2 - 5x + 1\n解答：f'(x) = 3x^2 + 4x - 5",
    ]
    return [{"text": templates[i % len(templates)]} for i in range(n)]


def get_builtin_medical_samples(n: int) -> List[Dict]:
    """内置医学样本"""
    templates = [
        "患者表现为发热、咳嗽、乏力。初步诊断为上呼吸道感染。建议多饮水、休息，必要时使用退烧药。",
        "高血压患者的日常管理包括：低盐饮食、规律运动、按时服药、定期监测血压。",
        "糖尿病的典型症状包括多饮、多食、多尿和体重减轻。诊断标准为空腹血糖≥7.0mmol/L。",
    ]
    return [{"text": templates[i % len(templates)]} for i in range(n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备校准数据")
    parser.add_argument("--domain", type=str, default="code",
                       choices=["code", "math", "medical", "all"])
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()
    
    if args.domain in ["code", "all"]:
        generate_code_calibration_data(
            os.path.join(args.output_dir, "calibration_code.jsonl"),
            args.num_samples
        )
    if args.domain in ["math", "all"]:
        generate_math_calibration_data(
            os.path.join(args.output_dir, "calibration_math.jsonl"),
            args.num_samples
        )
    if args.domain in ["medical", "all"]:
        generate_medical_calibration_data(
            os.path.join(args.output_dir, "calibration_medical.jsonl"),
            args.num_samples
        )
