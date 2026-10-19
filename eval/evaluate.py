"""
评测脚本

支持任务：
- humaneval: 代码生成（Pass@1）
- gsm8k: 数学推理（准确率）
- mmlu: 通用能力（防退化检测）

支持模式：
- native: 纯大模型
- call: UniFusion调用模式
- text_tool: 文本级Tool Use基线（大模型生成prompt调用小模型）
"""
import os
import json
import re
import torch
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import UniFusionConfig
from ..models import UniFusionModel


class Evaluator:
    """通用评测器"""
    
    def __init__(self, config: UniFusionConfig, model: UniFusionModel):
        self.config = config
        self.model = model
        self.results = {}
    
    @torch.no_grad()
    def generate_response(self, prompt: str, mode: str = "call",
                          max_new_tokens: int = 512) -> str:
        """生成单个回复"""
        tokenizer = self.model.large_tokenizer
        device = self.model.large_model.device
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode=mode,
            max_new_tokens=max_new_tokens,
            temperature=self.config.eval.temperature,
            top_p=self.config.eval.top_p,
        )
        
        # 只取新生成的部分
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response
    
    def evaluate_humaneval(self, num_samples: Optional[int] = None,
                           mode: str = "call") -> Dict:
        """
        HumanEval代码生成评测
        
        注意：这里只做生成，代码执行检查需要配合human-eval工具包
        生成的结果保存为jsonl，可用官方脚本计算pass@1
        """
        print(f"\n[HumanEval] 模式={mode}, 样本数={num_samples or 'all'}")
        
        # 尝试加载HumanEval数据集
        # 如果没有安装datasets，用内置的少量样本
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai/openai_humaneval", split="test")
            problems = list(dataset)
        except Exception as e:
            print(f"  警告: 无法加载HumanEval数据集 ({e})，使用内置样本")
            problems = self._get_builtin_humaneval_samples()
        
        if num_samples:
            problems = problems[:num_samples]
        
        results = []
        for prob in tqdm(problems, desc="HumanEval"):
            task_id = prob.get("task_id", prob.get("entry_point", "unknown"))
            prompt = prob["prompt"]
            
            # 构造输入
            full_prompt = f"请完成以下Python函数：\n\n{prompt}\n\n"
            response = self.generate_response(full_prompt, mode=mode, max_new_tokens=512)
            
            # 提取函数体（简单启发式）
            completion = self._extract_code_completion(prompt, response)
            
            results.append({
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
                "raw_response": response,
            })
        
        # 保存结果
        output_path = os.path.join(
            os.path.dirname(self.config.eval.output_path),
            f"humaneval_{mode}.jsonl"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        print(f"  生成完成，结果保存到 {output_path}")
        print(f"  提示: 使用官方脚本计算pass@1:")
        print(f"    pip install human-eval")
        print(f"    evaluate_functional_correctness {output_path}")
        
        return {"task": "humaneval", "mode": mode, "num_samples": len(results),
                "output_path": output_path, "note": "需用官方脚本执行代码检查"}
    
    def evaluate_gsm8k(self, num_samples: Optional[int] = None,
                       mode: str = "call") -> Dict:
        """GSM8K数学推理评测"""
        print(f"\n[GSM8K] 模式={mode}, 样本数={num_samples or 'all'}")
        
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="test")
            problems = list(dataset)
        except Exception as e:
            print(f"  警告: 无法加载GSM8K数据集 ({e})，使用内置样本")
            problems = self._get_builtin_gsm8k_samples()
        
        if num_samples:
            problems = problems[:num_samples]
        
        correct = 0
        total = 0
        results = []
        
        for prob in tqdm(problems, desc="GSM8K"):
            question = prob["question"]
            answer = prob["answer"]
            
            # 提取标准答案中的数字
            gold_num = self._extract_number(answer)
            
            prompt = f"请解答以下数学题，给出详细推理过程，最后用'答案：'标注最终数字。\n\n问题：{question}\n\n解答："
            response = self.generate_response(prompt, mode=mode, max_new_tokens=512)
            
            # 提取模型答案中的数字
            pred_num = self._extract_number(response)
            
            is_correct = (gold_num is not None and pred_num is not None 
                         and abs(gold_num - pred_num) < 1e-4)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": question,
                "gold": gold_num,
                "pred": pred_num,
                "correct": is_correct,
                "response": response[:500],
            })
        
        accuracy = correct / total if total > 0 else 0
        print(f"  准确率: {accuracy:.4f} ({correct}/{total})")
        
        return {"task": "gsm8k", "mode": mode, "accuracy": accuracy,
                "correct": correct, "total": total}
    
    def evaluate_mmlu(self, num_samples: Optional[int] = None,
                      mode: str = "call", subjects: Optional[List[str]] = None) -> Dict:
        """MMLU通用能力评测（防退化检测）"""
        print(f"\n[MMLU] 模式={mode}, 样本数={num_samples or 'all'}")
        
        try:
            from datasets import load_dataset
            dataset = load_dataset("cais/mmlu", "all", split="test")
            problems = list(dataset)
        except Exception as e:
            print(f"  警告: 无法加载MMLU数据集 ({e})，使用内置样本")
            problems = self._get_builtin_mmlu_samples()
        
        if subjects:
            problems = [p for p in problems if p.get("subject") in subjects]
        if num_samples:
            problems = problems[:num_samples]
        
        correct = 0
        total = 0
        
        for prob in tqdm(problems, desc="MMLU"):
            question = prob["question"]
            choices = prob["choices"]
            answer_idx = prob["answer"]
            
            # 构造选择题prompt
            choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            prompt = f"请选择正确答案（只输出字母）。\n\n问题：{question}\n{choices_str}\n\n答案："
            
            response = self.generate_response(prompt, mode=mode, max_new_tokens=10)
            
            # 提取答案字母
            pred_idx = self._extract_choice(response)
            if pred_idx == answer_idx:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"  准确率: {accuracy:.4f} ({correct}/{total})")
        
        return {"task": "mmlu", "mode": mode, "accuracy": accuracy,
                "correct": correct, "total": total}
    
    def evaluate_all(self, modes: List[str] = ["native", "call"]) -> Dict:
        """运行所有评测任务"""
        all_results = {}
        
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"评测模式: {mode}")
            print(f"{'='*60}")
            
            mode_results = {}
            
            if "humaneval" in self.config.eval.tasks:
                mode_results["humaneval"] = self.evaluate_humaneval(
                    num_samples=self.config.eval.num_samples, mode=mode
                )
            
            if "gsm8k" in self.config.eval.tasks:
                mode_results["gsm8k"] = self.evaluate_gsm8k(
                    num_samples=self.config.eval.num_samples, mode=mode
                )
            
            if "mmlu" in self.config.eval.tasks:
                mode_results["mmlu"] = self.evaluate_mmlu(
                    num_samples=self.config.eval.num_samples, mode=mode
                )
            
            all_results[mode] = mode_results
        
        # 保存汇总结果
        output_dir = os.path.dirname(self.config.eval.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.config.eval.output_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 打印对比表
        self._print_comparison(all_results)
        
        return all_results
    
    def _print_comparison(self, all_results: Dict):
        """打印模式对比表"""
        print("\n" + "=" * 60)
        print("评测结果对比")
        print("=" * 60)
        
        # 收集所有任务
        all_tasks = set()
        for mode_results in all_results.values():
            all_tasks.update(mode_results.keys())
        
        # 打印表头
        modes = list(all_results.keys())
        header = f"{'任务':<15}" + "".join(f"{m:<15}" for m in modes)
        print(header)
        print("-" * len(header))
        
        for task in sorted(all_tasks):
            row = f"{task:<15}"
            for mode in modes:
                if task in all_results[mode]:
                    r = all_results[mode][task]
                    if "accuracy" in r:
                        row += f"{r['accuracy']:.4f}         "
                    else:
                        row += f"生成完成        "
                else:
                    row += f"-              "
            print(row)
    
    # ===== 辅助函数 =====
    
    def _extract_code_completion(self, prompt: str, response: str) -> str:
        """从模型回复中提取代码补全"""
        # 简单策略：取第一个代码块，或者直接取回复
        code_block_pattern = r"```(?:python)?\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0]
        
        # 如果没有代码块，取回复中看起来像代码的部分
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class ') or in_code:
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip() == '' and len(code_lines) > 3:
                break
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return response
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取最后一个数字（通常是答案）"""
        # 优先找"答案："后面的数字
        answer_pattern = r"答案[：:]\s*([-+]?\d*\.?\d+)"
        matches = re.findall(answer_pattern, text)
        if matches:
            try:
                return float(matches[-1].replace(',', ''))
            except ValueError:
                pass
        
        # 否则找所有数字，取最后一个
        numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(',', ''))
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None
    
    def _extract_choice(self, text: str) -> Optional[int]:
        """从回复中提取选择题答案字母"""
        text = text.strip().upper()
        if text and text[0] in 'ABCD':
            return ord(text[0]) - ord('A')
        # 找"A."或"A)"等模式
        match = re.search(r'([ABCD])[\.\)\s]', text)
        if match:
            return ord(match.group(1)) - ord('A')
        return None
    
    def _get_builtin_humaneval_samples(self) -> List[Dict]:
        """内置的少量HumanEval样本（用于无网络环境测试）"""
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "entry_point": "has_close_elements",
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "entry_point": "separate_paren_groups",
            },
        ]
    
    def _get_builtin_gsm8k_samples(self) -> List[Dict]:
        """内置的少量GSM8K样本"""
        return [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 each. How much does she make every day at the farmers' market?",
                "answer": "She sells 16 - 3 - 4 = 9 eggs daily. So she makes 9 * 2 = 18 dollars. #### 18",
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "It takes 2/2=1 bolt of white fiber. So total is 2+1=3 bolts. #### 3",
            },
        ]
    
    def _get_builtin_mmlu_samples(self) -> List[Dict]:
        """内置的少量MMLU样本"""
        return [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": 1,
                "subject": "geography",
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "answer": 1,
                "subject": "astronomy",
            },
        ]


def run_evaluation(config: UniFusionConfig, adapter_path: Optional[str] = None,
                   modes: List[str] = ["native", "call"]):
    """运行完整评测流程"""
    print("=" * 60)
    print("UniFusion 评测")
    print("=" * 60)
    
    # 加载大模型
    print("\n[1/3] 加载大模型...")
    large_tokenizer = AutoTokenizer.from_pretrained(config.model.large_model_name)
    large_model = AutoModelForCausalLM.from_pretrained(
        config.model.large_model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.model.large_model_device,
    )
    if large_tokenizer.pad_token is None:
        large_tokenizer.pad_token = large_tokenizer.eos_token
    
    # 加载小模型
    print("\n[2/3] 加载专业小模型...")
    small_tokenizer = AutoTokenizer.from_pretrained(config.model.small_model_name)
    small_model = AutoModelForCausalLM.from_pretrained(
        config.model.small_model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.model.small_model_device,
    )
    if small_tokenizer.pad_token is None:
        small_tokenizer.pad_token = small_tokenizer.eos_token
    
    # 构建模型
    print("\n[3/3] 构建UniFusion模型...")
    model = UniFusionModel(config)
    model.set_models(large_model, small_model, large_tokenizer, small_tokenizer)
    # 评测必须走 eval 模式：否则适配层/门控/重采样器的 dropout 会在推理时持续生效，
    # 导致结果随机抖动（nn.Module 默认 training=True）。
    model.eval()
    
    # 加载适配层
    if adapter_path and os.path.exists(adapter_path):
        model.load_adapter(adapter_path)
        print(f"  适配层已加载: {adapter_path}")
    else:
        print("  警告: 未加载适配层，使用随机初始化（仅用于测试流程）")
    
    # 评测
    evaluator = Evaluator(config, model)
    results = evaluator.evaluate_all(modes=modes)
    
    print(f"\n评测完成，结果保存到 {config.eval.output_path}")
    return results


if __name__ == "__main__":
    from ..config import get_default_config
    config = get_default_config()
    run_evaluation(config)
