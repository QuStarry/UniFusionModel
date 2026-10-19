"""
适配层校准训练脚本

训练目标（自监督，不需要任务标注）：
1. P_in校准：大模型hook层隐状态 → 投影 → 逼近小模型同层的真实隐状态
2. P_out校准：循环一致性 P_out(P_in(h_large)) ≈ h_large（逐位置强约束），
   外加小模型输出投影后的均值/方差分布对齐
3. 端到端校准：融合后的模型在专业文本上的LM loss尽可能低
4. 原生保持：融合后logits与原生日志的KL正则（标准KD方向），防遗忘

校准数据：专业领域的无标注文本（如代码、医学文献），50-200条即可
"""
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Dict, Optional
from tqdm import tqdm

from ..config import UniFusionConfig
from ..models import UniFusionModel
from ..utils.hooks import HiddenStateHook, get_layers


class CalibrationDataset(Dataset):
    """校准数据集：专业领域的无标注文本"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = item.get("text", item.get("content", ""))
                    if text:
                        self.samples.append(text)
        
        print(f"[CalibrationDataset] 加载了 {len(self.samples)} 条校准样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class AdapterCalibrator:
    """适配层校准器"""
    
    def __init__(self, config: UniFusionConfig, model: UniFusionModel):
        self.config = config
        self.model = model
        self.train_config = config.train
        
        # 优化器只更新适配层和门控
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.train_config.warmup_steps,
            num_training_steps=self.train_config.max_steps,
        )
        
        self.global_step = 0
        self.loss_history = []
    
    def _compute_calibration_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算校准损失
        
        包含四个部分：
        1. hidden_mse: P_in逐位置对齐 + P_out的均值/方差分布对齐
        2. cycle: 循环一致性 P_out(P_in(h_large)) ≈ h_large（逐位置强约束）
        3. lm_loss: 融合后模型的端到端语言建模损失
        4. native_keep: 保持大模型原生输出能力的KL正则
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        losses = {}
        
        # ===== 损失1: 隐状态MSE对齐 =====
        # 分别跑大模型和小模型的完整前向，提取对应层的隐状态
        with torch.no_grad():
            # 大模型完整前向，提取hook层隐状态
            large_layers = get_layers(self.model.large_model)
            large_hook = HiddenStateHook(large_layers[self.model._large_hook_layer_idx])
            large_hook.register()
            self.model.large_model(input_ids=input_ids, attention_mask=attention_mask)
            h_large_hook = large_hook.output.detach()
            large_hook.remove()
            
            # 小模型完整前向，提取start层和output层隐状态
            small_layers = get_layers(self.model.small_model)
            small_start_hook = HiddenStateHook(small_layers[self.model._small_start_layer_idx])
            small_output_hook = HiddenStateHook(small_layers[self.model._small_output_layer_idx])
            small_start_hook.register()
            small_output_hook.register()
            
            small_input_ids = input_ids.to(self.model.small_model.device)
            small_attention_mask = attention_mask.to(self.model.small_model.device)
            self.model.small_model(input_ids=small_input_ids, attention_mask=small_attention_mask)
            
            h_small_start = small_start_hook.output.detach().to(input_ids.device)
            h_small_output = small_output_hook.output.detach().to(input_ids.device)
            small_start_hook.remove()
            small_output_hook.remove()
        
        # P_in对齐：P_in(h_large_hook) 应该接近 h_small_start
        # 主干模型是 bf16、适配层是 fp32，进适配层前必须先转 fp32
        h_large_f = h_large_hook.float()
        h_large_projected = self.model.adapter.forward_large_to_small(h_large_f)
        loss_p_in = F.mse_loss(h_large_projected, h_small_start.float())
        
        # P_out对齐(1)：分布约束——P_out(h_small_output) 的均值/方差应接近大模型hook层隐状态
        h_small_projected = self.model.adapter.forward_small_to_large(h_small_output.float())
        loss_p_out = F.mse_loss(
            h_small_projected.mean(dim=(0, 1)), h_large_f.mean(dim=(0, 1))
        ) + F.mse_loss(
            h_small_projected.std(dim=(0, 1)), h_large_f.std(dim=(0, 1))
        )
        
        losses["hidden_mse"] = self.train_config.hidden_mse_weight * (loss_p_in + loss_p_out)
        
        # P_out对齐(2)：循环一致性——P_out(P_in(h_large)) 应能还原 h_large。
        # 分布约束只是序列级统计量，这里补一条逐位置的强约束（P_in/P_out 互为逆映射）。
        h_cycle = self.model.adapter.forward_small_to_large(h_large_projected)
        losses["cycle"] = self.train_config.cycle_weight * F.mse_loss(h_cycle, h_large_f)
        
        # ===== 损失2: 端到端LM loss（融合后的模型应该能流畅生成专业文本）=====
        # 用input_ids作为labels（自回归）
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        outputs_call = self.model.forward_call(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        losses["lm_loss"] = self.train_config.lm_weight * outputs_call.loss
        
        # ===== 损失3: 原生输出保持（防止门控完全偏向小模型）=====
        with torch.no_grad():
            outputs_native = self.model.forward_native(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            native_logits = outputs_native.logits
        
        # 融合后的logits不应与原生logits相差太远（防遗忘）。
        # 方向为 KL(p_native ‖ q_fused)，即标准知识蒸馏的 teacher→student 方向。
        loss_kl = F.kl_div(
            F.log_softmax(outputs_call.logits, dim=-1),
            F.log_softmax(native_logits, dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        losses["native_keep"] = self.train_config.native_keep_weight * loss_kl
        
        # 总损失
        total_loss = sum(losses.values())
        losses["total"] = total_loss
        
        return losses
    
    def train(self, dataloader: DataLoader):
        """执行校准训练"""
        self.model.adapter.train()
        self.model.gating.train()
        
        progress_bar = tqdm(range(self.train_config.max_steps), desc="Calibrating")
        data_iter = iter(dataloader)
        
        for step in progress_bar:
            # 获取batch（循环使用数据）
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            batch = {k: v.to(self.model.large_model.device) for k, v in batch.items()}
            
            # 前向
            losses = self._compute_calibration_loss(batch)
            total_loss = losses["total"] / self.train_config.gradient_accumulation_steps
            
            # 反向
            total_loss.backward()
            
            if (step + 1) % self.train_config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    self.train_config.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # 日志
            self.global_step += 1
            self.loss_history.append({k: v.item() for k, v in losses.items()})
            
            if (step + 1) % self.train_config.logging_steps == 0:
                avg_losses = {
                    k: sum(l[k] for l in self.loss_history[-self.train_config.logging_steps:]) 
                       / min(self.train_config.logging_steps, len(self.loss_history))
                    for k in self.loss_history[0]
                }
                gate_val = self.model.gating.get_mean_gate_value()
                progress_bar.set_postfix({
                    "loss": f"{avg_losses['total']:.4f}",
                    "lm": f"{avg_losses.get('lm_loss', 0):.4f}",
                    "mse": f"{avg_losses.get('hidden_mse', 0):.4f}",
                    "cyc": f"{avg_losses.get('cycle', 0):.4f}",
                    "gate": f"{gate_val:.3f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })
            
            # 保存
            if (step + 1) % self.train_config.save_steps == 0:
                save_path = os.path.join(
                    self.train_config.output_dir,
                    f"adapter_step_{self.global_step}.pt"
                )
                self.model.save_adapter(save_path)
        
        # 最终保存
        final_path = os.path.join(self.train_config.output_dir, "adapter_final.pt")
        self.model.save_adapter(final_path)
        
        # 保存loss曲线
        loss_path = os.path.join(self.train_config.output_dir, "loss_history.json")
        with open(loss_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        
        print(f"[Calibrator] 训练完成，最终模型保存到 {final_path}")
        return final_path


def run_calibration(config: UniFusionConfig):
    """运行校准训练的完整流程"""
    print("=" * 60)
    print("UniFusion 适配层校准训练")
    print("=" * 60)
    
    # 设置种子
    torch.manual_seed(config.seed)
    
    # 创建输出目录
    os.makedirs(config.train.output_dir, exist_ok=True)
    
    # 加载大模型
    print("\n[1/4] 加载大模型...")
    large_tokenizer = AutoTokenizer.from_pretrained(config.model.large_model_name)
    large_model = AutoModelForCausalLM.from_pretrained(
        config.model.large_model_name,
        torch_dtype=torch.bfloat16 if config.model.large_model_dtype == "bfloat16" else torch.float16,
        device_map=config.model.large_model_device,
    )
    if large_tokenizer.pad_token is None:
        large_tokenizer.pad_token = large_tokenizer.eos_token
    print(f"  大模型加载完成: {config.model.large_model_name}")
    
    # 加载小模型
    print("\n[2/4] 加载专业小模型...")
    small_tokenizer = AutoTokenizer.from_pretrained(config.model.small_model_name)
    small_model = AutoModelForCausalLM.from_pretrained(
        config.model.small_model_name,
        torch_dtype=torch.bfloat16 if config.model.small_model_dtype == "bfloat16" else torch.float16,
        device_map=config.model.small_model_device,
    )
    if small_tokenizer.pad_token is None:
        small_tokenizer.pad_token = small_tokenizer.eos_token
    print(f"  小模型加载完成: {config.model.small_model_name}")
    
    # 构建UniFusion模型
    print("\n[3/4] 构建UniFusion模型...")
    model = UniFusionModel(config)
    model.set_models(large_model, small_model, large_tokenizer, small_tokenizer)
    print(f"  可训练参数量: {model.count_trainable_parameters():,}")
    
    # 准备数据
    print("\n[4/4] 准备校准数据...")
    dataset = CalibrationDataset(
        config.train.train_data_path,
        large_tokenizer,
        max_length=1024,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始校准训练...")
    print("=" * 60)
    calibrator = AdapterCalibrator(config, model)
    final_path = calibrator.train(dataloader)
    
    print("\n" + "=" * 60)
    print(f"校准完成！适配层保存到: {final_path}")
    print("=" * 60)
    
    return model, final_path


if __name__ == "__main__":
    from ..config import get_default_config
    config = get_default_config()
    run_calibration(config)
