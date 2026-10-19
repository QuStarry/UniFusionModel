"""
UniFusion 主模型

核心功能：
1. 大模型前向传播到指定层，提取隐状态
2. 通过适配层投影到小模型空间
3. 小模型从对应层继续前向传播
4. 小模型输出投影回大模型空间
5. 门控融合大模型原生输出和小模型增强输出
6. 融合后的隐状态注入回大模型，继续后续层

支持模式：
- native: 纯大模型（基线）
- call: 调用模式（核心验证）
- text_tool: 文本级Tool Use（对比基线）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .adapter import BidirectionalAdapter
from .gating import GatingFusion
from ..utils.hooks import HiddenStateHook, get_layers, get_hidden_size
from ..utils.layer_mapping import proportional_mapping


class UniFusionModel(nn.Module):
    """
    UniFusion: 大模型兼容融合小模型的统一框架
    
    验证实验阶段：聚焦调用模式（Call Mode）的隐状态级融合
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.adapter_config = config.adapter
        self.gating_config = config.gating
        
        # 模型将在外部加载后通过set_models注入
        self.large_model = None
        self.small_model = None
        self.large_tokenizer = None
        self.small_tokenizer = None
        
        # 可训练参数
        self.adapter = None  # BidirectionalAdapter
        self.gating = None   # GatingFusion
        
        # Hook
        self._large_hook = None
        self._small_hooks = {}
        
        # 层映射信息
        self._large_hook_layer_idx = None
        self._small_start_layer_idx = None
        self._small_output_layer_idx = None
        self._large_num_layers = None
        self._small_num_layers = None
    
    def set_models(self, large_model, small_model, 
                   large_tokenizer=None, small_tokenizer=None):
        """注入已加载的大模型和小模型（都冻结）"""
        self.large_model = large_model
        self.small_model = small_model
        self.large_tokenizer = large_tokenizer
        self.small_tokenizer = small_tokenizer
        
        # 冻结两个模型的所有参数
        for param in self.large_model.parameters():
            param.requires_grad = False
        for param in self.small_model.parameters():
            param.requires_grad = False
        
        # 获取维度信息
        large_dim = get_hidden_size(large_model)
        small_dim = get_hidden_size(small_model)
        large_layers = get_layers(large_model)
        small_layers = get_layers(small_model)
        
        self._large_num_layers = len(large_layers)
        self._small_num_layers = len(small_layers)
        
        # 计算层索引
        self._large_hook_layer_idx = (self.model_config.large_hook_layer 
            if self.model_config.large_hook_layer >= 0 
            else self._large_num_layers + self.model_config.large_hook_layer)
        
        # 小模型起始层：用比例映射
        self._small_start_layer_idx = proportional_mapping(
            self._large_num_layers, self._small_num_layers,
            self.model_config.large_hook_layer
        )
        
        # 小模型输出层
        self._small_output_layer_idx = (self.model_config.small_output_layer
            if self.model_config.small_output_layer >= 0
            else self._small_num_layers + self.model_config.small_output_layer)
        
        # 初始化适配层和门控
        self.adapter = BidirectionalAdapter(
            large_dim=large_dim,
            small_dim=small_dim,
            hidden_dim=self.adapter_config.hidden_dim,
            adapter_type=self.adapter_config.adapter_type,
            dropout=self.adapter_config.dropout,
            use_layer_norm=self.adapter_config.use_layer_norm,
            init_method=self.adapter_config.init_method,
        )
        
        self.gating = GatingFusion(
            hidden_dim=large_dim,
            gating_type=self.gating_config.gating_type,
            init_gate=self.gating_config.init_gate,
            temperature=self.gating_config.temperature,
        )
        
        print(f"[UniFusion] 模型初始化完成")
        print(f"  大模型: {self._large_num_layers}层, dim={large_dim}")
        print(f"  小模型: {self._small_num_layers}层, dim={small_dim}")
        print(f"  大模型hook层: {self._large_hook_layer_idx}")
        print(f"  小模型起始层: {self._small_start_layer_idx}")
        print(f"  小模型输出层: {self._small_output_layer_idx}")
        print(f"  适配层参数量: {self.adapter.count_parameters():,}")
    
    def _prepare_small_model_forward(self, input_ids: torch.Tensor, 
                                      attention_mask: torch.Tensor,
                                      device: torch.device):
        """
        为小模型的中间层前向传播准备attention_mask和position_ids
        
        对于decoder-only模型，需要构造causal attention mask
        """
        batch_size, seq_len = input_ids.shape
        
        # position_ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # attention mask: 转换为4D的causal mask [batch, 1, seq, seq]
        # 这里简化处理，大部分模型的层会自己处理mask
        # 我们传入原始的attention_mask，让层自己处理
        return position_ids, attention_mask
    
    def _run_small_model_from_layer(self, hidden_states: torch.Tensor,
                                     start_layer: int, end_layer: int,
                                     attention_mask: torch.Tensor,
                                     position_ids: torch.Tensor) -> torch.Tensor:
        """
        让小模型从指定层开始前向传播到指定层
        
        Args:
            hidden_states: 输入隐状态 [batch, seq, small_dim]
            start_layer: 起始层索引
            end_layer: 结束层索引（包含）
            attention_mask: 注意力mask
            position_ids: 位置id
        
        Returns:
            最终隐状态 [batch, seq, small_dim]
        """
        small_layers = get_layers(self.small_model)
        
        # 某些模型需要past_key_value，这里设为None（不使用KV cache）
        past_key_value = None
        
        for layer_idx in range(start_layer, end_layer + 1):
            layer = small_layers[layer_idx]
            
            # 不同模型的层forward签名略有不同，这里用通用方式调用
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=False,
            )
            
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        return hidden_states
    
    def forward_native(self, input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor,
                       labels: Optional[torch.Tensor] = None,
                       **kwargs) -> CausalLMOutputWithPast:
        """纯大模型前向（基线）"""
        return self.large_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def forward_call(self, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     labels: Optional[torch.Tensor] = None,
                     return_details: bool = False,
                     **kwargs) -> CausalLMOutputWithPast:
        """
        调用模式前向传播（核心方法）
        
        流程：
        1. 大模型前向到hook层，提取隐状态
        2. 适配层P_in投影到小模型空间
        3. 小模型从对应层继续前向
        4. 适配层P_out投影回大模型空间
        5. 门控融合
        6. 注入回大模型，继续后续层
        """
        large_layers = get_layers(self.large_model)
        device = input_ids.device
        
        # ===== Step 1: 大模型前向到hook层 =====
        # 手动运行大模型的前几层
        inputs_embeds = self.large_model.model.embed_tokens(input_ids)
        hidden_large = inputs_embeds
        
        # 构造大模型的position_ids和attention_mask
        batch_size, seq_len = input_ids.shape
        large_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 运行大模型前hook层（0到hook_layer-1）
        for layer_idx in range(self._large_hook_layer_idx):
            layer = large_layers[layer_idx]
            layer_outputs = layer(
                hidden_large,
                attention_mask=attention_mask,
                position_ids=large_position_ids,
                use_cache=False,
            )
            hidden_large = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        # 保存hook层的原生输出（用于融合）
        # 运行hook层本身
        hook_layer = large_layers[self._large_hook_layer_idx]
        hook_outputs = hook_layer(
            hidden_large,
            attention_mask=attention_mask,
            position_ids=large_position_ids,
            use_cache=False,
        )
        hidden_native = hook_outputs[0] if isinstance(hook_outputs, tuple) else hook_outputs
        
        # ===== Step 2: 适配层P_in投影到小模型空间 =====
        hidden_small_input = self.adapter.forward_large_to_small(hidden_native)
        
        # ===== Step 3: 小模型从对应层继续前向 =====
        # 构造小模型的position_ids和attention_mask
        small_position_ids, small_attention_mask = self._prepare_small_model_forward(
            input_ids, attention_mask, device
        )
        
        # 小模型需要在自己的device上
        hidden_small_input = hidden_small_input.to(self.small_model.device)
        small_attention_mask = small_attention_mask.to(self.small_model.device)
        small_position_ids = small_position_ids.to(self.small_model.device)
        
        # 运行小模型从start_layer到output_layer
        hidden_small_output = self._run_small_model_from_layer(
            hidden_small_input,
            start_layer=self._small_start_layer_idx,
            end_layer=self._small_output_layer_idx,
            attention_mask=small_attention_mask,
            position_ids=small_position_ids,
        )
        
        # 移回大模型device
        hidden_small_output = hidden_small_output.to(device)
        
        # ===== Step 4: 适配层P_out投影回大模型空间 =====
        hidden_enhanced = self.adapter.forward_small_to_large(hidden_small_output)
        
        # ===== Step 5: 门控融合 =====
        if return_details:
            hidden_fused, gate_value = self.gating(
                hidden_native, hidden_enhanced, return_gate=True
            )
        else:
            hidden_fused = self.gating(hidden_native, hidden_enhanced)
            gate_value = None
        
        # ===== Step 6: 大模型继续后续层 =====
        hidden_large = hidden_fused
        for layer_idx in range(self._large_hook_layer_idx + 1, self._large_num_layers):
            layer = large_layers[layer_idx]
            layer_outputs = layer(
                hidden_large,
                attention_mask=attention_mask,
                position_ids=large_position_ids,
                use_cache=False,
            )
            hidden_large = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        
        # 最终的norm和lm_head
        hidden_large = self.large_model.model.norm(hidden_large)
        logits = self.large_model.lm_head(hidden_large)
        
        # 计算loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        
        if return_details:
            output.gate_value = gate_value
            output.hidden_native = hidden_native
            output.hidden_enhanced = hidden_enhanced
        
        return output
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mode: str = "call",
                **kwargs) -> CausalLMOutputWithPast:
        """
        统一前向接口
        
        Args:
            mode: native / call
        """
        if mode == "native":
            return self.forward_native(input_ids, attention_mask, labels, **kwargs)
        elif mode == "call":
            return self.forward_call(input_ids, attention_mask, labels, **kwargs)
        else:
            raise ValueError(f"未知的模式: {mode}")
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 mode: str = "call", max_new_tokens: int = 512,
                 temperature: float = 0.0, top_p: float = 1.0,
                 **kwargs) -> torch.Tensor:
        """
        生成方法（用于评测）
        
        注意：当前实现是逐token生成，效率较低但逻辑清晰。
        验证阶段够用，后续可以优化为KV cache版本。
        """
        generated = input_ids.clone()
        gen_attention_mask = attention_mask.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(
                input_ids=generated,
                attention_mask=gen_attention_mask,
                mode=mode,
            )
            next_token_logits = outputs.logits[:, -1, :]
            
            # 采样
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # 拼接
            generated = torch.cat([generated, next_token], dim=-1)
            gen_attention_mask = torch.cat([
                gen_attention_mask, 
                torch.ones((gen_attention_mask.shape[0], 1), device=gen_attention_mask.device)
            ], dim=-1)
            
            # 检查EOS
            if self.large_tokenizer is not None:
                eos_token_id = self.large_tokenizer.eos_token_id
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated
    
    def get_trainable_parameters(self):
        """获取可训练参数（只有适配层和门控）"""
        params = []
        if self.adapter is not None:
            params.extend(self.adapter.parameters())
        if self.gating is not None:
            params.extend(self.gating.parameters())
        return params
    
    def count_trainable_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.get_trainable_parameters() if p.requires_grad)
    
    def save_adapter(self, path: str):
        """保存适配层和门控的权重"""
        torch.save({
            "adapter": self.adapter.state_dict() if self.adapter else None,
            "gating": self.gating.state_dict() if self.gating else None,
            "config": {
                "large_hook_layer": self.model_config.large_hook_layer,
                "small_start_layer": self._small_start_layer_idx,
                "small_output_layer": self.model_config.small_output_layer,
            }
        }, path)
        print(f"[UniFusion] 适配层已保存到 {path}")
    
    def load_adapter(self, path: str):
        """加载适配层和门控的权重"""
        checkpoint = torch.load(path, map_location="cpu")
        if self.adapter and checkpoint["adapter"]:
            self.adapter.load_state_dict(checkpoint["adapter"])
        if self.gating and checkpoint["gating"]:
            self.gating.load_state_dict(checkpoint["gating"])
        print(f"[UniFusion] 适配层已从 {path} 加载")
