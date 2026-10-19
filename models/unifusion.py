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

from .adapter import BidirectionalAdapter, CrossAttentionResampler
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
        # 变长/跨网格重采样器（Stage 1a，resampler_type != "none" 时启用）
        # 输入侧负责把大模型内容注入小模型，输出侧负责把小模型内容读回大模型网格
        self.resampler_in = None
        self.resampler_out = None
        self.resampler_in_gate = None
        
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
        
        # 变长/跨网格重采样器（Stage 1a）：允许大小模型 token 网格不一致
        # （不同 tokenizer、不同序列长度，乃至跨模态）。两个方向各一个交叉注意力，
        # query 取「目标侧自己的表示」（决定输出网格），kv 取「源侧隐状态」（提供内容）。
        self.resampler_in = None
        self.resampler_out = None
        self.resampler_in_gate = None
        if getattr(self.adapter_config, "resampler_type", "none") != "none":
            # 输入侧：query=小模型 start 层输入（小模型网格），kv=大模型 hook 层隐状态。
            # 这里刻意不做零初始化：「初始恒等」由零初始化的门控标量保证；若 o_proj 与
            # 门控同时为 0，grad_gate=<grad_z, injected>=0 且 grad_o_proj=gate*(...)=0，
            # 梯度会双重归零而永久死锁。
            self.resampler_in = CrossAttentionResampler(
                query_dim=small_dim,
                kv_dim=large_dim,
                num_heads=self.adapter_config.resampler_heads,
                dropout=self.adapter_config.dropout,
                zero_init=False,
            )
            # 输出侧：query=大模型 hook 层输出（大模型网格），kv=小模型输出隐状态。
            # 输出侧没有独立门控，改由 o_proj 零初始化来保证「初始 corr == 0」。
            # 注意由此形成的「分阶段自解锁」链条（已实测验证）：
            #   step0: corr==0 且 tanh(gate)==0，故 grad_gate=<grad_fused, corr>=0；
            #          但 grad_resampler_out ≠ 0（其乘子是 gating 的 init_gate≠0）；
            #   step1: o_proj 更新后 corr≠0 -> grad_gate≠0 -> 门控脱离 0；
            #   step2: 门控非零后 resampler_in 才开始拿到梯度。
            # 因此该链条依赖 gating.init_gate ≠ 0（默认 0.5）；若把它也置 0 会重新死锁。
            self.resampler_out = CrossAttentionResampler(
                query_dim=large_dim,
                kv_dim=small_dim,
                num_heads=self.adapter_config.resampler_heads,
                dropout=self.adapter_config.dropout,
                zero_init=True,
            )
            # 输入侧注入强度的标量门控：零初始化 -> tanh(0)=0 -> 初始注入恒为 0
            self.resampler_in_gate = nn.Parameter(torch.zeros(1))
        
        # 适配层/门控保持 fp32（bf16 参数会让 AdamW 的更新量被数值精度直接吞掉），
        # 但必须放到大模型所在设备上，前向时再做 dtype 转换。
        large_device = next(self.large_model.parameters()).device
        self.adapter.to(device=large_device)
        self.gating.to(device=large_device)
        if self.resampler_in is not None:
            self.resampler_in.to(device=large_device)
            self.resampler_out.to(device=large_device)
            self.resampler_in_gate.data = self.resampler_in_gate.data.to(large_device)
        
        print(f"[UniFusion] 模型初始化完成")
        print(f"  大模型: {self._large_num_layers}层, dim={large_dim}")
        print(f"  小模型: {self._small_num_layers}层, dim={small_dim}")
        print(f"  大模型hook层: {self._large_hook_layer_idx}")
        print(f"  小模型起始层: {self._small_start_layer_idx}")
        print(f"  小模型输出层: {self._small_output_layer_idx}")
        print(f"  适配层参数量: {self.adapter.count_parameters():,}")
    
    def _enhance_with_small_model(self, hidden_native: torch.Tensor,
                                  input_ids: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor],
                                  small_input_ids: Optional[torch.Tensor] = None,
                                  small_attention_mask: Optional[torch.Tensor] = None,
                                  past_key_values=None,
                                  cache_position: Optional[torch.Tensor] = None,
                                  use_cache: bool = False):
        """
        把大模型 hook 层的隐状态投影进小模型，让小模型从中间层继续前向，
        再把小模型的输出投影回大模型空间。

        实现要点：不再手写逐层循环（那样要自己拼 causal mask 和 RoPE，容易和
        transformers 版本不兼容），而是用小模型的 forward_pre_hook 把第 start 层
        的输入替换为适配层输出，mask / RoPE / causal 全部交给原生 forward 处理。

        Stage 1a（resampler_type="cross_attn"）：允许大小模型 token 网格不一致
        （不同 tokenizer / 不同序列长度，乃至跨模态）。此时小模型用自己的
        small_input_ids 前向，两侧各用一个交叉注意力重采样：
            输入侧 z_in     = z_native + tanh(gate) * Resampler_in(query=z_native, kv=h_large)
            输出侧 enhanced = h_large  + Resampler_out(query=h_large_native, kv=h_small_out)
        z_native 是小模型第 start 层的原生输入（本来就会被算出来、此前被丢弃），
        因此不增加额外前向。初始恒等由「零初始化门控 + 输出侧零初始化 o_proj」保证：
        `z_in == z_native`、`enhanced == h_large`，融合结果严格退化为主干原生行为。

        Args:
            hidden_native: 大模型 hook 层输出 [batch, n_new, large_dim]
            input_ids: 大模型的输入 token
            attention_mask: 大模型的注意力 mask
            small_input_ids: 小模型自己的输入 token（不同 tokenizer 时必填；
                None 表示复用 input_ids，即同 tokenizer 场景）
            small_attention_mask: 小模型的注意力 mask，None 表示复用 attention_mask
            past_key_values: 小模型自己的 KV cache（增量解码用）
            cache_position: 本次前向 token 在整条序列中的位置（增量解码用）
            use_cache: 是否维护 KV cache

        Returns:
            (增强后的隐状态 [batch, n_new, large_dim] fp32, 小模型新的 past_key_values)
        """
        small_layers = get_layers(self.small_model)
        small_device = next(self.small_model.parameters()).device
        small_dtype = next(self.small_model.parameters()).dtype

        use_resampler = self.resampler_in is not None
        small_ids = input_ids if small_input_ids is None else small_input_ids
        small_mask = attention_mask if small_attention_mask is None else small_attention_mask

        projected_input = None
        resampler_kv_mask = None
        if not use_resampler:
            projected_input = self.adapter.forward_large_to_small(
                hidden_native.float()
            ).to(device=small_device, dtype=small_dtype)
        elif attention_mask is not None and attention_mask.shape[1] == hidden_native.shape[1]:
            # kv_mask 必须与 hidden_native 的位置一一对应；增量解码下 attention_mask
            # 是整条序列而 hidden_native 只有新 token，长度对不上时保守地不传 mask
            resampler_kv_mask = attention_mask.to(hidden_native.device)

        captured: Dict[str, Any] = {}

        def _inject_pre_hook(module, args):
            # 替换 start 层的输入隐状态；前缀层的输出会被丢弃
            if not use_resampler:
                return (projected_input,) + tuple(args[1:])
            z_native = args[0]
            injected = self.resampler_in(
                z_native.float(), hidden_native.float(), kv_mask=resampler_kv_mask
            ).to(dtype=z_native.dtype)
            gate = torch.tanh(self.resampler_in_gate).to(dtype=z_native.dtype)
            return (z_native + gate * injected,) + tuple(args[1:])

        def _capture_output_hook(module, args, output):
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        pre_handle = small_layers[self._small_start_layer_idx].register_forward_pre_hook(
            _inject_pre_hook
        )
        out_handle = small_layers[self._small_output_layer_idx].register_forward_hook(
            _capture_output_hook
        )
        # 只跑 backbone（跳过 lm_head）：这里只要隐状态，不需要 [batch, seq, vocab] 的 logits
        small_backbone = getattr(self.small_model, "model", self.small_model)
        try:
            small_outputs = small_backbone(
                input_ids=small_ids.to(small_device),
                attention_mask=None if small_mask is None else small_mask.to(small_device),
                use_cache=use_cache,
                past_key_values=past_key_values,
                cache_position=None if cache_position is None else cache_position.to(small_device),
            )
        finally:
            pre_handle.remove()
            out_handle.remove()

        hidden_small_output = captured["hidden"].to(hidden_native.device)
        if not use_resampler:
            enhanced = self.adapter.forward_small_to_large(hidden_small_output.float())
        else:
            # 输出侧重采样：把小模型输出（小模型网格）读回到大模型网格上。
            # o_proj 零初始化 -> 初始 corr 恒为 0 -> enhanced == hidden_native，
            # 于是门控融合在初始时严格退化为大模型原生输出（无预热崩溃）。
            out_kv_mask = small_mask
            if out_kv_mask is not None and out_kv_mask.shape[1] != hidden_small_output.shape[1]:
                out_kv_mask = None
            corr = self.resampler_out(
                hidden_native.float(),
                hidden_small_output.float(),
                kv_mask=None if out_kv_mask is None else out_kv_mask.to(hidden_native.device),
            )
            enhanced = hidden_native.float() + corr
        return enhanced, getattr(small_outputs, "past_key_values", None)
    
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
                     small_input_ids: Optional[torch.Tensor] = None,
                     small_attention_mask: Optional[torch.Tensor] = None,
                     past_key_values=None,
                     small_past_key_values=None,
                     use_cache: bool = False,
                     cache_position: Optional[torch.Tensor] = None,
                     **kwargs) -> CausalLMOutputWithPast:
        """
        调用模式前向传播（核心方法）

        流程：
        1. 大模型原生前向；在 hook 层注册 forward_hook，拿到该层的原生隐状态
        2. 小模型从对应层继续前向（内含 P_in 投影），输出再经 P_out 投影回大模型空间
        3. 门控融合原生隐状态与增强隐状态
        4. 用融合结果替换 hook 层的输出，大模型继续后续层直到 lm_head / loss

        实现要点：不再手写逐层前向（那需要自行构造 4D causal mask 与 RoPE 的
        position_embeddings，和 transformers 版本强耦合），而是用原生 forward +
        forward_hook 替换该层输出，mask / RoPE / cache 全部交给 transformers 处理。

        增量解码（P1-11）：use_cache=True 时，大/小模型各自维护 KV cache。
        hook 层替换的是「层输出」而非「层输入」，因此每一层缓存的 K/V 都来自
        未受融合影响的输入，逐 token 复用与整段前向在数学上等价。小模型的
        cache 通过 outputs.small_past_key_values 透出，供调用方在下一步回填。

        Args:
            small_input_ids: 小模型自己的输入 token（Stage 1a 跨 tokenizer 时必填；
                None 表示复用 input_ids，即同 tokenizer 场景）
            small_attention_mask: 小模型的注意力 mask，None 表示复用 attention_mask
            past_key_values: 大模型的 KV cache（增量解码时传入）
            small_past_key_values: 小模型的 KV cache（增量解码时传入）
            use_cache: 是否维护 KV cache（训练时置 False）
            cache_position: 本次前向 token 在整条序列中的绝对位置（增量解码时必须给）
        """
        large_layers = get_layers(self.large_model)
        hook_layer = large_layers[self._large_hook_layer_idx]

        details: Dict[str, torch.Tensor] = {}
        small_cache: Dict[str, Any] = {}

        def _fuse_hook(module, args, output):
            hidden_native = output[0] if isinstance(output, tuple) else output

            hidden_enhanced, small_new_cache = self._enhance_with_small_model(
                hidden_native, input_ids, attention_mask,
                small_input_ids=small_input_ids,
                small_attention_mask=small_attention_mask,
                past_key_values=small_past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
            )
            small_cache["past_key_values"] = small_new_cache

            # 门控参数是 fp32，这里统一到 fp32 融合，再转回主干 dtype 以兼容后续层
            hidden_fused, gate_value = self.gating(
                hidden_native.float(), hidden_enhanced, return_gate=True
            )
            hidden_fused = hidden_fused.to(hidden_native.dtype)

            if return_details:
                details["hidden_native"] = hidden_native
                details["hidden_enhanced"] = hidden_enhanced
                details["gate_value"] = gate_value

            if isinstance(output, tuple):
                return (hidden_fused,) + tuple(output[1:])
            return hidden_fused

        handle = hook_layer.register_forward_hook(_fuse_hook)
        try:
            outputs = self.large_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            handle.remove()

        if use_cache:
            outputs.small_past_key_values = small_cache.get("past_key_values")

        if return_details:
            outputs.gate_value = details.get("gate_value")
            outputs.hidden_native = details.get("hidden_native")
            outputs.hidden_enhanced = details.get("hidden_enhanced")

        return outputs
    
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

        native 模式直接复用大模型自带的 generate（内部已做 KV cache）。
        call 模式走增量解码（P1-11）：预填充整段 prompt，之后每次只前向 1 个
        token，大模型与小模型各自维护 KV cache，避免每步重复计算整段序列。
        """
        if mode == "native":
            return self.large_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self._get_pad_token_id(),
                **kwargs,
            )

        # 跨 tokenizer 时逐 token 增量生成不成立（新 token 属于大模型词表，无法直接
        # 追加到小模型序列）。Stage 1a 只覆盖整段前向 / 校准，这里显式报错而非静默出错。
        if self.resampler_in is not None and self._uses_distinct_tokenizer():
            raise NotImplementedError(
                "大小模型 tokenizer 不同，逐 token 增量生成尚未支持（Stage 1a 仅覆盖"
                "整段前向与校准）。请使用同 tokenizer，或改用 forward_call 做整段前向。"
            )

        generated = input_ids.clone()
        gen_attention_mask = attention_mask.clone()
        device = input_ids.device

        # 增量解码状态：首步预填充整段 prompt，后续每次只喂 1 个新 token
        cur_input_ids = generated
        cur_cache_position = torch.arange(generated.shape[1], device=device)
        large_past = None
        small_past = None

        for _ in range(max_new_tokens):
            outputs = self.forward_call(
                input_ids=cur_input_ids,
                attention_mask=gen_attention_mask,
                use_cache=True,
                past_key_values=large_past,
                small_past_key_values=small_past,
                cache_position=cur_cache_position,
            )
            large_past = outputs.past_key_values
            small_past = getattr(outputs, "small_past_key_values", None)

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

            # 下一步只前向新 token，其绝对位置是当前序列长度 - 1
            cur_input_ids = next_token
            cur_cache_position = torch.tensor([generated.shape[1] - 1], device=device)

            # 检查EOS
            if self.large_tokenizer is not None:
                eos_token_id = self.large_tokenizer.eos_token_id
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return generated

    def _get_pad_token_id(self) -> Optional[int]:
        """取 pad_token_id，缺省时回退到 eos（Qwen2 二者常相同）"""
        if self.large_tokenizer is None:
            return None
        return self.large_tokenizer.pad_token_id or self.large_tokenizer.eos_token_id

    def _uses_distinct_tokenizer(self) -> bool:
        """大小模型 tokenizer 是否使用不同词表（用于判断能否逐 token 增量生成）"""
        if self.large_tokenizer is None or self.small_tokenizer is None:
            return False
        return (getattr(self.large_tokenizer, "vocab_size", None)
                != getattr(self.small_tokenizer, "vocab_size", None))
    
    def get_trainable_parameters(self):
        """获取可训练参数（适配层、门控和重采样器）"""
        params = []
        if self.adapter is not None:
            params.extend(self.adapter.parameters())
        if self.gating is not None:
            params.extend(self.gating.parameters())
        if self.resampler_in is not None:
            params.extend(self.resampler_in.parameters())
            params.extend(self.resampler_out.parameters())
            params.append(self.resampler_in_gate)
        return params
    
    def count_trainable_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.get_trainable_parameters() if p.requires_grad)
    
    def save_adapter(self, path: str):
        """保存适配层、门控和重采样器的权重"""
        torch.save({
            "adapter": self.adapter.state_dict() if self.adapter else None,
            "gating": self.gating.state_dict() if self.gating else None,
            "resampler_in": self.resampler_in.state_dict() if self.resampler_in else None,
            "resampler_out": self.resampler_out.state_dict() if self.resampler_out else None,
            "resampler_in_gate": (
                None if self.resampler_in_gate is None
                else self.resampler_in_gate.detach().cpu()
            ),
            "config": {
                "large_hook_layer": self.model_config.large_hook_layer,
                "small_start_layer": self._small_start_layer_idx,
                "small_output_layer": self.model_config.small_output_layer,
                "resampler_type": getattr(self.adapter_config, "resampler_type", "none"),
            }
        }, path)
        print(f"[UniFusion] 适配层已保存到 {path}")
    
    def load_adapter(self, path: str):
        """加载适配层、门控和重采样器的权重"""
        checkpoint = torch.load(path, map_location="cpu")
        if self.adapter and checkpoint.get("adapter"):
            self.adapter.load_state_dict(checkpoint["adapter"])
        if self.gating and checkpoint.get("gating"):
            self.gating.load_state_dict(checkpoint["gating"])
        if self.resampler_in is not None:
            if checkpoint.get("resampler_in"):
                self.resampler_in.load_state_dict(checkpoint["resampler_in"])
                self.resampler_out.load_state_dict(checkpoint["resampler_out"])
            gate = checkpoint.get("resampler_in_gate")
            if gate is not None:
                self.resampler_in_gate.data.copy_(gate.to(self.resampler_in_gate.device))
        print(f"[UniFusion] 适配层已从 {path} 加载")
