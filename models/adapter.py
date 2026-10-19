"""
适配层（Adapter）实现

适配层负责在大模型和小模型的隐空间之间做双向投影：
- P_in: 大模型隐状态 (d_large) → 小模型输入空间 (d_small)
- P_out: 小模型输出隐状态 (d_small) → 大模型隐空间 (d_large)

支持三种架构：
1. linear: 简单线性投影
2. mlp: 两层MLP（带非线性）
3. mlp_residual: 带残差连接的MLP（推荐，训练更稳定）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdapterLayer(nn.Module):
    """
    单向适配层：将输入从input_dim投影到output_dim
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2048,
                 adapter_type: str = "mlp_residual", dropout: float = 0.1,
                 use_layer_norm: bool = True, init_method: str = "identity"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapter_type = adapter_type
        
        if adapter_type == "linear":
            self.proj = nn.Linear(input_dim, output_dim)
            self._init_linear(self.proj, init_method)
        
        elif adapter_type == "mlp":
            self.proj_in = nn.Linear(input_dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, output_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self._init_linear(self.proj_in, init_method)
            self._init_linear(self.proj_out, "zero" if init_method == "identity" else init_method)
            # 初始化为接近恒等映射（如果维度相同）
        
        elif adapter_type == "mlp_residual":
            self.proj_in = nn.Linear(input_dim, hidden_dim)
            self.proj_out = nn.Linear(hidden_dim, output_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            # 残差投影（当input_dim != output_dim时需要）
            if input_dim != output_dim:
                self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)
                self._init_linear(self.residual_proj, "identity")
            else:
                self.residual_proj = nn.Identity()
            # 初始化为零，这样初始时适配层就是恒等映射（或残差投影）
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)
        
        else:
            raise ValueError(f"未知的适配层类型: {adapter_type}")
        
        # LayerNorm 作用在「分支输入」上（维度是 input_dim），刻意不挂在残差恒等路径上：
        # 若放在 forward 末尾对输出做归一化，proj_out 零初始化时的恒等映射就会被破坏。
        if use_layer_norm:
            self.ln = nn.LayerNorm(input_dim)
        else:
            self.ln = nn.Identity()
    
    def _init_linear(self, linear: nn.Linear, method: str):
        if method == "xavier":
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        elif method == "identity":
            # 尽可能初始化为接近恒等映射
            if linear.in_features == linear.out_features:
                nn.init.eye_(linear.weight)
            else:
                # 维度不同时，用截断的单位矩阵
                nn.init.zeros_(linear.weight)
                min_dim = min(linear.in_features, linear.out_features)
                linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        elif method == "zero":
            nn.init.zeros_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        else:
            raise ValueError(f"未知的初始化方法: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, seq_len, output_dim]
        """
        if self.adapter_type == "linear":
            out = self.proj(self.ln(x))
        
        elif self.adapter_type == "mlp":
            hidden = self.act(self.proj_in(self.ln(x)))
            hidden = self.drop(hidden)
            out = self.proj_out(hidden)
        
        elif self.adapter_type == "mlp_residual":
            # 残差恒等路径只走 residual_proj，不被 LayerNorm 触碰；
            # 初始时 proj_out 为零，所以 out == residual_proj(x)（恒等初始化成立）
            residual = self.residual_proj(x)
            hidden = self.act(self.proj_in(self.ln(x)))
            hidden = self.drop(hidden)
            out = residual + self.proj_out(hidden)
        
        return out


class BidirectionalAdapter(nn.Module):
    """
    双向适配层：包含P_in和P_out两个投影
    
    P_in: large → small（大模型隐状态投影到小模型输入空间）
    P_out: small → large（小模型输出隐状态投影回大模型空间）
    """
    
    def __init__(self, large_dim: int, small_dim: int, hidden_dim: int = 2048,
                 adapter_type: str = "mlp_residual", **kwargs):
        super().__init__()
        self.large_dim = large_dim
        self.small_dim = small_dim
        
        # 大模型→小模型的输入投影
        self.p_in = AdapterLayer(
            input_dim=large_dim, output_dim=small_dim,
            hidden_dim=hidden_dim, adapter_type=adapter_type, **kwargs
        )
        
        # 小模型→大模型的输出投影
        self.p_out = AdapterLayer(
            input_dim=small_dim, output_dim=large_dim,
            hidden_dim=hidden_dim, adapter_type=adapter_type, **kwargs
        )
    
    def forward_large_to_small(self, hidden_large: torch.Tensor) -> torch.Tensor:
        """大模型隐状态 → 小模型输入空间"""
        return self.p_in(hidden_large)
    
    def forward_small_to_large(self, hidden_small: torch.Tensor) -> torch.Tensor:
        """小模型输出隐状态 → 大模型空间"""
        return self.p_out(hidden_small)
    
    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossAttentionResampler(nn.Module):
    """
    变长 / 跨网格重采样器（Stage 1a）

    解决的问题：当目标模型与源模型的 token 网格不一致时（不同 tokenizer、不同
    层数、乃至跨模态），P_in 的「逐位置线性映射」不成立（输出长度必须等于输入长度）。
    这里改用交叉注意力：

        query  = 目标模型自己的 token 表示 [B, T_q, D_q]（决定输出网格与位置语义）
        kv     = 源模型/专家的隐状态       [B, T_kv, D_kv]（提供被读取的内容）
        output = [B, T_q, D_q]

    这样输出长度由 query 侧决定，与源侧长度 T_kv 解耦；且由于 query 侧自带位置
    语义，替换/叠加到目标模型的第 k 层输入时不会破坏其位置结构。

    默认对 o_proj 做零初始化：初始时本模块输出恒为 0，所以「叠加」用法下模型
    行为完全退化为目标模型原生行为，可安全接入（无预热崩溃）。该性质也是跨模态
    （Stage 2/3）能复用同一算子的前提。
    """
    
    def __init__(self, query_dim: int, kv_dim: int, attn_dim: Optional[int] = None,
                 num_heads: int = 8, dropout: float = 0.0, zero_init: bool = True):
        super().__init__()
        attn_dim = attn_dim or query_dim
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim({attn_dim}) 必须能被 num_heads({num_heads}) 整除"
            )
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        
        self.q_proj = nn.Linear(query_dim, attn_dim)
        self.k_proj = nn.Linear(kv_dim, attn_dim)
        self.v_proj = nn.Linear(kv_dim, attn_dim)
        self.o_proj = nn.Linear(attn_dim, query_dim)
        self.drop = nn.Dropout(dropout)
        
        if zero_init:
            nn.init.zeros_(self.o_proj.weight)
            nn.init.zeros_(self.o_proj.bias)
    
    def forward(self, query: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: 目标侧表示 [B, T_q, D_q]
            kv: 源侧隐状态 [B, T_kv, D_kv]
            kv_mask: 源侧有效位掩码 [B, T_kv]，1=有效；None 表示全有效
        
        Returns:
            [B, T_q, D_q]
        """
        batch, t_q, _ = query.shape
        t_kv = kv.shape[1]
        dtype = query.dtype
        
        q = self.q_proj(query).view(batch, t_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(batch, t_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(batch, t_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_mask = None
        empty_rows = None
        if kv_mask is not None:
            keep = kv_mask.to(torch.bool)
            # 整行无有效源时 SDPA 会得到 NaN，这里用第 0 位占位，事后把输出清零
            empty_rows = ~keep.any(dim=1)
            if empty_rows.any():
                keep = keep.clone()
                keep[empty_rows, 0] = True
            keep = keep[:, None, None, :]  # [B, 1, 1, T_kv]
            attn_mask = torch.zeros(keep.shape, dtype=dtype, device=query.device)
            attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, t_q, self.attn_dim)
        out = self.o_proj(out)
        
        if empty_rows is not None and empty_rows.any():
            out = out.clone()
            out[empty_rows] = 0.0
        return out
