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
