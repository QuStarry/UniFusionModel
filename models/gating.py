"""
门控融合层（Gating Fusion）

决定如何混合大模型原生输出和小模型增强输出：
    h_out = g * h_enhanced + (1 - g) * h_native

支持三种门控：
1. scalar: 全局标量门控
2. vector: 逐维向量门控
3. input_dependent: 输入相关的动态门控（推荐）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatingFusion(nn.Module):
    """
    门控融合模块
    
    融合大模型原生隐状态和小模型增强隐状态
    """
    
    def __init__(self, hidden_dim: int, gating_type: str = "input_dependent",
                 init_gate: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gating_type = gating_type
        self.temperature = temperature
        
        if gating_type == "scalar":
            # 全局标量门控
            self.gate_logit = nn.Parameter(torch.tensor(self._inv_sigmoid(init_gate)))
        
        elif gating_type == "vector":
            # 逐维向量门控
            self.gate_logits = nn.Parameter(torch.full((hidden_dim,), self._inv_sigmoid(init_gate)))
        
        elif gating_type == "input_dependent":
            # 输入相关的动态门控：根据当前隐状态预测门控值
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            # 初始化为接近init_gate
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.constant_(self.gate_net[-1].bias, self._inv_sigmoid(init_gate))
        
        else:
            raise ValueError(f"未知的门控类型: {gating_type}")
    
    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        """sigmoid的反函数，用于初始化"""
        import math
        return math.log(x / (1 - x)) if 0 < x < 1 else 0.0
    
    def get_gate(self, hidden_native: torch.Tensor) -> torch.Tensor:
        """
        计算门控值
        
        Args:
            hidden_native: 大模型原生隐状态 [batch, seq, dim]
        
        Returns:
            gate: 门控值，shape取决于gating_type
        """
        if self.gating_type == "scalar":
            return torch.sigmoid(self.gate_logit / self.temperature)
        
        elif self.gating_type == "vector":
            return torch.sigmoid(self.gate_logits / self.temperature)
        
        elif self.gating_type == "input_dependent":
            # 对序列维度平均后预测门控
            pooled = hidden_native.mean(dim=1)  # [batch, dim]
            gate_logit = self.gate_net(pooled)  # [batch, 1]
            return torch.sigmoid(gate_logit / self.temperature)  # [batch, 1]
    
    def forward(self, hidden_native: torch.Tensor, 
                hidden_enhanced: torch.Tensor,
                return_gate: bool = False) -> torch.Tensor:
        """
        融合隐状态
        
        Args:
            hidden_native: 大模型原生隐状态 [batch, seq, dim]
            hidden_enhanced: 小模型增强后的隐状态 [batch, seq, dim]
            return_gate: 是否返回门控值（用于分析）
        
        Returns:
            fused: 融合后的隐状态 [batch, seq, dim]
        """
        gate = self.get_gate(hidden_native)
        
        # 广播门控到正确的shape
        if self.gating_type == "scalar":
            # gate是标量
            fused = gate * hidden_enhanced + (1 - gate) * hidden_native
        elif self.gating_type == "vector":
            # gate是[dim]，广播到[batch, seq, dim]
            fused = gate.unsqueeze(0).unsqueeze(0) * hidden_enhanced + \
                    (1 - gate.unsqueeze(0).unsqueeze(0)) * hidden_native
        elif self.gating_type == "input_dependent":
            # gate是[batch, 1]，广播到[batch, seq, dim]
            fused = gate.unsqueeze(1) * hidden_enhanced + \
                    (1 - gate.unsqueeze(1)) * hidden_native
        
        if return_gate:
            return fused, gate
        return fused
    
    def get_mean_gate_value(self) -> float:
        """获取平均门控值（用于日志记录）"""
        if self.gating_type == "scalar":
            return torch.sigmoid(self.gate_logit).item()
        elif self.gating_type == "vector":
            return torch.sigmoid(self.gate_logits).mean().item()
        else:
            return 0.5  # input_dependent无法静态获取
