"""
大模型与小模型之间的层映射策略

核心问题：大模型和小模型层数不同，如何找到语义对应的层对？

策略：
1. 比例映射（proportional）：按层数比例线性映射
2. 语义对齐映射（semantic）：基于表征相似度自动寻找最佳匹配层
3. 手动映射（manual）：用户指定
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from .hooks import get_layers, HiddenStateHook


def proportional_mapping(n_large_layers: int, n_small_layers: int,
                         large_hook_layer: int) -> int:
    """
    比例映射：大模型的第L层对应小模型的第几层
    
    例如：大模型32层，hook在第24层（3/4处）
          小模型24层，则对应小模型第18层（3/4处）
    
    Args:
        n_large_layers: 大模型总层数
        n_small_layers: 小模型总层数
        large_hook_layer: 大模型的hook层（支持负数）
    
    Returns:
        小模型对应的层索引
    """
    large_idx = large_hook_layer if large_hook_layer >= 0 else n_large_layers + large_hook_layer
    ratio = large_idx / max(n_large_layers - 1, 1)
    small_idx = int(round(ratio * (n_small_layers - 1)))
    return max(0, min(small_idx, n_small_layers - 1))


@torch.no_grad()
def find_best_matching_layer(large_model, small_model, 
                             large_hook_layer: int,
                             calibration_inputs: List[Dict],
                             candidate_small_layers: Optional[List[int]] = None,
                             metric: str = "cosine") -> Tuple[int, Dict[int, float]]:
    """
    基于表征相似度，自动寻找小模型中与大模型指定层语义最匹配的层
    
    原理：用同一批输入分别通过大模型和小模型，计算大模型hook层的隐状态
    与小模型各候选层隐状态的相似度，选相似度最高的层。
    
    Args:
        large_model: 大模型
        small_model: 小模型
        large_hook_layer: 大模型的hook层
        calibration_inputs: 校准用的输入列表（每个是一个tokenized dict）
        candidate_small_layers: 小模型的候选层列表，None表示全部层
        metric: cosine / mse
    
    Returns:
        (best_layer_idx, {layer_idx: similarity_score})
    """
    large_layers = get_layers(large_model)
    small_layers = get_layers(small_model)
    
    n_small = len(small_layers)
    if candidate_small_layers is None:
        # 默认搜索中间到后面的层（前几层通常是低层特征，语义不匹配）
        candidate_small_layers = list(range(n_small // 2, n_small))
    
    # 注册大模型hook
    large_idx = large_hook_layer if large_hook_layer >= 0 else len(large_layers) + large_hook_layer
    large_hook = HiddenStateHook(large_layers[large_idx])
    large_hook.register()
    
    # 注册小模型多层hook
    small_hooks = {}
    for idx in candidate_small_layers:
        actual_idx = idx if idx >= 0 else n_small + idx
        hook = HiddenStateHook(small_layers[actual_idx])
        hook.register()
        small_hooks[actual_idx] = hook
    
    similarities = {idx: [] for idx in candidate_small_layers}
    
    try:
        for batch in calibration_inputs:
            # 大模型前向
            large_model(**{k: v.to(large_model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()})
            large_hidden = large_hook.output  # [batch, seq, dim_large]
            
            # 小模型前向
            small_model(**{k: v.to(small_model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()})
            
            # 计算相似度
            for idx, hook in small_hooks.items():
                small_hidden = hook.output  # [batch, seq, dim_small]
                
                # 为了计算相似度，需要统一维度（用平均池化到相同长度或取CLS token）
                # 简单起见，取每个序列的平均表征
                large_vec = large_hidden.mean(dim=1)  # [batch, dim_large]
                small_vec = small_hidden.mean(dim=1)  # [batch, dim_small]
                
                # 维度不同时，用投影到共同空间（这里简单用PCA风格的随机投影，或者只比较前min_dim维）
                min_dim = min(large_vec.shape[-1], small_vec.shape[-1])
                large_vec = large_vec[..., :min_dim]
                small_vec = small_vec[..., :min_dim]
                
                if metric == "cosine":
                    sim = F.cosine_similarity(large_vec, small_vec, dim=-1).mean().item()
                else:  # mse（越小越好，转成相似度）
                    dist = F.mse_loss(large_vec, small_vec).item()
                    sim = -dist
                
                similarities[idx].append(sim)
            
            # 清除hook缓存
            large_hook.clear()
            for hook in small_hooks.values():
                hook.clear()
    
    finally:
        large_hook.remove()
        for hook in small_hooks.values():
            hook.remove()
    
    # 平均相似度
    avg_similarities = {idx: sum(scores) / len(scores) for idx, scores in similarities.items()}
    best_layer = max(avg_similarities, key=avg_similarities.get)
    
    return best_layer, avg_similarities


def get_layer_mapping(strategy: str = "proportional", **kwargs) -> int:
    """
    获取层映射的统一接口
    
    Args:
        strategy: proportional / semantic / manual
        **kwargs: 策略特定参数
    
    Returns:
        小模型对应的层索引
    """
    if strategy == "proportional":
        return proportional_mapping(
            kwargs["n_large_layers"],
            kwargs["n_small_layers"],
            kwargs["large_hook_layer"]
        )
    elif strategy == "manual":
        return kwargs["small_start_layer"]
    elif strategy == "semantic":
        best_layer, _ = find_best_matching_layer(**kwargs)
        return best_layer
    else:
        raise ValueError(f"未知的层映射策略: {strategy}")
