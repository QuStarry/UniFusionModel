"""
隐状态提取与注入的 Hook 工具

核心功能：
1. 从大模型指定层提取隐状态
2. 将增强后的隐状态注入回大模型
3. 从小模型指定层提取输出隐状态
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable
from contextlib import contextmanager


class HiddenStateHook:
    """
    通用的隐状态提取/注入Hook
    
    用法：
        hook = HiddenStateHook(model.model.layers[layer_idx])
        # 前向传播后，hook.output 就是该层的输出隐状态
        # 或者设置 hook.inject_input = tensor 来替换该层的输入
    """
    
    def __init__(self, module: nn.Module, store_output: bool = True, store_input: bool = False):
        self.module = module
        self.store_output = store_output
        self.store_input = store_input
        self.output: Optional[torch.Tensor] = None
        self.input: Optional[torch.Tensor] = None
        self.inject_output: Optional[torch.Tensor] = None  # 如果设置，将替换该层的输出
        self.inject_input: Optional[torch.Tensor] = None   # 如果设置，将替换该层的输入
        self._hook = None
        self._enabled = False
    
    def _hook_fn(self, module, input_args, output):
        # 存储输入
        if self.store_input:
            if isinstance(input_args, tuple) and len(input_args) > 0:
                self.input = input_args[0].detach()
            else:
                self.input = input_args
        
        # 存储输出（Transformer层的输出通常是tuple，第一个元素是hidden_states）
        if self.store_output:
            if isinstance(output, tuple):
                self.output = output[0].detach()
            else:
                self.output = output.detach()
        
        # 注入输出（替换该层输出）
        if self.inject_output is not None:
            if isinstance(output, tuple):
                # 保持其他输出不变，只替换hidden_states
                new_output = (self.inject_output,) + output[1:]
                return new_output
            return self.inject_output
        
        return output
    
    def register(self):
        """注册Hook"""
        if self._hook is None:
            self._hook = self.module.register_forward_hook(self._hook_fn)
            self._enabled = True
    
    def remove(self):
        """移除Hook"""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
            self._enabled = False
    
    def clear(self):
        """清除缓存"""
        self.output = None
        self.input = None
        self.inject_output = None
        self.inject_input = None
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, *args):
        self.remove()


class MultiLayerHookManager:
    """
    多层Hook管理器，方便同时操作多个层
    
    用法：
        manager = MultiLayerHookManager()
        manager.add_hook("large_layer_k", large_model.model.layers[k])
        manager.add_hook("small_output", small_model.model.layers[-1])
        with manager:
            outputs = model(**inputs)
        hidden_k = manager["large_layer_k"].output
    """
    
    def __init__(self):
        self.hooks: Dict[str, HiddenStateHook] = {}
    
    def add_hook(self, name: str, module: nn.Module, **kwargs):
        hook = HiddenStateHook(module, **kwargs)
        self.hooks[name] = hook
        return hook
    
    def register_all(self):
        for hook in self.hooks.values():
            hook.register()
    
    def remove_all(self):
        for hook in self.hooks.values():
            hook.remove()
    
    def clear_all(self):
        for hook in self.hooks.values():
            hook.clear()
    
    def __getitem__(self, name: str) -> HiddenStateHook:
        return self.hooks[name]
    
    def __enter__(self):
        self.register_all()
        return self
    
    def __exit__(self, *args):
        self.remove_all()


@contextmanager
def capture_hidden_states(model: nn.Module, layer_indices: List[int], 
                          model_type: str = "decoder"):
    """
    上下文管理器：捕获模型指定层的隐状态
    
    Args:
        model: HuggingFace模型
        layer_indices: 要捕获的层索引列表
        model_type: decoder / encoder
    
    Yields:
        dict: {layer_idx: hidden_state_tensor}
    """
    hooks = []
    captured = {}
    
    layers = model.model.layers if hasattr(model, "model") else model.layers
    
    for idx in layer_indices:
        actual_idx = idx if idx >= 0 else len(layers) + idx
        hook = HiddenStateHook(layers[actual_idx])
        hook.register()
        hooks.append((actual_idx, hook))
    
    try:
        yield captured
        # 在前向传播后填充captured
        for idx, hook in hooks:
            captured[idx] = hook.output
    finally:
        for _, hook in hooks:
            hook.remove()


def get_layers(model: nn.Module):
    """获取模型的Transformer层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2风格
    else:
        raise ValueError(f"无法识别模型结构: {type(model)}")


def get_hidden_size(model: nn.Module) -> int:
    """获取模型的隐藏维度"""
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    layers = get_layers(model)
    return layers[0].hidden_size if hasattr(layers[0], "hidden_size") else None
