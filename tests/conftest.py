"""Stage 1a 测试夹具：极微型 Qwen2 模型 + 逐词 tokenizer（CPU，秒级可跑）

用逐词（WordLevel）tokenizer 是为了能精确控制词表大小，从而在单进程内
构造「跨 tokenizer / 跨 token 网格」的场景，无需下载任何真实模型。
"""
import copy
import os
import sys

import pytest
import torch
from tokenizers import Tokenizer, models as tk_models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

# unifusion.py 内部使用 ..utils 相对导入，必须让 UniFusionModel 作为「包」被导入，
# 因此把项目根目录的父目录（而非根目录本身）加入 sys.path。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(PROJECT_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from UniFusionModel.config import UniFusionConfig
from UniFusionModel.models import UniFusionModel  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """所有用例固定随机种子，保证初始恒等/梯度链路等断言可复现"""
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tiny_model_factory():
    """返回「构造极微型 Qwen2 因果 LM」的工厂函数"""

    def _build(vocab_size: int, hidden: int, layers: int, heads: int, intermediate: int):
        cfg = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            intermediate_size=intermediate,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        return Qwen2ForCausalLM(cfg)

    return _build


@pytest.fixture(scope="session")
def tiny_tokenizer_factory():
    """返回「构造指定词表大小的逐词 tokenizer」的工厂函数"""

    def _build(vocab_size: int, tag: str) -> PreTrainedTokenizerFast:
        vocab = {f"{tag}{i}": i for i in range(vocab_size - 1)}
        vocab["<pad>"] = vocab_size - 1
        tk = Tokenizer(tk_models.WordLevel(vocab=vocab, unk_token="<pad>"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tk, pad_token="<pad>", eos_token="<pad>"
        )

    return _build


@pytest.fixture(scope="session")
def tokenizers(tiny_tokenizer_factory):
    """(tok_a: 词表 120, tok_b: 词表 96)，词表不同 -> 触发「跨 tokenizer」路径"""
    return tiny_tokenizer_factory(120, "a"), tiny_tokenizer_factory(96, "b")


@pytest.fixture(scope="session")
def model_templates(tiny_model_factory):
    """大/小模型模板：词表不同（120 vs 96）、维度与层数也不同，仅作 deepcopy 源"""
    large = tiny_model_factory(120, 48, 6, 4, 96)
    small = tiny_model_factory(96, 32, 4, 4, 64)
    return large, small


@pytest.fixture
def build_unifusion(tokenizers, model_templates):
    """按 resampler_type 组装一个 UniFusionModel，返回 (model, config)"""
    tok_a, tok_b = tokenizers
    large, small = model_templates

    def _build(resampler_type="cross_attn", large_tokenizer=None,
               small_tokenizer=None, small_model=None, **train_overrides):
        cfg = UniFusionConfig()
        cfg.model.large_hook_layer = -2
        cfg.model.small_output_layer = -1
        cfg.adapter.resampler_type = resampler_type
        cfg.adapter.resampler_heads = 4
        for key, value in train_overrides.items():
            setattr(cfg.train, key, value)

        model = UniFusionModel(cfg)
        model.set_models(
            copy.deepcopy(large),
            copy.deepcopy(small) if small_model is None else copy.deepcopy(small_model),
            large_tokenizer if large_tokenizer is not None else tok_a,
            small_tokenizer if small_tokenizer is not None else tok_b,
        )
        return model, cfg

    return _build


@pytest.fixture(scope="session")
def batch_inputs():
    """大/小模型各自的 token 张量：长度刻意不同（11 vs 7）以覆盖变长网格"""
    return {
        "ids_large": torch.randint(0, 90, (2, 11)),
        "mask_large": torch.ones(2, 11, dtype=torch.long),
        "ids_small": torch.randint(0, 90, (2, 7)),
        "mask_small": torch.ones(2, 7, dtype=torch.long),
    }
