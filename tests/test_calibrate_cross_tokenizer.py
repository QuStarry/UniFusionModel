"""Stage 1a：校准损失在「跨 tokenizer / 跨网格」下的行为与梯度链路

覆盖：损失项齐全、跨网格时逐位对齐被正确跳过、初始梯度解锁链条、
以及同 tokenizer 时逐位对齐路径仍然生效（回归）。
"""
import json

import torch
from torch.utils.data import DataLoader

from UniFusionModel.trainers.calibrate import AdapterCalibrator, CalibrationDataset


def _write_calibration_data(path, num_lines=6):
    """写入若干条逐词文本；词取自 tok_a 词表，对 tok_b 会自然回落到 unk"""
    with open(path, "w") as f:
        for i in range(num_lines):
            words = " ".join(f"a{(i * 7 + j) % 90}" for j in range(12))
            f.write(json.dumps({"text": words}) + "\n")


def _make_calibration_setup(build_unifusion, tmp_path, resampler_type, use_small_tokenizer,
                            small_model=None, small_tokenizer=None):
    data_path = str(tmp_path / "calib.jsonl")
    _write_calibration_data(data_path)

    model, cfg = build_unifusion(
        resampler_type,
        train_data_path=data_path,
        output_dir=str(tmp_path / "out"),
        batch_size=2,
        logging_steps=1,
        small_model=small_model,
        small_tokenizer=small_tokenizer,
    )
    model.adapter.train()
    model.gating.train()

    dataset = CalibrationDataset(
        data_path, model.large_tokenizer, max_length=32,
        small_tokenizer=model.small_tokenizer if use_small_tokenizer else None,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    return AdapterCalibrator(cfg, model), batch


def test_cross_tokenizer_dataset_emits_small_grid(build_unifusion, tmp_path):
    """跨 tokenizer 时数据集必须额外给出小模型自己的 token 网格"""
    _, batch = _make_calibration_setup(
        build_unifusion, tmp_path, "cross_attn", use_small_tokenizer=True
    )
    assert "small_input_ids" in batch and "small_attention_mask" in batch


def test_cross_grid_skips_positional_alignment(build_unifusion, tmp_path):
    """跨网格时 P_in/P_out 的逐位对齐不成立，hidden_mse / cycle 应被置零"""
    calibrator, batch = _make_calibration_setup(
        build_unifusion, tmp_path, "cross_attn", use_small_tokenizer=True
    )
    losses = calibrator._compute_calibration_loss(batch)

    assert set(losses) == {"hidden_mse", "cycle", "lm_loss", "native_keep", "total"}
    assert losses["hidden_mse"].item() == 0.0
    assert losses["cycle"].item() == 0.0
    assert torch.isfinite(losses["lm_loss"])


def test_initial_unlock_chain_gradients(build_unifusion, tmp_path):
    """初始步：resampler_out 梯度非零、门控保持 0 且其梯度为 0（解锁链条起点）"""
    calibrator, batch = _make_calibration_setup(
        build_unifusion, tmp_path, "cross_attn", use_small_tokenizer=True
    )
    losses = calibrator._compute_calibration_loss(batch)
    losses["total"].backward()

    model = calibrator.model
    grad_out = max(
        p.grad.abs().max().item() for p in model.resampler_out.parameters()
        if p.grad is not None
    )
    assert grad_out > 0, "resampler_out 初始无梯度，自解锁链条断裂"
    assert abs(torch.tanh(model.resampler_in_gate).item()) < 1e-12
    assert model.resampler_in_gate.grad.abs().item() == 0.0


def test_resampler_in_trains_after_gate_unlocks(build_unifusion, tmp_path):
    """若干步之后门控脱离 0，输入侧重采样器才开始拿到梯度并被更新"""
    calibrator, batch = _make_calibration_setup(
        build_unifusion, tmp_path, "cross_attn", use_small_tokenizer=True
    )
    model = calibrator.model
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-3)
    init_in = {n: p.detach().clone() for n, p in model.resampler_in.named_parameters()}

    for _ in range(5):
        losses = calibrator._compute_calibration_loss(batch)
        losses["total"].backward()
        optimizer.step()
        optimizer.zero_grad()

    current_in = dict(model.resampler_in.named_parameters())
    delta_in = max(
        (current_in[n].detach() - p).abs().max().item() for n, p in init_in.items()
    )
    assert abs(torch.tanh(model.resampler_in_gate).item()) > 0, "门控未被更新"
    assert delta_in > 0, "输入侧重采样器仍未解锁"


def test_same_tokenizer_keeps_positional_alignment(build_unifusion, tmp_path,
                                                   tiny_model_factory, tokenizers):
    """回归：同 tokenizer（故同词表）+ 无重采样时，逐位对齐路径必须仍然生效"""
    tok_a, _ = tokenizers
    # 「同 tokenizer」意味着小模型必须共享同一词表，否则 pad/token id 会越界
    shared_vocab_small = tiny_model_factory(120, 32, 4, 4, 64)
    calibrator, batch = _make_calibration_setup(
        build_unifusion, tmp_path, "none", use_small_tokenizer=False,
        small_model=shared_vocab_small, small_tokenizer=tok_a,
    )
    losses = calibrator._compute_calibration_loss(batch)

    assert losses["hidden_mse"].item() > 0
    assert losses["cycle"].item() > 0
    losses["total"].backward()
