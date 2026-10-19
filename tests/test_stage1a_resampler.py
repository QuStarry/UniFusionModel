"""Stage 1a：变长/跨网格重采样器（方案 A：输入侧 + 输出侧对称重采样）

覆盖：初始恒等（跨 tokenizer + 跨长度）、hook 无泄漏、训练可达性（死锁解除）、
save/load 往返、generate 守卫、resampler 关闭时回退到原有逐位投影路径。
"""
import pytest
import torch

from UniFusionModel.utils.hooks import get_layers


def test_initial_identity_across_tokenizer_and_length(build_unifusion, batch_inputs):
    """T1：跨 tokenizer + 跨 token 长度时，初始 call 输出必须严格等于 native 输出

    输入侧由零初始化门控保证 z_in == z_native，输出侧由零初始化 o_proj 保证
    enhanced == h_large，故融合结果应精确退化为主干原生行为（无预热崩溃）。
    """
    model, _ = build_unifusion("cross_attn")
    model.eval()

    assert model.resampler_in is not None and model.resampler_out is not None

    with torch.no_grad():
        native = model.forward_native(batch_inputs["ids_large"], batch_inputs["mask_large"])
        call = model.forward_call(
            batch_inputs["ids_large"],
            batch_inputs["mask_large"],
            small_input_ids=batch_inputs["ids_small"],
            small_attention_mask=batch_inputs["mask_small"],
        )

    max_diff = (native.logits - call.logits).abs().max().item()
    assert max_diff < 1e-5, f"初始恒等被破坏，max|call-native|={max_diff:.3e}"


def test_forward_call_does_not_leak_hooks(build_unifusion, batch_inputs):
    """T3：多次 forward_call 后，大/小模型相关层的 hook 数量应保持不变"""
    model, _ = build_unifusion("cross_attn")
    model.eval()

    small_layers = get_layers(model.small_model)
    large_layers = get_layers(model.large_model)

    def _hook_counts():
        return (
            len(small_layers[model._small_start_layer_idx]._forward_pre_hooks),
            len(small_layers[model._small_output_layer_idx]._forward_hooks),
            len(large_layers[model._large_hook_layer_idx]._forward_hooks),
        )

    before = _hook_counts()
    with torch.no_grad():
        model.forward_call(
            batch_inputs["ids_large"],
            batch_inputs["mask_large"],
            small_input_ids=batch_inputs["ids_small"],
            small_attention_mask=batch_inputs["mask_small"],
        )
    assert _hook_counts() == before


def test_resampler_out_get_grad_and_gate_stays_zero_at_init(build_unifusion, batch_inputs):
    """T2-a：初始步 resampler_out 必须拿到非零梯度（解锁链条的起点）

    初始 corr == 0 且 tanh(gate) == 0，故 grad_gate = <grad_fused, corr> = 0；
    但 resampler_out 的梯度与其输出乘子（gating 的 init_gate != 0）解耦，必须非零，
    否则整个链条无法自解锁。
    """
    model, _ = build_unifusion("cross_attn")
    out = model.forward_call(
        batch_inputs["ids_large"],
        batch_inputs["mask_large"],
        labels=batch_inputs["ids_large"],
        small_input_ids=batch_inputs["ids_small"],
        small_attention_mask=batch_inputs["mask_small"],
    )
    out.loss.backward()

    grad_out = max(
        p.grad.abs().max().item() for p in model.resampler_out.parameters()
        if p.grad is not None
    )
    assert grad_out > 0

    assert abs(torch.tanh(model.resampler_in_gate).item()) < 1e-12
    assert model.resampler_in_gate.grad.abs().item() == 0.0


def test_resampler_parameters_unlock_within_few_steps(build_unifusion, batch_inputs):
    """T2-b：数步之后门控脱离 0，且两侧重采样器参数都被实际更新（无梯度死锁）"""
    model, _ = build_unifusion("cross_attn")
    model.large_model.eval()
    model.small_model.eval()

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-3)
    init_in = {n: p.detach().clone() for n, p in model.resampler_in.named_parameters()}
    init_out = {n: p.detach().clone() for n, p in model.resampler_out.named_parameters()}

    for _ in range(8):
        out = model.forward_call(
            batch_inputs["ids_large"],
            batch_inputs["mask_large"],
            labels=batch_inputs["ids_large"],
            small_input_ids=batch_inputs["ids_small"],
            small_attention_mask=batch_inputs["mask_small"],
        )
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    current_in = dict(model.resampler_in.named_parameters())
    current_out = dict(model.resampler_out.named_parameters())
    delta_in = max(
        (current_in[n].detach() - p).abs().max().item() for n, p in init_in.items()
    )
    delta_out = max(
        (current_out[n].detach() - p).abs().max().item() for n, p in init_out.items()
    )

    assert delta_out > 0, "输出侧重采样器未被训练"
    assert delta_in > 0, "输入侧重采样器未被训练（梯度死锁未解除）"
    assert abs(torch.tanh(model.resampler_in_gate).item()) > 0, "门控始终停留在 0"


def test_adapter_save_load_roundtrip(build_unifusion, batch_inputs, tmp_path):
    """T4：保存后加载到新实例，输出应与保存前完全一致"""
    model, _ = build_unifusion("cross_attn")
    path = str(tmp_path / "adapter.pt")
    model.save_adapter(path)

    restored, _ = build_unifusion("cross_attn")
    restored.load_adapter(path)

    model.eval()
    restored.eval()
    with torch.no_grad():
        a = model.forward_call(
            batch_inputs["ids_large"],
            batch_inputs["mask_large"],
            small_input_ids=batch_inputs["ids_small"],
            small_attention_mask=batch_inputs["mask_small"],
        )
        b = restored.forward_call(
            batch_inputs["ids_large"],
            batch_inputs["mask_large"],
            small_input_ids=batch_inputs["ids_small"],
            small_attention_mask=batch_inputs["mask_small"],
        )
    assert (a.logits - b.logits).abs().max().item() < 1e-5


def test_generate_same_tokenizer_is_supported(build_unifusion, batch_inputs,
                                              tiny_model_factory, tokenizers):
    """T5-a：同 tokenizer 时逐 token 增量解码应正常返回（长度 = prompt + new tokens）"""
    tok_a, _ = tokenizers
    shared_vocab_small = tiny_model_factory(120, 32, 4, 4, 64)
    model, _ = build_unifusion(
        "cross_attn", large_tokenizer=tok_a, small_tokenizer=tok_a,
        small_model=shared_vocab_small,
    )
    model.eval()

    with torch.no_grad():
        generated = model.generate(
            batch_inputs["ids_large"], batch_inputs["mask_large"],
            mode="call", max_new_tokens=3,
        )
    assert generated.shape == (2, 14)


def test_generate_cross_tokenizer_raises_not_implemented(build_unifusion, batch_inputs):
    """T5-b：跨 tokenizer 时增量解码不成立，必须显式报错而非静默出错"""
    model, _ = build_unifusion("cross_attn")  # tok_a vs tok_b
    model.eval()

    with pytest.raises(NotImplementedError):
        with torch.no_grad():
            model.generate(
                batch_inputs["ids_large"], batch_inputs["mask_large"],
                mode="call", max_new_tokens=2,
            )


def test_resampler_disabled_falls_back(build_unifusion, batch_inputs):
    """T6：resampler_type="none" 时不构造重采样器，输出形状回退到原生逐位路径"""
    model, _ = build_unifusion("none")

    assert model.resampler_in is None and model.resampler_out is None
    assert model.resampler_in_gate is None

    with torch.no_grad():
        out = model.forward_call(batch_inputs["ids_large"], batch_inputs["mask_large"])
    assert out.logits.shape == (2, 11, 120)
