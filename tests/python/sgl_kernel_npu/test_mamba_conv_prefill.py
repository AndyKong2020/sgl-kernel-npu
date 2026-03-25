from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
import sgl_kernel_npu  # noqa: F401
from sgl_kernel_npu.mamba import causal_conv1d as causal_conv1d_mod

ATOL = 5e-2
RTOL = 1e-2
MIN_MATCH_RATIO = 0.95
DEVICE = "npu"


def assert_match(actual: torch.Tensor, expected: torch.Tensor, min_ratio: float = MIN_MATCH_RATIO) -> None:
    actual_fp32 = actual.detach().to(torch.float32).cpu()
    expected_fp32 = expected.detach().to(torch.float32).cpu()
    diff = (actual_fp32 - expected_fp32).abs()
    tol = ATOL + RTOL * expected_fp32.abs()
    match_ratio = (diff <= tol).float().mean().item()
    assert match_ratio >= min_ratio, (
        f"match_ratio={match_ratio:.4f}, max_abs={diff.max().item():.6e}, "
        f"max_rel={(diff / (expected_fp32.abs() + 1e-6)).max().item():.6e}"
    )


def direct_prefill_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    width = weight.size(-1)
    out = F.conv1d(
        x.to(torch.float32),
        weight.to(torch.float32),
        bias.to(torch.float32),
        padding=width - 1,
        groups=x.size(1),
    )
    out = out[..., : x.size(-1)]
    return F.silu(out)


def wrapper_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    conv_states_ref = conv_states.clone() if conv_states is not None else None
    query_start_loc, cache_indices, has_initial_state = causal_conv1d_mod._ensure_prefill_metadata(
        x,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states_ref,
    )
    x_pad, initial_state_pad, seqlens, indices = causal_conv1d_mod.prepare_data(
        x,
        weight,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states_ref,
    )
    out, _ = causal_conv1d_mod.causal_conv1d_fn_native(
        x_pad,
        weight,
        bias,
        seqlens=seqlens,
        has_initial_state=has_initial_state,
        initial_states=initial_state_pad,
        return_final_states=True,
        activation=activation,
    )

    full_context = causal_conv1d_mod._make_prefill_context(
        x_pad,
        initial_state_pad,
        weight.size(-1),
    )
    final_states = causal_conv1d_mod._gather_final_states_from_context(
        full_context,
        seqlens,
        weight.size(-1),
    ).to(dtype=x.dtype)

    if conv_states_ref is not None and cache_indices is not None:
        conv_states_ref.index_copy_(0, cache_indices, final_states)
    if x.ndim == 3:
        return out, conv_states_ref
    out = out.transpose(1, 2).contiguous().view(out.size(0) * out.size(2), out.size(1))
    out = torch.index_select(out, 0, indices).transpose(0, 1).contiguous()
    pad_seq_len = x.size(-1) - out.size(-1)
    if pad_seq_len > 0:
        out = F.pad(out, (0, pad_seq_len))
    return out, conv_states_ref


@torch.no_grad()
def test_direct_prefill_op_dense() -> None:
    batch, dim, seqlen, width = 2, 256, 21, 4
    x = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=torch.float16) * 0.2
    weight = torch.randn(dim, 1, width, device=DEVICE, dtype=torch.float16) * 0.2
    bias = torch.randn(dim, device=DEVICE, dtype=torch.float16) * 0.2

    out = torch.ops.npu.causal_conv1d(x, weight, bias)
    ref = direct_prefill_reference(x, weight, bias)

    assert out.dtype == torch.float32
    assert_match(out, ref)


@torch.no_grad()
def test_causal_conv1d_fn_npu_dense_bias_none() -> None:
    batch, dim, seqlen, width = 3, 160, 15, 4
    x = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=torch.float16) * 0.2
    weight = torch.randn(dim, width, device=DEVICE, dtype=torch.float16) * 0.2

    out = causal_conv1d_mod.causal_conv1d_fn_npu(x, weight, bias=None, activation="silu")
    ref, _ = wrapper_reference(x, weight, bias=None, activation="silu")

    assert out.dtype == x.dtype
    assert_match(out, ref)


@torch.no_grad()
def test_causal_conv1d_fn_npu_varlen_updates_conv_states() -> None:
    dim, width = 192, 4
    seqlens = torch.tensor([9, 7, 5], device=DEVICE, dtype=torch.int32)
    query_start_loc = torch.cat(
        [
            torch.zeros(1, device=DEVICE, dtype=torch.int32),
            torch.cumsum(seqlens, dim=0),
        ]
    )
    x = torch.randn(dim, int(query_start_loc[-1].item()), device=DEVICE, dtype=torch.float16) * 0.2
    weight = torch.randn(dim, width, device=DEVICE, dtype=torch.float16) * 0.2
    bias = torch.randn(dim, device=DEVICE, dtype=torch.float16) * 0.2
    cache_indices = torch.tensor([6, 1, 4], device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True], device=DEVICE, dtype=torch.bool)
    conv_states = torch.randn(8, dim, width - 1, device=DEVICE, dtype=torch.float16) * 0.2

    conv_states_actual = conv_states.clone()
    out = causal_conv1d_mod.causal_conv1d_fn_npu(
        x,
        weight,
        bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        conv_states=conv_states_actual,
        activation="silu",
    )
    ref, conv_states_ref = wrapper_reference(
        x,
        weight,
        bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        conv_states=conv_states,
        activation="silu",
    )

    assert_match(out, ref)
    torch.testing.assert_close(conv_states_actual.cpu(), conv_states_ref.cpu(), rtol=0.0, atol=0.0)


@torch.no_grad()
def test_width_not_4_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    batch, dim, seqlen, width = 2, 128, 13, 3
    x = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=torch.float16) * 0.2
    weight = torch.randn(dim, width, device=DEVICE, dtype=torch.float16) * 0.2
    bias = torch.randn(dim, device=DEVICE, dtype=torch.float16) * 0.2

    def _unexpected_prefill_call(*args, **kwargs):
        raise AssertionError("width != 4 should stay on the native fallback path")

    monkeypatch.setattr(causal_conv1d_mod, "_causal_conv1d_prefill_npu", _unexpected_prefill_call)
    out = causal_conv1d_mod.causal_conv1d_fn_npu(x, weight, bias, activation="silu")
    ref, _ = wrapper_reference(x, weight, bias, activation="silu")

    assert_match(out, ref)
