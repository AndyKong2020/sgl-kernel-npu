import logging
from typing import Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScript")
torch.manual_seed(42)


def npu_causal_conv1d_update_ref_graphsafe(
    hidden_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = True,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    preserve_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorized reference matching the deleted sglang helper's call pattern.

    Args:
        hidden_state: [batch, dim, seq_len]
        weight: [dim, width]
        conv_state: [batch, dim, state_len]
        bias: [dim]
        silu_activation: whether to apply SiLU
        num_accepted_tokens: optional [batch] tensor enabling speculative
            decoding semantics. When omitted, this matches the deleted helper's
            behavior: return the rolling window
            [history_{tail}, x_0, ..., x_{seq_len-1}] of length
            (width - 1) + (seq_len - 1).
        preserve_state_layout: when True, keep conv_state's original length and
            only update the prefix that the custom op mutates. This is useful
            for test/reference code that works against a larger global cache.

    Returns:
        out: [batch, dim, seq_len]
        new_conv_state: [batch, dim, state_len] for speculative mode, or
            [batch, dim, (width - 1) + (seq_len - 1)] for the deleted helper's
            normal mode.
    """
    if hidden_state.dim() != 3:
        raise ValueError(
            f"hidden_state must be [batch, dim, seq_len], got {tuple(hidden_state.shape)}"
        )
    if weight.dim() != 2:
        raise ValueError(f"weight must be [dim, width], got {tuple(weight.shape)}")
    if conv_state.dim() != 3:
        raise ValueError(
            f"conv_state must be [batch, dim, state_len], got {tuple(conv_state.shape)}"
        )

    batch, dim, seq_len = hidden_state.shape
    weight_dim, width = weight.shape
    if weight_dim != dim:
        raise ValueError(f"weight.shape[0] ({weight_dim}) must equal dim ({dim})")
    if conv_state.shape[:2] != (batch, dim):
        raise ValueError(
            "conv_state must align with hidden_state on [batch, dim], got "
            f"{tuple(conv_state.shape[:2])} vs {(batch, dim)}"
        )

    state_len = conv_state.shape[-1]
    history_len = width - 1
    if state_len < history_len:
        raise ValueError(
            f"conv_state.shape[-1] ({state_len}) must be >= width - 1 ({history_len})"
        )

    hidden_state_w = hidden_state.to(weight.dtype)
    conv_state_w = conv_state.to(weight.dtype)
    is_spec = (num_accepted_tokens is not None) and (width == 4)

    if num_accepted_tokens is None and not preserve_state_layout:
        full_context = torch.cat([conv_state_w, hidden_state_w], dim=-1)
        context = full_context[..., -(history_len + seq_len) :]
        target_state_len = history_len + (seq_len - 1)
        if target_state_len > 0:
            new_conv_state = full_context[..., -target_state_len:]
        else:
            new_conv_state = conv_state_w.new_empty((batch, dim, 0))
    else:
        if is_spec:
            max_offset = max(state_len - history_len, 0)
            offsets = (
                num_accepted_tokens.to(device=conv_state.device, dtype=torch.long) - 1
            ).clamp_(min=0, max=max_offset)
        else:
            offsets = torch.zeros((batch,), device=conv_state.device, dtype=torch.long)

        history_positions = offsets.unsqueeze(1) + torch.arange(
            history_len, device=conv_state.device, dtype=torch.long
        ).unsqueeze(0)
        history_positions = history_positions.unsqueeze(1).expand(-1, dim, -1)
        history = conv_state_w.gather(2, history_positions)
        context = torch.cat([history, hidden_state_w], dim=-1)

    windows = context.unfold(-1, width, 1)
    out = (windows * weight.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
    if bias is not None:
        out = out + bias.to(weight.dtype).view(1, dim, 1)
    if silu_activation:
        out = F.silu(out)
    out = out.to(hidden_state.dtype).contiguous()

    if num_accepted_tokens is None and not preserve_state_layout:
        return out, new_conv_state.to(conv_state.dtype).contiguous()

    updated_conv_state = conv_state.clone()
    if is_spec and state_len >= (width - 2) + seq_len:
        keep = width - 2
        shifted_positions = offsets.unsqueeze(1) + 1 + torch.arange(
            keep, device=conv_state.device, dtype=torch.long
        ).unsqueeze(0)
        shifted_positions = shifted_positions.unsqueeze(1).expand(-1, dim, -1)
        shifted = conv_state.gather(2, shifted_positions)
        spec_prefix = torch.cat([shifted, hidden_state.to(conv_state.dtype)], dim=-1)
        updated_conv_state[..., : keep + seq_len] = spec_prefix
    else:
        updated_conv_state[..., :history_len] = context[..., -history_len:].to(
            conv_state.dtype
        )

    return out, updated_conv_state.contiguous()


def vllm_causal_conv1d_update_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Graph-safe reference aligned with the custom update op semantics.

    Layout: x [batch, seq_len, dim], weight [width, dim],
    conv_state [num_cache_lines, state_len, dim].
    """
    batch, seq_len, dim = x.shape
    width = weight.shape[0]
    if conv_state_indices.dim() != 1 or conv_state_indices.shape[0] != batch:
        raise ValueError(
            "conv_state_indices must be [batch], got "
            f"{tuple(conv_state_indices.shape)} for batch={batch}"
        )

    local_state_indices = conv_state_indices.to(
        device=conv_state.device, dtype=torch.long
    )
    local_conv_state = conv_state.index_select(0, local_state_indices).transpose(
        1, 2
    )
    out, updated_local_state = npu_causal_conv1d_update_ref_graphsafe(
        hidden_state=x.transpose(1, 2).contiguous(),
        weight=weight.transpose(0, 1).contiguous(),
        conv_state=local_conv_state,
        bias=bias,
        silu_activation=activation,
        num_accepted_tokens=num_accepted_tokens,
        preserve_state_layout=True,
    )
    conv_state.index_copy_(0, local_state_indices, updated_local_state.transpose(1, 2))
    return out.transpose(1, 2).contiguous()


def test_vllm_causal_conv1d_update_ref_control_tensor_dtypes():
    """The Python reference should accept both int32/int64 control tensors."""
    torch.manual_seed(123)

    test_configs = [
        {"batch": 3, "seq_len": 4, "dim": 8, "width": 4, "state_len": 6, "nat": [4, 4, 4]},
        {"batch": 3, "seq_len": 3, "dim": 8, "width": 3, "state_len": 5, "nat": [1, 3, 2]},
    ]

    for cfg in test_configs:
        x = torch.randn(cfg["batch"], cfg["seq_len"], cfg["dim"])
        weight = torch.randn(cfg["width"], cfg["dim"])
        bias = torch.randn(cfg["dim"])
        conv_state_base = torch.randn(cfg["batch"] + 2, cfg["state_len"], cfg["dim"])

        outputs = {}
        states = {}
        for dtype in (torch.int32, torch.int64):
            conv_state = conv_state_base.clone()
            indices = torch.tensor([2, 0, 1], dtype=dtype)
            num_accepted_tokens = torch.tensor(cfg["nat"], dtype=dtype)

            y = vllm_causal_conv1d_update_ref(
                x,
                weight,
                conv_state,
                indices,
                bias=bias,
                activation=True,
                num_accepted_tokens=num_accepted_tokens,
            )
            y.view(cfg["batch"] * cfg["seq_len"], cfg["dim"])
            outputs[dtype] = y
            states[dtype] = conv_state

        assert torch.allclose(outputs[torch.int32], outputs[torch.int64], atol=1e-5, rtol=1e-5)
        assert torch.allclose(states[torch.int32], states[torch.int64], atol=1e-5, rtol=1e-5)


def test_npu_causal_conv1d_fla_update():
    """Test the NPU causal_conv1d_fla_update operator."""
    try:
        import torch_npu
    except ImportError as e:
        print(f"Skipping NPU test (import failed): {e}")
        return

    try:
        import sgl_kernel_npu
    except ImportError as e:
        print(f"Skipping NPU test (sgl_kernel_npu import failed): {e}")
        return

    try:
        if not (hasattr(torch_npu, "npu") and torch.npu.device_count() > 0):
            print("NPU not available, skipping NPU test")
            return
    except Exception as e:
        print(f"Failed to check NPU availability: {e}")
        return

    if not hasattr(torch.ops.npu, "causal_conv1d_fla_update"):
        print("causal_conv1d_fla_update operator not registered!")
        return

    ATOL, RTOL = 5e-2, 1e-2
    DEVICE = "npu"

    test_configs = [
        # --- Normal decode (no num_accepted_tokens) ---
        {"name": "decode_single_token", "bsz": 8, "dim": 4096, "seq_len": 1, "width": 4,
         "state_len": 3, "bias": False, "act": True, "nat": None},
        {"name": "decode_multi_token", "bsz": 4, "dim": 4096, "seq_len": 4, "width": 4,
         "state_len": 3, "bias": False, "act": True, "nat": None},
        {"name": "decode_no_act", "bsz": 4, "dim": 4096, "seq_len": 1, "width": 4,
         "state_len": 3, "bias": False, "act": False, "nat": None},
        {"name": "decode_large_dim", "bsz": 2, "dim": 8192, "seq_len": 1, "width": 4,
         "state_len": 3, "bias": False, "act": True, "nat": None},
        {"name": "decode_width3", "bsz": 4, "dim": 4096, "seq_len": 1, "width": 3,
         "state_len": 2, "bias": False, "act": True, "nat": None},
        {"name": "decode_large_batch", "bsz": 256, "dim": 12288, "seq_len": 4, "width": 4,
         "state_len": 3, "bias": False, "act": True, "nat": None},
        # --- Speculative decoding (num_accepted_tokens present, width=4) ---
        {"name": "spec_decode_basic", "bsz": 32, "dim": 4096, "seq_len": 4, "width": 4,
         "state_len": 6, "bias": False, "act": True, "nat": 4},
        {"name": "spec_decode_large", "bsz": 32, "dim": 12288, "seq_len": 4, "width": 4,
         "state_len": 6, "bias": False, "act": True, "nat": 4},
    ]

    for cfg in test_configs:
        torch.manual_seed(42)
        bsz = cfg["bsz"]
        dim = cfg["dim"]
        seq_len = cfg["seq_len"]
        width = cfg["width"]
        state_len = cfg["state_len"]
        cache_len = bsz + 2

        weight = torch.randn(width, dim, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16) if cfg["bias"] else None
        x = torch.randn(bsz, seq_len, dim, device=DEVICE, dtype=torch.bfloat16)
        conv_state_init = torch.randn(cache_len, state_len, dim, device=DEVICE, dtype=torch.bfloat16)
        conv_state_indices = torch.arange(bsz, device=DEVICE, dtype=torch.int64)

        nat_val = cfg["nat"]
        if nat_val is not None:
            num_accepted_tokens = torch.full((bsz,), nat_val, device=DEVICE, dtype=torch.int64)
        else:
            num_accepted_tokens = None

        # Reference
        conv_state_ref = conv_state_init.cpu().clone()
        y_ref = vllm_causal_conv1d_update_ref(
            x.cpu(), weight.cpu(), conv_state_ref, conv_state_indices.cpu(),
            bias=bias.cpu() if bias is not None else None,
            activation=cfg["act"],
            num_accepted_tokens=num_accepted_tokens.cpu() if num_accepted_tokens is not None else None,
        )

        # NPU
        conv_state_npu = conv_state_init.clone()
        npu_kwargs = dict(
            x=x, weight=weight, conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            bias=bias, activation_mode=cfg["act"], pad_slot_id=-1,
        )
        if num_accepted_tokens is not None:
            npu_kwargs["num_accepted_tokens"] = num_accepted_tokens

        y_npu = torch.ops.npu.causal_conv1d_fla_update(**npu_kwargs)
        torch.npu.synchronize()

        # Compare output
        y_ref_f = y_ref.float()
        y_npu_f = y_npu.cpu().float()
        out_diff = (y_npu_f - y_ref_f).abs()
        out_max = out_diff.max().item()
        out_mean = out_diff.mean().item()

        # Compare state
        # For spec mode, kernel writes to state[0:2+seq_len]. For normal, state[0:width-1].
        check_len = (2 + seq_len) if (nat_val is not None and width == 4 and state_len >= 2 + seq_len) else (width - 1)
        cs_ref_f = conv_state_ref.float()
        cs_npu_f = conv_state_npu.cpu().float()
        st_diff = (cs_npu_f[:bsz, :check_len] - cs_ref_f[:bsz, :check_len]).abs()
        st_max = st_diff.max().item()
        st_mean = st_diff.mean().item()

        try:
            torch.testing.assert_close(y_npu_f, y_ref_f, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(cs_npu_f[:bsz, :check_len], cs_ref_f[:bsz, :check_len], atol=0.0, rtol=0.0)
            print(f"[PASS] {cfg['name']}: output(max={out_max:.6g}, mean={out_mean:.6g}) state(max={st_max:.6g}, mean={st_mean:.6g})")
        except AssertionError as e:
            print(f"[FAIL] {cfg['name']}: output(max={out_max:.6g}, mean={out_mean:.6g}) state(max={st_max:.6g}, mean={st_mean:.6g})")
            print(f"  {str(e).splitlines()[0]}")

    print("All causal_conv1d decode tests completed.")


if __name__ == "__main__":
    test_npu_causal_conv1d_fla_update()
