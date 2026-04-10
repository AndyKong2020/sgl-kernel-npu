import logging
from typing import Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScript")
torch.manual_seed(42)


def vllm_causal_conv1d_update_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """vLLM-style reference implementation for causal_conv1d_update.

    Faithfully models both normal decode and speculative decoding semantics:
      - Normal: reads history from state[0:width-1], writes back last (width-1) tokens.
      - Spec (num_accepted_tokens provided, width==4): reads history from
        state[offset:offset+width-1] where offset=accepted-1, writes back
        [shifted_hist, all_input_tokens] into state[0:2+seq_len].

    Layout: x [batch, seq_len, dim], weight [width, dim],
            conv_state [num_cache_lines, state_len, dim].
    """
    # Transpose to [batch, dim, seq_len] / [lines, dim, state_len] for unfold-based conv.
    batch, seq_len, dim = x.shape
    width = weight.shape[0]
    hidden_state = x.transpose(1, 2)         # [B, D, S]
    w = weight.transpose(0, 1)               # [D, K]
    cs = conv_state.transpose(1, 2)          # [lines, D, state_len]
    state_len = cs.shape[2]

    is_spec = (num_accepted_tokens is not None) and (width == 4)

    if is_spec:
        # Per-batch state-token offset: accepted - 1, clamped to [0, state_len - (width-1)]
        offsets = (num_accepted_tokens - 1).clamp(min=0, max=state_len - (width - 1))
    else:
        offsets = torch.zeros(batch, dtype=torch.long, device=x.device)

    out_list = []
    for b in range(batch):
        ci = int(conv_state_indices[b].item())
        off = int(offsets[b].item())
        hist = cs[ci, :, off:off + (width - 1)]   # [D, width-1]
        ctx = torch.cat([hist, hidden_state[b]], dim=-1).to(w.dtype)  # [D, width-1+S]

        # Correlation: out[s] = sum_k ctx[s+k] * w[k]
        windows = ctx.unfold(-1, width, 1)  # [D, S, K]
        o = (windows * w[:, None, :]).sum(dim=-1)  # [D, S]

        if bias is not None:
            o = o + bias[:, None].to(o.dtype)
        if activation:
            o = F.silu(o)

        out_list.append(o.to(x.dtype))

        # State writeback
        if is_spec and state_len >= 2 + seq_len:
            # Shift: [old[off+1], old[off+2], x0, x1, ..., x_{S-1}]
            keep = width - 2  # == 2 for width=4
            shifted = cs[ci, :, off + 1:off + 1 + keep] if off + 1 + keep <= state_len else cs[ci, :, :keep]
            new_state = torch.cat([shifted, hidden_state[b].to(cs.dtype)], dim=-1)  # [D, 2+S]
            cs[ci, :, :2 + seq_len] = new_state
        else:
            # Normal: last (width-1) tokens of ctx
            cs[ci, :, :width - 1] = ctx[:, -(width - 1):].to(cs.dtype)

    conv_state.copy_(cs.transpose(1, 2))
    out = torch.stack(out_list, dim=0).transpose(1, 2)  # [B, S, D]
    return out


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
