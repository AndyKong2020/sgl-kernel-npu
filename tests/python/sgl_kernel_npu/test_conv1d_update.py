import logging
from typing import Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScript")
torch.manual_seed(42)


def reference_causal_conv1d_update(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    """Reference implementation matching the FLA kernel convention.

    conv_state layout: [num_cache_lines, state_len, dim]
    History is stored at positions [0 .. width-2].
    The kernel reads history from the first (width-1) positions,
    processes all seq_len tokens, then writes back the last (width-1)
    tokens to positions [0 .. width-2].
    """
    # x: [batch, seq_len, dim], weight: [width, dim], conv_state: [cache_lines, state_len, dim]
    batch, seq_len, dim = x.shape
    width = weight.shape[0]
    state_prefix = width - 1

    x_f = x.float()
    w_f = weight.float()
    b_f = bias.float() if bias is not None else None
    cs_f = conv_state.float()

    y_out = torch.zeros_like(x_f)

    for b in range(batch):
        ci = int(conv_state_indices[b].item())
        hist = cs_f[ci, :state_prefix, :].clone()  # [width-1, dim]
        # Build extended sequence: [hist0, hist1, ..., hist_{w-2}, x0, x1, ..., x_{s-1}]
        ext = torch.cat([hist, x_f[b]], dim=0)  # [state_prefix + seq_len, dim]

        for t in range(seq_len):
            acc = torch.zeros(dim)
            for j in range(width):
                acc += w_f[j] * ext[t + j]
            if b_f is not None:
                acc += b_f
            if activation:
                acc = F.silu(acc)
            y_out[b, t] = acc

        # Write back last (width-1) tokens
        conv_state[ci, :state_prefix, :] = ext[-(state_prefix):].to(conv_state.dtype)

    return y_out.to(x.dtype)


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
        {"name": "decode_single_token", "bsz": 8, "dim": 4096, "seq_len": 1, "width": 4, "bias": False, "act": True},
        {"name": "decode_multi_token", "bsz": 4, "dim": 4096, "seq_len": 4, "width": 4, "bias": False, "act": True},
        {"name": "decode_no_act", "bsz": 4, "dim": 4096, "seq_len": 1, "width": 4, "bias": False, "act": False},
        {"name": "decode_large_dim", "bsz": 2, "dim": 8192, "seq_len": 1, "width": 4, "bias": False, "act": True},
        {"name": "decode_width3", "bsz": 4, "dim": 4096, "seq_len": 1, "width": 3, "bias": False, "act": True},
        {"name": "decode_large_batch", "bsz": 256, "dim": 12288, "seq_len": 4, "width": 4, "bias": False, "act": True},
    ]

    for cfg in test_configs:
        bsz = cfg["bsz"]
        dim = cfg["dim"]
        seq_len = cfg["seq_len"]
        width = cfg["width"]
        state_len = width - 1
        cache_len = bsz + 2

        weight = torch.randn(width, dim, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16) if cfg["bias"] else None
        x = torch.randn(bsz, seq_len, dim, device=DEVICE, dtype=torch.bfloat16)
        conv_state_init = torch.randn(cache_len, state_len, dim, device=DEVICE, dtype=torch.bfloat16)
        conv_state_indices = torch.arange(bsz, device=DEVICE, dtype=torch.int64)

        # Reference on CPU
        conv_state_ref_cpu = conv_state_init.cpu().clone()
        y_ref = reference_causal_conv1d_update(
            x.cpu(), weight.cpu(), conv_state_ref_cpu, conv_state_indices.cpu(),
            bias=bias.cpu() if bias is not None else None, activation=cfg["act"],
        )

        # NPU
        conv_state_npu = conv_state_init.clone()
        y_npu = torch.ops.npu.causal_conv1d_fla_update(
            x=x, weight=weight, conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            bias=bias, activation_mode=cfg["act"], pad_slot_id=-1,
        )
        torch.npu.synchronize()

        # Compare output
        y_ref_f = y_ref.cpu().float()
        y_npu_f = y_npu.cpu().float()
        out_diff = (y_npu_f - y_ref_f).abs()
        out_max = out_diff.max().item()
        out_mean = out_diff.mean().item()

        # Compare state (only the first bsz cache lines, first state_len positions)
        cs_ref_f = conv_state_ref_cpu.float()
        cs_npu_f = conv_state_npu.cpu().float()
        st_diff = (cs_npu_f[:bsz, :state_len] - cs_ref_f[:bsz, :state_len]).abs()
        st_max = st_diff.max().item()
        st_mean = st_diff.mean().item()

        try:
            torch.testing.assert_close(y_npu_f, y_ref_f, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(cs_npu_f[:bsz, :state_len], cs_ref_f[:bsz, :state_len], atol=0.0, rtol=0.0)
            print(f"[PASS] {cfg['name']}: output(max={out_max:.6g}, mean={out_mean:.6g}) state(max={st_max:.6g}, mean={st_mean:.6g})")
        except AssertionError as e:
            print(f"[FAIL] {cfg['name']}: output(max={out_max:.6g}, mean={out_mean:.6g}) state(max={st_max:.6g}, mean={st_mean:.6g})")
            print(f"  {str(e).splitlines()[0]}")

    print("All causal_conv1d decode tests completed.")


if __name__ == "__main__":
    test_npu_causal_conv1d_fla_update()
