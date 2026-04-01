"""Test causal_conv1d_update NPU kernel against torch reference.

Validates output and conv_state update for various dim sizes including
those requiring dim tiling (e.g. Qwen3.5-27B conv_dim=5120/10240).
"""

import torch
import torch.nn.functional as F
from typing import Optional

torch.manual_seed(42)


def torch_reference_causal_conv1d_update(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    """Pure-torch reference matching kernel semantics.

    Computes in float32 intermediate precision to match the NPU kernel,
    which casts bf16/fp16 -> fp32 for compute, then casts back.

    Args:
        hidden_state: [batch, seq_len, dim]
        conv_state:   [cache_len, state_len, dim]  (state_len = width - 1)
        weight:       [width, dim]
        conv_state_indices: [batch] int32
        bias:         [dim] optional
        activation:   whether to apply SiLU
    Returns:
        output: [batch, seq_len, dim]
    Side effect: conv_state is updated in-place.
    """
    hidden_state_t = hidden_state.transpose(1, 2)   # [B, D, S]
    weight_t = weight.transpose(0, 1)                # [D, K]
    conv_state_t = conv_state.transpose(1, 2)         # [C, D, W-1]

    bsz, hidden_size, seq_len = hidden_state_t.shape
    kernel_size = weight_t.shape[-1]
    orig_dtype = hidden_state_t.dtype

    # float32 intermediate to match kernel compute path
    full_context = torch.cat(
        [conv_state_t[conv_state_indices], hidden_state_t], dim=-1
    ).float()
    weight_f = weight_t.float()

    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len):]
    windows = computation_input.unfold(-1, kernel_size, 1)
    out = (windows * weight_f[None, :, None, :]).sum(dim=-1)

    if bias is not None:
        out = out + bias.float()[None, :, None]

    if activation:
        out = F.silu(out)

    out = out.to(orig_dtype)

    # Update conv_state in-place
    target_state_len = (kernel_size - 1) + (seq_len - 1)
    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:].to(orig_dtype)
    else:
        new_conv_state = torch.empty(
            bsz, hidden_size, 0,
            device=hidden_state_t.device, dtype=orig_dtype,
        )
    conv_state_t[conv_state_indices] = new_conv_state
    conv_state.copy_(conv_state_t.transpose(1, 2))

    return out.transpose(1, 2)


def test_npu_causal_conv1d_update(hidden_size, batch_size=1, seq_len=1,
                                   kernel_size=4, cache_len=10,
                                   dtype=torch.bfloat16,
                                   use_bias=False, activation=True):
    """Test NPU causal_conv1d_update against torch reference."""
    try:
        import torch_npu
        import sgl_kernel_npu
    except ImportError as e:
        print(f"Skipping NPU test (import failed): {e}")
        return False

    if not (hasattr(torch_npu, "npu") and torch.npu.device_count() > 0):
        print("NPU not available, skipping")
        return False

    if not hasattr(torch.ops.npu, "causal_conv1d_update"):
        print("causal_conv1d_update operator not registered!")
        return False

    DEVICE = "npu"
    CONV_STATE_LEN = kernel_size - 1 + seq_len - 1

    print(f"\n{'='*60}")
    print(f"Test: dim={hidden_size}, batch={batch_size}, seq={seq_len}, "
          f"width={kernel_size}, bias={use_bias}")
    print(f"{'='*60}")

    weight = torch.randn(kernel_size, hidden_size, device=DEVICE, dtype=dtype)
    bias_t = torch.randn(hidden_size, device=DEVICE, dtype=dtype) if use_bias else None
    hidden_state = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE, dtype=dtype)
    conv_state_init = torch.randn(cache_len, CONV_STATE_LEN, hidden_size, device=DEVICE, dtype=dtype)
    conv_state_indices = torch.arange(batch_size, device=DEVICE, dtype=torch.int32)

    # --- Torch reference ---
    conv_state_ref = conv_state_init.clone()
    out_ref = torch_reference_causal_conv1d_update(
        hidden_state=hidden_state,
        conv_state=conv_state_ref,
        weight=weight,
        conv_state_indices=conv_state_indices,
        bias=bias_t,
        activation=activation,
    )

    # --- NPU kernel ---
    conv_state_npu = conv_state_init.clone()
    empty_i32 = torch.empty(0, device=DEVICE, dtype=torch.int32)
    empty_dtype = torch.empty(0, device=DEVICE, dtype=dtype)

    try:
        out_npu = torch.ops.npu.causal_conv1d_update(
            x=hidden_state,
            weight=weight,
            conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            bias=bias_t if bias_t is not None else empty_dtype,
            num_accepted_tokens=empty_i32,
            activation_mode=activation,
            pad_slot_id=-1,
        )
    except Exception as e:
        print(f"FAIL: NPU kernel error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # --- Validate output ---
    out_npu_cpu = out_npu.cpu()
    out_ref_cpu = out_ref.cpu()

    assert out_npu_cpu.shape == out_ref_cpu.shape, \
        f"Shape mismatch: {out_npu_cpu.shape} vs {out_ref_cpu.shape}"

    diff = (out_npu_cpu - out_ref_cpu).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"  Output: max_abs_diff={max_abs:.6e}, mean_abs_diff={mean_abs:.6e}")

    # Both reference and kernel compute in float32 then cast to bf16,
    # so they should match exactly (zero diff).
    ATOL = 1e-3
    out_pass = max_abs <= ATOL
    print(f"  Output: {'PASS' if out_pass else 'FAIL'} (max_abs <= {ATOL})")

    # --- Validate state (only used cache slots) ---
    used_indices = conv_state_indices.cpu().long()
    state_npu_used = conv_state_npu.cpu()[used_indices]
    state_ref_used = conv_state_ref.cpu()[used_indices]
    state_diff = (state_npu_used - state_ref_used).abs()
    state_max = state_diff.max().item()
    state_pass = state_max <= 1e-6

    print(f"  State:  max_diff={state_max:.6e} (used slots only)")
    print(f"  State:  {'PASS' if state_pass else 'FAIL'}")

    passed = out_pass and state_pass
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = []

    # dim=4096: single tile, no tiling needed
    results.append(("dim=4096, bsz=1",
        test_npu_causal_conv1d_update(hidden_size=4096)))

    # dim=5120: Qwen3.5-27B hidden_size, requires dim tiling
    results.append(("dim=5120, bsz=1",
        test_npu_causal_conv1d_update(hidden_size=5120)))

    # dim=6144: Qwen3.5-0.8B conv_dim
    results.append(("dim=6144, bsz=1",
        test_npu_causal_conv1d_update(hidden_size=6144)))

    # dim=10240: large conv_dim, requires dim tiling
    results.append(("dim=10240, bsz=1",
        test_npu_causal_conv1d_update(hidden_size=10240)))

    # dim=2048, multi-batch
    results.append(("dim=2048, bsz=4",
        test_npu_causal_conv1d_update(hidden_size=2048, batch_size=4)))

    # dim=4096 with bias
    results.append(("dim=4096, bsz=1, bias",
        test_npu_causal_conv1d_update(hidden_size=4096, use_bias=True)))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        exit(1)
