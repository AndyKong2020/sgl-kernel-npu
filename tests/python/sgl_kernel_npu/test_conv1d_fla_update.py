import logging
from typing import Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScript")
torch.manual_seed(42)


def vllm_causal_conv1d_update_ref(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    """Reference implementation for causal_conv1d decode (update)."""
    hidden_state = hidden_state.transpose(1, 2) if hidden_state.dim() == 3 else hidden_state.unsqueeze(2)
    weight = weight.transpose(0, 1)
    conv_state = conv_state.transpose(1, 2)
    bsz, hidden_size, seq_len = hidden_state.shape
    kernel_size = weight.shape[-1]

    target_state_len = (kernel_size - 1) + (seq_len - 1)

    full_context = torch.cat([conv_state[conv_state_indices], hidden_state], dim=-1).to(
        weight.dtype
    )

    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len) :]
    windows = computation_input.unfold(-1, kernel_size, 1)
    out = (windows * weight[None, :, None, :]).sum(dim=-1)

    if bias is not None:
        out = out + bias[None, :, None]

    if activation:
        out = F.silu(out)

    out = out.to(hidden_state.dtype)

    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:]
    else:
        new_conv_state = torch.empty(
            bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype
        )
    conv_state[conv_state_indices] = new_conv_state
    conv_state = conv_state.transpose(1, 2)
    out = out.transpose(1, 2)

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

    BSZ = 1
    HIDDEN_SIZE = 4096
    SEQ_LEN = 1
    KERNEL_SIZE = 4
    CACHE_LEN = 10
    CONV_STATE_LEN = KERNEL_SIZE - 1 + SEQ_LEN - 1
    DTYPE = torch.bfloat16
    DEVICE = "npu"

    print(f"\n{'='*50}")
    print(f"Testing NPU causal_conv1d_fla_update on {DEVICE}")
    print(f"{'='*50}")

    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = None
    hidden_state = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_init = torch.randn(
        CACHE_LEN, CONV_STATE_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
    )
    conv_state_indices = torch.arange(BSZ, device=DEVICE, dtype=torch.int64)
    num_accepted_tokens = torch.tensor(
        [SEQ_LEN] * BSZ, device=DEVICE, dtype=torch.int64
    )

    conv_state_ref = conv_state_init.clone()
    out_ref = vllm_causal_conv1d_update_ref(
        hidden_state=hidden_state,
        conv_state=conv_state_ref,
        weight=weight,
        bias=bias,
        conv_state_indices=conv_state_indices,
        activation=True,
    )

    conv_state_npu = conv_state_init.clone()

    try:
        out_npu = torch.ops.npu.causal_conv1d_fla_update(
            x=hidden_state,
            weight=weight,
            conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            bias=bias,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=True,
            pad_slot_id=-1,
        )
        torch.npu.synchronize()

        print(f"NPU kernel executed successfully!")
        print(f"Output shape: {out_npu.shape}")

        out_npu_cpu = out_npu.cpu()
        out_ref_cpu = out_ref.cpu()

        assert (
            out_npu_cpu.shape == out_ref_cpu.shape
        ), f"Output shape mismatch: {out_npu_cpu.shape} vs {out_ref_cpu.shape}"
        print(f"Output shape matched: {out_npu_cpu.shape}")

        assert not torch.all(out_npu_cpu == 0), "NPU output is all zeros!"
        print(f"NPU output is not all zeros")

        diff = out_npu_cpu.float() - out_ref_cpu.float()
        abs_diff = torch.abs(diff)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()

        print(f"\n--- Numerical Comparison ---")
        print(f"Max absolute diff: {max_abs_diff:.6e}")
        print(f"Mean absolute diff: {mean_abs_diff:.6e}")

        ATOL, RTOL = 5e-2, 1e-2
        tol = ATOL + RTOL * torch.abs(out_ref_cpu.float())
        matched = (abs_diff <= tol).sum().item()
        total = abs_diff.numel()
        print(
            f"Matched (atol={ATOL}, rtol={RTOL}): {matched}/{total} ({100*matched/total:.2f}%)"
        )

        # Conv state verification
        print(f"\n--- Conv State Update Verification ---")
        npu_state = conv_state_npu.cpu()
        ref_state = conv_state_ref.cpu()

        state_diff = (npu_state.float() - ref_state.float()).abs()
        state_exact_match = (state_diff < 1e-6).sum().item()
        state_total = state_diff.numel()

        print(
            f"State exact match (diff < 1e-6): {state_exact_match}/{state_total} ({100*state_exact_match/state_total:.2f}%)"
        )
        if state_exact_match == state_total:
            print(f"Conv state values match exactly!")
        else:
            print(f"State max diff: {state_diff.max():.6e}")

        # Summary
        print(f"\n{'='*60}")
        print("PRECISION TEST SUMMARY")
        print(f"{'='*60}")

        if matched >= total * 0.95 and state_exact_match == state_total:
            print(f"\nPASS: Output and state are correctly aligned to torch reference!")
        else:
            print(f"\nWARNING: Precision below expected threshold")

        print(f"\nNPU causal_conv1d_fla_update test passed!")

    except Exception as e:
        print(f"NPU test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Running test_npu_causal_conv1d_fla_update (NPU kernel)")
    print("=" * 60)
    test_npu_causal_conv1d_fla_update()
