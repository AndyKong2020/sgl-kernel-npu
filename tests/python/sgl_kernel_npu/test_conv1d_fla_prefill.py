import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1


@dataclass
class CaseConfig:
    name: str
    dtype: torch.dtype
    dim: int
    width: int
    state_len: int
    num_cache_lines: int
    activation_mode: bool
    use_bias: bool
    input_mode: str
    batch: int
    seq_len: Optional[int] = None
    lengths: Optional[list[int]] = None
    cache_indices: Optional[list[int]] = None
    has_initial_state: Optional[list[bool]] = None


def make_query_start_loc(lengths: Iterable[int], device: torch.device) -> torch.Tensor:
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + int(length))
    out = torch.empty((len(qsl),), device=device, dtype=torch.int64)
    for idx, value in enumerate(qsl):
        out[idx] = int(value)
    return out


def make_device_long_tensor(
    values: Iterable[int], device: torch.device
) -> torch.Tensor:
    values = list(values)
    out = torch.empty((len(values),), device=device, dtype=torch.int64)
    for idx, value in enumerate(values):
        out[idx] = int(value)
    return out


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x


def reference_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    activation_mode: bool = False,
    pad_slot_id: int = PAD_SLOT_ID,
):
    width = weight.shape[0]
    state_prefix = width - 1
    dim = x.shape[-1]
    x_tokens = flatten_tokens(x)
    batch = x.shape[0] if x.dim() == 3 else query_start_loc.numel() - 1
    seq_len = x.shape[1] if x.dim() == 3 else None

    y_ref = torch.zeros((x_tokens.shape[0], dim), device=x.device, dtype=torch.float32)
    valid_mask = torch.zeros((x_tokens.shape[0],), device="cpu", dtype=torch.bool)
    conv_states_ref = conv_states.clone()

    weight_fp32 = weight.float()
    bias_fp32 = bias.float() if bias is not None else None

    for seq in range(batch):
        if x.dim() == 3:
            start = seq * seq_len
            length = seq_len
        else:
            start = int(query_start_loc[seq].item())
            end = int(query_start_loc[seq + 1].item())
            length = end - start

        if length <= 0:
            continue

        cache_idx = int(cache_indices[seq].item()) if cache_indices is not None else seq
        if cache_idx == pad_slot_id:
            continue

        valid_mask[start : start + length] = True

        use_init = False
        if has_initial_state is not None:
            use_init = bool(has_initial_state[seq].item())

        if use_init:
            hist_raw = conv_states[cache_idx, :state_prefix].clone()
        else:
            hist_raw = torch.zeros((state_prefix, dim), device=x.device, dtype=x.dtype)

        x_seg_raw = x_tokens[start : start + length]
        x_ext_raw = torch.cat([hist_raw, x_seg_raw], dim=0)
        x_ext = x_ext_raw.float()

        acc = torch.zeros((length, dim), device=x.device, dtype=torch.float32)
        for w in range(width):
            acc = acc + x_ext[w : w + length] * weight_fp32[w]

        if bias_fp32 is not None:
            acc = acc + bias_fp32
        if activation_mode:
            acc = F.silu(acc)

        y_ref[start : start + length] = acc.to(x.dtype).float()
        conv_states_ref[cache_idx, :state_prefix] = x_ext_raw[-state_prefix:]

    return y_ref, conv_states_ref, valid_mask


def make_case_tensors(case: CaseConfig, device: torch.device, pad_slot_id: int):
    if case.input_mode == "3d":
        assert case.seq_len is not None
        x = torch.randn(
            (case.batch, case.seq_len, case.dim), device=device, dtype=case.dtype
        )
        lengths = [case.seq_len] * case.batch
    else:
        assert case.lengths is not None
        x = torch.randn((sum(case.lengths), case.dim), device=device, dtype=case.dtype)
        lengths = case.lengths

    weight = torch.randn((case.width, case.dim), device=device, dtype=case.dtype)
    bias = (
        torch.randn((case.dim,), device=device, dtype=case.dtype)
        if case.use_bias
        else None
    )
    conv_states = torch.randn(
        (case.num_cache_lines, case.state_len, case.dim),
        device=device,
        dtype=case.dtype,
    )
    query_start_loc = make_query_start_loc(lengths, device)
    cache_indices = make_device_long_tensor(case.cache_indices, device)
    has_initial_state = make_device_long_tensor(
        [1 if v else 0 for v in case.has_initial_state], device
    )

    return (
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
    )


def summarize_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return diff.max().item(), diff.mean().item()


def run_positive_case(
    case: CaseConfig, device: torch.device, atol: float, rtol: float, pad_slot_id: int
):
    host_device = torch.device("cpu")
    (
        x_cpu,
        weight_cpu,
        bias_cpu,
        conv_states_cpu,
        query_start_loc_cpu,
        cache_indices_cpu,
        has_initial_state_cpu,
    ) = make_case_tensors(case, host_device, pad_slot_id)
    lengths = [case.seq_len] * case.batch if case.input_mode == "3d" else case.lengths
    x = x_cpu.to(device=device)
    weight = weight_cpu.to(device=device)
    bias = bias_cpu.to(device=device) if bias_cpu is not None else None
    conv_states_npu = conv_states_cpu.to(device=device)
    query_start_loc = make_query_start_loc(lengths, device)
    cache_indices = make_device_long_tensor(case.cache_indices, device)
    has_initial_state = make_device_long_tensor(
        [1 if v else 0 for v in case.has_initial_state], device
    )

    y_ref, conv_states_ref, valid_mask = reference_causal_conv1d(
        x=x_cpu,
        weight=weight_cpu,
        conv_states=conv_states_cpu,
        query_start_loc=query_start_loc_cpu,
        cache_indices=cache_indices_cpu,
        has_initial_state=has_initial_state_cpu,
        bias=bias_cpu,
        activation_mode=case.activation_mode,
        pad_slot_id=pad_slot_id,
    )

    y_npu = torch.ops.npu.causal_conv1d_fla_prefill(
        x,
        weight,
        conv_states_npu,
        query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=has_initial_state,
        bias=bias,
        activation_mode=case.activation_mode,
        pad_slot_id=pad_slot_id,
    )
    torch.npu.synchronize()

    valid_mask_cpu = valid_mask
    y_ref_cpu = y_ref
    y_npu_cpu = flatten_tokens(y_npu).cpu().float()
    conv_states_ref_cpu = conv_states_ref.float()
    conv_states_npu_cpu = conv_states_npu.cpu().float()

    y_ref_valid = y_ref_cpu[valid_mask_cpu]
    y_npu_valid = y_npu_cpu[valid_mask_cpu]
    if y_ref_valid.numel() > 0:
        torch.testing.assert_close(y_npu_valid, y_ref_valid, atol=atol, rtol=rtol)

    torch.testing.assert_close(
        conv_states_npu_cpu, conv_states_ref_cpu, atol=0.0, rtol=0.0
    )

    out_max_abs_diff, out_mean_abs_diff = (
        summarize_diff(y_npu_valid, y_ref_valid)
        if y_ref_valid.numel() > 0
        else (0.0, 0.0)
    )
    state_max_abs_diff, state_mean_abs_diff = summarize_diff(
        conv_states_npu_cpu, conv_states_ref_cpu
    )

    print(
        f"[PASS] {case.name}: "
        f"output(max={out_max_abs_diff:.6g}, mean={out_mean_abs_diff:.6g}) "
        f"state(max={state_max_abs_diff:.6g}, mean={state_mean_abs_diff:.6g})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--pad-slot-id", type=int, default=PAD_SLOT_ID)
    args = parser.parse_args()

    try:
        import sgl_kernel_npu  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Import failed: {exc}") from exc

    if not hasattr(torch.ops.npu, "causal_conv1d_fla_prefill"):
        raise SystemExit("torch.ops.npu.causal_conv1d_fla_prefill is not registered")

    if not hasattr(torch, "npu") or torch.npu.device_count() <= 0:
        raise SystemExit("NPU device is not available")

    torch.manual_seed(args.seed)
    device = torch.device("npu")

    positive_cases = [
        CaseConfig(
            name="dense3d_all_zero_no_bias",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=3,
            num_cache_lines=8,
            activation_mode=False,
            use_bias=False,
            input_mode="3d",
            batch=2,
            seq_len=6,
            cache_indices=[0, 1],
            has_initial_state=[False, False],
        ),
        CaseConfig(
            name="dense3d_mixed_bias_act",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=5,
            num_cache_lines=24,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=3,
            seq_len=4,
            cache_indices=[5, 12, 20],
            has_initial_state=[True, False, True],
        ),
        CaseConfig(
            name="varlen2d_all_one_bias_act",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=6,
            num_cache_lines=12,
            activation_mode=True,
            use_bias=True,
            input_mode="2d",
            batch=3,
            lengths=[3, 5, 2],
            cache_indices=[2, 4, 8],
            has_initial_state=[True, True, True],
        ),
        CaseConfig(
            name="varlen2d_mixed_pad_no_bias",
            dtype=torch.bfloat16,
            dim=4096,
            width=4,
            state_len=5,
            num_cache_lines=20,
            activation_mode=False,
            use_bias=False,
            input_mode="2d",
            batch=4,
            lengths=[2, 4, 1, 3],
            cache_indices=[3, args.pad_slot_id, 9, 15],
            has_initial_state=[True, False, False, True],
        ),
        CaseConfig(
            name="dense3d_fp16_bias_act",
            dtype=torch.float16,
            dim=1024,
            width=4,
            state_len=4,
            num_cache_lines=10,
            activation_mode=True,
            use_bias=True,
            input_mode="3d",
            batch=2,
            seq_len=5,
            cache_indices=[1, 7],
            has_initial_state=[True, False],
        ),
    ]

    for case in positive_cases:
        run_positive_case(
            case,
            device=device,
            atol=args.atol,
            rtol=args.rtol,
            pad_slot_id=args.pad_slot_id,
        )

    print("All causal_conv1d_fla prefill tests passed.")


if __name__ == "__main__":
    main()
