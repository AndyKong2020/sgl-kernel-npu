#include "kernel_operator.h"
#include "causal_conv1d_tiling.h"

#pragma push_macro("__global__")
#undef __global__
#define __global__ inline
#define causal_conv1d_fp32 causal_conv1d_fp32_impl
#include "causal_conv1d.cpp"
#undef causal_conv1d_fp32
#pragma pop_macro("__global__")

extern "C" __global__ __aicore__ void causal_conv1d_prefill_fp32(
    GM_ADDR xmtx,
    GM_ADDR wmtx,
    GM_ADDR bias,
    GM_ADDR outmtx,
    GM_ADDR workspace,
    GM_ADDR tiling_gm)
{
    auto *tiling = reinterpret_cast<__gm__ sglang::npu_kernel::CausalConv1dTilingData *>(tiling_gm);
    causal_conv1d_fp32_impl(
        xmtx,
        wmtx,
        bias,
        outmtx,
        workspace,
        tiling->batch,
        tiling->dim,
        tiling->seqlen,
        tiling->width);
}
