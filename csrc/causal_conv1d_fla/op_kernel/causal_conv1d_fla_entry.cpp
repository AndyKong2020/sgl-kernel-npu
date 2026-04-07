/*!
 * \file causal_conv1d_fla_entry.cpp
 * \brief Kernel entry points for causal_conv1d_fla (FLA optimized).
 *        Wraps the PR #50 kernel classes with explicit extern "C" entries
 *        for the sgl-kernel-npu PyTorch extension build system.
 */

#include "causal_conv1d_fn.h"
#include "causal_conv1d_update.h"

namespace {

__aicore__ inline int64_t ResolveFnPlan(int64_t baseDimCnt)
{
    if (baseDimCnt <= 0) {
        return FN_EXECUTION_PLAN_INVALID;
    }
    if (baseDimCnt <= 1) {
        return FN_EXECUTION_PLAN_CUTBS;
    }
    return FN_EXECUTION_PLAN_CUTBSD;
}

__aicore__ inline void LoadTilingData(GM_ADDR tiling, CausalConv1dTilingData &td)
{
    auto *src = reinterpret_cast<__gm__ uint8_t *>(tiling);
    auto *dst = reinterpret_cast<uint8_t *>(&td);
    for (uint32_t i = 0; i < sizeof(CausalConv1dTilingData); ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
__aicore__ inline void DispatchPrefill(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                       GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                       GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                       const CausalConv1dTilingData *tilingData)
{
    const int64_t fnPlan = ResolveFnPlan(tilingData->baseDimCnt);
    const int64_t width = tilingData->width;

    if (fnPlan == FN_EXECUTION_PLAN_CUTBS) {
        if (width == 4) {
            NsCausalConv1d::RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_4, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else if (width == 3) {
            NsCausalConv1d::RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_3, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else {
            NsCausalConv1d::RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_2, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        }
    } else {
        if (width == 4) {
            NsCausalConv1d::RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_4, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else if (width == 3) {
            NsCausalConv1d::RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_3, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        } else {
            NsCausalConv1d::RunCausalConv1dFn<T, CAUSAL_CONV1D_TPL_WIDTH_2, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD>(
                x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
                workspace, tilingData);
        }
    }
}

}  // namespace

extern "C" __global__ __aicore__ void causal_conv1d_fla_prefill_bfloat16_t(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
    GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dTilingData td;
    LoadTilingData(tiling, td);
    DispatchPrefill<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode,
                                numAcceptedTokens, y, nullptr, &td);
}

extern "C" __global__ __aicore__ void causal_conv1d_fla_prefill_half(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
    GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dTilingData td;
    LoadTilingData(tiling, td);
    DispatchPrefill<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode,
                         numAcceptedTokens, y, nullptr, &td);
}

extern "C" __global__ __aicore__ void causal_conv1d_fla_update_bfloat16_t(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
    GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dTilingData td;
    LoadTilingData(tiling, td);
    NsCausalConv1d::RunCausalConv1dUpdate<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices,
                                                      initialStateMode, numAcceptedTokens, y, nullptr, &td);
}

extern "C" __global__ __aicore__ void causal_conv1d_fla_update_half(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
    GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dTilingData td;
    LoadTilingData(tiling, td);
    NsCausalConv1d::RunCausalConv1dUpdate<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices,
                                                initialStateMode, numAcceptedTokens, y, nullptr, &td);
}
