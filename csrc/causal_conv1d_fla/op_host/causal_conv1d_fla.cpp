/*!
 * \file causal_conv1d_fla.cpp
 * \brief Host-side wrapper for causal_conv1d_fla (FLA optimized).
 *        Supports both prefill (runMode=0) and decode (runMode=1).
 */

#include "causal_conv1d_fla.h"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "common.h"
#include "causal_conv1d_tiling_data.h"
#include "stub/aclrtlaunch_causal_conv1d_fla_prefill_bfloat16_t.h"
#include "stub/aclrtlaunch_causal_conv1d_fla_prefill_half.h"
#include "stub/aclrtlaunch_causal_conv1d_fla_update_bfloat16_t.h"
#include "stub/aclrtlaunch_causal_conv1d_fla_update_half.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {
namespace {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr int64_t DIM_ALIGN_ELEMS = 16;
constexpr int64_t MAX_DIM_TILE_SIZE = 4096;
constexpr int64_t FN_UB_RESERVED_BYTES = 16 * 1024;  // Reserve 16KB for TPipe overhead and event IDs
constexpr int64_t RING_SLOT_CNT = 5;
constexpr int64_t FN_OUT_SLOT_CNT = 2;
constexpr int64_t FN_CALC_FP32_SLOT_CNT = 8;
constexpr int64_t BF16_FP16_ELEM_BYTES = 2;

// --- Tiling helpers (ported from PR #50) ---

inline int64_t CeilDivInt64(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t AlignDownInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return (value / align) * align;
}

inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return CeilDivInt64(value, align) * align;
}

struct DimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

struct VarlenTokenTileChoice {
    bool enabled = false;
    int64_t tokenBlockSize = 0;
    int64_t tokenBlockCnt = 0;
    int64_t gridSize = 0;
};

struct TokenCoreMappingChoice {
    int64_t tokenCoreBudget = 0;
    int64_t tokenBlocksPerCore = 0;
    int64_t tokenCoreTailCnt = 0;
    int64_t blockDim = 0;
};

DimTileChoice ChooseCanonicalUpdateBaseDimChoice(int64_t batch, int64_t dim, int32_t coreNum)
{
    constexpr int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};

    auto chooseOnce = [&](bool requireExactDiv) -> DimTileChoice {
        DimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        DimTileChoice bestUnder;

        for (int64_t baseDim : candidates) {
            if (requireExactDiv && (dim % baseDim != 0)) {
                continue;
            }
            const int64_t baseDimCnt = requireExactDiv ? (dim / baseDim) : CeilDivInt64(dim, baseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }
            if (gridSize >= static_cast<int64_t>(coreNum)) {
                const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
                if (gap < bestOverGap) {
                    bestOver = {baseDim, baseDimCnt, gridSize};
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && baseDim < bestUnder.baseDim)) {
                bestUnder = {baseDim, baseDimCnt, gridSize};
            }
        }
        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    DimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    return result;
}

int64_t ComputeFnUbLimitedBaseDim(uint64_t ubSize)
{
    if (ubSize <= static_cast<uint64_t>(FN_UB_RESERVED_BYTES)) {
        return 0;
    }
    const int64_t bytesPerElem = (RING_SLOT_CNT * BF16_FP16_ELEM_BYTES) + (FN_OUT_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                  (FN_CALC_FP32_SLOT_CNT * static_cast<int64_t>(sizeof(float)));
    const int64_t budgetBytes = static_cast<int64_t>(ubSize) - FN_UB_RESERVED_BYTES;
    const int64_t ubLimitedBaseDim = AlignDownInt64(budgetBytes / bytesPerElem, DIM_ALIGN_ELEMS);
    return std::min<int64_t>(MAX_DIM_TILE_SIZE, ubLimitedBaseDim);
}

DimTileChoice ChooseFnTokenFirstBaseDimChoice(int64_t dim, uint64_t ubSize)
{
    if (dim <= 0) {
        return {};
    }
    // Check if dim fits in UB; if not, split dim.
    const int64_t ubLimitedBaseDim = ComputeFnUbLimitedBaseDim(ubSize);
    if (ubLimitedBaseDim <= 0) {
        return {};
    }
    const int64_t baseDim = std::min<int64_t>(dim, std::min<int64_t>(ubLimitedBaseDim, MAX_DIM_TILE_SIZE));
    const int64_t baseDimCnt = CeilDivInt64(dim, baseDim);
    return {baseDim, baseDimCnt, baseDimCnt};
}

DimTileChoice ChooseFnTokenDimCoSplitBaseDimChoice(int64_t dim, uint64_t ubSize, int32_t coreNum)
{
    if (dim <= 0) {
        return {};
    }
    const int64_t ubLimitedBaseDim = ComputeFnUbLimitedBaseDim(ubSize);
    if (ubLimitedBaseDim <= 0) {
        return {};
    }

    DimTileChoice result;
    result.baseDim = ubLimitedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;

    if (coreNum == 0 || result.baseDimCnt <= 1 || result.baseDimCnt >= static_cast<int64_t>(coreNum) ||
        (coreNum % result.baseDimCnt == 0)) {
        return result;
    }

    int64_t adjustedBaseDimCnt = result.baseDimCnt;
    while (adjustedBaseDimCnt < static_cast<int64_t>(coreNum) && (coreNum % adjustedBaseDimCnt != 0)) {
        ++adjustedBaseDimCnt;
    }
    if (adjustedBaseDimCnt >= static_cast<int64_t>(coreNum)) {
        return result;
    }

    const int64_t adjustedBaseDim = AlignUpInt64(CeilDivInt64(dim, adjustedBaseDimCnt), DIM_ALIGN_ELEMS);
    if (adjustedBaseDim <= 0 || adjustedBaseDim > ubLimitedBaseDim || adjustedBaseDim > MAX_DIM_TILE_SIZE) {
        return result;
    }
    result.baseDim = adjustedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;
    return result;
}

int64_t ResolveFnTokenCoreBudget(int64_t baseDimCnt, int64_t fnPlan, int32_t coreNum)
{
    if (baseDimCnt <= 0 || coreNum == 0 || fnPlan == FN_EXECUTION_PLAN_INVALID) {
        return 0;
    }
    int64_t budget = static_cast<int64_t>(coreNum);
    if (fnPlan == FN_EXECUTION_PLAN_CUTBSD) {
        budget = std::max<int64_t>(1, budget / baseDimCnt);
    }
    return budget;
}

VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt, int64_t fnPlan, int32_t coreNum)
{
    VarlenTokenTileChoice choice;
    const int64_t budget = ResolveFnTokenCoreBudget(baseDimCnt, fnPlan, coreNum);
    if (cuSeqlen <= 0 || budget <= 0) {
        return choice;
    }
    choice.enabled = true;
    const int64_t idealBlockSize = CeilDivInt64(cuSeqlen, budget);
    choice.tokenBlockSize = (idealBlockSize > 0) ? idealBlockSize : 1;
    choice.tokenBlockCnt = CeilDivInt64(cuSeqlen, choice.tokenBlockSize);
    choice.gridSize = choice.tokenBlockCnt * baseDimCnt;
    return choice;
}

TokenCoreMappingChoice BuildFnTokenCoreMappingChoice(int64_t tokenBlockCnt, int64_t baseDimCnt, int64_t fnPlan,
                                                      int32_t coreNum)
{
    TokenCoreMappingChoice mapping;
    mapping.tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnPlan, coreNum);
    if (tokenBlockCnt <= 0 || mapping.tokenCoreBudget <= 0 || baseDimCnt <= 0) {
        return mapping;
    }
    mapping.tokenBlocksPerCore = CeilDivInt64(tokenBlockCnt, mapping.tokenCoreBudget);
    mapping.tokenCoreTailCnt =
        tokenBlockCnt - (std::max<int64_t>(0, mapping.tokenBlocksPerCore - 1) * mapping.tokenCoreBudget);
    if (mapping.tokenCoreTailCnt <= 0) {
        mapping.tokenCoreTailCnt = mapping.tokenCoreBudget;
    }
    mapping.blockDim = mapping.tokenCoreBudget * baseDimCnt;
    return mapping;
}

void CheckSameDevice(const at::Tensor &lhs, const at::Tensor &rhs, const char *lhs_name, const char *rhs_name)
{
    TORCH_CHECK(lhs.device() == rhs.device(), lhs_name, " and ", rhs_name, " must be on the same device");
}

void ValidateCommonInputs(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &conv_states,
                           const at::Tensor &bias)
{
    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported, got ", dtype);
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dtype, "conv_states dtype must match x dtype");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [width, dim], got shape ", weight.sizes());
    TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D [num_cache_lines, state_len, dim], got shape ",
                conv_states.sizes());

    const bool has_bias = bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D [dim], got shape ", bias.sizes());
        TORCH_CHECK(bias.scalar_type() == dtype, "bias dtype must match x dtype");
    }

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_states.is_contiguous(), "conv_states must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
}

void FillTilingCommon(CausalConv1dTilingData &td, int64_t dim, int64_t cuSeqlen, int64_t seqLen, int64_t inputMode,
                      int64_t width, int64_t stateLen, int64_t numCacheLines, int64_t batch, bool activation_mode,
                      int64_t pad_slot_id, bool has_bias)
{
    td.dim = dim;
    td.cuSeqlen = cuSeqlen;
    td.seqLen = seqLen;
    td.inputMode = inputMode;
    td.width = width;
    td.stateLen = stateLen;
    td.numCacheLines = numCacheLines;
    td.batch = batch;
    td.activationMode = activation_mode ? 1 : 0;
    td.padSlotId = pad_slot_id;
    td.hasBias = has_bias ? 1 : 0;
}

constexpr uint32_t MAX_CAPTURE_NUM = 1024;

uint64_t HashTilingData(const CausalConv1dTilingData &td)
{
    // Hash the scalar fields that determine the tiling configuration.
    // Array fields (tokenTileStartSeq/EndSeq) are derived from these, so not needed.
    auto tup = std::make_tuple(td.dim, td.cuSeqlen, td.seqLen, td.inputMode, td.width, td.stateLen,
                               td.numCacheLines, td.batch, td.activationMode, td.padSlotId, td.hasBias,
                               td.baseDim, td.baseDimCnt, td.hasNumAcceptedTokens, td.hasCacheIndices,
                               td.hasInitialStateMode, td.tokenBlockSize, td.tokenBlockCnt,
                               td.hasExplicitTokenSeqRanges, td.explicitTokenSeqRangeCount);
    return static_cast<uint64_t>(host_utils::TupleHasher::Hash(tup));
}

// Graph-mode compatible tiling cache: pre-allocated device buffer + hash dedup.
// On first encounter of a tiling config, copy H2D into a slot; on subsequent calls
// with the same config, return a zero-copy view of the cached slot.

static std::unordered_map<uint64_t, uint32_t> g_tilingCaptureMap;
static uint32_t g_tilingCaptureNum = 0;

bool TilingCacheHit(uint64_t hashValue)
{
    return g_tilingCaptureMap.find(hashValue) != g_tilingCaptureMap.end();
}

at::Tensor MakeTilingHostTensor(const CausalConv1dTilingData &td, int32_t tiling_size)
{
    auto host_buf = at::empty({tiling_size}, at::kByte);
    memcpy(host_buf.data_ptr<uint8_t>(), &td, sizeof(CausalConv1dTilingData));
    return host_buf;
}

void CopyTilingToDevice(const CausalConv1dTilingData &td, at::Tensor &tiling_tensor, const at::Tensor &x,
                        uint64_t hashValue)
{
    const int32_t tiling_size =
        static_cast<int32_t>((sizeof(CausalConv1dTilingData) + PADDING_BYTE - 1U) / PADDING_BYTE * PADDING_BYTE);

    static auto globalTilingBuffer = at::empty({tiling_size * static_cast<int64_t>(MAX_CAPTURE_NUM)},
                                               at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    if (TilingCacheHit(hashValue)) {
        // Cache hit: zero-copy view of existing slot
        const uint32_t slot = g_tilingCaptureMap[hashValue];
        tiling_tensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + tiling_size * slot,
                                      {tiling_size}, at::kByte);
    } else if (g_tilingCaptureNum >= MAX_CAPTURE_NUM) {
        // Overflow: one-shot copy via PyTorch H2D (graph-capture safe)
        auto host_buf = MakeTilingHostTensor(td, tiling_size);
        tiling_tensor = TorchNpuHelper::CopyTensorHostToDevice(host_buf);
    } else {
        // New config: copy to global buffer slot via PyTorch H2D (graph-capture safe)
        const uint32_t slot = g_tilingCaptureNum++;
        g_tilingCaptureMap[hashValue] = slot;
        auto host_buf = MakeTilingHostTensor(td, tiling_size);
        auto dev_buf = TorchNpuHelper::CopyTensorHostToDevice(host_buf);
        auto dst = globalTilingBuffer.narrow(0, tiling_size * slot, tiling_size);
        dst.copy_(dev_buf);
        tiling_tensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + tiling_size * slot,
                                      {tiling_size}, at::kByte);
    }
}

}  // namespace

// =============================================
// Prefill implementation (runMode = 0)
// =============================================
HOST_API at::Tensor causal_conv1d_fla_prefill_impl(const at::Tensor &x, const at::Tensor &weight,
                                                   const at::Tensor &conv_states, const at::Tensor &query_start_loc,
                                                   const at::Tensor &cache_indices,
                                                   const at::Tensor &initial_state_mode, const at::Tensor &bias,
                                                   bool activation_mode, int64_t pad_slot_id)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D [cu_seqlen, dim] or 3D [batch, seq_len, dim], got shape ",
                x.sizes());
    ValidateCommonInputs(x, weight, conv_states, bias);

    TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1D");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kLong, "query_start_loc dtype must be int64");

    const bool has_bias = bias.numel() > 0;
    const bool has_cache_indices = cache_indices.numel() > 0;
    const bool has_initial_state = initial_state_mode.numel() > 0;

    int64_t dim, cuSeqlen, seqLen, batch, inputMode;
    if (x.dim() == 2) {
        inputMode = 0;
        cuSeqlen = x.size(0);
        dim = x.size(1);
        seqLen = 0;
        batch = query_start_loc.size(0) - 1;
    } else {
        inputMode = 1;
        batch = x.size(0);
        seqLen = x.size(1);
        dim = x.size(2);
        cuSeqlen = batch * seqLen;
    }

    TORCH_CHECK(batch > 0, "batch must be > 0");
    const int64_t width = weight.size(0);
    TORCH_CHECK(width >= 2 && width <= 4, "Only width in [2,4] is supported, got ", width);
    TORCH_CHECK(weight.size(1) == dim, "weight.shape[1] must equal dim");
    TORCH_CHECK(dim % DIM_ALIGN_ELEMS == 0, "dim must be multiple of 16, got ", dim);

    const int64_t numCacheLines = conv_states.size(0);
    const int64_t stateLen = conv_states.size(1);
    TORCH_CHECK(conv_states.size(2) == dim, "conv_states.shape[2] must equal dim");
    TORCH_CHECK(stateLen >= width - 1, "conv_states.shape[1] must be >= width - 1");

    if (has_cache_indices) {
        TORCH_CHECK(cache_indices.dim() == 1 && cache_indices.size(0) == batch, "cache_indices must be 1D [batch]");
        TORCH_CHECK(cache_indices.scalar_type() == at::kLong, "cache_indices dtype must be int64");
    }
    if (has_initial_state) {
        TORCH_CHECK(initial_state_mode.dim() == 1 && initial_state_mode.size(0) == batch,
                    "initial_state_mode must be 1D [batch]");
        TORCH_CHECK(initial_state_mode.scalar_type() == at::kLong, "initial_state_mode dtype must be int64");
    }

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendc_platform != nullptr, "Failed to acquire AscendC platform manager");
    const int32_t core_num = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    TORCH_CHECK(core_num > 0, "AscendC returned invalid core_num");

    uint64_t ubSize = 0;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // Compute tiling data
    CausalConv1dTilingData td{};
    FillTilingCommon(td, dim, cuSeqlen, seqLen, inputMode, width, stateLen, numCacheLines, batch, activation_mode,
                     pad_slot_id, has_bias);
    td.hasCacheIndices = has_cache_indices ? 1 : 0;
    td.hasInitialStateMode = has_initial_state ? 1 : 0;
    td.hasNumAcceptedTokens = 0;

    // Choose dim tiling plan (respects UB size to avoid overflow)
    DimTileChoice baseDimChoice = ChooseFnTokenFirstBaseDimChoice(dim, ubSize);
    int64_t fnPlan;
    if (baseDimChoice.baseDimCnt <= 1) {
        fnPlan = FN_EXECUTION_PLAN_CUTBS;
    } else {
        fnPlan = FN_EXECUTION_PLAN_CUTBSD;
    }
    TORCH_CHECK(baseDimChoice.baseDim > 0 && baseDimChoice.baseDimCnt > 0, "Failed to choose valid baseDim for dim=",
                dim);

    baseDimChoice.gridSize = batch * baseDimChoice.baseDimCnt;
    td.baseDim = baseDimChoice.baseDim;
    td.baseDimCnt = baseDimChoice.baseDimCnt;

    // Token tiling
    td.tokenBlockSize = 0;
    td.tokenBlockCnt = 0;
    td.hasExplicitTokenSeqRanges = 0;
    td.explicitTokenSeqRangeCount = 0;

    VarlenTokenTileChoice tokenChoice =
        ChooseFnTokenBlockChoice(cuSeqlen, baseDimChoice.baseDimCnt, fnPlan, core_num);
    TORCH_CHECK(tokenChoice.enabled && tokenChoice.tokenBlockSize > 0 && tokenChoice.tokenBlockCnt > 0,
                "Failed to compute token tiling plan");

    td.tokenBlockSize = tokenChoice.tokenBlockSize;
    td.tokenBlockCnt = tokenChoice.tokenBlockCnt;

    // Explicit token-seq ranges for varlen mode.
    // Building these requires reading queryStartLoc from device (.cpu() triggers sync),
    // so only do it when the tiling config is new (cache miss).
    // For cache hits, the ranges are already stored in the cached tiling data.
    const bool needExplicitRanges = (inputMode == 0 && tokenChoice.tokenBlockCnt <= 128);
    if (!needExplicitRanges) {
        td.hasExplicitTokenSeqRanges = 0;
        td.explicitTokenSeqRangeCount = 0;
    }

    // Core mapping
    TokenCoreMappingChoice coreMapping =
        BuildFnTokenCoreMappingChoice(tokenChoice.tokenBlockCnt, baseDimChoice.baseDimCnt, fnPlan, core_num);
    TORCH_CHECK(coreMapping.blockDim > 0, "Failed to compute core mapping");

    // The kernel's actual grid depends on which Process path it takes:
    //   baseDimCnt > 1 → ProcessDefault, gridSize = batch * baseDimCnt
    //   baseDimCnt == 1 → ProcessVarlenTokenTiled, gridSize = tokenBlockCnt * baseDimCnt
    // Set blockDim to match so each block handles exactly one task.
    int64_t effectiveGridSize = (baseDimChoice.baseDimCnt > 1)
        ? (batch * baseDimChoice.baseDimCnt)
        : tokenChoice.gridSize;
    int32_t block_dim = static_cast<int32_t>(effectiveGridSize);
    if (block_dim <= 0) {
        block_dim = 1;
    }

    const int64_t workspace_size = static_cast<int64_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    // Check tiling cache before building expensive explicit ranges.
    // Only compute token-seq ranges on cache miss to avoid .cpu() sync on every call.
    const uint64_t tilingHash = HashTilingData(td);
    if (needExplicitRanges && !TilingCacheHit(tilingHash)) {
        at::Tensor qsl_cpu = query_start_loc.cpu();
        const int64_t *qslData = qsl_cpu.data_ptr<int64_t>();
        td.hasExplicitTokenSeqRanges = 1;
        td.explicitTokenSeqRangeCount = tokenChoice.tokenBlockCnt;
        int64_t seq = 0;
        for (int64_t tid = 0; tid < tokenChoice.tokenBlockCnt; ++tid) {
            const int64_t tokenStart = tid * tokenChoice.tokenBlockSize;
            const int64_t tokenEnd = tokenStart + tokenChoice.tokenBlockSize;
            while (seq < batch && qslData[seq + 1] <= tokenStart) {
                ++seq;
            }
            int64_t endSeq = seq;
            while (endSeq < batch && qslData[endSeq] < tokenEnd) {
                ++endSeq;
            }
            td.tokenTileStartSeq[tid] = seq;
            td.tokenTileEndSeq[tid] = endSeq;
        }
    }

    at::Tensor tiling_tensor;
    CopyTilingToDevice(td, tiling_tensor, x, tilingHash);

    auto byte_options = x.options().dtype(at::kByte);
    at::Tensor workspace_tensor = at::empty({workspace_size}, byte_options);
    at::Tensor y = at::empty_like(x);

    at::Tensor empty_data = at::empty({0}, x.options());
    at::Tensor empty_long = at::empty({0}, x.options().dtype(at::kLong));
    at::Tensor bias_arg = has_bias ? bias : empty_data;
    at::Tensor ci_arg = has_cache_indices ? cache_indices : empty_long;
    at::Tensor ism_arg = has_initial_state ? initial_state_mode : empty_long;
    at::Tensor nat_arg = empty_long;  // numAcceptedTokens unused for prefill


    if (x.scalar_type() == at::kBFloat16) {
        EXEC_KERNEL_CMD(causal_conv1d_fla_prefill_bfloat16_t, block_dim, x, weight, bias_arg, conv_states,
                        query_start_loc, ci_arg, ism_arg, nat_arg, y, workspace_tensor, tiling_tensor);
    } else {
        EXEC_KERNEL_CMD(causal_conv1d_fla_prefill_half, block_dim, x, weight, bias_arg, conv_states, query_start_loc,
                        ci_arg, ism_arg, nat_arg, y, workspace_tensor, tiling_tensor);
    }

    return y;
}

// =============================================
// Decode/Update implementation (runMode = 1)
// =============================================
HOST_API at::Tensor causal_conv1d_fla_update_impl(const at::Tensor &x, const at::Tensor &weight,
                                                  const at::Tensor &conv_state, const at::Tensor &conv_state_indices,
                                                  const at::Tensor &bias, const at::Tensor &num_accepted_tokens,
                                                  const at::Tensor &query_start_loc, bool activation_mode,
                                                  int64_t pad_slot_id)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D [batch, dim] or 3D [batch, seq_len, dim], got shape ",
                x.sizes());
    ValidateCommonInputs(x, weight, conv_state, bias);

    const bool has_bias = bias.numel() > 0;
    const bool has_indices = conv_state_indices.numel() > 0;
    const bool has_num_accept = num_accepted_tokens.numel() > 0;
    const bool has_query_loc = query_start_loc.numel() > 0;

    int64_t dim, batch, seqLen, cuSeqlen, inputMode;
    if (x.dim() == 2) {
        inputMode = 2;  // decode 2D
        batch = x.size(0);
        dim = x.size(1);
        seqLen = 1;
        cuSeqlen = batch;
    } else {
        inputMode = 1;
        batch = x.size(0);
        seqLen = x.size(1);
        dim = x.size(2);
        cuSeqlen = batch * seqLen;
    }

    TORCH_CHECK(batch > 0, "batch must be > 0");
    const int64_t width = weight.size(0);
    TORCH_CHECK(width >= 2 && width <= 4, "Only width in [2,4] is supported, got ", width);
    TORCH_CHECK(weight.size(1) == dim, "weight.shape[1] must equal dim");
    TORCH_CHECK(dim % DIM_ALIGN_ELEMS == 0, "dim must be multiple of 16, got ", dim);

    const int64_t stateLen = conv_state.size(1);
    TORCH_CHECK(conv_state.size(2) == dim, "conv_state.shape[2] must equal dim");
    TORCH_CHECK(stateLen >= width - 1, "conv_state.shape[1] must be >= width - 1");

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendc_platform != nullptr, "Failed to acquire AscendC platform manager");
    const int32_t core_num = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    TORCH_CHECK(core_num > 0, "AscendC returned invalid core_num");

    CausalConv1dTilingData td{};
    FillTilingCommon(td, dim, cuSeqlen, seqLen, inputMode, width, stateLen, conv_state.size(0), batch, activation_mode,
                     pad_slot_id, has_bias);
    td.hasCacheIndices = has_indices ? 1 : 0;
    td.hasInitialStateMode = 0;
    td.hasNumAcceptedTokens = has_num_accept ? 1 : 0;

    // Compute dim tiling for decode
    DimTileChoice baseDimChoice = ChooseCanonicalUpdateBaseDimChoice(batch, dim, core_num);
    TORCH_CHECK(baseDimChoice.baseDim > 0 && baseDimChoice.baseDimCnt > 0, "Failed to choose valid baseDim for dim=",
                dim);

    td.baseDim = baseDimChoice.baseDim;
    td.baseDimCnt = baseDimChoice.baseDimCnt;
    td.tokenBlockSize = 0;
    td.tokenBlockCnt = 0;
    td.hasExplicitTokenSeqRanges = 0;
    td.explicitTokenSeqRangeCount = 0;

    const int64_t gridSize = baseDimChoice.gridSize;
    // Use gridSize as blockDim so that each block handles exactly one task.
    // The NPU scheduler maps blocks to physical cores internally.
    int32_t block_dim = static_cast<int32_t>(gridSize);
    if (block_dim <= 0) {
        block_dim = 1;
    }

    const int64_t workspace_size = static_cast<int64_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    at::Tensor tiling_tensor;
    CopyTilingToDevice(td, tiling_tensor, x, HashTilingData(td));

    auto byte_options = x.options().dtype(at::kByte);
    at::Tensor workspace_tensor = at::empty({workspace_size}, byte_options);
    at::Tensor y = at::empty_like(x);

    at::Tensor empty_data = at::empty({0}, x.options());
    at::Tensor empty_long = at::empty({0}, x.options().dtype(at::kLong));
    at::Tensor bias_arg = has_bias ? bias : empty_data;
    at::Tensor qsl_arg = has_query_loc ? query_start_loc : empty_long;
    at::Tensor ci_arg = has_indices ? conv_state_indices : empty_long;
    at::Tensor ism_arg = empty_long;  // initialStateMode unused for decode
    at::Tensor nat_arg = has_num_accept ? num_accepted_tokens : empty_long;

    if (x.scalar_type() == at::kBFloat16) {
        EXEC_KERNEL_CMD(causal_conv1d_fla_update_bfloat16_t, block_dim, x, weight, bias_arg, conv_state, qsl_arg,
                        ci_arg, ism_arg, nat_arg, y, workspace_tensor, tiling_tensor);
    } else {
        EXEC_KERNEL_CMD(causal_conv1d_fla_update_half, block_dim, x, weight, bias_arg, conv_state, qsl_arg, ci_arg,
                        ism_arg, nat_arg, y, workspace_tensor, tiling_tensor);
    }

    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
