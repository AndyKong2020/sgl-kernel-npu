/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update_tiling.h
 * \brief Host-side tiling computation, aligned with vllm-ascend ChooseDimTileSize
 */

#ifndef CAUSAL_CONV1D_UPDATE_TILING_HOST_H_
#define CAUSAL_CONV1D_UPDATE_TILING_HOST_H_

#include <cstdint>
#include <limits>

// Host-side tiling data struct (must match kernel-side exactly)
struct CausalConv1dUpdateTilingData {
    int64_t dim;
    int64_t seqLen;
    int64_t batch;
    int64_t width;
    int64_t stateLen;

    int64_t activationMode;
    int64_t padSlotId;

    int64_t hasBias;
    int64_t hasIndices;
    int64_t hasNumAccept;
    int64_t hasQueryLoc;

    int64_t dimTileSize;
    int64_t blocksPerSeq;
};

namespace SGLang {
namespace CausalConv1dUpdate {

static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

struct DimTileChoice {
    int64_t dimTileSize = 0;
    int64_t blocksPerSeq = 0;
    int64_t gridSize = 0;
};

// Aligned with vllm-ascend ChooseDimTileSize
static inline DimTileChoice ChooseDimTileSize(int64_t batch, int64_t dim, int32_t coreNum)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};

    auto ChooseOnce = [&](bool requireExactDiv) -> DimTileChoice {
        DimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        DimTileChoice bestUnder;

        for (int64_t dimTileSize : candidates) {
            if (dimTileSize <= 0) {
                continue;
            }

            int64_t blocksPerSeq;
            if (requireExactDiv) {
                if (dim % dimTileSize != 0) {
                    continue;
                }
                blocksPerSeq = dim / dimTileSize;
            } else {
                blocksPerSeq = CeilDiv(dim, dimTileSize);
            }

            const int64_t gridSize = batch * blocksPerSeq;
            if (gridSize <= 0) {
                continue;
            }

            if (gridSize >= static_cast<int64_t>(coreNum)) {
                const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
                if (gap < bestOverGap) {
                    bestOver = {dimTileSize, blocksPerSeq, gridSize};
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && dimTileSize > bestUnder.dimTileSize)) {
                bestUnder = {dimTileSize, blocksPerSeq, gridSize};
            }
        }

        if (bestOver.dimTileSize > 0) return bestOver;
        if (bestUnder.dimTileSize > 0) return bestUnder;
        return {0, 0, 0};
    };

    // Try exact division first, then fall back to non-exact
    DimTileChoice choice = ChooseOnce(true);
    if (choice.dimTileSize > 0) return choice;
    choice = ChooseOnce(false);
    if (choice.dimTileSize > 0) return choice;
    // Fallback: one tile = full dim (must fit in MAX_BLOCK_DIM=4096)
    return {dim, 1, batch};
}

// Compute tiling data for kernel launch
inline void ComputeTilingData(
    const int64_t batch,
    const int64_t seq_len,
    const int64_t dim,
    const int64_t width,
    const int64_t state_len,
    const bool has_indices,
    const bool has_bias,
    const bool has_num_accept,
    const bool has_query_loc,
    const bool activation_mode,
    const int64_t pad_slot_id,
    const int32_t num_cores,
    CausalConv1dUpdateTilingData& tiling_data
) {
    tiling_data.dim = dim;
    tiling_data.seqLen = seq_len;
    tiling_data.batch = batch;
    tiling_data.width = width;
    tiling_data.stateLen = state_len;
    tiling_data.activationMode = activation_mode ? 1 : 0;
    tiling_data.padSlotId = pad_slot_id;
    tiling_data.hasBias = has_bias ? 1 : 0;
    tiling_data.hasIndices = has_indices ? 1 : 0;
    tiling_data.hasNumAccept = has_num_accept ? 1 : 0;
    tiling_data.hasQueryLoc = has_query_loc ? 1 : 0;

    auto choice = ChooseDimTileSize(batch, dim, num_cores);
    tiling_data.dimTileSize = choice.dimTileSize;
    tiling_data.blocksPerSeq = choice.blocksPerSeq;
}

} // namespace CausalConv1dUpdate
} // namespace SGLang

#endif // CAUSAL_CONV1D_UPDATE_TILING_HOST_H_
