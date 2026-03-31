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
 * \file causal_conv1d_update.h
 * \brief causal_conv1d_update kernel implementation.
 *        Kernel logic aligned with vllm-ascend PR#7495 causal_conv1d.h,
 *        only wrapper (tiling data parsing, entry points) differs.
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_update_tilingdata.h"
#include "causal_conv1d_update_common.h"

namespace CausalConv1dUpdateOp {

using namespace AscendC;
using sglang::npu_kernel::CausalConv1dUpdateTilingData;

template <typename T>
class CausalConv1dUpdate {
public:
    __aicore__ inline CausalConv1dUpdate() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convState, GM_ADDR convStateIndices,
                                GM_ADDR bias, GM_ADDR numAcceptedTokens, GM_ADDR queryStartLoc,
                                GM_ADDR y, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWeightAndBias(int32_t c0, int32_t dimTileSize);
    __aicore__ inline void InitRing(int32_t cacheIdx, int32_t stateTokenOffset, int32_t start, int32_t len,
                                    int32_t c0, int32_t dimTileSize, int32_t dim);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim);
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();

private:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;

    TEventID weightBiasMte2ToVEvent_;
    TEventID stateMte2ToVEvent_;
    TEventID inputMte2ToVEvent_[RING_SLOTS];
    TEventID inputVToMte2Event_;
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];
    TEventID stateWritebackMte3ToVEvent_;
    TEventID stateWritebackMte3ToMte2Event_;

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int32_t> convStateIndicesGm;
    GlobalTensor<int32_t> numAcceptGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<T> yGm;

    CausalConv1dUpdateTilingData tilingData_;

    bool weightCacheValid_ {false};
    int32_t cachedC0_ {-1};
    int32_t cachedDimTileSize_ {-1};
};

// ---------------------------------------------------------------------------
// Init: parse tiling data, set up GM tensors, allocate fixed-size buffers
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convState,
                                                   GM_ADDR convStateIndices, GM_ADDR bias,
                                                   GM_ADDR numAcceptedTokens, GM_ADDR queryStartLoc,
                                                   GM_ADDR y, GM_ADDR tiling)
{
    // --- sgl-specific wrapper: parse tiling from raw GM bytes ---
    auto td = reinterpret_cast<__gm__ CausalConv1dUpdateTilingData*>(tiling);
    tilingData_.dim = td->dim;
    tilingData_.seqLen = td->seqLen;
    tilingData_.batch = td->batch;
    tilingData_.width = td->width;
    tilingData_.stateLen = td->stateLen;
    tilingData_.activationMode = td->activationMode;
    tilingData_.padSlotId = td->padSlotId;
    tilingData_.hasBias = td->hasBias;
    tilingData_.hasIndices = td->hasIndices;
    tilingData_.hasNumAccept = td->hasNumAccept;
    tilingData_.hasQueryLoc = td->hasQueryLoc;
    tilingData_.dimTileSize = td->dimTileSize;
    tilingData_.blocksPerSeq = td->blocksPerSeq;

    weightCacheValid_ = false;
    cachedC0_ = -1;
    cachedDimTileSize_ = -1;

    // --- GM tensor setup (identical pattern to vllm) ---
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    if (tilingData_.hasBias != 0) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    }
    convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(convState));
    if (tilingData_.hasIndices != 0) {
        convStateIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(convStateIndices));
    }
    if (tilingData_.hasNumAccept != 0) {
        numAcceptGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numAcceptedTokens));
    }
    if (tilingData_.hasQueryLoc != 0) {
        queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
    }
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    // --- Fixed-size buffer allocation (matches vllm) ---
    pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 3) * MAX_BLOCK_DIM * sizeof(float));

    AllocEvents();
}

// ---------------------------------------------------------------------------
// AllocEvents / ReleaseEvents: pre-allocate event IDs (matches vllm)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::AllocEvents()
{
    weightBiasMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    stateMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    for (int32_t i = 0; i < RING_SLOTS; ++i) {
        inputMte2ToVEvent_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    }
    inputVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    stateWritebackMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    stateWritebackMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
}

template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    for (int32_t i = 0; i < RING_SLOTS; ++i) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_[i]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(inputVToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
}

// ---------------------------------------------------------------------------
// LoadWeightAndBias: per-row strided copy (identical to vllm)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::LoadWeightAndBias(int32_t c0, int32_t dimTileSize)
{
    const int32_t dim = static_cast<int32_t>(tilingData_.dim);
    const int32_t width = static_cast<int32_t>(tilingData_.width);
    const int32_t jStart = MAX_WIDTH - width;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    const bool hasBias = (tilingData_.hasBias != 0);

    for (int32_t j = 0; j < width; ++j) {
        const int32_t jDst = jStart + j;
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + c0;

        if constexpr (std::is_same<T, float>::value) {
            DataCopy(weightF[jDst * MAX_BLOCK_DIM], weightGm[weightOffset], dimTileSize);
        } else {
            DataCopy(weightF.ReinterpretCast<T>()[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM],
                     weightGm[weightOffset], dimTileSize);
        }
    }

    if (hasBias) {
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(biasF, biasGm[c0], dimTileSize);
        } else {
            DataCopy(biasF.ReinterpretCast<T>()[MAX_BLOCK_DIM], biasGm[c0], dimTileSize);
        }
    }

    SetFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    WaitFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);

    if constexpr (!std::is_same<T, float>::value) {
        for (int32_t j = 0; j < width; ++j) {
            const int32_t jDst = jStart + j;
            Cast(weightF[jDst * MAX_BLOCK_DIM],
                 weightF.ReinterpretCast<T>()[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM],
                 RoundMode::CAST_NONE, dimTileSize);
        }
        if (hasBias) {
            Cast(biasF, biasF.ReinterpretCast<T>()[MAX_BLOCK_DIM],
                 RoundMode::CAST_NONE, dimTileSize);
        }
    }

    if (!hasBias) {
        Duplicate(biasF, 0.0f, dimTileSize);
    }
}

// ---------------------------------------------------------------------------
// InitRing: load conv_state into ring buffer + prefetch first input (matches vllm)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::InitRing(int32_t cacheIdx, int32_t stateTokenOffset,
                                                       int32_t start, int32_t len,
                                                       int32_t c0, int32_t dimTileSize, int32_t dim)
{
    const int32_t stateLen = static_cast<int32_t>(tilingData_.stateLen);
    const int32_t width = static_cast<int32_t>(tilingData_.width);
    const int32_t ringStart = MAX_WIDTH - width;
    LocalTensor<T> ring = inBuf.Get<T>();

    // Load existing conv_state into ring buffer
    for (int32_t i = 0; i < (width - 1); ++i) {
        const int32_t pos = stateTokenOffset + i;
        const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                    static_cast<int64_t>(pos) * dim + c0;
        DataCopy(ring[(ringStart + i) * MAX_BLOCK_DIM], convStatesGm[stateOffset], dimTileSize);
    }
    SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);

    // Prefetch first input token
    if (len > 0) {
        const int32_t slot0 = SlotCurr(0);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + c0;
        DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
    }

    if (len > 1) {
        SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
    }
}

// ---------------------------------------------------------------------------
// RunSeq: compute conv1d with ring buffer (matches vllm)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::RunSeq(int32_t start, int32_t len, int32_t c0,
                                                     int32_t dimTileSize, int32_t dim)
{
    const int32_t width = static_cast<int32_t>(tilingData_.width);
    const int32_t jStart = MAX_WIDTH - width;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> accF = biasF[MAX_BLOCK_DIM];
    LocalTensor<float> tmpF = accF[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasActivation = (tilingData_.activationMode != 0);

    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);

        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotCurr]);

        // Prefetch next input token
        if (t + 1 < len) {
            const int32_t slotNext = SlotPrefetch(t);
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + c0;
            WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
            DataCopy(ring[slotNext * MAX_BLOCK_DIM], xGm[xOffsetNext], dimTileSize);
            SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotNext]);
        }

        // Initialize accumulator with bias
        DataCopy(accF, biasF, dimTileSize);
        PipeBarrier<PIPE_V>();

        // Convolution dot product
        for (int32_t j = jStart; j < MAX_WIDTH; ++j) {
            const int32_t tap = (MAX_WIDTH - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, dimTileSize);
            MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], dimTileSize);
        }

        // Activation (SiLU)
        if (hasActivation) {
            Silu(tmpF, accF, dimTileSize);
        }

        // Cast and write output (double-buffered, matches vllm)
        const int32_t outSlot = t & 1;
        LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
        if (t >= 2) {
            WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }
        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outSlotT, tmpF, dimTileSize);
            } else {
                DataCopy(outSlotT, accF, dimTileSize);
            }
        } else {
            if (hasActivation) {
                Cast(outSlotT, tmpF, RoundMode::CAST_RINT, dimTileSize);
            } else {
                Cast(outSlotT, accF, RoundMode::CAST_RINT, dimTileSize);
            }
        }

        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + c0;

        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        DataCopy(yGm[outOffset], outSlotT, dimTileSize);
        if (t + 2 < len) {
            SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if (t + 2 < len) {
            SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
        }
    }
}

// ---------------------------------------------------------------------------
// WriteBackState: write ring buffer back to conv_state (matches vllm)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                                             int32_t dimTileSize, int32_t dim)
{
    const int32_t stateLen = static_cast<int32_t>(tilingData_.stateLen);
    const int32_t width = static_cast<int32_t>(tilingData_.width);
    if (len <= 0) {
        return;
    }

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();

    for (int32_t pos = 0; pos < (width - 1); ++pos) {
        const int32_t tap = (width - 2) - pos;
        const int32_t slot = (tap == 0) ? SlotCurr(lastT) : SlotHist(lastT, tap);
        const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                    static_cast<int64_t>(pos) * dim + c0;
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], dimTileSize);
    }
}

// ---------------------------------------------------------------------------
// Process: grid distribution over (batch, dimBlock) pairs (matches vllm)
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CausalConv1dUpdate<T>::Process()
{
    const int32_t dim = static_cast<int32_t>(tilingData_.dim);
    const int32_t batch = static_cast<int32_t>(tilingData_.batch);
    const int32_t seqLen = static_cast<int32_t>(tilingData_.seqLen);
    const int32_t dimTileSize = static_cast<int32_t>(tilingData_.dimTileSize);
    const int32_t blocksPerSeq = static_cast<int32_t>(tilingData_.blocksPerSeq);
    const int32_t width = static_cast<int32_t>(tilingData_.width);

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (dimTileSize <= 0 || blocksPerSeq <= 0 || dimTileSize > MAX_BLOCK_DIM ||
        width < 2 || width > MAX_WIDTH) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * blocksPerSeq;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize;
         task += static_cast<int64_t>(blockNum)) {

        const int32_t seq = static_cast<int32_t>(task / blocksPerSeq);
        const int32_t dimBlockId = static_cast<int32_t>(task % blocksPerSeq);
        const int32_t c0 = dimBlockId * dimTileSize;
        if (c0 >= dim) {
            continue;
        }
        const int32_t dimTileSizeActual = (c0 + dimTileSize <= dim) ? dimTileSize : (dim - c0);

        // Determine sequence range
        int32_t start = 0;
        int32_t len = 0;
        if (tilingData_.hasQueryLoc != 0) {
            start = queryStartLocGm.GetValue(seq);
            const int32_t end = queryStartLocGm.GetValue(seq + 1);
            len = end - start;
        } else {
            start = seq * seqLen;
            len = seqLen;
        }

        if (len <= 0) {
            continue;
        }

        // Resolve cache index
        int32_t cacheIdx = seq;
        if (tilingData_.hasIndices != 0) {
            cacheIdx = convStateIndicesGm.GetValue(seq);
            if (cacheIdx == static_cast<int32_t>(tilingData_.padSlotId)) {
                continue;
            }
        }

        // Speculative decoding: determine state token offset (matches vllm)
        int32_t stateTokenOffset = 0;
        if (tilingData_.hasNumAccept != 0 && width == MAX_WIDTH) {
            const int32_t accepted = numAcceptGm.GetValue(seq);
            stateTokenOffset = accepted - 1;
            const int32_t maxOffset = static_cast<int32_t>(tilingData_.stateLen - (width - 1));
            if (stateTokenOffset < 0) {
                stateTokenOffset = 0;
            } else if (stateTokenOffset > maxOffset) {
                stateTokenOffset = maxOffset;
            }
        }

        // Weight caching across iterations (matches vllm)
        const bool weightCacheHit =
            weightCacheValid_ && (cachedC0_ == c0) && (cachedDimTileSize_ == dimTileSizeActual);
        if (!weightCacheHit) {
            LoadWeightAndBias(c0, dimTileSizeActual);
            weightCacheValid_ = true;
            cachedC0_ = c0;
            cachedDimTileSize_ = dimTileSizeActual;
        }

        InitRing(cacheIdx, stateTokenOffset, start, len, c0, dimTileSizeActual, dim);
        RunSeq(start, len, c0, dimTileSizeActual, dim);

        // Fence before state writeback (matches vllm)
        SetFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
        SetFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);

        WriteBackState(cacheIdx, len, c0, dimTileSizeActual, dim);

        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_MTE2>();
        PipeBarrier<PIPE_MTE3>();
    }

    ReleaseEvents();
}

} // namespace CausalConv1dUpdateOp
#endif // CAUSAL_CONV1D_UPDATE_H
