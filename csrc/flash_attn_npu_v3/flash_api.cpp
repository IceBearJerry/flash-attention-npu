#include <torch/extension.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

#include "mha_fwd_kvcache.cpp"
// #include "mha_fwd_kvcache_2.cpp"
#include "tilingdata.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_common.hpp"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

uint32_t GetQSBlockTile(int64_t kvSeqlen)
{
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}

uint32_t GetKSBlockTile(uint32_t kvSeqlen)
{
    uint32_t kSBlockTile = MAX_KV_STACK_LEN;
    return kSBlockTile;
}

struct BatchParams {
    uint32_t qSeqlen;
    uint32_t kvSeqlen;
    uint32_t curQNBlockTile;
    uint32_t qNBlockNumPerGroup;
    uint32_t curQNBlockNum;
    uint32_t curQSBlockTile;
    uint32_t curQSBlockNum;
    uint32_t curKSBlockTile;
    uint32_t curKSBlockNum;
};

struct SplitContext {
    int32_t batch_size;
    int32_t num_heads;
    int32_t num_heads_k;
    int32_t seqlen_q;
    int32_t head_size_v;
    int32_t* cu_seqlen_q_cpu;
    int32_t* seqlens_k_cpu;
    bool is_varlen_q;
    uint32_t blockDim;
};

BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize, const SplitContext& ctx) {
    BatchParams p;
    if (ctx.is_varlen_q) {
        p.qSeqlen = static_cast<uint32_t>(ctx.cu_seqlen_q_cpu[bIdx + 1] - ctx.cu_seqlen_q_cpu[bIdx]);
    } else {
        p.qSeqlen = static_cast<uint32_t>(ctx.seqlen_q);
    }
    p.kvSeqlen = static_cast<uint32_t>(ctx.seqlens_k_cpu[bIdx]);
    p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
    p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
    p.curQNBlockNum = p.qNBlockNumPerGroup * ctx.num_heads_k;
    p.curQSBlockTile = GetQSBlockTile(p.kvSeqlen);
    p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
    p.curKSBlockTile = GetKSBlockTile(p.kvSeqlen);
    p.curKSBlockNum = (p.kvSeqlen + p.curKSBlockTile - 1) / p.curKSBlockTile;
    return p;
}

void fillCoreInfoForFlashDecode(FAInferTilingData* tiling, uint32_t groupSize,
                                uint64_t perCoreTaskNum, const SplitContext& ctx) {
    int32_t nowBIdx = 0;
    int32_t nowN1Idx = 0;
    int32_t nowS1Idx = 0;
    int32_t nowS2Idx = 0;

    for (uint32_t coreIdx = 0; coreIdx < ctx.blockDim; coreIdx++) {
        tiling->coreInfo[coreIdx].startBIdx = 0;
        tiling->coreInfo[coreIdx].startN1Idx = 0;
        tiling->coreInfo[coreIdx].startS1Idx = 0;
        tiling->coreInfo[coreIdx].startS2Idx = 0;
        tiling->coreInfo[coreIdx].endBIdx = 0;
        tiling->coreInfo[coreIdx].endN1Idx = 0;
        tiling->coreInfo[coreIdx].endS1Idx = 0;
        tiling->coreInfo[coreIdx].endS2Idx = 0;
    }

    auto finishBatch = [&](uint32_t coreIdx) {
        BatchParams p = getBatchParams(ctx.batch_size - 1, groupSize, ctx);
        tiling->coreInfo[coreIdx].endBIdx = ctx.batch_size - 1;
        tiling->coreInfo[coreIdx].endN1Idx = p.curQNBlockNum - 1;
        tiling->coreInfo[coreIdx].endS1Idx = p.curQSBlockNum - 1;
        tiling->coreInfo[coreIdx].endS2Idx = p.curKSBlockNum;
        tiling->set_needCoreNum(coreIdx + 1);
    };

    for (uint32_t coreIdx = 0; coreIdx < ctx.blockDim; coreIdx++) {
        int32_t resTaskNum = static_cast<int32_t>(perCoreTaskNum);
        tiling->coreInfo[coreIdx].startBIdx = nowBIdx;
        tiling->coreInfo[coreIdx].startN1Idx = nowN1Idx;
        tiling->coreInfo[coreIdx].startS1Idx = nowS1Idx;
        tiling->coreInfo[coreIdx].startS2Idx = nowS2Idx;

        BatchParams p = getBatchParams(nowBIdx, groupSize, ctx);

        auto advanceCounters = [&]() {
            if (nowS2Idx == static_cast<int32_t>(p.curKSBlockNum)) { nowS1Idx++; nowS2Idx = 0; }
            if (nowS1Idx == static_cast<int32_t>(p.curQSBlockNum)) { nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0; }
            if (nowN1Idx == static_cast<int32_t>(p.curQNBlockNum)) { nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0; }
        };

        while (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize, ctx);
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                ? p.curQSBlockTile
                : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) - 1)
                ? p.curKSBlockTile
                : (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
            uint64_t singleS2Task = static_cast<uint64_t>(remainingQ) * remainingKV;
            resTaskNum -= static_cast<int32_t>(singleS2Task);
            nowS2Idx += 1;
        }

        if (resTaskNum <= 0) {
            tiling->coreInfo[coreIdx].endBIdx = nowBIdx;
            tiling->coreInfo[coreIdx].endN1Idx = nowN1Idx;
            tiling->coreInfo[coreIdx].endS1Idx = nowS1Idx;
            tiling->coreInfo[coreIdx].endS2Idx = nowS2Idx;
        }

        advanceCounters();
        if (nowBIdx < ctx.batch_size && resTaskNum <= 0) continue;
        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }

        while (nowBIdx < ctx.batch_size && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize, ctx);
            uint32_t remainingQ = p.qSeqlen * (ctx.num_heads - p.curQNBlockTile * nowN1Idx) - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingKV = p.kvSeqlen;
            uint32_t remainingInBatch = remainingQ * remainingKV;

            if (resTaskNum >= static_cast<int32_t>(remainingInBatch)) {
                resTaskNum -= remainingInBatch;
                nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0;
            } else {
                break;
            }
        }

        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowN1Idx < static_cast<int32_t>(p.curQNBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = p.qSeqlen * p.curQNBlockTile - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingInN1 = remainingQ * p.kvSeqlen;
            if (resTaskNum >= static_cast<int32_t>(remainingInN1)) {
                resTaskNum -= remainingInN1;
                nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                ? p.curQSBlockTile
                : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint64_t remainingInS1 = static_cast<uint64_t>(remainingQ) * p.kvSeqlen;
            if (resTaskNum >= static_cast<int64_t>(remainingInS1)) {
                resTaskNum -= static_cast<int32_t>(remainingInS1);
                nowS1Idx++; nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize, ctx);

        while (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < static_cast<int32_t>(p.curQSBlockNum) - 1)
                ? p.curQSBlockTile
                : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < static_cast<int32_t>(p.curKSBlockNum) - 1)
                ? p.curKSBlockTile
                : (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
            uint64_t singleS2Task = static_cast<uint64_t>(remainingQ) * remainingKV;
            resTaskNum -= static_cast<int32_t>(singleS2Task);
            nowS2Idx += 1;
        }

        if (nowBIdx == ctx.batch_size) { finishBatch(coreIdx); break; }

        tiling->coreInfo[coreIdx].endBIdx = nowBIdx;
        tiling->coreInfo[coreIdx].endN1Idx = nowN1Idx;
        tiling->coreInfo[coreIdx].endS1Idx = nowS1Idx;
        tiling->coreInfo[coreIdx].endS2Idx = nowS2Idx;

        advanceCounters();
    }
}

void fillSplitInfoForFlashDecode(FAInferTilingData* tiling, uint32_t groupSize,
                                 const SplitContext& ctx) {
    constexpr uint32_t SIZE_OF_32BIT = 4;

    for (uint32_t splitIdx = 0; splitIdx < ctx.blockDim + 1; splitIdx++) {
        tiling->splitInfo[splitIdx].batchIdx = 0;
        tiling->splitInfo[splitIdx].headStartIdx = 0;
        tiling->splitInfo[splitIdx].headEndIdx = 0;
        tiling->splitInfo[splitIdx].qStartIdx = 0;
        tiling->splitInfo[splitIdx].qEndIdx = 0;
        tiling->splitInfo[splitIdx].splitNum = 0;
        tiling->splitInfo[splitIdx].lseTaskOffset = 0;
        tiling->splitInfo[splitIdx].oTaskOffset = 0;
    }

    int64_t currentLseTaskOffset = 0;
    int64_t currentOTaskOffset = 0;
    int32_t splitIdx = -1;
    int32_t prevBIdx = -1;
    int32_t prevN1Idx = -1;
    int32_t prevS1Idx = -1;

    for (uint32_t coreIdx = 0; coreIdx < ctx.blockDim; coreIdx++) {
        int32_t startBIdx = tiling->coreInfo[coreIdx].startBIdx;
        int32_t startN1Idx = tiling->coreInfo[coreIdx].startN1Idx;
        int32_t startS1Idx = tiling->coreInfo[coreIdx].startS1Idx;
        int32_t startS2Idx = tiling->coreInfo[coreIdx].startS2Idx;
        int32_t endBIdx = tiling->coreInfo[coreIdx].endBIdx;
        int32_t endN1Idx = tiling->coreInfo[coreIdx].endN1Idx;
        int32_t endS1Idx = tiling->coreInfo[coreIdx].endS1Idx;
        int32_t endS2Idx = tiling->coreInfo[coreIdx].endS2Idx;

        tiling->coreInfo[coreIdx].firstSplitKVTaskLseOffset = 0;
        tiling->coreInfo[coreIdx].firstSplitKVTaskOOffset = 0;

        bool foundFirstSplitKV = false;
        for (int BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
            BatchParams p = getBatchParams(BIdx, groupSize, ctx);

            int curStartN1 = (BIdx == startBIdx) ? startN1Idx : 0;
            int curEndN1 = (BIdx == endBIdx) ? endN1Idx : static_cast<int>(p.curQNBlockNum) - 1;

            for (int N1Idx = curStartN1; N1Idx <= curEndN1; N1Idx++) {
                int curStartS1 = (BIdx == startBIdx && N1Idx == startN1Idx) ? startS1Idx : 0;
                int curEndS1 = (BIdx == endBIdx && N1Idx == endN1Idx) ? endS1Idx : static_cast<int>(p.curQSBlockNum) - 1;

                for (int S1Idx = curStartS1; S1Idx <= curEndS1; S1Idx++) {
                    int curStartS2 = (BIdx == startBIdx && N1Idx == startN1Idx && S1Idx == startS1Idx) ? startS2Idx : 0;
                    int curEndS2 = (BIdx == endBIdx && N1Idx == endN1Idx && S1Idx == endS1Idx) ? endS2Idx : static_cast<int>(p.curKSBlockNum);

                    int coveredS2 = curEndS2 - curStartS2;
                    bool isSplitKV = (coveredS2 > 0 && coveredS2 < static_cast<int>(p.curKSBlockNum));

                    int64_t tmpLseOffset = currentLseTaskOffset;
                    int64_t tmpOOffset = currentOTaskOffset;

                    uint32_t N1IdxPerGroup = N1Idx % p.qNBlockNumPerGroup;
                    uint32_t kvHeadIdx = N1Idx / p.qNBlockNumPerGroup;
                    uint32_t currentHeadStart = kvHeadIdx * groupSize + N1IdxPerGroup * p.curQNBlockTile;
                    uint32_t currentHeadEnd = std::min(currentHeadStart + p.curQNBlockTile, (kvHeadIdx + 1) * groupSize);

                    uint32_t currentQStart = S1Idx * p.curQSBlockTile;
                    uint32_t currentQEnd = std::min(currentQStart + p.curQSBlockTile, p.qSeqlen);

                    uint32_t headLen = currentHeadEnd - currentHeadStart;
                    uint32_t qLen = currentQEnd - currentQStart;

                    if (isSplitKV) {
                        if (BIdx != prevBIdx || N1Idx != prevN1Idx || S1Idx != prevS1Idx) {
                            splitIdx++;
                            if (splitIdx < static_cast<int32_t>(ctx.blockDim) + 1) {
                                tiling->splitInfo[splitIdx].batchIdx = BIdx;
                                tiling->splitInfo[splitIdx].splitNum = 0;
                                tiling->splitInfo[splitIdx].headStartIdx = currentHeadStart;
                                tiling->splitInfo[splitIdx].headEndIdx = currentHeadEnd;
                                tiling->splitInfo[splitIdx].qStartIdx = currentQStart;
                                tiling->splitInfo[splitIdx].qEndIdx = currentQEnd;
                                tiling->splitInfo[splitIdx].lseTaskOffset = currentLseTaskOffset;
                                tiling->splitInfo[splitIdx].oTaskOffset = currentOTaskOffset;
                            }
                            prevBIdx = BIdx;
                            prevN1Idx = N1Idx;
                            prevS1Idx = S1Idx;
                        }
                        if (splitIdx >= 0 && splitIdx < static_cast<int32_t>(ctx.blockDim) + 1) {
                            tiling->splitInfo[splitIdx].splitNum++;
                            currentLseTaskOffset += static_cast<int64_t>(headLen) * qLen;
                            currentOTaskOffset += static_cast<int64_t>(headLen) * qLen * ctx.head_size_v;
                        }

                        if (!foundFirstSplitKV) {
                            foundFirstSplitKV = true;
                            tiling->coreInfo[coreIdx].firstSplitKVTaskLseOffset = tmpLseOffset;
                            tiling->coreInfo[coreIdx].firstSplitKVTaskOOffset = tmpOOffset;
                        }
                    }
                }
            }
        }
    }

    uint32_t actualSplitNum = (splitIdx + 1 > static_cast<int32_t>(ctx.blockDim))
        ? ctx.blockDim : static_cast<uint32_t>(splitIdx + 1);
    tiling->set_totalSplitNodeNum(actualSplitNum);
    tiling->set_splitLseTotalSize(currentLseTaskOffset * SIZE_OF_32BIT);
    tiling->set_splitOTotalSize(currentOTaskOffset * SIZE_OF_32BIT);
}

void splitBN2S1GS2(FAInferTilingData* tiling, const SplitContext& ctx) {
    uint64_t totalTaskNum = 0;
    uint32_t groupSize = ctx.num_heads / ctx.num_heads_k;

    for (int32_t batchIdx = 0; batchIdx < ctx.batch_size; batchIdx++) {
        BatchParams p = getBatchParams(batchIdx, groupSize, ctx);
        totalTaskNum += static_cast<uint64_t>(ctx.num_heads) * p.qSeqlen * p.kvSeqlen;
    }
    uint64_t perCoreTaskNum = (totalTaskNum + ctx.blockDim - 1) / ctx.blockDim;
    fillCoreInfoForFlashDecode(tiling, groupSize, perCoreTaskNum, ctx);
    fillSplitInfoForFlashDecode(tiling, groupSize, ctx);
}

std::vector<at::Tensor>
mha_fwd(at::Tensor q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        at::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        at::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        std::optional<at::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<at::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        std::optional<at::Tensor> q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> cu_seqlens_q_,  // b+1
        std::optional<at::Tensor> cu_seqlens_k_,  // b+1
        std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
        std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int64_t> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int64_t> max_seqlen_k_,
        std::optional<at::Tensor> page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<at::Tensor> kv_batch_idx_, // b. indices to index into the KV cache
        std::optional<at::Tensor> leftpad_k_, // b
        std::optional<at::Tensor> rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        std::optional<at::Tensor> rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        std::optional<at::Tensor> seqlens_rotary_, // b
        std::optional<at::Tensor> q_descale_,  // (b, h_k), not (b, h)
        std::optional<at::Tensor> k_descale_,  // (b, h_k)
        std::optional<at::Tensor> v_descale_,  // (b, h_k)
        std::optional<float> softmax_scale_,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        float softcap,
        bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        std::optional<at::Tensor> scheduler_metadata_,  // (b + 1)
        int64_t num_splits,
        std::optional<bool> pack_gqa_,
        int64_t sm_margin
        )
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    auto hostStart = std::chrono::steady_clock::now();

    auto q_dtype = q.dtype();
    bool is_bf16 = q_dtype == torch::kBFloat16;
    bool is_fp16 = q_dtype == torch::kFloat16;
    TORCH_CHECK(is_bf16 || is_fp16, "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor v must have contiguous last dimension");
    at::Tensor tiling_cpu_tensor = at::empty({static_cast<int64_t>(sizeof(FAInferTilingData))}, at::device(c10::kCPU).dtype(at::kByte));

    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    std::memset(tiling_cpu_ptr, 0, sizeof(FAInferTilingData));
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t launchBlockDim = blockDim;
    at::Tensor seqlens_k, block_table, out;
    at::Tensor k_, v_, rotary_cos, rotary_sin, cache_batch_idx, alibi_slopes;

    at::Tensor cu_seqlens_q, cu_seqlens_k;
    at::Tensor out_accum, softmax_lse_accum;
    float softmax_scale;

    const bool paged_KV = page_table_.has_value();
    const bool is_varlen_q = cu_seqlens_q_.has_value();
    const bool is_varlen_kv = cu_seqlens_k_.has_value();

    if (paged_KV) {
        auto page_table = page_table_.value();
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype int32");
        TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }

    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then paged table is not supported");
    }

    if (seqused_k_.has_value()) {
        seqlens_k = seqused_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    }

    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!rotary_sin_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!seqlens_rotary_.has_value(), "NPU FlashAttention does not support seqlens_rotary");
    TORCH_CHECK(!q_descale_.has_value(), "NPU FlashAttention does not support q_descale");
    TORCH_CHECK(!k_descale_.has_value(), "NPU FlashAttention does not support k_descale");
    TORCH_CHECK(!v_descale_.has_value(), "NPU FlashAttention does not support v_descale");
    TORCH_CHECK(softcap == 0.0f, "NPU FlashAttention does not support softcap");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right");
    TORCH_CHECK(attention_chunk == 0, "NPU FlashAttention does not support attention_chunk");
    TORCH_CHECK(!scheduler_metadata_.has_value(), "NPU FlashAttention does not support scheduler_metadata");
    TORCH_CHECK(num_splits == 1 || num_splits == 0, "NPU FlashAttention only supports num_splits=1 or num_splits=0");
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(), "NPU FlashAttention does not support pack_gqa");

    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then paged table is not supported");
    }
    if (k_new_.has_value()) {
        k_ = k_new_.value();
    }
    if (v_new_.has_value()) {
        v_ = v_new_.value();
    }
    if (rotary_cos_.has_value()) {
        rotary_cos = rotary_cos_.value();
    }
    if (rotary_sin_.has_value()) {
        rotary_sin = rotary_sin_.value();
    }
    if (kv_batch_idx_.has_value()) {
        cache_batch_idx = kv_batch_idx_.value();
    }
    if (paged_KV) {
        block_table = page_table_.value();
    }
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "output must have the same dtype as inputs");
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    }  else {
        out = torch::empty_like(q);
    }
    const auto sizes = q.sizes();
    
    int batch_size = 0;
    int seqlen_q = 0;
    int num_heads = 0;
    int head_size_og = 0;
    if (is_varlen_q) {
        batch_size = cu_seqlens_q.size(0) - 1;
        seqlen_q = static_cast<int>(max_seqlen_q_.value());
        num_heads = sizes[1];
        head_size_og = sizes[2];
    } else {
        batch_size = sizes[0];
        seqlen_q = sizes[1];
        num_heads = sizes[2];
        head_size_og = sizes[3];
    }
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 128 : k.size(1);
    const int num_heads_k = k.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    at::Tensor seqlenk_cpu_tensor = seqlens_k.to(at::Device(at::kCPU));
    int32_t* seqlens_k_cpu = static_cast<int32_t *>(seqlenk_cpu_tensor.data_ptr());
    int32_t* cu_seqlen_q_cpu = nullptr;
    at::Tensor cu_seqlen_q_cpu_tensor;
    if (is_varlen_q) {
        cu_seqlen_q_cpu_tensor = cu_seqlens_q.to(at::Device(at::kCPU));
        cu_seqlen_q_cpu = static_cast<int32_t *>(cu_seqlen_q_cpu_tensor.data_ptr());
    }
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq));
    tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(is_causal));
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
    int32_t max_kv_seqlen = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        max_kv_seqlen = std::max(max_kv_seqlen, seqlens_k_cpu[i]);
    }
    tiling_cpu_ptr->set_maxKvSeqlen(static_cast<uint32_t>(max_kv_seqlen));

    uint32_t totalTaskNum = 0;
    uint32_t groupSize = num_heads / num_heads_k;
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = seqlen_q;
        if (is_varlen_q) {
            qSeqlen = *(cu_seqlen_q_cpu + batchIdx + 1) - *(cu_seqlen_q_cpu + batchIdx);
        }
        uint64_t kvSeqlen = *(seqlens_k_cpu + batchIdx);
        uint64_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;
        uint64_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
        }
        totalTaskNum += curTaskNum;
    }
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);

    int64_t maxQSeqlenCalc = 0;
    int64_t minQSeqlenCalc = std::numeric_limits<int64_t>::max();
    int64_t maxKVSeqlenCalc = 0;
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        int64_t qSeqlenVal = seqlen_q;
        int64_t kvSeqlenVal = *(seqlens_k_cpu + batchIdx);
        if (is_varlen_q) {
            qSeqlenVal = *(cu_seqlen_q_cpu + batchIdx + 1) - *(cu_seqlen_q_cpu + batchIdx);
            if (is_varlen_kv) {
                kvSeqlenVal = *(seqlens_k_cpu + batchIdx + 1) - *(seqlens_k_cpu + batchIdx);
            }
        }
        maxQSeqlenCalc = std::max(maxQSeqlenCalc, qSeqlenVal);
        minQSeqlenCalc = std::min(minQSeqlenCalc, qSeqlenVal);
        maxKVSeqlenCalc = std::max(maxKVSeqlenCalc, kvSeqlenVal);
    }
    uint32_t numTasks = static_cast<uint32_t>(batch_size * num_heads_k);
    bool isLongSeq = (static_cast<double>(numTasks) <= 0.8 * blockDim) &&
        (maxKVSeqlenCalc >= static_cast<int64_t>(blockDim) * 512);
    bool isShortSeq = (static_cast<double>(numTasks) <= 0.4 * blockDim) &&
        (maxKVSeqlenCalc >= 1024);
    bool flashDecodeFlag = paged_KV && is_varlen_q &&
        (maxQSeqlenCalc * groupSize <= 128) && (maxQSeqlenCalc <= 16) &&
        (maxKVSeqlenCalc >= 1024) && (minQSeqlenCalc > 0) && (isLongSeq || isShortSeq);

    SplitContext splitCtx;
    splitCtx.batch_size = batch_size;
    splitCtx.num_heads = num_heads;
    splitCtx.num_heads_k = num_heads_k;
    splitCtx.seqlen_q = seqlen_q;
    splitCtx.head_size_v = head_size_og;
    splitCtx.cu_seqlen_q_cpu = cu_seqlen_q_cpu;
    splitCtx.seqlens_k_cpu = seqlens_k_cpu;
    splitCtx.is_varlen_q = is_varlen_q;
    splitCtx.blockDim = blockDim;
    if (flashDecodeFlag) {
        splitBN2S1GS2(tiling_cpu_ptr, splitCtx);
        auto needCoreNum = tiling_cpu_ptr->get_needCoreNum();
        if (needCoreNum != 0) {
            launchBlockDim = needCoreNum;
        }
    }

    uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 512;
    uint64_t PRELANCH_NUM = 3;
    uint64_t mm1OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        2 * PRELANCH_NUM;
    uint64_t mm2OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;
    uint64_t UpdateSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;
    uint64_t splitLseTotalSize = tiling_cpu_ptr->get_splitLseTotalSize();
    uint64_t splitOTotalSize = tiling_cpu_ptr->get_splitOTotalSize();
    int64_t workSpaceSize = static_cast<int64_t>(mm1OutSize + smOnlineOutSize + mm2OutSize
        + UpdateSize + splitLseTotalSize + splitOTotalSize);

    at::Tensor workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
    at::Tensor softmaxlse = at::empty({batch_size, seqlen_q, num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    if (is_varlen_q) {
        softmaxlse = at::empty({sizes[0], num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    }
    softmaxlse.fill_(std::numeric_limits<float>::infinity());
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);
    at::Tensor mask_gpu_tensor;
    if (is_causal) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
    }
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    at::Tensor seqlenk_gpu_tensor;
    at::Tensor seqlenq_gpu_tensor;
    if (is_varlen_q) {
        seqlenq_gpu_tensor = cu_seqlens_q;
    } else {
        seqlenq_gpu_tensor = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kInt));
    }
    if (is_varlen_kv) {
        seqlenk_gpu_tensor = cu_seqlens_k;
    } else {
        seqlenk_gpu_tensor = seqlens_k;
    }
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(q.data_ptr());
    auto kDevice = static_cast<uint8_t *>(k.data_ptr());
    auto vDevice = static_cast<uint8_t *>(v.data_ptr());
    uint8_t * blockTableDevice = nullptr;
    uint8_t * maskDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(block_table.data_ptr());
    }
    if (is_causal) {
        maskDevice = static_cast<uint8_t *>(mask_gpu_tensor.data_ptr());
    }
    auto oDevice = static_cast<uint8_t *>(out.data_ptr());
    auto qSeqDevice = static_cast<uint8_t *>(seqlenq_gpu_tensor.data_ptr());
    auto kvSeqDevice = static_cast<uint8_t *>(seqlenk_gpu_tensor.data_ptr());
    auto workspaceDevice = static_cast<uint8_t *>(workspace_tensor.data_ptr());
    auto tilingDevice = static_cast<uint8_t *>(tiling_gpu_tensor.data_ptr());
    auto softmaxLseDevice = static_cast<uint8_t *>(softmaxlse.data_ptr());
    if (is_bf16) {
        if (paged_KV) {
            if (is_causal) {
                if (is_varlen_q) {
                    if (flashDecodeFlag) {
                        SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    } else {
                        SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    }
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    if (flashDecodeFlag) {
                        SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    } else {
                        SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    }
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        } else {
            if (is_causal) {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        }
    } else {
        if (paged_KV) {
            if (is_causal) {
                if (is_varlen_q) { 
                    if (flashDecodeFlag) {
                        SplitFuse::FAInfer<half, half, float, true, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    } else {
                        SplitFuse::FAInfer<half, half, float, true, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    }
                } else {
                    SplitFuse::FAInfer<half, half, float, true, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    if (flashDecodeFlag) {
                        SplitFuse::FAInfer<half, half, float, true, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    } else {
                        SplitFuse::FAInfer<half, half, float, true, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                    }
                } else {
                    SplitFuse::FAInfer<half, half, float, true, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        } else {
            if (is_causal) {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<half, half, float, false, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<half, half, float, false, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    SplitFuse::FAInfer<half, half, float, false, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    SplitFuse::FAInfer<half, half, float, false, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<launchBlockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        }
    }
    return {out, softmaxlse, out_accum, softmax_lse_accum};
}

PYBIND11_MODULE(flash_attn_npu_3, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache");
}