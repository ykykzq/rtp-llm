#include "rtp_llm/cpp/cache/CacheStoreWriter.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <vector>

namespace rtp_llm {

void runtimeWriteCacheStore(const CacheStoreInputs&     cache_store_inputs,
                            const KvCacheInfo&          kv_cache,
                            bool                        mla_kvcache,
                            std::shared_ptr<CacheStore> cache_store) {
    auto& param = cache_store_inputs;
    if (param.warmup) {
        RTP_LLM_LOG_DEBUG("is warmup, so ignore writeCacheStore");
        return;
    }
    if (!param.pd_separation || param.context_batch_size == 0) {
        RTP_LLM_LOG_DEBUG("pd_separation = %d, context_batch_size = %d, so ignore writeCacheStore",
                          param.pd_separation,
                          param.context_batch_size);
        return;
    }
    if (!cache_store) {
        RTP_LLM_LOG_DEBUG("cache_store is null, skip writeCacheStore");
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(param.host_kv_cache_offset.defined(), "failed to get host_kv_cache_offset");
    const int32_t* offset_addr          = nullptr;
    size_t         max_blocks_per_batch = 0;

    const bool   has_group_types = param.kv_cache_group_types_host.defined();
    const size_t group_num       = has_group_types ? static_cast<size_t>(param.kv_cache_group_types_host.size(0)) : 1;

    int gid = 0;
    if (param.kv_cache_layer_to_group_host.defined() && param.layer_id >= 0
        && static_cast<size_t>(param.layer_id) < static_cast<size_t>(param.kv_cache_layer_to_group_host.numel())) {
        gid = param.kv_cache_layer_to_group_host.data_ptr<int32_t>()[param.layer_id];
    }

    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int32_t>(group_num), "invalid kv cache group id [%d]", gid);

    bool is_grouped_layout = has_group_types && group_num > 1;
    if (param.kv_cache_layer_to_group_host.defined() && group_num > 1) {
        std::vector<bool> used_groups(group_num, false);
        size_t            used_group_num = 0;
        const auto*       layer_to_group = param.kv_cache_layer_to_group_host.data_ptr<int32_t>();
        for (int64_t layer = 0; layer < param.kv_cache_layer_to_group_host.numel(); ++layer) {
            const int layer_gid = layer_to_group[layer];
            RTP_LLM_CHECK_WITH_INFO(layer_gid >= 0 && static_cast<size_t>(layer_gid) < group_num,
                                    "invalid kv cache group id [%d] at layer [%ld]",
                                    layer_gid,
                                    static_cast<long>(layer));
            if (!used_groups[static_cast<size_t>(layer_gid)]) {
                used_groups[static_cast<size_t>(layer_gid)] = true;
                ++used_group_num;
            }
        }
        is_grouped_layout = used_group_num > 1;
    }

    if (param.host_kv_cache_offset.dim() == 3) {
        const auto group_offset_view = param.host_kv_cache_offset[static_cast<int64_t>(gid)];
        max_blocks_per_batch         = group_offset_view.size(1);
        offset_addr                  = group_offset_view.data_ptr<int32_t>();
    } else {
        max_blocks_per_batch = param.host_kv_cache_offset.size(1);
        offset_addr          = param.host_kv_cache_offset.data_ptr<int32_t>();
    }

    const auto seq_size_per_block = param.tokens_per_block;
    auto       kv_cache_data      = (uint64_t*)kv_cache.kv_cache_buffer.data_ptr();
    auto kv_scale_data = kv_cache.kv_scale_buffer.defined() ? (uint64_t*)kv_cache.kv_scale_buffer.data_ptr() : nullptr;

    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_pd_separation.numel()),
                            "size not same");
    RTP_LLM_CHECK_WITH_INFO(param.context_batch_size == static_cast<size_t>(param.request_id.numel()),
                            "context batch size and request id size is not same");

    RTP_LLM_LOG_DEBUG("write cache store, context_batch_size is %ld", param.context_batch_size);

    for (size_t batch_id = 0; batch_id < param.context_batch_size; batch_id++) {
        if (*(param.request_pd_separation.data_ptr<bool>() + batch_id) == false) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.defined() && param.input_lengths_host.defined(),
                                "failed to get prefix_length_host and input_length_host for cache store");
        RTP_LLM_CHECK_WITH_INFO(param.prefix_lengths_host.data_ptr<int>()[batch_id] % seq_size_per_block == 0,
                                "prefix_length %% seq_size_per_block != 0");
        int reuse_block_num = param.prefix_lengths_host.data_ptr<int>()[batch_id] / seq_size_per_block;
        int block_num =
            (param.input_lengths_host.data_ptr<int>()[param.decoder_batch_size + batch_id] + seq_size_per_block - 1)
            / seq_size_per_block;
        auto request_id     = *(param.request_id.data_ptr<int64_t>() + batch_id);
        auto event          = param.pre_created_event ? param.pre_created_event : runtimeCreateEvent();
        auto request_blocks = std::make_shared<RequestBlockBuffer>(std::to_string(request_id), event);
        RTP_LLM_LOG_DEBUG(
            "write cache store, request id is %ld, blocks num is %ld", request_id, block_num + reuse_block_num);

        CacheGroupType group_type = CacheGroupType::FULL;
        group_type = static_cast<CacheGroupType>(param.kv_cache_group_types_host.data_ptr<int32_t>()[gid]);

        const int total_blocks = block_num + reuse_block_num;
        if (total_blocks <= 0) {
            continue;
        }

        auto addBlock = [&](int index, CacheGroupType group_type) {
            RTP_LLM_CHECK_WITH_INFO(index >= 0 && index < static_cast<int>(max_blocks_per_batch),
                                    "invalid block index=%d (max_blocks_per_batch=%zu)",
                                    index,
                                    max_blocks_per_batch);
            auto block_id = *(offset_addr + (param.decoder_batch_size + batch_id) * max_blocks_per_batch + index);
            std::string cache_key;
            cache_key =
                makeCacheKey(param.model_id, param.cache_keys[batch_id * max_blocks_per_batch + index], param.layer_id);

            void*                 kv_addr = (void*)((int8_t*)kv_cache_data + block_id * param.kv_block_stride_bytes);
            std::shared_ptr<void> kv_block_addr(kv_addr, [](void* p) {});

            if (is_grouped_layout || mla_kvcache) {
                request_blocks->addBlock("kv_" + cache_key, kv_block_addr, param.kv_block_stride_bytes, true, true);
            } else {
                const uint32_t        kv_half = static_cast<uint32_t>(param.kv_block_stride_bytes / 2);
                void*                 k_addr  = kv_addr;
                void*                 v_addr  = (void*)((int8_t*)kv_addr + kv_half);
                std::shared_ptr<void> k_block_addr(k_addr, [](void* p) {});
                std::shared_ptr<void> v_block_addr(v_addr, [](void* p) {});
                request_blocks->addBlock("k_" + cache_key, k_block_addr, kv_half, true, true);
                request_blocks->addBlock("v_" + cache_key, v_block_addr, kv_half, true, true);
            }

            if (kv_scale_data) {
                void* kv_scale_addr = (void*)((int8_t*)kv_scale_data + block_id * param.kv_scale_stride_bytes);
                std::shared_ptr<void> kv_scale_block_addr(kv_scale_addr, [](void* p) {});
                if (is_grouped_layout || mla_kvcache) {
                    request_blocks->addBlock(
                        "kv_scale_" + cache_key, kv_scale_block_addr, param.kv_scale_stride_bytes, true, true);
                } else {
                    const uint32_t        sc_half = static_cast<uint32_t>(param.kv_scale_stride_bytes / 2);
                    void*                 k_sc    = kv_scale_addr;
                    void*                 v_sc    = (void*)((int8_t*)kv_scale_addr + sc_half);
                    std::shared_ptr<void> k_scale_block_addr(k_sc, [](void* p) {});
                    std::shared_ptr<void> v_scale_block_addr(v_sc, [](void* p) {});
                    request_blocks->addBlock("k_scale_" + cache_key, k_scale_block_addr, sc_half, true, true);
                    request_blocks->addBlock("v_scale_" + cache_key, v_scale_block_addr, sc_half, true, true);
                }
            }
        };

        if (group_type == CacheGroupType::LINEAR) {
            addBlock(total_blocks - 1, group_type);
        } else {
            for (int index = 0; index < total_blocks; ++index) {
                addBlock(index, group_type);
            }
        }

        auto storeCallback = [layer_id = param.layer_id, model_id = param.model_id, gid, request_id, request_blocks](
                                 bool success, CacheStoreErrorCode ec) {
            if (!success) {
                RTP_LLM_LOG_WARNING("PD_CACHE_KEY_WRITE_FAILED request_id=%ld model_id=%zu local_layer_id=%d gid=%d "
                                    "error_code=%d error=%s buffer={%s}",
                                    static_cast<long>(request_id),
                                    model_id,
                                    layer_id,
                                    gid,
                                    static_cast<int>(ec),
                                    ErrorCodeToString(transCacheStoreErrorCode(ec)).c_str(),
                                    request_blocks->debugInfo().c_str());
            }
        };
        cache_store->store(request_blocks, storeCallback);
    }
}

void execWriteCacheStore(const CacheStoreInputs&     inputs,
                         const KvCacheInfo&          kv_cache,
                         bool                        mla_kvcache,
                         std::shared_ptr<CacheStore> cache_store) {
    runtimeWriteCacheStore(inputs, kv_cache, mla_kvcache, std::move(cache_store));
}

}  // namespace rtp_llm
