> 当前合入口径：本文档来自 `sglang-tly` 的 DCU 建设记录，已合入当前 `sglang/` 作为覆盖矩阵与推进依据。当前主线按 BW1000 实机处理，默认验证节点为 `10.16.1.66`，容器为 `dxl-sglang`；文中出现的 `10.16.1.58`、`sgl-test`、`K100` 保留为 tly 历史调试记录，不作为新建目录或当前 CI 命名依据。

# DCU CSV 覆盖未复测合入清单

更新时间：2026-05-22

本清单记录本轮按 `sglang.csv` 历史覆盖直接启用、但当前没有重新实机验证的 DCU 注册测试。

代码标记：每个对应文件的 `register_dcu_ci()` 前均有：

```python
# DCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical DCU coverage; not re-tested in this framework pass.
```

当前标记数量：52 个。

| 序号 | 测试文件 | CSV casefile | 当前状态 |
|---:|---|---|---|
| 1 | `test/registered/attention/test_torch_native_attention_backend.py` | `test_torch_native_attention_backend.py` | 已启用，CSV 覆盖，当前未复测 |
| 2 | `test/registered/attention/test_triton_attention_backend.py` | `test_triton_attention_backend.py` | 已启用，CSV 覆盖，当前未复测 |
| 3 | `test/registered/attention/test_triton_attention_kernels.py` | `test_triton_attention_kernels.py` | 已启用，CSV 覆盖，当前未复测 |
| 4 | `test/registered/attention/test_triton_sliding_window.py` | `test_triton_sliding_window.py` | 已启用，CSV 覆盖，当前未复测 |
| 5 | `test/registered/backends/test_torch_compile.py` | `test_torch_compile.py` | 已启用，CSV 覆盖，当前未复测 |
| 6 | `test/registered/core/test_deterministic.py` | `test_deterministic.py` | 已启用，CSV 覆盖，当前未复测 |
| 7 | `test/registered/core/test_gpt_oss_1gpu.py` | `test_gpt_oss_1gpu.py` | 已启用，CSV 覆盖，当前未复测 |
| 8 | `test/registered/core/test_hidden_states.py` | `test_hidden_states.py` | 已启用，CSV 覆盖，当前未复测 |
| 9 | `test/registered/distributed/test_data_parallelism.py` | `test_data_parallelism.py` | 已启用，CSV 覆盖，当前未复测 |
| 10 | `test/registered/distributed/test_load_weights_from_remote_instance.py` | `test_load_weights_from_remote_instance.py` | 已启用，CSV 覆盖，当前未复测 |
| 11 | `test/registered/distributed/test_pp_single_node.py` | `test_pp_single_node.py` | 已启用，CSV 覆盖，当前未复测 |
| 12 | `test/registered/embedding/test_embedding_models.py` | `models/test_embedding_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 13 | `test/registered/embedding/test_openai_embedding.py` | `openai_server/basic/test_openai_embedding.py` | 已启用，CSV 覆盖，当前未复测 |
| 14 | `test/registered/hicache/test_hicache_storage.py` | `hicache/test_hicache_storage.py` | 已启用，CSV 覆盖，当前未复测 |
| 15 | `test/registered/hicache/test_hicache_storage_3fs_backend.py` | `hicache/test_hicache_storage_3fs_backend.py` | 已启用，CSV 覆盖，当前未复测 |
| 16 | `test/registered/hicache/test_hicache_storage_file_backend.py` | `hicache/test_hicache_storage_file_backend.py` | 已启用，CSV 覆盖，当前未复测 |
| 17 | `test/registered/lora/test_lora_backend.py` | `lora/test_lora_backend.py` | 已启用，CSV 覆盖，当前未复测 |
| 18 | `test/registered/lora/test_lora_eviction.py` | `lora/test_lora_eviction.py` | 已启用，CSV 覆盖，当前未复测 |
| 19 | `test/registered/lora/test_lora_qwen3.py` | `lora/test_lora_qwen3.py` | 已启用，CSV 覆盖，当前未复测 |
| 20 | `test/registered/lora/test_lora_tp.py` | `lora/test_lora_tp.py` | 已启用，CSV 覆盖，当前未复测 |
| 21 | `test/registered/lora/test_multi_lora_backend.py` | `lora/test_multi_lora_backend.py` | 已启用，CSV 覆盖，当前未复测 |
| 22 | `test/registered/mla/test_mla.py` | `test_mla.py` | 已启用，CSV 覆盖，当前未复测 |
| 23 | `test/registered/mla/test_mla_deepseek_v3.py` | `test_mla_deepseek_v3.py` | 已启用，CSV 覆盖，当前未复测 |
| 24 | `test/registered/mla/test_mla_fp8.py` | `test_mla_fp8.py` | 已启用，CSV 覆盖，当前未复测 |
| 25 | `test/registered/models/test_compressed_tensors_models.py` | `models/test_compressed_tensors_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 26 | `test/registered/models/test_cross_encoder_models.py` | `models/test_cross_encoder_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 27 | `test/registered/models/test_generation_models.py` | `models/test_generation_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 28 | `test/registered/models/test_qwen_models.py` | `models/test_qwen_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 29 | `test/registered/models/test_reward_models.py` | `models/test_reward_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 30 | `test/registered/models/test_transformers_models.py` | `models/test_transformers_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 31 | `test/registered/models/test_vlm_models.py` | `models/test_vlm_models.py` | 已启用，CSV 覆盖，当前未复测 |
| 32 | `test/registered/moe/test_fused_moe.py` | `test_fused_moe.py` | 已启用，CSV 覆盖，当前未复测 |
| 33 | `test/registered/moe/test_torch_compile_moe.py` | `test_torch_compile_moe.py` | 已启用，CSV 覆盖，当前未复测 |
| 34 | `test/registered/openai_server/basic/test_openai_server.py` | `openai_server/basic/test_openai_server.py` | 已启用，CSV 覆盖，当前未复测 |
| 35 | `test/registered/openai_server/features/test_enable_thinking.py` | `openai_server/features/test_enable_thinking.py` | 已启用，CSV 覆盖，当前未复测 |
| 36 | `test/registered/openai_server/features/test_reasoning_content.py` | `openai_server/features/test_reasoning_content.py` | 已启用，CSV 覆盖，当前未复测 |
| 37 | `test/registered/quant/test_awq.py` | `quant/test_awq.py` | 已启用，CSV 覆盖，当前未复测 |
| 38 | `test/registered/quant/test_block_int8.py` | `quant/test_block_int8.py` | 已启用，CSV 覆盖，当前未复测 |
| 39 | `test/registered/quant/test_eval_fp8_accuracy.py` | `test_eval_fp8_accuracy.py` | 已启用，CSV 覆盖，当前未复测 |
| 40 | `test/registered/quant/test_torchao.py` | `test_torchao.py` | 已启用，CSV 覆盖，当前未复测 |
| 41 | `test/registered/rl/test_multi_instance_release_memory_occupation.py` | `test_multi_instance_release_memory_occupation.py` | 已启用，CSV 覆盖，当前未复测 |
| 42 | `test/registered/rl/test_update_weights_from_distributed.py` | `rl/test_update_weights_from_distributed.py` | 已启用，CSV 覆盖，当前未复测 |
| 43 | `test/registered/rl/test_update_weights_from_tensor.py` | `rl/test_update_weights_from_tensor.py` | 已启用，CSV 覆盖，当前未复测 |
| 44 | `test/registered/rotary/test_mrope.py` | `rotary_embedding/test_mrope.py` | 已启用，CSV 覆盖，当前未复测 |
| 45 | `test/registered/scheduler/test_abort.py` | `test_abort.py` | 已启用，CSV 覆盖，当前未复测 |
| 46 | `test/registered/scheduler/test_chunked_prefill.py` | `test_chunked_prefill.py` | 已启用，CSV 覆盖，当前未复测 |
| 47 | `test/registered/scheduler/test_no_chunked_prefill.py` | `test_no_chunked_prefill.py` | 已启用，CSV 覆盖，当前未复测 |
| 48 | `test/registered/scheduler/test_no_overlap_scheduler.py` | `test_no_overlap_scheduler.py` | 已启用，CSV 覆盖，当前未复测 |
| 49 | `test/registered/scheduler/test_retract_decode.py` | `test_retract_decode.py` | 已启用，CSV 覆盖，当前未复测 |
| 50 | `test/registered/tokenizer/test_multi_tokenizer.py` | `test_multi_tokenizer.py` | 已启用，CSV 覆盖，当前未复测 |
| 51 | `test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py` | `batch_invariant/test_batch_invariant_ops.py` | 已启用，CSV 覆盖，当前未复测 |
| 52 | `test/registered/vlm/test_vision_chunked_prefill.py` | `test_vision_chunked_prefill.py` | 已启用，CSV 覆盖，当前未复测 |
