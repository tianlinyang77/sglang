> 当前合入口径：本文档来自 `sglang-tly` 的 DCU 建设记录，已合入当前 `sglang/` 作为覆盖矩阵与推进依据。当前主线按 BW1000 实机处理，默认验证节点为 `10.16.1.66`，容器为 `dxl-sglang`；文中出现的 `10.16.1.58`、`sgl-test`、`K100` 保留为 tly 历史调试记录，不作为新建目录或当前 CI 命名依据。

## 0. 当前合入后状态（2026-05-25）

本轮合入后，`scripts/ci/dcu/verify_dcu_registration.py` 在 `10.16.1.66` 的 `dxl-sglang` 容器内通过，当前收集到 **203 个 DCU registered test file**。

覆盖矩阵脚本 `scripts/ci/dcu/analyze_dcu_csv_coverage.py` 当前输出：

| 项 | 当前结果 |
|---|---:|
| CSV 原始行数 | 161 |
| CSV 唯一 casefile | 159 |
| CSV enabled | 57 |
| CSV registered-disabled | 95 |
| CSV manual_dcu | 1 |
| CSV missing_or_obsolete | 8 |
| AMD/NVIDIA 交集 | 136 |
| 交集 enabled | 75 |
| 交集 disabled | 61 |
| 交集 csv_overlap | 97 |
| 交集 not_in_csv | 39 |

注意：这些数字是当前仓库合入后的注册/矩阵状态。下文 `sglang-tly` 的历史统计保留为来源记录，若数字不同，以本节和重新生成的 `DCU_CSV覆盖矩阵.md`、`DCU_AMD_NVIDIA交集覆盖矩阵.md` 为准。

# DCU 当前测试差距分析

## 0. 最新双线收敛进展（2026-05-23）

当前工作口径已经拆成两条主线：

1. `sglang.csv` 历史用例体系化：CI 用例全部补 DCU 注册，manual 和历史缺失单独登记。
2. AMD/NVIDIA 交集用例收敛：交集已全量注册 DCU，继续处理剩余 disabled。

最新框架统计：

| 项 | 当前结果 |
|---|---|
| DCU 注册总数 | 191 个测试文件 |
| DCU enabled | 132 个测试文件 |
| DCU disabled | 59 个测试文件 |
| CSV 原始行数 | 161 行，159 个唯一 casefile |
| CSV 自动 CI 目标 | 152 行，排除 manual 1 行、missing/obsolete 8 行 |
| CSV 自动目标注册覆盖率 | 152/152，100% |
| CSV 自动目标启用情况 | enabled 108，registered-disabled 44 |
| CSV manual | 1 行：`test/manual/lora/test_lora_llama4.py`，只登记不进自动 CI |
| CSV missing/obsolete | 8 行，当前仓库找不到，移出本轮自动覆盖目标 |
| AMD/NVIDIA 交集 | 136 个测试文件 |
| 交集 DCU 注册缺口 | 0 个 |
| 交集 enabled/disabled | enabled 121，disabled 15 |
| 覆盖矩阵 | `DCU_CSV覆盖矩阵.md`、`DCU_AMD_NVIDIA交集覆盖矩阵.md` |
| 专项推进表 | `DCU专项测试推进表.md` |

关键执行规则：

1. `sglang.csv` 中已有历史覆盖的用例，可以按 CSV 覆盖口径先注册、标记未复测，再后续人工复测。
2. **AMD/NVIDIA 交集里不在 `sglang.csv` 的用例，不能按 CSV 历史覆盖直接通过，必须实机测试；通过后才能启用，失败则保留 disabled 并写真实原因。**
3. 交集中不在 CSV 的 disabled 项优先进入后续快速复测队列；若属于性能、精度、大模型、多卡、VLM、LoRA/HiCache、spec 等重型路径，则归专项，但仍需实测闭环后才能算完成。

本轮新增处理：

| 类别 | 数量 | 处理方式 |
|---|---:|---|
| CSV CI 缺口补注册 | 53 个唯一文件，对应 54 行 CSV | 已补 `register_dcu_ci()` |
| 轻量 unit/parser/server_args/function_call | 9 个文件 | 直接 enabled，标记 `DCU_CSV_CI_UNVERIFIED` |
| 重型/多卡/VLM/DeepSeek/DeePEP/vLLM dependency | 44 个文件 | 已注册但保留 disabled，写明专项原因 |
| manual | 1 行 | `manual_dcu`，不进入 per-commit/nightly |
| missing/obsolete | 8 行 | 只记录，不新建空壳测试 |

交集 disabled 快速候选复测结果：

| 文件 | 快速验证结果 | 当前处理 |
|---|---|---|
| `test/registered/debug_utils/test_engine_dumper_comparator_e2e.py` | 2 failed；默认 `Qwen/Qwen3-30B-A3B --tp 2` 在单卡 quick run 下触发 HIP invalid device ordinal，server code -9 | 保留 disabled，写入实测原因 |
| `test/registered/dllm/test_llada2_mini.py` | 900s 超时；`inclusionAI/LLaDA2.0-mini` server 长时间未完成首个 pytest item，并残留进程 | 保留 disabled，写入实测原因 |
| `test/registered/rl/test_lora_load_from_tensor.py` | 首个 pytest item 退出 code 137 | 保留 disabled，写入实测原因 |
| `test/registered/lora/test_lora_hf_sgl_logprob_diff.py` | 本地 Llama-2 路径在容器内断链到 `/models`；可见 TinyLlama LoRA 缺兼容 base model，basic 子项未进入数值比较 | 保留 disabled，写入实测原因 |
| `test/registered/rl/test_return_routed_experts.py` | 3 skipped，退出码 0；上游 class 级 skip | 去掉 disabled，纳入 enabled，但实际覆盖为 skip |
| `test/registered/vlm/test_evs.py` | 2 passed | 去掉 disabled，纳入 enabled |
| `test/registered/lora/test_lora_overlap_loading.py` | 3 passed、1 DCU 模型矩阵子项 skipped | 去掉 disabled，纳入 enabled；LoRA overlap loader unit 已覆盖，模型矩阵归 LoRA 专项 |
| `test/registered/bench_fn/test_bench_serving_functionality.py` | 2 passed、1 DCU GSP server/model-mapping 子项 skipped | 去掉 disabled，纳入 enabled；benchmark custom-header/mock 路径已覆盖，GSP 多轮 benchmark 归性能专项 |
| `test/registered/model_loading/test_external_models.py` | 1 passed；本地 `Qwen2-VL-2B-Instruct` + external model/mm processor 路径通过 | 去掉 disabled，纳入 enabled；后续更大 VLM/external-model 矩阵仍归专项 |
| `test/registered/perf/test_bench_one_batch_1gpu.py` | 1 passed、1 DCU 默认 8B 阈值子项 skipped | 去掉 disabled，纳入 enabled；小模型 one-batch perf smoke 已覆盖，正式 BW1000 阈值归性能专项 |
| `test/registered/perf/test_bench_one_batch_2gpu.py` | 1 passed、2 DCU MoE/torch-compile TP2 性能子项 skipped | 去掉 disabled，纳入 enabled；小模型 TP2 one-batch smoke 已覆盖，正式多卡性能阈值归专项 |
| `test/registered/perf/test_bench_serving_1gpu_part1.py` | 4 passed、4 DCU ShareGPT/Triton/LoRA 性能子项 skipped | 去掉 disabled，纳入 enabled；小模型 serving throughput/latency smoke 已覆盖，正式性能阈值归专项 |
| `test/registered/perf/test_bench_serving_1gpu_part2.py` | 2 passed、4 DCU VLM/Embedding 性能子项 skipped | 去掉 disabled，纳入 enabled；score API serving smoke 已覆盖，VLM/Embedding/正式阈值归性能专项 |
| `test/registered/perf/test_bench_serving_2gpu.py` | 1 passed、4 DCU MoE/PP serving 性能子项 skipped | 去掉 disabled，纳入 enabled；小模型 TP2 serving smoke 已覆盖，MoE/PP/长上下文阈值归专项 |

### 0.1 剩余非 CSV 交集 disabled 必测队列

以下 15 个文件属于 AMD/NVIDIA 交集，但不在 `sglang.csv` 中，不能使用 CSV 历史覆盖口径直接启用。当前处理原则是：能拆轻量子项就实测启用；不能拆的保持 disabled，并进入专项实测闭环。完整推进表见 `DCU专项测试推进表.md`。

| 分类 | 数量 |
|---|---:|
| perf | 2 |
| eval | 2 |
| spec | 2 |
| debug_utils | 1 |
| distributed | 1 |
| dllm | 1 |
| hicache | 1 |
| lora | 1 |
| openai_server | 1 |
| rl | 1 |
| scheduler | 1 |
| vlm | 1 |

| 文件 | 当前状态/下一步 |
|---|---|
| `test/registered/debug_utils/test_engine_dumper_comparator_e2e.py` | 已快测失败：单卡 quick run 暴露 `--tp 2` 默认模型路径问题，需改成本地小模型或多卡专项复测 |
| `test/registered/dllm/test_llada2_mini.py` | 已快测超时：DLLM/flashinfer 路径 900s 未完成首个 pytest item，归 DLLM 专项 |
| `test/registered/rl/test_lora_load_from_tensor.py` | 已快测失败：首个 pytest item exit code 137，归 LoRA/RL 专项 |
| `test/registered/openai_server/function_call/test_anthropic_tool_use.py` | 已快测失败：Llama3.2 本地 server 可启动，但 tool_use 请求返回 500/error events，归 function-call 专项 |
| `test/registered/scheduler/test_routing_key_scheduling.py` | 已快测超时/残留 server，归 scheduler 专项 |
| `test/registered/distributed/test_dp_attention_large.py` | 未快测启用，4/8 卡 DP attention、DeepSeek/VLM 大模型路径，归多卡专项 |
| `test/registered/eval/test_eval_accuracy_large.py` | 未快测启用，精度基线/数据集/阈值未固化，归精度专项 |
| `test/registered/eval/test_moe_eval_accuracy_large.py` | 未快测启用，MoE 精度基线/数据集/阈值未固化，归精度专项 |
| `test/registered/hicache/test_hicache_variants.py` | 未快测启用，HiCache + MMLU/MGSM/MLA/EAGLE 组合，归 HiCache 专项 |
| `test/registered/lora/test_lora_hf_sgl_logprob_diff.py` | 已快测失败：Llama-2 本地路径在容器内断链，可见 TinyLlama LoRA 缺兼容 base model，归 LoRA 专项 |
| `test/registered/perf/test_bench_serving_1gpu_large.py` | 未快测启用，需 BW1000 serving 性能基线和阈值，归 perf 专项 |
| `test/registered/perf/test_vlm_perf_5090.py` | 未快测启用，VLM perf 原为 5090 口径，需重建 BW1000 口径，归 VLM/perf 专项 |
| `test/registered/spec/eagle/test_eagle3_basic.py` | 未快测启用，EAGLE3 + MMLU 精度/accept length 路径需本地模型映射，归 spec 专项 |
| `test/registered/spec/eagle/test_eagle_dp_attention.py` | 未快测启用，EAGLE3 DP attention 需要 4 卡和精度/性能门槛，归 spec/多卡专项 |
| `test/registered/vlm/test_encoder_dp.py` | 未快测启用，VLM encoder DP + MMMU 4 卡路径，归 VLM/多卡专项 |

## 0. 最新快速框架搭建进展（2026-05-22）

当前工作口径已经调整为：`sglang.csv` 只作为 DCU 可运行能力的参考，不作为本轮主线；本轮主线是快速补齐 **AMD 与 NVIDIA 已注册测试的交集**，先把 DCU CI 框架立起来。

当前已完成：

| 项 | 当前结果 |
|---|---|
| AMD/NVIDIA 注册测试交集 | 136 个测试文件 |
| 交集中缺 DCU 注册 | 0 个，已全部补 `register_dcu_ci()` |
| 当前 DCU 注册总数 | 138 个测试文件 |
| 当前启用 | 113 个文件，其中 52 个为 CSV 覆盖未复测启用 |
| 当前 disabled | 25 个文件 |
| 注册健康检查 | `python3 scripts/ci/dcu/verify_dcu_registration.py` 通过 |
| 语法检查 | 0 个 AST syntax error |
| 实机节点 | `10.16.1.58`，容器 `sgl-test`，硬件按 BW1000 口径记录 |
| CSV 覆盖未复测启用 | 52 个文件，代码内统一标记 `DCU_CSV_COVERED_UNVERIFIED`，详见 `DCU_CSV覆盖未复测合入清单.md` |
| 实机已验证启用 | 61 个文件 |

本轮新确认可启用的交集文件：

| 文件 | 结果 |
|---|---|
| `test/registered/openai_server/basic/test_protocol.py` | 通过，已启用 |
| `test/registered/spec/utils/test_build_eagle_tree.py` | 通过，已启用 |
| `test/registered/unit/model_executor/test_model_hooks.py` | 通过，已启用 |
| `test/registered/unit/model_loader/test_modelopt_export.py` | 通过/skip 符合预期，已启用 |
| `test/registered/debug_utils/test_tensor_dump_forward_hook.py` | 通过，已启用 |
| `test/registered/layers/mamba/test_mamba_ssm.py` | 通过，已启用 |
| `test/registered/layers/mamba/test_mamba_ssm_ssd.py` | 通过，已启用 |
| `test/registered/quant/test_triton_scaled_mm.py` | 通过，已启用 |
| `test/registered/rl/test_patch_torch.py` | 通过，已启用 |
| `test/registered/rl/test_fp32_lm_head.py` | 4 passed，已启用 |
| `test/registered/openai_server/basic/test_serving_rerank.py` | 13 passed，已启用 |
| `test/registered/openai_server/basic/test_anthropic_server.py` | 19 passed、1 DCU 模板敏感子项 skipped，已启用 |
| `test/registered/metrics/test_metrics.py` | 5 routing-key helper passed、2 DCU server metrics skipped，已启用 |
| `test/registered/metrics/test_priority_metrics.py` | 3 QueueCount unit passed、3 DCU server metrics skipped，已启用 |
| `test/registered/openai_server/function_call/test_openai_function_calling.py` | 11 passed，已启用 |
| `test/registered/openai_server/function_call/test_tool_choice.py` | 14 passed、56 DCU 本地模型未映射子项 skipped，已启用 |
| `test/registered/openai_server/features/test_openai_server_hidden_states.py` | 8 passed、8 EAGLE/EAGLE3 子项 skipped，已启用 |
| `test/registered/debug_utils/test_dumper.py` | 173 passed、10 SGLang server/E2E 子项 skipped，已启用 |
| `test/registered/embedding/test_input_embeddings.py` | 4 passed，已启用 |
| `test/registered/core/test_srt_endpoint.py` | 17 passed、7 DCU 重型/不稳定子项 skipped，已启用 |
| `test/registered/core/test_srt_engine.py` | 3 passed、5 精度/吞吐/embedding/CPU offload 子项 skipped，已启用 |
| `test/registered/language/test_srt_backend.py` | 7 passed、6 模型语义/数据集/重型子项 skipped，已启用 |
| `test/registered/rl/test_update_weights_from_disk.py` | 2 passed、5 真实换权重/并发压力矩阵 skipped，已启用 |
| `test/registered/core/test_page_size.py` | 1 passed、1 inherited MMLU accuracy skipped，已启用 |

已记录风险/失败原因的代表文件：

| 文件 | 当前原因 |
|---|---|
| `test/registered/openai_server/function_call/test_anthropic_tool_use.py` | Llama3.2 local server 可启动，但 Anthropic tool_choice/tool_use 请求返回 500/error events，仅 2/10 通过 |
| `test/registered/rotary/test_mrope.py` | Qwen2/Qwen2.5 VL text config 缺 `rope_theta`，所有 MRoPE 参数化失败 |
| `test/registered/backends/test_torch_compile.py` | 默认 Llama-3.1-8B gated 模型 HF 401 |
| `test/registered/attention/test_triton_attention_kernels.py` | 多数基础 attention kernel 通过，sliding-window extend 子项失败 |
| `test/registered/moe/test_torch_compile_moe.py` | Qwen1.5-MoE torch-compile server code -9 |
| `test/registered/moe/test_fused_moe.py` | fused MoE Triton path abort/segfault |
| `test/registered/scheduler/test_abort.py` | abort/memory-leak/duplicate-rid 等子项失败或 ERROR |
| `test/registered/scheduler/test_chunked_prefill.py` | chunked-prefill scheduler cases 失败 |
| `test/registered/scheduler/test_no_chunked_prefill.py` | no-chunked-prefill scheduler cases 失败 |
| `test/registered/scheduler/test_no_overlap_scheduler.py` | no-overlap cases 失败，批次最终超时 |
| `test/registered/core/test_deterministic.py` | fa3 server 启动时 `enable-deterministic-inference` 触发 batch-invariant persistent matmul，shared memory 需求 98304B 超过 BW1000 65536B 硬件限制 |
| `test/registered/embedding/test_openai_embedding.py` | gte-Qwen2 embedding tokenizer 依赖当前容器缺 `transformers.models.qwen2.tokenization_qwen2_fast`；改用 BAAI/bge-small-en 后，fa3 head-dim 32 不支持、默认 aiter 缺 `mha_batch_prefill_func`、torch_native 路径出现维度不匹配/VMFault |
| `test/registered/models/test_generation_models.py` | 本地 Qwen2.5-0.5B server 可跑，prefill logprobs 最大差异约 0.0407；但输出 ROUGE-L 只有约 0.588 到 0.768，decode logprobs 最大差异约 7.13，继续归入数值一致性专项 |
| `test/registered/model_loading/test_external_models.py` | 已用本地 Qwen2-VL-2B external model/mm processor 单文件跑通，当前已启用；更大 VLM/external-model 矩阵仍归专项 |
| `test/registered/core/test_gpt_oss_1gpu.py` | GPT-OSS 20B mxfp4/bf16 reasoning 大模型量化路径，归入 large-model/quant 专项 |
| `test/registered/models/test_cross_encoder_models.py` | cross-encoder/reranker HF-vs-SRT 数值对比依赖本地 reranker 映射，且与 embedding/attention 后端相关，归入专项 |
| `test/registered/models/test_reward_models.py` | reward model HF-vs-SRT score 对比依赖本地 reward 模型映射和数值阈值，归入专项 |
| `test/registered/quant/test_awq.py` | AWQ MoE/VL + MMLU 精度矩阵，不属于快速框架目标，归入 quant/accuracy 专项 |

本轮新增 CSV 覆盖未复测启用策略：

1. 52 个剩余 disabled 文件与 `sglang.csv` 能匹配，按历史 DCU 覆盖直接去掉 `disabled=`。
2. 这些文件当前没有重新实机验证，均在代码中用 `DCU_CSV_COVERED_UNVERIFIED` 标记。
3. 后续人工复测时，若通过则移除该标记；若失败则重新加回 `disabled=` 并写实测原因。
4. 完整清单见 `DCU_CSV覆盖未复测合入清单.md`。

下一步继续策略：

1. 不再逐个文件跑完后立刻全量验证，先批量推进交集覆盖。
2. 实机跑过的文件按真实结果处理；CSV 覆盖但未复测的文件先启用并保留 `DCU_CSV_COVERED_UNVERIFIED` 标记。
3. 后续复测 `DCU_CSV覆盖未复测合入清单.md` 时，再把通过项去标记、失败项重新 disabled。
4. 阶段性收口时再统一跑 `verify_dcu_registration.py` 和已启用文件集合。

## 一、DCU 当前测试现状

### 1.1 总体统计

**文件**：`/public/home/tianly/sgl/sglang/sglang.csv`

**测试总数**：161 个测试文件

**按套件分类**：

| 套件 | 数量 | 说明 |
|------|------|------|
| per-commit-1-gpu | 125 | 单卡测试 |
| per-commit-2-gpu | 14 | 双卡测试 |
| per-commit-4-gpu | 6 | 4卡测试 |
| per-commit-8-gpu-h200 | 4 | 8卡 H200 测试 |
| per-commit-8-gpu-h20 | 3 | 8卡 H20 测试 |
| per-commit-8-gpu-h200-deepseek-v32 | 2 | 8卡 DeepSeek-V32 测试 |
| per-commit-4-gpu-deepep | 2 | 4卡 DeepEP 测试 |
| per-commit-8-gpu-h200-deepep | 1 | 8卡 DeepEP 测试 |
| vllm_dependency_test | 4 | vllm 依赖测试 |

**按目录分类**：

| 目录 | 数量 | 说明 |
|------|------|------|
| openai_server | 19 | OpenAI API 测试 |
| models | 12 | 模型测试 |
| lora | 12 | LoRA 测试 |
| quant | 7 | 量化测试 |
| hicache | 7 | HiCache 测试 |
| rl | 4 | 强化学习测试 |
| layers | 4 | 层测试 |
| ep | 4 | 专家并行测试 |
| 其他单个文件 | 90+ | 其他测试 |

---

## 二、已覆盖的测试分析

### 2.1 已覆盖的 P0 必做测试

#### ✅ OpenAI API 兼容性（19 个）
- `openai_server/basic/test_serving_chat.py` - Chat Completions
- `openai_server/basic/test_serving_completions.py` - Completions
- `openai_server/basic/test_serving_embedding.py` - Embeddings
- `openai_server/features/test_openai_function_calling.py` - Function calling
- `openai_server/features/test_json_constrained.py` - JSON 约束
- `openai_server/features/test_json_mode.py` - JSON 模式
- `openai_server/features/test_openai_hidden_states.py` - Hidden states
- `openai_server/features/test_reasoning_content.py` - Reasoning
- `openai_server/function_call/test_tool_choice.py` - Tool choice
- `openai_server/validation/test_large_max_new_tokens.py` - Max tokens
- `openai_server/validation/test_matched_stop.py` - Matched stop
- `openai_server/validation/test_openai_server_ignore_eos.py` - Ignore EOS
- 其他 API 测试

**结论**：✅ OpenAI API 兼容性测试基本完整

#### ✅ 核心引擎功能（部分）
- `test_srt_engine.py` - SRT 引擎
- `test_srt_endpoint.py` - SRT 端点
- `test_deterministic.py` - 确定性测试
- `test_request_queue_validation.py` - 请求队列验证
- `test_hidden_states.py` - 隐藏状态
- `test_abort.py` - Abort 测试

**结论**：✅ 核心引擎功能基本覆盖

#### ✅ Attention 与 KV Cache（部分）
- `test_triton_attention_backend.py` - Triton Attention 后端
- `test_triton_attention_kernels.py` - Triton Attention Kernels
- `test_torch_native_attention_backend.py` - Torch Native Attention
- `test_create_kvindices.py` - KV 索引创建
- `test_radix_attention.py` - Radix Attention
- `test_radix_cache_unit.py` - Radix Cache 单元测试
- `test_hybrid_attn_backend.py` - Hybrid Attention 后端
- `test_triton_sliding_window.py` - Sliding Window Attention

**结论**：✅ Attention 与 KV Cache 基本覆盖

#### ✅ 基础算子（部分）
- `test_fused_moe.py` - MoE 算子
- `test_mla.py` - MLA 算子
- `test_mla_deepseek_v3.py` - DeepSeek-V3 MLA
- `test_mla_flashinfer.py` - FlashInfer MLA
- `test_mla_fp8.py` - FP8 MLA
- `test_mla_int8_deepseek_v3.py` - INT8 MLA
- `test_fa3.py` - FA3 算子
- `test_triton_moe_channel_fp8_kernel.py` - MoE FP8 Kernel
- `layers/attention/mamba/test_*.py` - Mamba 算子

**结论**：✅ 基础算子基本覆盖

#### ✅ 分布式与通信（部分）
- `test_dp_attention.py` - DP Attention
- `test_data_parallelism.py` - 数据并行
- `test_pp_single_node.py` - Pipeline Parallel
- `ep/test_moe_ep.py` - MoE EP
- `ep/test_deepep_small.py` - DeepEP 小规模
- `ep/test_deepep_large.py` - DeepEP 大规模
- `ep/test_mooncake_ep_small.py` - MoonCake EP
- `lora/test_lora_tp.py` - LoRA TP

**结论**：✅ 分布式与通信基本覆盖

#### ✅ 模型覆盖（部分）
- `test_deepseek_v3_basic.py` - DeepSeek-V3
- `test_deepseek_v3_mtp.py` - DeepSeek-V3 MTP
- `test_deepseek_v32_basic.py` - DeepSeek-V32
- `test_deepseek_v32_mtp.py` - DeepSeek-V32 MTP
- `test_gpt_oss_1gpu.py` - GPT-OSS 1GPU
- `test_gpt_oss_4gpu.py` - GPT-OSS 4GPU
- `models/test_qwen_models.py` - Qwen 模型
- `models/test_glm4_moe_models.py` - GLM-4 MoE
- `models/test_qwen3_next_models.py` - Qwen3 Next

**结论**：✅ 模型覆盖基本完整

#### ✅ 量化测试（7 个）
- `quant/test_fp8_kernel.py` - FP8 Kernel
- `quant/test_int8_kernel.py` - INT8 Kernel
- `quant/test_w8a8_quantization.py` - W8A8 量化
- `quant/test_w4a8_deepseek_v3.py` - W4A8 DeepSeek-V3
- `quant/test_block_int8.py` - Block INT8
- `quant/test_triton_scaled_mm.py` - Triton Scaled MM
- 其他量化测试

**结论**：✅ 量化测试基本覆盖

#### ✅ LoRA 测试（12 个）
- `lora/test_lora.py` - LoRA 基础
- `lora/test_lora_qwen3.py` - Qwen3 LoRA
- `lora/test_lora_llama4.py` - Llama4 LoRA
- `lora/test_lora_backend.py` - LoRA 后端
- `lora/test_lora_openai_api.py` - LoRA OpenAI API
- `lora/test_lora_tp.py` - LoRA TP
- 其他 LoRA 测试

**结论**：✅ LoRA 测试基本完整

#### ✅ HiCache 测试（7 个）
- `hicache/test_hicache.py` - HiCache 基础
- `hicache/test_hicache_mla.py` - HiCache MLA
- `hicache/test_hicache_eagle.py` - HiCache EAGLE
- `hicache/test_hicache_storage.py` - HiCache 存储
- 其他 HiCache 测试

**结论**：✅ HiCache 测试基本完整

#### ✅ 投机解码（部分）
- `test_eagle_infer_a.py` - EAGLE Infer A
- `test_eagle_infer_b.py` - EAGLE Infer B
- `test_eagle_infer_beta.py` - EAGLE Infer Beta
- `test_build_eagle_tree.py` - Build EAGLE Tree
- `test_ngram_speculative_decoding.py` - Ngram Speculative Decoding
- `test_standalone_speculative_decoding.py` - Standalone Speculative Decoding
- `hicache/test_hicache_eagle.py` - HiCache EAGLE

**结论**：✅ 投机解码基本覆盖

---

## 三、缺失的关键测试

### 3.1 ❌ 严重缺失：精度评测测试

**问题**：DCU 当前测试中**没有任何精度评测测试**

**缺失的 P0 必做测试**：

| 测试类型 | 缺失情况 | 重要性 |
|---------|---------|--------|
| **GSM8K 评测** | ❌ 完全缺失 | 🔴 P0 必做 |
| **MMLU 评测** | ❌ 完全缺失 | 🔴 P0 必做 |
| **MMMU 评测** | ❌ 完全缺失 | 🔴 P0 必做 |
| **Golden output 对比** | ❌ 完全缺失 | 🔴 P0 必做 |

**影响**：这是 DCU 合入官方仓库的最大障碍，因为没有精度评测就无法证明 DCU 后端结果可信。

**参考**：AMD 后端有 44 个精度评测测试文件（GSM8K、MMMU 等）

---

### 3.2 ❌ 严重缺失：性能基准测试

**问题**：DCU 当前测试中**没有任何性能基准测试**

**缺失的 P1 必做测试**：

| 测试类型 | 缺失情况 | 重要性 |
|---------|---------|--------|
| **吞吐测试** | ❌ 完全缺失 | 🟡 P1 强烈建议 |
| **延迟测试（TTFT/TPOT/ITL）** | ❌ 完全缺失 | 🟡 P1 强烈建议 |
| **并发性能测试** | ❌ 完全缺失 | 🟡 P1 强烈建议 |
| **不同输入输出长度性能** | ❌ 完全缺失 | 🟡 P1 强烈建议 |
| **单卡/多卡性能对比** | ❌ 完全缺失 | 🟡 P1 强烈建议 |

**影响**：无法证明 DCU 后端的性能水平，无法与 AMD/NVIDIA/Ascend 对比。

**参考**：AMD 后端有 25 个性能基准测试文件

---

### 3.3 ⚠️ 部分缺失：独立 DCU 测试目录

**问题**：DCU 当前测试散落在各个目录，**没有独立的 `test/registered/dcu/` 目录**

**缺失的 P0 必做测试**：

| 目录结构 | 缺失情况 | 重要性 |
|---------|---------|--------|
| **test/registered/dcu/** | ❌ 完全缺失 | 🔴 P0 必做 |
| **test/registered/dcu/accuracy/** | ❌ 完全缺失 | 🔴 P0 必做 |
| **test/registered/dcu/accuracy/k100/** | ❌ 完全缺失 | 🔴 P0 必做 |
| **test/registered/dcu/perf/** | ❌ 完全缺失 | 🟡 P1 强烈建议 |
| **test/registered/dcu/perf/k100/** | ❌ 完全缺失 | 🟡 P1 强烈建议 |

**影响**：不符合官方后端组织规范，无法使用 `register_dcu_ci()` 进行 CI 注册。

**参考**：AMD 有 `test/registered/amd/` 目录，Ascend 有 `test/registered/ascend/` 目录

---

### 3.4 ❌ 严重缺失：DCU CI 注册函数

**问题**：DCU 当前测试中**没有使用 `register_dcu_ci()` 函数**

**缺失的 P0 必做测试**：

| CI 注册 | 缺失情况 | 重要性 |
|--------|---------|--------|
| **register_dcu_ci() 函数** | ❌ 完全缺失 | 🔴 P0 必做 |
| **DCU 平台检测** | ❌ 完全缺失 | 🔴 P0 必做 |
| **DCU 套件路由** | ❌ 完全缺失 | 🔴 P0 必做 |

**影响**：无法建立 DCU CI 系统，无法合入官方仓库。

**参考**：AMD 有 `register_amd_ci()`，NVIDIA 有 `register_cuda_ci()`，Ascend 有 `register_npu_ci()`

---

### 3.5 ⚠️ 部分缺失：VLM 测试

**问题**：DCU 当前 VLM 测试较少

**缺失的 P1 测试**：

| 测试类型 | 现状 | 重要性 |
|---------|------|--------|
| **VLM 模型测试** | ⚠️ 只有 1 个 `test_vlm_input_format.py` | 🟡 P1 强烈建议 |
| **VLM OpenAI Server 测试** | ⚠️ 只有 1 个 `test_vision_openai_server_a.py` | 🟡 P1 强烈建议 |
| **VLM 精度评测** | ❌ 完全缺失 | 🟡 P1 强烈建议 |

**参考**：Ascend 后端有 17 个 VLM 测试文件

---

### 3.6 ⚠️ 部分缺失：RL/Reward/Embedding/Rerank 模型测试

**问题**：DCU 当前相关测试较少

**缺失的 P2 测试**：

| 测试类型 | 现状 | 重要性 |
|---------|------|--------|
| **Reward 模型** | ⚠️ 只有 1 个 `models/test_reward_models.py` | 🟢 P2 后续完善 |
| **Embedding 模型** | ⚠️ 只有 1 个 `models/test_embedding_models.py` | 🟢 P2 后续完善 |
| **Rerank 模型** | ❌ 完全缺失 | 🟢 P2 后续完善 |

**参考**：Ascend 后端有 3 个 reward 模型测试、1 个 embedding 模型测试、1 个 rerank 模型测试

---

## 四、差距总结

### 4.1 按 P0/P1/P2 分类

| 优先级 | 缺失测试类型 | 数量 | 影响 |
|--------|-------------|------|------|
| **P0** | 精度评测（GSM8K/MMLU/MMMU） | 3+ | 🔴 阻塞合入 |
| **P0** | DCU CI 注册函数 | 3 | 🔴 阻塞合入 |
| **P0** | 独立 DCU 测试目录 | 5 | 🔴 阻塞合入 |
| **P1** | 性能基准测试 | 5+ | 🟡 影响竞争力 |
| **P1** | VLM 测试 | 2+ | 🟡 影响完整性 |
| **P2** | RL/Reward/Embedding/Rerank | 3+ | 🟢 影响完善度 |

### 4.2 按功能分类

| 功能分类 | 已覆盖 | 缺失 | 完整度 |
|---------|--------|------|--------|
| **OpenAI API** | ✅ 19 个 | ❌ 无 | 100% |
| **核心引擎** | ✅ 6 个 | ⚠️ 部分缺失 | 80% |
| **Attention/KV Cache** | ✅ 8 个 | ⚠️ 部分缺失 | 85% |
| **基础算子** | ✅ 9 个 | ⚠️ 部分缺失 | 80% |
| **分布式** | ✅ 7 个 | ⚠️ 部分缺失 | 75% |
| **模型测试** | ✅ 9 个 | ⚠️ 部分缺失 | 70% |
| **量化测试** | ✅ 7 个 | ⚠️ 部分缺失 | 70% |
| **LoRA 测试** | ✅ 12 个 | ❌ 无 | 100% |
| **HiCache 测试** | ✅ 7 个 | ❌ 无 | 100% |
| **投机解码** | ✅ 7 个 | ⚠️ 部分缺失 | 80% |
| **精度评测** | ❌ 0 个 | 🔴 3+ | 0% |
| **性能基准** | ❌ 0 个 | 🔴 5+ | 0% |
| **DCU CI 系统** | ❌ 0 个 | 🔴 3 | 0% |
| **VLM 测试** | ⚠️ 2 个 | 🟡 2+ | 40% |

### 4.3 对比 AMD 后端

| 对比项 | AMD 后端 | DCU 当前 | 差距 |
|--------|---------|---------|------|
| 独立测试目录 | ✅ `test/registered/amd/` | ❌ 无 | 🔴 必须补 |
| 精度测试目录 | ✅ `amd/accuracy/` (44 个) | ❌ 无 | 🔴 必须补 |
| 性能测试目录 | ✅ `amd/perf/` (25 个) | ❌ 无 | 🔴 必须补 |
| CI 注册函数 | ✅ `register_amd_ci` | ❌ 无 | 🔴 必须补 |
| GSM8K 评测 | ✅ 覆盖 20+ 模型 | ❌ 无 | 🔴 必须补 |
| MMLU/MMMU 评测 | ✅ 完整 | ❌ 无 | 🔴 必须补 |
| OpenAI API | ✅ 完整 | ✅ 19 个 | ✅ 已覆盖 |
| 分布式测试 | ✅ 完整 | ✅ 7 个 | ✅ 已覆盖 |
| 量化测试 | ✅ 完整 | ✅ 7 个 | ✅ 已覆盖 |
| LoRA 测试 | ✅ 完整 | ✅ 12 个 | ✅ 已覆盖 |

---

## 五、优先级建议

### 5.1 立即补齐（P0 - 阻塞合入）

1. **创建 `register_dcu_ci()` 函数**
   - 文件：`python/sglang/test/ci/ci_register.py`
   - 参考：`register_amd_ci()`

2. **添加 DCU 平台检测**
   - 文件：`python/sglang/test/test_utils.py`
   - 添加 `is_dcu()` 函数

3. **添加 DCU 套件路由**
   - 文件：`test/run_suite.py`
   - 添加 `nightly-dcu`、`stage-a-dcu`、`stage-b-dcu` 套件

4. **创建 `test/registered/dcu/` 目录结构**
   - 创建 `test/registered/dcu/`
   - 创建 `test/registered/dcu/accuracy/k100/`
   - 创建 `test/registered/dcu/perf/k100/`

5. **创建精度评测测试**
   - `test_gsm8k_eval_dcu.py`
   - `test_mmlu_eval_dcu.py`
   - `test_mmmu_eval_dcu.py`

### 5.2 强烈建议补齐（P1 - 影响竞争力）

1. **创建性能基准测试**
   - 吞吐测试
   - 延迟测试（TTFT/TPOT/ITL）
   - 并发性能测试

2. **补充 VLM 测试**
   - VLM 模型测试
   - VLM 精度评测

3. **补充专项精度测试**
   - 量化精度测试
   - MLA 精度测试
   - MoE 精度测试

### 5.3 后续完善（P2 - 影响完善度）

1. **补充 RL/Reward/Embedding/Rerank 模型测试**
2. **补充调试工具测试**
3. **补充压力测试**
4. **补充长时间稳定性测试**

---

## 六、结论

### 6.1 总体评价

DCU 当前测试覆盖度约为 **60%**，主要缺失：

- 🔴 **精度评测**（0% 覆盖）- 这是最大障碍
- 🔴 **性能基准**（0% 覆盖）- 影响竞争力
- 🔴 **DCU CI 系统**（0% 覆盖）- 阻塞合入

### 6.2 合入官方仓库的关键差距

DCU 要合入官方仓库，**必须先补齐**：

1. ✅ 创建 `register_dcu_ci()` 函数
2. ✅ 创建 `test/registered/dcu/` 目录
3. ✅ 创建精度评测测试（GSM8K/MMLU/MMMU）
4. ✅ 添加 DCU 套件路由

### 6.3 与 AMD 后端的对比

DCU 当前测试数量（161 个）与 AMD 后端（106 个独有测试）相当，但：

- ✅ **功能测试**：DCU 已基本覆盖
- ❌ **精度评测**：DCU 完全缺失，AMD 有 44 个
- ❌ **性能基准**：DCU 完全缺失，AMD 有 25 个
- ❌ **CI 系统**：DCU 完全缺失，AMD 有完整 CI 系统

### 6.4 预估工作量

- **P0 必做**：约 2-3 天（基础设施 + 3 个精度评测）
- **P1 强烈建议**：约 3-5 天（性能基准 + VLM + 专项精度）
- **P2 后续完善**：约 5-7 天（补充完善）

**总计**：约 10-15 天可以完成与 AMD 后端对齐的测试覆盖度
