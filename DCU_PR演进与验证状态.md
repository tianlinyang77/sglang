# DCU PR 演进与验证状态

> 维护约定：本文档作为当前 DCU/BW1000 PR 的持续演进记录。后续每轮验证、打开测例、disabled 原因变更、suite 数字变化，都继续更新本文档。

## 1. 当前口径

| 项 | 内容 |
|---|---|
| 主线仓库 | `/public/home/dingxl/sglang` |
| donor 仓库 | `/public/home/dingxl/sglang-tly` |
| 目标硬件 | BW1000 |
| 验证节点 | `10.16.1.66` |
| 容器 | `dxl-sglang` |
| 模型根目录 | `/public/opendas/DL_DATA/llm-models` |
| CI 入口 | `test/registered/` + `register_dcu_ci(...)` |
| DCU 专项目录 | `test/registered/dcu/` |

### 1.1 可见卡变量规则

DCU 外层验证命令只设置一种 visible devices 变量，默认使用：

```bash
-e HIP_VISIBLE_DEVICES=<card_ids>
```

不要在同一条外层命令里同时设置 `HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`。  
说明：SGLang server 子进程内部为了兼容 CUDA 语义设置 `CUDA_VISIBLE_DEVICES` 是正常现象；这里限制的是 CI wrapper/远端命令的外层环境变量。

## 2. 为什么不走 `test/srt/`

`test/srt/` 是历史 runtime/model 测试集合，适合参考迁移，但不适合作为 DCU 主线 CI 沉淀位置：

| 对比项 | `test/srt/` | `test/registered/dcu/` |
|---|---|---|
| CI 注册 | 不强制硬件注册，难按 suite 调度 | 每个文件显式 `register_dcu_ci(...)` |
| 硬件隔离 | 默认 CUDA/NVIDIA 语义较多 | DCU/BW1000 参数集中维护 |
| 模型路径 | 容易走远端 HF/gated 模型 | 默认本地模型路径，可用环境变量覆盖 |
| 分层能力 | smoke/nightly/perf/accuracy 边界弱 | 可按 stage/nightly/accuracy/vlm/perf 分层 |
| 官方合入 | 迁移成本高，容易影响 legacy 逻辑 | 与 AMD/Ascend registered 风格一致 |

因此本 PR 选择 `test/srt/` 作为参考来源，正式 DCU CI 建设放在 `test/registered/dcu/` 和 broad registered 覆盖中。

## 3. 演进阶段

### Phase 1：DCU 第三阶段精度测试

目标是补齐 BW1000 精度体系，目录落到：

```text
test/registered/dcu/accuracy/bw1000/
```

已建设：

| 测试 | 数据集 | 模型 | 当前状态 |
|---|---|---|---|
| `test_gsm8k_eval_dcu.py` | GSM8K | Qwen2.5-7B-Instruct | 10 样本 smoke 跑通，历史 3 次 score `0.700/0.700/0.700` |
| `test_mmlu_eval_dcu.py` | MMLU | Qwen2.5-7B-Instruct | 50 样本 smoke 跑通，历史 3 次 score `0.720/0.720/0.780` |
| `test_mmmu_eval_dcu.py` | MMMU | Qwen2.5-VL-3B-Instruct | 10 样本 smoke 跑通，历史 3 次 score `0.400/0.400/0.400` |

已支持：

- 本地数据路径优先。
- 文本/VLM 样本数环境变量拆分。
- MMLU 本地 parquet 数据读取。
- MMMU 本地 parquet 数据读取。
- `nightly-dcu-accuracy` 与 `nightly-dcu-vlm` 分层。

后续需要做：

- GSM8K 固定 nightly 样本 3 次基线。
- MMLU 5000 样本 3 次基线。
- MMMU 100 样本 3 次基线。
- 按 3 次最小值保守下浮固化 nightly 阈值。

### Phase 2：DCU 专项目录建设

当前 DCU 专项目录共有 13 个测试文件：

```text
test/registered/dcu/
├── accuracy/bw1000/      # GSM8K / MMLU / MMMU
├── embedding/bw1000/     # Qwen embedding smoke
├── interface/            # DCU smoke
├── kernels/              # sgl-kernel supported whitelist
├── moe/bw1000/           # Qwen3 MoE smoke
├── reranker/bw1000/      # Qwen3 reranker smoke
├── srt/bw1000/           # Qwen2.5 dense server smoke
└── vlm_models/bw1000/    # Qwen2.5-VL server smoke
```

专项 smoke 已验证情况：

| 类别 | 文件 | 模型/内容 | 状态 |
|---|---|---|---|
| interface | `test_dcu_smoke.py` | DCU 基础可用性 | 已通过 |
| dense 0.5B | `test_qwen25_0p5b_server_dcu.py` | Qwen2.5-0.5B-Instruct | 已通过 |
| dense 1.5B | `test_qwen25_1p5b_server_dcu.py` | Qwen2.5-1.5B-Instruct | 已通过 |
| dense 7B | `test_qwen25_7b_server_dcu.py` | Qwen2.5-7B-Instruct | 已通过 |
| VLM | `test_qwen25_vl_3b_server_dcu.py` | Qwen2.5-VL-3B-Instruct | 已通过 |
| MoE base | `test_qwen3_30b_moe_server_dcu.py` | Qwen3-30B-A3B，TP2 | 已通过 |
| MoE instruct | `test_qwen3_30b_moe_instruct_dcu.py` | Qwen3-30B-A3B-Instruct，TP2 | 已通过 |
| embedding | `test_gte_qwen2_embedding_dcu.py` | Qwen3-Embedding-0.6B | 已通过 |
| reranker | `test_qwen3_reranker_dcu.py` | Qwen3-Reranker-0.6B | 已通过 |
| kernel | `test_sgl_kernel_supported_dcu.py` | 当前 BW1000 支持白名单 | 已通过 |

### Phase 3：sgl-kernel DCU 验证

已对 `sgl-kernel/tests` 做过一轮 DCU 可运行性扫描，并输出文档：

- `DCU_BW1000_sgl-kernel测试汇总.md`
- `DCU_BW1000_基础镜像到测试全流程.md`

结论：

- 一部分 kernel 单测可以直接在 DCU 上通过，适合纳入 whitelist。
- 一部分失败来自底层算子未接入、注册缺失、Triton/ROCm 后端 crash 或 timeout。
- 当前 registered 中使用 `test/registered/dcu/kernels/test_sgl_kernel_supported_dcu.py` 只跑已知支持集合。
- 后续新增 kernel 支持时，先在原始 `sgl-kernel/tests` 三轮验证，再加入 whitelist。

### Phase 4：合入 `sglang-tly` 的 DCU 注册覆盖

合入原则：

- 当前 `sglang/` 为主线，不整目录覆盖。
- `sglang-tly` 的 `k100/` 旧口径不作为主线目录。
- `stage-a-dcu` / `stage-b-dcu` 迁移到当前标准 suite。
- broad registration 作为覆盖信息迁入，不等同于 BW1000 实测通过。
- 未验证或失败项保持 disabled，并写清原因。

合入内容：

| 模块 | 内容 |
|---|---|
| CI registry | 增强 `register_dcu_ci(...)` 和 suite 分层 |
| run_suite | 支持 DCU suite 收集、list、分区 |
| wrapper | `dcu_ci_exec.sh` 支持 `--container-name`、本地模型环境变量、`PYTHONPATH` |
| container | `dcu_ci_start_container.sh` 支持镜像和容器名参数 |
| workflow | PR / nightly DCU workflow 分层 |
| coverage | `analyze_dcu_csv_coverage.py` 覆盖矩阵脚本 |
| docs | 合入差距分析、建设步骤、专项推进、CSV 覆盖矩阵等文档 |

### Phase 5：PR baseline broad 用例验证

目标是把 broad registered 中可作为 PR 门禁的测试分成三类：

1. 本地模型可映射且三轮稳定：打开。
2. 本地化后仍失败：保持 disabled，写清实测原因。
3. 依赖缺失、gated/远端模型、重型多卡/专项路径：保持 disabled，后续专项处理。

## 4. 当前量化状态

最近一次统计时间：2026-05-26。

### 4.1 注册与 suite 状态

| 项 | 当前值 |
|---|---:|
| DCU registered test file | 203 |
| `stage-b-test-1-gpu-small-dcu` enabled | 73 |
| `stage-b-test-1-gpu-small-dcu` skipped/disabled | 61 |
| CSV rows | 161 |
| CSV unique casefiles | 159 |
| CSV enabled | 72 |
| CSV registered-disabled | 80 |
| CSV missing/obsolete | 8 |
| AMD/NVIDIA intersection | 136 |
| AMD/NVIDIA intersection enabled | 87 |
| AMD/NVIDIA intersection disabled | 49 |
| 仍带 `DCU PR baseline deferred` 的文件 | 17 |
| 已写具体 `DCU Stage-B deferred` 原因的文件 | 18 |
| 带 `DCU BW1000 validated` 记录的文件 | 23 |

### 4.2 最近验证命令

注册检查：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang -e HIP_VISIBLE_DEVICES=2 python3 scripts/ci/dcu/verify_dcu_registration.py'
```

结果：

```text
Collected 203 DCU registered test file(s)
OK: DCU registration looks healthy.
```

Stage-B small list：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test -e HIP_VISIBLE_DEVICES=2 python3 run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --list'
```

结果：

```text
Enabled 73 test(s)
Skipped 61 test(s)
```

覆盖矩阵：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang python3 scripts/ci/dcu/analyze_dcu_csv_coverage.py'
```

结果：

```text
CSV Coverage Matrix
  csv_rows: 161
  unique_casefiles: 159
  enabled: 72
  manual_dcu: 1
  missing_or_obsolete: 8
  registered_disabled: 80

AMD/NVIDIA Intersection Matrix
  intersection: 136
  disabled: 49
  enabled: 87
  csv_overlap: 97
  not_in_csv: 39
```

## 5. 已三轮验证并打开的 broad PR 用例

以下文件已在 BW1000 / `dxl-sglang` 下完成三轮验证，或为本阶段明确三轮 smoke 后打开：

| 文件 | 验证摘要 |
|---|---|
| `test/registered/openai_server/function_call/test_openai_function_calling.py` | 3 轮通过，每轮 `11 passed` |
| `test/registered/rl/test_fp32_lm_head.py` | 3 轮通过，每轮 `4 passed` |
| `test/registered/spec/utils/test_build_eagle_tree.py` | 3 轮通过，每轮 `1 passed` |
| `test/registered/attention/test_triton_attention_kernels.py` | 3 轮通过，每轮 `7 passed + 3 subtests` |
| `test/registered/model_loading/test_external_models.py` | 本地 Qwen2-VL 路径，3 轮通过，每轮 `1 passed` |
| `test/registered/layers/mamba/test_mamba_ssm.py` | 3 轮通过，每轮 `324 passed` |
| `test/registered/layers/mamba/test_mamba_ssm_ssd.py` | 3 轮通过，每轮 `62 passed` |
| `test/registered/quant/test_triton_scaled_mm.py` | 3 轮通过，每轮 `1 passed + 3 subtests` |
| `test/registered/embedding/test_input_embeddings.py` | 3 轮通过，每轮 `4 passed` |
| `test/registered/embedding/test_openai_embedding.py` | 3 轮通过，每轮 `10 passed + 5 subtests` |
| `test/registered/vlm/test_evs.py` | 3 轮通过，每轮 `2 passed` |
| `test/registered/openai_server/features/test_enable_thinking.py` | 本地 Qwen3-0.6B，3 轮通过 |
| `test/registered/openai_server/features/test_reasoning_content.py` | 本地 DeepSeek-R1-Distill-Qwen-7B，3 轮通过 |
| `test/registered/unit/entrypoints/openai/test_serving_embedding.py` | 3 轮 smoke 通过 |
| `test/registered/unit/function_call/test_function_call_parser.py` | 3 轮 smoke 通过 |
| `test/registered/unit/function_call/test_json_schema_constraint.py` | 3 轮 smoke 通过 |
| `test/registered/unit/observability/test_metrics_utils.py` | 3 轮 smoke 通过 |
| `test/registered/unit/parser/test_harmony_parser.py` | 3 轮 smoke 通过 |
| `test/registered/unit/parser/test_jinja_template_utils.py` | 3 轮 smoke 通过 |
| `test/registered/unit/parser/test_reasoning_parser.py` | 3 轮 smoke 通过 |
| `test/registered/unit/server_args/test_server_args.py` | 3 轮 smoke 通过 |

说明：`test/registered/tokenizer/test_multi_tokenizer.py`、`test/registered/unit/model_loader/test_modelopt_loader.py` 也有 `DCU BW1000 validated` 注释，但验证结论是保留 disabled，原因见下一节。

## 6. 已验证但保持 disabled 的重点项

| 文件 | 当前原因 |
|---|---|
| `test/registered/attention/test_triton_sliding_window.py` | 使用 `google/gemma-3-4b-it`，触发 gated HF 401，需要本地模型映射 |
| `test/registered/rotary/test_mrope.py` | 本地 Qwen2/Qwen2.5 VL config 可加载，但缺 `rope_theta`，32 个参数化用例失败 |
| `test/registered/embedding/test_embedding_models.py` | 本地 gte-Qwen2 映射后卡在 HFRunner/SRTRunner logits 对比初始化，未进 DCU |
| `test/registered/models/test_cross_encoder_models.py` | 本地 bge-reranker-base 映射后卡在 HFRunner/SRTRunner cross-encoder 对比初始化 |
| `test/registered/moe/test_fused_moe.py` | `fused_moe_triton` 在 BW1000 abort/segfault，栈在 `fused_experts_impl` / Triton launcher |
| `test/registered/hicache/test_hicache_storage.py` | server 启动 VMFault，scheduler exit `-6` / `EOFError`，未进入 MMLU |
| `test/registered/rl/test_patch_torch.py` | multiprocessing CUDA tensor transfer 在 `HIP_VISIBLE_DEVICES=2,3` 下 3 个 subtest 失败，cleanup 超时 |
| `test/registered/rl/test_return_routed_experts.py` | 上游测试类本身 `unittest.skip` flaky，当前收集结果为 `3 skipped` |
| `test/registered/openai_server/features/test_openai_server_hidden_states.py` | base subtests 通过，但 EAGLE/EAGLE3 依赖 gated/远端 Llama 与 draft 模型 |
| `test/registered/openai_server/function_call/test_tool_choice.py` | Llama/Qwen 有本地候选，但 Mistral/LFM 远端/缺失，文件级门禁不能打开 |
| `test/registered/tokenizer/test_multi_tokenizer.py` | 本地 Qwen2.5-7B run 失败在 TTFT latency gate，约 20s vs 11s 阈值 |
| `test/registered/unit/model_loader/test_modelopt_loader.py` | `modelopt_fp8` 在 ROCm/HIP 不支持，BW1000 6 个测试失败 |
| `test/registered/quant/test_torchao.py` | `torchao` 未安装，量化路径依赖缺失 |
| `test/registered/eval/test_eval_accuracy_large.py` | 依赖 `human_eval`，且与 dedicated `nightly-dcu-accuracy` 重叠 |

## 7. 仍待 PR baseline 验证的 17 个文件

| 文件 | 当前处理建议 |
|---|---|
| `test/registered/backends/test_torch_compile.py` | 本地 Qwen2.5-7B 可跑，但有 throughput 门槛，建议专项单跑 3 轮 |
| `test/registered/core/test_gpt_oss_1gpu.py` | GPT-OSS 模型/量化路径重，先确认本地模型 |
| `test/registered/lora/test_lora_eviction.py` | 需要本地 base + adapter 映射 |
| `test/registered/lora/test_multi_lora_backend.py` | 需要本地 LoRA 矩阵映射 |
| `test/registered/mla/test_mla_deepseek_v3.py` | MLA/DeepSeek 后端数值路径，建议专项 |
| `test/registered/models/test_compressed_tensors_models.py` | FP8/compressed tensors 本地模型未确认 |
| `test/registered/models/test_qwen_models.py` | 可映射到本地 Qwen2.5/Qwen3，但涉及 GSM8K 分数门槛 |
| `test/registered/models/test_transformers_models.py` | transformers fallback + eval，耗时较长 |
| `test/registered/moe/test_torch_compile_moe.py` | MoE + torch compile，模型/后端风险高 |
| `test/registered/quant/test_awq.py` | 本地 AWQ 模型未确认，AWQ backend 风险高 |
| `test/registered/quant/test_eval_fp8_accuracy.py` | FP8 accuracy 本地模型未确认 |
| `test/registered/radix_cache/test_radix_attention.py` | 历史 900s timeout，建议 nightly/manual 复测 |
| `test/registered/rl/test_multi_instance_release_memory_occupation.py` | 多实例/内存释放，涉及多进程和显存行为 |
| `test/registered/rl/test_update_weights_from_disk.py` | 需要确认 base 与 non-Instruct 本地路径成对存在 |
| `test/registered/rl/test_update_weights_from_distributed.py` | distributed update weights，建议多卡专项 |
| `test/registered/scheduler/test_no_overlap_scheduler.py` | scheduler + MMLU，耗时和分数门槛需复测 |
| `test/registered/scheduler/test_retract_decode.py` | retract + MMLU，耗时和稳定性需复测 |

## 8. 后续维护流程

每轮继续验证时，按以下顺序更新本文档：

1. 在 `10.16.1.66 / dxl-sglang` 上执行测试，外层只设置一种 visible devices 变量。
2. 单文件通过 3 轮后，更新对应测试文件中的 `register_dcu_ci(...)`，移除 disabled，并加 `DCU BW1000 validated` 注释。
3. 失败或不适合作为 PR gate 的测试，保持 disabled，并把原因改成实测原因。
4. 更新本文档：
   - 第 4 节数字。
   - 第 5 节已打开列表。
   - 第 6 节已验证但 disabled 列表。
   - 第 7 节剩余待验证列表。
5. 重新运行：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang -e HIP_VISIBLE_DEVICES=<card_ids> python3 scripts/ci/dcu/verify_dcu_registration.py'
```

6. 必要时同步生成覆盖矩阵：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang python3 scripts/ci/dcu/analyze_dcu_csv_coverage.py'
```

## 9. 当前结论

本 PR 已从“DCU 专项 smoke/精度补齐”推进到“官方 broad registered 覆盖迁入 + PR baseline 分批打开”的阶段。

当前最重要的进展是：

- DCU registered 总覆盖达到 203 个文件。
- `stage-b-test-1-gpu-small-dcu` 当前 enabled 73 个、skipped/disabled 61 个。
- CSV 覆盖 enabled 72 个，registered-disabled 80 个。
- AMD/NVIDIA 交集 enabled 87 个，disabled 49 个。
- 仍需处理的 PR baseline deferred 收敛到 17 个。
- 已通过三轮并打开一批 function calling、embedding、Mamba、Triton attention/kernel、external model、unit/parser 等低风险路径。

下一步重点不是继续扩大注册数量，而是围绕剩余 17 个 PR baseline deferred 做本地模型映射和专项复测，把能稳定复现的继续打开，把无法本地化或后端不支持的写清 disabled 原因。
