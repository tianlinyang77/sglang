> 当前合入口径：本文档来自 `sglang-tly` 的 DCU 建设记录，已合入当前 `sglang/` 作为覆盖矩阵与推进依据。当前主线按 BW1000 实机处理，默认验证节点为 `10.16.1.66`，容器为 `dxl-sglang`；文中出现的 `10.16.1.58`、`sgl-test`、`K100` 保留为 tly 历史调试记录，不作为新建目录或当前 CI 命名依据。

# DCU 专项测试推进表

## 1. 定位

本文用于收敛 AMD/NVIDIA 交集中不在 `sglang.csv` 的剩余 disabled 测试。

这些用例不能使用 CSV 历史覆盖口径直接启用，必须走 DCU 实机验证。验证通过后去掉 `disabled=`；验证失败或需要专项资源时保留 `disabled=`，并写真实原因。

当前统计：

| 项 | 数量 |
|---|---:|
| AMD/NVIDIA 交集 | 136 |
| 交集 enabled | 121 |
| 交集 disabled | 15 |
| 非 CSV 交集 disabled | 15 |

## 2. 推进规则

1. 快速框架阶段优先保留轻量/unit/mock/小模型子项。
2. 能跳子项就不跳整文件。
3. 性能、精度、多卡、大模型、VLM、LoRA/HiCache、speculative decoding 默认进入专项。
4. 专项不是永久跳过；每个专项必须补齐模型、数据、硬件、阈值和复测命令。
5. 通过项才允许启用；失败项必须保留真实错误，不写泛化原因。

## 3. 专项总览

| 专项 | 数量 | 当前目标 |
|---|---:|---|
| 性能基线 | 2 | 建 BW1000 perf baseline，固化吞吐/延迟阈值 |
| 精度评测 | 2 | 建 GSM8K/MMLU/MoE eval 的模型、数据集、阈值闭环 |
| VLM/多卡 | 2 | 补 VLM/DP attention 多卡模型映射和 4/8 卡复测 |
| LoRA/HiCache | 2 | 补可用 base model、adapter、storage 后复测 |
| Spec/EAGLE | 2 | 补 target/draft 本地模型和 accept length/精度门槛 |
| 已知失败专项 | 5 | 针对 debug/DLLM/function-call/RL/scheduler 失败原因修复或复测 |

## 4. 性能基线专项

目标：先把 BW1000 的 perf smoke 和正式性能基线拆开。当前 `test_bench_one_batch_1gpu.py` 已启用小模型 one-batch smoke，`test_bench_one_batch_2gpu.py` 已启用小模型 TP2 one-batch smoke，`test_bench_serving_1gpu_part1.py` 已启用小模型 serving throughput/latency smoke，`test_bench_serving_1gpu_part2.py` 已启用 score API serving smoke，`test_bench_serving_2gpu.py` 已启用小模型 TP2 serving smoke；剩余项需要单独跑 3 次基线后再固化门槛。

| 文件 | 当前状态 | 前置条件 | 下一步 |
|---|---|---|---|
| `test/registered/perf/test_bench_serving_1gpu_large.py` | registered-disabled | 1 卡，大模型 serving 基线 | 用固定模型跑 3 次，拆 smoke 与 threshold |
| `test/registered/perf/test_vlm_perf_5090.py` | registered-disabled | BW1000 VLM perf 口径 | 不沿用 5090 阈值，重建 BW1000 阈值 |

## 5. 精度评测专项

目标：和第三阶段精度建设对齐，不直接套 CUDA/AMD 阈值。

| 文件 | 当前状态 | 前置条件 | 下一步 |
|---|---|---|---|
| `test/registered/eval/test_eval_accuracy_large.py` | registered-disabled | 本地文本模型、GSM8K/MMLU 数据集、阈值 | 先 10/50 样本 smoke，再跑 3 次基线 |
| `test/registered/eval/test_moe_eval_accuracy_large.py` | registered-disabled | 本地 MoE 模型、数据集、阈值 | 先确认 MoE server 稳定，再接 eval |

## 6. VLM/多卡专项

目标：把 VLM、多卡 DP attention 和 MMMU 路径从快速框架里拆出来，使用独立 nightly 或后续专项 suite。

| 文件 | 当前状态 | 前置条件 | 下一步 |
|---|---|---|---|
| `test/registered/distributed/test_dp_attention_large.py` | registered-disabled | 4/8 卡资源，DeepSeek/VLM 本地模型 | 按模型拆子类复测，避免单卡 quick run |
| `test/registered/vlm/test_encoder_dp.py` | registered-disabled | VLM 模型、4 卡资源、lmms-eval 数据 | 先用 1 个本地 VLM 跑小样本，再扩 MMMU |

## 7. LoRA/HiCache 专项

目标：建立可复测的本地 base model、LoRA adapter、HiCache storage 组合。

| 文件 | 当前状态 | 前置条件 | 下一步 |
|---|---|---|---|
| `test/registered/lora/test_lora_hf_sgl_logprob_diff.py` | 已快测失败 | 当前 Llama-2 路径在容器内断链；TinyLlama LoRA 缺兼容 base model | 补可用 base+adapter 后复测 basic，再决定 full/chunked |
| `test/registered/hicache/test_hicache_variants.py` | registered-disabled | 本地模型、HiCache storage、MLA/EAGLE 模型 | 先拆 standard/page-size 子项，MLA/EAGLE 后置 |

## 8. Spec/EAGLE 专项

目标：补齐 target/draft 模型路径和 BW1000 accept length/精度阈值。

| 文件 | 当前状态 | 前置条件 | 下一步 |
|---|---|---|---|
| `test/registered/spec/eagle/test_eagle3_basic.py` | registered-disabled | Llama3.1 target、EAGLE3 draft 本地路径 | 映射本地模型后先小样本 MMLU |
| `test/registered/spec/eagle/test_eagle_dp_attention.py` | registered-disabled | 4 卡，Qwen3 target/draft，DP attention | 多卡窗口复测 |

## 9. 已知失败专项

目标：这些不是缺登记，而是已经有实测失败或超时，需要按失败原因修。

| 文件 | 当前失败原因 | 下一步 |
|---|---|---|
| `test/registered/debug_utils/test_engine_dumper_comparator_e2e.py` | 默认 `Qwen/Qwen3-30B-A3B --tp 2` 在单卡 quick run 下触发 HIP invalid device ordinal，server exit code -9 | 改本地小模型 E2E 或放入多卡 debug 专项 |
| `test/registered/dllm/test_llada2_mini.py` | 900s 超时，DLLM/flashinfer 路径未完成首个 pytest item | 查 server 卡住阶段，确认是否 flashinfer/模型路径问题 |
| `test/registered/openai_server/function_call/test_anthropic_tool_use.py` | Llama3.2 本地 server 可启动，但 tool_use 请求返回 500/error events，2/10 通过 | 定位 Anthropic tool_use parser/server error |
| `test/registered/rl/test_lora_load_from_tensor.py` | 首个 LoRA load-from-tensor pytest item exit code 137 | 查 OOM/进程被杀原因，必要时缩小 tensor/model |
| `test/registered/scheduler/test_routing_key_scheduling.py` | quick-validation 窗口未完成，且可能残留 routing-key server | 拆小请求和 server 生命周期，单独 scheduler 专项复测 |

## 10. 阶段验收

每个专项完成时至少满足：

1. 对应文件有 DCU 实机运行记录。
2. 通过项去掉 `disabled=`。
3. 失败项保留真实错误原因。
4. 需要阈值的专项有 BW1000 阈值来源。
5. `python3 scripts/ci/dcu/verify_dcu_registration.py` 通过。
6. `DCU_AMD_NVIDIA交集覆盖矩阵.md` 中对应项状态同步更新。
