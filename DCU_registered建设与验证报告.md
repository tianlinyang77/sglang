# DCU registered 建设与验证报告

## 1. 背景

SGLang 原有测试体系里存在两类测试入口：

- `test/srt/`：历史模型与 runtime 测试集合，更多用于上游开发者本地验证。
- `test/registered/`：当前 CI 注册测试集合，通过 `ci_register.py` 显式声明硬件、suite、预估耗时、nightly 属性等元信息，再由 `test/run_suite.py` 和 GitHub workflow 收集执行。

本次 DCU/BW1000 建设选择沉淀到 `test/registered/dcu/`，而不是继续在 `test/srt/` 下新增 DCU 目录，主要原因如下。

| 对比项 | `test/srt/` | `test/registered/dcu/` |
|---|---|---|
| CI 注册 | 不强制注册，难以按硬件和 suite 精确调度 | 每个文件调用 `register_dcu_ci(...)`，可被 `run_suite.py --hw dcu` 收集 |
| 硬件隔离 | 测试经常默认 CUDA/NVIDIA 语义，需要逐个适配 | DCU 专用目录，可集中维护 BW1000/K100 等差异 |
| 模型路径 | 多数依赖默认 HF 模型名，容易触发下载或 gated 模型失败 | 默认使用本地模型路径，环境变量可覆盖，路径缺失可 skip/fail fast |
| 分层能力 | 不天然区分 smoke、nightly、accuracy、kernel、VLM | 可按 suite 分层：per-commit、nightly、accuracy、vlm、kernel |
| 维护成本 | 历史测例混杂，迁移时容易影响上游通用逻辑 | 与 `amd/`、`ascend/` 同风格，便于长期维护和官方合入 |

因此，`test/srt/` 当前作为迁移参考来源使用；正式 DCU CI 测例沉淀在 `test/registered/dcu/`。

## 2. 建设目标

本阶段目标不是简单搬运模型测例，而是建立一套可长期维护的 DCU registered CI 分层：

1. 用 `register_dcu_ci(...)` 把 DCU 测例纳入统一 CI 收集。
2. 按功能类型划分目录，避免所有模型混在一个文件夹里。
3. 默认使用 BW1000 实机已验证的本地模型路径。
4. 启动参数统一吸收 DCU cookbook 和实测经验：`fa3`、`page-size 64`、VLM 的 `mm-attention-backend fa3` 等。
5. smoke 测试只验证“server 能启动、接口能返回非空结果”；精度阈值集中放到 accuracy/nightly。
6. 对当前 DCU 不支持或暂不稳定的路径先做白名单/环境变量控制，避免把不可复现失败放进主线门禁。


### 2.1 sglang-tly 合入补充（2026-05-25）

本轮以当前 `sglang/` 为主线，吸收 `sglang-tly/` 的 DCU 注册覆盖、CI workflow、覆盖分析脚本和文档矩阵。合入规则是：

- 当前 BW1000 专项目录不回退到 `k100/` 命名。
- `sglang-tly` 中的 `stage-a-dcu`、`stage-b-dcu` 迁移到当前标准 suite。
- 已有 BW1000 实测通过的专项测例保持当前实现。
- `sglang-tly` 的 broad registration 作为覆盖信息迁入；未复测、disabled 或专项失败原因保留，不直接等同于当前 BW1000 通过。
- 新增 `scripts/ci/dcu/analyze_dcu_csv_coverage.py` 作为 CSV 与 AMD/NVIDIA 交集覆盖矩阵分析入口。

迁入的文档矩阵包括 `DCU当前测试差距分析.md`、`DCU_CI建设步骤清单.md`、`DCU专项测试推进表.md`、`DCU_CSV覆盖矩阵.md`、`DCU_AMD_NVIDIA交集覆盖矩阵.md` 和 `DCU_CSV覆盖未复测合入清单.md`。

## 3. 目录建设情况

当前 DCU registered 目录结构如下：

```text
test/registered/dcu/
├── README.md
├── interface/
│   └── test_dcu_smoke.py
├── accuracy/
│   ├── README.md
│   └── bw1000/
│       ├── test_gsm8k_eval_dcu.py
│       ├── test_mmlu_eval_dcu.py
│       └── test_mmmu_eval_dcu.py
├── srt/
│   └── bw1000/
│       ├── test_qwen25_0p5b_server_dcu.py
│       ├── test_qwen25_1p5b_server_dcu.py
│       └── test_qwen25_7b_server_dcu.py
├── vlm_models/
│   └── bw1000/
│       └── test_qwen25_vl_3b_server_dcu.py
├── moe/
│   └── bw1000/
│       ├── test_qwen3_30b_moe_server_dcu.py
│       └── test_qwen3_30b_moe_instruct_dcu.py
├── embedding/
│   └── bw1000/
│       └── test_gte_qwen2_embedding_dcu.py
├── reranker/
│   └── bw1000/
│       └── test_qwen3_reranker_dcu.py
└── kernels/
    └── test_sgl_kernel_supported_dcu.py
```

## 4. Suite 注册情况

### 4.1 Per-commit suite

| Suite | 当前用途 | 代表测例 |
|---|---|---|
| `stage-a-test-1-gpu-small-dcu` | 最小 DCU smoke | `interface/test_dcu_smoke.py` |
| `stage-b-test-1-gpu-small-dcu` | 单卡模型/API/kernel smoke | dense、VLM、embedding、reranker、kernel whitelist |
| `stage-b-test-1-gpu-large-dcu` | 较重单卡后端/attention 测试 | attention、MLA |
| `stage-b-test-2-gpu-large-dcu` | 多卡数据并行类测试 | distributed |
| `stage-c-test-large-8-gpu-dcu` | 8 卡大测试预留 | 后续扩展 |

### 4.2 Nightly suite

| Suite | 当前用途 | 代表测例 |
|---|---|---|
| `nightly-dcu` | MoE 与较重 kernel nightly | Qwen3 MoE、kernel nightly |
| `nightly-dcu-accuracy` | 文本精度 nightly | GSM8K、MMLU |
| `nightly-dcu-vlm` | VLM/MMMU nightly | MMMU、VLM 相关 |
| `nightly-dcu-perf` | 性能 nightly | bench serving / one batch |
| `nightly-dcu-1-gpu/4-gpu/8-gpu` | 后续按卡数扩展 | 预留/部分已有 |

## 5. 统一工具与参数

新增/使用了 `python/sglang/test/dcu_utils.py` 作为 DCU 测试公共 helper，集中管理：

- 默认 server args。
- 本地模型路径读取与缺失处理。
- OpenAI API base URL。
- chat/generate/embedding/reranker 请求检查。
- 环境变量整数读取。

默认文本参数：

```text
--attention-backend fa3
--page-size 64
--trust-remote-code
--log-level warning
--log-level-http warning
```

VLM 额外参数：

```text
--mm-attention-backend fa3
--enable-multimodal
```

MoE 实测需要 TP1卡死，TP2正常：

```text
--tp-size 2
```

## 6. 模型覆盖情况

| 类型 | 默认模型 | 目录 | 状态 |
|---|---|---|---|
| Dense text 1.5B | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-1.5B-Instruct` | `srt/bw1000/` | 已通过 |
| Dense text 7B | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct` | `srt/bw1000/` | 已通过 |
| VLM | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-3B-Instruct` | `vlm_models/bw1000/` | 已通过 |
| MoE base | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B` | `moe/bw1000/` | TP2 已通过 |
| MoE instruct | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B-Instruct-2507` | `moe/bw1000/` | TP2 已通过 |
| Embedding | `/public/opendas/DL_DATA/llm-models/vllm-optest-models/Qwen/Qwen3-Embedding-0.6B` | `embedding/bw1000/` | 已通过 |
| Reranker | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Reranker-0.6B` | `reranker/bw1000/` | 已通过 |
| Accuracy text | Qwen2.5-7B-Instruct | `accuracy/bw1000/` | GSM8K/MMLU smoke 已跑通 |
| Accuracy VLM | Qwen2.5-VL-3B-Instruct | `accuracy/bw1000/` | MMMU smoke 已跑通 |

### Embedding 模型调整说明

原计划默认使用：

```text
/public/opendas/DL_DATA/llm-models/vllm-optest-models/Alibaba-NLP/gte-Qwen2-1.5B-instruct
```

但当前 `dxl-sglang` 容器中的 transformers 版本为 `5.3.0`，该模型动态 tokenizer 会导入：

```text
transformers.models.qwen2.tokenization_qwen2_fast
```

当前环境中该模块不存在，因此启动失败。即使加 `--tokenizer-mode slow` 也会在 import 阶段失败。为保证 registered smoke 可复现，默认模型改为已验证通过的：

```text
/public/opendas/DL_DATA/llm-models/vllm-optest-models/Qwen/Qwen3-Embedding-0.6B
```

`gte-Qwen2-1.5B-instruct` 后续可在 tokenizer/transformers 兼容性修复后通过 `SGLANG_DCU_EMBEDDING_MODEL` 切回验证。

### MoE TP2 说明

Qwen3-30B-A3B 两个 MoE 模型目录约 57G。单卡可以开始加载，但实测会在 fused MoE 权重加载/布局转换阶段非常慢，表现为 server 端口未 ready，父进程等待 scheduler ready。

改用两卡 TP2 后，16 个 checkpoint shard 能快速加载并完成短生成。因此 MoE smoke 默认使用：

```text
--tp-size 2
```

## 7. 实机验证环境

| 项 | 内容 |
|---|---|
| 节点 | `10.16.1.66` |
| 硬件 | BW1000 x 8 |
| 容器 | `dxl-sglang` |
| wrapper | `scripts/ci/dcu/dcu_ci_exec.sh` |
| 工作目录 | `/workspace/sglang` |
| 模型挂载 | `/public/opendas/DL_DATA/llm-models` |
| DCU 监控 | `hy-smi` |

推荐执行方式：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && \
DCU_CI_CONTAINER=dxl-sglang \
bash scripts/ci/dcu/dcu_ci_exec.sh \
-w /workspace/sglang \
-e CUDA_VISIBLE_DEVICES=1 \
-- python3 -m pytest -q test/registered/dcu/srt/bw1000/test_qwen25_1p5b_server_dcu.py'
```

## 8. 验证结果汇总

### 8.1 静态与注册检查

| 检查项 | 结果 | 说明 |
|---|---|---|
| `py_compile` | 通过 | 覆盖 DCU helper、dense、VLM、embedding、reranker、MoE、kernel、`run_suite.py` |
| `verify_dcu_registration.py` | 通过 | 收集到 49 个 DCU registered test file |
| `run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --list` | 通过 | 列出 30 个 stage-b DCU small 测例，不执行 pytest |

### 8.2 单项实机 smoke

| 测试文件 | 卡隔离 | 结果 | 备注 |
|---|---|---|---|
| `test_qwen25_1p5b_server_dcu.py` | `CUDA_VISIBLE_DEVICES=1` | `2 passed in 82.31s` | server 命令确认 `--device cuda` |
| `test_qwen25_7b_server_dcu.py` | 单卡 | `2 passed in 722.52s` | 需要较长 timeout，已调到 1800s |
| `test_qwen25_vl_3b_server_dcu.py` | 单卡 | `1 passed in 372.08s` | 需要 `--mm-attention-backend fa3` |
| `test_qwen3_reranker_dcu.py` | 单卡 | `1 passed in 221.33s` | `/v1/rerank` 返回 score |
| `test_sgl_kernel_supported_dcu.py` | 单卡 | `1 passed in 89.18s` | 只跑当前 BW1000 支持白名单 |
| `test_gte_qwen2_embedding_dcu.py` | `CUDA_VISIBLE_DEVICES=3` | `2 passed in 81.92s` | 默认已改为 Qwen3-Embedding-0.6B |
| `test_qwen3_30b_moe_server_dcu.py` | `CUDA_VISIBLE_DEVICES=5,6` | `1 passed in 131.83s` | 默认 TP2 |
| `test_qwen3_30b_moe_instruct_dcu.py` | `CUDA_VISIBLE_DEVICES=5,6` | `1 passed in 112.01s` | 默认 TP2 |

### 8.3 精度 smoke

| 测试 | 数据集 | 模型 | 当前状态 |
|---|---|---|---|
| GSM8K | GSM8K | Qwen2.5-7B-Instruct | 10 样本 smoke 已跑通，历史 3 次 score 为 `0.700/0.700/0.700` |
| MMLU | MMLU | Qwen2.5-7B-Instruct | 50 样本 smoke 已跑通，历史 3 次 score 为 `0.720/0.720/0.780` |
| MMMU | MMMU | Qwen2.5-VL-3B-Instruct | 10 样本 smoke 已跑通，历史 3 次 score 为 `0.400/0.400/0.400` |

说明：以上是 smoke 阈值参考，不等同于 nightly 阈值。nightly 阈值仍需按固定样本数连续 3 次基线后固化。

## 9. 已定位问题与处理

| 问题 | 现象 | 根因 | 当前处理 |
|---|---|---|---|
| 可见卡环境变量混用 | `No HIP GPUs are available` | 同时设置 `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES` 会干扰 torch/SGLang 设备识别 | 并行测试统一只使用 `CUDA_VISIBLE_DEVICES` |
| `/generate` 401 | dense generate 接口报未授权 | server 启动带 `api_key`，helper 请求未带 Authorization | `assert_generate_non_empty()` 支持传入 api key |
| 7B/VLM 启动超时 | 600s 内未 ready | 模型加载和图捕获耗时较长 | 7B/VLM timeout 调到 1800s |
| gte-Qwen2 embedding 启动失败 | `ModuleNotFoundError: tokenization_qwen2_fast` | 当前 transformers 5.3.0 缺少该 fast tokenizer 模块，模型动态 tokenizer import 失败 | 默认改用 Qwen3-Embedding-0.6B，gte 保留环境变量覆盖 |
| Qwen3 MoE 单卡长时间不 ready | 父进程等待 scheduler ready，端口未监听 | 单卡加载 57G MoE 时 fused MoE 权重加载/布局转换慢且不稳定 | 默认改为 TP2，两卡实测通过 |
| sgl-kernel 全量测试失败较多 | 多个 kernel 文件 fail/timeout | 部分算子未接入或当前 DCU 路径不支持 | registered kernel 只跑 BW1000 已通过白名单 |

## 10. 当前完成度

| 模块 | 完成度 | 说明 |
|---|---|---|
| DCU suite 注册 | 已完成 | `HWBackend.DCU`、per-commit、nightly suite 已接入 |
| 目录规范 | 已完成 | `test/registered/dcu/README.md` 已说明目录和 env |
| Dense smoke | 已完成 | 1.5B、7B 均通过 |
| VLM smoke | 已完成 | Qwen2.5-VL-3B 通过 |
| Embedding smoke | 已完成 | Qwen3-Embedding-0.6B 通过 |
| Reranker smoke | 已完成 | Qwen3-Reranker-0.6B 通过 |
| MoE smoke | 已完成 | base/instruct TP2 均通过 |
| Kernel whitelist | 已完成首版 | 只运行当前 DCU 支持集合 |
| Accuracy smoke | 已完成首版 | GSM8K/MMLU/MMMU 小样本跑通 |
| Nightly 阈值 | 待继续 | 需要固定样本数连续 3 次基线 |
| Workflow 收敛 | 进行中 | 需确保 MoE 两卡资源、accuracy/vlm env 注入 |

## 11. 后续建议

1. **把 MoE suite 的资源语义明确成两卡**
   - MoE 当前虽然注册在 `nightly-dcu`，但默认需要 `CUDA_VISIBLE_DEVICES=5,6` 这类两卡环境。
   - 建议后续单独拆出 `nightly-dcu-moe` 或在 workflow 中为 MoE 文件分配两卡。

2. **继续固化精度 nightly 阈值**
   - GSM8K：全量或固定 1319 样本，连续 3 次。
   - MMLU：固定 5000 样本，连续 3 次。
   - MMMU：固定 100 样本，连续 3 次。
   - 阈值按最小值下浮 1-3 个百分点，不直接复用 smoke 阈值。

3. **补 embedding gte-Qwen2 兼容路径**
   - 可选方案一：升级/调整 transformers，使 `tokenization_qwen2_fast` 可导入。
   - 可选方案二：修正模型动态 tokenizer 缓存或模型文件。
   - 可选方案三：长期以 Qwen3-Embedding-0.6B 作为 DCU smoke 默认，gte-Qwen2 作为专项兼容测试。

4. **kernel 白名单持续维护**
   - 已通过集合进入 registered。
   - 当前 fail/timeout 的 kernel 文件先不进门禁，等底层算子适配后再逐步放开。

5. **保持 `/test/srt` 只作参考**
   - 后续迁移模型类别时继续参考 `test/srt/`、`test/registered/amd/`、`test/registered/ascend/` 的测试逻辑。
   - 正式 DCU CI 仍统一放到 `test/registered/dcu/`，保持注册、分层、模型路径和硬件语义清晰。

## 12. 结论

DCU registered 测试体系已经形成首版可维护结构：从最小 smoke、dense server、VLM、MoE、embedding、reranker、kernel whitelist 到 accuracy smoke 都有对应目录和注册 suite。

当前实机验证表明，BW1000 上核心 smoke 链路已经跑通；主要剩余工作是 nightly 阈值固化、MoE 两卡资源编排、以及 gte-Qwen2 embedding tokenizer 兼容性收敛。
