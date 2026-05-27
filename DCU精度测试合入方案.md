# DCU/BW1000 精度测试合入方案

## 背景

第三阶段补齐 DCU 精度 CI。`step3.md` 中的 `k100` 是早期规划名称，当前已按实机环境修正为 BW1000，目录固定为：

```text
test/registered/dcu/accuracy/bw1000/
```

首批用例复用 SGLang 现有 eval 框架，不另建评测逻辑：

- `test_gsm8k_eval_dcu.py`：GSM8K 文本数学推理精度。
- `test_mmlu_eval_dcu.py`：MMLU 文本通用知识精度。
- `test_mmmu_eval_dcu.py`：MMMU 多模态理解精度。

## 模型与数据

默认文本模型：

```text
/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct
```

默认 VLM 模型：

```text
/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-3B-Instruct
```

如果 Qwen2.5-VL-3B 不存在，MMMU 测试会 fallback 到：

```text
/public/opendas/DL_DATA/llm-models/qwen2/Qwen2-VL-2B-Instruct
```

MMLU 默认本地数据：

```text
/public/opendas/DL_DATA/llm-models/datasets/mmlu
```

MMMU 默认本地数据：

```text
/public/opendas/DL_DATA/llm-models/multimodal-datasets/MMMU
```

本地模型路径一旦配置必须存在，避免 CI 隐式下载 gated 模型。本地数据存在时优先使用本地数据；本地路径不存在且没有显式配置时才允许沿用上游远程默认。

## 环境变量

文本精度：

- `SGLANG_DCU_GSM8K_MODEL`
- `SGLANG_DCU_MMLU_MODEL`
- `SGLANG_DCU_GSM8K_DATA_PATH`
- `SGLANG_DCU_MMLU_DATASET_PATH`
- `SGLANG_DCU_GSM8K_NUM_EXAMPLES`
- `SGLANG_DCU_MMLU_NUM_EXAMPLES`
- `SGLANG_DCU_EVAL_NUM_EXAMPLES`：兼容旧调试命令的 fallback。
- `SGLANG_DCU_GSM8K_THRESHOLD`
- `SGLANG_DCU_MMLU_THRESHOLD`
- `SGLANG_DCU_GSM8K_SERVER_ARGS`
- `SGLANG_DCU_MMLU_SERVER_ARGS`

多模态精度：

- `SGLANG_DCU_MMMU_MODEL`
- `SGLANG_DCU_MMMU_DATASET_PATH`
- `SGLANG_DCU_MMMU_NUM_EXAMPLES`
- `SGLANG_DCU_MMMU_NUM_THREADS`
- `SGLANG_DCU_MMMU_THRESHOLD`
- `SGLANG_DCU_MMMU_LATENCY_THRESHOLD`
- `SGLANG_DCU_MMMU_SERVER_ARGS`

## 默认启动参数

文本模型默认：

```text
--attention-backend fa3 --page-size 64 --log-level warning --log-level-http warning --trust-remote-code
```

VLM 默认：

```text
--attention-backend fa3 --mm-attention-backend fa3 --page-size 64 --log-level warning --log-level-http warning --enable-multimodal --trust-remote-code
```

`--mem-fraction-static` 不在测试里写死，按模型和节点状态通过 `SGLANG_DCU_*_SERVER_ARGS` 覆盖。

## Suite 与 workflow

- GSM8K/MMLU 注册到 `nightly-dcu-accuracy`。
- MMMU 注册到 `nightly-dcu-vlm`。
- `.github/workflows/nightly-test-dcu.yml` 支持 workflow dispatch 选择 `nightly-dcu-accuracy`、`nightly-dcu-vlm`、`nightly-dcu-perf`。

## 阈值策略

当前 smoke 阈值来自 BW1000 三次小样本实测：

- GSM8K 10 样本：`0.700 / 0.700 / 0.700`，默认阈值 `0.65`。
- MMLU 50 样本：`0.720 / 0.720 / 0.780`，默认阈值 `0.68`。
- MMMU 10 样本：`0.400 / 0.400 / 0.400`，默认阈值 `0.35`。

Nightly 阈值不能直接复用 smoke。需要固定模型、固定数据、固定样本数后连续跑三次：

- GSM8K：全量或固定 1319 样本。
- MMLU：固定 5000 样本。
- MMMU：固定 100 样本。

GSM8K/MMLU 取三次最小值下浮 1 到 2 个百分点；MMMU 取三次最小值下浮 2 到 3 个百分点。MMMU latency 先记录，稳定后再按三次最大值的 1.5 倍设置门禁。
