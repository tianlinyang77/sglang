# DCU/BW1000 test/srt 模型映射与迁移建议

## 1. 目标

本文梳理 `test/srt/` 中现有 CPU、XPU、Ascend、configs 测试涉及的模型，并映射到当前 DCU/BW1000 集群模型目录：

```text
/public/opendas/DL_DATA/llm-models
```

目标不是直接复用 `test/srt/` 原脚本，而是判断每类测试在 DCU 上是否有可用模型基础，后续迁移到：

```text
test/registered/dcu/
```

## 2. 总体结论

`test/srt/` 不是 DCU 专用测试目录，现有文件主要分为：

| 目录/文件 | 原定位 | DCU 复用方式 |
|---|---|---|
| `test/srt/cpu/` | CPU / Intel AMX 后端测试 | 部分模型与测试思路可迁移；原脚本多测 CPU kernel，不算 DCU 硬件覆盖 |
| `test/srt/xpu/` | Intel XPU 后端测试 | 模型可部分对应；启动参数需改成 DCU |
| `test/srt/ascend/` | Ascend NPU 后端测试 | Qwen/DeepSeek 相关测试思路可迁移；后端参数、模型路径、阈值都要重写 |
| `test/srt/configs/` | benchmark/server 配置样例 | 可参考 workload 形态；不建议直接纳入 DCU CI |
| `test/srt/run_suite.py` | 旧 suite 调度入口 | 当前 AMD suite 基本为空，CUDA/AMD 大量测试已迁移到 `test/registered/` |

当前 DCU 集群模型覆盖较好，能找到以下主要类别：

```text
Qwen2.5 dense
Qwen2.5 VL
Qwen3 dense
Qwen3 MoE
Qwen3 VL
DeepSeek V2/V3/R1
Mixtral
Embedding / Reranker
AWQ / GPTQ / W8A8 / FP8 量化模型
EAGLE draft 模型
```

缺口主要是：

```text
meta-llama/Llama-3.2-1B
meta-llama/Llama-3.2-1B-Instruct
deepseek-ai/DeepSeek-OCR
inclusionAI/LLaDA2.0-mini
部分完全同名的 Ascend 专用量化模型
```

## 3. test/srt 中的模型来源

### 3.1 `python/sglang/test/test_utils.py` 默认模型

`test/srt/cpu/` 和 `test/srt/xpu/` 中很多测试不是直接写模型路径，而是引用 `test_utils.py` 中的默认模型常量。

| 默认常量 | 原模型 | 当前集群候选 | 匹配状态 | 建议 |
|---|---|---|---|---|
| `DEFAULT_MODEL_NAME_FOR_TEST` | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct` | 同左 | 完全匹配 | DCU 首批文本主模型 |
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` | `meta-llama/Llama-3.2-1B-Instruct` | 暂未发现完全同名 | 缺失 | 用 `Qwen2.5-1.5B-Instruct` 替代 |
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE` | `meta-llama/Llama-3.2-1B` | 暂未发现完全同名 | 缺失 | 用 `Qwen2.5-1.5B` 或 `Qwen3-0.6B` 替代 |
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE` | `Qwen/Qwen3-Reranker-0.6B` | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Reranker-0.6B` | 同族本地匹配 | 可做 reranker 专项 |
| `DEFAULT_MOE_MODEL_NAME_FOR_TEST` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `/public/opendas/DL_DATA/llm-models/mixtral/mixtral_vllm` | 同族候选 | 需确认 HF 结构和 DCU MoE 支持 |
| `DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE` | `Qwen/Qwen1.5-MoE-A2.7B` | 暂未发现完全同名 | 缺失 | 可用 Qwen3 MoE 替代 |
| `DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 暂未发现完全同名 | 缺失 | 可用 Qwen3 MoE Instruct 替代 |
| `DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | `/public/opendas/DL_DATA/llm-models/vllm-optest-models/Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 模型存在但当前镜像 tokenizer 不兼容 | 当前默认改用 Qwen3-Embedding-0.6B；gte 待 transformers/tokenizer 兼容后再启用 |
| `DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME_FOR_TEST` | `cross-encoder/ms-marco-MiniLM-L6-v2` | 暂未发现完全同名 | 缺失 | 暂不纳入首批 |
| `DEFAULT_MLA_MODEL_NAME_FOR_TEST` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | `/public/opendas/DL_DATA/llm-models/deepseek-v2/DeepSeek-V2-Lite`、`DeepSeek-V2-Lite-Chat` | 同族候选 | 可做 DeepSeek/MLA 专项 |
| `DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST` | `neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8` | 有 DeepSeek FP8/Channel-FP8 类模型 | 同类候选 | 量化专项后置 |
| `DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Next-80B-A3B-Instruct` | 完全匹配 | 大模型 nightly 后置 |
| `DEFAULT_MODEL_NAME_FOR_TEST_VL_PP` | `Qwen/Qwen3-VL-2B-Thinking` | 有 `Qwen3-VL-*` 多个候选 | 同族候选 | VLM 专项可扩展 |
| `DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP` | `zai-org/GLM-4.1V-9B-Thinking` | `/public/opendas/DL_DATA/llm-models/GLM-4.1V-9B-Thinking`、`glm4/GLM-4.1V-9B-Thinking-*` | 同族候选 | VLM 后续扩展 |
| `DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8` | `Qwen/Qwen3-1.7B-FP8` | 有 `Qwen3-0.6B-FP8`、`qwen3-14B-fp8`、`qwen3-32B-fp8` | 同类候选 | FP8 kernel 未完全覆盖，后置 |
| `DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE` | `gaunernst/DeepSeek-V2-Lite-Chat-FP8` | 有 DeepSeek FP8 类模型 | 同类候选 | 后置 |
| `DEFAULT_MODEL_NAME_FOR_TEST_W8A8` | `RedHatAI/Llama-3.2-3B-quantized.w8a8` | 有 Qwen/GLM/DeepSeek W8A8 类模型 | 同类候选 | 后置 |
| `DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE` | `nytopop/Qwen3-30B-A3B.w8a8` | 有 `Qwen3.5-*W8A8`、`Qwen3-30B-A3B`、AWQ/FP8 类模型 | 同类候选 | 后置 |
| `DEFAULT_TARGET_MODEL_EAGLE` | `meta-llama/Llama-2-7b-chat-hf` | 暂未发现完全同名 | 缺失 | 可考虑换 Qwen EAGLE |
| `DEFAULT_DRAFT_MODEL_EAGLE` | `lmsys/sglang-EAGLE-llama2-chat-7B` | 暂未发现完全同名 | 缺失 | 可考虑换 Qwen EAGLE |
| `DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN` | `Qwen/Qwen3-30B-A3B` | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B` | 同族本地匹配 | EAGLE 专项后置 |
| `DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN` | `Tengyunw/qwen3_30b_moe_eagle3` | `/public/opendas/DL_DATA/llm-models/eagle/EAGLE-Qwen2-72B-Instruct` | 不完全匹配 | 需单独找 draft/target 对 |
| `DEFAULT_TARGET_MODEL_NGRAM` | `Qwen/Qwen2.5-Coder-7B-Instruct` | 有 Qwen2.5-Coder-32B-Instruct，未见 7B | 同族候选 | 可用已有 Coder 模型替代 |
| `DEFAULT_REASONING_MODEL_NAME_FOR_TEST` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `/public/opendas/DL_DATA/llm-models/deepseek-r1/DeepSeek-R1-Distill-Qwen-7B` | 完全匹配 | 可做 reasoning 专项 |
| `DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST` | `deepseek-ai/DeepSeek-V3-0324` | `/public/opendas/DL_DATA/llm-models/deepseek-v3/DeepSeek-V3-0324-bf16` | 同族候选 | 大模型后置 |
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN` | `Qwen/Qwen2.5-1.5B-Instruct` | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-1.5B-Instruct` | 完全匹配 | DCU smoke 首选 |
| `DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST` | `Qwen/Qwen2.5-VL-3B-Instruct` | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-3B-Instruct` | 完全匹配 | DCU VLM smoke 首选 |

## 4. 按 test/srt 子目录汇总

### 4.1 `test/srt/cpu/`

这部分主要服务 CPU / Intel AMX，不是 DCU 硬件测试。

| 测试文件 | 原测试内容 | 涉及模型 | DCU 可迁移性 |
|---|---|---|---|
| `cpu/test_activation.py` | CPU activation kernel | `dummy` | 不建议迁移为模型测试；可参考 kernel 单测思路 |
| `cpu/test_binding.py` | CPU binding/接口 | 多为 CPU 路径 | 不建议作为 DCU CI |
| `cpu/test_bmm.py` | CPU BMM | 无真实模型 | 可参考算子测试，但应走 sgl-kernel DCU 白名单 |
| `cpu/test_causal_conv1d.py` | CPU causal conv1d | 无真实模型 | 当前 DCU sgl-kernel 对应 op 未完全注册，后置 |
| `cpu/test_cpu_graph.py` | CPU graph + MLA eval | `DeepSeek-Coder-V2-Lite-Instruct` | 可迁移为 DCU DeepSeek/MLA 专项 |
| `cpu/test_decode.py` | CPU decode attention | 无真实模型 | 不建议首批 |
| `cpu/test_extend.py` | CPU extend/prefill | 无真实模型 | 不建议首批 |
| `cpu/test_flash_attn.py` | CPU flash attention fallback | 无真实模型 | 不建议首批 |
| `cpu/test_gemm.py` | CPU GEMM | 无真实模型 | DCU 应走 kernel 专项 |
| `cpu/test_intel_amx_attention_backend_a.py` | Intel AMX 文本模型 latency/MMLU | Qwen2.5-7B、DeepSeek-Coder-V2-Lite | 可迁移测试思路，后端参数需替换为 DCU |
| `cpu/test_intel_amx_attention_backend_b.py` | Intel AMX FP8 | Qwen3 FP8、DeepSeek FP8 MoE | 有同类模型，DCU FP8 kernel 后置 |
| `cpu/test_intel_amx_attention_backend_c.py` | Intel AMX W8A8 | Llama W8A8、Qwen3 W8A8 MoE | 有同类模型，DCU W8A8 后置 |
| `cpu/test_mamba.py` | CPU Mamba | 无明确本地首选模型 | 后置 |
| `cpu/test_mla.py` | CPU MLA | 多为 synthetic | 可参考，不作为首批模型测试 |
| `cpu/test_moe.py` | CPU MoE | synthetic / CPU MoE | 可参考 MoE 测试逻辑 |
| `cpu/test_norm.py` | CPU norm | synthetic | DCU norm kernel 当前未完全注册，后置 |
| `cpu/test_qkv_proj_with_rope.py` | QKV + RoPE | synthetic | 可参考算子级测试 |
| `cpu/test_qwen3.py` | Qwen3 配置/逻辑 | 无真实模型启动 | 可参考 Qwen3 单测 |
| `cpu/test_rope.py` | RoPE 逻辑 | `dummy` | 可参考单测 |
| `cpu/test_shared_expert.py` | shared expert 逻辑 | `dummy` | 可参考 MoE 单测 |
| `cpu/test_topk.py` | CPU top-k / MoE routing | synthetic | DCU 已有 sgl-kernel topk 通过，可迁移到 kernel CI |

### 4.2 `test/srt/xpu/`

这部分服务 Intel XPU。

| 测试文件 | 原测试内容 | 涉及模型 | 集群候选 | DCU 可迁移性 |
|---|---|---|---|---|
| `xpu/test_intel_xpu_backend.py` | XPU backend latency/attention | `Qwen2.5-1.5B-Instruct`、`Llama-3.2-1B` | Qwen2.5-1.5B 有；Llama 1B 缺失 | 可迁移 Qwen 小模型 smoke |
| `xpu/test_deepseek_ocr.py` | DeepSeek OCR VLM/OCR | `deepseek-ai/DeepSeek-OCR` | 未发现完全同名 | 暂不迁移 |

迁移时需要删除或替换：

```text
--device xpu
--attention-backend intel_xpu
```

替换为 DCU 当前稳定参数：

```text
--attention-backend fa3
--page-size 64
--trust-remote-code
```

### 4.3 `test/srt/ascend/`

这部分服务 Ascend NPU，但模型类别覆盖最丰富，适合借鉴迁移。

| 测试文件 | 原测试内容 | 原模型 | 集群候选 | 迁移优先级 |
|---|---|---|---|---|
| `ascend/test_ascend_tp1_bf16.py` | TP1 BF16 + GSM8K | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P0 |
| `ascend/test_ascend_tp2_bf16.py` | TP2 BF16 + GSM8K | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P1 |
| `ascend/test_ascend_graph_tp1_bf16.py` | graph TP1 | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P1 |
| `ascend/test_ascend_graph_tp2_bf16.py` | graph TP2 | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P2 |
| `ascend/test_ascend_compile_graph_tp1_bf16.py` | compile graph TP1 | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P2 |
| `ascend/test_ascend_hicache_mha.py` | HiCache MHA | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P2 |
| `ascend/test_ascend_sampling_backend.py` | sampling backend | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P1 |
| `ascend/test_ascend_piecewise_graph_prefill.py` | piecewise graph prefill | Qwen2.5-7B-Instruct | `qwen2.5/Qwen2.5-7B-Instruct` | P2 |
| `ascend/test_ascend_tp4_bf16.py` | TP4 Qwen3 MoE | Qwen3-30B-A3B-Instruct-2507 | `qwen3/Qwen3-30B-A3B-Instruct-2507` | P2 |
| `ascend/test_ascend_gptq_moe.py` | GPTQ MoE | Qwen3-30B-A3B-GPTQ-Int4 | `vllm-awq-models/Qwen3-30B-A3B-*`、`qwen3/Qwen3-30B-A3B` | P3 |
| `ascend/test_ascend_autoround_dense.py` | AutoRound dense | Qwen3-8B-int4-AutoRound | `qwen3/Qwen3-8B`，AutoRound 同名未发现 | P3 |
| `ascend/test_ascend_autoround_moe.py` | AutoRound MoE | Qwen3-30B-A3B-Instruct-2507-int4-AutoRound | `qwen3/Qwen3-30B-A3B-Instruct-2507`、AWQ 候选 | P3 |
| `ascend/test_ascend_w8a8_quantization.py` | W8A8 quantization | Qwen2.5-0.5B W8A8 | 有 Qwen2.5/Qwen3/GLM W8A8 类模型 | P3 |
| `ascend/test_ascend_w4a4_quantization.py` | W4A4 quantization | Qwen3-32B-w4a4-LAOS | Qwen3-32B / Qwen3 FP8 候选 | P3 |
| `ascend/test_ascend_mla_w8a8int8.py` | DeepSeek MLA W8A8 | DeepSeek-V2-Lite-W8A8 | `deepseek-v2/DeepSeek-V2-Lite*` | P3 |
| `ascend/test_ascend_mla_fia_w8a8int8.py` | DeepSeek MLA FIA W8A8 | DeepSeek-V2-Lite-W8A8 | `deepseek-v2/DeepSeek-V2-Lite*` | P3 |
| `ascend/test_ascend_hicache_mla.py` | HiCache MLA | DeepSeek-V2-Lite-W8A8 | `deepseek-v2/DeepSeek-V2-Lite*` | P3 |
| `ascend/test_ascend_deepep.py` | DeepEP / DeepSeek R1 | DeepSeek-R1-0528-W8A8 | DeepSeek R1/FP8/INT8/AWQ 类模型 | P3 |
| `ascend/test_embed_interpolate_unittest.py` | Qwen3-VL vision embedding interpolate | dummy config, `device=npu` | Qwen3-VL 模型丰富 | P2，可改为 DCU VLM unit |
| `ascend/test_llada2_mini_ascend.py` | LLaDA2 mini | LLaDA2.0-mini | 未发现同名 | 暂不迁移 |

迁移时需要替换：

```text
--attention-backend ascend
--disable-cuda-graph
Ascend/NPU device 参数
/root/.cache/modelscope/hub/models/...
Ascend 专用精度阈值
```

替换为：

```text
--attention-backend fa3
--page-size 64
--trust-remote-code
必要时加 --mm-attention-backend fa3 --enable-multimodal
/public/opendas/DL_DATA/llm-models/...
DCU/BW1000 实测阈值
```

### 4.4 `test/srt/configs/`

这些 YAML 是 benchmark/server 对比配置，主要包含：

| 配置文件 | 原模型 | 集群候选 | 建议 |
|---|---|---|---|
| `random_config.yaml` | `meta-llama/Llama-3.1-8B-Instruct` | 未发现完全同名 | 可用 Qwen2.5-7B-Instruct 替代 |
| `random_flashinfer_vs_triton_config.yaml` | `meta-llama/Llama-3.1-8B-Instruct` + flashinfer/triton 对比 | 未发现完全同名 | DCU 不适合直接套 flashinfer/triton 对比 |
| `sharegpt_config.yaml` | `meta-llama/Llama-3.1-8B-Instruct` | 未发现完全同名 | 可参考 ShareGPT workload |
| `deepseek_v3.yaml` | `deepseek-ai/DeepSeek-V3-0324` | `deepseek-v3/DeepSeek-V3-0324-bf16` | 大模型 nightly/性能专项后置 |
| `deepseek_v3_long_context.yaml` | `deepseek-ai/DeepSeek-V3-0324` | `deepseek-v3/DeepSeek-V3-0324-bf16` | 长上下文专项后置 |
| `llama_405b.yaml` | `nvidia/Llama-3.1-405B-Instruct-FP8` | 未发现完全同名 | 暂不迁移 |

## 5. DCU 可优先建设的模型矩阵

### 5.1 P0：首批 smoke / accuracy

| 类别 | 模型路径 | 用途 |
|---|---|---|
| 小文本模型 | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-1.5B-Instruct` | server smoke、基础生成 |
| 主文本模型 | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct` | GSM8K、MMLU、采样、TP1 |
| VLM 小模型 | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-3B-Instruct` | MMMU、VLM server smoke |

### 5.2 P1：功能专项

| 类别 | 模型路径 | 用途 |
|---|---|---|
| Qwen3 dense | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B` | Qwen3 dense server |
| Qwen3 MoE | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B` | MoE routing / MoE server；BW1000 实测需 TP2 |
| Qwen3 MoE Instruct | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B-Instruct-2507` | MoE accuracy / TP；BW1000 实测需 TP2 |
| DeepSeek V2 | `/public/opendas/DL_DATA/llm-models/deepseek-v2/DeepSeek-V2-Lite` | MLA 专项 |
| DeepSeek V2 Chat | `/public/opendas/DL_DATA/llm-models/deepseek-v2/DeepSeek-V2-Lite-Chat` | MLA chat/accuracy |
| Embedding | `/public/opendas/DL_DATA/llm-models/vllm-optest-models/Qwen/Qwen3-Embedding-0.6B` | embedding endpoint；已在 BW1000 通过，gte-Qwen2 当前 tokenizer 不兼容 |
| Reranker | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Reranker-0.6B` | score/reranker endpoint |

### 5.3 P2：nightly / 大模型专项

| 类别 | 模型路径 | 用途 |
|---|---|---|
| DeepSeek V3 | `/public/opendas/DL_DATA/llm-models/deepseek-v3/DeepSeek-V3-0324-bf16` | MLA/大模型/TP8 |
| DeepSeek R1 reasoning | `/public/opendas/DL_DATA/llm-models/deepseek-r1/DeepSeek-R1-Distill-Qwen-7B` | reasoning model |
| Qwen3 Next | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Next-80B-A3B-Instruct` | hybrid/mamba 后置 |
| Mixtral | `/public/opendas/DL_DATA/llm-models/mixtral/mixtral_vllm` | MoE 对照 |
| Qwen3 VL | `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-VL-8B-Instruct` | VLM 扩展 |
| GLM VLM | `/public/opendas/DL_DATA/llm-models/GLM-4.1V-9B-Thinking` | VLM 扩展 |

### 5.4 P3：量化专项

| 类别 | 候选路径 | 说明 |
|---|---|---|
| Qwen2.5 W8A8 | `/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-72B-quantized.w8a8` | 大模型，需确认 DCU W8A8 路径 |
| Qwen3 AWQ | `/public/opendas/DL_DATA/llm-models/vllm-awq-models/Qwen3-30B-A3B-AWQ` | AWQ 后置 |
| Qwen3 FP8 | `/public/opendas/DL_DATA/llm-models/vllm-fp8-models/Qwen3-30B-A3B-Instruct-2507-FP8` | FP8 kernel 当前不宜首批 |
| DeepSeek FP8 | `/public/opendas/DL_DATA/llm-models/vllm-fp8-models/DeepSeek-R1-Channel-FP8` | FP8 后置 |
| DeepSeek INT8 | `/public/opendas/DL_DATA/llm-models/deepseek0519/DeepSeek-R1-Channel-INT8` | INT8 后置 |
| GLM W8A8 | `/public/opendas/DL_DATA/llm-models/glm4.7/GLM-4.7-W8A8` | W8A8 后置 |

量化类模型虽然很多，但当前 DCU `sgl-kernel/tests` 已验证出 FP8/INT8/GPTQ/QServe 等不少算子未注册或不完整。因此不建议首批纳入门禁。

## 6. 建议迁移路线

### Phase A：先补 DCU SRT smoke

新增目录建议：

```text
test/registered/dcu/srt/bw1000/
```

首批测试建议：

```text
test_qwen25_1p5b_server_dcu.py
test_qwen25_7b_server_dcu.py
test_qwen25_7b_sampling_dcu.py
test_qwen25_vl_3b_server_dcu.py
```

目标：

- 验证 DCU server 能启动。
- 验证 OpenAI-compatible `/v1/chat/completions` 或 SGLang client 请求能返回。
- 验证 `fa3 + page_size=64` 的默认参数不回退。

### Phase B：迁移 test/srt 中最有价值的文本测试

从 Ascend/Intel AMX 测试中抽取：

```text
Qwen2.5-7B-Instruct + GSM8K/MMLU
Qwen2.5-7B-Instruct + sampling backend
Qwen2.5-7B-Instruct + TP1/TP2
```

注意：这部分不能复用原阈值，必须按 BW1000 实机重新定。

### Phase C：补 MoE / DeepSeek / VLM

候选：

```text
Qwen3-30B-A3B
Qwen3-30B-A3B-Instruct-2507
DeepSeek-V2-Lite
DeepSeek-V2-Lite-Chat
Qwen2.5-VL-3B-Instruct
```

目标：

- 覆盖 MoE routing。
- 覆盖 MLA。
- 覆盖 multimodal。

### Phase D：量化后置

量化模型包括 AWQ/GPTQ/W8A8/FP8，但建议在 DCU kernel 支持情况进一步明确后再接入。

优先处理顺序：

```text
W8A8 INT8
AWQ
GPTQ
FP8
QServe
```

## 7. 一句话结论

`test/srt` 的模型类别在当前 DCU 集群里基本都能找到候选，尤其是 Qwen2.5、Qwen3、Qwen-VL、DeepSeek、Embedding/Reranker、量化模型都比较完整。

但 `test/srt` 原脚本不能直接作为 DCU CI 使用，因为它们绑定了 CPU/Intel AMX、XPU、Ascend 的后端参数、路径和阈值。正确做法是以本文模型映射为基础，把可迁移项重写到 `test/registered/dcu/`，并使用 BW1000 实机重新确定启动参数和门禁阈值。

