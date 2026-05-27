# DCU/BW1000 sgl-kernel 测试验证汇总

## 1. 背景

本次验证目标是梳理 `sglang/sgl-kernel/tests` 下官方 sgl-kernel 测试在 DCU/BW1000 环境中的可运行情况，为后续 DCU CI 建设提供测试白名单和问题清单。

本次测试不依赖大模型权重，主要覆盖 sgl-kernel 中的底层算子、采样、MoE、KV cache、量化、通信等 kernel 级功能。

## 2. 测试环境

| 项目 | 内容 |
|---|---|
| 节点 | `10.16.1.66` |
| 硬件 | DCU BW1000 |
| 容器 | `dxl-sglang` |
| 仓库路径 | `/workspace/sglang` |
| sgl-kernel 路径 | `/workspace/sglang/sgl-kernel` |
| 测试汇总 | `/tmp/sgl_kernel_dcu_tests_installed_20260522_100156/summary.tsv` |
| 本地备份 | `/public/home/dingxl/sgl_kernel_dcu_tests_installed_20260522_100156_summary.tsv` |

基础软件包来自当前 DCU SGLang 镜像：

| 包 | 版本 |
|---|---|
| `torch` | `2.9.0+das.opt1.dtk2604.2605091832.g21362a` |
| `sglang` | `0.5.10rc0+das.opt2.alpha.dtk2604.torch290.2605081216.g9f67de` |
| `sglang-kernel` | `0.4.0+das.opt1.dtk2604.torch290.2605081216.g9f67de` |
| `aiter` | `0.1.2+das.opt1.dtk2604.torch290.2605071840.g1f8f50` |
| HIP | `6.3.26113` |

## 3. 测试方法

本次测试使用容器内已安装的 `sglang-kernel` wheel 进行验证。

关键点：

1. 使用 installed wheel，而不是直接加载本地 `sgl-kernel/python`。
2. `PYTHONPATH` 只指向 `/workspace/sglang/python`。
3. 开启 Hugging Face 离线模式，避免测试过程中隐式下载样例数据。
4. 单文件执行 pytest，并为每个文件设置 240 秒超时。

典型命令如下：

```bash
ssh 10.16.1.66 'docker exec dxl-sglang bash -lc "
cd /workspace/sglang/sgl-kernel
export PYTHONPATH=/workspace/sglang/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python3 -m pytest tests/test_topk.py -q
"'
```

注意：不要设置 `PYTHONPATH=/workspace/sglang/sgl-kernel/python`。否则会绕过已安装 wheel，导致 `common_ops.so` 等扩展加载路径错误，产生误判。

## 4. 总体结果

本次共验证 47 个测试文件。

| 状态 | 文件数 | 说明 |
|---|---:|---|
| PASS | 15 | pytest 返回成功，其中 12 个有实际通过用例，3 个为全 skipped |
| FAIL | 29 | 多数为 DCU/ROCm 侧算子未注册或 CUDA-only 判断未适配 |
| SKIP | 2 | pytest 直接跳过，无有效 DCU 覆盖 |
| TIMEOUT | 1 | 240 秒超时，文件内大量 case 失败/跳过 |

有效结论：

- 当前可作为 DCU/BW1000 有效覆盖的 sgl-kernel 测试文件为 12 个。
- 另有 5 个文件虽然不一定报错，但实际没有 DCU 覆盖价值，暂不建议纳入 CI。
- 失败项主要不是 pytest 注册问题，而是当前 DCU 版 `sglang-kernel` wheel 的算子接入范围不完整，或测试本身仍包含 CUDA-only 假设。

## 5. 当前可跑通的测试

以下测试在 DCU/BW1000 容器中跑通，并且存在实际 passed case。

| 测试文件 | 结果 | 耗时 | 覆盖内容 | CI 建议 |
|---|---:|---:|---|---|
| `tests/speculative/test_eagle_utils.py` | `1 passed` | 11.60s | EAGLE speculative decoding 工具逻辑 | 可进 smoke |
| `tests/test_activation.py` | `945 passed` | 46.22s | activation kernel | 可进 smoke 或 nightly |
| `tests/test_amd_deterministic_custom_allreduce.py` | `1 passed` | 35.85s | AMD/DCU allreduce 确定性 | 建议 nightly |
| `tests/test_amd_nccl_allreduce_determinism.py` | `1 passed` | 33.07s | RCCL/NCCL 风格 allreduce 确定性 | 建议 nightly |
| `tests/test_apply_token_bitmask_inplace.py` | `1 passed` | 11.35s | token bitmask 原地更新 | 可进 smoke |
| `tests/test_kvcacheio.py` | `144 passed, 48 skipped` | 147.26s | KV cache I/O | 建议 nightly |
| `tests/test_merge_state_v2.py` | `150 passed` | 17.55s | attention state merge v2 | 可进 smoke |
| `tests/test_moe_align.py` | `4368 passed, 72 skipped` | 165.96s | MoE token 对齐与 routing 辅助逻辑 | 建议 nightly |
| `tests/test_moe_topk_sigmoid.py` | `774 passed` | 12.13s | MoE top-k sigmoid | 可进 smoke |
| `tests/test_moe_topk_softmax.py` | `738 passed` | 12.79s | MoE top-k softmax | 可进 smoke |
| `tests/test_topk.py` | `112 passed` | 49.50s | top-k kernel | 可进 smoke |
| `tests/test_torch_defaults_reset.py` | `2 passed` | 0.04s | torch 默认配置恢复 | 可进 smoke |

建议第一版 DCU sgl-kernel CI 白名单：

```text
tests/speculative/test_eagle_utils.py
tests/test_activation.py
tests/test_apply_token_bitmask_inplace.py
tests/test_merge_state_v2.py
tests/test_moe_topk_sigmoid.py
tests/test_moe_topk_softmax.py
tests/test_topk.py
tests/test_torch_defaults_reset.py
```

建议放到 nightly 的扩展项：

```text
tests/test_amd_deterministic_custom_allreduce.py
tests/test_amd_nccl_allreduce_determinism.py
tests/test_kvcacheio.py
tests/test_moe_align.py
```

## 6. 跳过或无有效覆盖的测试

以下文件本次没有形成有效 DCU 覆盖。

| 测试文件 | 结果 | 原因 |
|---|---|---|
| `tests/test_cutlass_mla.py` | `1 skipped` | CUTLASS/CUDA 相关条件不满足 |
| `tests/test_hadamard.py` | `1 skipped` | 当前环境不满足测试条件 |
| `tests/test_cutlass_w4a8_moe_mm.py` | `605 skipped` | pytest 返回 PASS，但所有 case 都 skipped |
| `tests/test_es_mxfp8_blockscaled_moe.py` | `8 skipped` | pytest 返回 PASS，但所有 case 都 skipped |
| `tests/test_fused_qk_norm_rope.py` | `200 skipped` | pytest 返回 PASS，但所有 case 都 skipped |

这几项不建议直接作为 DCU CI 通过标准，因为没有验证真实 DCU kernel 路径。

## 7. 失败分类

### 7.1 DCU/ROCm wheel 未注册对应算子

这是失败最多的一类。典型报错为：

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute '<op_name>'
```

涉及文件：

| 测试文件 | 缺失算子示例 |
|---|---|
| `tests/speculative/test_ngram_utils.py` | `reconstruct_indices_from_tree_mask` |
| `tests/speculative/test_speculative_sampling.py` | `tree_speculative_sampling_target_only` |
| `tests/test_awq_dequant.py` | `awq_dequantize` |
| `tests/test_bmm_fp8.py` | `bmm_fp8` |
| `tests/test_causal_conv1d.py` | `causal_conv1d_fwd` |
| `tests/test_copy.py` | `copy_to_gpu_no_ce` |
| `tests/test_dsv3_fused_a_gemm.py` | `dsv3_fused_a_gemm` |
| `tests/test_dsv3_router_gemm.py` | `dsv3_router_gemm` |
| `tests/test_fp8_blockwise_gemm.py` | `fp8_blockwise_scaled_mm` |
| `tests/test_fp8_gemm.py` | `fp8_scaled_mm` |
| `tests/test_gptq_kernel.py` | `gptq_gemm` |
| `tests/test_int8_gemm.py` | `int8_scaled_mm` |
| `tests/test_kimi_k2_moe_fused_gate.py` | `kimi_k2_moe_fused_gate` |
| `tests/test_merge_state.py` | `merge_state` |
| `tests/test_moe_fused_gate.py` | `moe_fused_gate` |
| `tests/test_norm.py` | `rmsnorm` |
| `tests/test_per_token_quant_fp8.py` | `sgl_per_token_quant_fp8` |
| `tests/test_qserve_w4a8_per_chn_gemm.py` | `qserve_w4a8_per_chn_gemm` |
| `tests/test_qserve_w4a8_per_group_gemm.py` | `qserve_w4a8_per_group_gemm` |

结论：这不是测试注册问题，而是当前 DCU/ROCm 版 sgl-kernel wheel 尚未接入或注册这些算子。后续需要从 `common_extension_rocm.cc`、`setup_hip.py`、`setup_rocm.py` 等路径补齐 DCU kernel 编译与注册。

### 7.2 测试中存在 CUDA-only 判断

典型报错为：

```text
TypeError: '>=' not supported between instances of 'NoneType' and 'str'
```

原因是 DCU 环境下 `torch.version.cuda is None`，但测试代码直接执行了类似：

```python
torch.version.cuda >= "12.3"
```

涉及文件：

```text
tests/test_es_fp8_blockwise_moe.py
tests/test_flash_attention.py
tests/test_flash_attn_sparse.py
tests/test_flashmla.py
tests/test_fp8_blockwise_moe.py
```

结论：这类测试需要先做 HIP/DCU 条件分支适配。未适配前，不能直接加入 DCU CI。

### 7.3 空间/stream 扩展未加载

涉及文件：

```text
tests/spatial/test_greenctx_stream.py
```

关键报错：

```text
ImportError: Failed to load sgl_kernel.spatial_ops extension. Ensure CUDA Driver >= 12.4
```

结论：当前 `spatial_ops` 扩展路径仍是 CUDA 假设，DCU 环境未接入对应实现或加载逻辑。

### 7.4 通信/分布式 runtime 问题

涉及文件：

```text
tests/test_custom_allreduce.py
tests/test_mscclpp.py
```

关键现象：

```text
AssertionError: Process 0 failed with exit code 1
AssertionError: libcudart is not loaded in the current process
NotImplementedError
```

结论：这类测试依赖自定义 allreduce 或 MSCCl++ 相关 runtime，当前 DCU 环境未完全适配。建议暂不作为通用 smoke，用单独专项跟踪。

### 7.5 外部依赖或离线数据缺失

涉及文件：

```text
tests/test_sampling.py
tests/test_gguf.py
```

原因：

- `tests/test_sampling.py` 缺少 `flashinfer`。
- `tests/test_gguf.py` 需要 Hugging Face 样例仓库 `Isotr0py/test-gguf-sample`，当前离线模式下本地无缓存。

结论：这类不是 DCU kernel 直接失败。需要补依赖或预置测试数据后再判断是否可纳入。

### 7.6 超时项

涉及文件：

```text
tests/test_per_token_group_quant_8bit.py
```

现象：

- 240 秒超时。
- 日志中持续出现大量 `F` 和 `s`，说明该文件内大量 case 在失败/跳过之间推进。

结论：应先单独抽取小规模 case 定位失败原因，不建议直接纳入 CI。

## 8. 结论

当前 `sgl-kernel/tests` 在 DCU/BW1000 上不是整体可直接复用的状态，需要分层处理。

可以先接入 DCU CI 的部分：

- activation、top-k、MoE top-k、merge_state_v2、token bitmask 等基础 kernel。
- KV cache I/O、MoE align、allreduce determinism 可以放 nightly。

暂不建议接入的部分：

- 大量 FP8/INT8/GPTQ/QServe/Mamba/MoE fused gate/norm 测试，因为 DCU wheel 中相关算子还没有注册。
- FlashAttention/FlashMLA/ES FP8 MoE 相关测试，因为测试代码仍有 CUDA-only 判断。
- GGUF 和 sampling，因为依赖或离线数据不完整。
- MSCCl++、custom allreduce 旧测试路径，因为当前 runtime 假设和 DCU 环境不完全匹配。

整体判断：

```text
当前不是“测试写错了”或“CI 没注册”的问题为主，
而是 DCU 版 sgl-kernel 当前只接入了部分官方 kernel 能力。
CI 建设应先纳入已验证通过的白名单，再把失败项拆成 DCU kernel 接入 backlog。
```

## 9. 后续建议

### 9.1 第一阶段：建立 DCU sgl-kernel smoke 白名单

建议先固定以下文件：

```text
tests/speculative/test_eagle_utils.py
tests/test_activation.py
tests/test_apply_token_bitmask_inplace.py
tests/test_merge_state_v2.py
tests/test_moe_topk_sigmoid.py
tests/test_moe_topk_softmax.py
tests/test_topk.py
tests/test_torch_defaults_reset.py
```

目标是保证 DCU 基础 kernel 能力在每次 CI 中不回退。

### 9.2 第二阶段：nightly 扩展覆盖

建议 nightly 加入：

```text
tests/test_amd_deterministic_custom_allreduce.py
tests/test_amd_nccl_allreduce_determinism.py
tests/test_kvcacheio.py
tests/test_moe_align.py
```

这些文件耗时更长或依赖通信语义，更适合 nightly。

### 9.3 第三阶段：失败项 backlog

建议将失败项按以下方向拆解：

| 方向 | 代表文件 | 处理方式 |
|---|---|---|
| ROCm/DCU op 注册补齐 | `test_norm.py`, `test_merge_state.py`, `test_moe_fused_gate.py` | 补 `common_extension_rocm.cc` 与构建脚本 |
| 量化 kernel 接入 | `test_fp8_gemm.py`, `test_int8_gemm.py`, `test_gptq_kernel.py`, `test_qserve_*` | 确认 DCU kernel 是否已有实现，再补注册和测试 |
| CUDA-only 测试适配 | `test_flash_attention.py`, `test_flashmla.py` | 加 HIP/DCU 条件分支，避免直接比较 `torch.version.cuda` |
| 外部依赖补齐 | `test_sampling.py`, `test_gguf.py` | 安装依赖或预置离线样例数据 |
| 通信专项 | `test_custom_allreduce.py`, `test_mscclpp.py` | 单独验证 DCU runtime 和通信后端 |

## 10. 附：本次完整文件级结果

| 测试文件 | 状态 | 耗时 | 结果摘要 |
|---|---|---:|---|
| `tests/spatial/test_greenctx_stream.py` | FAIL | 21s | `1 failed in 13.93s` |
| `tests/speculative/test_eagle_utils.py` | PASS | 18s | `1 passed in 11.60s` |
| `tests/speculative/test_ngram_utils.py` | FAIL | 17s | `1 failed in 11.51s` |
| `tests/speculative/test_speculative_sampling.py` | FAIL | 17s | `2 failed in 11.38s` |
| `tests/test_activation.py` | PASS | 53s | `945 passed in 46.22s` |
| `tests/test_amd_deterministic_custom_allreduce.py` | PASS | 40s | `1 passed in 35.85s` |
| `tests/test_amd_nccl_allreduce_determinism.py` | PASS | 39s | `1 passed in 33.07s` |
| `tests/test_apply_token_bitmask_inplace.py` | PASS | 18s | `1 passed in 11.35s` |
| `tests/test_awq_dequant.py` | FAIL | 21s | `112 failed in 15.30s` |
| `tests/test_bmm_fp8.py` | FAIL | 22s | `6 failed, 2 skipped in 15.15s` |
| `tests/test_causal_conv1d.py` | FAIL | 50s | `760 failed in 44.57s` |
| `tests/test_copy.py` | FAIL | 18s | `2 failed in 11.63s` |
| `tests/test_custom_allreduce.py` | FAIL | 43s | `1 failed in 35.97s` |
| `tests/test_cutlass_mla.py` | SKIP | 18s | `1 skipped in 11.24s` |
| `tests/test_cutlass_w4a8_moe_mm.py` | PASS | 18s | `605 skipped in 12.22s` |
| `tests/test_dsv3_fused_a_gemm.py` | FAIL | 21s | `16 failed in 15.41s` |
| `tests/test_dsv3_router_gemm.py` | FAIL | 22s | `32 failed in 15.23s` |
| `tests/test_es_fp8_blockwise_moe.py` | FAIL | 16s | `1 error in 10.69s` |
| `tests/test_es_mxfp8_blockscaled_moe.py` | PASS | 15s | `8 skipped in 11.25s` |
| `tests/test_flash_attention.py` | FAIL | 6s | `1 error in 0.13s` |
| `tests/test_flash_attn_sparse.py` | FAIL | 16s | `1 error in 10.74s` |
| `tests/test_flashmla.py` | FAIL | 17s | `1 error in 11.59s` |
| `tests/test_fp8_blockwise_gemm.py` | FAIL | 45s | `576 failed in 38.13s` |
| `tests/test_fp8_blockwise_moe.py` | FAIL | 17s | `1 error in 11.47s` |
| `tests/test_fp8_gemm.py` | FAIL | 40s | `500 failed in 33.79s` |
| `tests/test_fused_qk_norm_rope.py` | PASS | 34s | `200 skipped, 6 warnings in 17.57s` |
| `tests/test_gguf.py` | FAIL | 19s | `2 warnings, 1 error in 12.47s` |
| `tests/test_gptq_kernel.py` | FAIL | 47s | `12 failed, 6 warnings in 25.21s` |
| `tests/test_hadamard.py` | SKIP | 17s | `1 skipped in 11.60s` |
| `tests/test_int8_gemm.py` | FAIL | 66s | `1260 failed in 60.12s` |
| `tests/test_kimi_k2_moe_fused_gate.py` | FAIL | 42s | `46 failed, 6 warnings in 22.37s` |
| `tests/test_kvcacheio.py` | PASS | 154s | `144 passed, 48 skipped in 147.26s` |
| `tests/test_merge_state.py` | FAIL | 18s | `1 failed in 12.24s` |
| `tests/test_merge_state_v2.py` | PASS | 24s | `150 passed in 17.55s` |
| `tests/test_moe_align.py` | PASS | 173s | `4368 passed, 72 skipped in 165.96s` |
| `tests/test_moe_fused_gate.py` | FAIL | 31s | `396 failed in 25.18s` |
| `tests/test_moe_topk_sigmoid.py` | PASS | 18s | `774 passed in 12.13s` |
| `tests/test_moe_topk_softmax.py` | PASS | 19s | `738 passed in 12.79s` |
| `tests/test_mscclpp.py` | FAIL | 34s | `1 failed in 28.77s` |
| `tests/test_norm.py` | FAIL | 27s | `224 failed in 20.99s` |
| `tests/test_per_token_group_quant_8bit.py` | TIMEOUT | 240s | `timeout` |
| `tests/test_per_token_quant_fp8.py` | FAIL | 17s | `15 failed in 11.94s` |
| `tests/test_qserve_w4a8_per_chn_gemm.py` | FAIL | 28s | `270 failed in 22.07s` |
| `tests/test_qserve_w4a8_per_group_gemm.py` | FAIL | 27s | `270 failed in 22.14s` |
| `tests/test_sampling.py` | FAIL | 5s | `1 error in 0.12s` |
| `tests/test_topk.py` | PASS | 55s | `112 passed in 49.50s` |
| `tests/test_torch_defaults_reset.py` | PASS | 6s | `2 passed in 0.04s` |

