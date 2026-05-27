# DCU/BW1000 从基础镜像到测试全流程

本文记录当前 `10.16.1.66` 节点上，从基础 SGLang DCU 镜像启动容器，到运行 SGLang DCU accuracy 测试和 `sgl-kernel/tests` 算子测试的完整复现流程。

## 1. 当前基准环境

验证节点：

```text
10.16.1.66
```

容器名：

```text
dxl-sglang
```

基础镜像：

```text
10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260510-2235
```

镜像 ID：

```text
sha256:34e3ef2e9542d581e48d1f498d3e320f583e6789a5e09ab71d19bb89cf047692
```

容器内关键软件版本：

```text
torch          2.9.0+das.opt1.dtk2604.2605091832.g21362a
sglang         0.5.10rc0+das.opt2.alpha.dtk2604.torch290.2605081216.g9f67de
sglang-kernel 0.4.0+das.opt1.dtk2604.torch290.2605081216.g9f67de
aiter          0.1.2+das.opt1.dtk2604.torch290.2605071840.g1f8f50
datasets       4.8.5
pandas         2.3.3
pyarrow        24.0.0
pytest         9.0.3
HIP            6.3.26113
```

DCU 信息：

```text
device_count: 8
device0: BW200, UBB BW1000
arch: gfx936:sramecc+:xnack-
```

## 2. 启动基础容器

当前使用的启动脚本在用户目录：

```text
/public/home/dingxl/dockerrun.sh
```

核心内容如下：

```bash
IMAGE_NAME="34e3ef2e9542"
CONTAINER_NAME="dxl-sglang"

docker run -dit \
    --name "$CONTAINER_NAME" \
    --privileged \
    --network=host \
    --ipc=host \
    --shm-size=200G \
    --cap-add=SYS_PTRACE \
    --device=/dev/dri:/dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /public/opendas/DL_DATA/ImageNet-pytorch:/mnt/resnet50/dataset:ro \
    -v /public/opendas/DL_DATA/llm-models:/public/opendas/DL_DATA/llm-models:ro \
    -v /public/home/dingxl:/workspace \
    --security-opt seccomp=unconfined \
    "$IMAGE_NAME" \
    /bin/bash
```

启动：

```bash
ssh 10.16.1.66 'bash /public/home/dingxl/dockerrun.sh'
```

进入容器：

```bash
ssh 10.16.1.66 'docker exec -it dxl-sglang bash'
```

容器内仓库路径：

```text
/workspace/sglang
```

模型和数据路径：

```text
/public/opendas/DL_DATA/llm-models
```

## 3. 环境自检

确认 DCU 可见：

```bash
ssh 10.16.1.66 'docker exec dxl-sglang hy-smi'
```

确认 PyTorch/HIP：

```bash
ssh 10.16.1.66 'docker exec dxl-sglang python3 -c "
import torch
print(\"torch\", torch.__version__)
print(\"hip\", torch.version.hip)
print(\"cuda_available\", torch.cuda.is_available())
print(\"device_count\", torch.cuda.device_count())
print(\"device0\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print(\"arch\", getattr(torch.cuda.get_device_properties(0), \"gcnArchName\", None) if torch.cuda.is_available() else None)
"'
```

确认 SGLang 和 kernel 包：

```bash
ssh 10.16.1.66 'docker exec dxl-sglang python3 -m pip show torch sglang sglang-kernel aiter datasets pandas pyarrow pytest | egrep "^(Name|Version|Location):"'
```

确认 `sgl_kernel` 安装版 wheel 可用：

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/sgl-kernel dxl-sglang bash -lc "
PYTHONPATH=/workspace/sglang/python python3 -c '\''import sgl_kernel; print(sgl_kernel.__file__); from sgl_kernel import silu_and_mul, topk_softmax; print(\"ops import ok\")'\''
"'
```

注意：跑 `sgl-kernel/tests` 时不要把下面路径放进 `PYTHONPATH`：

```text
/workspace/sglang/sgl-kernel/python
```

否则 Python 会优先使用源码目录里的 `sgl_kernel` 包，但那里没有编译好的 `common_ops.so`，会导致大量：

```text
ModuleNotFoundError: No module named 'common_ops'
```

正确做法是使用镜像里安装好的：

```text
/usr/local/lib/python3.10/dist-packages/sgl_kernel
```

## 4. 当前仓库侧改动要求

当前测试不是只依赖基础镜像，还依赖 `/workspace/sglang` 当前代码。

与 DCU accuracy 相关的关键改动包括：

- `python/sglang/srt/server_args.py`
  - DCU 默认 attention backend 走 `fa3`。
  - DCU + FA3 + 非 MLA 场景下默认 `page_size=64`。
- `python/sglang/test/run_eval.py`
  - MMLU/MMMU 支持传入本地 `dataset_path`。
- `python/sglang/test/simple_eval_mmlu.py`
  - 支持本地 MMLU parquet root。
  - 按官方 MMLU subject 顺序读取，避免额外目录导致抽样口径偏移。
- `python/sglang/test/simple_eval_mmmu_vlm.py`
  - 支持本地 MMMU parquet 数据路径。
- `test/registered/dcu/accuracy/bw1000/`
  - 新增 GSM8K、MMLU、MMMU 三个 DCU/BW1000 精度测试。

## 5. DCU accuracy 测试

### 5.1 模型和数据路径

默认文本模型：

```text
/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct
```

默认 VLM 模型：

```text
/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-3B-Instruct
```

MMLU 本地数据：

```text
/public/opendas/DL_DATA/llm-models/datasets/mmlu
```

MMMU 本地数据：

```text
/public/opendas/DL_DATA/llm-models/multimodal-datasets/MMMU
```

### 5.2 server 参数

文本模型默认参数：

```text
--attention-backend fa3
--page-size 64
--log-level warning
--log-level-http warning
--trust-remote-code
```

VLM 默认额外参数：

```text
--mm-attention-backend fa3
--enable-multimodal
```

### 5.3 单文件 smoke

GSM8K：

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/test \
  -e PYTHONPATH=/workspace/sglang/python \
  dxl-sglang \
  python3 -m pytest -q registered/dcu/accuracy/bw1000/test_gsm8k_eval_dcu.py'
```

MMLU：

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/test \
  -e PYTHONPATH=/workspace/sglang/python \
  dxl-sglang \
  python3 -m pytest -q registered/dcu/accuracy/bw1000/test_mmlu_eval_dcu.py'
```

MMMU：

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/test \
  -e PYTHONPATH=/workspace/sglang/python \
  dxl-sglang \
  python3 -m pytest -q registered/dcu/accuracy/bw1000/test_mmmu_eval_dcu.py'
```

### 5.4 三文件组合 smoke

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/test \
  -e PYTHONPATH=/workspace/sglang/python \
  dxl-sglang \
  python3 -m pytest \
    registered/dcu/accuracy/bw1000/test_gsm8k_eval_dcu.py \
    registered/dcu/accuracy/bw1000/test_mmlu_eval_dcu.py \
    registered/dcu/accuracy/bw1000/test_mmmu_eval_dcu.py \
    -q'
```

当前已验证通过：

```text
3 passed, 2 warnings in 282.80s
```

当前 smoke 阈值：

```text
GSM8K 10 samples >= 0.65
MMLU  50 samples >= 0.68
MMMU  10 samples >= 0.35
```

nightly 阈值还需要单独跑大样本三次基线后固化。

## 6. DCU registration 检查

容器内执行：

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang dxl-sglang python3 scripts/ci/dcu/verify_dcu_registration.py'
```

当前已验证结果：

```text
OK: DCU registration looks healthy.
Collected 40 DCU registered test file(s)
```

## 7. sgl-kernel/tests 测试

### 7.1 测试定位

`sgl-kernel/tests` 不是模型测试，不需要 Qwen/Llama 模型。

它主要测试底层算子：

- activation
- topk
- MoE align/topk/gate
- norm
- GEMM/quant
- attention merge state
- speculative decoding 小算子
- kvcache transfer
- 多卡 allreduce

这些测试一般自己构造 tensor：

```python
torch.randn(..., device="cuda")
torch.empty(..., device="cuda")
torch.randint(..., device="cuda")
```

在 DCU/DTK PyTorch 里仍然使用 `torch.cuda` API。

### 7.2 单文件运行方式

注意：只设置 `/workspace/sglang/python`，不要设置 `/workspace/sglang/sgl-kernel/python`。

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/sgl-kernel \
  -e PYTHONPATH=/workspace/sglang/python \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  dxl-sglang \
  python3 -m pytest -q tests/test_topk.py --tb=short --disable-warnings'
```

### 7.3 文件级批量验证脚本

下面命令会逐文件运行 `sgl-kernel/tests`，每个文件超时 240 秒，并输出 summary。

```bash
ssh 10.16.1.66 "docker exec -i -w /workspace/sglang/sgl-kernel dxl-sglang bash -s" <<'BASH'
set +e
LOGDIR=/tmp/sgl_kernel_dcu_tests_installed_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
export PYTHONPATH=/workspace/sglang/python:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
SUMMARY="$LOGDIR/summary.tsv"
printf "file\tstatus\trc\tduration_s\tlast_line\n" > "$SUMMARY"

for f in $(find tests -type f -name "test_*.py" | sort); do
  safe=$(echo "$f" | tr "/" "_")
  log="$LOGDIR/${safe}.log"
  echo "===RUN $f==="
  start=$(date +%s)
  timeout --kill-after=15s 240s python3 -m pytest -q "$f" --tb=short --disable-warnings > "$log" 2>&1
  rc=$?
  end=$(date +%s)
  dur=$((end-start))
  if [ "$rc" -eq 0 ]; then
    status=PASS
  elif [ "$rc" -eq 5 ]; then
    status=SKIP
  elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    status=TIMEOUT
  else
    status=FAIL
  fi
  last=$(grep -E "passed|failed|skipped|error|ERROR|FAILED|SKIPPED|no tests ran|ImportError|ModuleNotFoundError|AttributeError|RuntimeError|AssertionError|Segmentation fault|core dumped|Timeout" "$log" | tail -n 1 | tr "\t" " " | cut -c1-220)
  printf "%s\t%s\t%s\t%s\t%s\n" "$f" "$status" "$rc" "$dur" "$last" >> "$SUMMARY"
  echo "===RESULT $status rc=$rc dur=${dur}s $f==="
done

echo "===SUMMARY_FILE $SUMMARY==="
cat "$SUMMARY"
BASH
```

当前汇总文件已复制到：

```text
/public/home/dingxl/sgl_kernel_dcu_tests_installed_20260522_100156_summary.tsv
```

远端容器原始日志：

```text
/tmp/sgl_kernel_dcu_tests_installed_20260522_100156/
```

## 8. sgl-kernel 当前验证结论

当前有效验证结果：

```text
47 个测试文件
14 个 PASS
2 个 SKIP
1 个 TIMEOUT
30 个 FAIL
```

适合先纳入 DCU kernel CI 的白名单：

```text
tests/speculative/test_eagle_utils.py
tests/test_activation.py
tests/test_amd_deterministic_custom_allreduce.py
tests/test_amd_nccl_allreduce_determinism.py
tests/test_apply_token_bitmask_inplace.py
tests/test_merge_state_v2.py
tests/test_moe_topk_sigmoid.py
tests/test_moe_topk_softmax.py
tests/test_topk.py
tests/test_torch_defaults_reset.py
```

可用但较重，建议放 nightly：

```text
tests/test_kvcacheio.py
tests/test_moe_align.py
```

### 8.1 失败原因分类

第一类：DCU wheel 未注册对应 op。

典型报错：

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'xxx'
```

这类不是 pytest 注册问题，而是当前镜像里的 DCU 版 `sglang-kernel` wheel 没把对应 op 编入/注册到 `torch.ops.sgl_kernel`。

代表文件：

```text
tests/test_awq_dequant.py
tests/test_bmm_fp8.py
tests/test_causal_conv1d.py
tests/test_copy.py
tests/test_dsv3_fused_a_gemm.py
tests/test_dsv3_router_gemm.py
tests/test_fp8_blockwise_gemm.py
tests/test_fp8_gemm.py
tests/test_gptq_kernel.py
tests/test_int8_gemm.py
tests/test_kimi_k2_moe_fused_gate.py
tests/test_merge_state.py
tests/test_moe_fused_gate.py
tests/test_norm.py
tests/test_per_token_quant_fp8.py
tests/test_qserve_w4a8_per_chn_gemm.py
tests/test_qserve_w4a8_per_group_gemm.py
```

第二类：CUDA-only 判断没兼容 HIP。

典型报错：

```text
TypeError: '>=' not supported between instances of 'NoneType' and 'str'
```

原因是测试代码直接判断：

```python
torch.version.cuda >= "12.3"
```

DCU/ROCm 下 `torch.version.cuda` 是 `None`。

代表文件：

```text
tests/test_flash_attention.py
tests/test_flash_attn_sparse.py
tests/test_flashmla.py
tests/test_es_fp8_blockwise_moe.py
tests/test_fp8_blockwise_moe.py
```

第三类：依赖或数据缺失。

```text
tests/test_sampling.py
```

缺 `flashinfer`。

```text
tests/test_gguf.py
```

离线模式下没有 HuggingFace sample cache。

第四类：多卡通信/runtime 失败。

```text
tests/test_custom_allreduce.py
tests/test_mscclpp.py
```

这类不是单纯 op 注册问题，需要单独查多卡通信栈、进程启动方式、RCCL/NCCL 兼容层。

第五类：超时且已有大量失败。

```text
tests/test_per_token_group_quant_8bit.py
```

240 秒未结束，且中间已经出现大量失败，不建议直接进 CI。

## 9. 清理测试产物

本次验证后需要清理 pytest 缓存：

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/sgl-kernel dxl-sglang rm -rf \
  .pytest_cache \
  tests/__pycache__ \
  tests/spatial/__pycache__ \
  tests/speculative/__pycache__ \
  python/sgl_kernel/__pycache__'
```

确认 DCU 没有残留占用：

```bash
ssh 10.16.1.66 'docker exec dxl-sglang hy-smi'
```

期望看到 8 张卡 `VRAM%` 和 `HCU%` 都回到 0。

## 10. 可复现性结论

当前测试可以复现，但需要同时满足：

```text
同类 BW1000 DCU 节点
同一个基础镜像 34e3ef2e9542
同样的 docker run 挂载
当前 /workspace/sglang 代码
/public/opendas/DL_DATA/llm-models 模型和数据存在
跑 sgl-kernel 时使用安装版 sglang-kernel wheel
```

如果只拿裸基础镜像，不挂当前 repo、不挂模型数据、不带这些环境变量和代码改动，则不能完整复现当前 accuracy 和 sgl-kernel 验证结果。
