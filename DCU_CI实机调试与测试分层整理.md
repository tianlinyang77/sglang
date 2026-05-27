# DCU CI 实机调试与测试分层整理

## 当前实机约定

- 验证节点：`10.16.1.66`
- 主要容器：`dxl-sglang`
- 容器内仓库：`/workspace/sglang`
- 容器内测试工作目录：`/workspace/sglang/test`
- DCU 监控命令：`hy-smi`

用户本地调试时可直接使用：

```bash
ssh 10.16.1.66 'docker exec -w /workspace/sglang/test -e PYTHONPATH=/workspace/sglang/python dxl-sglang python3 -m pytest registered/dcu/accuracy/bw1000/test_gsm8k_eval_dcu.py -q'
```

## 测试分层

DCU registered tests 当前按能力拆分：

- `interface/`：CI wrapper、server 启动、health-check、基础 API。
- `accuracy/bw1000/`：GSM8K、MMLU、MMMU 精度评测。
- `perf/k100/`：已有性能用例目录，后续如迁移硬件命名再单独调整。
- `disaggregation/`：PD-disaggregation 专项。
- `basic_function/`：后端、量化、并行策略、runtime 选项。
- `llm_models/`：DCU LLM 模型兼容性。
- `vlm_models/`：DCU VLM 模型兼容性。

## 第三阶段精度进展

已完成：

- 建立 `test/registered/dcu/accuracy/bw1000/`。
- GSM8K、MMLU、MMMU 三个测试均复用 `sglang.test.run_eval`。
- 文本测试默认使用 `fa3` attention 和 `page_size=64`。
- VLM 测试默认使用 `fa3` text/mm attention、`page_size=64` 和 `--enable-multimodal`。
- MMLU 支持本地 parquet root，读取 `*/test-*.parquet`。
- MMMU 支持本地 parquet 数据路径。
- GSM8K/MMLU 注册到 `nightly-dcu-accuracy`。
- MMMU 注册到 `nightly-dcu-vlm`。
- workflow 增加 `nightly-dcu-vlm` 独立 job。

待完成：

- GSM8K nightly 固定样本连续三次基线。
- MMLU 5000 样本连续三次基线。
- MMMU 100 样本连续三次基线。
- 根据 nightly 基线替换当前 smoke 阈值。
- 观察 MMMU latency 是否稳定，再决定是否打开严格 latency 门禁。

## 验证命令

静态检查：

```bash
python3 -m py_compile \
  test/registered/dcu/accuracy/bw1000/test_gsm8k_eval_dcu.py \
  test/registered/dcu/accuracy/bw1000/test_mmlu_eval_dcu.py \
  test/registered/dcu/accuracy/bw1000/test_mmmu_eval_dcu.py \
  python/sglang/test/run_eval.py \
  python/sglang/test/simple_eval_mmlu.py \
  python/sglang/test/simple_eval_mmmu_vlm.py
```

注册检查：

```bash
python3 scripts/ci/dcu/verify_dcu_registration.py
```

组合 smoke：

```bash
cd test
PYTHONPATH=../python python3 -m pytest \
  registered/dcu/accuracy/bw1000/test_gsm8k_eval_dcu.py \
  registered/dcu/accuracy/bw1000/test_mmlu_eval_dcu.py \
  registered/dcu/accuracy/bw1000/test_mmmu_eval_dcu.py \
  -q
```

## 调试注意

- `hy-smi` 看到 0 占用不一定说明没走 DCU；GSM8K/MMLU 小样本 decode 窗口很短，采样间隔可能错过峰值。
- 判断是否走 DCU 更可靠的信息来自 server 启动日志、SGLang backend 选择和 decode throughput。
- cookbook 中的大模型 speculative/EAGLE 参数不要直接套进 CI 精度 smoke；当前只吸收稳定的 DCU 启动参数。
