# DCU 剩余测试建设方案

## 阶段目标

当前第三阶段聚焦 DCU/BW1000 精度测试体系。目标不是只增加三个 pytest 文件，而是形成可维护的 nightly 门禁：

- 本地模型和本地数据优先。
- 样本数、阈值、server 参数可通过环境变量覆盖。
- smoke 和 nightly 阈值分开管理。
- GitHub workflow 可单独运行文本精度、VLM 精度和性能 suite。

## 已完成项

- 新增 `test/registered/dcu/accuracy/README.md`。
- 新增 `test/registered/dcu/accuracy/bw1000/test_gsm8k_eval_dcu.py`。
- 新增 `test/registered/dcu/accuracy/bw1000/test_mmlu_eval_dcu.py`。
- 新增 `test/registered/dcu/accuracy/bw1000/test_mmmu_eval_dcu.py`。
- 新增 MMLU 本地 parquet root 支持。
- 保留 MMMU 本地 parquet 支持。
- 拆分 `SGLANG_DCU_GSM8K_NUM_EXAMPLES` 和 `SGLANG_DCU_MMLU_NUM_EXAMPLES`，保留 `SGLANG_DCU_EVAL_NUM_EXAMPLES` 作为兼容 fallback。
- `.github/workflows/nightly-test-dcu.yml` 增加 `nightly-dcu-vlm` 输入选项和独立 job。

## 待建设项

1. 文本精度 nightly 基线

   - GSM8K 固定 1319 样本或全量连续跑三次。
   - MMLU 固定 5000 样本连续跑三次。
   - 根据三次最小值下浮 1 到 2 个百分点固化 nightly 阈值。

2. 多模态精度 nightly 基线

   - MMMU 固定 100 样本连续跑三次。
   - 根据三次最小值下浮 2 到 3 个百分点固化 nightly 阈值。
   - 记录 latency，稳定后按最大值的 1.5 倍设置保守 latency 门禁。

3. CI 环境收敛

   - 确认官方 runner 的 BW1000 模型挂载路径与 `DCU_MODEL_HOST_PATH` 一致。
   - 确认 `lm-eval`、simple-evals 依赖和 parquet 读取依赖在 DCU CI 镜像中预置。
   - 如 CI infra 提供 BW1000 专用 runner label，再把 workflow 的 runner 名称从旧 label 迁移过去。

4. 后续扩展

   - 增加更多 BW1000 文本模型和 VLM 模型矩阵。
   - 按硬件新增 `accuracy/bw1100/` 等目录。
   - 将稳定的长耗时 accuracy suite 与 per-commit smoke 明确分层。

## 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| 模型路径不可用 | CI 隐式下载或直接失败 | 本地路径缺失时 fail fast；VLM 无默认模型时 skip |
| MMLU 远程 CSV 不稳定 | nightly 偶发失败 | 优先使用本地 parquet root |
| MMMU 图片数据或 VLM 模型过大 | nightly 超时 | 先 100 样本并单独放入 `nightly-dcu-vlm` |
| smoke 阈值复用到 nightly | 误判精度 | nightly 必须单独三次基线 |
| DCU runtime VMFault | 精度链路无法启动 | 先固定已跑通模型和 fa3/page_size 参数，后端问题单独归因 |
