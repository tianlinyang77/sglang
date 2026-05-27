# DCU sglang-tly 合入内容与验证计划

## 1. 背景

本次合入目标是以当前 `sglang/` 为主线，吸收同事仓库 `sglang-tly/` 中已有的 DCU 注册覆盖、CI workflow、覆盖矩阵脚本和文档沉淀。

当前主线按 BW1000 实机口径处理，默认验证节点为 `10.16.1.66`，容器为 `dxl-sglang`。`sglang-tly` 中历史使用的 `stage-a-dcu`、`stage-b-dcu`、`k100/` 命名没有作为当前主线继续使用，而是统一迁移到当前 BW1000 suite 和目录体系。

需要特别说明：本次完成的是注册体系和 CI 基础设施合入，不代表所有新增注册用例都已在 BW1000 上实测通过。未复测或已知失败的用例仍保持 disabled，并保留原因。

## 2. 本次合入内容

| 模块 | 修改项 | 说明 |
|---|---|---|
| DCU 注册覆盖 | 迁入 `sglang-tly` 的 broad registration | 当前 `verify_dcu_registration.py` 可收集 203 个 DCU registered test file |
| Suite 命名 | 旧 suite 映射到当前 suite | `stage-a-dcu`、`stage-b-dcu` 不再作为实际 suite 使用 |
| DCU 专项目录 | 保留当前 BW1000 主线 | 不引入 `test/registered/dcu/*/k100` 作为主线目录 |
| SRT smoke | 新增 Qwen2.5-0.5B server smoke | 迁移自 tly 的 server smoke，落到 `srt/bw1000/` |
| CI 执行脚本 | 增强容器名兼容 | `dcu_ci_exec.sh` 支持 `--container-name` 和 `DCU_CI_CONTAINER_NAME` |
| CI 启动脚本 | 增强镜像和容器参数 | `dcu_ci_start_container.sh` 支持 `--image`、`--container-name` |
| 依赖安装脚本 | 增强容器名兼容 | `dcu_ci_install_dependency.sh` 支持 `DCU_CI_CONTAINER_NAME` |
| 覆盖分析 | 新增 `analyze_dcu_csv_coverage.py` | 用于生成 CSV 覆盖矩阵和 AMD/NVIDIA 交集覆盖矩阵 |
| PR workflow | 吸收 tly 的结构化调度 | 支持路径过滤、手动触发、runner/image 参数、PR gate、分区运行 |
| Nightly workflow | 保留当前分层 | 继续使用 `nightly-dcu-accuracy`、`nightly-dcu-vlm`、`nightly-dcu-perf` |
| 文档 | 迁入 tly 核心文档和矩阵 | 补充当前 BW1000/10.16.1.66/dxl-sglang 合入口径说明 |

## 3. 关键状态

| 项 | 当前结果 |
|---|---:|
| DCU registered test file | 203 |
| `stage-b-test-1-gpu-small-dcu` enabled | 53 |
| `stage-b-test-1-gpu-small-dcu` disabled/skipped | 81 |
| CSV enabled | 57 |
| CSV registered-disabled | 95 |
| AMD/NVIDIA 交集 enabled | 75 |
| AMD/NVIDIA 交集 disabled | 61 |

已知保留策略：

- 已 disabled 的测试继续 disabled。
- disabled 原因保留，用于后续专项复测。
- `DCU_CSV_CI_UNVERIFIED`、`DCU_CSV_COVERED_UNVERIFIED` 这类未复测标记继续保留。
- 注册覆盖不等同于实机通过。

## 4. 已完成验证

已在 `10.16.1.66` 的 `dxl-sglang` 容器中完成以下验证。

### 4.1 DCU 注册收集

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang python3 scripts/ci/dcu/verify_dcu_registration.py'
```

结果：

```text
Collected 203 DCU registered test file(s)
OK: DCU registration looks healthy.
```

### 4.2 Stage-B suite 枚举

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --list'
```

结果：

```text
Enabled 53 test(s)
Skipped 81 test(s)
```

### 4.3 覆盖矩阵脚本

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang python3 scripts/ci/dcu/analyze_dcu_csv_coverage.py'
```

结果：

```text
CSV Coverage Matrix
  csv_rows: 161
  unique_casefiles: 159
  enabled: 57
  manual_dcu: 1
  missing_or_obsolete: 8
  registered_disabled: 95

AMD/NVIDIA Intersection Matrix
  intersection: 136
  disabled: 61
  enabled: 75
  csv_overlap: 97
  not_in_csv: 39
```

### 4.4 静态检查

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang python3 -m py_compile test/run_suite.py scripts/ci/dcu/verify_dcu_registration.py scripts/ci/dcu/analyze_dcu_csv_coverage.py test/registered/dcu/srt/bw1000/test_qwen25_0p5b_server_dcu.py'
```

结果：退出码为 0。

## 5. 下一步测试计划

### Phase 1：静态和注册验证

目标：确认合入后 CI 元信息没有问题，不启动模型。

建议命令：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang python3 scripts/ci/dcu/verify_dcu_registration.py'
```

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite stage-a-test-1-gpu-small-dcu --list'
```

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --list'
```

验收标准：

- `verify_dcu_registration.py` 通过。
- `run_suite --list` 能列出 stage-a/stage-b 测试。
- 不出现 `stage-a-dcu`、`stage-b-dcu` 作为实际 suite。

### Phase 2：DCU 专项 smoke 回归

目标：验证当前 BW1000 专项目录没有因合入回退。

建议优先跑这些文件：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 -m pytest registered/dcu/interface/test_dcu_smoke.py -v'
```

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 -m pytest registered/dcu/srt/bw1000/test_qwen25_0p5b_server_dcu.py -v'
```

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 -m pytest registered/dcu/srt/bw1000/test_qwen25_1p5b_server_dcu.py -v'
```

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 -m pytest registered/dcu/kernels/test_sgl_kernel_supported_dcu.py -v'
```

验收标准：

- interface smoke 通过。
- Qwen2.5 dense smoke 至少 0.5B/1.5B 通过。
- kernel whitelist 通过。
- 如果 7B、VLM、MoE、embedding、reranker 因卡占用或环境失败，需要单独记录失败原因，不直接判定合入失败。

### Phase 3：Stage-A / Stage-B 分区运行

目标：验证当前 PR suite 可以作为 CI 门禁运行。

建议先跑 Stage-A：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite stage-a-test-1-gpu-small-dcu --timeout-per-file 1800'
```

Stage-B 建议分区跑，避免一次占用过久：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --auto-partition-id 0 --auto-partition-size 2 --timeout-per-file 1800 --continue-on-error'
```

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --auto-partition-id 1 --auto-partition-size 2 --timeout-per-file 1800 --continue-on-error'
```

验收标准：

- Stage-A 通过。
- Stage-B enabled 用例尽量通过。
- 失败项需要分类：环境问题、模型路径问题、DCU 后端问题、测试本身需要 disabled。

### Phase 4：Nightly 分层验证

目标：验证 nightly workflow 的三个方向可独立运行。

精度：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite nightly-dcu-accuracy --nightly --timeout-per-file 3600 --continue-on-error'
```

VLM：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite nightly-dcu-vlm --nightly --timeout-per-file 7200 --continue-on-error'
```

性能：

```bash
ssh 10.16.1.66 'cd /public/home/dingxl/sglang && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name dxl-sglang -w /workspace/sglang/test python3 run_suite.py --hw dcu --suite nightly-dcu-perf --nightly --timeout-per-file 3600 --continue-on-error'
```

验收标准：

- 三类 nightly suite 可以独立启动。
- accuracy/VLM/perf 失败时输出清晰的模型路径、数据集路径、阈值或性能原因。
- 不把未建立 BW1000 基线的性能/精度项直接设为强门禁。

### Phase 5：disabled 用例复测与打开

目标：逐步把 registered-disabled 转成 enabled。

建议顺序：

1. 先复测轻量 unit/parser/utils/openai_server mock 类。
2. 再复测 scheduler、sampling、core server 类。
3. 再复测 VLM、LoRA、HiCache、MoE、quant、spec、distributed。
4. 最后处理 perf 和 accuracy 阈值固化。

每打开一个 disabled 用例都需要满足：

- BW1000 实机至少跑通一次。
- 如果是门禁用例，建议连续 3 次稳定。
- 文档里记录模型路径、启动参数、失败/通过结果。
- 删除 disabled 前保留对应验证证据。

## 6. 当前建议结论

本次合入可以作为 DCU registered 覆盖建设的阶段性完成点。下一步重点不是继续补注册数量，而是按 suite 分层逐批实测，把当前 registered-disabled 中风险较低的用例逐步打开，并把高风险项沉淀到专项测试计划中。
