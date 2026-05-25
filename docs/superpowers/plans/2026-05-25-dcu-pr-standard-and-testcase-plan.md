# DCU PR 规范与测试用例建设多 Agent 计划

## 1. 目标

本计划用于把 DCU 当前测试建设从“注册框架和实机 smoke 已跑通”推进到“可按官方 PR CI 规范验证、可持续扩充测试用例”的状态。

本轮目标分两条主线：

1. **PR 规范对齐**：在 `tianlinyang77/sgl-dcu-test.git` 中验证官方式 PR gate、PR 自动触发、manual dispatch、matrix 分片、真实 DCU runner 执行，以及最终 check 聚合。
2. **测试用例建设**：基于当前 191 个 DCU registered test 的状态，收敛 PR 前置门禁、nightly、manual/specialty 三层测试计划，并优先推进剩余专项。

## 2. 多 Agent 分工

| Agent | 负责范围 | 输出 |
|---|---|---|
| 测试现状分析 agent | 阅读 `DCU_CI前置合入测试计划.md`、`DCU推理Cookbook对齐指导.md`、`DCU_CI实机调试与测试分层整理.md`、差距分析和专项推进表 | 当前测试能力、缺口、PR/nightly/manual 分层、下一批优先级 |
| PR 规范 agent | 阅读官方 workflows、gate、slash handler、前一轮 DCU PR CI E2E spec/plan | 官方合入流程规范、阶段 A 必做项、延后项、PR 拆分、验收证据 |
| 实测环境 agent | 只读探测 `10.16.1.58` 和 `sgl-test` | 节点/容器/脚本可用性、安全 smoke 命令 |

本地补充验证：

- 本地静态注册检查：`191` 个 DCU registered test，`132` enabled，`59` disabled。
- suite 分布：
  - `stage-a-dcu`: total `1`，enabled `1`，disabled `0`
  - `stage-b-dcu`: total `133`，enabled `108`，disabled `25`
  - `nightly-dcu`: total `57`，enabled `23`，disabled `34`
- 远端 `10.16.1.58` / `sgl-test` 最小实测：`stage-a-dcu` 通过，1/1 文件，2 个 unittest，耗时 `18.13s`。

## 3. 当前状态判断

### 3.1 已完成能力

- DCU CI 注册框架已具备：
  - `HWBackend.DCU`
  - `register_dcu_ci()`
  - `test/run_suite.py --hw dcu`
  - `stage-a-dcu`
  - `stage-b-dcu`
  - `nightly-dcu`
  - `scripts/ci/dcu/verify_dcu_registration.py`
- `stage-a-dcu` 可作为 PR CI 心跳测试，已在 `10.16.1.58` 容器 `sgl-test` 内通过。
- `stage-b-dcu` 已形成较大 registered surface，当前 108 个 enabled、25 个 disabled。
- CSV 自动 CI 目标已完成注册覆盖：`152/152`。
- AMD/NVIDIA 交集 DCU 注册已闭合：`136/136`。
- 小模型 server 路径已经打通：
  - 本地 `Qwen2.5-0.5B-Instruct`
  - `fa3 + --page-size 64`
  - `/health`
  - `/v1/models`
  - `/v1/chat/completions`
- 已验证并纳入或候选纳入的轻量能力包括：
  - unit/managers
  - unit/mem_cache
  - attention helper
  - Mamba causal conv
  - repeat interleave
  - OpenAI mock handler
  - request length / request queue / ignore_eos
  - EBNF / JSON mode / matched stop
  - large max new tokens
  - profiling
  - scheduler/request logger
  - sampling penalty/original logprobs
  - constrained decoding
  - priority scheduling
  - tokenizer skip init
  - radix attention

### 3.2 主要缺口

当前缺口不在“能不能注册”，而在“能不能作为可信合入信号”：

- GitHub PR workflow 还未在 PR 上端到端证明。
- `run-ci` gate、PR 自动触发、matrix 分片、manual dispatch 尚未形成 GitHub run 证据链。
- `stage-b-dcu` enabled 面很大，但不是全部都有最新完整 suite 级复跑证据。
- 59 个 disabled 仍需要分层处理，其中非 CSV AMD/NVIDIA 交集 disabled 15 个必须走实机验证或专项闭环。
- 精度、性能、大模型、多卡、VLM、LoRA/HiCache、Spec/EAGLE 仍缺模型、数据、阈值和复测命令。
- 部分问题已有真实失败原因：
  - `test_block_int8.py` / W8A8 Block INT8 Fused MoE 曾触发 Triton JIT 段错误。
  - `test_hidden_states.py` 有 HF vs SRT hidden states 数值差异。
  - `test_routing_key_scheduling.py` 快速窗口内未完成且可能残留 server。
  - `test_anthropic_tool_use.py` tool_use 请求返回 500/error events。
  - `test_lora_load_from_tensor.py` 首个 pytest item exit code 137。

## 4. PR 规范建设计划

### 4.1 阶段 A 必做

在 `tianlinyang77/sgl-dcu-test.git` 中按上一份 E2E plan 推进：

1. `ci-smoke-minimal` 分支验证普通 runner 上的 GitHub Actions 编排。
2. `sglang-dcu-mirror` 分支验证真实 SGLang DCU 代码和真实 DCU runner。
3. 复用 `pr-gate.yml`，验证：
   - draft PR 阻断
   - 缺 `run-ci` label 阻断
   - 加 `run-ci` label 后自动触发
4. `pr-test-dcu.yml` 至少包含：
   - `check-changes`
   - `call-gate`
   - `validate-config`
   - `stage-a-dcu`
   - matrix `stage-b-dcu`
   - `pr-test-dcu-finish`
5. `stage-b-dcu` matrix 要设置：
   - `strategy.fail-fast: false`
   - 分片日志包含 `partition_id` 和 `partition_size`
6. `workflow_dispatch` 支持独立触发：
   - `stage-a-dcu`
   - `stage-b-dcu`
7. 证据文档记录具体 GitHub Actions run URL。

### 4.2 建议补充 `pr-test-dcu-finish`

官方 CUDA/AMD workflow 都有 finish job 聚合状态。DCU 如果要模拟 required check，建议阶段 A 就加入：

- `pr-test-dcu-finish` 依赖 `check-changes`、`validate-config`、`stage-a-dcu`、`stage-b-dcu`
- 打印各 job result
- 任一 required job `failure` / `cancelled` 时失败
- 作为后续 required check 的稳定名字

### 4.3 延后内容

以下不进入阶段 A：

- `/rerun-stage` for DCU
- `/rerun-ut` for DCU
- `nightly-dcu` 自动 nightly 链路
- CODEOWNERS / PR template / Merge Oncall
- 15 个非 CSV 交集 disabled 收敛
- 精度/性能/VLM/多卡/LoRA/HiCache/Spec 阈值

## 5. 测试用例分层计划

### 5.1 PR 前置门禁

PR 前置门禁必须稳、短、可重复，不把重型模型问题混进第一阶段。

必选：

```bash
python3 scripts/ci/dcu/verify_dcu_registration.py
python3 run_suite.py --hw dcu --suite stage-a-dcu --timeout-per-file 120
```

保守 `stage-b-dcu` 首批：

- 轻量 unit/helper
- OpenAI mock handler
- 小 tensor / parser / server_args / function_call helper
- 已实机通过的小模型 server validation 中最稳定子集

建议做法：

1. 先在 10.16.1.58 跑 `stage-a-dcu`。
2. 再选 10 到 30 个已验证文件组成 `stage-b-dcu-pr` 口径。
3. 完整复跑 3 次后再作为 PR required check 候选。

### 5.2 Nightly

Nightly 适合覆盖稳定但耗时更长的真实 server 和 debug 工具：

- `test/registered/dcu/server/k100/test_dcu_sglang_server_smoke.py`
- `test/registered/debug_utils/test_crash_dump.py`
- `test/registered/debug_utils/test_soft_watchdog.py`
- 更完整 OpenAI/core/sampling/scheduler/cache validation
- Qwen2.5/Qwen3 小模型 server 回归
- perf smoke，不带正式阈值

### 5.3 Manual/Specialty

专项必须保留模型、数据、硬件、阈值和复测命令，不允许只写泛化 disabled 原因。

专项队列：

| 专项 | 当前重点 |
|---|---|
| 精度评测 | GSM8K/MMLU 10/50 样本 smoke，再跑正式阈值 |
| 性能基线 | BW1000 serving/one-batch，单卡/双卡，跑 3 次建阈值 |
| VLM/多卡 | Qwen2.5-VL/Qwen3-VL、MMMU、encoder DP、DP attention large |
| LoRA/HiCache | 补 base model、adapter、storage，先 basic 后 full |
| Spec/EAGLE | 补 target/draft 本地路径和 accept length/精度门槛 |
| MoE/FP8/量化 | block int8 crash、MLA FP8、MoE torch-compile |
| 已知失败 | dumper E2E、DLLM、tool_use、RL LoRA、routing key |

## 6. 10.16.1.58 实测计划

### 6.1 环境事实

- 节点：`10.16.1.58`
- 主机名：`bw18`
- 宿主机仓库：`/public/home/tianly/sgl/sglang-tly`
- 容器：`sgl-test`
- 容器内仓库：`/home/sgl/sglang-tly`
- Python：`3.10.12`
- 容器镜像：`10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260518-2235`
- 设备：8 张 HCU 可见，`hy-smi` 可输出 0-7 状态表
- 注意：`hy-smi` 前有 `Open mkfd failed` 噪声，但能继续输出设备表

### 6.2 安全预检

```bash
ssh 10.16.1.58 'docker ps --format "{{.Names}} {{.Status}}" | grep sgl-test'
ssh 10.16.1.58 'timeout 10s docker exec sgl-test hy-smi'
ssh 10.16.1.58 'cd /public/home/tianly/sgl/sglang-tly && bash scripts/ci/dcu/dcu_ci_exec.sh --container-name sgl-test -w /home/sgl/sglang-tly -- python3 scripts/ci/dcu/verify_dcu_registration.py'
```

### 6.3 最小 smoke

已在 2026-05-25 执行并通过：

```bash
ssh 10.16.1.58 'cd /public/home/tianly/sgl/sglang-tly && timeout 300s bash scripts/ci/dcu/dcu_ci_exec.sh --container-name sgl-test -w /home/sgl/sglang-tly/test -- python3 run_suite.py --hw dcu --suite stage-a-dcu --timeout-per-file 120'
```

结果：

```text
Test Summary: 1/1 passed
Ran 2 tests
Success. Time elapsed: 18.13s
```

### 6.4 下一步推荐实测顺序

1. 注册健康检查。
2. `stage-a-dcu`。
3. `stage-b-dcu` 小分片 dry-run 口径：先用 matrix 参数验证分片选择，不急着跑全量。
4. 选 10 到 30 个已验证稳定文件组成 PR 前置候选清单，跑 3 次。
5. 再扩到当前 enabled 的 `stage-b-dcu` 大集合。
6. 失败项按文件归入专项，不把大集合失败直接当作 CI 框架失败。

## 7. 下一批建设优先级

### P0：PR 规范闭环

- 完成 `sgl-dcu-test` 双分支验证。
- 加 `pr-test-dcu-finish`。
- 记录 GitHub Actions 证据。
- 确认真实 DCU runner label/image 变量。

### P1：PR 前置测试基线

- 固定 `stage-a-dcu`。
- 从 108 个 enabled `stage-b-dcu` 中选最稳子集做 PR gate。
- 对候选子集做 3 次实机复跑。
- 每次复跑记录 commit、节点、容器、命令、通过/失败文件。

### P2：小模型 server/nightly

- 继续使用本地 `Qwen2.5-0.5B-Instruct`。
- 固定 `fa3 + --page-size 64`。
- 扩展 OpenAI/core/sampling/scheduler/cache validation。
- server 类测试统一检查残留进程。

### P3：专项并行推进

- perf：建立 BW1000 阈值，不复用 CUDA/5090 阈值。
- eval：GSM8K/MMLU 小样本 smoke。
- VLM：先本地 VLM 小样本，再 MMMU。
- LoRA/HiCache：补本地 base/adapter/storage。
- Spec/EAGLE：补 target/draft 模型。
- MoE/FP8：先最小复现 crash，再决定启用范围。

## 8. 验收标准

### PR 规范验收

- 无 `run-ci` PR 被 gate 阻断。
- 加 `run-ci` 后 DCU workflow 自动执行。
- `workflow_dispatch` 可单独跑 `stage-a-dcu` / `stage-b-dcu`。
- `stage-b-dcu` matrix 分片日志清晰。
- `pr-test-dcu-finish` 能聚合状态并给出稳定 check 名称。
- 真实 DCU runner 上 `stage-a-dcu` 通过。

### 测试用例建设验收

- `verify_dcu_registration.py` 持续通过。
- PR 前置门禁清单明确，且至少 3 次实机复跑通过。
- Nightly 清单明确，server 类测试有残留进程检查。
- 15 个非 CSV 交集 disabled 每个都有专项归属、复测命令和前置条件。
- 精度和性能专项不再只登记空壳条目，而是具备模型、数据、阈值和运行记录。
