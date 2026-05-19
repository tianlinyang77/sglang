# DCU CI PR1 基础设施设计

## 背景

DCU 当前已有来自 `sglang.csv` 的测试清单和较多功能测试基础，但尚未接入 SGLang 官方注册式 CI 体系。当前阻塞项包括缺少 `register_dcu_ci()`、缺少 `HWBackend.DCU`、缺少 `--hw dcu` 路由、缺少 DCU suite 列表，以及缺少 `test/registered/dcu/` 目录骨架。

本 PR1 只建设最小可合入基础设施，不批量迁移测试，不新增 GitHub workflow，不新增精度或性能测试。

## 目标

PR1 的目标是让仓库具备 DCU 测试注册和 suite dry-run 能力：

- 测试文件可以通过 `register_dcu_ci()` 声明 DCU CI 注册信息。
- `test/run_suite.py` 可以接受 `--hw dcu`。
- `stage-a-dcu`、`stage-b-dcu`、`nightly-dcu` 可以被识别为合法 DCU suite。
- `test/registered/dcu/` 目录结构存在，供后续 PR 放置 DCU 专项测试。
- 在没有 DCU 测试文件注册时，suite dry-run 应正常输出空 suite 提示，不应崩溃。

## 非目标

PR1 不做以下事情：

- 不给 125 个通用测试批量添加 `register_dcu_ci()`。
- 不新增 GSM8K、MMLU、MMMU 精度评测。
- 不新增性能 benchmark。
- 不新增或接入 `.github/workflows/pr-test-dcu.yml`。
- DCU runner 名称、容器镜像、安装脚本或机器调度策略留到后续 workflow PR 中确定。

## 代码设计

### CI 注册层

文件：`python/sglang/test/ci/ci_register.py`

新增 `HWBackend.DCU`，并新增 `register_dcu_ci()` no-op marker。该函数与 `register_amd_ci()`、`register_cuda_ci()`、`register_npu_ci()` 保持一致，仅用于 AST 解析，不在运行时执行逻辑。

更新 `__all__` 和 `REGISTER_MAPPING`，使 `collect_tests()` 能把 DCU 注册调用解析成 `CIRegistry`，其中 `backend` 为 `HWBackend.DCU`。

### Suite 路由层

文件：`test/run_suite.py`

更新 `HW_MAPPING`，新增 `"dcu": HWBackend.DCU`。

在 `PER_COMMIT_SUITES` 中为 DCU 添加：

- `stage-a-dcu`
- `stage-b-dcu`

在 `NIGHTLY_SUITES` 中为 DCU 添加：

- `nightly-dcu`

现有 `filter_tests()`、`collect_tests()`、`auto_partition()`、`run_unittest_files()` 流程保持不变。

### 测试工具层

文件：`python/sglang/test/test_utils.py`

新增 DCU 相关测试辅助能力：

- `is_in_dcu_ci()`：读取 `SGLANG_IS_IN_CI_DCU`。
- `is_dcu()`：根据 `DCU_VISIBLE_DEVICES` 或 `HIP_VISIBLE_DEVICES` 判断 DCU 环境。
- `DEFAULT_URL_FOR_DCU_TEST`：复用当前 `DEFAULT_URL_FOR_TEST`。

如果处于 DCU CI 环境，则复用大模型 CI 的长启动超时策略，将 `DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH` 放宽到 `3600`。

### 目录骨架

创建以下目录骨架并使用 `.gitkeep` 保留空目录：

```text
test/registered/dcu/
├── .gitkeep
├── accuracy/
│   └── k100/
│       └── .gitkeep
├── perf/
│   └── k100/
│       └── .gitkeep
└── disaggregation/
    └── .gitkeep
```

## 数据流

1. 后续测试文件调用 `register_dcu_ci()` 并传入常量形式的 `est_time`、`suite` 和可选 `nightly`。
2. `test/run_suite.py` 扫描 `test/registered/**/*.py`。
3. `collect_tests()` 使用 AST 解析注册调用。
4. `REGISTER_MAPPING` 将 `register_dcu_ci` 映射到 `HWBackend.DCU`。
5. `filter_tests()` 根据 `--hw dcu`、`--suite`、`--nightly` 筛选 DCU 测试。
6. 当前 PR1 没有注册测试时，空 suite 正常提示并跳过。

## 错误处理

- 未知 suite 保持现有行为：打印 warning。
- 无测试命中保持现有行为：提示增量迁移期间可预期并跳过。
- 注册参数解析保持现有严格规则：`est_time`、`suite`、`nightly`、`disabled` 必须是支持的常量参数。
- 不改变已有 CUDA、AMD、NPU、CPU 注册和路由行为。

## 验证计划

至少执行以下验证：

```bash
python3 -m py_compile python/sglang/test/ci/ci_register.py test/run_suite.py python/sglang/test/test_utils.py
python3 test/run_suite.py --hw dcu --suite stage-a-dcu --continue-on-error
python3 test/run_suite.py --hw dcu --suite stage-b-dcu --continue-on-error
python3 test/run_suite.py --hw dcu --suite nightly-dcu --nightly --continue-on-error
```

预期结果：

- Python 文件编译通过。
- 三个 DCU suite 可被 CLI 识别。
- 当前没有注册测试时输出空 suite 提示，不因 `--hw dcu` 或 suite 路由缺失失败。

## 后续 PR 边界

PR2 再开始给通用测试分批添加 `register_dcu_ci()`，优先选择核心、OpenAI API、Attention、KV Cache 等 P0 子集。

PR3 再新增 DCU 精度评测，例如 GSM8K、MMLU、MMMU。

PR4 再新增性能 benchmark、量化专项、大模型和多卡专项测试。
