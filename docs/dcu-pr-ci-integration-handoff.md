# DCU PR CI Integration Handoff

## Current Goal

This branch merges the validated DCU PR-CI control flow into the testcase-updated
`sglang` tree. The immediate goal is to prove the official-style PR CI flow
end to end, not to expand the required DCU testcase count.

## What Is Integrated

- `PR Test (DCU)` workflow now follows the official PR CI shape:
  `check-changes -> call-gate -> validate-config -> Stage A -> Stage B matrix -> PR Test (DCU) finish`.
- PR label gate uses the shared `pr-gate.yml`, with DCU-specific `workflow-id`
  and resource-gate job configuration.
- `workflow_dispatch` supports isolated DCU stage reruns by dropdown or typed
  `target_stage`.
- `/rerun-stage` can dispatch `PR Test (DCU)`.
- `/rerun-ut` can detect `register_dcu_ci(...)` files and dispatch `Rerun UT (DCU)`.
- `run_suite.py` supports repeated `--include-file` filters, so PR CI can use a
  conservative baseline while the full enabled suite keeps evolving.
- `run_suite.py --list` no longer imports full SGLang runtime dependencies.
- DCU container startup exports both `SGLANG_IS_IN_CI=1` and
  `SGLANG_IS_IN_CI_DCU=1`.

## Suite Naming Decision

The testcase-updated tree is authoritative for suite names. This integration
keeps:

- `stage-a-test-1-gpu-small-dcu`
- `stage-b-test-1-gpu-small-dcu`
- existing `nightly-dcu*` suites

It does not restore the older short names `stage-a-dcu` / `stage-b-dcu`.

## PR Baseline Used By Workflow

Stage A is intentionally narrowed to the DCU framework smoke file:

- `test/registered/dcu/interface/test_dcu_smoke.py`

Stage B matrix is intentionally narrowed to two CI-flow smoke files:

- `test/registered/dcu/interface/test_dcu_stage_b_flow_0.py`
- `test/registered/dcu/interface/test_dcu_stage_b_flow_1.py`

With `--auto-partition-size 2`, each matrix partition selects one file. This
keeps the first official-flow validation focused on CI mechanics.

The Qwen2.5 server files remain registered in the tree, but they are not used
as PR-required baseline in this validation step because the current DCU image
failed server startup with `ModuleNotFoundError: aiter.ops.triton.gemm`.

## Required Repository Variables

- `DCU_CI_RUNNER_LABEL`
- `DCU_CI_IMAGE`

Optional for test repository debugging:

- `DCU_CI_CONTAINER_NAME`
- `DCU_CI_SKIP_PULL=1`

## Local Verification Already Run

```bash
python3 -m py_compile test/run_suite.py scripts/ci/utils/slash_command_handler.py scripts/ci/dcu/verify_dcu_registration.py
python3 scripts/ci/dcu/verify_dcu_registration.py
PYTHONPATH=python python3 test/run_suite.py --hw dcu --suite stage-a-test-1-gpu-small-dcu --include-file registered/dcu/interface/test_dcu_smoke.py --list
PYTHONPATH=python python3 test/run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --include-file registered/dcu/interface/test_dcu_stage_b_flow_0.py --include-file registered/dcu/interface/test_dcu_stage_b_flow_1.py --auto-partition-id 0 --auto-partition-size 2 --list
PYTHONPATH=python python3 test/run_suite.py --hw dcu --suite stage-b-test-1-gpu-small-dcu --include-file registered/dcu/interface/test_dcu_stage_b_flow_0.py --include-file registered/dcu/interface/test_dcu_stage_b_flow_1.py --auto-partition-id 1 --auto-partition-size 2 --list
python3 -c "import pathlib, yaml; [yaml.safe_load(path.read_text()) for path in [pathlib.Path('.github/workflows/pr-test-dcu.yml'), pathlib.Path('.github/workflows/pr-gate.yml'), pathlib.Path('.github/workflows/rerun-ut-dcu.yml')]]; print('workflow yaml ok')"
git diff --check
```

Observed results:

- DCU registration is healthy: `16` DCU registered test files collected in
  this minimal validation branch.
- Stage A dry-run selected `1` enabled file: `test_dcu_smoke.py`.
- Stage B partition `0/2` selected `test_dcu_stage_b_flow_0.py`.
- Stage B partition `1/2` selected `test_dcu_stage_b_flow_1.py`.
- Workflow YAML parsing passed.
- `git diff --check` passed.

## Not In This Step

- Expanding all enabled DCU tests into required PR signal.
- Solving remaining disabled/deferred tests.
- Enabling slash-command policy, CODEOWNERS, PR template, or merge-oncall flow.
- Promoting Qwen2.5 SRT server files to required PR baseline before the DCU
  runtime image provides the needed `aiter.ops.triton.gemm` module.
