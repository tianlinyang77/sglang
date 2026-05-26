# DCU PR CI E2E Results

## Minimal GitHub Orchestration

- Repository: `tianlinyang77/sgl-dcu-test`
- PR: `https://github.com/tianlinyang77/sgl-dcu-test/pull/1`
- Successful run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26381591833`
- Head commit: `4e831ba ci: trigger dcu workflow on pr labels`
- Slash-command smoke branch: `dcu-slash-smoke`

Validated behavior:

- `run-ci` label gate blocks PR CI before the label is present.
- Adding `run-ci` triggers the workflow through the `labeled` pull request event.
- `Check changes`, `Call PR gate`, `Stage A DCU smoke`, two `Stage B DCU smoke` matrix partitions, and `PR Test (DCU) finish` completed successfully.

## Real DCU Mirror Status

This repository now carries the real SGLang DCU PR workflow shape in
`.github/workflows/pr-test-dcu.yml`.

The first real DCU mirror PR validated the official-style PR chain end to end:

- PR: `https://github.com/tianlinyang77/sgl-dcu-test/pull/2`
- Successful run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26390659853`
- Head commit: `4d1e07b0c1bb4c3b9a9823dc51580957dac458e9`
- Required check candidate: `PR Test (DCU) finish`
- Temporary runner: `nmz26-dcu-pr` on `10.16.1.26`
- Runner label: `dcu-nmz26`
- Container image: `10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.10rc0-ubuntu22.04-dtk26.04-py3.10-20260518-2235`

Validated behavior:

- Missing `run-ci` label blocks the PR workflow at `call-gate`.
- Adding `run-ci` triggers the workflow again.
- `validate-config` fails clearly when `DCU_CI_RUNNER_LABEL` or `DCU_CI_IMAGE` is missing.
- With both variables set, `stage-a-dcu` and the two `stage-b-dcu` matrix partitions run on the DCU runner.
- `PR Test (DCU) finish` completes successfully after both Stage B matrix partitions pass.
- `/rerun-stage stage-a-dcu` and `/rerun-stage stage-b-dcu` are routed by the shared slash-command handler to `PR Test (DCU)` using the same workflow_dispatch pattern as official PR Test and PR Test (AMD).
- `/rerun-ut` supports DCU registered files through `Rerun UT (DCU)`, resolving runner and image from the same DCU repository variables as the PR workflow.
- The shared `pr-gate` cooldown lookup is parameterized so DCU gate checks count DCU workflow runs instead of hard-coding `pr-test.yml`.

Successful job timeline:

- `Check changes`: success, `2026-05-25T08:14:56Z` to `2026-05-25T08:15:20Z`
- `Call PR gate / pr-gate`: success, `2026-05-25T08:15:23Z` to `2026-05-25T08:15:30Z`
- `Validate DCU config`: success, `2026-05-25T08:15:33Z` to `2026-05-25T08:15:39Z`
- `Stage A DCU smoke`: success on `nmz26-dcu-pr`, `2026-05-25T08:15:43Z` to `2026-05-25T08:16:56Z`
- `Stage B DCU smoke (0)`: success on `nmz26-dcu-pr`, `2026-05-25T08:17:01Z` to `2026-05-25T08:39:09Z`
- `Stage B DCU smoke (1)`: success on `nmz26-dcu-pr`, `2026-05-25T08:39:13Z` to `2026-05-25T09:04:12Z`
- `PR Test (DCU) finish`: success, `2026-05-25T09:04:30Z` to `2026-05-25T09:04:33Z`

Stability reruns on the same PR baseline:

- Run 1: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26390659853`
  - Head commit: `4d1e07b0c1bb4c3b9a9823dc51580957dac458e9`
  - Result: `PR Test (DCU) finish` success
  - Stage A: `2026-05-25T08:15:43Z` to `2026-05-25T08:16:56Z`
  - Stage B partition 0: `2026-05-25T08:17:01Z` to `2026-05-25T08:39:09Z`
  - Stage B partition 1: `2026-05-25T08:39:13Z` to `2026-05-25T09:04:12Z`
- Run 2: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26392722768`, attempt 1
  - Head commit: `a03caa88fab9baaf170b41c5c72aa44a3ccfeb14`
  - Result: `PR Test (DCU) finish` success
  - Stage A: `2026-05-25T09:10:22Z` to `2026-05-25T09:11:37Z`
  - Stage B partition 0: `2026-05-25T09:11:42Z` to `2026-05-25T09:33:20Z`
  - Stage B partition 1: `2026-05-25T09:33:23Z` to `2026-05-25T09:58:26Z`
- Run 3: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26392722768`, attempt 2
  - Head commit: `a03caa88fab9baaf170b41c5c72aa44a3ccfeb14`
  - Result: `PR Test (DCU) finish` success
  - Stage A: `2026-05-25T10:01:54Z` to `2026-05-25T10:03:03Z`
  - Stage B partition 0: `2026-05-25T10:03:08Z` to `2026-05-25T10:25:07Z`
  - Stage B partition 1: `2026-05-25T10:25:11Z` to `2026-05-25T10:49:52Z`
- Control-flow alignment run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26404025409`
  - Head commit: `5488a4d ci: align dcu pr control flow`
  - Result: `PR Test (DCU) finish` success
  - Included additions: DCU `pr-gate` workflow-id parameterization, `/rerun-stage` routing to `PR Test (DCU)`, and `Rerun UT (DCU)` workflow presence on the mirror branch
  - Stage B partition 0 and partition 1 both completed successfully on `nmz26-dcu-pr`

Slash-command runtime validation on PR #3:

- PR: `https://github.com/tianlinyang77/sgl-dcu-test/pull/3`
- Default branch fix: changed the test repository default branch from `ci-smoke-minimal` to `main` so `issue_comment` workflows run from the branch that contains `slash-command-handler.yml`.
- Temporary test-repo permission: added `tianlinyang77` to `.github/CI_PERMISSIONS.json` on the smoke PR branch so the official permission gate accepts `/rerun-stage` and `/rerun-ut`.
- `/tag-run-ci-label`
  - Slash handler run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26426708606`, success
  - Result: PR #3 labels changed from `documentation` to `documentation` + `run-ci`
  - Note: PR #3 only changed docs/permission files, so this validates the official label slash entrypoint without re-running the DCU path-filtered PR workflow.
- `/rerun-stage stage-a-dcu`
  - Slash handler run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26426456809`, success
  - Dispatched DCU run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26426462188`, display title `[stage-a-dcu]`, success
  - `Check changes`: success, `2026-05-26T01:05:30Z` to `2026-05-26T01:05:34Z`
  - `Validate DCU config`: success, `2026-05-26T01:05:37Z` to `2026-05-26T01:05:42Z`
  - `Call PR gate`: skipped as expected for `workflow_dispatch`
  - `Stage A DCU smoke`: success, `2026-05-26T01:05:45Z` to `2026-05-26T01:06:49Z`
  - `Stage B DCU smoke`: skipped as expected for a stage-A-only rerun
  - `PR Test (DCU) finish`: success, `2026-05-26T01:06:52Z` to `2026-05-26T01:06:55Z`
- `/rerun-ut test/registered/dcu/interface/test_dcu_smoke.py::TestDCUSmoke.test_import_sglang`
  - Slash handler run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26426531375`, success
  - Dispatched DCU UT run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26426537606`, display title `[rerun-ut-dcu]`, success
  - `Validate DCU config`: success, `2026-05-26T01:08:21Z` to `2026-05-26T01:08:28Z`
  - `Rerun UT (DCU)`: success, `2026-05-26T01:08:32Z` to `2026-05-26T01:09:31Z`
- Post-run process check on `10.16.1.26` at `2026-05-26 09:18:05 CST`: no matching `ci_sglang_dcu_26426462188*` or `ci_sglang_dcu_26426537606*` containers; no residual `run_suite.py`, `sglang serve`, pytest, or `python3 .*test_` processes beyond the `pgrep` command itself.

## Current Flow-Smoke PR CI Validation

The current validation scope is to prove the official-style PR CI registration,
triggering, gate, matrix, runner, container, and finish-check flow. It is not
intended to prove broad DCU test coverage yet.

- PR: `https://github.com/tianlinyang77/sgl-dcu-test/pull/3`
- Successful run: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26435615195`
- Head commit: `3f8cb44bee0a3197e126ea54572cafa96feb33ed`
- Commit title: `ci: narrow dcu pr flow smoke`
- Required check candidate: `PR Test (DCU) finish`
- Lint run on the same commit: `https://github.com/tianlinyang77/sgl-dcu-test/actions/runs/26435615071`, success
- Temporary runner: `nmz26-dcu-pr` on `10.16.1.26`
- Runner label: `dcu-nmz26`

Current required flow-smoke baseline:

- `stage-a-dcu`: `test/registered/dcu/interface/test_dcu_smoke.py`
- `stage-b-dcu`: `test/registered/unit/managers/test_prefill_adder.py`
- `stage-b-dcu`: `test/registered/unit/mem_cache/test_radix_cache_unit.py`

Successful job timeline:

- `Check changes`: success, `2026-05-26T06:12:31Z` to `2026-05-26T06:12:39Z`
- `Call PR gate / pr-gate`: success, `2026-05-26T06:12:42Z` to `2026-05-26T06:12:49Z`
- `Validate DCU config`: success, `2026-05-26T06:12:52Z` to `2026-05-26T06:13:00Z`
- `Stage A DCU smoke`: success, `2026-05-26T06:13:04Z` to `2026-05-26T06:14:16Z`
- `Stage B DCU smoke (0)`: success, `2026-05-26T06:14:20Z` to `2026-05-26T06:15:38Z`
- `Stage B DCU smoke (1)`: success, `2026-05-26T06:15:42Z` to `2026-05-26T06:16:56Z`
- `PR Test (DCU) finish`: success, `2026-05-26T06:16:59Z` to `2026-05-26T06:17:04Z`

Stage B matrix evidence:

- Partition `0/2` printed `Running stage-b-dcu partition 0/2` and ran only
  `test/registered/unit/mem_cache/test_radix_cache_unit.py`.
- Partition `1/2` printed `Running stage-b-dcu partition 1/2` and ran only
  `test/registered/unit/managers/test_prefill_adder.py`.

Post-rerun process check on `10.16.1.26`:

- No active `ci_sglang_dcu_26392722768_*` containers after attempt 2.
- No residual `run_suite.py`, `sglang serve`, pytest, or `python3 .*test_` processes from the CI job.
- HCU 0-3 returned to `0%` VRAM and `0%` HCU utilization after the run.
- No active `ci_sglang_dcu_26404025409_*` containers and no residual `run_suite.py`, `sglang serve`, pytest, or `python3 .*test_` processes after the control-flow alignment run.

Environment fixes required before the successful run:

- Mount `/opt/hyhal` into the CI container when present so `rocm-smi` dependencies resolve.
- Allow `DCU_CI_SKIP_SGLANG_BUILD=1` in the test repository because the shared image already carries the build artifacts and NMZ reports `gfx938`, while the current `sgl-kernel` build path expects `gfx942` or `gfx950`.
- Set `PYTHONDONTWRITEBYTECODE=1` and clean/chown the checkout after container jobs to prevent root-owned `__pycache__` files from breaking the next checkout.
- Mount `/public/opendas/DL_DATA/llm-models` into the CI container so the PR baseline can resolve local model paths.

## Earlier Broader PR Baseline Evidence

Earlier validation also proved a broader but slower PR baseline. This is kept as
historical evidence only; the current official-flow validation uses the smaller
flow-smoke baseline above.

- `stage-a-dcu`: `test/registered/dcu/interface/test_dcu_smoke.py`
- `stage-b-dcu`: 28 files with BW1000 `sgl-test` evidence

All other DCU `stage-b-dcu` registrations remain present for coverage tracking,
but are deferred from the required PR signal until they have repeat BW1000
evidence or a dedicated nightly/manual specialty plan.

## Real DCU Runner Evidence

Environment:

- Host: `10.16.1.58`
- Container: `sgl-test`
- Wrapper: `scripts/ci/dcu/dcu_ci_exec.sh`
- Workdir in container: `/home/sgl/sglang-tly/test`

Commands and results:

- `python3 run_suite.py --hw dcu --suite stage-a-dcu --timeout-per-file 120`
  - Result: passed
  - Summary: `1/1` file passed; the file runs 2 pytest tests
  - Elapsed: `18.55s`
- `python3 run_suite.py --hw dcu --suite stage-b-dcu --auto-partition-id 0 --auto-partition-size 2 --timeout-per-file 900`
  - First attempt result: failed on `test/registered/radix_cache/test_radix_attention.py`
  - Failure class: timeout after `900s`
  - Action: deferred `test_radix_attention.py` from PR-required baseline to nightly/manual until radix-cache server integration is repeatable inside the PR budget
  - Rerun result after deferral: passed
  - Summary: `15/15` files passed
  - Elapsed: `1154.57s`
- `python3 run_suite.py --hw dcu --suite stage-b-dcu --auto-partition-id 1 --auto-partition-size 2 --timeout-per-file 900`
  - Result: passed
  - Summary: `13/13` files passed
  - Elapsed: `1362.03s`

Post-run process check:

- `pgrep -af "sglang serve|python3 .*run_suite|python3 .*test_" || true`
- Result: no residual server or test processes; only the `pgrep` command matched.
