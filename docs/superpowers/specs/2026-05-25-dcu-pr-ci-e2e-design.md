# DCU PR CI End-to-End Design

## Background

The DCU test registration framework is already in place: `register_dcu_ci()`,
`HWBackend.DCU`, `run_suite.py --hw dcu`, the `stage-a-dcu`,
`stage-b-dcu`, and `nightly-dcu` suites, DCU registration collection,
container wrappers, and local or real-machine smoke tests.

The remaining gap for official-style merge readiness is the GitHub PR CI
chain. The current DCU work proves that tests can be registered and executed,
but it does not yet prove that the official PR flow can trigger, gate, shard,
and execute DCU CI end to end.

This design focuses only on the PR CI trigger path. It does not attempt to
finish the remaining disabled DCU test areas or the full official merge
operations process.

## Goals

- Prove a DCU PR workflow can be triggered automatically by pull requests.
- Reuse the official `run-ci` label gate semantics.
- Support manual `workflow_dispatch` for `stage-a-dcu` and `stage-b-dcu`.
- Prove `stage-b-dcu` can run through a GitHub Actions matrix partition.
- First validate GitHub Actions orchestration on a normal runner with a mock
  suite.
- Then validate the same workflow shape on a real DCU runner with the mirrored
  SGLang code.
- Keep evidence links for gate behavior, automatic PR execution, manual
  dispatch, partition logs, and real DCU execution.

## Non-Goals

- Do not implement `/rerun-stage` for DCU in this phase.
- Do not implement `/rerun-ut` for DCU in this phase.
- Do not wire `nightly-dcu` into an automatic nightly dependency chain.
- Do not close the remaining 15 disabled non-CSV intersection tests.
- Do not add new accuracy, performance, VLM, multi-card, LoRA, HiCache, or
  speculative decoding thresholds.
- Do not solve repository-level merge operations such as CODEOWNERS, PR
  templates, Merge Oncall, or full pre-commit policy.

## Test Repository Strategy

Use the existing test repository:

```text
https://github.com/tianlinyang77/sgl-dcu-test.git
```

Use one repository with two validation branches:

- `ci-smoke-minimal`: minimal GitHub Actions reproduction.
- `sglang-dcu-mirror`: full mirror of the current SGLang DCU branch.

This keeps labels, Actions permissions, repository variables, runner access,
and evidence links in one GitHub repository while still separating orchestration
debugging from real DCU execution.

## Architecture

### Minimal Orchestration Branch

The `ci-smoke-minimal` branch contains only the files required to reproduce the
official PR CI control flow:

- `.github/workflows/pr-gate.yml`
- `.github/workflows/pr-test-dcu.yml`
- A lightweight fake `test/run_suite.py` or equivalent mock suite runner.
- Minimal fake test files or scripts that print selected suite and partition
  information.

This branch validates GitHub behavior only. It does not validate SGLang runtime
or DCU hardware behavior.

### Full Mirror Branch

The `sglang-dcu-mirror` branch contains the current SGLang DCU code, including:

- `python/sglang/test/ci/ci_register.py`
- `test/run_suite.py`
- `scripts/ci/dcu/*`
- `test/registered/dcu/*`
- `.github/workflows/pr-test-dcu.yml`

This branch validates that the same workflow shape runs against real SGLang
suite registration and DCU container execution.

## Workflow Design

Add or complete `.github/workflows/pr-test-dcu.yml` with a structure aligned to
the official PR workflows, especially `pr-test-amd.yml`.

The workflow supports:

- `pull_request`
- `workflow_dispatch`

The primary jobs are:

- `call-gate`: reuse `.github/workflows/pr-gate.yml`.
- `check-changes`: use paths filtering to decide whether DCU CI should run.
- `validate-config`: fail early when required DCU runner or image settings are
  missing in the full mirror path.
- `stage-a-dcu`: run the smallest DCU suite without partitioning.
- `stage-b-dcu`: run the larger DCU suite through matrix partitioning.

The workflow should keep `nightly-dcu` out of the PR automatic path for this
phase.

## Runner And Image Configuration

Use mixed configuration:

- PR-triggered full mirror runs read:
  - `vars.DCU_CI_RUNNER_LABEL`
  - `vars.DCU_CI_IMAGE`
- `workflow_dispatch` keeps optional overrides for:
  - `runner_label`
  - `image`
  - `target_stage`
  - `partition_size`
  - `continue_on_error`

This supports official-style repository configuration while preserving manual
debug flexibility.

The minimal branch can run on `ubuntu-latest` and should not require DCU
variables.

## Data Flow

### Minimal PR Flow

1. Open a PR from `ci-smoke-minimal` to the repository default branch.
2. Without the `run-ci` label, `call-gate` blocks the workflow and DCU jobs do
   not run.
3. Add the `run-ci` label.
4. The workflow executes:
   `check-changes -> call-gate -> stage-a-dcu -> stage-b-dcu`.
5. `stage-b-dcu` runs at least two matrix partitions.
6. Each partition prints its `partition_id` and `partition_size`.

### Full Mirror PR Flow

1. Push the current SGLang DCU code to `sglang-dcu-mirror`.
2. Configure repository variables for the real DCU runner and container image.
3. Open a PR from `sglang-dcu-mirror` to the repository default branch.
4. Verify the no-label gate behavior.
5. Add `run-ci`.
6. Run `validate-config`.
7. Start the DCU container through `scripts/ci/dcu/dcu_ci_start_container.sh`.
8. Run `scripts/ci/dcu/verify_dcu_registration.py`.
9. Run:

```bash
python3 run_suite.py --hw dcu --suite stage-a-dcu
```

10. Run `stage-b-dcu` through matrix partitioning:

```bash
python3 run_suite.py \
  --hw dcu \
  --suite stage-b-dcu \
  --auto-partition-id "${PARTITION_ID}" \
  --auto-partition-size "${PARTITION_SIZE}"
```

### Manual Dispatch Flow

The same workflow supports manual dispatch for:

- `stage-a-dcu`
- `stage-b-dcu`

Manual dispatch can override runner, image, partition size, and
`continue_on_error`. This proves the base capability needed for later
slash-command integration without implementing slash commands in this phase.

## Error Handling

### Gate Failures

Missing `run-ci`, draft PRs, and cooldown failures should fail at `call-gate`.
This is expected behavior and must be recorded as positive evidence of official
gate alignment.

### Path Filtering Skips

The DCU paths filter must include:

- `.github/workflows/pr-test-dcu.yml`
- `.github/workflows/pr-gate.yml`
- `scripts/ci/dcu/**`
- `test/**`
- `python/sglang/test/ci/**`

The minimal branch should include enough changed files to avoid accidental
skips during orchestration validation.

### Missing Runner Or Image Configuration

The full mirror workflow should fail early in `validate-config` when required
repository variables or dispatch inputs are missing. The failure message should
name the missing setting.

### Container Startup Failures

Container startup failures should preserve `docker pull`, `docker run`, and
device or mount logs. The workflow should not convert these failures into
success.

### Registration Failures

The full mirror path should run `verify_dcu_registration.py` before suite
execution. Unknown suites, no DCU registrations, or invalid registration data
should fail before the suite runs.

### Partition Failures

`stage-b-dcu` should use `strategy.fail-fast: false`. One failing partition
must not cancel the other partitions, so the final evidence can identify
whether failures are isolated to a partition or shared across all partitions.

### Test Failures

PR-triggered full mirror runs should default to `continue_on_error=false`.
Manual dispatch may use `continue_on_error=true` for debugging and broader log
collection.

## Validation Evidence

The implementation phase should record these six evidence items:

1. Minimal PR without `run-ci`: gate blocks DCU jobs.
2. Minimal PR with `run-ci`: automatic PR workflow runs `stage-a-dcu` and
   matrixed `stage-b-dcu`.
3. Minimal manual dispatch: `stage-a-dcu` and `stage-b-dcu` can be triggered
   independently.
4. Full mirror configuration validation: missing variables fail clearly, and
   configured variables pass validation.
5. Full mirror real DCU `stage-a-dcu`: registration verification and suite
   execution pass on the DCU runner.
6. Full mirror real DCU `stage-b-dcu`: matrix partitions run with clear
   partition logs. If a test fails, the failure is attributable to the specific
   partition or test file rather than to workflow orchestration.

## Acceptance Criteria

- The `tianlinyang77/sgl-dcu-test.git` repository contains the two validation
  branches.
- The minimal branch proves the official-style PR gate and matrix mechanics on
  a normal runner.
- The full mirror branch proves real DCU `stage-a-dcu` execution on a DCU
  runner.
- `stage-b-dcu` runs through GitHub matrix partitioning in the full mirror
  branch, with useful logs for each partition.
- Evidence links are collected for every validation item above.
- Slash commands, nightly automation, and disabled test convergence remain out
  of scope for this phase.
