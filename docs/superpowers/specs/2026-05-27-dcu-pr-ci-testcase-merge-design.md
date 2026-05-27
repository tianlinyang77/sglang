# DCU PR-CI and Testcase Merge Design

## Goal

Merge the already-validated DCU PR-CI flow into the testcase-updated SGLang
workspace at `/public/home/tianly/sgl/sglang`.

The merge goal is control-flow alignment with the official PR CI process:
`pull_request` trigger, `run-ci` gate, config validation, DCU runner execution,
matrix partitioning, and `PR Test (DCU) finish`.

This merge is not intended to expand the number of required DCU PR tests. The
current testcase repository remains the source of truth for enabled/disabled
test coverage and BW1000 validation status.

## Current Inputs

Target testcase workspace:

- Path: `/public/home/tianly/sgl/sglang`
- Current branch at design time: `dcu-pr-ci-testcase-integration`
- Previous branch: `v0.5.10rc0_dev`
- Role: primary integration target
- State: large dirty worktree containing the latest testcase registrations,
  DCU specialty tests, workflow drafts, and validation documents.

PR-CI donor workspace:

- Path: `/public/home/tianly/sgl/sglang-tly/.worktrees/dcu-pr-flow-smoke`
- Branch: `dcu-pr-flow-smoke`
- Role: donor for validated PR-CI control-flow changes
- Latest pushed fork branch: `tianlinyang77/sglang:dcu-pr-flow-smoke`
- Latest donor commit at design time:
  `b2a4511d71da9c0dc699029759ff8beecf279170`

Status document:

- Path: `/public/home/tianly/sgl/DCU_PR演进与验证状态.md`
- Role: testcase state reference
- Current key facts: 203 DCU registered files, 73 enabled
  `stage-b-test-1-gpu-small-dcu` tests, 61 skipped/disabled in that suite, and
  a set of BW1000-validated specialty tests.

## Direction Decision

Use `/public/home/tianly/sgl/sglang` as the merge target and bring the PR-CI
flow into it.

Do not merge the testcase work into the older PR-CI donor branch. The testcase
workspace has the fresher suite names, registered test state, disabled reasons,
and specialty DCU files. The PR-CI donor branch is narrower and should be used
as a source of proven CI control-flow patches.

## Alternatives Considered

### Option A: PR-CI into testcase workspace

This is the recommended path.

Benefits:

- Preserves the latest testcase work as the source of truth.
- Keeps current suite naming such as `stage-a-test-1-gpu-small-dcu` and
  `stage-b-test-1-gpu-small-dcu`.
- Reduces risk of overwriting large uncommitted testcase updates.
- Produces an internal MR that naturally reads as testcase baseline plus CI
  integration.

Trade-off:

- Requires careful manual reconciliation of overlapping files such as
  `.github/workflows/pr-test-dcu.yml`, `python/sglang/test/ci/ci_register.py`,
  `test/run_suite.py`, and `scripts/ci/dcu/*`.

### Option B: testcase workspace into PR-CI donor branch

This is not recommended.

Benefits:

- Starts from a branch where PR-CI E2E evidence is already compact and clean.

Trade-off:

- High risk of losing or flattening the much larger testcase update set.
- Harder to preserve the exact disabled reasons and BW1000 validation state.
- More likely to force large conflict resolution against test files that should
  remain owned by the testcase work.

### Option C: create a third clean branch and replay both sides

This is safer than Option B but heavier than needed.

Benefits:

- Cleanest history if time allows.
- Makes each imported commit explicit.

Trade-off:

- The testcase workspace is already a large dirty worktree, so replaying it into
  a new clean branch adds time and risk without improving the immediate merge
  outcome.

## Merge Architecture

The integrated branch should have two conceptual layers:

1. Testcase baseline layer
   - Commit the current `/public/home/tianly/sgl/sglang` testcase state first.
   - This protects the colleague's enabled/disabled status, BW1000 specialty
     tests, and broad registered updates before PR-CI changes are applied.

2. PR-CI flow layer
   - Apply only the CI flow pieces needed to match the validated official PR
     behavior.
   - Keep testcase ownership in the target workspace.
   - Prefer the target workspace's suite names and testcase lists over donor
     names when conflicts arise.

## Files to Reconcile

Workflow and gate files:

- `.github/workflows/pr-test-dcu.yml`
- `.github/workflows/nightly-test-dcu.yml`
- `.github/workflows/pr-gate.yml`
- `.github/workflows/rerun-ut-dcu.yml`

CI runner and helper files:

- `scripts/ci/dcu/dcu_ci_exec.sh`
- `scripts/ci/dcu/dcu_ci_start_container.sh`
- `scripts/ci/dcu/dcu_ci_install_dependency.sh`
- `scripts/ci/dcu/verify_dcu_registration.py`
- `scripts/ci/dcu/analyze_dcu_csv_coverage.py`
- `scripts/ci/utils/slash_command_handler.py`

Registry and suite files:

- `python/sglang/test/ci/ci_register.py`
- `test/run_suite.py`
- `test/registered/dcu/**`
- broad `test/registered/**` files already updated in the target workspace

Docs:

- Keep `/public/home/tianly/sgl/DCU_PR演进与验证状态.md` as the high-level
  testcase status source.
- Bring over or adapt `docs/dcu-ci-e2e-results.md` for PR-CI E2E evidence if
  the target branch does not already carry an equivalent document.

## Conflict Policy

Use the testcase workspace as authoritative for:

- enabled/disabled test state
- disabled reasons
- suite names and target stage names
- BW1000 model paths and specialty testcase files
- current registered test counts

Use the PR-CI donor branch as authoritative for:

- official-style `run-ci` gate behavior
- `PR Test (DCU) finish` aggregation behavior
- workflow_dispatch stage rerun behavior where compatible with target suite names
- inline `git diff` path filtering instead of `dorny/paths-filter@v3`
- E2E evidence structure and required check naming

When both sides edit the same file, merge behavior rather than blindly choosing
one side.

## Expected Integrated Behavior

The final integrated branch should support:

- PR trigger types: `opened`, `synchronize`, `reopened`, `labeled`,
  `ready_for_review`.
- Missing `run-ci` label blocks DCU PR jobs through `pr-gate`.
- Adding `run-ci` triggers or allows DCU PR jobs.
- `validate-config` checks runner label and image variables before DCU jobs.
- Stage A runs the target workspace's DCU smoke suite.
- Stage B runs the target workspace's current PR baseline suite with matrix
  partitioning.
- The required check candidate remains `PR Test (DCU) finish`.

## Validation Plan

Static validation:

- `python3 scripts/ci/dcu/verify_dcu_registration.py`
- `python3 -m py_compile test/run_suite.py`
- YAML parse for `.github/workflows/pr-test-dcu.yml`
- `git diff --check`

Local or remote DCU validation should use the target workspace's current node
and container from the status document unless redirected:

- Host: `10.16.1.66`
- Container: `dxl-sglang`
- Visible card variable: `HIP_VISIBLE_DEVICES=<card_ids>` only

GitHub PR validation remains a flow proof:

- Without `run-ci`: gate fails and DCU jobs do not run.
- With `run-ci`: gate, config validation, Stage A, Stage B matrix, and finish
  succeed.

## Out of Scope

- Expanding the required PR testcase count.
- Resolving all disabled DCU tests.
- Changing broad registered testcase ownership beyond conflict reconciliation.
- Solving full `sgl-kernel` architecture policy for every DCU card type.
- Slash command parity beyond preserving already-validated PR-CI hooks that fit
  the target branch.

## Implementation Boundary

Implementation should not start by copying entire directories between repos.
It should first make a baseline commit of the target workspace state, then
apply PR-CI changes file by file with explicit conflict review.

The merge is complete only after static checks pass and the integrated workflow
shape matches the successful fork evidence already collected for
`tianlinyang77/sglang`.
