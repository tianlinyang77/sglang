# DCU PR-CI Testcase Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the validated DCU PR-CI control flow into the testcase-updated `/public/home/tianly/sgl/sglang` workspace without losing current testcase state.

**Architecture:** Use `/public/home/tianly/sgl/sglang` as the integration target. First commit the current testcase worktree as a baseline, then apply the PR-CI donor changes from `/public/home/tianly/sgl/sglang-tly/.worktrees/dcu-pr-flow-smoke` file by file, preserving target suite names and testcase ownership.

**Tech Stack:** Git, GitHub Actions YAML, Bash CI scripts, Python test registry, SGLang `test/run_suite.py`, DCU registered tests.

---

### Task 1: Protect Current Testcase State

**Files:**
- Modify: current git index only
- Commit: all current testcase workspace changes in `/public/home/tianly/sgl/sglang`

- [ ] **Step 1: Confirm branch**

Run:

```bash
git status --short --branch
```

Expected: branch is `dcu-pr-ci-testcase-integration`.

- [ ] **Step 2: Commit testcase baseline**

Run:

```bash
git add -A
git commit -m "test: baseline dcu testcase integration state"
```

Expected: a baseline commit protects the current testcase worktree before PR-CI flow changes.

### Task 2: Reconcile PR-CI Workflow

**Files:**
- Modify: `.github/workflows/pr-test-dcu.yml`
- Check: `.github/workflows/pr-gate.yml`
- Check: `.github/workflows/rerun-ut-dcu.yml`
- Check: `.github/workflows/nightly-test-dcu.yml`

- [ ] **Step 1: Compare target and donor workflows**

Run:

```bash
diff -u .github/workflows/pr-test-dcu.yml /public/home/tianly/sgl/sglang-tly/.worktrees/dcu-pr-flow-smoke/.github/workflows/pr-test-dcu.yml
```

Expected: differences include target suite names and donor gate/path-filter behavior.

- [ ] **Step 2: Apply donor control-flow behavior**

Edit `.github/workflows/pr-test-dcu.yml` so it keeps target suite names but includes:

```yaml
resource-gate-jobs: check-changes,pr-gate,call-gate,validate-config,PR Test (DCU) finish
```

and uses inline `git diff` path filtering instead of `dorny/paths-filter@v3`.

- [ ] **Step 3: Preserve target suite names**

Ensure workflow_dispatch supports the target names:

```text
stage-a-test-1-gpu-small-dcu
stage-b-test-1-gpu-small-dcu
```

Expected: workflow can run target Stage A and Stage B suites.

### Task 3: Reconcile CI Scripts and Registry

**Files:**
- Modify if needed: `scripts/ci/dcu/dcu_ci_exec.sh`
- Modify if needed: `scripts/ci/dcu/dcu_ci_start_container.sh`
- Modify if needed: `scripts/ci/dcu/verify_dcu_registration.py`
- Modify if needed: `scripts/ci/dcu/analyze_dcu_csv_coverage.py`
- Modify if needed: `python/sglang/test/ci/ci_register.py`
- Modify if needed: `test/run_suite.py`

- [ ] **Step 1: Compare donor files**

Run:

```bash
for f in scripts/ci/dcu/dcu_ci_exec.sh scripts/ci/dcu/dcu_ci_start_container.sh scripts/ci/dcu/verify_dcu_registration.py scripts/ci/dcu/analyze_dcu_csv_coverage.py python/sglang/test/ci/ci_register.py test/run_suite.py; do
  diff -u "$f" "/public/home/tianly/sgl/sglang-tly/.worktrees/dcu-pr-flow-smoke/$f" || true
done
```

Expected: identify flow-only improvements that are missing from target.

- [ ] **Step 2: Keep target testcase semantics**

When conflicts appear, preserve target values for:

```text
stage-a-test-1-gpu-small-dcu
stage-b-test-1-gpu-small-dcu
nightly-dcu-accuracy
nightly-dcu-vlm
```

Expected: target testcase suite state remains authoritative.

- [ ] **Step 3: Import only flow improvements**

Bring in missing donor behavior only when it supports PR flow:

```text
config validation
matrix partition arguments
container cleanup
E2E evidence compatibility
slash/rerun hooks where already present
```

Expected: no broad testcase enabled/disabled state is overwritten by donor state.

### Task 4: Reconcile Docs

**Files:**
- Create or modify: `docs/dcu-ci-e2e-results.md`
- Keep external reference: `/public/home/tianly/sgl/DCU_PR演进与验证状态.md`

- [ ] **Step 1: Copy PR-CI evidence if missing**

If `docs/dcu-ci-e2e-results.md` is missing, copy it from the donor branch.

- [ ] **Step 2: Add testcase status reference**

Ensure the doc states that testcase counts and enabled/disabled state come from:

```text
/public/home/tianly/sgl/DCU_PR演进与验证状态.md
```

Expected: PR-CI flow evidence and testcase status are not mixed into one ambiguous source.

### Task 5: Validate Integration

**Files:**
- Check: `.github/workflows/pr-test-dcu.yml`
- Check: `test/run_suite.py`
- Check: `scripts/ci/dcu/verify_dcu_registration.py`

- [ ] **Step 1: Run registration check**

Run:

```bash
python3 scripts/ci/dcu/verify_dcu_registration.py
```

Expected: `OK: DCU registration looks healthy.`

- [ ] **Step 2: Compile run_suite**

Run:

```bash
python3 -m py_compile test/run_suite.py
```

Expected: exit code `0`.

- [ ] **Step 3: Parse workflow YAML**

Run:

```bash
python3 -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/pr-test-dcu.yml').read_text())"
```

Expected: exit code `0`.

- [ ] **Step 4: Check whitespace**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

### Task 6: Commit PR-CI Integration

**Files:**
- Commit all reconciled PR-CI changes after validation

- [ ] **Step 1: Review diff**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected: diff contains PR-CI integration changes after the testcase baseline commit.

- [ ] **Step 2: Commit integration**

Run:

```bash
git add -A
git commit -m "ci: integrate dcu pr flow with testcase baseline"
```

Expected: second commit contains PR-CI reconciliation only.
