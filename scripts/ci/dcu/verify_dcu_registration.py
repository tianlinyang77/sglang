from __future__ import annotations

import ast
import glob
import importlib.util
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_ci_register():
    path = os.path.join(REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py")
    spec = importlib.util.spec_from_file_location("dcu_ci_register", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ci_register from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_dcu_suites():
    run_suite_path = os.path.join(REPO_ROOT, "test", "run_suite.py")
    with open(run_suite_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    per_commit = set()
    nightly = set()

    def extract_from_dict(node: ast.Dict) -> set[str]:
        suites = set()
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Attribute):
                continue
            if not isinstance(key.value, ast.Name) or key.value.id != "HWBackend":
                continue
            if key.attr != "DCU" or not isinstance(value, ast.List):
                continue
            suites.update(
                item.value for item in value.elts if isinstance(item, ast.Constant)
            )
        return suites

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not stmt.targets or not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        if name == "PER_COMMIT_SUITES" and isinstance(stmt.value, ast.Dict):
            per_commit = extract_from_dict(stmt.value)
        elif name == "NIGHTLY_SUITES" and isinstance(stmt.value, ast.Dict):
            nightly = extract_from_dict(stmt.value)

    return per_commit, nightly


def main() -> int:
    ci_register = _load_ci_register()
    per_commit_suites, nightly_suites = _extract_dcu_suites()

    if not per_commit_suites and not nightly_suites:
        print("ERROR: no DCU suites configured in test/run_suite.py", file=sys.stderr)
        return 2

    registered_dir = os.path.join(REPO_ROOT, "test", "registered")
    files = [
        path
        for path in glob.glob(os.path.join(registered_dir, "**", "*.py"), recursive=True)
        if not path.endswith(("conftest.py", "__init__.py"))
    ]

    tests = ci_register.collect_tests(files, sanity_check=False)
    dcu_tests = [test for test in tests if test.backend == ci_register.HWBackend.DCU]

    print("Configured DCU suites:")
    print(f"  per-commit ({len(per_commit_suites)}):")
    for suite in sorted(per_commit_suites):
        print(f"    - {suite}")
    print(f"  nightly    ({len(nightly_suites)}):")
    for suite in sorted(nightly_suites):
        print(f"    - {suite}")
    print()

    print(f"Collected {len(dcu_tests)} DCU registered test file(s):")
    by_suite: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for test in dcu_tests:
        rel_path = os.path.relpath(test.filename, REPO_ROOT)
        by_suite[(test.suite, test.nightly)].append(
            f"{rel_path} (est={test.est_time}s, disabled={test.disabled})"
        )

    for suite, nightly in sorted(by_suite):
        print(f"  [{suite}, nightly={nightly}]")
        for line in sorted(by_suite[(suite, nightly)]):
            print(f"    - {line}")
    print()

    known_suites = per_commit_suites | nightly_suites
    unknown_suites = sorted({suite for suite, _ in by_suite if suite not in known_suites})
    if unknown_suites:
        print("ERROR: unknown DCU suite(s):", file=sys.stderr)
        for suite in unknown_suites:
            print(f"  - {suite}", file=sys.stderr)
        return 3

    if not dcu_tests:
        print("ERROR: no DCU registered tests collected", file=sys.stderr)
        return 4

    print("OK: DCU registration looks healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
