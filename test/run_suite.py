import argparse
import glob
import importlib.util
import os
import sys
from typing import List, Optional, Tuple

try:
    import tabulate
except ModuleNotFoundError:

    class _TabulateFallback:
        @staticmethod
        def tabulate(rows, headers=(), tablefmt=None):
            table = [list(headers)] if headers else []
            table.extend(rows)
            if not table:
                return ""
            widths = [
                max(len(str(row[i])) for row in table)
                for i in range(max(len(row) for row in table))
            ]

            def fmt(row):
                return " | ".join(
                    str(row[i]).ljust(widths[i]) for i in range(len(widths))
                )

            lines = [fmt(row) for row in table]
            if headers:
                lines.insert(1, "-+-".join("-" * width for width in widths))
            return "\n".join(lines)

    tabulate = _TabulateFallback()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def _load_ci_register():
    path = os.path.join(
        REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
    )
    spec = importlib.util.spec_from_file_location("run_suite_ci_register", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ci_register from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ci_utils():
    python_dir = os.path.join(REPO_ROOT, "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    from sglang.test.ci.ci_utils import TestFile, run_unittest_files

    return TestFile, run_unittest_files


_ci_register = _load_ci_register()
CIRegistry = _ci_register.CIRegistry
HWBackend = _ci_register.HWBackend
auto_partition = _ci_register.auto_partition
collect_tests = _ci_register.collect_tests

HW_MAPPING = {
    "cpu": HWBackend.CPU,
    "cuda": HWBackend.CUDA,
    "amd": HWBackend.AMD,
    "npu": HWBackend.NPU,
    "dcu": HWBackend.DCU,
}

# Per-commit test suites (run on every PR)
PER_COMMIT_SUITES = {
    HWBackend.CPU: ["stage-a-test-cpu"],
    HWBackend.AMD: [
        "stage-a-test-1-gpu-small-amd",
        "stage-b-test-1-gpu-small-amd",
        "stage-b-test-1-gpu-small-amd-nondeterministic",
        "stage-b-test-1-gpu-small-amd-mi35x",
        "stage-b-test-large-8-gpu-35x-disaggregation-amd",
        "stage-b-test-1-gpu-large-amd",
        "stage-b-test-2-gpu-large-amd",
        "stage-c-test-4-gpu-amd",
        "stage-c-test-large-8-gpu-amd",
        "stage-c-test-large-8-gpu-amd-mi35x",
    ],
    HWBackend.CUDA: [
        "stage-a-test-1-gpu-small",
        "stage-b-test-1-gpu-small",
        "stage-b-test-1-gpu-large",
        "stage-b-test-2-gpu-large",
        "stage-b-test-4-gpu-b200",
        "stage-b-kernel-unit-1-gpu-large",
        "stage-b-kernel-unit-8-gpu-h200",
        "stage-b-kernel-benchmark-1-gpu-large",
        "stage-c-test-4-gpu-h100",
        "stage-c-test-4-gpu-b200",
        "stage-c-test-4-gpu-gb200",
        "stage-c-test-8-gpu-h20",
        "stage-c-test-8-gpu-h200",
        "stage-c-test-8-gpu-b200",
        "stage-c-test-deepep-4-gpu-h100",
        "stage-c-test-deepep-8-gpu-h200",
    ],
    HWBackend.NPU: [
        "stage-a-test-1-gpu-small",
        "stage-b-test-1-npu-a2",
        "stage-b-test-2-npu-a2",
        "stage-b-test-4-npu-a3",
        "stage-b-test-16-npu-a3",
    ],
    HWBackend.DCU: [
        "stage-a-test-1-gpu-small-dcu",
        "stage-b-test-1-gpu-small-dcu",
        "stage-b-test-1-gpu-large-dcu",
        "stage-b-test-2-gpu-large-dcu",
        "stage-c-test-large-8-gpu-dcu",
    ],
}

# Nightly test suites (run nightly, organized by GPU configuration)
NIGHTLY_SUITES = {
    HWBackend.CUDA: [
        "nightly-1-gpu",
        "nightly-2-gpu",
        "nightly-4-gpu",
        "nightly-4-gpu-b200",
        "nightly-8-gpu",
        "nightly-8-gpu-h200",
        "nightly-8-gpu-h20",
        "nightly-8-gpu-b200",
        "nightly-8-gpu-h200-basic",  # Basic tests for large models on H200
        "nightly-8-gpu-b200-basic",  # Basic tests for large models on B200
        "nightly-8-gpu-common",  # Common tests that run on both H200 and B200
        "nightly-kernel-1-gpu",
        "nightly-kernel-8-gpu-h200",
        # Eval and perf suites (2-gpu)
        "nightly-eval-text-2-gpu",
        "nightly-eval-vlm-2-gpu",
        "nightly-perf-text-2-gpu",
        "nightly-perf-vlm-2-gpu",
    ],
    HWBackend.AMD: [
        "nightly-amd",
        "nightly-amd-1-gpu",
        "nightly-amd-1-gpu-mi35x",
        "nightly-amd-1-gpu-zimage-turbo",
        "nightly-amd-4-gpu",
        "nightly-amd-8-gpu",
        "nightly-amd-vlm",
        # MI35x 8-GPU suite (different model configs)
        "nightly-amd-8-gpu-mi35x",
    ],
    HWBackend.CPU: [],
    HWBackend.NPU: [
        "nightly-1-npu-a3",
        "nightly-2-npu-a3",
        "nightly-4-npu-a3",
        "nightly-8-npu-a3",
        "nightly-16-npu-a3",
    ],
    HWBackend.DCU: [
        "nightly-dcu",
        "nightly-dcu-1-gpu",
        "nightly-dcu-4-gpu",
        "nightly-dcu-8-gpu",
        "nightly-dcu-accuracy",
        "nightly-dcu-perf",
        "nightly-dcu-vlm",
    ],
}


def filter_tests(
    ci_tests: List[CIRegistry], hw: HWBackend, suite: str, nightly: bool = False
) -> List[CIRegistry]:
    ci_tests = [
        t
        for t in ci_tests
        if t.backend == hw and t.suite == suite and t.nightly == nightly
    ]

    valid_suites = (
        NIGHTLY_SUITES.get(hw, []) if nightly else PER_COMMIT_SUITES.get(hw, [])
    )

    if suite not in valid_suites:
        print(
            f"Warning: Unknown suite {suite} for backend {hw.name}, nightly={nightly}"
        )

    enabled_tests = [t for t in ci_tests if t.disabled is None]
    skipped_tests = [t for t in ci_tests if t.disabled is not None]

    return enabled_tests, skipped_tests


def _include_file_keys(filename: str, repo_root: str, test_root: str) -> set[str]:
    filename = os.path.normpath(os.path.abspath(filename))
    keys = {filename}

    for root in (repo_root, test_root):
        root = os.path.normpath(os.path.abspath(root))
        relpath = os.path.normpath(os.path.relpath(filename, root))
        if relpath != os.pardir and not relpath.startswith(os.pardir + os.sep):
            keys.add(relpath)

    return keys


def _matches_include_file(
    test: CIRegistry, include_file: str, repo_root: str, test_root: str
) -> bool:
    include_file = os.path.normpath(include_file)
    return include_file in _include_file_keys(test.filename, repo_root, test_root)


def filter_include_files(
    ci_tests: List[CIRegistry],
    skipped_tests: List[CIRegistry],
    include_files: List[str],
    repo_root: str,
    test_root: str,
) -> Tuple[List[CIRegistry], List[CIRegistry], Optional[str]]:
    include_files = [os.path.normpath(path) for path in include_files if path]
    if not include_files:
        return ci_tests, skipped_tests, None

    selected_tests = [
        test
        for test in ci_tests
        if any(
            _matches_include_file(test, include_file, repo_root, test_root)
            for include_file in include_files
        )
    ]
    selected_skipped_tests = [
        test
        for test in skipped_tests
        if any(
            _matches_include_file(test, include_file, repo_root, test_root)
            for include_file in include_files
        )
    ]

    errors = []
    all_suite_tests = ci_tests + skipped_tests
    for include_file in include_files:
        matched_tests = [
            test
            for test in all_suite_tests
            if _matches_include_file(test, include_file, repo_root, test_root)
        ]
        if not matched_tests:
            errors.append(f"{include_file} was not found in the selected suite.")
            continue

        disabled_tests = [test for test in matched_tests if test.disabled is not None]
        for test in disabled_tests:
            errors.append(f"{include_file} is disabled: {test.disabled}")

    return selected_tests, selected_skipped_tests, "\n".join(errors) if errors else None


def pretty_print_tests(
    args, ci_tests: List[CIRegistry], skipped_tests: List[CIRegistry]
):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} "
            f"(0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    headers = ["Hardware", "Suite", "Nightly", "Partition"]
    rows = [[hw.name, suite, str(nightly), partition_info]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"

    if skipped_tests:
        msg += f"⚠️  Skipped {len(skipped_tests)} test(s):\n"
        for t in skipped_tests:
            reason = t.disabled or "disabled"
            msg += f"  - {t.filename} (reason: {reason})\n"
        msg += "\n"

    if len(ci_tests) == 0:
        msg += f"No tests found for hw={hw.name}, suite={suite}, nightly={nightly}\n"
        msg += "This is expected during incremental migration. Skipping.\n"
    else:
        total_est_time = sum(t.est_time for t in ci_tests)
        msg += (
            f"✅ Enabled {len(ci_tests)} test(s) (est total {total_est_time:.1f}s):\n"
        )
        for t in ci_tests:
            msg += f"  - {t.filename} (est_time={t.est_time})\n"

    print(msg, flush=True)


def run_a_suite(args):
    hw = HW_MAPPING[args.hw]
    suite = args.suite
    nightly = args.nightly
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    # Registered tests under test/registered/
    files = [
        f
        for f in glob.glob(
            os.path.join(SCRIPT_DIR, "registered", "**", "*.py"), recursive=True
        )
        if not f.endswith("/conftest.py")
        and not f.endswith("/__init__.py")
        and not f.endswith("/cpu/utils.py")
    ]

    # JIT kernel tests and benchmarks (live alongside kernel source)
    jit_kernel_dir = os.path.join(REPO_ROOT, "python", "sglang", "jit_kernel")
    files += glob.glob(
        os.path.join(jit_kernel_dir, "tests", "**", "test_*.py"), recursive=True
    )
    files += glob.glob(
        os.path.join(jit_kernel_dir, "benchmark", "**", "bench_*.py"), recursive=True
    )

    # Strict: all discovered files must have proper registration
    sanity_check = True

    all_tests = collect_tests(files, sanity_check=sanity_check)
    ci_tests, skipped_tests = filter_tests(all_tests, hw, suite, nightly)

    ci_tests, skipped_tests, include_error = filter_include_files(
        ci_tests,
        skipped_tests,
        args.include_file,
        repo_root=REPO_ROOT,
        test_root=SCRIPT_DIR,
    )
    if include_error:
        print(include_error, file=sys.stderr, flush=True)
        return 1

    if auto_partition_size:
        ci_tests = auto_partition(ci_tests, auto_partition_id, auto_partition_size)

    pretty_print_tests(args, ci_tests, skipped_tests)

    if args.list:
        return 0

    TestFile, run_unittest_files = _load_ci_utils()
    runnable_tests = [
        TestFile(name=test.filename, estimated_time=test.est_time)
        for test in ci_tests
    ]

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    return run_unittest_files(
        runnable_tests,
        timeout_per_file=timeout,
        continue_on_error=args.continue_on_error,
        enable_retry=args.enable_retry,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CI test suites from test/registered/"
    )
    parser.add_argument(
        "--hw",
        type=str,
        choices=HW_MAPPING.keys(),
        required=True,
        help="Hardware backend to run tests on.",
    )
    parser.add_argument("--suite", type=str, required=True, help="Test suite to run.")
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Run nightly tests instead of per-commit tests.",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds (default: 1200).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (default: False, useful for nightly tests).",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    parser.add_argument(
        "--include-file",
        action="append",
        default=[],
        help=(
            "Restrict the selected suite to a registered file. Can be repeated. "
            "Accepts paths relative to repo root or test/."
        ),
    )
    parser.add_argument(
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures (not code errors)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2)",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60)",
    )
    parser.add_argument(
        "--retry-timeout-increase",
        type=int,
        default=600,
        help="Additional timeout in seconds when retry is enabled (default: 600)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected test files without running them.",
    )
    args = parser.parse_args()

    # Validate auto-partition arguments
    if (args.auto_partition_id is not None) != (args.auto_partition_size is not None):
        parser.error(
            "--auto-partition-id and --auto-partition-size must be specified together."
        )
    if args.auto_partition_size is not None:
        if args.auto_partition_size <= 0:
            parser.error("--auto-partition-size must be positive.")
        if not 0 <= args.auto_partition_id < args.auto_partition_size:
            parser.error(
                f"--auto-partition-id must be in range [0, {args.auto_partition_size}), "
                f"but got {args.auto_partition_id}"
            )

    exit_code = run_a_suite(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
