import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FakeCIRegistry:
    filename: str
    est_time: float = 1.0
    disabled: str | None = None


def _load_run_suite():
    ci_register = types.ModuleType("sglang.test.ci.ci_register")
    ci_register.CIRegistry = FakeCIRegistry
    ci_register.HWBackend = types.SimpleNamespace(
        CPU="CPU",
        CUDA="CUDA",
        AMD="AMD",
        DCU="DCU",
        NPU="NPU",
    )
    ci_register.auto_partition = lambda files, rank, size: files
    ci_register.collect_tests = lambda files, sanity_check=True: []

    ci_utils = types.ModuleType("sglang.test.ci.ci_utils")
    ci_utils.run_unittest_files = lambda *args, **kwargs: 0

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.test": types.ModuleType("sglang.test"),
        "sglang.test.ci": types.ModuleType("sglang.test.ci"),
        "sglang.test.ci.ci_register": ci_register,
        "sglang.test.ci.ci_utils": ci_utils,
    }
    old_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        path = Path(__file__).resolve().parents[1] / "test" / "run_suite.py"
        spec = importlib.util.spec_from_file_location("run_suite_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_include_file_filter_accepts_repo_relative_paths():
    run_suite = _load_run_suite()
    repo_root = "/repo"
    test_root = "/repo/test"
    enabled = [
        FakeCIRegistry("/repo/test/registered/unit/a.py"),
        FakeCIRegistry("/repo/test/registered/unit/b.py"),
    ]

    selected, skipped, error = run_suite.filter_include_files(
        enabled,
        [],
        ["test/registered/unit/b.py"],
        repo_root,
        test_root,
    )

    assert error is None
    assert [test.filename for test in selected] == ["/repo/test/registered/unit/b.py"]
    assert skipped == []


def test_include_file_filter_accepts_test_relative_paths_and_hides_unselected_skips():
    run_suite = _load_run_suite()
    repo_root = "/repo"
    test_root = "/repo/test"
    enabled = [
        FakeCIRegistry("/repo/test/registered/unit/a.py"),
        FakeCIRegistry("/repo/test/registered/unit/b.py"),
    ]
    disabled = [
        FakeCIRegistry(
            "/repo/test/registered/unit/deferred.py",
            disabled="not part of flow smoke",
        )
    ]

    selected, skipped, error = run_suite.filter_include_files(
        enabled,
        disabled,
        ["registered/unit/a.py"],
        repo_root,
        test_root,
    )

    assert error is None
    assert [test.filename for test in selected] == ["/repo/test/registered/unit/a.py"]
    assert skipped == []


def test_include_file_filter_reports_disabled_or_missing_selected_files():
    run_suite = _load_run_suite()
    repo_root = "/repo"
    test_root = "/repo/test"
    disabled = [
        FakeCIRegistry(
            "/repo/test/registered/unit/deferred.py",
            disabled="not part of flow smoke",
        )
    ]

    selected, skipped, error = run_suite.filter_include_files(
        [],
        disabled,
        ["registered/unit/deferred.py", "registered/unit/missing.py"],
        repo_root,
        test_root,
    )

    assert selected == []
    assert skipped == disabled
    assert "registered/unit/deferred.py is disabled" in error
    assert "registered/unit/missing.py was not found" in error
