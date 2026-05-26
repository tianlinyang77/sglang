import unittest
import os

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
)

# RadixAttention server integration tests
register_cuda_ci(est_time=100, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=100, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(
    est_time=100,
    suite="stage-b-dcu",
    disabled=(
        "DCU PR baseline deferred: timed out after 900s on BW1000 "
        "sgl-test stage-b-dcu partition 0; keep in nightly/manual until "
        "radix-cache server integration is repeatable within PR budget."
    ),
)

DEFAULT_DCU_RADIX_ATTENTION_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


def _is_dcu_test():
    return is_dcu() or is_in_dcu_ci()


def _get_model():
    if _is_dcu_test():
        return os.environ.get(
            "SGLANG_DCU_RADIX_ATTENTION_MODEL",
            os.environ.get(
                "SGLANG_DCU_SERVER_SMOKE_MODEL",
                DEFAULT_DCU_RADIX_ATTENTION_MODEL,
            ),
        )
    return DEFAULT_SMALL_MODEL_NAME_FOR_TEST


def _get_launch_args(*base_args: str):
    launch_args = list(base_args)
    env = None
    if _is_dcu_test():
        launch_args += [
            "--attention-backend",
            "fa3",
            "--page-size",
            "64",
            "--trust-remote-code",
            "--disable-cuda-graph",
            "--context-length",
            "2048",
        ]
        env = {
            "SGLANG_USE_MODELSCOPE": "1",
            "SGLANG_USE_LIGHTOP": "1",
            **os.environ,
        }
    return launch_args, env


class TestRadixCacheFCFS(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model()
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args, env = _get_launch_args(
            "--chunked-prefill-size",
            "128",
            "--max-total-tokens",
            "4096" if _is_dcu_test() else "20000",
            "--schedule-policy",
            "fcfs",
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestRadixCacheLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model()
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args, env = _get_launch_args(
            "--chunked-prefill-size",
            "128",
            "--max-total-tokens",
            "4096" if _is_dcu_test() else "20000",
            "--schedule-policy",
            "lpm",
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )


class TestRadixCacheNonOverlapLPM(TestRadixCacheFCFS):
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model()
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args, env = _get_launch_args(
            "--disable-overlap-schedule",
            "--chunked-prefill-size",
            "128",
            "--max-total-tokens",
            "4096" if _is_dcu_test() else "20000",
            "--schedule-policy",
            "lpm",
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )


if __name__ == "__main__":
    envs.SGLANG_TEST_RETRACT.set(True)
    envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.set(1)
    unittest.main()
