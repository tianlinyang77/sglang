import io
import os
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=120, suite="nightly-amd-1-gpu", nightly=True)
register_dcu_ci(est_time=120, suite="nightly-dcu", nightly=True)

DEFAULT_DCU_SOFT_WATCHDOG_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class BaseTestSoftWatchdog:
    env_override = None
    expected_message = None

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        with cls.env_override():
            if is_dcu() or is_in_dcu_ci():
                model = os.environ.get(
                    "SGLANG_DCU_SOFT_WATCHDOG_MODEL",
                    os.environ.get(
                        "SGLANG_DCU_SERVER_SMOKE_MODEL",
                        DEFAULT_DCU_SOFT_WATCHDOG_MODEL,
                    ),
                )
                other_args = [
                    "--soft-watchdog-timeout",
                    "20",
                    "--skip-server-warmup",
                    "--attention-backend",
                    "fa3",
                    "--page-size",
                    "64",
                    "--trust-remote-code",
                    "--disable-cuda-graph",
                    "--context-length",
                    "2048",
                    "--max-total-tokens",
                    "4096",
                    "--max-running-requests",
                    "8",
                    "--chunked-prefill-size",
                    "2048",
                ]
                env = {
                    "SGLANG_USE_MODELSCOPE": "1",
                    "SGLANG_USE_LIGHTOP": "1",
                    **os.environ,
                }
            else:
                model = "Qwen/Qwen3-0.6B"
                other_args = [
                    "--soft-watchdog-timeout",
                    "20",
                    "--skip-server-warmup",
                ]
                env = None
            cls.process = popen_launch_server(
                model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(cls.stdout, cls.stderr),
                env=env,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_watchdog_triggers(self):
        print("Start call /generate API", flush=True)
        try:
            requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "Hello, please repeat this sentence for 1000 times.",
                    "sampling_params": {"max_new_tokens": 100, "temperature": 0},
                },
                timeout=30,
            )
        except requests.exceptions.ReadTimeout as e:
            print(f"requests.post timeout (but expected): {e}")
        print("End call /generate API", flush=True)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(self.expected_message, combined_output)


class TestSoftWatchdogDetokenizer(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_DETOKENIZER.override(30)
    expected_message = "DetokenizerManager watchdog timeout"


class TestSoftWatchdogTokenizer(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_TOKENIZER.override(30)
    expected_message = "TokenizerManager watchdog timeout"


class TestSoftWatchdogSchedulerInit(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.override(30)
    expected_message = "Scheduler watchdog timeout"


if __name__ == "__main__":
    unittest.main()
