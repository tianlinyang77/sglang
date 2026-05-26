import unittest
import os
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=66, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=66, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(
    est_time=66,
    suite="stage-b-dcu",
)

DEFAULT_DCU_PYTORCH_SAMPLING_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class TestPyTorchSamplingBackend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_dcu() or is_in_dcu_ci():
            cls.model = os.environ.get(
                "SGLANG_DCU_PYTORCH_SAMPLING_MODEL",
                os.environ.get(
                    "SGLANG_DCU_SERVER_SMOKE_MODEL",
                    DEFAULT_DCU_PYTORCH_SAMPLING_MODEL,
                ),
            )
            other_args = [
                "--sampling-backend",
                "pytorch",
                "--disable-radix-cache",
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
            cls.model = DEFAULT_MODEL_NAME_FOR_TEST
            other_args = ["--sampling-backend", "pytorch", "--disable-radix-cache"]
            env = None
        cls.base_url = DEFAULT_URL_FOR_TEST
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

    def test_mmlu(self):
        if is_dcu() or is_in_dcu_ci():
            self.skipTest("DCU stage-b skips MMLU accuracy; covered by accuracy suites.")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)

    def test_greedy(self):

        first_text = None

        # ensure the answer is identical across single response
        for _ in range(5):
            response_single = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of Germany is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            ).json()
            text = response_single["text"]
            if first_text is None:
                first_text = text

            self.assertEqual(text, first_text)

        first_text = None

        response_batch = requests.post(
            self.base_url + "/generate",
            json={
                "text": ["The capital of Germany is"] * 10,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        ).json()

        # ensure the answer is identical among the batch
        for i in range(10):
            text = response_batch[i]["text"]
            if first_text is None:
                first_text = text
            self.assertEqual(text, first_text)


if __name__ == "__main__":
    unittest.main()
