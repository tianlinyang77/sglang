import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
    popen_launch_server,
)
register_dcu_ci(est_time=120, suite="stage-b-dcu", disabled="DCU PR baseline deferred: core/server path needs BW1000 model-runtime repeat validation before required CI.")

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

DEFAULT_DCU_PAGE_SIZE_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)
DCU_PAGE_SIZE_MODEL_ENV = "SGLANG_DCU_PAGE_SIZE_MODEL"


def _is_dcu_path():
    return is_dcu() or is_in_dcu_ci()


class TestPageSize(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_DEBUG_MEMORY_POOL"] = "1"
        cls.model = (
            os.environ.get(DCU_PAGE_SIZE_MODEL_ENV, DEFAULT_DCU_PAGE_SIZE_MODEL)
            if _is_dcu_path()
            else DEFAULT_MODEL_NAME_FOR_TEST
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--page-size", 4, "--chunked-prefill-size", 128]
        env = None
        if _is_dcu_path():
            other_args = [
                "--attention-backend",
                "fa3",
                "--page-size",
                "64",
                "--trust-remote-code",
                "--disable-cuda-graph",
                "--context-length",
                "2048",
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "8",
                "--chunked-prefill-size",
                "2048",
            ]
            env = {
                "SGLANG_USE_MODELSCOPE": "1",
                "SGLANG_USE_LIGHTOP": "1",
            }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_generate_with_configured_page_size(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Name one programming language.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()["text"].strip()), 0)

    def test_mmlu(self):
        if _is_dcu_path():
            self.skipTest("DCU quick framework skips inherited MMLU accuracy path.")
        MMLUMixin.test_mmlu(self)


if __name__ == "__main__":
    unittest.main()
