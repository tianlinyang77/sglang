import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)


register_dcu_ci(est_time=300, suite="nightly-dcu", nightly=True)


DEFAULT_DCU_SERVER_SMOKE_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)
MODEL_ENV = "SGLANG_DCU_SERVER_SMOKE_MODEL"


class TestDCUSGLangServerSmoke(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(MODEL_ENV, DEFAULT_DCU_SERVER_SMOKE_MODEL)
        if not os.path.isdir(cls.model):
            if MODEL_ENV in os.environ:
                raise FileNotFoundError(f"{MODEL_ENV} does not exist: {cls.model}")
            raise unittest.SkipTest(
                f"Default DCU server smoke model does not exist: {cls.model}"
            )

        port = find_available_port(11001)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
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
            ],
            env={
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "SGLANG_USE_MODELSCOPE": "1",
                "SGLANG_USE_LIGHTOP": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_health(self):
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)

    def test_models(self):
        response = requests.get(f"{self.base_url}/v1/models", timeout=10)
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertTrue(any(model["id"] == self.model for model in data))

    def test_chat_completion(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, reply with one short sentence.",
                    }
                ],
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
