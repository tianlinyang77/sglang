"""
python3 -m unittest openai_server.validation.test_large_max_new_tokens.TestLargeMaxNewTokens.test_chat_completion
"""

import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=41, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=41, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(
    est_time=41,
    suite="stage-b-dcu",
)

DEFAULT_DCU_LARGE_MAX_NEW_TOKENS_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class TestLargeMaxNewTokens(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_dcu() or is_in_dcu_ci():
            cls.model = os.environ.get(
                "SGLANG_DCU_LARGE_MAX_NEW_TOKENS_MODEL",
                os.environ.get(
                    "SGLANG_DCU_SERVER_SMOKE_MODEL",
                    DEFAULT_DCU_LARGE_MAX_NEW_TOKENS_MODEL,
                ),
            )
            other_args = (
                "--max-total-tokens",
                "1536",
                "--context-length",
                "8192",
                "--decode-log-interval",
                "2",
                "--max-running-requests",
                "8",
                "--attention-backend",
                "fa3",
                "--page-size",
                "64",
                "--trust-remote-code",
                "--disable-cuda-graph",
                "--chunked-prefill-size",
                "2048",
            )
            env = {
                "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256",
                "SGLANG_USE_MODELSCOPE": "1",
                "SGLANG_USE_LIGHTOP": "1",
                **os.environ,
            }
        else:
            cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            other_args = (
                "--max-total-token",
                "1536",
                "--context-len",
                "8192",
                "--decode-log-interval",
                "2",
            )
            env = {"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256", **os.environ}
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def run_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Please repeat the world 'hello' for 10000 times.",
                },
            ],
            temperature=0,
        )
        return response

    def test_chat_completion(self):
        num_requests = 4

        futures = []
        with ThreadPoolExecutor(num_requests) as executor:
            # Send multiple requests
            for i in range(num_requests):
                futures.append(executor.submit(self.run_chat_completion))

            # Ensure that they are running concurrently
            pt = 0
            while pt >= 0:
                time.sleep(5)
                lines = open(STDERR_FILENAME).readlines()
                for line in lines[pt:]:
                    print(line, end="", flush=True)
                    if f"#running-req: {num_requests}" in line:
                        all_requests_running = True
                        pt = -1
                        break
                    pt += 1

        assert all_requests_running


if __name__ == "__main__":
    unittest.main()
