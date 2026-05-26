import os

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=6, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=47, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(
    est_time=47,
    suite="stage-b-dcu",
)

DEFAULT_DCU_IGNORE_EOS_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class TestOpenAIServerIgnoreEOS(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_dcu() or is_in_dcu_ci():
            cls.model = os.environ.get(
                "SGLANG_DCU_IGNORE_EOS_MODEL",
                os.environ.get(
                    "SGLANG_DCU_SERVER_SMOKE_MODEL", DEFAULT_DCU_IGNORE_EOS_MODEL
                ),
            )
            other_args = (
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
            )
            env = {
                "SGLANG_USE_MODELSCOPE": "1",
                "SGLANG_USE_LIGHTOP": "1",
            }
        else:
            cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            other_args = None
            env = None
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
            env=env,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ignore_eos(self):
        """
        Test that ignore_eos=True allows generation to continue beyond EOS token
        and reach the max_tokens limit.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        max_tokens = 200

        response_default = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": False},
        )

        response_ignore_eos = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )

        default_tokens = len(
            self.tokenizer.encode(response_default.choices[0].message.content)
        )
        ignore_eos_tokens = len(
            self.tokenizer.encode(response_ignore_eos.choices[0].message.content)
        )

        # Check if ignore_eos resulted in more tokens or exactly max_tokens
        # The ignore_eos response should either:
        # 1. Have more tokens than the default response (if default stopped at EOS before max_tokens)
        # 2. Have exactly max_tokens (if it reached the max_tokens limit)
        self.assertTrue(
            ignore_eos_tokens > default_tokens or ignore_eos_tokens >= max_tokens,
            f"ignore_eos did not generate more tokens: {ignore_eos_tokens} vs {default_tokens}",
        )

        self.assertEqual(
            response_ignore_eos.choices[0].finish_reason,
            "length",
            f"Expected finish_reason='length' for ignore_eos=True, got {response_ignore_eos.choices[0].finish_reason}",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
