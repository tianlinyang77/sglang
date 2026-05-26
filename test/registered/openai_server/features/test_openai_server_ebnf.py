import os
import re

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

register_cuda_ci(est_time=7, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(
    est_time=20,
    suite="stage-b-dcu",
)

DEFAULT_DCU_EBNF_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


# -------------------------------------------------------------------------
#    EBNF Test Class: TestOpenAIServerEBNF
#    Launches the server with xgrammar, has only EBNF tests
# -------------------------------------------------------------------------
class TestOpenAIServerEBNF(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_dcu() or is_in_dcu_ci():
            cls.model = os.environ.get(
                "SGLANG_DCU_EBNF_MODEL",
                os.environ.get("SGLANG_DCU_SERVER_SMOKE_MODEL", DEFAULT_DCU_EBNF_MODEL),
            )
            other_args = [
                "--grammar-backend",
                "xgrammar",
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
            }
        else:
            cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            # passing xgrammar specifically
            other_args = ["--grammar-backend", "xgrammar"]
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

    def test_ebnf(self):
        """
        Ensure we can pass `ebnf` to the local openai server
        and that it enforces the grammar.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        ebnf_grammar = r"""
        root ::= "Hello" | "Hi" | "Hey"
        """
        pattern = re.compile(r"^(Hello|Hi|Hey)[.!?]*\s*$")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful EBNF test bot."},
                {"role": "user", "content": "Say a greeting (Hello, Hi, or Hey)."},
            ],
            temperature=0,
            max_tokens=32,
            extra_body={"ebnf": ebnf_grammar},
        )
        text = response.choices[0].message.content.strip()
        self.assertTrue(len(text) > 0, "Got empty text from EBNF generation")
        self.assertRegex(text, pattern, f"Text '{text}' doesn't match EBNF choices")

    def test_ebnf_strict_json(self):
        """
        A stricter EBNF that produces exactly {"name":"Alice"} format
        with no trailing punctuation or extra fields.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        ebnf_grammar = r"""
        root    ::= "{" pair "}"
        pair    ::= "\"name\"" ":" string
        string  ::= "\"" [A-Za-z]+ "\""
        """
        pattern = re.compile(r'^\{"name":"[A-Za-z]+"\}$')

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "EBNF mini-JSON generator."},
                {
                    "role": "user",
                    "content": "Generate single key JSON with only letters.",
                },
            ],
            temperature=0,
            max_tokens=64,
            extra_body={"ebnf": ebnf_grammar},
        )
        text = response.choices[0].message.content.strip()
        self.assertTrue(len(text) > 0, "Got empty text from EBNF strict JSON test")
        self.assertRegex(
            text, pattern, f"Text '{text}' not matching the EBNF strict JSON shape"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
