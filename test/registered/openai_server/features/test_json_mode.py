import json
import os
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_amd_ci,
    is_in_dcu_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=109, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(
    est_time=180,
    suite="stage-b-dcu",
)

DEFAULT_DCU_JSON_MODE_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class JSONModeMixin:
    """Mixin class containing JSON mode test methods"""

    def test_json_mode_response(self):
        """Test that response_format json_object (also known as "json mode") produces valid JSON, even without a system prompt that mentions JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content

        print(f"Response ({len(text)} characters): {text}")

        # Verify the response is valid JSON
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            self.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")

        # Verify it's actually an object (dict)
        self.assertIsInstance(js_obj, dict, f"Response is not a JSON object: {text}")

    def test_json_mode_with_streaming(self):
        """Test that streaming with json_object response (also known as "json mode") format works correctly, even without a system prompt that mentions JSON."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream=True,
        )

        # Collect all chunks
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_response = "".join(chunks)

        print(
            f"Concatenated Response ({len(full_response)} characters): {full_response}"
        )

        # Verify the combined response is valid JSON
        try:
            js_obj = json.loads(full_response)
        except json.JSONDecodeError as e:
            self.fail(
                f"Streamed response is not valid JSON. Error: {e}. Response: {full_response}"
            )

        self.assertIsInstance(js_obj, dict)


class ServerWithGrammarBackend(CustomTestCase):
    """Base class for tests requiring a grammar backend server"""

    backend = "xgrammar"

    @classmethod
    def setUpClass(cls):
        if is_dcu() or is_in_dcu_ci():
            cls.model = os.environ.get(
                "SGLANG_DCU_JSON_MODE_MODEL",
                os.environ.get(
                    "SGLANG_DCU_SERVER_SMOKE_MODEL", DEFAULT_DCU_JSON_MODE_MODEL
                ),
            )
            other_args = [
                "--max-running-requests",
                "10",
                "--grammar-backend",
                cls.backend,
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
                "--chunked-prefill-size",
                "2048",
            ]
            env = {
                "SGLANG_USE_MODELSCOPE": "1",
                "SGLANG_USE_LIGHTOP": "1",
            }
        else:
            cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            other_args = [
                "--max-running-requests",
                "10",
                "--grammar-backend",
                cls.backend,
            ]
            env = None
        cls.base_url = DEFAULT_URL_FOR_TEST

        if is_in_amd_ci():
            other_args.append("--constrained-json-disable-any-whitespace")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestJSONModeXGrammar(ServerWithGrammarBackend, JSONModeMixin):
    backend = "xgrammar"


class TestJSONModeOutlines(ServerWithGrammarBackend, JSONModeMixin):
    backend = "outlines"


class TestJSONModeLLGuidance(ServerWithGrammarBackend, JSONModeMixin):
    backend = "llguidance"


if __name__ == "__main__":
    unittest.main()
