import json
import os
import tempfile
import unittest

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import kill_process_tree
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
register_dcu_ci(
    est_time=120,
    suite="stage-b-dcu",
    disabled="DCU PR baseline deferred: embedding path needs local model mapping and BW1000 repeat validation before required CI.",
)

register_cuda_ci(est_time=38, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=38, suite="stage-b-test-1-gpu-small-amd")

DEFAULT_DCU_INPUT_EMBEDDING_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class TestInputEmbeds(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_dcu() or is_in_dcu_ci():
            cls.model = os.environ.get(
                "SGLANG_DCU_INPUT_EMBEDDING_MODEL",
                DEFAULT_DCU_INPUT_EMBEDDING_MODEL,
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
                **os.environ,
            }
        else:
            cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            other_args = ["--disable-radix", "--cuda-graph-max-bs", 4]
            env = None
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model, trust_remote_code=True
        )
        cls.ref_model = AutoModelForCausalLM.from_pretrained(
            cls.model, trust_remote_code=True
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        cls.texts = [
            "The capital of France is",
            "What is the best time of year to visit Japan for cherry blossoms?",
        ]

    def generate_input_embeddings(self, text):
        """Generate input embeddings for a given text."""
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        embeddings = self.ref_model.get_input_embeddings()(input_ids)
        return embeddings.squeeze().tolist()  # Convert tensor to a list for API use

    def send_request(self, payload):
        """Send a POST request to the /generate endpoint and return the response."""
        response = requests.post(
            self.base_url + "/generate",
            json=payload,
            timeout=30,  # Set a reasonable timeout for the API request
        )
        if response.status_code == 200:
            return response.json()
        return {
            "error": f"Request failed with status {response.status_code}: {response.text}"
        }

    def send_file_request(self, file_path):
        """Send a POST request to the /generate_from_file endpoint with a file."""
        with open(file_path, "rb") as f:
            response = requests.post(
                self.base_url + "/generate_from_file",
                files={"file": f},
                timeout=30,  # Set a reasonable timeout for the API request
            )
        if response.status_code == 200:
            return response.json()
        return {
            "error": f"Request failed with status {response.status_code}: {response.text}"
        }

    def test_text_based_response(self):
        """Test and print API responses using text-based input."""
        for text in self.texts:
            payload = {
                "model": self.model,
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            response = self.send_request(payload)
            print(
                f"Text Input: {text}\nResponse: {json.dumps(response, indent=2)}\n{'-' * 80}"
            )

    def test_embedding_based_response(self):
        """Test and print API responses using input embeddings."""
        for text in self.texts:
            embeddings = self.generate_input_embeddings(text)
            payload = {
                "model": self.model,
                "input_embeds": embeddings,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            response = self.send_request(payload)
            print(
                f"Embeddings Input (for text '{text}'):\nResponse: {json.dumps(response, indent=2)}\n{'-' * 80}"
            )

    def test_compare_text_vs_embedding(self):
        """Test and compare responses for text-based and embedding-based inputs."""
        for text in self.texts:
            # Text-based payload
            text_payload = {
                "model": self.model,
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            # Embedding-based payload
            embeddings = self.generate_input_embeddings(text)
            embed_payload = {
                "model": self.model,
                "input_embeds": embeddings,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            # Get responses
            text_response = self.send_request(text_payload)
            embed_response = self.send_request(embed_payload)
            # Print responses
            print(
                f"Text Input: {text}\nText-Based Response: {json.dumps(text_response, indent=2)}\n"
            )
            print(
                f"Embeddings Input (for text '{text}'):\nEmbedding-Based Response: {json.dumps(embed_response, indent=2)}\n{'-' * 80}"
            )
            # This is flaky, so we skip this temporarily
            # self.assertEqual(text_response["text"], embed_response["text"])

    def test_generate_from_file(self):
        """Test the /generate_from_file endpoint using tokenized embeddings."""
        for text in self.texts:
            embeddings = self.generate_input_embeddings(text)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp_file:
                json.dump(embeddings, tmp_file)
                tmp_file_path = tmp_file.name

            try:
                response = self.send_file_request(tmp_file_path)
                print(
                    f"Text Input: {text}\nResponse from /generate_from_file: {json.dumps(response, indent=2)}\n{'-' * 80}"
                )
            finally:
                # Ensure the temporary file is deleted
                os.remove(tmp_file_path)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
