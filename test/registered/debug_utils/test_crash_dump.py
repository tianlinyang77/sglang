import glob
import os
import pickle
import tempfile
import time
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

register_cuda_ci(est_time=40, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=40, suite="nightly-amd-1-gpu", nightly=True)
register_dcu_ci(est_time=40, suite="nightly-dcu", nightly=True)

DEFAULT_DCU_CRASH_DUMP_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class TestCrashDump(CustomTestCase):
    crash_dump_folder = None
    MAX_NEW_TOKENS = 4
    NUM_REQUESTS_BEFORE_CRASH = 5

    @classmethod
    def setUpClass(cls):
        cls.crash_dump_folder = tempfile.mkdtemp(prefix="crash_dump_test_")

        with envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.override(
            cls.NUM_REQUESTS_BEFORE_CRASH * cls.MAX_NEW_TOKENS + 10
        ):
            if is_dcu() or is_in_dcu_ci():
                model = os.environ.get(
                    "SGLANG_DCU_CRASH_DUMP_MODEL",
                    os.environ.get("SGLANG_DCU_SERVER_SMOKE_MODEL", DEFAULT_DCU_CRASH_DUMP_MODEL),
                )
                other_args = [
                    "--crash-dump-folder",
                    cls.crash_dump_folder,
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
                    "--crash-dump-folder",
                    cls.crash_dump_folder,
                    "--skip-server-warmup",
                ]
                env = None
            cls.process = popen_launch_server(
                model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                env=env,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_crash_dump_generated(self):
        """Test that crash dump file is generated after server crash."""
        # Send multiple requests to trigger the crash
        for i in range(self.NUM_REQUESTS_BEFORE_CRASH * 2):
            try:
                response = requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": f"Hello, this is request {i}.",
                        "sampling_params": {
                            "max_new_tokens": self.MAX_NEW_TOKENS,
                            "temperature": 0,
                        },
                    },
                    timeout=30,
                )
            except requests.exceptions.RequestException:
                # Connection error expected after crash
                pass

        # Wait for crash dump to be written
        time.sleep(5)

        # Find the crash dump file
        dump_pattern = os.path.join(self.crash_dump_folder, "*", "crash_dump_*.pkl")
        dump_files = glob.glob(dump_pattern)

        # Check that a dump file was created
        self.assertTrue(
            len(dump_files) > 0,
            f"No crash dump file found in {self.crash_dump_folder}. "
            f"Pattern: {dump_pattern}",
        )

        # Read the dump file and verify contents
        dump_file = dump_files[0]
        with open(dump_file, "rb") as f:
            dump_data = pickle.load(f)

        # Verify the dump structure
        self.assertIn("server_args", dump_data)
        self.assertIn("requests", dump_data)

        # Check that there are more than 5 requests in the dump
        requests_list = dump_data["requests"]
        self.assertGreater(
            len(requests_list),
            self.NUM_REQUESTS_BEFORE_CRASH,
            f"Expected more than {self.NUM_REQUESTS_BEFORE_CRASH} requests in dump, but got {len(requests_list)}",
        )

        # Verify each request tuple has the expected structure (obj, out, created_time, finish_time)
        for i, req_tuple in enumerate(requests_list):
            self.assertIsInstance(
                req_tuple,
                tuple,
                f"Request {i} should be a tuple, got {type(req_tuple)}",
            )
            self.assertGreaterEqual(
                len(req_tuple),
                4,
                f"Request {i} tuple should have at least 4 elements",
            )


if __name__ == "__main__":
    unittest.main()
