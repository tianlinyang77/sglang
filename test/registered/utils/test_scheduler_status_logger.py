import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import requests

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
register_dcu_ci(est_time=120, suite="stage-b-dcu")

DEFAULT_DCU_SCHEDULER_STATUS_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class TestSchedulerStatusLogger(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.temp_dir)
        env = os.environ.copy()
        env["SGLANG_LOG_SCHEDULER_STATUS_TARGET"] = cls.temp_dir
        env["SGLANG_LOG_SCHEDULER_STATUS_INTERVAL"] = "1"
        model = "Qwen/Qwen3-0.6B"
        other_args = ["--skip-server-warmup", "--enable-metrics"]
        if is_dcu() or is_in_dcu_ci():
            model = DEFAULT_DCU_SCHEDULER_STATUS_MODEL
            env["SGLANG_USE_MODELSCOPE"] = "1"
            env["SGLANG_USE_LIGHTOP"] = "1"
            other_args.extend(
                [
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
            )
        cls.process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def test_scheduler_status_dump(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(2)

        events = list(_find_log_events(self.temp_dir, "scheduler.status"))
        print(f"{events=}")
        self.assertGreater(len(events), 0, "scheduler.status event not found")
        data = events[0]
        for field in ["timestamp", "rank", "running_rids", "queued_rids"]:
            self.assertIn(field, data)
        self.assertIsInstance(data["running_rids"], list)
        self.assertIsInstance(data["queued_rids"], list)


def _find_log_events(log_dir: str, event_name: str):
    for f in Path(log_dir).glob("*.log"):
        for line in f.read_text().splitlines():
            if line.startswith("{"):
                data = json.loads(line)
                if data.get("event") == event_name:
                    yield data


if __name__ == "__main__":
    unittest.main()
