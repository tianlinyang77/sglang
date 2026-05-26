"""
Performance tests for single GPU - LLM throughput/latency and LoRA tests.
Works on 5090 (32GB).
"""

import asyncio
import itertools
import os
import unittest

import requests

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    is_in_dcu_ci,
    run_bench_serving,
    write_github_step_summary,
)
register_dcu_ci(
    est_time=120,
    suite="stage-b-dcu",
    disabled="DCU PR baseline deferred: performance path needs BW1000 baseline and thresholds before required CI.",
)

register_cuda_ci(est_time=1000, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=1100, suite="stage-b-test-1-gpu-large-amd")

DCU_MODEL = os.environ.get(
    "SGLANG_DCU_SMALL_MODEL",
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct",
)


def _dcu_server_args(extra_args=None):
    args = [
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--disable-cuda-graph",
        "--context-length",
        "4096",
        "--max-total-tokens",
        "8192",
        "--max-running-requests",
        "8",
    ]
    if extra_args:
        args.extend(extra_args)
    return args


def _dcu_run_bench(num_prompts, request_rate, extra_server_args=None):
    return run_bench_serving(
        model=DCU_MODEL,
        num_prompts=num_prompts,
        request_rate=request_rate,
        other_server_args=_dcu_server_args(extra_server_args),
        random_input_len=128,
        random_output_len=64,
    )


class TestBenchServing1GPUPart1(CustomTestCase):
    def test_offline_throughput_default(self):
        if is_in_dcu_ci():
            res = _dcu_run_bench(num_prompts=20, request_rate=float("inf"))
            self.assertGreater(res["output_throughput"], 0)
            return

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3050)
            else:
                self.assertGreater(res["output_throughput"], 3800)

    def test_offline_throughput_non_stream_small_batch_size(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps ShareGPT/non-stream throughput in the perf specialty track.")

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=200,
            request_rate=float("inf"),
            other_server_args=["--max-running-requests", "10"],
            dataset_name="sharegpt",
            random_input_len=None,
            random_output_len=None,
            disable_stream=True,
            need_warmup=True,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_non_stream_small_batch_size\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 1000)
            else:
                self.assertGreater(res["output_throughput"], 1050)

    def test_offline_throughput_without_radix_cache(self):
        if is_in_dcu_ci():
            res = _dcu_run_bench(
                num_prompts=20,
                request_rate=float("inf"),
                extra_server_args=["--disable-radix-cache"],
            )
            self.assertGreater(res["output_throughput"], 0)
            return

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_without_radix_cache\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3050)
            else:
                self.assertGreater(res["output_throughput"], 3800)

    def test_offline_throughput_without_chunked_prefill(self):
        if is_in_dcu_ci():
            res = _dcu_run_bench(
                num_prompts=20,
                request_rate=float("inf"),
                extra_server_args=["--chunked-prefill-size", "-1"],
            )
            self.assertGreater(res["output_throughput"], 0)
            return

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=["--chunked-prefill-size", "-1"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_without_chunked_prefill\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 2600)

    def test_offline_throughput_with_triton_attention_backend(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass validates fa3; Triton backend stays in the attention/perf specialty track.")

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--attention-backend",
                "triton",
                "--context-length",
                "8192",
            ],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_with_triton_attention_backend\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3500)
            else:
                self.assertGreater(res["output_throughput"], 3700)

    def test_online_latency_default(self):
        if is_in_dcu_ci():
            res = _dcu_run_bench(num_prompts=10, request_rate=1)
            self.assertGreater(res["median_e2e_latency_ms"], 0)
            self.assertGreater(res["median_ttft_ms"], 0)
            self.assertGreater(res["median_itl_ms"], 0)
            return

        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=100,
            request_rate=1,
            other_server_args=[],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_latency_default\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 11000)
            if is_in_amd_ci():
                self.assertLess(res["median_ttft_ms"], 115)
            else:
                self.assertLess(res["median_ttft_ms"], 86)
            self.assertLess(res["median_itl_ms"], 10)

    def test_lora_online_latency(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps LoRA latency in the LoRA/perf specialty track.")

        if is_in_amd_ci():
            pass

        res = self._run_lora_latency_test(enable_background_task=False)

        if is_in_ci():
            write_github_step_summary(
                f"### test_lora_online_latency\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 2400)
            self.assertLess(res["median_ttft_ms"], 58)

    def test_lora_online_latency_with_concurrent_adapter_updates(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps LoRA adapter update latency in the LoRA/perf specialty track.")

        if is_in_amd_ci():
            pass

        res = self._run_lora_latency_test(enable_background_task=True)

        if is_in_ci():
            write_github_step_summary(
                f"### test_lora_online_latency\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 4000)
            self.assertLess(res["median_ttft_ms"], 80)

    def _run_lora_latency_test(self, enable_background_task: bool):
        """
        Run a latency test for LoRA with the specified background task setting.
        """

        async def lora_loader_unloader_task(
            base_url: str,
            start_event: asyncio.Event,
            stop_event: asyncio.Event,
        ):
            """
            A background task that repeatedly loads and unloads a LoRA adapter.
            """
            await start_event.wait()

            path_cycler = itertools.cycle(
                [
                    "pbevan11/llama-3.1-8b-ocr-correction",
                    "faridlazuarda/valadapt-llama-3.1-8B-it-chinese",
                    "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                ]
            )
            load_url = f"{base_url}/load_lora_adapter"
            unload_url = f"{base_url}/unload_lora_adapter"
            num_updates = 0

            while not stop_event.is_set():
                lora_path = next(path_cycler)
                response = await asyncio.to_thread(
                    requests.post,
                    load_url,
                    json={"lora_name": lora_path, "lora_path": lora_path},
                )
                self.assertTrue(
                    response.ok, f"Failed to load LoRA adapter: {response.text}"
                )
                num_updates += 1

                if stop_event.is_set():
                    break

                await asyncio.sleep(1)

                response = await asyncio.to_thread(
                    requests.post,
                    unload_url,
                    json={"lora_name": lora_path},
                )
                self.assertTrue(
                    response.ok, f"Failed to unload LoRA adapter: {response.text}"
                )
                num_updates += 1

                await asyncio.sleep(1)

        background_task = lora_loader_unloader_task if enable_background_task else None
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=400,
            request_rate=8,
            other_server_args=[
                "--enable-lora",
                "--max-loras-per-batch",
                "1",
                "--disable-radix-cache",
                "--random-seed",
                "42",
                "--mem-fraction-static",
                "0.8",
                "--lora-paths",
                "nvidia/llama-3.1-nemoguard-8b-topic-control",
                "--max-lora-rank",
                "256",
            ],
            dataset_name="random",
            random_input_len=256,
            random_output_len=256,
            lora_name=["nvidia/llama-3.1-nemoguard-8b-topic-control"],
            background_task=background_task,
        )

        return res


if __name__ == "__main__":
    unittest.main()
