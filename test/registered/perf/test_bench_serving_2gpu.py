"""
Performance tests for 2-GPU that need large GPUs (H200 80GB) - MoE and Pipeline Parallel tests.
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    is_in_dcu_ci,
    run_bench_serving,
    write_github_step_summary,
)
register_dcu_ci(
    est_time=120,
    suite="nightly-dcu",
    nightly=True,
)

register_cuda_ci(est_time=600, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=1100, suite="stage-b-test-2-gpu-large-amd")


class TestBenchServing2GPU(CustomTestCase):
    @staticmethod
    def _dcu_server_args(extra_args=None):
        args = [
            "--tp",
            "2",
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

    @unittest.skipIf(not is_in_dcu_ci(), "DCU-only TP2 serving smoke.")
    def test_dcu_tp2_serving_smoke(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=self._dcu_server_args(),
            random_input_len=128,
            random_output_len=64,
        )
        self.assertGreater(res["output_throughput"], 0)

    def test_moe_offline_throughput_default(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps MoE TP2 serving throughput in the perf/MoE specialty track.")

        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=300,
            request_rate=float("inf"),
            other_server_args=["--tp", "2"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_offline_throughput_default\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 2100)
            else:
                self.assertGreater(res["output_throughput"], 2200)

    def test_moe_offline_throughput_without_radix_cache(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps MoE TP2 no-radix throughput in the perf/MoE specialty track.")

        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=300,
            request_rate=float("inf"),
            other_server_args=["--tp", "2", "--disable-radix-cache"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_offline_throughput_without_radix_cache\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 2100)
            else:
                self.assertGreater(res["output_throughput"], 2200)

    def test_pp_offline_throughput_default_decode(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps pipeline-parallel decode throughput in the perf/PP specialty track.")

        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=1000,
            request_rate=float("inf"),
            random_input_len=1,
            random_output_len=1024,
            other_server_args=["--pp-size", "2"],
            need_warmup=True,
            seed=42,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_pp_offline_throughput_default_decode\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 6700)

    def test_pp_long_context_prefill(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps long-context PP prefill in the perf/PP specialty track.")

        res = run_bench_serving(
            model="meta-llama/Llama-3.3-70B-Instruct",
            num_prompts=4,
            request_rate=float("inf"),
            random_input_len=128000,
            random_output_len=1,
            dataset_name="random",
            other_server_args=[
                "--quantization",
                "fp8",
                "--pp-size",
                "2",
            ]
            + (["--mem-fraction-static", "0.7"] if is_in_amd_ci() else []),
            need_warmup=False,
            seed=42,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_pp_long_context_latency_prefill\n"
                f"input_throughput: {res['input_throughput']:.2f} ms\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["input_throughput"], 3000)
            else:
                self.assertGreater(res["input_throughput"], 4000)


if __name__ == "__main__":
    unittest.main()
