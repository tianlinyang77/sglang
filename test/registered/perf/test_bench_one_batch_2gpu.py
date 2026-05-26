import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    is_in_dcu_ci,
    run_bench_offline_throughput,
    run_bench_one_batch,
    write_github_step_summary,
)
register_dcu_ci(
    est_time=120,
    suite="nightly-dcu",
    nightly=True,
)

register_cuda_ci(est_time=180, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=630, suite="stage-b-test-2-gpu-large-amd")


class TestBenchOneBatch2GPU(CustomTestCase):
    @staticmethod
    def _dcu_bench_args():
        if not is_in_dcu_ci():
            return []
        return [
            "--tp",
            "2",
            "--attention-backend",
            "fa3",
            "--page-size",
            "64",
            "--disable-cuda-graph",
        ]

    @unittest.skipIf(not is_in_dcu_ci(), "DCU-only TP2 small-model smoke.")
    def test_dcu_tp2_bs1_small(self):
        _, output_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            ["--cuda-graph-max-bs", "2"] + self._dcu_bench_args(),
        )
        self.assertGreater(output_throughput, 0)

    def test_moe_tp2_bs1(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps MoE TP2 throughput in the perf/MoE specialty track.")

        output_throughput = run_bench_offline_throughput(
            DEFAULT_MOE_MODEL_NAME_FOR_TEST, ["--tp", "2", "--cuda-graph-max-bs", "2"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_moe_tp2_bs1 (Mixtral-8x7B)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(output_throughput, 85)
            else:
                self.assertGreater(output_throughput, 125)

    def test_torch_compile_tp2_bs1(self):
        if is_in_dcu_ci():
            self.skipTest("DCU quick pass keeps torch-compile TP2 throughput in the perf/compile specialty track.")

        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            ["--tp", "2", "--enable-torch-compile", "--cuda-graph-max-bs", "2"],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_torch_compile_tp2_bs1 (Mixtral-8x7B)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(output_throughput, 200)
            else:
                self.assertGreater(output_throughput, 220)


if __name__ == "__main__":
    unittest.main()
