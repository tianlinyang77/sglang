import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_dcu_ci,
    is_in_ci,
    run_bench_offline_throughput,
    run_bench_one_batch,
    write_github_step_summary,
)

register_dcu_ci(
    est_time=120,
    suite="nightly-dcu",
    nightly=True,
)

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-large-amd")


class TestBenchOneBatch1GPU(CustomTestCase):
    @staticmethod
    def _dcu_bench_args():
        if not is_in_dcu_ci():
            return []
        return [
            "--attention-backend",
            "fa3",
            "--page-size",
            "64",
            "--disable-cuda-graph",
        ]

    def test_bs1_small(self):
        _, output_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            ["--cuda-graph-max-bs", "2"] + self._dcu_bench_args(),
        )
        self.assertGreater(output_throughput, 50)

    @unittest.skipIf(
        is_in_dcu_ci(),
        "DCU_CSV_NOT_APPLICABLE: default 8B throughput threshold needs BW1000 perf baseline.",
    )
    def test_bs1_default(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST, ["--cuda-graph-max-bs", "2"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs1_default (llama-3.1-8b)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 135)


if __name__ == "__main__":
    unittest.main()
