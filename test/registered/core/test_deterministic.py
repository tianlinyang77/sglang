"""
Usage:
cd test/srt
python3 -m unittest test_deterministic.TestDeterministic.TESTCASE

Note that there is also `python/sglang/test/test_deterministic.py` as an interactive test. We are converting that
test into unit tests so that's easily reproducible in CI.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)
from sglang.test.test_utils import is_dcu, is_in_amd_ci, is_in_dcu_ci

register_cuda_ci(est_time=278, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=278, suite="stage-b-test-1-gpu-small-amd")
# DCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical DCU coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=278,
    suite="stage-b-dcu",
    disabled="DCU PR baseline deferred: core/server path needs BW1000 model-runtime repeat validation before required CI.",
)

DEFAULT_DCU_DETERMINISTIC_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


@unittest.skipIf(is_in_amd_ci() or is_dcu() or is_in_dcu_ci(), "Skip for AMD/DCU CI.")
class TestFlashinferDeterministic(TestDeterministicBase):
    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        args = list(COMMON_SERVER_ARGS)
        args.extend(
            [
                "--attention-backend",
                "flashinfer",
            ]
        )
        return args


@unittest.skipIf(is_in_amd_ci(), "Skip for AMD CI.")
class TestFa3Deterministic(TestDeterministicBase):
    # Test with fa3 attention backend
    @classmethod
    def get_model(cls):
        if is_dcu() or is_in_dcu_ci():
            return os.environ.get(
                "SGLANG_DCU_DETERMINISTIC_MODEL",
                DEFAULT_DCU_DETERMINISTIC_MODEL,
            )
        return super().get_model()

    @classmethod
    def get_server_args(cls):
        args = list(COMMON_SERVER_ARGS)
        args.extend(
            [
                "--attention-backend",
                "fa3",
            ]
        )
        if is_dcu() or is_in_dcu_ci():
            args.extend(
                [
                    "--page-size",
                    "64",
                    "--trust-remote-code",
                    "--disable-cuda-graph",
                    "--context-length",
                    "4096",
                    "--max-total-tokens",
                    "8192",
                    "--max-running-requests",
                    "8",
                    "--chunked-prefill-size",
                    "4096",
                ]
            )
        return args


@unittest.skipIf(is_dcu() or is_in_dcu_ci(), "DCU quick framework uses fa3 only.")
class TestTritonDeterministic(TestDeterministicBase):
    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        args = list(COMMON_SERVER_ARGS)
        args.extend(
            [
                "--attention-backend",
                "triton",
            ]
        )
        return args


if __name__ == "__main__":
    unittest.main()
