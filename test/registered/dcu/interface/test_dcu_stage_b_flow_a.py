import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.test_utils import CustomTestCase

register_dcu_ci(est_time=2, suite="stage-b-dcu")


class TestDCUStageBFlowA(CustomTestCase):
    """Flow-smoke test used to exercise DCU PR matrix routing."""

    def test_dcu_ci_env_marker(self):
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping marker check.")
        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI"), "1")


if __name__ == "__main__":
    unittest.main()
