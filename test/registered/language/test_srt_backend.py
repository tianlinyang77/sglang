import os
import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_programs import (
    test_decode_int as run_decode_int_program,
    test_decode_json_regex as run_decode_json_regex_program,
    test_dtype_gen as run_dtype_gen_program,
    test_expert_answer as run_expert_answer_program,
    test_few_shot_qa as run_few_shot_qa_program,
    test_gen_min_new_tokens as run_gen_min_new_tokens_program,
    test_hellaswag_select as run_hellaswag_select_program,
    test_mt_bench as run_mt_bench_program,
    test_parallel_decoding as run_parallel_decoding_program,
    test_regex as run_regex_program,
    test_select as run_select_program,
    test_stream as run_stream_program,
    test_tool_use as run_tool_use_program,
)
register_dcu_ci(
    est_time=120,
    suite="stage-b-dcu",
    disabled="DCU PR baseline deferred: test is registered for DCU coverage but lacks three-pass BW1000 PR-gate repeat evidence.",
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
)

register_cuda_ci(est_time=80, suite="stage-a-test-1-gpu-small")
register_amd_ci(est_time=120, suite="stage-a-test-1-gpu-small-amd")

DEFAULT_DCU_SRT_BACKEND_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)


class TestSRTBackend(CustomTestCase):
    backend = None

    @classmethod
    def setUpClass(cls):
        if is_dcu() or is_in_dcu_ci():
            model_path = os.environ.get(
                "SGLANG_DCU_SRT_BACKEND_MODEL", DEFAULT_DCU_SRT_BACKEND_MODEL
            )
            runtime_kwargs = {
                "model_path": model_path,
                "attention_backend": "fa3",
                "page_size": 64,
                "trust_remote_code": True,
                "disable_cuda_graph": True,
                "context_length": 2048,
                "max_total_tokens": 4096,
                "max_running_requests": 8,
                "chunked_prefill_size": 2048,
            }
        else:
            runtime_kwargs = {
                "model_path": DEFAULT_MODEL_NAME_FOR_TEST,
                "cuda_graph_max_bs": 4,
                "mem_fraction_static": 0.7,
            }
        cls.backend = sgl.Runtime(
            **runtime_kwargs,
        )
        sgl.set_default_backend(cls.backend)

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown()

    def test_few_shot_qa(self):
        if is_dcu() or is_in_dcu_ci():
            self.skipTest("DCU quick framework skips model-semantic few-shot QA.")
        run_few_shot_qa_program()

    def test_mt_bench(self):
        run_mt_bench_program()

    def test_select(self):
        run_select_program(check_answer=False)

    def test_decode_int(self):
        if is_dcu() or is_in_dcu_ci():
            self.skipTest("DCU quick framework skips model-semantic integer decoding.")
        run_decode_int_program()

    @unittest.skip("Skip this flaky test.")
    def test_decode_json_regex(self):
        run_decode_json_regex_program()

    def test_expert_answer(self):
        run_expert_answer_program(check_answer=not (is_dcu() or is_in_dcu_ci()))

    def test_tool_use(self):
        if is_dcu() or is_in_dcu_ci():
            self.skipTest("DCU quick framework skips model-semantic tool-use generation.")
        run_tool_use_program()

    def test_parallel_decoding(self):
        if is_dcu() or is_in_dcu_ci():
            self.skipTest("DCU quick framework skips heavier parallel decoding.")
        run_parallel_decoding_program()

    def test_stream(self):
        run_stream_program()

    def test_regex(self):
        run_regex_program()

    def test_dtype_gen(self):
        run_dtype_gen_program()

    def test_hellaswag_select(self):
        if is_dcu() or is_in_dcu_ci():
            self.skipTest("DCU quick framework skips HellaSwag dataset/accuracy coverage.")
        # Run twice to capture more bugs
        for _ in range(2):
            accuracy, latency = run_hellaswag_select_program()
            self.assertGreater(accuracy, 0.60)

    def test_gen_min_new_tokens(self):
        run_gen_min_new_tokens_program()


if __name__ == "__main__":
    unittest.main()
