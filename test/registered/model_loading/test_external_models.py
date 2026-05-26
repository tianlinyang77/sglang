import os
import unittest

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_dcu_ci,
)
from sglang.test.test_utils import CustomTestCase, is_in_dcu_ci

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")
register_dcu_ci(
    est_time=120,
    suite="stage-b-dcu",
    disabled="DCU PR baseline deferred: test is registered for DCU coverage but lacks three-pass BW1000 PR-gate repeat evidence.",
)
register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd")

DCU_QWEN2_VL_2B_MODEL = os.environ.get(
    "SGLANG_DCU_QWEN2_VL_MODEL",
    "/public/opendas/DL_DATA/llm-models/qwen2/Qwen2-VL-2B-Instruct",
)


class TestExternalModels(CustomTestCase):
    def test_external_model(self):
        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
        prompt = "Today is a sunny day and I like"
        model_path = (
            DCU_QWEN2_VL_2B_MODEL
            if is_in_dcu_ci()
            else "Qwen/Qwen2-VL-2B-Instruct"
        )

        engine_kwargs = dict(
            model_path=model_path,
            cuda_graph_max_bs=1,
            max_total_tokens=64,
            enable_multimodal=True,
        )
        if is_in_dcu_ci():
            engine_kwargs.update(
                attention_backend="fa3",
                mm_attention_backend="fa3",
                page_size=64,
                disable_cuda_graph=True,
                trust_remote_code=True,
            )

        engine = sgl.Engine(**engine_kwargs)
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
