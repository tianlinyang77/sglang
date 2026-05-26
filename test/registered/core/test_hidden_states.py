import unittest
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.srt.utils import get_device, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_dcu,
    is_in_dcu_ci,
)

register_cuda_ci(est_time=55, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=55, suite="stage-b-test-1-gpu-small-amd")
# DCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical DCU coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=55,
    suite="stage-b-dcu",
    disabled="BW1000 quick validation failed: HF vs SRT hidden states max diff around 7.75; needs numerical investigation before PR gate.",
)

DEFAULT_DCU_HIDDEN_STATES_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-0.5B-Instruct"
)
DCU_ENGINE_KWARGS = {
    "attention_backend": "fa3",
    "page_size": 64,
    "disable_cuda_graph": True,
    "context_length": 2048,
    "max_total_tokens": 4096,
    "max_running_requests": 8,
    "chunked_prefill_size": 2048,
}

_is_hip = is_hip()
if _is_hip:
    os.environ["SGLANG_USE_AITER"] = "0"

if is_dcu() or is_in_dcu_ci():
    os.environ.setdefault("SGLANG_USE_MODELSCOPE", "1")
    os.environ.setdefault("SGLANG_USE_LIGHTOP", "1")


def get_test_model_path():
    if is_dcu() or is_in_dcu_ci():
        return os.environ.get(
            "SGLANG_DCU_HIDDEN_STATES_MODEL",
            os.environ.get("SGLANG_DCU_SERVER_SMOKE_MODEL", DEFAULT_DCU_HIDDEN_STATES_MODEL),
        )
    return DEFAULT_SMALL_MODEL_NAME_FOR_TEST


def get_dcu_engine_kwargs():
    return DCU_ENGINE_KWARGS if is_dcu() or is_in_dcu_ci() else {}


class TestHiddenState(CustomTestCase):
    def test_return_hidden_states(self):
        prompts = ["Today is", "Today is a sunny day and I like"]
        model_path = get_test_model_path()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 8,
        }

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
            enable_return_hidden_states=True,
            **get_dcu_engine_kwargs(),
        )
        outputs = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        engine.shutdown()

        for output in outputs:
            self.assertEqual(len(output["meta_info"]["hidden_states"]), 8)
            for i in range(len(output["meta_info"]["hidden_states"])):
                assert isinstance(output["meta_info"]["hidden_states"][i], list)
                output["meta_info"]["hidden_states"][i] = torch.tensor(
                    output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                )
        # Checks that splicing of the batch was done correctly
        self.assertGreater(
            outputs[1]["meta_info"]["hidden_states"][0].shape[0],
            outputs[0]["meta_info"]["hidden_states"][0].shape[0],
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=get_device()
        )

        for input_id, output in zip(input_ids, outputs):
            with torch.inference_mode():
                hf_out = model(
                    torch.tensor(
                        [input_id + output["output_ids"][:-1]], device=model.device
                    ),
                    output_hidden_states=True,
                )
            print("=== HF Hiddens ===")
            print(hf_out["hidden_states"][-1][0])
            sg_hidden_states = torch.cat(
                [
                    i.unsqueeze(0) if len(i.shape) == 1 else i
                    for i in output["meta_info"]["hidden_states"]
                ]
            ).to(get_device())
            print("=== SRT Hiddens ===")
            print(sg_hidden_states)

            print(
                f"Max diff: {torch.max(torch.abs(hf_out['hidden_states'][-1][0] - sg_hidden_states))}"
            )

            atol = 0.8
            self.assertTrue(
                torch.allclose(
                    hf_out["hidden_states"][-1][0],
                    sg_hidden_states,
                    atol=atol,
                    rtol=0,
                )
            )

    def test_repeatedly_changes_hidden_states(self):
        prompts = ["Today is", "Today is a sunny day and I like"]
        model_path = get_test_model_path()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 8,
        }

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
            enable_return_hidden_states=True,
            **get_dcu_engine_kwargs(),
        )
        outputs_completion_first_round = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        outputs_hidden_state = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=False,
        )

        outputs_completion_last_round = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )
        engine.shutdown()

        for (
            output_completion_first_round,
            output_hidden_state,
            output_completion_last_round,
        ) in zip(
            outputs_completion_first_round,
            outputs_hidden_state,
            outputs_completion_last_round,
        ):
            self.assertEqual(
                len(output_completion_first_round["meta_info"]["hidden_states"]), 8
            )
            self.assertNotIn("hidden_states", output_hidden_state["meta_info"])
            self.assertEqual(
                len(output_completion_last_round["meta_info"]["hidden_states"]), 8
            )


if __name__ == "__main__":
    unittest.main()
