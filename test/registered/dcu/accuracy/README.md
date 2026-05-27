# DCU accuracy tests

This directory contains DCU-specific accuracy evaluations.  Hardware-specific
baselines live under subdirectories such as `bw1000/`.

The tests reuse the shared SGLang eval entry point (`sglang.test.run_eval`) and
the common CI helpers from `sglang.test.test_utils`.  They should only define
DCU model selection, launch arguments, sample sizes, thresholds, and result
checks.

## Environment variables

Text accuracy tests use these common overrides:

- `SGLANG_DCU_GSM8K_MODEL`
- `SGLANG_DCU_MMLU_MODEL`
- `SGLANG_DCU_GSM8K_DATA_PATH`
- `SGLANG_DCU_MMLU_DATASET_PATH`
- `SGLANG_DCU_GSM8K_NUM_EXAMPLES`
- `SGLANG_DCU_MMLU_NUM_EXAMPLES`
- `SGLANG_DCU_GSM8K_NUM_THREADS`
- `SGLANG_DCU_MMLU_NUM_THREADS`
- `SGLANG_DCU_GSM8K_THRESHOLD`
- `SGLANG_DCU_MMLU_THRESHOLD`
- `SGLANG_DCU_EVAL_NUM_EXAMPLES`

`SGLANG_DCU_EVAL_NUM_EXAMPLES` is kept as a compatibility fallback for older
debug commands. Prefer the per-dataset variables above so GSM8K and MMLU can
use different nightly sample counts.

The current BW1000 smoke defaults are 10 GSM8K examples and 50 MMLU examples.
Their conservative default thresholds are based on three smoke runs:
`GSM8K >= 0.65` and `MMLU >= 0.68`. Nightly defaults are 1319 GSM8K examples
and 5000 MMLU examples when set from the workflow, but their thresholds should
be replaced after dedicated BW1000 baseline runs.

MMLU prefers the local parquet cache under
`/public/opendas/DL_DATA/llm-models/datasets/mmlu` when
`SGLANG_DCU_MMLU_DATASET_PATH` is unset and the cache exists. The local parquet
reader consumes `*/test-*.parquet` files with `question`, `subject`, `choices`,
and `answer` fields. Numeric answers `0` to `3` are mapped to `A` to `D`.
Known MMLU subject directories are loaded in the same subject order as the
simple-evals CSV baseline, so seeded smoke samples stay comparable.

MMMU uses:

- `SGLANG_DCU_MMMU_MODEL`
- `SGLANG_DCU_MMMU_THRESHOLD`
- `SGLANG_DCU_MMMU_LATENCY_THRESHOLD`
- `SGLANG_DCU_MMMU_NUM_EXAMPLES`
- `SGLANG_DCU_MMMU_NUM_THREADS`
- `SGLANG_DCU_MMMU_DATASET_PATH`

When `SGLANG_DCU_MMMU_MODEL` or `SGLANG_DCU_MMMU_DATASET_PATH` are unset, the
BW1000 MMMU test first checks the local DCU model/data cache under
`/public/opendas/DL_DATA/llm-models`. If those defaults are absent, the VLM test
is skipped instead of downloading gated models implicitly.

The current MMMU smoke default is 10 local examples with 4 worker threads and a
conservative `MMMU >= 0.35` threshold. This should be replaced with a
100-sample threshold after the BW1000 nightly baseline is collected.

The default DCU launch arguments follow the stable BW1000 smoke configuration:
text tests use `--attention-backend fa3 --page-size 64`, while VLM tests add
`--mm-attention-backend fa3 --enable-multimodal`. Memory sizing is intentionally
left to per-test server argument overrides instead of being fixed in the tests.

Server launch arguments can be replaced per test:

- `SGLANG_DCU_GSM8K_SERVER_ARGS`
- `SGLANG_DCU_MMLU_SERVER_ARGS`
- `SGLANG_DCU_MMMU_SERVER_ARGS`

If a model path is local, the path must exist.  This avoids accidental fallback
to gated Hugging Face downloads in DCU CI.
