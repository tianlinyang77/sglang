# DCU Registered Tests

This directory holds DCU backend-specific registered CI tests.

## Layout

- `interface/`: minimal smoke, API, and health-check level tests.
- `accuracy/`: DCU accuracy evaluation tests such as GSM8K, MMLU, and MMMU.
- `perf/`: DCU throughput, latency, TTFT, TPOT, ITL, and concurrency benchmarks.
- `disaggregation/`: PD-disaggregation tests that need DCU-specific implementations.
- `basic_function/`: backend, quantization, parallel strategy, and runtime option tests.
- `llm_models/`: DCU-specific LLM model tests.
- `vlm_models/`: DCU-specific VLM model tests.

## Registering a test

Add a module-level registration near the top of the file:

```python
from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(est_time=120, suite="stage-b-dcu")
```

For nightly tests, set `nightly=True` and use `suite="nightly-dcu"`.

## Running locally

```bash
python3 test/run_suite.py --hw dcu --suite stage-a-dcu
python3 test/run_suite.py --hw dcu --suite stage-b-dcu
python3 test/run_suite.py --hw dcu --suite nightly-dcu --nightly
```

## Static verification

```bash
python3 scripts/ci/dcu/verify_dcu_registration.py
```
