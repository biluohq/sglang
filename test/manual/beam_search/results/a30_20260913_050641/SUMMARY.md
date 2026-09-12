# A30 Test Run: Stopped at Smoke Correctness Gate

- Worker: 4252179, NVIDIA A30, SM80, 56 SMs, MIG disabled.
- Driver: 535.161.08; power limit: 165 W.
- Python: 3.11.2; PyTorch: 2.7.1; CUDA runtime: 12.6; Triton: 3.3.1.
- Git revision: `dbdb0fe06bf7ce0fc3744fa1c2c932ecb46fec9e`.
- GPU process queries before and after testing showed no other compute processes.
- Policy: default candidate, not frozen or performance-validated.
- No kernel, test tolerance, dependency, driver, or clock changes were made.

## Results

| Stage | Result |
| --- | --- |
| Environment and FP16 PyTorch smoke | Passed |
| CPU self-test | 38 tests passed |
| Smoke dry-run | Passed |
| Smoke `single` | Passed |
| Smoke `mha_tail` | Failed LSE correctness |
| Smoke `gqa_tail`, `independent` | Not run after failure |
| Calibrate (39), main (330), trace (13), stress (150) | Not run: smoke gate failed |

Of 536 planned matrix cases: 1 passed, 1 failed, 534 were not run.
This is not a completed matrix or a performance acceptance result.

## Failure

Case: FP16, beams=3, output_tokens=16, pattern=grouped, q_heads=8,
kv_heads=8, head_dim=128, seed=43. Default query_tile=16, tile=32,
prefix_splits=2, warps=4, stages=2; suffix splits requested: 1, 2, 4.

The failure is at `test_beam_kv_attention.py:2923`, the shared/auto
LSE assertion, using unchanged `atol=1e-4`, `rtol=1e-4`:

```text
Mismatched elements: 4 / 24 (16.7%)
Greatest absolute difference: 0.0009704748198977597 at index (1, 4)
Greatest relative difference: 0.00034664241475592874 at index (1, 4)
```

The matrix stopped immediately. Its manifest has `complete=false`.
The failed case has no successful result JSON. The shared-prefix MMA gate
was not reached, so this run does not claim that gate passed.

## Diagnostic Context

Pre-existing artifacts in `../a30_20260913_045514/diagnosis.json` and
`precision_probe.json` record the same numerical failure at shared mode,
suffix split 1, layer 0, with zero common prefix for all three rows.
They identify the same error in suffix LSE and record an isolated FP32
Q/K diagnostic reducing LSE error to approximately 2.57e-7. These are
earlier diagnostic artifacts, not new successful tests from this run.

Source inspection shows the MHA gather stage computes
`tl.sum(q[None, :] * k, 1)` in
`python/sglang/kernels/ops/attention/decode_attention.py:335`.
The evidence points to the MHA gather precision path and its compatibility
with the strict FP64-reference LSE check. No fix was applied, and the
earlier FP32 probe is not a benchmark or a proposed production fix.

## Artifacts and Next Step

- `suite_status.json`: commands, timestamps, exit statuses.
- `environment.log`, `nvidia_smi.log`, `gpu_processes.log`: environment.
- `self_test.log`, `smoke_dry_run.log`: preflight evidence.
- `smoke/manifest.json`, `smoke/mha_tail.log`: failure evidence.
- `smoke/single.json`, `smoke/kernels/single/`: successful baseline and compiler dumps.
- `run_suite.py`: exact one-shot orchestration; refuses to overwrite this run.

Resolve the LSE precision discrepancy without relaxing tolerance, then
rerun smoke in a fresh output directory. Only after correctness and MMA
gates pass should calibration and coverage matrices continue.
