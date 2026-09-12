# A30 Precision Smoke: Passed

- Commit: `181ab8c4c6b267d20186a561b03d6da5c17f95cc`.
- Worker: 4252179; NVIDIA A30, SM80, 56 SMs, MIG disabled.
- Python 3.11.2; PyTorch 2.7.1; CUDA runtime 12.6; Triton 3.3.1.
- Driver 535.161.08; power limit 165 W.
- No GPU compute processes were present before or after the run.
- Worker CPU self-test: 41 tests passed.
- No local kernel, tolerance, dependency, or driver changes.

## Command

Run directly from the repository root on the worker, without any historical
`run_suite.py`:

```bash
python test/manual/beam_search/test_beam_kv_attention.py matrix \
  --preset a30-smoke \
  --output-dir test/manual/beam_search/results/a30_smoke_precision_01
```

The matrix used 10 warmups, 30 timing samples, graph batch 64, and suffix
splits 1, 2, 4. All five cases enabled partial-LSE checking.

## Correctness and Compilation

`manifest.json`: `complete=true`, `all_cases_validated=true`,
`resource_skipped=0`; all five child processes returned 0.

| Case | Status | Maximum output absolute error | Maximum merged LSE absolute error | Maximum partial LSE absolute error |
| --- | --- | ---: | ---: | ---: |
| single | Passed | 8.793e-4 | 1.160e-7 | 1.160e-7 |
| mha_tail | Passed | 4.874e-4 | 2.573e-7 | 2.573e-7 |
| mha_prefix | Passed | 2.691e-4 | 7.247e-7 | 7.974e-7 |
| gqa_tail | Passed | 4.226e-4 | 7.903e-7 | 7.670e-7 |
| independent | Passed | 9.742e-4 | 2.726e-7 | 2.726e-7 |

There are 39 mode/split/layer correctness records across the five cases.
The LSE tolerances remain `atol=1e-4`, `rtol=1e-4`; FP16 output tolerances
remain `atol=0.002`, `rtol=0.002`.

`mha_prefix` selected one shared group with a 62-token common prefix.
Its auto policy selected no shared group under the default thresholds.
Every captured shared-prefix kernel has 16 static PTX MMA instructions and
zero Triton-reported spills. SASS exports succeeded for all captured kernels.
Compiler dumps remain local under the existing ignore rules.

## Performance Scope

For the four multi-beam cases, shared total speedup versus the best tested
gather split is 0.176-0.218x; auto is 0.181-0.218x. These small smoke cases
are slower on the shared/auto paths and do not establish a performance win.

The baseline is the new precision-matched experimental gather with LSE
merge, not the unmodified production gather. Do not compare these speedups
directly with historical results using the old baseline.

Only the requested smoke matrix was run. Calibration, main, trace, and
stress matrices were not run in this invocation. No policy was frozen.
