# Beam Triton Attention Benchmark

This benchmark compares the production Triton gather stage with the
beam-aware prefix-sharing path. It has one command:

```bash
.venv-beam-kv/bin/python -B test/manual/beam_search/test_beam_kv_attention.py benchmark \
  --beams 32 \
  --seq-len 128 \
  --shared-prefix 127 \
  --q-heads 8 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype float16 \
  --layers 32 \
  --execution graph \
  --graph-batch 16 \
  --warmup 20 \
  --repeats 100 \
  --output /tmp/beam_attention.json
```

The output contains two primary comparisons:

- `baseline_kernel` versus `beam_kernel`: metadata is prebuilt and only the
  complete attention pipelines are timed.
- `baseline_route_e2e` versus `beam_route_e2e`: both routes start from the
  same row-wise paths, build their runtime metadata, and finish with the final
  attention output.

The beam pipeline also reports `beam_prefix_stage`, `beam_suffix_stage`, and
`beam_merge_stage` to identify where time is spent. Compilation, allocation,
and correctness references are outside timed regions.

Every run first compares both paths with an FP64 PyTorch reference. A numerical
mismatch fails the benchmark instead of producing timing results.

Use `--cache-mode warm` for steady-state CUDA Graph replay. Use
`--cache-mode scrub --eviction-mib 256` for best-effort cache pressure; scrub
time is excluded.

The current backend shares the prompt and the committed history of sibling
beams when parent metadata is available. Its conservative dispatch policy
currently enables only sufficiently large FP16 MHA beam groups with head
dimension 128 on NVIDIA CUDA, page size 1, and a pure beam decode batch. Small
groups, GQA, BF16, other head dimensions, mixed or CUDA-graph serving,
quantized, MLA, sliding-window, DCP, sink, and score-mod cases fall back to the
normal Triton backend. The benchmark can still measure those candidate shapes
and reports whether the production backend would select them.
