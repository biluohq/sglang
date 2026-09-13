# A30 Beam Triton Benchmark

Environment: NVIDIA A30 24 GiB, CUDA 12.6, PyTorch 2.7.1, Triton 3.4.0.

All cases passed the mandatory baseline/new-path comparison against the FP64
reference. Timings use warm CUDA Graph replay. `kernel` excludes metadata;
`route_e2e` starts from row-wise KV paths and includes metadata construction.

## Width Sweep

All width-sweep cases use FP16 MHA, head dimension 128, sequence length 128,
and a 127-token shared prefix.

| Beam width | Layers | Kernel speedup | Route E2E speedup |
| ---: | ---: | ---: | ---: |
| 64 | 8 | 1.323x | 1.270x |
| 128 | 8 | 1.615x | 1.568x |
| 256 | 8 | 1.706x | 1.662x |
| 512 | 8 | 1.753x | 1.749x |
| 1024 | 8 | 1.713x | 1.721x |
| 2048 | 1 | 1.713x | 1.651x |

## Boundary And Coverage Cases

| Case | Kernel speedup | Route E2E speedup | Production selection |
| --- | ---: | ---: | --- |
| MHA k=3, seq=64, prefix=63, 1 layer | 0.708x | 0.563x | fallback |
| MHA k=8, seq=128, prefix=127, 32 layers | 1.024x | 1.006x | fallback |
| MHA k=16, seq=128, prefix=127, 32 layers | 1.031x | 1.017x | beam |
| MHA k=32, seq=256, prefix=128, 32 layers | 1.107x | 1.100x | beam |
| MHA k=32, seq=512, prefix=400, 8 layers | 1.316x | 1.294x | beam |
| MHA k=16, 8 requests, seq=128, 8 layers | 1.619x | 1.568x | beam |
| BF16 MHA k=16, head dim 64, 8 layers | 0.797x | 0.749x | fallback |
| FP16 GQA k=32, 8/2 heads, 32 layers | 0.853x | 0.834x | fallback |

The initial production policy enables only FP16 MHA with head dimension 128,
group width at least 8, prefix length at least 64, and at least 1024 saved
logical KV references per group. Other cases use the ordinary Triton backend.

These are synthetic attention-route measurements, not full model/server
latency. CUDA Graph serving currently falls back to Triton until beam-plan
buffers become capture-stable.
