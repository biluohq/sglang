"""Benchmark the production Triton baseline against beam-aware attention.

This file intentionally has one command and two timing scopes:

* kernel: metadata is prebuilt; only the complete attention paths are timed.
* route_e2e: both paths start from the same row-wise KV mapping and finish
  with the final attention output. Metadata preparation is included.

The benchmark uses synthetic Q/K/V but calls the production kernel entry
points. Correctness is a mandatory, untimed gate before measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch
import triton
from sglang.kernels.ops.attention.beam_decode_attention import (
    balanced_beam_chunks,
    beam_prefix_attention_fwd,
    build_suffix_indices,
    merge_prefix_suffix,
    shared_prefix_attention_fwd,
)
from sglang.kernels.ops.attention.decode_attention import (
    _decode_att_m_fwd,
    _decode_grouped_att_m_fwd,
)


@dataclass
class LayerBuffers:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    baseline_out: torch.Tensor
    beam_out: torch.Tensor
    baseline_partial: torch.Tensor
    baseline_lse: torch.Tensor
    empty_prefix_out: torch.Tensor
    empty_prefix_lse: torch.Tensor
    prefix_out: torch.Tensor
    prefix_lse: torch.Tensor
    suffix_out: torch.Tensor
    suffix_lse: torch.Tensor
    output_lse: torch.Tensor


@dataclass
class BenchmarkCase:
    rows: int
    seq_len: int
    shared_prefix: int
    group_beams: int
    paths: torch.Tensor
    seq_lens: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    group_rows: torch.Tensor
    group_widths: torch.Tensor
    prefix_lens: torch.Tensor
    row_prefix_lens: torch.Tensor
    suffix_indptr: torch.Tensor
    suffix_indices: torch.Tensor
    num_kv_splits: torch.Tensor
    layers: list[LayerBuffers]


def build_case(args) -> BenchmarkCase:
    if args.shared_prefix < 0:
        args.shared_prefix = args.seq_len - 1
    if not 1 <= args.shared_prefix < args.seq_len:
        raise ValueError("shared-prefix must be in [1, seq-len)")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")

    gqa = args.q_heads // args.kv_heads
    group_limit = args.group_beams or min(16, max(2, 16 // gqa))
    if group_limit < 2:
        raise ValueError("group-beams must be at least 2")
    if group_limit * gqa > 16:
        raise ValueError("group-beams times the GQA ratio must not exceed 16")

    rows = args.requests * args.beams
    paths = []
    groups = []
    prefix_lens = []
    slot = 0
    for request in range(args.requests):
        shared = list(range(slot, slot + args.shared_prefix))
        slot += args.shared_prefix
        request_rows = list(range(request * args.beams, (request + 1) * args.beams))
        for _ in request_rows:
            suffix = list(range(slot, slot + args.seq_len - args.shared_prefix))
            slot += len(suffix)
            paths.append(shared + suffix)
        for chunk in balanced_beam_chunks(request_rows, group_limit):
            groups.append(chunk)
            prefix_lens.append(args.shared_prefix)

    device = torch.device("cuda")
    paths_tensor = torch.tensor(paths, dtype=torch.int64, device=device)
    seq_lens = torch.full((rows,), args.seq_len, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(
        0,
        (rows + 1) * args.seq_len,
        args.seq_len,
        dtype=torch.int32,
        device=device,
    )
    kv_indices = paths_tensor.flatten().clone()

    group_beams = max(len(chunk) for chunk in groups)
    padded_groups = [
        chunk + [chunk[0]] * (group_beams - len(chunk)) for chunk in groups
    ]
    group_rows = torch.tensor(padded_groups, dtype=torch.int64, device=device)
    group_widths = torch.tensor(
        [len(chunk) for chunk in groups], dtype=torch.int32, device=device
    )
    prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    row_prefix_lens = torch.full(
        (rows,), args.shared_prefix, dtype=torch.int32, device=device
    )
    suffix_indptr = torch.empty(rows + 1, dtype=torch.int32, device=device)
    suffix_indices = torch.empty_like(kv_indices)
    build_suffix_indices(
        kv_indptr,
        kv_indices,
        row_prefix_lens,
        suffix_indptr,
        suffix_indices,
        max_seq_len=args.seq_len,
    )
    num_kv_splits = torch.ones(rows, dtype=torch.int32, device=device)

    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    layers = []
    for _ in range(args.layers):
        q = torch.randn(
            rows,
            args.q_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        k = torch.randn(
            slot,
            args.kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        v = torch.randn(k.shape, dtype=dtype, device=device, generator=generator)
        partial_shape = (rows, args.q_heads, 1, args.head_dim)
        layers.append(
            LayerBuffers(
                q=q,
                k=k,
                v=v,
                baseline_out=torch.empty_like(q),
                beam_out=torch.empty_like(q),
                baseline_partial=torch.empty(
                    partial_shape, dtype=torch.float32, device=device
                ),
                baseline_lse=torch.empty(
                    partial_shape[:-1], dtype=torch.float32, device=device
                ),
                empty_prefix_out=torch.zeros(
                    q.shape, dtype=torch.float32, device=device
                ),
                empty_prefix_lse=torch.full(
                    q.shape[:2], -float("inf"), dtype=torch.float32, device=device
                ),
                prefix_out=torch.empty(q.shape, dtype=torch.float32, device=device),
                prefix_lse=torch.empty(q.shape[:2], dtype=torch.float32, device=device),
                suffix_out=torch.empty(
                    partial_shape, dtype=torch.float32, device=device
                ),
                suffix_lse=torch.empty(
                    partial_shape[:-1], dtype=torch.float32, device=device
                ),
                output_lse=torch.empty(q.shape[:2], dtype=torch.float32, device=device),
            )
        )

    return BenchmarkCase(
        rows=rows,
        seq_len=args.seq_len,
        shared_prefix=args.shared_prefix,
        group_beams=group_beams,
        paths=paths_tensor,
        seq_lens=seq_lens,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        group_rows=group_rows,
        group_widths=group_widths,
        prefix_lens=prefix_lens_tensor,
        row_prefix_lens=row_prefix_lens,
        suffix_indptr=suffix_indptr,
        suffix_indices=suffix_indices,
        num_kv_splits=num_kv_splits,
        layers=layers,
    )


def baseline_attention(layer: LayerBuffers, case: BenchmarkCase, scale: float) -> None:
    stage = (
        _decode_att_m_fwd
        if layer.q.shape[1] == layer.k.shape[1]
        else _decode_grouped_att_m_fwd
    )
    stage(
        layer.q,
        layer.k,
        layer.v,
        layer.baseline_partial,
        layer.baseline_lse,
        case.kv_indptr,
        case.kv_indices,
        case.num_kv_splits,
        1,
        scale,
        0.0,
        page_size=1,
    )
    merge_prefix_suffix(
        layer.empty_prefix_out,
        layer.empty_prefix_lse,
        layer.baseline_partial,
        layer.baseline_lse,
        case.num_kv_splits,
        layer.baseline_out,
        layer.output_lse,
    )


def beam_attention(layer: LayerBuffers, case: BenchmarkCase, scale: float) -> None:
    beam_prefix_attention_fwd(
        layer.q,
        layer.k,
        layer.v,
        layer.beam_out,
        layer.output_lse,
        case.kv_indptr,
        case.kv_indices,
        case.group_rows,
        case.group_widths,
        case.prefix_lens,
        case.suffix_indptr,
        case.suffix_indices,
        case.num_kv_splits,
        layer.prefix_out,
        layer.prefix_lse,
        layer.suffix_out,
        layer.suffix_lse,
        group_beams=case.group_beams,
        max_kv_splits=1,
        sm_scale=scale,
    )


def suffix_attention(layer: LayerBuffers, case: BenchmarkCase, scale: float) -> None:
    stage = (
        _decode_att_m_fwd
        if layer.q.shape[1] == layer.k.shape[1]
        else _decode_grouped_att_m_fwd
    )
    stage(
        layer.q,
        layer.k,
        layer.v,
        layer.suffix_out,
        layer.suffix_lse,
        case.suffix_indptr,
        case.suffix_indices,
        case.num_kv_splits,
        1,
        scale,
        0.0,
        page_size=1,
    )


def reference_attention(
    layer: LayerBuffers, case: BenchmarkCase, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    q = layer.q[:rows].double()
    head_map = torch.arange(q.shape[1], device=q.device) // (
        q.shape[1] // layer.k.shape[1]
    )
    slots = case.paths[:rows]
    keys = layer.k[slots][:, :, head_map, :].double()
    values = layer.v[slots][:, :, head_map, :].double()
    scores = torch.einsum("bhd,bthd->bht", q, keys) / math.sqrt(q.shape[-1])
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.einsum("bht,bthd->bhd", scores.softmax(dim=-1), values)
    return output, lse


def measure_cuda(call, args, scrub):
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    calls_per_sample = 1
    measured = call
    if args.execution == "graph":
        calls_per_sample = args.graph_batch if scrub is None else 1
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls_per_sample):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        measured = graph.replay
        measured()
        torch.cuda.synchronize()

    samples = []
    for _ in range(args.repeats):
        if scrub is not None:
            scrub.add_(1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        measured()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / calls_per_sample)
    samples.sort()
    return {
        "median_us": statistics.median(samples),
        "p10_us": samples[int((len(samples) - 1) * 0.1)],
        "p90_us": samples[int((len(samples) - 1) * 0.9)],
    }


def run_benchmark(args):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("benchmark requires an NVIDIA CUDA GPU")
    if min(args.beams, args.requests, args.seq_len, args.layers) < 1:
        raise ValueError("shape parameters must be positive")
    case = build_case(args)
    scale = 1.0 / math.sqrt(args.head_dim)

    for layer in case.layers:
        baseline_attention(layer, case, scale)
        beam_attention(layer, case, scale)
    torch.cuda.synchronize()

    reference_rows = min(args.reference_rows, case.rows)
    tolerance = 0.01 if args.dtype == "bfloat16" else 0.002
    max_output_error = 0.0
    max_lse_error = 0.0
    for layer in case.layers:
        expected, expected_lse = reference_attention(layer, case, reference_rows)
        torch.testing.assert_close(
            layer.baseline_out[:reference_rows].double(),
            expected,
            atol=tolerance,
            rtol=tolerance,
        )
        torch.testing.assert_close(
            layer.beam_out[:reference_rows].double(),
            expected,
            atol=tolerance,
            rtol=tolerance,
        )
        torch.testing.assert_close(
            layer.output_lse[:reference_rows].double(),
            expected_lse,
            atol=1e-4,
            rtol=1e-4,
        )
        max_output_error = max(
            max_output_error,
            (layer.beam_out[:reference_rows].double() - expected).abs().max().item(),
        )
        max_lse_error = max(
            max_lse_error,
            (layer.output_lse[:reference_rows].double() - expected_lse)
            .abs()
            .max()
            .item(),
        )

    def prepare_baseline():
        case.kv_indptr[0].zero_()
        torch.cumsum(
            case.seq_lens,
            dim=0,
            dtype=case.kv_indptr.dtype,
            out=case.kv_indptr[1:],
        )
        case.kv_indices.copy_(case.paths.flatten())

    def prepare_beam():
        prepare_baseline()
        build_suffix_indices(
            case.kv_indptr,
            case.kv_indices,
            case.row_prefix_lens,
            case.suffix_indptr,
            case.suffix_indices,
            max_seq_len=case.seq_len,
        )

    def baseline_kernel():
        for layer in case.layers:
            baseline_attention(layer, case, scale)

    def beam_kernel():
        for layer in case.layers:
            beam_attention(layer, case, scale)

    def baseline_route():
        prepare_baseline()
        baseline_kernel()

    def beam_route():
        prepare_beam()
        beam_kernel()

    first = case.layers[0]

    def prefix_stage():
        shared_prefix_attention_fwd(
            first.q,
            first.k,
            first.v,
            case.kv_indptr,
            case.kv_indices,
            case.group_rows,
            case.group_widths,
            case.prefix_lens,
            first.prefix_out,
            first.prefix_lse,
            group_beams=case.group_beams,
            sm_scale=scale,
        )

    def suffix_stage():
        suffix_attention(first, case, scale)

    def merge_stage():
        merge_prefix_suffix(
            first.prefix_out,
            first.prefix_lse,
            first.suffix_out,
            first.suffix_lse,
            case.num_kv_splits,
            first.beam_out,
            first.output_lse,
        )

    scrub = (
        torch.empty(args.eviction_mib << 20, dtype=torch.uint8, device="cuda")
        if args.cache_mode == "scrub"
        else None
    )
    measurements = {
        "baseline_kernel": measure_cuda(baseline_kernel, args, scrub),
        "beam_kernel": measure_cuda(beam_kernel, args, scrub),
        "beam_prefix_stage": measure_cuda(prefix_stage, args, scrub),
        "beam_suffix_stage": measure_cuda(suffix_stage, args, scrub),
        "beam_merge_stage": measure_cuda(merge_stage, args, scrub),
        "baseline_route_e2e": measure_cuda(baseline_route, args, scrub),
        "beam_route_e2e": measure_cuda(beam_route, args, scrub),
    }
    baseline_kernel_us = measurements["baseline_kernel"]["median_us"]
    beam_kernel_us = measurements["beam_kernel"]["median_us"]
    baseline_e2e_us = measurements["baseline_route_e2e"]["median_us"]
    beam_e2e_us = measurements["beam_route_e2e"]["median_us"]
    return {
        "status": "completed",
        "parameters": vars(args),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "correctness": {
            "reference_rows": reference_rows,
            "max_output_abs_error": max_output_error,
            "max_lse_abs_error": max_lse_error,
        },
        "layout": {
            "rows": case.rows,
            "groups": case.group_rows.shape[0],
            "group_beams": case.group_beams,
            "shared_prefix": case.shared_prefix,
            "suffix_len": case.seq_len - case.shared_prefix,
            "baseline_kv_references": case.rows * case.seq_len,
            "beam_kv_references": (
                case.group_rows.shape[0] * case.shared_prefix
                + case.rows * (case.seq_len - case.shared_prefix)
            ),
            "production_backend_eligible": (
                args.q_heads == args.kv_heads
                and args.dtype == "float16"
                and args.head_dim == 128
                and min(case.group_widths.tolist()) >= 8
                and case.shared_prefix >= 64
                and min(
                    (width - 1) * case.shared_prefix
                    for width in case.group_widths.tolist()
                )
                >= 1024
            ),
        },
        "measurements": measurements,
        "speedup": {
            "kernel": baseline_kernel_us / beam_kernel_us,
            "route_e2e": baseline_e2e_us / beam_e2e_us,
        },
        "scope": {
            "kernel": "prebuilt metadata; complete attention output",
            "route_e2e": (
                "common row-wise paths -> KV metadata -> complete attention output"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bench = subparsers.add_parser("benchmark")
    bench.add_argument("--beams", type=int, default=32)
    bench.add_argument("--requests", type=int, default=1)
    bench.add_argument("--seq-len", type=int, default=128)
    bench.add_argument("--shared-prefix", type=int, default=-1)
    bench.add_argument("--q-heads", type=int, default=8)
    bench.add_argument("--kv-heads", type=int, default=8)
    bench.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    bench.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    bench.add_argument("--layers", type=int, default=1)
    bench.add_argument("--group-beams", type=int)
    bench.add_argument("--execution", choices=("graph", "eager"), default="graph")
    bench.add_argument("--graph-batch", type=int, default=64)
    bench.add_argument("--cache-mode", choices=("warm", "scrub"), default="warm")
    bench.add_argument("--eviction-mib", type=int, default=256)
    bench.add_argument("--warmup", type=int, default=20)
    bench.add_argument("--repeats", type=int, default=100)
    bench.add_argument("--reference-rows", type=int, default=2)
    bench.add_argument("--seed", type=int, default=43)
    bench.add_argument("--output")
    args = parser.parse_args()

    if args.output and Path(args.output).exists():
        parser.error("output file already exists")
    result = run_benchmark(args)
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
