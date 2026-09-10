"""Experimental shared-tile beam attention, not registered with serving backends."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _beam_decode_attention(
    Q,
    K,
    V,
    Visible,
    ValidSlots,
    QueryValid,
    ScanEnds,
    Out,
    LSE,
    BEAMS: tl.constexpr,
    CAPACITY: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP_BEAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    group = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.program_id(2).to(tl.int64)
    GQA: tl.constexpr = Q_HEADS // KV_HEADS
    m = tl.arange(0, BLOCK_M)
    beams = group * GROUP_BEAMS + m // GQA
    rows = request * BEAMS + beams
    heads = kv_head * GQA + m % GQA
    row_in_bounds = (m < GROUP_BEAMS * GQA) & (beams < BEAMS)
    query_valid = tl.load(QueryValid + rows, row_in_bounds, other=0)
    active = row_in_bounds & query_valid
    d = tl.arange(0, DIM)
    q = tl.load(
        Q + (rows[:, None] * Q_HEADS + heads[:, None]) * DIM + d[None, :],
        active[:, None],
        other=0.0,
    )
    end = tl.load(ScanEnds + request)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DIM), tl.float32)
    for start in range(0, end, BLOCK_N):
        slots = start + tl.arange(0, BLOCK_N)
        in_bounds = (slots < end) & (slots < CAPACITY)
        valid = (
            tl.load(ValidSlots + request * CAPACITY + slots, in_bounds, other=0)
            & in_bounds
        )
        offsets = ((request * CAPACITY + slots) * KV_HEADS + kv_head)[
            :, None
        ] * DIM + d[None, :]
        k = tl.load(K + offsets, valid[:, None], other=0.0)
        v = tl.load(V + offsets, valid[:, None], other=0.0)
        visible = tl.load(
            Visible + rows[:, None] * CAPACITY + slots[None, :],
            active[:, None] & in_bounds[None, :],
            other=0,
        )
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(
            visible & valid[None, :] & active[:, None], scores, -float("inf")
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # Entirely masked rows must not evaluate exp(-inf - -inf).
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        alpha = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(scores - safe_maximum[:, None])
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities.to(v.dtype), v
        )
        denominator = denominator * alpha + tl.sum(probabilities, axis=1)
        maximum = new_maximum
    divisor = tl.where(denominator > 0.0, denominator, 1.0)
    output = accumulator / divisor[:, None]
    lse = tl.where(denominator > 0.0, maximum + tl.log(divisor), -float("inf"))
    tl.store(
        Out + (rows[:, None] * Q_HEADS + heads[:, None]) * DIM + d[None, :],
        output,
        row_in_bounds[:, None],
    )
    tl.store(LSE + rows * Q_HEADS + heads, lse, row_in_bounds)


def beam_decode_attention_fwd(
    q,
    k,
    v,
    visible,
    valid_slots,
    query_valid,
    scan_ends,
    out,
    lse,
    *,
    beams_per_request,
    group_beams=8,
    tile_tokens=64,
    scale=None,
):
    """NHD buffers; scan_ends must stay in [0, capacity], with holes marked invalid."""
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("expected Q [rows, q_heads, dim] and matching NHD K/V")
    if scan_ends.ndim != 1 or not scan_ends.numel() or beams_per_request < 1:
        raise ValueError("expected nonempty scan_ends and positive beams_per_request")
    requests = scan_ends.numel()
    rows, heads, dim = q.shape
    if rows != requests * beams_per_request or k.shape[0] % requests:
        raise ValueError("Q rows and KV capacity must match the request count")
    capacity, kv_heads = k.shape[0] // requests, k.shape[1]
    if capacity < 1 or kv_heads < 1 or heads < 1 or heads % kv_heads:
        raise ValueError("expected positive capacity and integral GQA head ratio")
    if dim not in (64, 128) or k.shape[2] != dim:
        raise ValueError("the prototype supports head_dim 64 or 128")
    if group_beams not in (4, 8, 16) or tile_tokens not in (32, 64, 128):
        raise ValueError("group_beams must be 4/8/16 and tile_tokens 32/64/128")
    group_queries = group_beams * (heads // kv_heads)
    if group_queries > 128:
        raise ValueError("group_beams times the GQA ratio must not exceed 128")
    if (
        visible.shape != (rows, capacity)
        or valid_slots.shape != (requests, capacity)
        or query_valid.shape != (rows,)
        or out.shape != q.shape
        or lse.shape != (rows, heads)
    ):
        raise ValueError("metadata or output shapes do not match Q/K/V")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("the prototype supports FP16/BF16 Q/K/V")
    if any(t.dtype != q.dtype for t in (k, v, out)) or lse.dtype != torch.float32:
        raise ValueError("Q/K/V/output dtypes must match, and LSE must be FP32")
    if (
        any(t.dtype != torch.bool for t in (visible, valid_slots, query_valid))
        or scan_ends.dtype != torch.int32
    ):
        raise ValueError("visibility buffers must be bool and scan_ends must be int32")
    tensors = (q, k, v, visible, valid_slots, query_valid, scan_ends, out, lse)
    if not q.is_cuda or any(
        t.device != q.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("all buffers must be contiguous on the same CUDA device")
    if torch.version.hip is not None:
        raise ValueError("this experimental kernel targets NVIDIA CUDA only")
    scale = 1.0 / math.sqrt(dim) if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    _beam_decode_attention[
        (triton.cdiv(beams_per_request, group_beams), kv_heads, requests)
    ](
        q,
        k,
        v,
        visible,
        valid_slots,
        query_valid,
        scan_ends,
        out,
        lse,
        BEAMS=beams_per_request,
        CAPACITY=capacity,
        Q_HEADS=heads,
        KV_HEADS=kv_heads,
        DIM=dim,
        GROUP_BEAMS=group_beams,
        BLOCK_M=max(16, triton.next_power_of_2(group_queries)),
        BLOCK_N=tile_tokens,
        SCALE=scale,
        num_warps=4,
        num_stages=1,
    )
    return out, lse
