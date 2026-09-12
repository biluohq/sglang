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
    QUERY_ROWS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    USE_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    group = tl.program_id(0)
    kv_head = tl.program_id(1)
    request_split = tl.program_id(2).to(tl.int64)
    request = request_split // NUM_SPLITS
    split = request_split % NUM_SPLITS
    GQA: tl.constexpr = Q_HEADS // KV_HEADS
    m = tl.arange(0, BLOCK_M)
    query_indices = group * QUERY_ROWS + m
    beams = query_indices // GQA
    rows = request * BEAMS + beams
    heads = kv_head * GQA + query_indices % GQA
    row_in_bounds = (m < QUERY_ROWS) & (beams < BEAMS)
    query_valid = tl.load(QueryValid + rows, row_in_bounds, other=0)
    active = row_in_bounds & query_valid
    d = tl.arange(0, DIM)
    q = tl.load(
        Q + (rows[:, None] * Q_HEADS + heads[:, None]) * DIM + d[None, :],
        active[:, None],
        other=0.0,
    )
    end = tl.load(ScanEnds + request)
    split_size = tl.cdiv(tl.cdiv(end, BLOCK_N), NUM_SPLITS) * BLOCK_N
    split_start = split * split_size
    split_end = tl.minimum(split_start + split_size, end)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DIM), tl.float32)
    for start in range(split_start, split_end, BLOCK_N):
        slots = start + tl.arange(0, BLOCK_N)
        in_bounds = (slots < split_end) & (slots < CAPACITY)
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
        if USE_DOT:
            scores = tl.dot(q, tl.trans(k)) * SCALE
        else:
            scores = (
                tl.sum(q.to(tl.float32)[:, None, :] * k.to(tl.float32)[None, :, :], 2)
                * SCALE
            )
        scores = tl.where(
            visible & valid[None, :] & active[:, None], scores, -float("inf")
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # Entirely masked rows must not evaluate exp(-inf - -inf).
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        alpha = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(scores - safe_maximum[:, None])
        if USE_DOT:
            contribution = tl.dot(probabilities.to(v.dtype), v)
        else:
            contribution = tl.sum(
                probabilities[:, :, None] * v.to(tl.float32)[None, :, :], 1
            )
        accumulator = accumulator * alpha[:, None] + contribution
        denominator = denominator * alpha + tl.sum(probabilities, axis=1)
        maximum = new_maximum
    divisor = tl.where(denominator > 0.0, denominator, 1.0)
    output = accumulator / divisor[:, None]
    lse = tl.where(denominator > 0.0, maximum + tl.log(divisor), -float("inf"))
    output_rows = split * TOTAL_ROWS + rows
    tl.store(
        Out + (output_rows[:, None] * Q_HEADS + heads[:, None]) * DIM + d[None, :],
        output,
        row_in_bounds[:, None],
    )
    tl.store(LSE + output_rows * Q_HEADS + heads, lse, row_in_bounds)


@triton.jit
def _merge_beam_splits(
    Partial,
    PartialLSE,
    Out,
    LSE,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    splits = tl.arange(0, BLOCK_SPLITS)
    d = tl.arange(0, DIM)
    indices = (splits * ROWS + row) * HEADS + head
    scores = tl.load(PartialLSE + indices, splits < NUM_SPLITS, other=-float("inf"))
    maximum = tl.max(scores, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(scores - safe_maximum)
    denominator = tl.sum(weights, 0)
    partial = tl.load(
        Partial + indices[:, None] * DIM + d[None, :],
        splits[:, None] < NUM_SPLITS,
        other=0.0,
    )
    output = tl.sum(weights[:, None] * partial, 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    merged_lse = tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf"))
    tl.store(Out + (row * HEADS + head) * DIM + d, output)
    tl.store(LSE + row * HEADS + head, merged_lse)


@triton.jit
def _reparent_beam_paths(
    Previous,
    Parents,
    NewSlots,
    Paths,
    BEAMS: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    LENGTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    parent = tl.load(Parents + row).to(tl.int64) + row // BEAMS * BEAMS
    positions = tl.arange(0, BLOCK_T)
    slots = tl.load(
        Previous + parent * PATH_WIDTH + positions, positions < LENGTH - 1, other=0
    )
    current = tl.load(NewSlots + row)
    slots = tl.where(positions == LENGTH - 1, current, slots)
    tl.store(Paths + row * PATH_WIDTH + positions, slots, positions < PATH_WIDTH)


@triton.jit
def _group_prefix_metadata(
    Paths,
    Order,
    SeqLens,
    PrefixLens,
    RowPrefix,
    RowGroup,
    BEAMS: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    GROUP_BEAMS: tl.constexpr,
    GROUPS_PER_REQUEST: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_B: tl.constexpr,
    MIN_PREFIX: tl.constexpr,
    MIN_SAVED: tl.constexpr,
    PROMPT_LEN: tl.constexpr,
):
    group = tl.program_id(0)
    request = group // GROUPS_PER_REQUEST
    local_group = group % GROUPS_PER_REQUEST
    start = local_group * GROUP_BEAMS
    count = tl.minimum(GROUP_BEAMS, BEAMS - start)
    first = tl.load(Order + request * BEAMS + start).to(tl.int64)
    last = tl.load(Order + request * BEAMS + start + count - 1).to(tl.int64)
    length = tl.load(SeqLens + first) - PROMPT_LEN
    positions = tl.arange(0, BLOCK_T)
    a = tl.load(
        Paths + first * PATH_WIDTH + PROMPT_LEN + positions,
        positions < length,
        other=-1,
    )
    b = tl.load(
        Paths + last * PATH_WIDTH + PROMPT_LEN + positions,
        positions < length,
        other=-2,
    )
    common = tl.min(tl.where((positions < length) & (a != b), positions, length), 0)
    use_prefix = (
        (count > 1) & (common >= MIN_PREFIX) & ((count - 1) * common >= MIN_SAVED)
    )
    common = tl.where(use_prefix, common, 0)
    tl.store(PrefixLens + group, common)
    offsets = tl.arange(0, BLOCK_B)
    members = start + offsets
    member_valid = (offsets < GROUP_BEAMS) & (members < BEAMS)
    rows = tl.load(Order + request * BEAMS + members, member_valid, other=0)
    tl.store(RowPrefix + rows, common, member_valid)
    tl.store(RowGroup + rows, group, member_valid)


@triton.jit
def _suffix_indices(
    Paths,
    RowPrefix,
    Indptr,
    Indices,
    PATH_WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PROMPT_LEN: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    begin = tl.load(Indptr + row)
    length = tl.load(Indptr + row + 1) - begin
    prefix = tl.load(RowPrefix + row)
    offsets = tl.arange(0, BLOCK_T)
    positions = tl.where(offsets < PROMPT_LEN, offsets, prefix + offsets)
    slots = tl.load(Paths + row * PATH_WIDTH + positions, offsets < length, other=0)
    tl.store(Indices + begin + offsets, slots, offsets < length)


@triton.jit
def _shared_prefix_attention(
    Q,
    K,
    V,
    Paths,
    Order,
    PrefixLens,
    Active,
    Partial,
    PartialLSE,
    BEAMS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP_BEAMS: tl.constexpr,
    GROUPS_PER_REQUEST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREFIX_SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    PROMPT_LEN: tl.constexpr,
):
    group = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2).to(tl.int64)
    request = group // GROUPS_PER_REQUEST
    local_group = group % GROUPS_PER_REQUEST
    GQA: tl.constexpr = Q_HEADS // KV_HEADS
    m = tl.arange(0, BLOCK_M)
    members = local_group * GROUP_BEAMS + m // GQA
    valid_row = (m < GROUP_BEAMS * GQA) & (members < BEAMS)
    rows = tl.load(Order + request * BEAMS + members, valid_row, other=0).to(tl.int64)
    active = tl.load(Active + rows, valid_row, other=0) & valid_row
    heads = kv_head * GQA + m % GQA
    dims = tl.arange(0, DIM)
    q = tl.load(
        Q + (rows[:, None] * Q_HEADS + heads[:, None]) * DIM + dims[None, :],
        active[:, None],
        other=0.0,
    )
    first = tl.load(Order + request * BEAMS + local_group * GROUP_BEAMS).to(tl.int64)
    length = tl.load(PrefixLens + group)
    split_size = tl.cdiv(tl.cdiv(length, BLOCK_N), PREFIX_SPLITS) * BLOCK_N
    begin = split * split_size
    end = tl.minimum(begin + split_size, length)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, DIM), tl.float32)
    for start in range(begin, end, BLOCK_N):
        positions = start + tl.arange(0, BLOCK_N)
        slots = tl.load(
            Paths + first * PATH_WIDTH + PROMPT_LEN + positions,
            positions < end,
            other=0,
        ).to(tl.int64)
        offsets = (slots[:, None] * KV_HEADS + kv_head) * DIM + dims[None, :]
        k = tl.load(K + offsets, positions[:, None] < end, other=0.0)
        v = tl.load(V + offsets, positions[:, None] < end, other=0.0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(
            active[:, None] & (positions[None, :] < end), scores, -float("inf")
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        weights = tl.exp(scores - safe_maximum[:, None])
        alpha = tl.exp(maximum - safe_maximum)
        acc = acc * alpha[:, None] + tl.dot(weights.to(v.dtype), v)
        denominator = denominator * alpha + tl.sum(weights, 1)
        maximum = new_maximum
    divisor = tl.where(denominator > 0, denominator, 1.0)
    out_row = split * TOTAL_ROWS + rows
    tl.store(
        Partial + (out_row[:, None] * Q_HEADS + heads[:, None]) * DIM + dims[None, :],
        acc / divisor[:, None],
        valid_row[:, None],
    )
    tl.store(
        PartialLSE + out_row * Q_HEADS + heads,
        tl.where(denominator > 0, maximum + tl.log(divisor), -float("inf")),
        valid_row,
    )


@triton.jit
def _merge_prefix_suffix(
    PrefixOut,
    PrefixLSE,
    SuffixOut,
    SuffixLSE,
    Indptr,
    Active,
    Out,
    LSE,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PREFIX_SPLITS: tl.constexpr,
    SUFFIX_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    active = tl.load(Active + row)
    ids = tl.arange(0, BLOCK_SPLITS)
    dims = tl.arange(0, DIM)
    pvalid = ids < PREFIX_SPLITS
    pindex = (ids * ROWS + row) * HEADS + head
    plse = tl.load(PrefixLSE + pindex, pvalid & active, other=-float("inf"))
    pout = tl.load(
        PrefixOut + pindex[:, None] * DIM + dims[None, :],
        pvalid[:, None] & active,
        other=0.0,
    )
    suffix_len = tl.load(Indptr + row + 1) - tl.load(Indptr + row)
    suffix_chunk = tl.cdiv(tl.cdiv(suffix_len, SUFFIX_SPLITS), 32) * 32
    svalid = (ids < SUFFIX_SPLITS) & (ids * suffix_chunk < suffix_len) & active
    sindex = (row * HEADS + head) * SUFFIX_SPLITS + ids
    slse = tl.load(SuffixLSE + sindex, svalid, other=-float("inf"))
    sout = tl.load(
        SuffixOut + sindex[:, None] * DIM + dims[None, :], svalid[:, None], other=0.0
    )
    maximum = tl.maximum(tl.max(plse, 0), tl.max(slse, 0))
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    pw, sw = tl.exp(plse - safe_maximum), tl.exp(slse - safe_maximum)
    denominator = tl.sum(pw, 0) + tl.sum(sw, 0)
    acc = tl.sum(pw[:, None] * pout + sw[:, None] * sout, 0)
    divisor = tl.where(denominator > 0, denominator, 1.0)
    tl.store(Out + (row * HEADS + head) * DIM + dims, acc / divisor)
    tl.store(
        LSE + row * HEADS + head,
        tl.where(denominator > 0, maximum + tl.log(divisor), -float("inf")),
    )


def group_prefix_metadata(
    paths,
    order,
    seq_lens,
    prefix_lens,
    row_prefix,
    row_group,
    indptr,
    indices,
    lengths,
    *,
    beams,
    group_beams,
    min_prefix=0,
    min_saved=0,
    prompt_len=0,
):
    """Paths and tree order are lockstep per request; all buffers stay caller-owned."""
    groups = triton.cdiv(beams, group_beams)
    _group_prefix_metadata[(order.numel() // beams * groups,)](
        paths,
        order,
        seq_lens,
        prefix_lens,
        row_prefix,
        row_group,
        BEAMS=beams,
        PATH_WIDTH=paths.shape[1],
        GROUP_BEAMS=group_beams,
        GROUPS_PER_REQUEST=groups,
        BLOCK_T=triton.next_power_of_2(max(1, paths.shape[1] - prompt_len)),
        BLOCK_B=triton.next_power_of_2(group_beams),
        MIN_PREFIX=min_prefix,
        MIN_SAVED=min_saved,
        PROMPT_LEN=prompt_len,
    )
    torch.sub(seq_lens, row_prefix, out=lengths)
    indptr[0].zero_()
    torch.cumsum(lengths, 0, dtype=indptr.dtype, out=indptr[1:])
    _suffix_indices[(paths.shape[0],)](
        paths,
        row_prefix,
        indptr,
        indices,
        PATH_WIDTH=paths.shape[1],
        BLOCK_T=triton.next_power_of_2(paths.shape[1]),
        PROMPT_LEN=prompt_len,
    )


def shared_prefix_attention(
    q,
    k,
    v,
    paths,
    order,
    prefix_lens,
    active,
    partial,
    partial_lse,
    *,
    beams,
    group_beams,
    tile,
    splits,
    prompt_len=0,
    num_warps=4,
    num_stages=2,
):
    gqa = q.shape[1] // k.shape[1]
    groups = triton.cdiv(beams, group_beams)
    _shared_prefix_attention[(order.numel() // beams * groups, k.shape[1], splits)](
        q,
        k,
        v,
        paths,
        order,
        prefix_lens,
        active,
        partial,
        partial_lse,
        BEAMS=beams,
        TOTAL_ROWS=q.shape[0],
        PATH_WIDTH=paths.shape[1],
        Q_HEADS=q.shape[1],
        KV_HEADS=k.shape[1],
        DIM=q.shape[2],
        GROUP_BEAMS=group_beams,
        GROUPS_PER_REQUEST=groups,
        BLOCK_M=max(16, triton.next_power_of_2(group_beams * gqa)),
        BLOCK_N=tile,
        PREFIX_SPLITS=splits,
        SCALE=1.0 / math.sqrt(q.shape[2]),
        PROMPT_LEN=prompt_len,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def merge_prefix_suffix(
    prefix, prefix_lse, suffix, suffix_lse, indptr, active, out, lse
):
    ps, ss = prefix.shape[0], suffix.shape[2]
    _merge_prefix_suffix[(out.shape[0], out.shape[1])](
        prefix,
        prefix_lse,
        suffix,
        suffix_lse,
        indptr,
        active,
        out,
        lse,
        ROWS=out.shape[0],
        HEADS=out.shape[1],
        DIM=out.shape[2],
        PREFIX_SPLITS=ps,
        SUFFIX_SPLITS=ss,
        BLOCK_SPLITS=triton.next_power_of_2(max(ps, ss)),
        num_warps=4,
    )


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
    query_tile=16,
    compute="dot",
    num_splits=1,
    partial_out=None,
    partial_lse=None,
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
    if group_beams not in (1, 2, 4, 8, 16) or tile_tokens not in (16, 32, 64, 128):
        raise ValueError("group_beams must be 1/2/4/8/16 and tile_tokens 16/32/64/128")
    if compute not in ("dot", "simt") or num_splits not in (1, 2, 4, 8):
        raise ValueError("compute must be dot/simt and num_splits 1/2/4/8")
    if query_tile not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("query_tile must be a power of two up to 128")
    if compute == "simt" and query_tile > 8:
        raise ValueError("SIMT query_tile must be at most 8")
    group_queries = min(group_beams, beams_per_request) * (heads // kv_heads)
    query_rows = min(group_queries, query_tile)
    block_m = triton.next_power_of_2(query_rows)
    if compute == "dot":
        block_m = max(16, block_m)
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
    if num_splits > 1:
        if (
            partial_out is None
            or partial_lse is None
            or partial_out.shape != (num_splits, rows, heads, dim)
            or partial_lse.shape != (num_splits, rows, heads)
            or partial_out.dtype != torch.float32
            or partial_lse.dtype != torch.float32
        ):
            raise ValueError("split KV requires correctly sized FP32 partial buffers")
        tensors += (partial_out, partial_lse)
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
        (
            triton.cdiv(beams_per_request * (heads // kv_heads), query_rows),
            kv_heads,
            requests * num_splits,
        )
    ](
        q,
        k,
        v,
        visible,
        valid_slots,
        query_valid,
        scan_ends,
        partial_out if num_splits > 1 else out,
        partial_lse if num_splits > 1 else lse,
        BEAMS=beams_per_request,
        CAPACITY=capacity,
        Q_HEADS=heads,
        KV_HEADS=kv_heads,
        DIM=dim,
        QUERY_ROWS=query_rows,
        TOTAL_ROWS=rows,
        NUM_SPLITS=num_splits,
        USE_DOT=compute == "dot",
        BLOCK_M=block_m,
        BLOCK_N=tile_tokens,
        SCALE=scale,
        num_warps=4,
        num_stages=1,
    )
    if num_splits > 1:
        _merge_beam_splits[(rows, heads)](
            partial_out,
            partial_lse,
            out,
            lse,
            ROWS=rows,
            HEADS=heads,
            DIM=dim,
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=4,
        )
    return out, lse
