# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam-aware decode attention kernels used by BeamTritonAttnBackend.

The backend divides every beam row into:

* a prefix shared by all rows in a group, loaded once per KV tile; and
* a private suffix processed by the production Triton gather kernel.

The two online-softmax states are merged into the final output. This module
does not contain the retired visibility-mask/shared-pool scan implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import (
    _decode_att_m_fwd,
    _decode_grouped_att_m_fwd,
    _extract_kv_strides,
)


def balanced_beam_chunks(rows: list[int], limit: int) -> list[list[int]]:
    """Split rows into near-equal chunks without producing singleton tails."""
    if len(rows) < 2:
        return []
    num_chunks = (len(rows) + limit - 1) // limit
    small, extra = divmod(len(rows), num_chunks)
    chunks = []
    offset = 0
    for index in range(num_chunks):
        size = small + (index < extra)
        chunks.append(rows[offset : offset + size])
        offset += size
    return chunks


@triton.jit
def _copy_suffix_indices(
    KvIndptr,
    KvIndices,
    RowPrefixLens,
    SuffixIndptr,
    SuffixIndices,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    source_begin = tl.load(KvIndptr + row)
    source_end = tl.load(KvIndptr + row + 1)
    prefix_len = tl.load(RowPrefixLens + row)
    target_begin = tl.load(SuffixIndptr + row)
    suffix_len = source_end - source_begin - prefix_len
    offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    slots = tl.load(
        KvIndices + source_begin + prefix_len + offsets,
        offsets < suffix_len,
        other=0,
    )
    tl.store(SuffixIndices + target_begin + offsets, slots, offsets < suffix_len)


@triton.jit
def _beam_prefix_attention(
    Q,
    K,
    V,
    KvIndptr,
    KvIndices,
    GroupRows,
    GroupWidths,
    PrefixLens,
    PrefixOut,
    PrefixLSE,
    K_SLOT_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    V_SLOT_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
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
    width = tl.load(GroupWidths + group)
    gqa: tl.constexpr = Q_HEADS // KV_HEADS

    query_offsets = tl.arange(0, BLOCK_M)
    member_offsets = query_offsets // gqa
    valid_query = (member_offsets < width) & (query_offsets < GROUP_BEAMS * gqa)
    rows = tl.load(
        GroupRows + group * GROUP_BEAMS + member_offsets,
        valid_query,
        other=0,
    ).to(tl.int64)
    heads = kv_head * gqa + query_offsets % gqa
    dims = tl.arange(0, DIM)
    q = tl.load(
        Q + (rows[:, None] * Q_HEADS + heads[:, None]) * DIM + dims[None, :],
        valid_query[:, None],
        other=0.0,
    )

    source_row = tl.load(GroupRows + group * GROUP_BEAMS).to(tl.int64)
    source_begin = tl.load(KvIndptr + source_row)
    prefix_len = tl.load(PrefixLens + group)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DIM), tl.float32)

    for start in range(0, prefix_len, BLOCK_N):
        positions = start + tl.arange(0, BLOCK_N)
        valid_token = positions < prefix_len
        slots = tl.load(
            KvIndices + source_begin + positions,
            valid_token,
            other=0,
        ).to(tl.int64)
        k = tl.load(
            K
            + slots[:, None] * K_SLOT_STRIDE
            + kv_head * K_HEAD_STRIDE
            + dims[None, :],
            valid_token[:, None],
            other=0.0,
        )
        v = tl.load(
            V
            + slots[:, None] * V_SLOT_STRIDE
            + kv_head * V_HEAD_STRIDE
            + dims[None, :],
            valid_token[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(
            valid_query[:, None] & valid_token[None, :],
            scores,
            -float("inf"),
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        old_scale = tl.exp(maximum - safe_maximum)
        weights = tl.exp(scores - safe_maximum[:, None])
        contribution = tl.dot(weights.to(v.dtype), v)
        accumulator = accumulator * old_scale[:, None] + contribution
        denominator = denominator * old_scale + tl.sum(weights, axis=1)
        maximum = new_maximum

    divisor = tl.where(denominator > 0.0, denominator, 1.0)
    output = accumulator / divisor[:, None]
    lse = tl.where(denominator > 0.0, maximum + tl.log(divisor), -float("inf"))
    tl.store(
        PrefixOut + (rows[:, None] * Q_HEADS + heads[:, None]) * DIM + dims[None, :],
        output,
        valid_query[:, None],
    )
    tl.store(
        PrefixLSE + rows * Q_HEADS + heads,
        lse,
        valid_query,
    )


@triton.jit
def _merge_prefix_suffix(
    PrefixOut,
    PrefixLSE,
    SuffixOut,
    SuffixLSE,
    NumSuffixSplits,
    Out,
    OutLSE,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    MAX_SUFFIX_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    dims = tl.arange(0, DIM)
    splits = tl.arange(0, BLOCK_SPLITS)

    prefix_lse = tl.load(PrefixLSE + row * HEADS + head)
    prefix_out = tl.load(
        PrefixOut + (row * HEADS + head) * DIM + dims,
    ).to(tl.float32)

    num_suffix_splits = tl.load(NumSuffixSplits + row)
    valid_split = splits < num_suffix_splits
    suffix_state = (row * HEADS + head) * MAX_SUFFIX_SPLITS + splits
    suffix_lse = tl.load(SuffixLSE + suffix_state, valid_split, other=-float("inf"))
    suffix_out = tl.load(
        SuffixOut + suffix_state[:, None] * DIM + dims[None, :],
        valid_split[:, None],
        other=0.0,
    ).to(tl.float32)

    maximum = tl.maximum(prefix_lse, tl.max(suffix_lse, axis=0))
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    prefix_weight = tl.exp(prefix_lse - safe_maximum)
    suffix_weights = tl.exp(suffix_lse - safe_maximum)
    denominator = prefix_weight + tl.sum(suffix_weights, axis=0)
    accumulator = prefix_weight * prefix_out + tl.sum(
        suffix_weights[:, None] * suffix_out, axis=0
    )
    divisor = tl.where(denominator > 0.0, denominator, 1.0)
    tl.store(Out + (row * HEADS + head) * DIM + dims, accumulator / divisor)
    tl.store(
        OutLSE + row * HEADS + head,
        tl.where(denominator > 0.0, maximum + tl.log(divisor), -float("inf")),
    )


def build_suffix_indices(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    row_prefix_lens: torch.Tensor,
    suffix_indptr: torch.Tensor,
    suffix_indices: torch.Tensor,
    *,
    max_seq_len: int,
) -> None:
    """Build packed per-row suffix indices from the production KV index stream."""
    rows = row_prefix_lens.numel()
    suffix_lens = kv_indptr[1 : rows + 1] - kv_indptr[:rows] - row_prefix_lens
    suffix_indptr[0].zero_()
    torch.cumsum(suffix_lens, dim=0, dtype=suffix_indptr.dtype, out=suffix_indptr[1:])
    block_t = 256
    _copy_suffix_indices[(rows, triton.cdiv(max_seq_len, block_t))](
        kv_indptr,
        kv_indices,
        row_prefix_lens,
        suffix_indptr,
        suffix_indices,
        BLOCK_T=block_t,
    )


def shared_prefix_attention_fwd(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    group_rows: torch.Tensor,
    group_widths: torch.Tensor,
    prefix_lens: torch.Tensor,
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    *,
    group_beams: int,
    sm_scale: float,
    tile_tokens: int = 32,
) -> None:
    _, q_heads, dim = q.shape
    kv_heads = k_buffer.shape[1]
    gqa = q_heads // kv_heads
    query_rows = group_beams * gqa
    block_m = max(16, triton.next_power_of_2(query_rows))
    k_slot_stride, k_head_stride, _, _ = _extract_kv_strides(k_buffer, 1)
    v_slot_stride, v_head_stride, _, _ = _extract_kv_strides(v_buffer, 1)
    _beam_prefix_attention[(group_rows.shape[0], kv_heads)](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        group_rows,
        group_widths,
        prefix_lens,
        prefix_out,
        prefix_lse,
        K_SLOT_STRIDE=k_slot_stride,
        K_HEAD_STRIDE=k_head_stride,
        V_SLOT_STRIDE=v_slot_stride,
        V_HEAD_STRIDE=v_head_stride,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        DIM=dim,
        GROUP_BEAMS=group_beams,
        BLOCK_M=block_m,
        BLOCK_N=tile_tokens,
        SCALE=sm_scale,
        num_warps=4,
        num_stages=2,
    )


def merge_prefix_suffix(
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_out: torch.Tensor,
    suffix_lse: torch.Tensor,
    suffix_num_kv_splits: torch.Tensor,
    out: torch.Tensor,
    out_lse: torch.Tensor,
) -> None:
    rows, heads, dim = out.shape
    max_kv_splits = suffix_out.shape[2]
    _merge_prefix_suffix[(rows, heads)](
        prefix_out,
        prefix_lse,
        suffix_out,
        suffix_lse,
        suffix_num_kv_splits,
        out,
        out_lse,
        HEADS=heads,
        DIM=dim,
        MAX_SUFFIX_SPLITS=max_kv_splits,
        BLOCK_SPLITS=triton.next_power_of_2(max_kv_splits),
        num_warps=4,
    )


def beam_prefix_attention_fwd(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    out: torch.Tensor,
    out_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    group_rows: torch.Tensor,
    group_widths: torch.Tensor,
    prefix_lens: torch.Tensor,
    suffix_indptr: torch.Tensor,
    suffix_indices: torch.Tensor,
    suffix_num_kv_splits: torch.Tensor,
    prefix_out: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_out: torch.Tensor,
    suffix_lse: torch.Tensor,
    *,
    group_beams: int,
    max_kv_splits: int,
    sm_scale: float,
    logit_cap: float = 0.0,
    tile_tokens: int = 32,
) -> torch.Tensor:
    """Execute one complete beam-aware decode attention layer.

    Metadata and workspaces are caller-owned so this function is suitable for
    both backend integration and allocation-free kernel benchmarking.
    """
    if q.ndim != 3 or k_buffer.ndim != 3 or v_buffer.ndim != 3:
        raise ValueError("expected Q/K/V in NHD layout")
    if q.shape[-1] != k_buffer.shape[-1] or k_buffer.shape != v_buffer.shape:
        raise ValueError("beam attention currently requires equal Q/K/V head dims")
    if q.shape[-1] not in (64, 128):
        raise ValueError("beam attention supports head_dim 64 or 128")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("beam attention supports FP16/BF16")
    if group_beams < 2 or tile_tokens not in (16, 32, 64, 128):
        raise ValueError("invalid beam group or KV tile size")
    if max_kv_splits < 1:
        raise ValueError("max_kv_splits must be positive")

    _, q_heads, _ = q.shape
    kv_heads = k_buffer.shape[1]
    if q_heads % kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    if group_beams * (q_heads // kv_heads) > 16:
        raise ValueError("beam group must contain at most 16 query rows per KV head")
    if out.shape != q.shape or out_lse.shape != q.shape[:2]:
        raise ValueError("output shapes do not match Q")
    if group_rows.ndim != 2 or group_rows.shape[1] != group_beams:
        raise ValueError("group_rows shape does not match group_beams")
    if group_widths.numel() != group_rows.shape[0]:
        raise ValueError("group widths do not match group rows")

    shared_prefix_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        group_rows,
        group_widths,
        prefix_lens,
        prefix_out,
        prefix_lse,
        group_beams=group_beams,
        sm_scale=sm_scale,
        tile_tokens=tile_tokens,
    )

    suffix_stage = (
        _decode_att_m_fwd if q_heads == kv_heads else _decode_grouped_att_m_fwd
    )
    suffix_stage(
        q,
        k_buffer,
        v_buffer,
        suffix_out,
        suffix_lse,
        suffix_indptr,
        suffix_indices,
        suffix_num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
        page_size=1,
    )
    merge_prefix_suffix(
        prefix_out,
        prefix_lse,
        suffix_out,
        suffix_lse,
        suffix_num_kv_splits,
        out,
        out_lse,
    )
    return out
