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
"""Beam-aware Triton attention backend.

Version one shares the immutable prompt and, when the committed parent map is
available, the generated history of sibling beams. Private suffixes keep using
the production Triton gather stage. Unsupported or mixed batches fall back to
TritonAttnBackend without changing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

_MIN_GROUP_BEAMS = 8
_MIN_PREFIX_TOKENS = 64
_MIN_SAVED_KV_REFERENCES = 1024


@dataclass
class BeamForwardMetadata:
    group_rows: torch.Tensor
    group_widths: torch.Tensor
    prefix_lens: torch.Tensor
    suffix_indptr: torch.Tensor
    suffix_indices: torch.Tensor
    suffix_num_kv_splits: torch.Tensor
    prefix_out: torch.Tensor
    prefix_lse: torch.Tensor
    output_lse: torch.Tensor
    group_beams: int


class BeamTritonAttnBackend(TritonAttnBackend):
    """Triton backend with an optimized path for pure beam decode batches."""

    def __init__(self, model_runner: ModelRunner, skip_prefill: bool = False):
        super().__init__(model_runner, skip_prefill=skip_prefill)
        from sglang.kernels.ops.attention.beam_decode_attention import (
            balanced_beam_chunks,
            beam_prefix_attention_fwd,
            build_suffix_indices,
        )

        self.balanced_beam_chunks = balanced_beam_chunks
        self.beam_prefix_attention_fwd = torch.compiler.disable(
            beam_prefix_attention_fwd
        )
        self.build_beam_suffix_indices = torch.compiler.disable(build_suffix_indices)
        self.beam_forward_metadata: BeamForwardMetadata | None = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self.beam_forward_metadata = self._build_beam_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # The initial implementation deliberately uses the proven Triton path in
        # captured graphs. Beam plan buffers become capture-stable in a follow-up.
        self.beam_forward_metadata = None
        return super().init_forward_metadata_out_graph(forward_batch, in_capture)

    def _build_beam_metadata(
        self, forward_batch: ForwardBatch
    ) -> BeamForwardMetadata | None:
        tail = forward_batch.beam_tail
        if (
            tail is None
            or not forward_batch.forward_mode.is_decode()
            or self.use_mla
            or self.page_size != 1
            or self.dcp_size != 1
            or self.sliding_window_size not in (None, -1)
            or forward_batch.seq_lens_cpu is None
        ):
            return None

        gqa = self.num_head // self.num_kv_head
        # The first tuned policy is deliberately conservative. A30 measurements
        # show no win yet for GQA or small beam groups.
        if gqa != 1:
            return None
        group_limit = min(16, max(2, 16 // max(1, gqa)))
        rows_by_group = []
        prefixes = []
        covered_rows = []
        base = tail.num_base_rows
        for entry in tail.entries:
            rows = [entry.leader_idx, *range(base + entry.start, base + entry.end)]
            group = entry.group
            parent_indices = (
                group.attention_parent_indices
                if group.num_generated == group.num_committed
                else None
            )
            parent_groups = []
            if parent_indices is not None and len(parent_indices) == len(rows):
                by_parent = {}
                for row, parent in zip(rows, parent_indices):
                    by_parent.setdefault(parent, []).append(row)
                parent_groups = list(by_parent.values())
            use_parent_groups = parent_groups and all(
                len(parent_group) >= _MIN_GROUP_BEAMS for parent_group in parent_groups
            )
            source_groups = parent_groups if use_parent_groups else [rows]
            prefix_len = (
                group.attention_prefix_len if use_parent_groups else group.prompt_len
            )
            for chunk in (
                chunk
                for source_group in source_groups
                for chunk in self.balanced_beam_chunks(source_group, group_limit)
            ):
                rows_by_group.append(chunk)
                prefixes.append(prefix_len)
                covered_rows.extend(chunk)

        if (
            not rows_by_group
            or any(prefix <= 0 for prefix in prefixes)
            or any(
                len(chunk) < _MIN_GROUP_BEAMS
                or prefix < _MIN_PREFIX_TOKENS
                or (len(chunk) - 1) * prefix < _MIN_SAVED_KV_REFERENCES
                for chunk, prefix in zip(rows_by_group, prefixes)
            )
            or sorted(covered_rows) != list(range(forward_batch.batch_size))
        ):
            return None

        group_beams = max(len(chunk) for chunk in rows_by_group)
        device = forward_batch.seq_lens.device
        padded_rows = [
            chunk + [chunk[0]] * (group_beams - len(chunk)) for chunk in rows_by_group
        ]
        group_rows = torch.tensor(padded_rows, dtype=torch.int64, device=device)
        group_widths = torch.tensor(
            [len(chunk) for chunk in rows_by_group],
            dtype=torch.int32,
            device=device,
        )
        prefix_lens = torch.tensor(prefixes, dtype=torch.int32, device=device)
        row_prefix_values = [0] * forward_batch.batch_size
        for chunk, prefix_len in zip(rows_by_group, prefixes):
            for row in chunk:
                row_prefix_values[row] = prefix_len
        row_prefix_lens = torch.tensor(
            row_prefix_values, dtype=torch.int32, device=device
        )

        seq_lens_cpu = [int(value) for value in forward_batch.seq_lens_cpu.tolist()]
        seq_lens = forward_batch.seq_lens[: forward_batch.batch_size].to(torch.int32)
        if any(
            prefix > seq_lens_cpu[row]
            for chunk, prefix in zip(rows_by_group, prefixes)
            for row in chunk
        ):
            return None
        suffix_lens = seq_lens - row_prefix_lens
        suffix_indptr = torch.zeros(
            forward_batch.batch_size + 1, dtype=torch.int32, device=device
        )
        suffix_tokens = sum(seq_lens_cpu) - sum(
            len(chunk) * prefix for chunk, prefix in zip(rows_by_group, prefixes)
        )
        suffix_indices = torch.empty(
            suffix_tokens,
            dtype=self.forward_metadata.kv_indices.dtype,
            device=device,
        )
        max_seq_len = max(seq_lens_cpu)
        self.build_beam_suffix_indices(
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            row_prefix_lens,
            suffix_indptr,
            suffix_indices,
            max_seq_len=max_seq_len,
        )

        suffix_num_kv_splits = torch.empty(
            forward_batch.batch_size, dtype=torch.int32, device=device
        )
        self.get_num_kv_splits(suffix_num_kv_splits, suffix_lens.clamp_min(1))
        shape = (
            forward_batch.batch_size,
            self.num_head,
            self.v_head_dim,
        )
        return BeamForwardMetadata(
            group_rows=group_rows,
            group_widths=group_widths,
            prefix_lens=prefix_lens,
            suffix_indptr=suffix_indptr,
            suffix_indices=suffix_indices,
            suffix_num_kv_splits=suffix_num_kv_splits,
            prefix_out=torch.empty(shape, dtype=torch.float32, device=device),
            prefix_lse=torch.empty(shape[:2], dtype=torch.float32, device=device),
            output_lse=torch.empty(shape[:2], dtype=torch.float32, device=device),
            group_beams=group_beams,
        )

    def _can_run_beam_attention(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        *,
        sinks,
        score_mod,
        aux_tensors,
    ) -> bool:
        key_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        return (
            self.beam_forward_metadata is not None
            and q.dtype == torch.float16
            and key_buffer.dtype == q.dtype
            and value_buffer.dtype == q.dtype
            and layer.qk_head_dim == layer.v_head_dim
            and layer.qk_head_dim == 128
            and layer.k_scale is None
            and layer.v_scale is None
            and not layer.logit_cap
            and sinks is None
            and score_mod is None
            and aux_tensors is None
            and q.shape[0] == forward_batch.batch_size
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
    ):
        if not self._can_run_beam_attention(
            q,
            layer,
            forward_batch,
            sinks=sinks,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        ):
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                sinks=sinks,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
            )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        output = torch.empty_like(q)
        if save_kv_cache:
            self._set_kv_buffer(
                forward_batch,
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                ),
                k,
                v,
            )

        metadata = self.beam_forward_metadata
        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        output_view = output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        self.beam_prefix_attention_fwd(
            q_view,
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            output_view,
            metadata.output_lse,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            metadata.group_rows,
            metadata.group_widths,
            metadata.prefix_lens,
            metadata.suffix_indptr,
            metadata.suffix_indices,
            metadata.suffix_num_kv_splits,
            metadata.prefix_out,
            metadata.prefix_lse,
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            group_beams=metadata.group_beams,
            max_kv_splits=self.max_kv_splits,
            sm_scale=layer.scaling,
            logit_cap=0.0,
        )
        return output
