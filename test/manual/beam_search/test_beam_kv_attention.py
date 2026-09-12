"""Standalone beam-KV experiments; CPU modes require only the standard library."""

from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import importlib.util
import itertools
import json
import linecache
import logging
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import unittest
from array import array
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


@dataclass
class BeamSnapshot:
    paths: list[list[int]]
    slots: list[int | None]
    capacity: int
    trace: list[dict[str, int]]

    def slot_paths(self):
        locations = {
            node: slot for slot, node in enumerate(self.slots) if node is not None
        }
        return [[locations[node] for node in path] for path in self.paths]

    def mask(self):
        result = [[False] * len(self.slots) for _ in self.paths]
        for row, path in zip(result, self.slot_paths()):
            for slot in path:
                row[slot] = True
        return result


def simulate_tree(
    beams,
    steps,
    pattern="shared",
    allocator="reuse",
    concentration=0.8,
    seed=42,
    initial_capacity=None,
):
    if beams < 1 or steps < 0:
        raise ValueError("beams must be positive and steps must be nonnegative")
    if pattern not in ("shared", "independent", "grouped", "random"):
        raise ValueError("unknown tree pattern")
    if allocator not in ("reuse", "append"):
        raise ValueError("allocator must be reuse or append")
    if not 0 <= concentration <= 1:
        raise ValueError("concentration must be between zero and one")
    if initial_capacity is not None and initial_capacity < 1:
        raise ValueError("initial_capacity must be positive")
    rng = random.Random(seed)
    limit = beams * steps
    capacity = min(limit, initial_capacity or beams)
    slots, free, locations, trace = [], [], {}, []
    paths = [[] for _ in range(beams)]
    expansions = reused = freed = 0
    for step in range(steps):
        if pattern == "shared":
            parents = [0] * beams
        elif pattern == "independent":
            parents = list(range(beams))
        elif pattern == "grouped":
            width = (beams + 1) // 2
            parents = [i // width * width for i in range(beams)]
        else:
            favored = rng.randrange(beams)
            parents = [
                favored if rng.random() < concentration else rng.randrange(beams)
                for _ in range(beams)
            ]
        inherited = [paths[parent] for parent in parents]
        retained = {node for path in inherited for node in path}
        if allocator == "reuse":
            for node in sorted(locations.keys() - retained):
                slot = locations.pop(node)
                slots[slot] = None
                heapq.heappush(free, slot)
                freed += 1
        paths = []
        for beam, parent_path in enumerate(inherited):
            node = step * beams + beam
            if free:
                slot = heapq.heappop(free)
                slots[slot] = node
                reused += 1
            else:
                slot = len(slots)
                slots.append(node)
            locations[node] = slot
            paths.append(parent_path + [node])
        if len(slots) > capacity:
            while capacity < len(slots):
                capacity = min(limit, max(1, capacity * 2))
            expansions += 1
        live = len({node for path in paths for node in path})
        trace.append(
            {
                "step": step + 1,
                "live_nodes": live,
                "held_nodes": len(locations),
                "scan_slots": len(slots),
                "capacity": capacity,
                "generated_nodes": (step + 1) * beams,
                "freed_nodes": freed,
                "reused_slots": reused,
                "expansions": expansions,
            }
        )
    return BeamSnapshot(paths, slots, capacity, trace)


def relayout(snapshot, spacing=1, seed=0):
    if spacing < 1:
        raise ValueError("spacing must be positive")
    positions = [i * spacing for i in range(len(snapshot.slots))]
    random.Random(seed).shuffle(positions)
    size = (len(snapshot.slots) - 1) * spacing + 1 if snapshot.slots else 0
    slots = [None] * size
    for node, slot in zip(snapshot.slots, positions):
        slots[slot] = node
    return BeamSnapshot(snapshot.paths, slots, len(slots), snapshot.trace)


def traffic_model(snapshot, group_beams=8, tile=64, head_dim=128, element_bytes=2):
    if min(group_beams, tile, head_dim, element_bytes) < 1:
        raise ValueError("tile, group, dimension, and element size must be positive")
    beams, steps = len(snapshot.paths), len(snapshot.paths[0])
    scan = len(snapshot.slots)
    groups = (beams + group_beams - 1) // group_beams
    live = len({node for path in snapshot.paths for node in path})
    kv_bytes = 2 * head_dim * element_bytes
    sectors_per_kv = 2 * math.ceil(head_dim * element_bytes / 32)
    scan_refs, gather_refs = groups * scan, beams * steps
    return {
        "kind": "analytical_not_measured",
        "scope": "one_layer_one_KV_head_no_GQA_head_repetition",
        "beams": beams,
        "steps": steps,
        "group_beams": group_beams,
        "tile_tokens": tile,
        "live_nodes": live,
        "capacity": snapshot.capacity,
        "scan_slots": scan,
        "scan_tile_loads": groups * math.ceil(scan / tile),
        "gather_tile_loads": beams * math.ceil(steps / tile),
        "scan_kv_references": scan_refs,
        "gather_kv_references": gather_refs,
        "scan_bytes_no_cache": scan_refs * kv_bytes,
        "scan_bytes_slot_valid_mask_no_cache": groups
        * sum(node is not None for node in snapshot.slots)
        * kv_bytes,
        "gather_bytes_no_cache": gather_refs * kv_bytes,
        "scan_32b_sectors_aligned": scan_refs * sectors_per_kv,
        "gather_32b_sectors_aligned": gather_refs * sectors_per_kv,
        "gather_bytes_cold_ideal_reuse": live * kv_bytes,
        "scan_logical_pairs": beams * scan,
        "gather_logical_pairs": beams * steps,
        "read_reduction_no_cache": gather_refs / scan_refs if scan_refs else None,
        "compute_amplification": scan / steps if steps else None,
        "assumptions": (
            "aligned vectors, full scan, no tile padding loads or metadata traffic; "
            "sectors are not DRAM commands"
        ),
    }


def cpu_kv(snapshot, head_dim=4, seed=7):
    keys, values = [], []
    for node in snapshot.slots:
        if node is None:
            keys.append([math.nan] * head_dim)
            values.append([math.nan] * head_dim)
        else:
            rng = random.Random(seed * 1_000_003 + node)
            keys.append([rng.uniform(-1, 1) for _ in range(head_dim)])
            values.append([rng.uniform(-1, 1) for _ in range(head_dim)])
    return keys, values


def reference_attention(query, keys, values, path):
    if not path:
        return [0.0] * len(query), -math.inf
    scores = [
        math.fsum(q * k for q, k in zip(query, keys[slot])) / math.sqrt(len(query))
        for slot in path
    ]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(weights)
    output = [
        math.fsum(w * values[slot][d] for w, slot in zip(weights, path)) / denominator
        for d in range(len(query))
    ]
    return output, maximum + math.log(denominator)


def tiled_attention(query, keys, values, visible, tile):
    maximum, denominator = -math.inf, 0.0
    output = [0.0] * len(query)
    for start in range(0, len(visible), tile):
        slots = [s for s in range(start, min(start + tile, len(visible))) if visible[s]]
        if not slots:
            continue
        scores = [
            sum(query[d] * keys[s][d] for d in range(len(query)))
            / math.sqrt(len(query))
            for s in slots
        ]
        new_maximum = max(maximum, max(scores))
        scale = math.exp(maximum - new_maximum)
        weights = [math.exp(score - new_maximum) for score in scores]
        output = [
            output[d] * scale + sum(w * values[s][d] for w, s in zip(weights, slots))
            for d in range(len(query))
        ]
        denominator = denominator * scale + sum(weights)
        maximum = new_maximum
    if not denominator:
        return output, -math.inf
    return [v / denominator for v in output], maximum + math.log(denominator)


def merge_states(left, right):
    left_out, left_lse = left
    right_out, right_lse = right
    if left_lse == -math.inf:
        return right
    if right_lse == -math.inf:
        return left
    maximum = max(left_lse, right_lse)
    a, b = math.exp(left_lse - maximum), math.exp(right_lse - maximum)
    return (
        [(a * x + b * y) / (a + b) for x, y in zip(left_out, right_out)],
        maximum + math.log(a + b),
    )


class TestBeamKVCPU(unittest.TestCase):
    def assert_state_close(self, actual, expected):
        self.assertEqual(len(actual[0]), len(expected[0]))
        for a, b in zip(actual[0], expected[0]):
            self.assertAlmostEqual(a, b, places=11)
        if math.isinf(expected[1]):
            self.assertEqual(actual[1], expected[1])
        else:
            self.assertAlmostEqual(actual[1], expected[1], places=11)

    def test_live_bounds_and_allocator_accounting(self):
        for beams in (1, 3, 8):
            for pattern in ("shared", "independent", "grouped", "random"):
                for allocator in ("reuse", "append"):
                    with self.subTest(
                        beams=beams, pattern=pattern, allocator=allocator
                    ):
                        snapshot = simulate_tree(beams, 13, pattern, allocator)
                        for row in snapshot.trace:
                            t, live = row["step"], row["live_nodes"]
                            self.assertLessEqual(t + beams - 1, live)
                            self.assertLessEqual(live, beams * t)
                            self.assertLessEqual(row["held_nodes"], row["scan_slots"])
                            self.assertLessEqual(row["scan_slots"], row["capacity"])
                            self.assertLessEqual(row["capacity"], beams * 13)
                            self.assertEqual(
                                row["generated_nodes"] - row["freed_nodes"],
                                row["held_nodes"],
                            )
                            if pattern == "shared":
                                self.assertEqual(live, t + beams - 1)
                            elif pattern == "independent":
                                self.assertEqual(live, beams * t)
                            if allocator == "reuse":
                                self.assertEqual(row["held_nodes"], live)
                            else:
                                self.assertEqual(row["scan_slots"], beams * t)

    def test_recycled_slots_do_not_alias_live_nodes(self):
        snapshot = simulate_tree(8, 40, "shared")
        nodes = [node for node in snapshot.slots if node is not None]
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertGreater(snapshot.trace[-1]["reused_slots"], 0)
        self.assertEqual(set(nodes), {node for path in snapshot.paths for node in path})
        for path in snapshot.paths:
            self.assertEqual([node // 8 for node in path], list(range(40)))

    def test_seed_reproducibility(self):
        a = simulate_tree(8, 20, "random", seed=3)
        self.assertEqual(a, simulate_tree(8, 20, "random", seed=3))
        self.assertNotEqual(a.paths, simulate_tree(8, 20, "random", seed=4).paths)

    def test_random_concentration_one_coalesces(self):
        snapshot = simulate_tree(7, 12, "random", concentration=1)
        self.assertEqual(snapshot.trace[-1]["live_nodes"], 18)

    def test_empty_history(self):
        snapshot = simulate_tree(3, 0)
        self.assertEqual(snapshot.slots, [])
        self.assertEqual(snapshot.slot_paths(), [[], [], []])
        self.assert_state_close(
            tiled_attention([1, 2], [], [], [], 4), ([0, 0], -math.inf)
        )
        self.assertEqual(traffic_model(snapshot)["scan_tile_loads"], 0)

    def test_mask_matches_path(self):
        snapshot = relayout(simulate_tree(5, 9, "random"), spacing=3)
        for mask, path in zip(snapshot.mask(), snapshot.slot_paths()):
            self.assertEqual({i for i, valid in enumerate(mask) if valid}, set(path))
            self.assertEqual(sum(mask), 9)

    def test_scan_and_gather_equivalence(self):
        rng = random.Random(19)
        for trial in range(24):
            snapshot = simulate_tree(
                rng.choice((1, 3, 8)),
                rng.randint(1, 12),
                rng.choice(("shared", "independent", "grouped", "random")),
                seed=trial,
            )
            snapshot = relayout(snapshot, spacing=rng.choice((1, 3)), seed=trial)
            keys, values = cpu_kv(snapshot)
            for visible, path in zip(snapshot.mask(), snapshot.slot_paths()):
                query = [rng.uniform(-2, 2) for _ in range(4)]
                expected = reference_attention(query, keys, values, path)
                scan_path = [s for s, flag in enumerate(visible) if flag]
                self.assert_state_close(
                    reference_attention(query, keys, values, scan_path), expected
                )
                for tile in (1, 3, 16):
                    self.assert_state_close(
                        tiled_attention(query, keys, values, visible, tile), expected
                    )

    def test_relayout_preserves_holes_and_padding(self):
        snapshot = BeamSnapshot([[0, 1]], [0, None, 1, None], 8, [])
        shuffled = relayout(snapshot, spacing=1, seed=3)
        self.assertEqual(len(shuffled.slots), 4)
        self.assertEqual(shuffled.slots.count(None), 2)
        self.assertEqual(len(relayout(snapshot, spacing=3).slots), 10)

    def test_slot_permutation_invariance(self):
        original = simulate_tree(4, 8, "grouped")
        shuffled = relayout(original, spacing=4, seed=91)
        q = [0.2, -0.4, 0.1, 0.8]
        k1, v1 = cpu_kv(original)
        k2, v2 = cpu_kv(shuffled)
        for p1, p2 in zip(original.slot_paths(), shuffled.slot_paths()):
            self.assert_state_close(
                reference_attention(q, k1, v1, p1), reference_attention(q, k2, v2, p2)
            )

    def test_split_softmax_merge(self):
        q, keys, values = (
            [0.4, -0.2],
            [[2, 3], [-1, 4], [5, 6]],
            [[1, 0], [0, 1], [2, 4]],
        )
        whole = reference_attention(q, keys, values, [0, 1, 2])
        for boundary in range(4):
            left = reference_attention(q, keys, values, list(range(boundary)))
            right = reference_attention(q, keys, values, list(range(boundary, 3)))
            self.assert_state_close(merge_states(left, right), whole)
        empty = ([0.0, 0.0], -math.inf)
        self.assertEqual(merge_states(empty, empty), empty)

    def test_masked_nan_and_extreme_scores(self):
        q, keys, values = (
            [1000.0],
            [[-1000.0], [math.nan], [1000.0]],
            [[2.0], [math.nan], [4.0]],
        )
        self.assert_state_close(
            tiled_attention(q, keys, values, [True, False, True], 2),
            ([4.0], 1_000_000.0),
        )
        self.assert_state_close(
            tiled_attention(q, keys, values, [False] * 3, 2), ([0.0], -math.inf)
        )

    def test_traffic_counts_and_partial_groups(self):
        snapshot = simulate_tree(8, 128, "independent")
        cost = traffic_model(snapshot, 8, 64)
        self.assertEqual(cost["gather_tile_loads"], 16)
        self.assertEqual(cost["scan_tile_loads"], 16)
        self.assertEqual(cost["gather_32b_sectors_aligned"], 16384)
        snapshot = simulate_tree(9, 1)
        cost = traffic_model(snapshot, 8, 64)
        self.assertEqual(cost["scan_tile_loads"], 2)
        self.assertEqual(cost["scan_kv_references"], 18)
        self.assertEqual(cost["scan_logical_pairs"], 81)

    def test_growth_does_not_change_layout(self):
        small = simulate_tree(6, 17, "random", initial_capacity=6)
        large = simulate_tree(6, 17, "random", initial_capacity=102)
        self.assertEqual(small.paths, large.paths)
        self.assertEqual(small.slots, large.slots)
        self.assertGreater(small.trace[-1]["expansions"], 0)
        self.assertEqual(large.trace[-1]["expansions"], 0)

    def test_invalid_parameters(self):
        for kwargs in (
            {"beams": 0},
            {"steps": -1},
            {"pattern": "unknown"},
            {"allocator": "unknown"},
            {"concentration": 2},
            {"initial_capacity": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_tree(**({"beams": 4, "steps": 8} | kwargs))
        with self.assertRaises(ValueError):
            relayout(simulate_tree(1, 1), spacing=0)
        with self.assertRaises(ValueError):
            traffic_model(simulate_tree(1, 1), tile=0)


class TestBenchmarkHarness(unittest.TestCase):
    def fake_cuda(self):
        torch = MagicMock()
        torch.cuda.graph.side_effect = lambda *a, **kw: nullcontext()
        torch.cuda.Event.side_effect = lambda **kw: Mock(
            elapsed_time=Mock(return_value=1.0)
        )
        return torch

    def test_warm_graph_averages_replay_batch(self):
        torch, call = self.fake_cuda(), Mock()
        result = measure_cuda(torch, call, 2, 3, None, "graph", 8)
        self.assertEqual(call.call_count, 10)
        self.assertEqual(result["calls_per_sample"], 8)
        self.assertEqual(result["median_us"], 125.0)
        self.assertEqual(torch.cuda.CUDAGraph.return_value.replay.call_count, 4)
        for args in torch.cuda.Event.call_args_list:
            self.assertEqual(args.kwargs, {"enable_timing": True})

    def test_scrub_is_outside_single_call_graph(self):
        torch, call, scrub = self.fake_cuda(), Mock(), Mock()
        operations = Mock()
        operations.attach_mock(torch.cuda.CUDAGraph.return_value.replay, "replay")
        operations.attach_mock(scrub.add_, "scrub")
        result = measure_cuda(torch, call, 2, 3, scrub, "graph", 64)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(result["calls_per_sample"], 1)
        self.assertEqual(result["median_us"], 1000.0)
        self.assertEqual(
            [call[0] for call in operations.mock_calls],
            ["replay", "scrub", "replay", "scrub", "replay", "scrub", "replay"],
        )

    def test_eager_does_not_capture(self):
        torch, call = self.fake_cuda(), Mock()
        result = measure_cuda(torch, call, 2, 3, None, "eager", 64)
        self.assertEqual(call.call_count, 5)
        self.assertEqual(result["sample_unit"], "eager_call")
        torch.cuda.CUDAGraph.assert_not_called()
        torch.cuda.graph.assert_not_called()

    def test_invalid_timing_parameters(self):
        for kwargs in (
            {"warmup": 0},
            {"repeats": 0},
            {"graph_batch": 0},
            {"execution": "auto"},
        ):
            parameters = {"warmup": 1, "repeats": 1, "scrub": None} | kwargs
            with self.assertRaises(ValueError):
                measure_cuda(self.fake_cuda(), Mock(), **parameters)

    def test_summary_uses_best_split_in_same_execution(self):
        rows = [
            {
                "method": "B_pool_gather",
                "splits": s,
                "median_us": duration,
                "execution": mode,
                "scope": "total",
            }
            for s, duration, mode in (
                (1, 20, "graph"),
                (4, 40, "graph"),
                (1, 100, "eager"),
            )
        ]
        rows.append(
            {
                "method": "C_shared_scan",
                "group_beams": 8,
                "tile": 64,
                "compute": "dot",
                "query_tile": 16,
                "query_rows": 8,
                "shared_splits": 1,
                "median_us": 10,
                "execution": "graph",
                "scope": "total",
            }
        )
        summary = summarize_timings(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["baseline_splits"], 1)
        self.assertEqual(summary[0]["observed_speedup"], 2.0)

    def test_inspection_restores_jit_method(self):
        kernel = SimpleNamespace(
            name="test",
            hash="abc",
            metadata=SimpleNamespace(
                target="sm70", shared=1024, num_warps=4, num_stages=1
            ),
            asm={"ptx": "fma.rn.f32; mma.sync.aligned;"},
            n_regs=32,
            n_spills=0,
        )
        jit = SimpleNamespace(run=Mock(return_value=kernel))
        original = jit.run
        data = inspect_compilation(lambda: jit.run(grid=(2, 8, 1)), [jit])
        self.assertIs(jit.run, original)
        self.assertEqual(data[0]["grid"], [2, 8, 1])
        self.assertEqual(data[0]["ptx_mma_instruction_count"], 1)
        self.assertEqual(data[0]["ptx_fma_instruction_count"], 1)
        self.assertEqual(data[0]["registers_per_thread"], 32)
        jit.run.side_effect = RuntimeError("compilation failed")
        with self.assertRaisesRegex(RuntimeError, "compilation failed"):
            inspect_compilation(lambda: jit.run(), [jit])
        self.assertIs(jit.run, original)

    def test_matrix_is_staged_and_supports_selection(self):
        args = SimpleNamespace(
            preset="quick",
            python=None,
            case=None,
            output_dir="/tmp/beam-matrix-test",
            graph_batch=32,
            repeats=10,
            warmup=3,
        )
        commands = matrix_commands(args)
        self.assertEqual(len(commands), 12)
        self.assertEqual(len({case["output"] for case in commands}), 12)
        self.assertIn("--dtype", commands[0]["command"])
        self.assertIn("float16", commands[0]["command"])
        args.case = ["timing_control"]
        self.assertEqual(len(matrix_commands(args)), 1)
        args.case = ["missing"]
        with self.assertRaises(ValueError):
            matrix_commands(args)
        self.assertGreater(len(matrix_cases("full")), len(matrix_cases("quick")))

    def test_padded_geometry_counts_real_blocks(self):
        args = SimpleNamespace(q_heads=8, kv_heads=8, beams=8, steps=128)
        snapshot = simulate_tree(8, 128)
        config = {
            "group_beams": 8,
            "tile": 64,
            "query_rows": 8,
            "compute": "dot",
            "shared_splits": 4,
        }
        geometry = padded_geometry(args, snapshot, config)
        self.assertEqual(geometry["block_m"], 16)
        self.assertEqual(geometry["blocks_per_request"], 32)
        self.assertEqual(geometry["padded_pair_ratio"], 3.0)

    def test_query_cap_deduplicates_large_beam_groups(self):
        args = SimpleNamespace(
            beams=32,
            q_heads=8,
            kv_heads=2,
            query_tile=None,
            group_beams=[4, 8, 16],
            compute=["dot", "simt"],
            tile=[16, 32],
            shared_splits=[1, 2, 4],
        )
        configs = shared_configs(args)
        self.assertEqual(len(configs), 12)
        self.assertEqual(
            {c["query_rows"] for c in configs if c["compute"] == "dot"}, {16}
        )
        self.assertEqual(
            {c["query_rows"] for c in configs if c["compute"] == "simt"}, {4}
        )
        args.beams = 1
        self.assertEqual({c["query_rows"] for c in shared_configs(args)}, {4})
        args.query_tile = 16
        with self.assertRaises(ValueError):
            shared_configs(args)

    def test_split_partition_and_merge_match_full_attention(self):
        snapshot = relayout(simulate_tree(3, 17), spacing=3, seed=8)
        keys, values = cpu_kv(snapshot)
        query = [0.2, -0.8, 0.5, 0.1]
        for tile, splits, end in itertools.product(
            (16, 32), (1, 2, 4, 8), (0, 1, 17, len(keys))
        ):
            split_size = math.ceil(math.ceil(end / tile) / splits) * tile
            for path in snapshot.slot_paths():
                retained = [slot for slot in path if slot < end]
                result = ([0.0] * 4, -math.inf)
                pieces = []
                for split in range(splits):
                    slots = [
                        slot
                        for slot in retained
                        if split * split_size
                        <= slot
                        < min((split + 1) * split_size, end)
                    ]
                    pieces.extend(slots)
                    result = merge_states(
                        result, reference_attention(query, keys, values, slots)
                    )
                self.assertEqual(sorted(pieces), sorted(retained))
                expected = reference_attention(query, keys, values, retained)
                for actual, wanted in zip(result[0], expected[0]):
                    self.assertAlmostEqual(actual, wanted, places=11)
                if retained:
                    self.assertAlmostEqual(result[1], expected[1], places=11)
                else:
                    self.assertEqual(result[1], -math.inf)

    def test_source_subset_does_not_import_runtime_or_modify_function_bodies(self):
        source = (
            "import missing_runtime\nCONST = 7\ndef f(x: int):\n    return x + CONST\n"
        )
        with patch.object(Path, "read_text", return_value=source):
            module = load_source_subset(
                Path("numeric.py"), "beam_subset_test", ["f"], ["CONST"], {}
            )
        try:
            self.assertEqual(module.f(3), 10)
            self.assertIs(module.f.__annotations__["x"], int)
            self.assertEqual(module.f.__code__.co_firstlineno, 3)
            self.assertEqual(
                module.source_sha256, hashlib.sha256(source.encode()).hexdigest()
            )
        finally:
            sys.modules.pop("beam_subset_test", None)
        with (
            patch.object(Path, "read_text", return_value=source),
            self.assertRaises(RuntimeError),
        ):
            load_source_subset(
                Path("numeric.py"), "beam_missing_test", ["missing"], [], {}
            )

    def test_standalone_loader_launches_existing_cuda_baseline_without_runtime(self):
        launches = []

        class FakeJIT:
            def __init__(self, function):
                self.fn = function

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launches.append((self.fn.__name__, grid, kwargs))

                return launch

        class TensorShape:
            def __init__(self, shape, dtype="float16"):
                self.shape = shape
                self.ndim = len(shape)
                self.dtype = dtype
                self.device = "cuda:0"
                self.is_cuda = True

            def stride(self, axis):
                return math.prod(self.shape[axis + 1 :])

            def numel(self):
                return math.prod(self.shape)

            def is_contiguous(self):
                return True

        torch = ModuleType("torch")
        torch.float16, torch.bfloat16 = "float16", "bfloat16"
        torch.float32, torch.bool, torch.int32 = "float32", "bool", "int32"
        torch.version = SimpleNamespace(hip=None)
        triton = ModuleType("triton")
        language = ModuleType("triton.language")
        language.constexpr = int
        triton.language = language
        triton.jit = FakeJIT
        triton.cdiv = lambda x, y: (x + y - 1) // y
        triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
        runtime_before = {name for name in sys.modules if name.startswith("sglang")}
        isolated_names = (
            "beam_experiment_shared",
            "beam_experiment_gather",
            "beam_experiment_gather_fp32",
            "beam_experiment_score_mod",
        )
        with patch.dict(
            sys.modules, {"torch": torch, "triton": triton, "triton.language": language}
        ):
            try:
                shared, gather = load_standalone_kernels(torch, triton)
                self.assertTrue(callable(shared.beam_decode_attention_fwd))
                self.assertEqual(gather.pruned_pdl_blocks, 2)
                self.assertNotEqual(gather.loaded_source_sha256, gather.source_sha256)
                for name in ("_fwd_kernel_stage2", "_fwd_grouped_kernel_stage1"):
                    loaded = "".join(
                        linecache.getlines(
                            getattr(gather, name).fn.__code__.co_filename
                        )
                    )
                    self.assertNotIn("tl.extra.cuda.gdc_", loaded)
                _, precise = load_standalone_kernels(torch, triton, mha_qk_fp32=True)
                self.assertTrue(precise.mha_qk_fp32)
                self.assertFalse(gather.mha_qk_fp32)
                self.assertEqual(precise.source_sha256, gather.source_sha256)
                self.assertNotEqual(
                    precise.loaded_source_sha256, gather.loaded_source_sha256
                )
                source = "".join(
                    linecache.getlines(
                        precise._fwd_kernel_stage1.fn.__code__.co_filename
                    )
                )
                self.assertIn("q.to(tl.float32)[None, :] * k.to(tl.float32)", source)
                for kv_heads, expected_stage in (
                    (8, "_fwd_kernel_stage1"),
                    (2, "_fwd_grouped_kernel_stage1"),
                ):
                    launches.clear()
                    q = TensorShape((9, 8, 128))
                    kv = TensorShape((96, kv_heads, 128))
                    partial = TensorShape((9, 8, 4, 128))
                    lse = TensorShape((9, 8, 4))
                    indptr = TensorShape((10,))
                    gather.decode_attention_fwd(
                        q,
                        kv,
                        kv,
                        q,
                        indptr,
                        None,
                        partial,
                        lse,
                        None,
                        4,
                        1 / math.sqrt(128),
                        1.0,
                        1.0,
                        enable_lean=False,
                    )
                    self.assertEqual(
                        [entry[0] for entry in launches],
                        [expected_stage, "_fwd_kernel_stage2"],
                    )
                    self.assertEqual(launches[0][1][0], 9)
                    self.assertEqual(launches[0][2]["PAGE_SIZE"], 1)
                    self.assertFalse(launches[1][2]["USE_PDL"])
                for compute, splits in itertools.product(("dot", "simt"), (1, 4)):
                    launches.clear()
                    q = TensorShape((18, 8, 128))
                    kv = TensorShape((192, 2, 128))
                    metadata = (
                        TensorShape((18, 96), "bool"),
                        TensorShape((2, 96), "bool"),
                        TensorShape((18,), "bool"),
                        TensorShape((2,), "int32"),
                    )
                    workspace = (
                        TensorShape((splits, 18, 8, 128), "float32"),
                        TensorShape((splits, 18, 8), "float32"),
                    )
                    shared.beam_decode_attention_fwd(
                        q,
                        kv,
                        kv,
                        *metadata,
                        q,
                        TensorShape((18, 8), "float32"),
                        beams_per_request=9,
                        group_beams=16,
                        tile_tokens=16,
                        query_tile=4,
                        compute=compute,
                        num_splits=splits,
                        partial_out=workspace[0],
                        partial_lse=workspace[1],
                    )
                    self.assertEqual(launches[0][1], (9, 2, 2 * splits))
                    self.assertEqual(launches[0][2]["QUERY_ROWS"], 4)
                    self.assertEqual(
                        launches[0][2]["BLOCK_M"], 16 if compute == "dot" else 4
                    )
                    self.assertEqual(len(launches), 2 if splits > 1 else 1)
                    if splits > 1:
                        self.assertEqual(launches[1][0], "_merge_beam_splits")
                        self.assertEqual(launches[1][1], (18, 8))
                launches.clear()
                grouped_prefix = TensorShape((2, 18, 8, 128), "float32")
                grouped_lse = TensorShape((2, 18, 8), "float32")
                paths = TensorShape((18, 96), "int32")
                order = TensorShape((18,), "int64")
                shared.shared_prefix_attention(
                    q,
                    kv,
                    kv,
                    paths,
                    order,
                    None,
                    None,
                    grouped_prefix,
                    grouped_lse,
                    beams=9,
                    group_beams=3,
                    tile=32,
                    splits=2,
                    prompt_len=64,
                )
                self.assertEqual(launches[0][1], (6, 2, 2))
                self.assertEqual(launches[0][2]["PROMPT_LEN"], 64)
                self.assertEqual(launches[0][2]["BLOCK_M"], 16)
                shared.merge_prefix_suffix(
                    grouped_prefix,
                    grouped_lse,
                    TensorShape((18, 8, 4, 128), "float32"),
                    TensorShape((18, 8, 4), "float32"),
                    None,
                    None,
                    q,
                    TensorShape((18, 8), "float32"),
                )
                self.assertEqual(launches[1][0], "_merge_prefix_suffix")
                self.assertEqual(launches[1][2]["BLOCK_SPLITS"], 4)
                self.assertTrue(launches[1][2]["HAS_PREFIX"])
                shared.merge_prefix_suffix(
                    None,
                    None,
                    TensorShape((18, 8, 4, 128), "float32"),
                    TensorShape((18, 8, 4), "float32"),
                    None,
                    None,
                    q,
                    TensorShape((18, 8), "float32"),
                )
                self.assertFalse(launches[2][2]["HAS_PREFIX"])
                self.assertEqual(launches[2][2]["PREFIX_SPLITS"], 0)
                launches.pop()
                torch.sub, torch.cumsum = Mock(), Mock()
                shared.group_prefix_metadata(
                    paths,
                    order,
                    None,
                    None,
                    None,
                    None,
                    MagicMock(),
                    None,
                    None,
                    beams=9,
                    group_beams=3,
                    prompt_len=64,
                )
                self.assertEqual(launches[2][0], "_group_prefix_metadata")
                self.assertEqual(launches[2][2]["BLOCK_B"], 4)
                self.assertEqual(launches[3][0], "_suffix_indices")
                torch.sub.assert_called_once()
                torch.cumsum.assert_called_once()
                self.assertEqual(
                    {name for name in sys.modules if name.startswith("sglang")},
                    runtime_before,
                )
            finally:
                for name in isolated_names:
                    sys.modules.pop(name, None)

    def test_query_tiles_cover_each_beam_head_once(self):
        for beams, gqa, cap in itertools.product(
            (1, 3, 9, 32), (1, 3, 4, 16), (1, 4, 16)
        ):
            query_rows = min(min(16, beams) * gqa, cap)
            coordinates = []
            for group in range(math.ceil(beams * gqa / query_rows)):
                for row in range(query_rows):
                    index = group * query_rows + row
                    if index // gqa < beams:
                        coordinates.append((index // gqa, index % gqa))
            self.assertEqual(
                coordinates, list(itertools.product(range(beams), range(gqa)))
            )

    def test_isolated_matrix_selects_existing_python(self):
        args = SimpleNamespace(
            preset="compiler",
            python="/opt/venvs/triton31/bin/python",
            case=None,
            output_dir="/tmp/beam-compiler",
            graph_batch=64,
            repeats=10,
            warmup=3,
        )
        for case in matrix_commands(args):
            self.assertEqual(case["command"][0], args.python)
            self.assertIn("--standalone", case["command"])
            self.assertIn("--query-tile", case["command"])
        self.assertEqual(len(matrix_cases("v100-tune")), 9)


def load_source_subset(
    path, module_name, functions, constants, namespace, *, mha_qk_fp32=False
):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    selected, found = [], set()
    requested = set(functions) | set(constants)
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            name = target.id if isinstance(target, ast.Name) else None
        if name in requested:
            selected.append(node)
            found.add(name)
    if found != requested:
        raise RuntimeError(
            f"standalone loader missing definitions: {requested - found}"
        )
    pdl_blocks = [
        child
        for node in selected
        for child in ast.walk(node)
        if isinstance(child, ast.If)
        and isinstance(child.test, ast.Name)
        and child.test.id == "USE_PDL"
    ]
    loaded_source = source
    filename = str(path)
    if pdl_blocks or mha_qk_fp32:
        lines = source.splitlines(keepends=True)
        for node in pdl_blocks:
            if node.orelse:
                raise RuntimeError(
                    "standalone PDL pruning does not support an else branch"
                )
            lines[node.lineno - 1] = " " * node.col_offset + "pass\n"
            for i in range(node.lineno, node.end_lineno):
                lines[i] = "\n"
        if mha_qk_fp32:
            promote_mha_qk(lines, selected)
        loaded_source = "".join(lines)
        filename = f"{path}::<{module_name}>"
        linecache.cache[filename] = (len(loaded_source), None, lines, filename)
        reparsed = ast.parse(loaded_source, filename=filename)
        starts = {node.lineno for node in selected}
        selected = [node for node in reparsed.body if node.lineno in starts]
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(namespace)
    sys.modules[module_name] = module
    # Triton inspection must see the same specialized source as Python execution.
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename,
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    module.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    module.loaded_source_sha256 = hashlib.sha256(loaded_source.encode()).hexdigest()
    module.pruned_pdl_blocks = len(pdl_blocks)
    module.mha_qk_fp32 = mha_qk_fp32
    return module


def promote_mha_qk(lines, definitions):
    kernel = next(
        (
            node
            for node in definitions
            if isinstance(node, ast.FunctionDef) and node.name == "_fwd_kernel_stage1"
        ),
        None,
    )
    if kernel is None:
        raise RuntimeError("FP32 QK specialization requires the original MHA stage-1")
    expected = ast.dump(ast.parse("qk = tl.sum(q[None, :] * k, 1)").body[0])
    matches = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign) and ast.dump(node) == expected
    ]
    if len(matches) != 1 or matches[0].lineno != matches[0].end_lineno:
        raise RuntimeError(
            "MHA QK expression changed; review the precision specialization"
        )
    node = matches[0]
    lines[node.lineno - 1] = (
        " "
        * node.col_offset
        + "qk = tl.sum(q.to(tl.float32)[None, :] * k.to(tl.float32), "
        "1, dtype=tl.float32)\n"
    )


def load_standalone_kernels(torch, triton, *, mha_qk_fp32=False):
    from typing import Tuple

    import triton.language as tl

    root = Path(__file__).resolve().parents[3] / "python/sglang/kernels/ops/attention"
    shared_name = "beam_experiment_shared"
    spec = importlib.util.spec_from_file_location(
        shared_name, root / "beam_decode_attention.py"
    )
    shared = importlib.util.module_from_spec(spec)
    sys.modules[shared_name] = shared
    spec.loader.exec_module(shared)
    auxiliary = load_source_subset(
        root / "score_mod.py",
        "beam_experiment_score_mod",
        ["unpack_aux_tensors"],
        [],
        {},
    )
    gather = load_source_subset(
        root / "decode_attention.py",
        "beam_experiment_gather_fp32" if mha_qk_fp32 else "beam_experiment_gather",
        [
            "tanh",
            "_grouped_head_tiles",
            "_extract_kv_strides",
            "_mla_tuning_applies",
            "_mla_launch_plan",
            "_fwd_kernel_stage1",
            "_fwd_grouped_kernel_stage1",
            "_fwd_kernel_stage2",
            "_decode_att_m_fwd",
            "_decode_grouped_att_m_fwd",
            "_decode_softmax_reducev_fwd",
            "decode_attention_fwd_normal",
            "decode_attention_fwd_grouped",
            "decode_attention_fwd",
        ],
        ["_MIN_BLOCK_KV", "_GROUPED_BLOCK_H", "_MLA_BLOCK_N"],
        {
            "torch": torch,
            "triton": triton,
            "tl": tl,
            "Tuple": Tuple,
            "_is_hip": False,
            "_is_gfx1250": False,
            "logger": logging.getLogger("beam_experiment_gather"),
            "unpack_aux_tensors": auxiliary.unpack_aux_tensors,
        },
        mha_qk_fp32=mha_qk_fp32,
    )
    return shared, gather


def load_cuda(standalone=False, *, mha_qk_fp32=False):
    try:
        import torch
        import triton
    except ImportError as error:
        raise RuntimeError(
            "GPU modes require CUDA-enabled PyTorch and Triton; "
            "nothing was installed or benchmarked"
        ) from error
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError(
            "GPU modes require an NVIDIA CUDA GPU; no GPU checks were run"
        )
    if standalone:
        shared, gather = load_standalone_kernels(torch, triton, mha_qk_fp32=mha_qk_fp32)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
        try:
            from sglang.kernels.ops.attention import beam_decode_attention as shared
            from sglang.kernels.ops.attention import decode_attention as gather
        except ImportError as error:
            raise RuntimeError(
                "runtime imports unavailable; use --standalone "
                "for a torch/triton-only experiment"
            ) from error
    return torch, triton, shared, gather


def prepare_gpu_case(args, torch):
    if (
        min(
            args.requests,
            args.q_heads,
            args.kv_heads,
            args.scatter_factor,
            args.scan_spacing,
        )
        < 1
    ):
        raise ValueError(
            "requests, heads, spacing, and scatter factor must be positive"
        )
    if args.q_heads % args.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    snapshots = [snapshot_from_args(args, args.seed + r) for r in range(args.requests)]
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    shape = (
        args.requests * max(1, args.beams * args.steps),
        args.kv_heads,
        args.head_dim,
    )
    node_keys = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=dtype
    )
    node_values = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=dtype
    )
    q = torch.randn(
        (args.requests * args.beams, args.q_heads, args.head_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device="cuda", dtype=dtype)
    query_valid = torch.ones(q.shape[0], dtype=torch.bool, device="cuda")

    def make_layout(spacing):
        layouts = [relayout(s, spacing, args.seed + r) for r, s in enumerate(snapshots)]
        capacity = max(1, max(len(s.slots) for s in layouts))
        keys = torch.full(
            (args.requests * capacity, args.kv_heads, args.head_dim),
            math.nan,
            dtype=dtype,
            device="cuda",
        )
        values = torch.full_like(keys, math.nan)
        valid = torch.zeros((args.requests, capacity), dtype=torch.bool, device="cuda")
        ends = torch.tensor(
            [len(s.slots) for s in layouts], dtype=torch.int32, device="cuda"
        )
        source_slots, node_ids, local_paths, global_paths = [], [], [], []
        for r, layout in enumerate(layouts):
            for slot, node in enumerate(layout.slots):
                if node is not None:
                    source_slots.append(r * capacity + slot)
                    node_ids.append(r * max(1, args.beams * args.steps) + node)
            for path in layout.slot_paths():
                local_paths.append(path)
                global_paths.append([r * capacity + slot for slot in path])
        slots_tensor = torch.tensor(source_slots, dtype=torch.int64, device="cuda")
        ids_tensor = torch.tensor(node_ids, dtype=torch.int64, device="cuda")
        keys[slots_tensor] = node_keys[ids_tensor]
        values[slots_tensor] = node_values[ids_tensor]
        valid.view(-1)[slots_tensor] = True
        paths = torch.tensor(global_paths, dtype=torch.int64, device="cuda").reshape(
            q.shape[0], args.steps
        )
        local = torch.tensor(local_paths, dtype=torch.int64, device="cuda").reshape(
            q.shape[0], args.steps
        )
        return {
            "snapshots": layouts,
            "capacity": capacity,
            "k": keys,
            "v": values,
            "valid": valid,
            "ends": ends,
            "paths": paths,
            "local_paths": local,
        }

    dense = make_layout(args.scan_spacing)
    scattered = make_layout(args.scan_spacing * args.scatter_factor)
    visible = torch.zeros(
        (q.shape[0], dense["capacity"]), dtype=torch.bool, device="cuda"
    )
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
    return {
        "q": q,
        "query_valid": query_valid,
        "dense": dense,
        "scattered": scattered,
        "visible": visible,
        "out": output,
        "lse": lse,
        "source_trace": [s.trace[-1] if s.trace else None for s in snapshots],
    }


def rebuild_mask(case):
    visible = case["visible"]
    visible.zero_()
    visible.scatter_(1, case["dense"]["local_paths"], True)
    visible.logical_and_(case["query_valid"][:, None])


def shared_configs(args):
    configs = []
    seen = set()
    gqa = args.q_heads // args.kv_heads
    for compute, group, tile, split in itertools.product(
        args.compute, args.group_beams, args.tile, args.shared_splits
    ):
        cap = args.query_tile or (4 if compute == "simt" else 16)
        if compute == "simt" and cap > 8:
            raise ValueError("SIMT --query-tile must be at most 8")
        query_rows = min(min(group, args.beams) * gqa, cap)
        key = (compute, query_rows, tile, split)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            {
                "compute": compute,
                "group_beams": group,
                "query_tile": cap,
                "query_rows": query_rows,
                "tile": tile,
                "shared_splits": split,
            }
        )
    return configs


def shared_runner(args, case, shared_fwd, config):
    dense = case["dense"]
    split = config["shared_splits"]
    partial_out, partial_lse = case["shared_workspace"].get(split, (None, None))

    def run():
        shared_fwd(
            case["q"],
            dense["k"],
            dense["v"],
            case["visible"],
            dense["valid"],
            case["query_valid"],
            dense["ends"],
            case["out"],
            case["lse"],
            beams_per_request=args.beams,
            group_beams=config["group_beams"],
            tile_tokens=config["tile"],
            query_tile=config["query_tile"],
            compute=config["compute"],
            num_splits=split,
            partial_out=partial_out,
            partial_lse=partial_lse,
        )

    def total():
        rebuild_mask(case)
        run()

    return run, total


def gather_runner(args, case, layout, torch, gather_fwd, split_count):
    q = case["q"]
    rows, heads, dim = q.shape
    indptr = torch.arange(rows + 1, dtype=torch.int32, device=q.device) * args.steps
    indices = layout["paths"].flatten().clone()
    splits = torch.full((rows,), split_count, dtype=torch.int32, device=q.device)
    partial = torch.empty(
        (rows, heads, split_count, dim), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty(
        (rows, heads, split_count), dtype=torch.float32, device=q.device
    )
    output = torch.empty_like(q)

    def metadata():
        indices.copy_(layout["paths"].flatten())

    def run():
        gather_fwd(
            q,
            layout["k"],
            layout["v"],
            output,
            indptr,
            indices,
            partial,
            partial_lse,
            splits,
            split_count,
            1.0 / math.sqrt(dim),
            1.0,
            1.0,
            enable_lean=False,
        )

    def total():
        metadata()
        run()

    return {
        "run": run,
        "metadata": metadata,
        "total": total,
        "out": output,
        "workspace_bytes": (partial.numel() + partial_lse.numel()) * 4,
        "metadata_bytes": sum(
            t.numel() * t.element_size() for t in (indptr, indices, splits)
        ),
    }


def torch_reference(args, case, torch):
    q, layout = case["q"], case["dense"]
    result = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    lse = torch.full(q.shape[:2], -math.inf, dtype=torch.float64, device=q.device)
    if args.steps == 0:
        return result, lse
    head_map = torch.arange(args.q_heads, device=q.device) // (
        args.q_heads // args.kv_heads
    )
    for row in range(q.shape[0]):
        keys = layout["k"][layout["paths"][row]][:, head_map, :].double()
        values = layout["v"][layout["paths"][row]][:, head_map, :].double()
        scores = torch.einsum("hd,thd->ht", q[row].double(), keys) / math.sqrt(
            args.head_dim
        )
        allowed = (
            layout["local_paths"][row] < layout["ends"][row // args.beams]
        ) & case["query_valid"][row]
        scores.masked_fill_(~allowed[None, :], -math.inf)
        row_lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.where(
            torch.isfinite(row_lse[:, None]), scores.softmax(dim=-1), 0.0
        )
        result[row] = torch.einsum("ht,thd->hd", probabilities, values)
        lse[row] = row_lse
    return result, lse


def verify_gpu(args, case, torch, shared_fwd, baselines):
    rebuild_mask(case)
    reference, reference_lse = torch_reference(args, case, torch)
    tolerance = 1e-2 if args.dtype == "bfloat16" else 2e-3
    errors = []
    if args.steps:
        for label, baseline in baselines.items():
            baseline["run"]()
            torch.testing.assert_close(
                baseline["out"].double(), reference, atol=tolerance, rtol=tolerance
            )
            errors.append(
                {
                    "method": label,
                    "max_abs_error": (baseline["out"].double() - reference)
                    .abs()
                    .max()
                    .item(),
                }
            )
        layout = case["dense"]
        path = layout["paths"][0].tolist()
        cpu_state = reference_attention(
            case["q"][0, 0].double().tolist(),
            layout["k"][layout["paths"][0], 0].double().tolist(),
            layout["v"][layout["paths"][0], 0].double().tolist(),
            list(range(len(path))),
        )
        torch.testing.assert_close(
            reference[0, 0],
            torch.tensor(cpu_state[0], dtype=torch.float64, device="cuda"),
            atol=1e-10,
            rtol=1e-10,
        )
    saved_ends = case["dense"]["ends"].clone()
    configs = shared_configs(args)
    for config in configs:
        run, _ = shared_runner(args, case, shared_fwd, config)
        run()
        torch.testing.assert_close(
            case["out"].double(), reference, atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            case["lse"].double(), reference_lse, atol=1e-4, rtol=1e-4
        )
        errors.append(
            {
                "method": "shared",
                **config,
                "max_abs_error": (case["out"].double() - reference).abs().max().item(),
            }
        )
        case["query_valid"][-1] = False
        rebuild_mask(case)
        run()
        torch.testing.assert_close(
            case["out"][-1], torch.zeros_like(case["out"][-1]), atol=0, rtol=0
        )
        assert torch.isneginf(case["lse"][-1]).all().item()
        case["query_valid"].fill_(True)
        rebuild_mask(case)
        case["dense"]["ends"].zero_()
        run()
        torch.testing.assert_close(
            case["out"], torch.zeros_like(case["out"]), atol=0, rtol=0
        )
        assert torch.isneginf(case["lse"]).all().item()
        case["dense"]["ends"].copy_(saved_ends)
        if args.check_graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            for end in (0, 1, config["tile"] - 1, config["tile"] + 1, None):
                if end is None:
                    case["dense"]["ends"].copy_(saved_ends)
                else:
                    case["dense"]["ends"].copy_(saved_ends.clamp(max=end))
                graph.replay()
                expected, expected_lse = torch_reference(args, case, torch)
                torch.testing.assert_close(
                    case["out"].double(), expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    case["lse"].double(), expected_lse, atol=1e-4, rtol=1e-4
                )
    torch.cuda.synchronize()
    return {
        "output_atol": tolerance,
        "output_rtol": tolerance,
        "errors": errors,
        "graph_checked": args.check_graph,
        "graph_configurations_checked": len(configs) if args.check_graph else 0,
    }


def measure_cuda(
    torch, call, warmup, repeats, scrub, execution="graph", graph_batch=64
):
    if min(warmup, repeats, graph_batch) < 1:
        raise ValueError("warmup, repeats, and graph_batch must be positive")
    if execution not in ("eager", "graph"):
        raise ValueError("execution must be eager or graph")
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    calls_per_sample = 1
    measured_call = call
    if execution == "graph":
        # Scrubbing once before a batch would leave all but the first call warm.
        calls_per_sample = graph_batch if scrub is None else 1
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls_per_sample):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        measured_call = graph.replay
        measured_call()
        torch.cuda.synchronize()
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(repeats)
    ]
    for start, end in events:
        if scrub is not None:
            scrub.add_(1)
        start.record()
        measured_call()
        end.record()
    torch.cuda.synchronize()
    samples = sorted(
        start.elapsed_time(end) * 1000.0 / calls_per_sample for start, end in events
    )
    return {
        "execution": execution,
        "calls_per_sample": calls_per_sample,
        "sample_count": repeats,
        "sample_unit": "per_call_average_of_replay"
        if execution == "graph"
        else "eager_call",
        "host_submission": (
            "one replay submission gap amortized over calls_per_sample"
            if execution == "graph"
            else "Python dispatch gaps may be included in the GPU event interval"
        ),
        "median_us": statistics.median(samples),
        "p10_us": samples[int((len(samples) - 1) * 0.1)],
        "p90_us": samples[int((len(samples) - 1) * 0.9)],
    }


def summarize_timings(timings):
    summaries = []
    for execution in sorted({row["execution"] for row in timings}):
        for scope in sorted({row["scope"] for row in timings}):
            rows = [
                r
                for r in timings
                if r["execution"] == execution and r["scope"] == scope
            ]
            for method in ("A_scattered_gather", "B_pool_gather"):
                gather = [r for r in rows if r["method"] == method]
                shared = [r for r in rows if r["method"] == "C_shared_scan"]
                if not gather or not shared:
                    continue
                baseline = min(gather, key=lambda r: r["median_us"])
                best = min(shared, key=lambda r: r["median_us"])
                summaries.append(
                    {
                        "execution": execution,
                        "scope": scope,
                        "baseline": method,
                        "baseline_splits": baseline["splits"],
                        "baseline_median_us": baseline["median_us"],
                        "shared_group_beams": best["group_beams"],
                        "shared_tile": best["tile"],
                        "shared_config": {
                            key: best[key]
                            for key in (
                                "compute",
                                "query_tile",
                                "query_rows",
                                "shared_splits",
                            )
                        },
                        "shared_median_us": best["median_us"],
                        "observed_speedup": baseline["median_us"] / best["median_us"]
                        if best["median_us"] > 0
                        else None,
                    }
                )
    return summaries


def inspect_compilation(call, jit_functions, dump_dir=None):
    captured = []
    originals = []
    for jit in jit_functions:
        previous = jit.__dict__.get("run")
        original = jit.run

        def capture(*values, _original=original, **kwargs):
            kernel = _original(*values, **kwargs)
            captured.append((kernel, kwargs.get("grid")))
            return kernel

        originals.append((jit, previous))
        jit.run = capture
    try:
        call()
    finally:
        for jit, previous in originals:
            if previous is None:
                del jit.run
            else:
                jit.run = previous
    result = []
    for kernel, grid in captured:
        metadata = kernel.metadata
        ptx = kernel.asm.get("ptx", "")
        info = {
            "name": kernel.name,
            "hash": kernel.hash,
            "grid": list(grid) if isinstance(grid, tuple) else str(grid),
            "target": str(metadata.target),
            "registers_per_thread": getattr(kernel, "n_regs", None),
            "spills_reported_by_triton": getattr(kernel, "n_spills", None),
            "shared_memory_bytes": metadata.shared,
            "num_warps": metadata.num_warps,
            "num_stages": getattr(metadata, "num_stages", None),
            "ptx_mma_instruction_count": len(
                re.findall(r"\b(?:mma\.sync|wgmma\.mma_async|tcgen05\.mma)\b", ptx)
            ),
            "ptx_fma_instruction_count": len(re.findall(r"\bfma\.", ptx)),
        }
        if dump_dir is not None:
            stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{kernel.name}-{kernel.hash}")
            files = {}
            for extension in ("ptx", "ttgir", "cubin", "sass"):
                try:
                    code = kernel.asm[extension]
                except Exception as error:
                    info[f"{extension}_unavailable"] = str(error)
                    continue
                target = dump_dir / f"{stem}.{extension}"
                if not target.exists():
                    if isinstance(code, bytes):
                        target.write_bytes(code)
                    else:
                        target.write_text(code)
                files[extension] = str(target)
            info["files"] = files
        result.append(info)
    if not result:
        raise RuntimeError(
            "no Triton launches captured; compiler inspection is unavailable"
        )
    return result


def gpu_environment(torch):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,mig.mode.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        driver = f"unavailable: {error}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "multiprocessor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "driver_query": driver,
    }


def padded_geometry(args, snapshot, config):
    gqa = args.q_heads // args.kv_heads
    block_m = 1 << (config["query_rows"] - 1).bit_length()
    if config["compute"] == "dot":
        block_m = max(16, block_m)
    groups = math.ceil(args.beams * gqa / config["query_rows"])
    slots = math.ceil(len(snapshot.slots) / config["tile"]) * config["tile"]
    valid_pairs = args.beams * args.steps * gqa
    return {
        **config,
        "block_m": block_m,
        "blocks_per_request": groups * args.kv_heads * config["shared_splits"],
        "padded_slots": slots,
        "padded_pair_ratio": groups * block_m * slots / valid_pairs
        if valid_pairs
        else None,
    }


def run_gpu(args):
    if (
        min(
            args.warmup, args.repeats, args.graph_batch, args.eviction_mib, *args.splits
        )
        < 1
    ):
        raise ValueError(
            "warmup, repeats, graph_batch, eviction_mib, and splits must be positive"
        )
    if args.mode == "benchmark" and args.steps < 1:
        raise ValueError(
            "benchmark requires steps > 0; use gpu-check for empty history"
        )
    torch, triton, shared_module, gather_module = load_cuda(args.standalone)
    shared_fwd = shared_module.beam_decode_attention_fwd
    gather_fwd = gather_module.decode_attention_fwd
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    case = prepare_gpu_case(args, torch)
    configs = shared_configs(args)
    case["shared_workspace"] = {
        split: (
            torch.empty((split, *case["q"].shape), dtype=torch.float32, device="cuda"),
            torch.empty(
                (split, *case["q"].shape[:2]), dtype=torch.float32, device="cuda"
            ),
        )
        for split in args.shared_splits
        if split > 1
    }
    baselines = {}
    if args.steps:
        for name, layout in (
            ("A_scattered_gather", "scattered"),
            ("B_pool_gather", "dense"),
        ):
            for split in args.splits:
                baseline = gather_runner(
                    args, case, case[layout], torch, gather_fwd, split
                )
                baseline.update(method=name, splits=split)
                baselines[f"{name}_split{split}"] = baseline
    correctness = verify_gpu(args, case, torch, shared_fwd, baselines)
    result = {
        "kind": "measured_cuda" if args.mode == "benchmark" else "cuda_correctness",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "environment": gpu_environment(torch),
        "parameters": vars(args),
        "correctness": correctness,
        "source_allocator_state": case["source_trace"],
        "layout_note": (
            "offline slot permutation preserves holes; "
            "no runtime compaction or VMM is measured"
        ),
        "timing_note": (
            "explicit eager and graph results using ordinary CUDA events; "
            "warm graphs amortize one replay submission over graph_batch calls; "
            "scrub graphs contain one call and retain replay submission overhead"
        ),
        "scan_note": (
            "KV loads skip globally invalid slots; "
            "dense tile compute and mask loads still span S"
        ),
        "scope": (
            "prebuilt parent paths; excludes selection, model forward, "
            "VMM, and initial data placement"
        ),
        "metadata_scope": (
            "gather copies path indices; shared zeroes/scatters the ancestry mask"
        ),
        "buffer_bytes": {
            "dense_kv": 2 * case["dense"]["k"].numel() * case["q"].element_size(),
            "scattered_kv": 2
            * case["scattered"]["k"].numel()
            * case["q"].element_size(),
            "shared_visibility": case["visible"].numel(),
            "shared_valid_slots": case["dense"]["valid"].numel(),
            "shared_output_and_lse": case["out"].numel() * case["out"].element_size()
            + case["lse"].numel() * 4,
            "shared_workspace_by_split": {
                str(split): sum(t.numel() * t.element_size() for t in workspace)
                for split, workspace in case["shared_workspace"].items()
            },
            "gather_workspace_by_variant": {
                name: b["workspace_bytes"] for name, b in baselines.items()
            },
            "gather_metadata_by_variant": {
                name: b["metadata_bytes"] for name, b in baselines.items()
            },
        },
    }
    gather_jits = (
        gather_module._fwd_kernel_stage1,
        gather_module._fwd_grouped_kernel_stage1,
        gather_module._fwd_kernel_stage2,
    )
    compiled = []
    for baseline in baselines.values():
        compiled.append(
            {
                "method": baseline["method"],
                "splits": baseline["splits"],
                "kernels": inspect_compilation(baseline["run"], gather_jits, dump_dir),
            }
        )
    for config in configs:
        run, _ = shared_runner(args, case, shared_fwd, config)
        compiled.append(
            {
                "method": "C_shared_scan",
                **config,
                "kernels": inspect_compilation(
                    run,
                    (
                        shared_module._beam_decode_attention,
                        shared_module._merge_beam_splits,
                    ),
                    dump_dir,
                ),
            }
        )
    result["compiled_kernels"] = compiled
    result["compiler_note"] = (
        "PTX counts are static instructions, not executed counts; "
        "inspect SASS to confirm lowering; missing disassembly is reported explicitly"
    )
    result["padded_geometry"] = [
        padded_geometry(args, snapshot, config)
        for snapshot in case["dense"]["snapshots"]
        for config in configs
    ]
    result["source_hashes"] = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in (("shared", shared_module), ("gather", gather_module))
    }
    result["loader"] = (
        "CUDA-only source subset; disabled PDL blocks removed, numeric bodies unchanged"
        if args.standalone
        else "SGLang runtime imports"
    )
    if args.standalone:
        result["standalone_source"] = {
            "loaded_gather_sha256": gather_module.loaded_source_sha256,
            "pruned_pdl_blocks": gather_module.pruned_pdl_blocks,
            "use_pdl": False,
        }
    if args.mode == "gpu-check":
        return result
    scrub = (
        torch.zeros(args.eviction_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if args.cache_mode == "scrub"
        else None
    )
    result["cache_note"] = (
        "warm repeats"
        if scrub is None
        else (
            "best-effort cache pressure before each sample; scrub time excluded, "
            "not a guaranteed cold-cache state"
        )
    )
    timings = []
    executions = ("eager", "graph") if args.execution == "both" else (args.execution,)
    workloads = []
    for baseline in baselines.values():
        for scope in args.scopes:
            workloads.append(
                (
                    {
                        "method": baseline["method"],
                        "splits": baseline["splits"],
                        "scope": scope,
                    },
                    baseline[scope],
                )
            )
    for config in configs:
        run, total = shared_runner(args, case, shared_fwd, config)
        calls = {"run": run, "metadata": lambda: rebuild_mask(case), "total": total}
        for scope in args.scopes:
            workloads.append(
                ({"method": "C_shared_scan", **config, "scope": scope}, calls[scope])
            )
    random.Random(args.seed).shuffle(workloads)
    for fields, call in workloads:
        for execution in executions:
            label = "/".join(f"{key}={value}" for key, value in fields.items())
            with torch.cuda.nvtx.range(f"beam_kv/{execution}/{label}"):
                measurement = measure_cuda(
                    torch,
                    call,
                    args.warmup,
                    args.repeats,
                    scrub,
                    execution=execution,
                    graph_batch=args.graph_batch,
                )
            timings.append({**fields, **measurement})
    result["timings"] = timings
    result["comparisons"] = summarize_timings(timings)
    result["analytical_scan_geometry"] = [
        traffic_model(snapshot, group, tile, args.head_dim)
        for snapshot in case["dense"]["snapshots"]
        for group in args.group_beams
        for tile in args.tile
    ]
    result["analytical_geometry_note"] = (
        "uncapped conceptual beam groups only; "
        "actual capped query/split geometry is in padded_geometry"
    )
    result["hardware_counters"] = (
        "not collected; use Nsight Compute for sectors, L2/DRAM traffic, and occupancy"
    )
    return result


def matrix_cases(preset):
    if preset.startswith("a30-"):
        return [
            (name, "a30-case", parameters) for name, parameters in a30_cases(preset)
        ]
    if preset == "v100-tune":
        common = {
            "standalone": True,
            "compute": ["dot", "simt"],
            "group_beams": [4, 16],
            "tile": [16, 32],
            "shared_splits": [1, 2, 4],
        }
        return [
            (name, mode, common | overrides)
            for name, mode, overrides in (
                (
                    "check_mha",
                    "gpu-check",
                    {"beams": 9, "steps": 33, "scan_spacing": 2, "check_graph": True},
                ),
                (
                    "check_gqa",
                    "gpu-check",
                    {
                        "beams": 9,
                        "steps": 33,
                        "kv_heads": 2,
                        "scan_spacing": 2,
                        "check_graph": True,
                    },
                ),
                (
                    "check_empty",
                    "gpu-check",
                    {"beams": 1, "steps": 0, "check_graph": True},
                ),
                ("k8_shared", "benchmark", {"beams": 8, "steps": 128}),
                ("k32_shared", "benchmark", {"beams": 32, "steps": 128}),
                (
                    "k32_independent",
                    "benchmark",
                    {"beams": 32, "steps": 128, "pattern": "independent"},
                ),
                (
                    "k32_random",
                    "benchmark",
                    {"beams": 32, "steps": 128, "pattern": "random"},
                ),
                ("gqa_shared", "benchmark", {"beams": 32, "steps": 128, "kv_heads": 2}),
                (
                    "requests8_shared",
                    "benchmark",
                    {"beams": 32, "steps": 128, "requests": 8},
                ),
            )
        ]
    if preset == "compiler":
        return [
            (
                name,
                mode,
                {
                    "standalone": True,
                    "compute": ["dot"],
                    "group_beams": [4, 8, 16],
                    "tile": [32, 64],
                    "shared_splits": [1],
                    "query_tile": 128,
                }
                | overrides,
            )
            for name, mode, overrides in (
                (
                    "compiler_check",
                    "gpu-check",
                    {"beams": 9, "steps": 33, "check_graph": True},
                ),
                ("compiler_k8", "benchmark", {"beams": 8, "steps": 128}),
                ("compiler_k32", "benchmark", {"beams": 32, "steps": 128}),
                (
                    "compiler_requests8",
                    "benchmark",
                    {"beams": 32, "steps": 128, "requests": 8},
                ),
            )
        ]
    cases = [
        (
            "check_mha_tails",
            "gpu-check",
            {"beams": 9, "steps": 17, "scan_spacing": 2, "check_graph": True},
        ),
        (
            "check_gqa_tails",
            "gpu-check",
            {
                "beams": 9,
                "steps": 17,
                "kv_heads": 2,
                "scan_spacing": 2,
                "check_graph": True,
            },
        ),
        ("check_empty", "gpu-check", {"beams": 1, "steps": 0, "check_graph": True}),
        (
            "timing_control",
            "benchmark",
            {"beams": 8, "steps": 128, "execution": "both"},
        ),
        ("tile128_diagnostic", "gpu-check", {"beams": 8, "steps": 128, "tile": [128]}),
    ]
    for pattern in ("shared", "independent", "random"):
        cases.append(
            (
                f"k32_{pattern}",
                "benchmark",
                {"beams": 32, "steps": 128, "pattern": pattern},
            )
        )
    cases.extend(
        [
            ("k128_shared", "benchmark", {"beams": 128, "steps": 128}),
            (
                "requests8_shared",
                "benchmark",
                {"beams": 32, "steps": 128, "requests": 8},
            ),
            ("gqa_shared", "benchmark", {"beams": 32, "steps": 128, "kv_heads": 2}),
            (
                "scrub_shared",
                "benchmark",
                {"beams": 32, "steps": 128, "cache_mode": "scrub"},
            ),
        ]
    )
    if preset == "full":
        cases.extend(
            [
                (
                    "k8_independent",
                    "benchmark",
                    {"beams": 8, "steps": 128, "pattern": "independent"},
                ),
                (
                    "k128_independent",
                    "benchmark",
                    {"beams": 128, "steps": 128, "pattern": "independent"},
                ),
                (
                    "k128_random",
                    "benchmark",
                    {"beams": 128, "steps": 128, "pattern": "random"},
                ),
                (
                    "requests8_independent",
                    "benchmark",
                    {
                        "beams": 32,
                        "steps": 128,
                        "requests": 8,
                        "pattern": "independent",
                    },
                ),
                (
                    "requests8_gqa",
                    "benchmark",
                    {"beams": 32, "steps": 128, "requests": 8, "kv_heads": 2},
                ),
                (
                    "scrub_independent",
                    "benchmark",
                    {
                        "beams": 32,
                        "steps": 128,
                        "cache_mode": "scrub",
                        "pattern": "independent",
                    },
                ),
                (
                    "gqa_random",
                    "benchmark",
                    {"beams": 32, "steps": 128, "kv_heads": 2, "pattern": "random"},
                ),
                (
                    "short_decode",
                    "benchmark",
                    {"beams": 32, "steps": 16, "head_dim": 64},
                ),
                ("long_decode", "benchmark", {"beams": 32, "steps": 1024}),
                ("holes", "benchmark", {"beams": 32, "steps": 128, "scan_spacing": 4}),
                (
                    "append_only",
                    "benchmark",
                    {"beams": 32, "steps": 128, "allocator": "append"},
                ),
            ]
        )
    return cases


def matrix_commands(args):
    cases = matrix_cases(args.preset)
    if args.case:
        unknown = set(args.case) - {name for name, _, _ in cases}
        if unknown:
            raise ValueError(f"unknown matrix cases: {sorted(unknown)}")
        cases = [case for case in cases if case[0] in args.case]
    root = Path(args.output_dir)
    commands = []
    for name, mode, overrides in cases:
        parameters = {
            "dtype": "float16",
            "group_beams": [4, 8, 16],
            "tile": [32, 64],
            "splits": [1, 2, 4],
            "shared_splits": [1],
            "compute": ["dot"],
            "execution": "graph",
            "graph_batch": args.graph_batch,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "output": str(root / f"{name}.json"),
            "dump_dir": str(root / "kernels" / name),
            **overrides,
        }
        if mode == "a30-case":
            parameters = {
                "graph_batch": args.graph_batch,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "output": str(root / f"{name}.json"),
                **overrides,
            }
            if args.preset in ("a30-smoke", "a30-calibrate"):
                parameters["dump_dir"] = str(root / "kernels" / name)
            if args.preset == "a30-smoke":
                parameters["check_partial_lse"] = True
            if args.policy:
                parameters["policy"] = args.policy
        command = [
            args.python or sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            mode,
        ]
        for key, value in parameters.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.append(flag)
                command.extend(
                    str(v) for v in (value if isinstance(value, list) else [value])
                )
        commands.append(
            {"name": name, "command": command, "output": parameters["output"]}
        )
    return commands


def run_matrix(args):
    if min(args.graph_batch, args.repeats, args.warmup) < 1:
        raise ValueError("graph_batch, repeats, and warmup must be positive")
    commands = matrix_commands(args)
    if args.dry_run:
        return {"kind": "matrix_dry_run", "cases": commands}
    root = Path(args.output_dir)
    if root.exists():
        raise ValueError(
            "matrix output directory already exists; select a fresh directory"
        )
    root.mkdir(parents=True)
    manifest = {"kind": "matrix_run", "complete": False, "cases": []}
    for index, item in enumerate(commands):
        print(
            f"[{index + 1}/{len(commands)}] {item['name']}", file=sys.stderr, flush=True
        )
        log = root / f"{item['name']}.log"
        with log.open("w") as stream:
            process = subprocess.run(
                item["command"], stdout=stream, stderr=subprocess.STDOUT
            )
        record = {**item, "returncode": process.returncode, "log": str(log)}
        if process.returncode == 0:
            result = json.loads(Path(item["output"]).read_text())
            record["status"] = result.get("status", "completed")
        manifest["cases"].append(record)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"matrix stopped at {item['name']}; inspect {log}")
    manifest["complete"] = True
    manifest["resource_skipped"] = sum(
        c.get("status") == "resource_skipped" for c in manifest["cases"]
    )
    manifest["all_cases_validated"] = manifest["resource_skipped"] == 0
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def add_gpu_arguments(parser):
    add_tree_arguments(parser)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--group-beams",
        type=int,
        nargs="+",
        choices=(1, 2, 4, 8, 16),
        default=[4, 8, 16],
    )
    parser.add_argument(
        "--tile", type=int, nargs="+", choices=(16, 32, 64, 128), default=[32, 64]
    )
    parser.add_argument(
        "--compute", nargs="+", choices=("dot", "simt"), default=["dot"]
    )
    parser.add_argument(
        "--query-tile",
        type=int,
        choices=(1, 2, 4, 8, 16, 32, 64, 128),
        help="cap query rows per KV head; default dot=16, SIMT=4",
    )
    parser.add_argument(
        "--shared-splits", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1]
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="load CUDA numeric kernels without importing the SGLang runtime",
    )
    parser.add_argument(
        "--scan-spacing",
        type=int,
        default=1,
        help="spacing between stored slots in the shared pool",
    )
    parser.add_argument(
        "--scatter-factor",
        type=int,
        default=8,
        help="additional spacing for baseline A",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="existing decode kernel KV split counts to compare",
    )
    parser.add_argument(
        "--execution", choices=("eager", "graph", "both"), default="both"
    )
    parser.add_argument(
        "--graph-batch", type=int, default=64, help="calls per warm graph replay sample"
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=("run", "metadata", "total"),
        default=["run", "metadata", "total"],
    )
    parser.add_argument(
        "--dump-dir", help="export compiled PTX/TTGIR/cubin/SASS to this directory"
    )
    parser.add_argument(
        "--output", help="write JSON to a new file; refuse to overwrite"
    )
    parser.add_argument("--cache-mode", choices=("warm", "scrub"), default="warm")
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--check-graph",
        action="store_true",
        help="check replay with changing scan lengths, not VMM mappings",
    )


def add_tree_arguments(parser):
    parser.add_argument("--beams", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--pattern",
        choices=("shared", "independent", "grouped", "random"),
        default="shared",
    )
    parser.add_argument("--allocator", choices=("reuse", "append"), default="reuse")
    parser.add_argument("--concentration", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-capacity", type=int)


def snapshot_from_args(args, seed=None):
    return simulate_tree(
        args.beams,
        args.steps,
        args.pattern,
        args.allocator,
        args.concentration,
        args.seed if seed is None else seed,
        args.initial_capacity,
    )


A30_MAIN_WIDTHS = (1, 2, 3, 6, 8, 12, 16, 24, 32, 64, 128)
A30_STRESS_WIDTHS = (256, 512, 1024, 2048, 4096)
A30_OUTPUTS = (3, 16, 32, 64, 128)


def parent_trace(beams, steps, pattern, seed):
    rng = random.Random(seed)
    trace = []
    for step in range(steps):
        if step == 0 or pattern == "independent":
            parents = range(beams)
        elif pattern == "shared":
            parents = [rng.randrange(beams)] * beams
        elif pattern == "grouped":
            width = max(2, math.ceil(beams / 4))
            parents = [min(i // width * width, beams - 1) for i in range(beams)]
        else:
            favored = rng.randrange(beams)
            parents = [
                favored if rng.random() < 0.5 else rng.randrange(beams)
                for _ in range(beams)
            ]
        trace.append(array("i", parents))
    return trace


def trace_snapshot(trace, beams, depth, compact=True):
    rank = list(range(beams))
    for parents in trace[:depth]:
        order = sorted(range(beams), key=lambda row: rank[parents[row]])
        rank = [0] * beams
        for i, row in enumerate(order):
            rank[row] = i
    paths = array("i", [0]) * (beams * depth)
    for row in range(beams):
        ancestor = row
        for position in range(depth - 1, -1, -1):
            paths[row * depth + position] = position * beams + ancestor
            ancestor = trace[position][ancestor]
    nodes = sorted(set(paths))
    if compact:
        locations = {node: slot for slot, node in enumerate(nodes)}
        slots = array("i", (locations[node] for node in paths))
    else:
        slots = paths
    order = sorted(range(beams), key=rank.__getitem__)
    return slots, array("i", order), len(nodes)


def group_prefix_oracle(
    paths, order, beams, depth, group_beams, min_prefix=0, min_saved=0
):
    prefixes = []
    for start in range(0, beams, group_beams):
        members = order[start : start + group_beams]
        first, last = members[0], members[-1]
        common = 0
        while (
            common < depth
            and paths[first * depth + common] == paths[last * depth + common]
        ):
            common += 1
        if (
            len(members) < 2
            or common < min_prefix
            or (len(members) - 1) * common < min_saved
        ):
            common = 0
        prefixes.append(common)
    return prefixes


def a30_memory_estimate(
    beams,
    depth,
    requests,
    q_heads,
    kv_heads,
    dim,
    layers,
    query_tile,
    splits,
    reference_rows,
    prompt_len=0,
):
    rows = beams * requests
    slots = rows * depth + requests * prompt_len
    width = max(1, depth + prompt_len)
    groups = requests * math.ceil(beams / max(1, query_tile // (q_heads // kv_heads)))
    kv = 2 * slots * kv_heads * dim * 2 * layers
    paths = rows * width * 4
    metadata = paths * 5 + rows * 128 + groups * 4
    query_output = rows * q_heads * (dim * 2 * 2 + 4) * layers
    partial = rows * q_heads * (dim + 1) * 4 * splits * 2
    reference = (
        min(rows, reference_rows) * q_heads * width * (dim * 24 + 32)
        + rows * q_heads * (dim + 1) * 8 * 4
    )
    return {
        "kv_upper_bound_bytes": kv,
        "metadata_bytes": metadata,
        "query_output_bytes": query_output,
        "partial_workspace_bytes": partial,
        "reference_chunk_bytes": reference,
        "estimated_peak_bytes": math.ceil(
            1.25 * (kv + metadata + query_output + partial + reference)
        )
        + (256 << 20),
        "dense_ancestry_mask_bytes": 0,
    }


def a30_cases(preset):
    if preset in ("a30-main", "a30-stress"):
        widths = A30_MAIN_WIDTHS if preset == "a30-main" else A30_STRESS_WIDTHS
        return [
            (
                f"k{k}_n{n}_{pattern}_{heads}",
                {"beams": k, "output_tokens": n, "pattern": pattern, "kv_heads": kv},
            )
            for k, n, pattern, (heads, kv) in itertools.product(
                widths,
                A30_OUTPUTS,
                ("shared", "grouped", "independent"),
                (("mha", 8), ("gqa", 2)),
            )
        ]
    if preset == "a30-smoke":
        return [
            ("single", {"beams": 1, "output_tokens": 3}),
            ("mha_tail", {"beams": 3, "output_tokens": 16, "pattern": "grouped"}),
            (
                "mha_prefix",
                {
                    "beams": 3,
                    "output_tokens": 64,
                    "pattern": "shared",
                    "require_shared_prefix": True,
                },
            ),
            ("gqa_tail", {"beams": 6, "output_tokens": 32, "kv_heads": 2}),
            (
                "independent",
                {"beams": 12, "output_tokens": 3, "pattern": "independent"},
            ),
        ]
    if preset == "a30-calibrate":
        return [
            (
                f"resources_w{warps}_p{stages}",
                {
                    "beams": 32,
                    "output_tokens": 64,
                    "query_tile": 32,
                    "tile": 64,
                    "warps": warps,
                    "stages": stages,
                },
            )
            for warps, stages in ((4, 1), (4, 3), (8, 2))
        ] + [
            (
                f"k{k}_q{q}_t{tile}_s{split}_{heads}",
                {
                    "beams": k,
                    "output_tokens": 64,
                    "query_tile": q,
                    "tile": tile,
                    "prefix_splits": split,
                    "kv_heads": kv,
                },
            )
            for k, (heads, kv), (q, tile, split) in itertools.product(
                (8, 32, 128),
                (("mha", 8), ("gqa", 2)),
                (
                    (16, 32, 1),
                    (16, 64, 2),
                    (32, 32, 2),
                    (32, 64, 4),
                    (32, 128, 1),
                    (16, 32, 8),
                ),
            )
        ]
    if preset == "a30-trace":
        return [
            (
                "random_seed47",
                {
                    "beams": 24,
                    "output_tokens": 32,
                    "pattern": "random",
                    "seed": 47,
                    "trace": True,
                },
            ),
            (
                "random_seed71",
                {
                    "beams": 24,
                    "output_tokens": 32,
                    "pattern": "random",
                    "seed": 71,
                    "trace": True,
                },
            ),
            ("trace_shared", {"beams": 24, "output_tokens": 32, "trace": True}),
            (
                "trace_grouped",
                {"beams": 24, "output_tokens": 32, "pattern": "grouped", "trace": True},
            ),
            ("layers8", {"beams": 32, "output_tokens": 64, "layers": 8}),
            (
                "layers32",
                {"beams": 32, "output_tokens": 64, "layers": 32, "kv_heads": 2},
            ),
            ("requests4", {"beams": 24, "output_tokens": 64, "requests": 4}),
            ("bf16", {"beams": 32, "output_tokens": 64, "dtype": "bfloat16"}),
            ("head64", {"beams": 12, "output_tokens": 64, "head_dim": 64}),
            ("prompt512", {"beams": 12, "output_tokens": 32, "prompt_len": 512}),
            ("prompt2048", {"beams": 12, "output_tokens": 32, "prompt_len": 2048}),
            ("prompt8192", {"beams": 12, "output_tokens": 32, "prompt_len": 8192}),
            (
                "cache_pressure",
                {"beams": 32, "output_tokens": 64, "cache_mode": "scrub"},
            ),
        ]
    raise ValueError(f"unknown A30 preset: {preset}")


def a30_reference(torch, q, k, v, paths, length, out, lse, chunk_rows):
    head_map = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[1])
    for start in range(0, q.shape[0], chunk_rows):
        end = min(start + chunk_rows, q.shape[0])
        slots = paths[start:end, :length].long()
        keys = k[slots][:, :, head_map, :].double()
        values = v[slots][:, :, head_map, :].double()
        scores = torch.einsum("bhd,bthd->bht", q[start:end].double(), keys) / math.sqrt(
            q.shape[2]
        )
        out[start:end] = torch.einsum("bht,bthd->bhd", scores.softmax(-1), values)
        lse[start:end] = torch.logsumexp(scores, -1)


def check_a30_partial_lse(
    torch,
    q,
    k,
    indptr,
    indices,
    partial_lse,
    prefix_lse,
    paths,
    row_prefix,
    prompt_len,
    prefix_tile,
    chunk_rows,
):
    head_map = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[1])
    ptr = indptr.cpu().tolist()
    counts = row_prefix.cpu().tolist() if prefix_lse is not None else [0] * q.shape[0]
    max_errors = {"suffix": 0.0, "prefix": 0.0}
    for start in range(0, q.shape[0], chunk_rows):
        for row in range(start, min(start + chunk_rows, q.shape[0])):
            suffix_length = ptr[row + 1] - ptr[row]
            suffix_splits = partial_lse.shape[2]
            span = math.ceil(math.ceil(suffix_length / suffix_splits) / 32) * 32
            for split in range(suffix_splits):
                begin, end = split * span, min((split + 1) * span, suffix_length)
                if begin >= end:
                    continue
                slots = indices[ptr[row] + begin : ptr[row] + end].long()
                keys = k[slots][:, head_map, :].double()
                expected = torch.logsumexp(
                    torch.einsum("hd,thd->ht", q[row].double(), keys)
                    / math.sqrt(q.shape[2]),
                    -1,
                )
                actual = partial_lse[row, :, split].double()
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
                max_errors["suffix"] = max(
                    max_errors["suffix"], (actual - expected).abs().max().item()
                )
            if prefix_lse is None:
                continue
            span = (
                math.ceil(math.ceil(counts[row] / prefix_tile) / prefix_lse.shape[0])
                * prefix_tile
            )
            for split in range(prefix_lse.shape[0]):
                begin, end = split * span, min((split + 1) * span, counts[row])
                actual = prefix_lse[split, row].double()
                if begin >= end:
                    if not torch.isneginf(actual).all().item():
                        raise AssertionError(
                            "empty prefix partition must have negative infinite LSE"
                        )
                    continue
                slots = paths[row, prompt_len + begin : prompt_len + end].long()
                keys = k[slots][:, head_map, :].double()
                expected = torch.logsumexp(
                    torch.einsum("hd,thd->ht", q[row].double(), keys)
                    / math.sqrt(q.shape[2]),
                    -1,
                )
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
                max_errors["prefix"] = max(
                    max_errors["prefix"], (actual - expected).abs().max().item()
                )
    return max_errors


def parse_parent_trace(raw, requests, depth, beams):
    data = raw.get("parents") if isinstance(raw, dict) else None
    if not isinstance(data, list) or len(data) != requests:
        raise ValueError("trace JSON must contain parents[request][step][beam]")
    for request in data:
        if not isinstance(request, list) or len(request) != depth:
            raise ValueError("trace step count does not match output_tokens")
        for step in request:
            if (
                not isinstance(step, list)
                or len(step) != beams
                or any(
                    type(parent) is not int or not 0 <= parent < beams
                    for parent in step
                )
            ):
                raise ValueError(
                    "trace shape or parent indices do not match the experiment"
                )
    return [[array("i", step) for step in request] for request in data]


def a30_candidate_policy(args):
    policy = {
        "schema": "beam-prefix-policy-v1",
        "min_prefix": args.min_prefix,
        "min_saved": args.min_saved,
    }
    if args.policy:
        policy = json.loads(Path(args.policy).read_text())
    if not isinstance(policy, dict) or policy.get("schema") != "beam-prefix-policy-v1":
        raise ValueError("unknown policy schema")
    choices = {
        "query_tile": (16, 32, 64),
        "tile": (32, 64, 128),
        "prefix_splits": (1, 2, 4, 8),
        "suffix_splits": (1, 2, 4, 8),
        "warps": (4, 8),
        "stages": (1, 2, 3),
    }
    if set(policy) - {"schema", "min_prefix", "min_saved", *choices}:
        raise ValueError("unknown policy field")
    for name in ("min_prefix", "min_saved"):
        if type(policy.get(name)) is not int or policy[name] < 0:
            raise ValueError("policy thresholds must be nonnegative integers")
    for name, allowed in choices.items():
        if name in policy and (
            type(policy[name]) is not int or policy[name] not in allowed
        ):
            raise ValueError(f"invalid policy {name}")
    return policy


def run_a30(args):
    policy = a30_candidate_policy(args)
    args = argparse.Namespace(**vars(args))
    for key in ("query_tile", "tile", "prefix_splits", "warps", "stages"):
        if key in policy:
            setattr(args, key, policy[key])
    if args.parent_trace and not args.trace:
        raise ValueError("--parent-trace requires --trace")
    positive = (
        args.beams,
        args.requests,
        args.output_tokens,
        args.q_heads,
        args.kv_heads,
        args.layers,
        args.reference_rows,
        args.repeats,
        args.warmup,
    )
    if min(positive) < 1 or args.output_tokens < 2 or args.prompt_len < 0:
        raise ValueError("positive sizes and at least two output tokens are required")
    if args.q_heads % args.kv_heads or args.query_tile < args.q_heads // args.kv_heads:
        raise ValueError("query tile must accommodate an integral GQA head group")
    if "gather" not in args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError("modes must be unique and include the gather baseline")
    if args.require_shared_prefix and ("shared" not in args.modes or args.beams < 2):
        raise ValueError(
            "--require-shared-prefix needs shared mode and at least two beams"
        )
    if args.graph_batch < 1 or args.eviction_mib < 1:
        raise ValueError("graph_batch and eviction_mib must be positive")
    if not 0 < args.memory_fraction <= 0.8 or args.memory_gib <= 0:
        raise ValueError("memory fraction must be in (0, 0.8] and memory GiB positive")
    torch, triton, shared, gather = load_cuda(True, mha_qk_fp32=True)
    original_gather = None
    if args.original_diagnostic:
        shared, original_gather = load_standalone_kernels(torch, triton)
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError(
            "A30 experiments require SM80 or newer; use the V100 presets on SM70"
        )
    min_prefix, min_saved = policy["min_prefix"], policy["min_saved"]
    depth = args.output_tokens - 1
    width = depth + args.prompt_len
    rows = args.beams * args.requests
    group_beams = max(
        1, min(args.beams, args.query_tile // (args.q_heads // args.kv_heads))
    )
    groups = args.requests * math.ceil(args.beams / group_beams)
    free, _ = torch.cuda.mem_get_info()
    budget = min(int(args.memory_gib * (1 << 30)), int(free * args.memory_fraction))
    estimate = a30_memory_estimate(
        args.beams,
        depth,
        args.requests,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.layers,
        args.query_tile,
        max(args.prefix_splits, policy.get("suffix_splits", 1), *args.splits),
        args.reference_rows,
        args.prompt_len,
    )
    if args.cache_mode == "scrub":
        estimate["estimated_peak_bytes"] += args.eviction_mib * (1 << 20)
    allocated_at_start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    header = {
        "kind": "a30_grouped_prefix",
        "parameters": vars(args),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "environment": gpu_environment(torch),
        "memory_budget_bytes": budget,
        "memory_estimate": estimate,
        "policy_status": "loaded_candidate"
        if args.policy
        else "uncalibrated_candidate",
        "policy": policy,
        "scope": "attention only; no model, top-k, VMM, or serving integration",
        "precision_contract": {
            "mha_qk": (
                "FP16/BF16 storage, FP32 multiplication/reduction/scaling "
                "in the experimental gather clone"
            ),
            "gqa_qk": "unchanged Tensor Core dot with FP32 accumulation",
            "gather": (
                "precision-matched stage-1 plus prefix-free LSE merge; "
                "not the unmodified serving baseline"
            ),
            "lse_atol": 1e-4,
            "lse_rtol": 1e-4,
            "output_atol": 0.01 if args.dtype == "bfloat16" else 0.002,
            "output_rtol": 0.01 if args.dtype == "bfloat16" else 0.002,
        },
        "trace_timing": (
            "sum of per-step replay medians with fixed prior-step snapshots; "
            "parent uploads and prior-path snapshots excluded from GPU intervals"
        ),
        "selection_note": (
            "thresholds are frozen before the run; comparisons report the best "
            "measured suffix split for diagnosis, not an online performance oracle"
        ),
        "prompt_semantics": (
            "prompt remains on the per-beam gather path; "
            "only beam-KV prefix sharing is changed"
        ),
        "reference_semantics": (
            "synthetic Q/K/V with parent-selection paths; not generated model tokens"
        ),
        "storage_semantics": (
            "snapshot cases place retained nodes compactly before timing; trace "
            "cases use append-only step slots; no online KV compaction is timed"
        ),
    }
    if estimate["estimated_peak_bytes"] > budget:
        return header | {
            "status": "resource_skipped",
            "reason": "estimated peak exceeds explicit budget",
        }
    build_start = time.perf_counter()
    traces = (
        parse_parent_trace(
            json.loads(Path(args.parent_trace).read_text()),
            args.requests,
            depth,
            args.beams,
        )
        if args.parent_trace
        else [
            parent_trace(args.beams, depth, args.pattern, args.seed + r)
            for r in range(args.requests)
        ]
    )
    snapshots = [
        trace_snapshot(trace, args.beams, depth, compact=not args.trace)
        for trace in traces
    ]
    spans = [args.beams * depth if args.trace else size for _, _, size in snapshots]
    bases, cursor = [], 0
    flat_paths, order_cpu = array("i"), array("i")
    for request, (slots, order, _) in enumerate(snapshots):
        bases.append(cursor)
        for row in range(args.beams):
            flat_paths.extend(range(cursor, cursor + args.prompt_len))
            flat_paths.extend(
                cursor + args.prompt_len + slot
                for slot in slots[row * depth : (row + 1) * depth]
            )
        order_cpu.extend(request * args.beams + row for row in order)
        cursor += args.prompt_len + spans[request]
    cpu_build_ms = (time.perf_counter() - build_start) * 1000
    paths = torch.tensor(flat_paths, dtype=torch.int32, device="cuda").reshape(
        rows, width
    )
    original_paths = paths.clone()
    previous = paths.clone()
    parents = torch.empty(rows, dtype=torch.int64, device="cuda")
    new_slots = torch.empty(rows, dtype=torch.int32, device="cuda")
    ranks = torch.arange(args.beams, device="cuda", dtype=torch.int64).repeat(
        args.requests, 1
    )
    initial_ranks = ranks.clone()
    tree_order = torch.tensor(order_cpu, dtype=torch.int64, device="cuda")
    offsets = (
        torch.arange(args.requests, device="cuda", dtype=torch.int64)[:, None]
        * args.beams
    )
    rank_values = torch.arange(args.beams, device="cuda", dtype=torch.int64).expand(
        args.requests, -1
    )
    last_parents = torch.tensor(
        [p for trace in traces for p in trace[-1]], dtype=torch.int64, device="cuda"
    )
    # Recover the preceding tree ranks for a genuine last-step metadata replay.
    for request, trace in enumerate(traces):
        _, prior_order, _ = trace_snapshot(trace, args.beams, depth - 1)
        ranks[request, torch.tensor(prior_order, dtype=torch.int64, device="cuda")] = (
            torch.arange(args.beams, device="cuda")
        )
    initial_ranks.copy_(ranks)
    sequence = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    prefix = torch.empty(groups, dtype=torch.int32, device="cuda")
    row_prefix = torch.empty(rows, dtype=torch.int32, device="cuda")
    row_group = torch.empty(rows, dtype=torch.int32, device="cuda")
    lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
    indptr = torch.empty(rows + 1, dtype=torch.int32, device="cuda")
    suffix_indices = torch.empty(rows * width, dtype=torch.int32, device="cuda")
    baseline_indices = torch.empty_like(suffix_indices)
    baseline_indptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda") * width
    active = torch.ones(rows, dtype=torch.bool, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    saved_initial_ranks = initial_ranks.clone()
    saved_last_parents = last_parents.clone()
    layer_data = []
    for _ in range(args.layers):
        q = torch.randn(
            (rows, args.q_heads, args.head_dim),
            generator=generator,
            dtype=dtype,
            device="cuda",
        )
        k = torch.randn(
            (cursor, args.kv_heads, args.head_dim),
            generator=generator,
            dtype=dtype,
            device="cuda",
        )
        v = torch.randn(k.shape, generator=generator, dtype=dtype, device="cuda")
        layer_data.append(
            (
                q,
                k,
                v,
                torch.empty_like(q),
                torch.empty(q.shape[:2], dtype=torch.float32, device="cuda"),
            )
        )
    pref_out = torch.empty(
        (args.prefix_splits, rows, args.q_heads, args.head_dim),
        dtype=torch.float32,
        device="cuda",
    )
    pref_lse = torch.empty(pref_out.shape[:-1], dtype=torch.float32, device="cuda")
    max_splits = max(policy.get("suffix_splits", 1), *args.splits)
    suffix_workspace = torch.empty(
        (rows * args.q_heads * max_splits * args.head_dim,),
        dtype=torch.float32,
        device="cuda",
    )
    suffix_lse_workspace = torch.empty(
        (rows * args.q_heads * max_splits,), dtype=torch.float32, device="cuda"
    )
    split_tensor = torch.empty(rows, dtype=torch.int32, device="cuda")
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    def order_update(source_ranks, parent_indices):
        parent_ranks = torch.gather(
            source_ranks, 1, parent_indices.reshape(args.requests, args.beams)
        )
        result = torch.argsort(parent_ranks, dim=1, stable=True)
        tree_order.copy_((result + offsets).flatten())
        return result

    def metadata(mode):
        if mode == "gather" or args.beams == 1:
            if not args.trace:
                baseline_indices.copy_(paths.flatten())
            else:
                row_prefix.zero_()
                baseline_indptr[0].zero_()
                torch.cumsum(
                    sequence, 0, dtype=baseline_indptr.dtype, out=baseline_indptr[1:]
                )
                shared._suffix_indices[(rows,)](
                    paths,
                    row_prefix,
                    baseline_indptr,
                    baseline_indices,
                    PATH_WIDTH=width,
                    BLOCK_T=triton.next_power_of_2(width),
                    PROMPT_LEN=args.prompt_len,
                )
            return
        result_order = order_update(initial_ranks, last_parents)
        ranks.scatter_(1, result_order, rank_values)
        shared.group_prefix_metadata(
            paths,
            tree_order,
            sequence,
            prefix,
            row_prefix,
            row_group,
            indptr,
            suffix_indices,
            lengths,
            beams=args.beams,
            group_beams=group_beams,
            min_prefix=min_prefix if mode == "auto" else 0,
            min_saved=min_saved if mode == "auto" else 0,
            prompt_len=args.prompt_len,
        )

    def execute(mode, split, data):
        q, k, v, out, lse = data
        part = suffix_workspace[: rows * args.q_heads * split * args.head_dim].view(
            rows, args.q_heads, split, args.head_dim
        )
        part_lse = suffix_lse_workspace[: rows * args.q_heads * split].view(
            rows, args.q_heads, split
        )
        stage = (
            gather._decode_att_m_fwd
            if args.q_heads == args.kv_heads
            else gather._decode_grouped_att_m_fwd
        )
        if mode == "gather" or args.beams == 1:
            stage(
                q,
                k,
                v,
                part,
                part_lse,
                baseline_indptr,
                baseline_indices,
                split_tensor,
                split,
                1 / math.sqrt(args.head_dim),
                0.0,
            )
            shared.merge_prefix_suffix(
                None, None, part, part_lse, baseline_indptr, active, out, lse
            )
            return
        shared.shared_prefix_attention(
            q,
            k,
            v,
            paths,
            tree_order,
            prefix,
            active,
            pref_out,
            pref_lse,
            beams=args.beams,
            group_beams=group_beams,
            tile=args.tile,
            splits=args.prefix_splits,
            prompt_len=args.prompt_len,
            num_warps=args.warps,
            num_stages=args.stages,
        )
        stage(
            q,
            k,
            v,
            part,
            part_lse,
            indptr,
            suffix_indices,
            split_tensor,
            split,
            1 / math.sqrt(args.head_dim),
            0.0,
        )
        shared.merge_prefix_suffix(
            pref_out, pref_lse, part, part_lse, indptr, active, out, lse
        )

    scrub = (
        torch.zeros(args.eviction_mib * (1 << 20), dtype=torch.uint8, device="cuda")
        if args.cache_mode == "scrub"
        else None
    )
    modes = ("gather",) if args.beams == 1 else tuple(args.modes)
    tests, timings, diagnostics, group_reports = [], [], [], []
    original_reports = []
    ref_out = torch.empty_like(layer_data[0][0], dtype=torch.float64)
    ref_lse = torch.empty(
        layer_data[0][0].shape[:2], dtype=torch.float64, device="cuda"
    )
    tolerance = 0.01 if dtype == torch.bfloat16 else 0.002
    if original_gather is not None:
        metadata("gather")
        for split in args.splits:
            split_tensor.fill_(split)
            part = suffix_workspace[: rows * args.q_heads * split * args.head_dim].view(
                rows, args.q_heads, split, args.head_dim
            )
            part_lse = suffix_lse_workspace[: rows * args.q_heads * split].view(
                rows, args.q_heads, split
            )
            records = []

            def original_call(data):
                q, k, v, out, _ = data
                original_gather.decode_attention_fwd(
                    q,
                    k,
                    v,
                    out,
                    baseline_indptr,
                    baseline_indices,
                    part,
                    part_lse,
                    split_tensor,
                    split,
                    1 / math.sqrt(args.head_dim),
                    1.0,
                    1.0,
                    enable_lean=False,
                )

            for layer, data in enumerate(layer_data):
                original_call(data)
                a30_reference(
                    torch,
                    *data[:3],
                    paths,
                    width,
                    ref_out,
                    ref_lse,
                    args.reference_rows,
                )
                output_error = (data[3].double() - ref_out).abs().max().item()
                output_close = torch.allclose(
                    data[3].double(), ref_out, atol=tolerance, rtol=tolerance
                )
                shared.merge_prefix_suffix(
                    None,
                    None,
                    part,
                    part_lse,
                    baseline_indptr,
                    active,
                    data[3],
                    data[4],
                )
                records.append(
                    {
                        "layer": layer,
                        "max_abs_error": output_error,
                        "output_matches_tolerance": output_close,
                        "max_lse_abs_error": (data[4].double() - ref_lse)
                        .abs()
                        .max()
                        .item(),
                        "lse_matches_strict_tolerance": torch.allclose(
                            data[4].double(), ref_lse, atol=1e-4, rtol=1e-4
                        ),
                    }
                )

            def original_compute():
                for data in layer_data:
                    original_call(data)

            original_reports.append(
                {
                    "mode": "gather_original_diagnostic",
                    "split": split,
                    "precision_certified": False,
                    "correctness": records,
                    "scope": (
                        "unmodified gather attention only; "
                        "LSE extraction excluded from timing"
                    ),
                    "timing": measure_cuda(
                        torch,
                        original_compute,
                        args.warmup,
                        args.repeats,
                        scrub,
                        args.execution,
                        args.graph_batch,
                    ),
                    "loaded_gather_sha256": original_gather.loaded_source_sha256,
                }
            )
    for mode in modes:
        metadata(mode)
        if mode != "gather":
            expected = []
            for slots, order, _ in snapshots:
                expected.extend(
                    group_prefix_oracle(
                        slots,
                        order,
                        args.beams,
                        depth,
                        group_beams,
                        min_prefix if mode == "auto" else 0,
                        min_saved if mode == "auto" else 0,
                    )
                )
            torch.testing.assert_close(
                prefix.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
            )
            selected_prefixes = prefix.cpu().tolist()
            if (
                mode == "shared"
                and args.require_shared_prefix
                and not any(selected_prefixes)
            ):
                raise AssertionError("smoke case must execute a nonempty shared prefix")
            sizes = [
                min(group_beams, args.beams - start)
                for _ in range(args.requests)
                for start in range(0, args.beams, group_beams)
            ]
            group_reports.append(
                {
                    "mode": mode,
                    "prefix_lengths": selected_prefixes,
                    "selected_groups": sum(c > 0 for c in selected_prefixes),
                    "logical_beam_kv_references": sum(
                        c + size * (depth - c)
                        for size, c in zip(sizes, selected_prefixes)
                    ),
                    "baseline_beam_kv_references": rows * depth,
                }
            )
        mode_splits = (
            [policy["suffix_splits"]]
            if mode == "auto" and "suffix_splits" in policy
            else args.splits
        )
        for split in mode_splits:
            split_tensor.fill_(split)
            for layer, data in enumerate(layer_data):
                execute(mode, split, data)
                a30_reference(
                    torch,
                    *data[:3],
                    paths,
                    width,
                    ref_out,
                    ref_lse,
                    args.reference_rows,
                )
                torch.testing.assert_close(
                    data[3].double(), ref_out, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    data[4].double(), ref_lse, atol=1e-4, rtol=1e-4
                )
                partial_errors = None
                if args.check_partial_lse:
                    partial_errors = check_a30_partial_lse(
                        torch,
                        data[0],
                        data[1],
                        baseline_indptr if mode == "gather" else indptr,
                        baseline_indices if mode == "gather" else suffix_indices,
                        suffix_lse_workspace[: rows * args.q_heads * split].view(
                            rows, args.q_heads, split
                        ),
                        None if mode == "gather" else pref_lse,
                        paths,
                        row_prefix,
                        args.prompt_len,
                        args.tile,
                        args.reference_rows,
                    )
                if mode != "gather" and args.beams <= 12 and args.layers == 1:
                    active[-1] = False
                    execute(mode, split, data)
                    torch.testing.assert_close(
                        data[3][-1], torch.zeros_like(data[3][-1]), rtol=0, atol=0
                    )
                    if not torch.isneginf(data[4][-1]).all().item():
                        raise AssertionError(
                            "inactive query must have negative infinite LSE"
                        )
                    active.fill_(True)
                    prefix.fill_(depth)
                    row_prefix.fill_(depth)
                    lengths.fill_(args.prompt_len)
                    indptr.copy_(
                        torch.arange(rows + 1, device="cuda", dtype=torch.int32)
                        * args.prompt_len
                    )
                    shared._suffix_indices[(rows,)](
                        paths,
                        row_prefix,
                        indptr,
                        suffix_indices,
                        PATH_WIDTH=width,
                        BLOCK_T=triton.next_power_of_2(width),
                        PROMPT_LEN=args.prompt_len,
                    )
                    # Duplicate histories provide a valid fully shared-prefix boundary.
                    saved_paths = paths.clone()
                    for start in range(0, rows, args.beams):
                        for local in range(0, args.beams, group_beams):
                            members = tree_order[
                                start + local : start
                                + min(local + group_beams, args.beams)
                            ]
                            paths[members] = paths[members[0]].clone()
                    execute(mode, split, data)
                    a30_reference(
                        torch,
                        *data[:3],
                        paths,
                        width,
                        ref_out,
                        ref_lse,
                        args.reference_rows,
                    )
                    torch.testing.assert_close(
                        data[3].double(), ref_out, atol=tolerance, rtol=tolerance
                    )
                    torch.testing.assert_close(
                        data[4].double(), ref_lse, atol=1e-4, rtol=1e-4
                    )
                    paths.copy_(saved_paths)
                    metadata(mode)
                    execute(mode, split, data)
                    a30_reference(
                        torch,
                        *data[:3],
                        paths,
                        width,
                        ref_out,
                        ref_lse,
                        args.reference_rows,
                    )
                tests.append(
                    {
                        "mode": mode,
                        "split": split,
                        "layer": layer,
                        "max_abs_error": (data[3].double() - ref_out)
                        .abs()
                        .max()
                        .item(),
                        "max_lse_abs_error": (data[4].double() - ref_lse)
                        .abs()
                        .max()
                        .item(),
                        "precision": "matched_fp32_qk",
                        "partial_lse_max_abs_error": partial_errors,
                    }
                )

            def compute(mode=mode, split=split):
                for data in layer_data:
                    execute(mode, split, data)

            def total(mode=mode, compute=compute):
                metadata(mode)
                compute()

            functions = (
                shared._shared_prefix_attention,
                shared._merge_prefix_suffix,
                gather._fwd_kernel_stage1,
                gather._fwd_grouped_kernel_stage1,
                gather._fwd_kernel_stage2,
            )
            diagnostics.append(
                {
                    "mode": mode,
                    "split": split,
                    "kernels": inspect_compilation(
                        lambda: execute(mode, split, layer_data[0]), functions, dump_dir
                    ),
                }
            )
            operations = [
                ("metadata", lambda mode=mode: metadata(mode)),
                ("attention_all_layers", compute),
                ("total", total),
            ]
            if args.layers > 1:
                operations.append(
                    (
                        "attention_first_layer",
                        lambda mode=mode, split=split: execute(
                            mode, split, layer_data[0]
                        ),
                    )
                )
            for scope, call in operations:
                timings.append(
                    {
                        "mode": mode,
                        "split": split,
                        "scope": scope,
                        "layers": args.layers,
                        **measure_cuda(
                            torch,
                            call,
                            args.warmup,
                            args.repeats,
                            scrub,
                            args.execution,
                            args.graph_batch,
                        ),
                    }
                )
            if args.trace:
                paths.zero_()
                for request, base in enumerate(bases):
                    paths[
                        request * args.beams : (request + 1) * args.beams,
                        : args.prompt_len,
                    ] = torch.arange(
                        base, base + args.prompt_len, device="cuda", dtype=torch.int32
                    )
                ranks.copy_(torch.arange(args.beams, device="cuda").expand_as(ranks))
                step_records = []
                for step in range(depth):
                    previous.copy_(paths)
                    parents.copy_(
                        torch.tensor(
                            [p for trace in traces for p in trace[step]],
                            dtype=torch.int64,
                            device="cuda",
                        )
                    )
                    new_slots.copy_(
                        torch.tensor(
                            [
                                base + args.prompt_len + step * args.beams + row
                                for base in bases
                                for row in range(args.beams)
                            ],
                            dtype=torch.int32,
                            device="cuda",
                        )
                    )
                    sequence.fill_(args.prompt_len + step + 1)
                    baseline_indptr.copy_(
                        torch.arange(rows + 1, dtype=torch.int32, device="cuda")
                        * (args.prompt_len + step + 1)
                    )
                    initial_ranks.copy_(ranks)
                    last_parents.copy_(parents)

                    def advance():
                        shared._reparent_beam_paths[(rows,)](
                            previous,
                            parents,
                            new_slots,
                            paths,
                            BEAMS=args.beams,
                            PATH_WIDTH=width,
                            LENGTH=args.prompt_len + step + 1,
                            BLOCK_T=triton.next_power_of_2(width),
                        )
                        metadata(mode)

                    advance()
                    oracle_paths = array("i")
                    oracle_order = array("i")
                    for request, (trace, base) in enumerate(zip(traces, bases)):
                        step_paths, step_order, _ = trace_snapshot(
                            trace, args.beams, step + 1, compact=False
                        )
                        for row in range(args.beams):
                            oracle_paths.extend(range(base, base + args.prompt_len))
                            oracle_paths.extend(
                                base + args.prompt_len + slot
                                for slot in step_paths[
                                    row * (step + 1) : (row + 1) * (step + 1)
                                ]
                            )
                        oracle_order.extend(
                            request * args.beams + row for row in step_order
                        )
                    torch.testing.assert_close(
                        paths[:, : args.prompt_len + step + 1].cpu(),
                        torch.tensor(oracle_paths, dtype=torch.int32).reshape(rows, -1),
                        rtol=0,
                        atol=0,
                    )
                    if mode != "gather":
                        torch.testing.assert_close(
                            tree_order.cpu(),
                            torch.tensor(oracle_order, dtype=torch.int64),
                            rtol=0,
                            atol=0,
                        )
                    for data in layer_data:
                        execute(mode, split, data)
                        a30_reference(
                            torch,
                            *data[:3],
                            paths,
                            args.prompt_len + step + 1,
                            ref_out,
                            ref_lse,
                            args.reference_rows,
                        )
                        torch.testing.assert_close(
                            data[3].double(), ref_out, atol=tolerance, rtol=tolerance
                        )
                        torch.testing.assert_close(
                            data[4].double(), ref_lse, atol=1e-4, rtol=1e-4
                        )

                    def step_total():
                        advance()
                        compute()

                    step_records.append(
                        {
                            "kv_len": step + 1,
                            **measure_cuda(
                                torch,
                                step_total,
                                args.warmup,
                                args.repeats,
                                scrub,
                                args.execution,
                                args.graph_batch,
                            ),
                        }
                    )
                timings.append(
                    {
                        "mode": mode,
                        "split": split,
                        "scope": "trace_sum",
                        "steps": step_records,
                        "median_us_sum": sum(s["median_us"] for s in step_records),
                    }
                )
                paths.copy_(original_paths)
                initial_ranks.copy_(saved_initial_ranks)
                last_parents.copy_(saved_last_parents)
                sequence.fill_(width)
                baseline_indptr.copy_(
                    torch.arange(rows + 1, dtype=torch.int32, device="cuda") * width
                )
                metadata(mode)
                tests.append(
                    {
                        "mode": mode,
                        "split": split,
                        "trace_steps_checked": depth,
                        "path_checked": True,
                        "tree_order_checked": mode != "gather",
                    }
                )
    summary = []
    for mode in modes:
        selected = min(
            (r for r in timings if r["mode"] == mode and r["scope"] == "total"),
            key=lambda r: r["median_us"],
        )
        baseline = min(
            (r for r in timings if r["mode"] == "gather" and r["scope"] == "total"),
            key=lambda r: r["median_us"],
        )
        summary.append(
            {
                "mode": mode,
                "best_split": selected["split"],
                "total_median_us": selected["median_us"],
                "speedup_vs_best_gather": baseline["median_us"] / selected["median_us"],
            }
        )
    return header | {
        "status": "completed",
        "torch_peak_allocated_delta_bytes": torch.cuda.max_memory_allocated()
        - allocated_at_start,
        "correctness": tests,
        "timings": timings,
        "comparisons": summary,
        "compiled_kernels": diagnostics,
        "original_gather_diagnostics": original_reports,
        "grouping": group_reports,
        "cpu_fixture_build_ms": cpu_build_ms,
        "live_beam_nodes": [s[2] for s in snapshots],
        "global_common_prefix": [
            depth
            if args.beams == 1
            else group_prefix_oracle(slots, order, args.beams, depth, args.beams)[0]
            for slots, order, _ in snapshots
        ],
        "ancestor_counts_by_depth": [
            [len(set(slots[position::depth])) for position in range(depth)]
            for slots, _, _ in snapshots
        ],
        "kv_len": depth,
        "kv_storage_slots": cursor,
        "group_beams": group_beams,
        "actual_kv_bytes": sum(
            (k.numel() + v.numel()) * k.element_size() for _, k, v, _, _ in layer_data
        ),
        "source_hashes": {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (("shared", shared), ("gather", gather))
        },
        "loaded_gather_sha256": gather.loaded_source_sha256,
        "parent_trace_sha256": hashlib.sha256(
            Path(args.parent_trace).read_bytes()
        ).hexdigest()
        if args.parent_trace
        else None,
        "auto_note": (
            "zero-prefix groups use full suffix gather but still pay "
            "metadata/merge launches; k=1 bypasses them"
        ),
        "hardware_counters": (
            "not collected; compiler resources are not hardware bandwidth counters"
        ),
    }


class TestA30Grouping(unittest.TestCase):
    def test_qk_promotion_changes_only_the_mha_expression(self):
        source = (
            "def _fwd_kernel_stage1(q, k):\n"
            "    qk = tl.sum(q[None, :] * k, 1)\n"
            "    qk *= sm_scale_withk\n"
            "def _fwd_grouped_kernel_stage1(q, k):\n"
            "    qk = tl.dot(q, k)\n"
        )
        lines = source.splitlines(keepends=True)
        promote_mha_qk(lines, ast.parse(source).body)
        self.assertEqual(len(lines), len(source.splitlines()))
        self.assertEqual(lines[0], source.splitlines(keepends=True)[0])
        self.assertEqual(lines[2:], source.splitlines(keepends=True)[2:])
        expression = ast.parse("".join(lines)).body[0].body[0].value
        self.assertEqual(
            ast.unparse(expression.args[0]),
            "q.to(tl.float32)[None, :] * k.to(tl.float32)",
        )
        self.assertEqual(ast.unparse(expression.keywords[0].value), "tl.float32")
        changed = source.replace("q[None, :] * k", "q[None, :] + k")
        with self.assertRaises(RuntimeError):
            promote_mha_qk(changed.splitlines(keepends=True), ast.parse(changed).body)

    def test_smoke_has_zero_and_positive_prefix_regressions(self):
        cases = dict(a30_cases("a30-smoke"))
        self.assertEqual(len(cases), 5)
        for name, expect_positive in (("mha_tail", False), ("mha_prefix", True)):
            c = cases[name]
            depth = c["output_tokens"] - 1
            paths, order, _ = trace_snapshot(
                parent_trace(c["beams"], depth, c["pattern"], 43), c["beams"], depth
            )
            common = group_prefix_oracle(paths, order, c["beams"], depth, c["beams"])
            self.assertEqual(any(common), expect_positive)
        self.assertTrue(cases["mha_prefix"]["require_shared_prefix"])

    def test_prefix_free_lse_merge_matches_full_path(self):
        rng = random.Random(7)
        q = [rng.uniform(-1, 1) for _ in range(8)]
        keys = [[rng.uniform(-1, 1) for _ in q] for _ in range(67)]
        values = [[rng.uniform(-1, 1) for _ in q] for _ in keys]
        for length, splits in itertools.product((0, 1, 15, 33, 67), (1, 2, 4, 8)):
            span = math.ceil(math.ceil(length / splits) / 32) * 32
            merged = ([0.0] * len(q), -math.inf)
            for split in range(splits):
                path = list(range(split * span, min((split + 1) * span, length)))
                merged = merge_states(
                    merged, reference_attention(q, keys, values, path)
                )
            expected = reference_attention(q, keys, values, list(range(length)))
            for a, b in zip(merged[0], expected[0]):
                self.assertAlmostEqual(a, b, places=12)
            if length:
                self.assertAlmostEqual(merged[1], expected[1], places=12)
            else:
                self.assertEqual(merged[1], -math.inf)

    def test_parent_trace_input_validation(self):
        valid = {"parents": [[[0, 1], [1, 1]]]}
        actual = parse_parent_trace(valid, 1, 2, 2)
        self.assertEqual([list(step) for step in actual[0]], valid["parents"][0])
        for invalid in (
            [],
            {},
            {"parents": [None]},
            {"parents": [[None, [0, 1]]]},
            {"parents": [[[0, 2], [0, 0]]]},
            {"parents": [[[True, 0], [0, 0]]]},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_parent_trace(invalid, 1, 2, 2)

    def test_budget_skip_occurs_before_gpu_data_allocations(self):
        parser = argparse.ArgumentParser()
        add_a30_arguments(parser)
        args = parser.parse_args(
            ["--beams", "4096", "--output-tokens", "128", "--memory-gib", "0.01"]
        )
        torch = MagicMock()
        torch.__version__ = "mock"
        torch.cuda.get_device_capability.return_value = (8, 0)
        torch.cuda.mem_get_info.return_value = (24 << 30, 24 << 30)
        torch.cuda.memory_allocated.return_value = 0
        triton = SimpleNamespace(__version__="mock")
        with patch.dict(
            run_a30.__globals__,
            {
                "load_cuda": Mock(return_value=(torch, triton, None, None)),
                "gpu_environment": Mock(return_value={}),
            },
        ):
            result = run_a30(args)
        self.assertEqual(result["status"], "resource_skipped")
        torch.randn.assert_not_called()
        torch.empty.assert_not_called()
        torch.tensor.assert_not_called()

    def test_policy_validation_and_fixed_matrix_policy(self):
        args = SimpleNamespace(policy=None, min_prefix=16, min_saved=128)
        self.assertEqual(a30_candidate_policy(args)["min_saved"], 128)
        args.policy = "policy.json"
        policy = {
            "schema": "beam-prefix-policy-v1",
            "min_prefix": 8,
            "min_saved": 64,
            "query_tile": 32,
            "suffix_splits": 2,
        }
        with patch.object(Path, "read_text", return_value=json.dumps(policy)):
            self.assertEqual(a30_candidate_policy(args), policy)
        for invalid in (
            [],
            policy | {"min_saved": True},
            policy | {"query_tile": 7},
            policy | {"unknown": 1},
        ):
            with (
                patch.object(Path, "read_text", return_value=json.dumps(invalid)),
                self.assertRaises(ValueError),
            ):
                a30_candidate_policy(args)
        matrix = SimpleNamespace(
            preset="a30-main",
            case=None,
            output_dir="/tmp/a30-dry",
            python=None,
            graph_batch=64,
            repeats=2,
            warmup=1,
            policy="policy.json",
        )
        self.assertTrue(
            all("--policy" in c["command"] for c in matrix_commands(matrix))
        )

    def test_prompt_and_suffix_indices_do_not_duplicate_shared_nodes(self):
        path = [100, 101, 10, 11, 12, 13]
        for common in range(5):
            suffix_len = len(path) - common
            packed = [path[i if i < 2 else i + common] for i in range(suffix_len)]
            shared = path[2 : 2 + common]
            self.assertEqual(sorted(packed + shared), sorted(path))
            self.assertEqual(packed[:2], path[:2])

    def test_raw_trace_snapshot_matches_forward_reparent(self):
        for beams, pattern in itertools.product(
            (1, 3, 6, 24), ("shared", "grouped", "independent", "random")
        ):
            trace = parent_trace(beams, 17, pattern, 11)
            histories = [[] for _ in range(beams)]
            for step, parents in enumerate(trace):
                histories = [
                    histories[parent] + [step * beams + row]
                    for row, parent in enumerate(parents)
                ]
                paths, _, _ = trace_snapshot(trace, beams, step + 1, compact=False)
                self.assertEqual(list(paths), [s for path in histories for s in path])

    def test_tree_order_prefix_matches_all_members(self):
        for beams, depth, pattern in itertools.product(
            (1, 3, 6, 24), (1, 3, 17), ("shared", "grouped", "independent", "random")
        ):
            trace = parent_trace(beams, depth, pattern, 7)
            paths, order, _ = trace_snapshot(trace, beams, depth)
            self.assertEqual(sorted(order), list(range(beams)))
            for size in (1, 2, 4, 8):
                actual = group_prefix_oracle(paths, order, beams, depth, size)
                for group, start in enumerate(range(0, beams, size)):
                    members = order[start : start + size]
                    expected = 0
                    if len(members) > 1:
                        while (
                            expected < depth
                            and len({paths[r * depth + expected] for r in members}) == 1
                        ):
                            expected += 1
                    self.assertEqual(actual[group], expected)

    def test_split_attention_uses_only_path_nodes(self):
        trace = parent_trace(6, 9, "grouped", 3)
        paths, order, size = trace_snapshot(trace, 6, 9)
        rng = random.Random(3)
        keys = [[rng.random() for _ in range(4)] for _ in range(size)]
        values = [[rng.random() for _ in range(4)] for _ in range(size)]
        q = [0.1, 0.2, -0.3, 0.4]
        common = group_prefix_oracle(paths, order, 6, 9, 2)
        for group, start in enumerate(range(0, 6, 2)):
            for row in order[start : start + 2]:
                path = paths[row * 9 : (row + 1) * 9]
                c = common[group]
                actual = merge_states(
                    reference_attention(q, keys, values, path[:c]),
                    reference_attention(q, keys, values, path[c:]),
                )
                expected = reference_attention(q, keys, values, path)
                for a, b in zip(actual[0], expected[0]):
                    self.assertAlmostEqual(a, b, places=12)
                self.assertAlmostEqual(actual[1], expected[1], places=12)

    def test_original_matrix_is_preserved(self):
        cases = a30_cases("a30-main") + a30_cases("a30-stress")
        pairs = {(c["beams"], c["output_tokens"]) for _, c in cases}
        self.assertEqual(len(pairs), 80)
        for k, n in itertools.product(
            (1, 3, 6, 12, 24, 128, 256, 512, 1024, 2048, 4096), A30_OUTPUTS
        ):
            self.assertIn((k, n), pairs)
        self.assertEqual(len(cases), 480)

    def test_large_case_has_no_dense_mask(self):
        cost = a30_memory_estimate(4096, 128, 1, 8, 8, 128, 1, 16, 4, 1)
        self.assertEqual(cost["kv_upper_bound_bytes"], 2 << 30)
        self.assertEqual(cost["dense_ancestry_mask_bytes"], 0)
        self.assertLess(cost["estimated_peak_bytes"], 4 << 30)

    def test_selection_thresholds_and_zero_prefix(self):
        trace = parent_trace(6, 9, "independent", 2)
        paths, order, _ = trace_snapshot(trace, 6, 9)
        self.assertEqual(group_prefix_oracle(paths, order, 6, 9, 4), [0, 0])
        trace = parent_trace(6, 9, "shared", 2)
        paths, order, _ = trace_snapshot(trace, 6, 9)
        self.assertEqual(
            group_prefix_oracle(paths, order, 6, 9, 4, min_prefix=16), [0, 0]
        )


def add_a30_arguments(parser):
    parser.add_argument("--beams", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument(
        "--pattern",
        choices=("shared", "grouped", "independent", "random"),
        default="shared",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--query-tile", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--tile", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--prefix-splits", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--stages", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--splits", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2, 4]
    )
    parser.add_argument("--min-prefix", type=int, default=16)
    parser.add_argument("--min-saved", type=int, default=128)
    parser.add_argument(
        "--policy", help="frozen candidate policy JSON, not a per-case oracle"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("gather", "shared", "auto"),
        default=["gather", "shared", "auto"],
    )
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--reference-rows", type=int, default=1)
    parser.add_argument(
        "--original-diagnostic",
        action="store_true",
        help="measure original gather separately; its LSE is not precision-certified",
    )
    parser.add_argument(
        "--require-shared-prefix",
        action="store_true",
        help="require a nonempty shared prefix in forced-shared mode",
    )
    parser.add_argument(
        "--check-partial-lse",
        action="store_true",
        help="validate stage-1 LSE before merging; intended for smoke cases",
    )
    parser.add_argument("--prompt-len", type=int, default=0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--parent-trace", help="JSON parents[request][step][beam] from a real model"
    )
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--graph-batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cache-mode", choices=("warm", "scrub"), default="warm")
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--output")
    parser.add_argument("--dump-dir")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser(
        "self-test", help="run standard-library CPU correctness tests"
    )
    sim = subparsers.add_parser(
        "simulate", help="report theoretical counts, not GPU measurements"
    )
    add_tree_arguments(sim)
    sim.add_argument("--group-beams", type=int, default=8)
    sim.add_argument("--tile", type=int, default=64)
    sim.add_argument("--head-dim", type=int, default=128)
    sim.add_argument("--element-bytes", type=int, default=2)
    sim.add_argument("--history", action="store_true")
    add_gpu_arguments(
        subparsers.add_parser(
            "gpu-check", help="check CUDA outputs; fail if CUDA dependencies are absent"
        )
    )
    add_gpu_arguments(
        subparsers.add_parser(
            "benchmark", help="compare existing gather and shared-tile CUDA kernels"
        )
    )
    add_a30_arguments(
        subparsers.add_parser(
            "a30-case", help="grouped-prefix attention with budgeted storage"
        )
    )
    matrix = subparsers.add_parser(
        "matrix", help="run staged GPU experiments in isolated processes"
    )
    matrix.add_argument(
        "--preset",
        choices=(
            "quick",
            "full",
            "v100-tune",
            "compiler",
            "a30-smoke",
            "a30-calibrate",
            "a30-main",
            "a30-stress",
            "a30-trace",
        ),
        default="quick",
    )
    matrix.add_argument(
        "--python",
        help="Python executable in an existing environment; installs nothing",
    )
    matrix.add_argument(
        "--output-dir",
        required=True,
        help="fresh directory for JSON, logs, and assembly",
    )
    matrix.add_argument(
        "--case", nargs="+", help="run only named cases from the preset"
    )
    matrix.add_argument(
        "--dry-run", action="store_true", help="print commands without importing CUDA"
    )
    matrix.add_argument(
        "--policy", help="frozen A30 policy shared by every matrix case"
    )
    matrix.add_argument("--graph-batch", type=int, default=64)
    matrix.add_argument("--warmup", type=int, default=10)
    matrix.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args(
        argv if argv is not None else sys.argv[1:] or ["self-test"]
    )
    if args.mode == "self-test":
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
            for test_case in (TestBeamKVCPU, TestBenchmarkHarness, TestA30Grouping)
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    try:
        if args.mode in ("gpu-check", "benchmark", "a30-case"):
            if args.output and Path(args.output).exists():
                raise ValueError("output file already exists; select a new path")
            result = run_a30(args) if args.mode == "a30-case" else run_gpu(args)
        elif args.mode == "matrix":
            result = run_matrix(args)
        else:
            snapshot = snapshot_from_args(args)
            result = traffic_model(
                snapshot, args.group_beams, args.tile, args.head_dim, args.element_bytes
            )
            result["allocator"] = args.allocator
            result["pattern"] = args.pattern
            result["final_state"] = snapshot.trace[-1] if snapshot.trace else None
            if args.history:
                result["history"] = snapshot.trace
        output = json.dumps(result, indent=2, allow_nan=False)
        if args.mode in ("gpu-check", "benchmark", "a30-case") and args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as stream:
                stream.write(output + "\n")
        else:
            print(output)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
