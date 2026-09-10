"""Standalone beam-KV experiments; CPU modes require only the standard library."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import statistics
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


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


def load_cuda():
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    try:
        from sglang.kernels.ops.attention.beam_decode_attention import (
            beam_decode_attention_fwd,
        )
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    except ImportError as error:
        raise RuntimeError(
            "GPU modes require this checkout's SGLang runtime dependencies"
        ) from error
    return torch, triton, beam_decode_attention_fwd, decode_attention_fwd


def prepare_gpu_case(args, torch):
    if (
        min(
            args.requests,
            args.q_heads,
            args.kv_heads,
            args.scatter_factor,
            args.scan_spacing,
            args.splits,
        )
        < 1
    ):
        raise ValueError(
            "requests, heads, spacing, scatter factor, and splits must be positive"
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


def shared_runner(args, case, shared_fwd, group, tile):
    dense = case["dense"]

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
            group_beams=group,
            tile_tokens=tile,
        )

    def total():
        rebuild_mask(case)
        run()

    return run, total


def gather_runner(args, case, layout, torch, gather_fwd):
    q = case["q"]
    rows, heads, dim = q.shape
    indptr = torch.arange(rows + 1, dtype=torch.int32, device=q.device) * args.steps
    indices = layout["paths"].flatten().clone()
    splits = torch.full((rows,), args.splits, dtype=torch.int32, device=q.device)
    partial = torch.empty(
        (rows, heads, args.splits, dim), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty(
        (rows, heads, args.splits), dtype=torch.float32, device=q.device
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
            args.splits,
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
    for group in args.group_beams:
        for tile in args.tile:
            run, _ = shared_runner(args, case, shared_fwd, group, tile)
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
                    "group_beams": group,
                    "tile": tile,
                    "max_abs_error": (case["out"].double() - reference)
                    .abs()
                    .max()
                    .item(),
                }
            )
    run, _ = shared_runner(args, case, shared_fwd, args.group_beams[0], args.tile[0])
    case["query_valid"][-1] = False
    rebuild_mask(case)
    run()
    torch.testing.assert_close(
        case["out"][-1], torch.zeros_like(case["out"][-1]), atol=0, rtol=0
    )
    assert torch.isneginf(case["lse"][-1]).all().item()
    case["query_valid"].fill_(True)
    rebuild_mask(case)
    saved_ends = case["dense"]["ends"].clone()
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
        for end in (0, 1, None):
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
    }


def measure_cuda(torch, call, warmup, repeats, scrub):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    try:
        probe = torch.cuda.Event(enable_timing=True, external=True)
    except TypeError:
        probe = None

    if probe is None:
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
            call()
            end.record()
    else:
        events = [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(repeats)
        ]
        graph = torch.cuda.CUDAGraph()
        # Event nodes keep Python submission gaps outside the measured intervals.
        with torch.cuda.graph(graph):
            for start, end in events:
                if scrub is not None:
                    scrub.add_(1)
                start.record()
                call()
                end.record()
        graph.replay()
    torch.cuda.synchronize()
    samples = sorted(start.elapsed_time(end) * 1000.0 for start, end in events)
    return {
        "median_us": statistics.median(samples),
        "p10_us": samples[int((len(samples) - 1) * 0.1)],
        "p90_us": samples[int((len(samples) - 1) * 0.9)],
    }


def run_gpu(args):
    if min(args.warmup, args.repeats) < 1 or args.eviction_mib < 1:
        raise ValueError("warmup, repeats, and eviction_mib must be positive")
    if args.mode == "benchmark" and args.steps < 1:
        raise ValueError(
            "benchmark requires steps > 0; use gpu-check for empty history"
        )
    torch, triton, shared_fwd, gather_fwd = load_cuda()
    case = prepare_gpu_case(args, torch)
    baselines = (
        {
            name: gather_runner(args, case, case[layout], torch, gather_fwd)
            for name, layout in (
                ("A_scattered_gather", "scattered"),
                ("B_pool_gather", "dense"),
            )
        }
        if args.steps
        else {}
    )
    correctness = verify_gpu(args, case, torch, shared_fwd, baselines)
    result = {
        "kind": "measured_cuda" if args.mode == "benchmark" else "cuda_correctness",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "parameters": vars(args),
        "correctness": correctness,
        "source_allocator_state": case["source_trace"],
        "layout_note": (
            "offline slot permutation preserves holes; "
            "no runtime compaction or VMM is measured"
        ),
        "timing_note": (
            "CUDA Graph replay with external event nodes when supported; "
            "otherwise batched CUDA event timing; excludes scrub intervals"
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
            "gather_workspace_per_method": next(iter(baselines.values()))[
                "workspace_bytes"
            ]
            if baselines
            else 0,
            "gather_metadata_per_method": next(iter(baselines.values()))[
                "metadata_bytes"
            ]
            if baselines
            else 0,
        },
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
    for name, baseline in baselines.items():
        for scope in ("run", "metadata", "total"):
            timings.append(
                {
                    "method": name,
                    "scope": scope,
                    **measure_cuda(
                        torch, baseline[scope], args.warmup, args.repeats, scrub
                    ),
                }
            )
    for group in args.group_beams:
        for tile in args.tile:
            run, total = shared_runner(args, case, shared_fwd, group, tile)
            for scope, call in (
                ("run", run),
                ("metadata", lambda: rebuild_mask(case)),
                ("total", total),
            ):
                timings.append(
                    {
                        "method": "C_shared_scan",
                        "group_beams": group,
                        "tile": tile,
                        "scope": scope,
                        **measure_cuda(torch, call, args.warmup, args.repeats, scrub),
                    }
                )
    result["timings"] = timings
    result["analytical_scan_geometry"] = [
        traffic_model(snapshot, group, tile, args.head_dim)
        for snapshot in case["dense"]["snapshots"]
        for group in args.group_beams
        for tile in args.tile
    ]
    result["hardware_counters"] = (
        "not collected; use Nsight Compute for sectors, L2/DRAM traffic, and occupancy"
    )
    return result


def add_gpu_arguments(parser):
    add_tree_arguments(parser)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--group-beams", type=int, nargs="+", choices=(4, 8, 16), default=[4, 8, 16]
    )
    parser.add_argument(
        "--tile", type=int, nargs="+", choices=(32, 64, 128), default=[32, 64, 128]
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
        "--splits", type=int, default=4, help="existing decode kernel's KV split count"
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
    args = parser.parse_args(
        argv if argv is not None else sys.argv[1:] or ["self-test"]
    )
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBeamKVCPU)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    try:
        if args.mode in ("gpu-check", "benchmark"):
            result = run_gpu(args)
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
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
