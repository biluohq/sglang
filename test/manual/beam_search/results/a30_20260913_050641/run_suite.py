import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
ENTRY = REPO / "test/manual/beam_search/test_beam_kv_attention.py"
STATE = ROOT / "suite_status.json"
STAGES = ("smoke", "calibrate", "main", "trace", "stress")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(state):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)


def run(state, name, command):
    record = {"name": name, "command": command, "started_at": now()}
    state["steps"].append(record)
    state["current_step"] = name
    save(state)
    print(f"{now()} START {name}", flush=True)
    started = time.monotonic()
    with (ROOT / f"{name}.log").open("x") as stream:
        result = subprocess.run(
            command, cwd=REPO, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    record.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started,
                  finished_at=now())
    save(state)
    print(f"{now()} END {name}: rc={result.returncode}", flush=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {ROOT / (name + '.log')}")


def smoke_gate():
    directory = ROOT / "smoke"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"] and manifest["all_cases_validated"]
    assert manifest["resource_skipped"] == 0 and len(manifest["cases"]) == 4
    report = []
    for case in manifest["cases"]:
        data = json.loads(Path(case["output"]).read_text())
        assert case["returncode"] == 0 and data["status"] == "completed"
        assert data["correctness"]
        kernels = [
            kernel
            for diagnostic in data["compiled_kernels"]
            for kernel in diagnostic["kernels"]
            if kernel["name"] == "_shared_prefix_attention"
        ]
        if case["name"] != "single":
            assert kernels, f"No shared prefix kernels: {case['name']}"
            assert all(k["ptx_mma_instruction_count"] > 0 for k in kernels)
        report.append({
            "case": case["name"],
            "correctness_records": len(data["correctness"]),
            "prefix_kernel_count": len(kernels),
            "mma_counts": [k["ptx_mma_instruction_count"] for k in kernels],
        })
    (ROOT / "smoke_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    if STATE.exists():
        raise RuntimeError("Refusing to overwrite an existing suite run")
    state = {
        "worker_id": 4252179, "started_at": now(), "status": "running",
        "python": sys.executable, "policy": "default_candidate_not_frozen",
        "steps": [], "planned_matrix_cases": 536,
    }
    save(state)
    try:
        run(state, "nvidia_smi", [
            "nvidia-smi",
            "--query-gpu=name,driver_version,mig.mode.current,memory.total,memory.free,"
            "power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu",
            "--format=csv",
        ])
        run(state, "gpu_processes", [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ])
        run(state, "environment", [sys.executable, "-B", "-c", """
import sys, torch, triton
print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Triton:", triton.__version__)
assert torch.cuda.is_available() and torch.version.hip is None
print("GPU:", torch.cuda.get_device_name())
print("Capability:", torch.cuda.get_device_capability())
print("Properties:", torch.cuda.get_device_properties(0))
print("Free/total bytes:", torch.cuda.mem_get_info())
assert torch.cuda.get_device_capability()[0] >= 8
x = torch.randn(128, 128, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print("FP16 PyTorch smoke: OK")
"""])
        run(state, "git_revision", ["git", "rev-parse", "HEAD"])
        run(state, "git_status", ["git", "status", "--short"])
        run(state, "self_test", [sys.executable, "-B", str(ENTRY), "self-test"])
        run(state, "smoke_dry_run", [
            sys.executable, "-B", str(ENTRY), "matrix", "--preset", "a30-smoke",
            "--output-dir", str(ROOT / "smoke"), "--dry-run",
        ])
        for stage in STAGES:
            run(state, stage, [
                sys.executable, "-B", str(ENTRY), "matrix", "--preset", f"a30-{stage}",
                "--output-dir", str(ROOT / stage),
            ])
            manifest = json.loads((ROOT / stage / "manifest.json").read_text())
            state["steps"][-1]["matrix_summary"] = {
                "case_count": len(manifest["cases"]),
                "complete": manifest["complete"],
                "resource_skipped": manifest["resource_skipped"],
                "all_cases_validated": manifest["all_cases_validated"],
            }
            if stage == "smoke":
                state["smoke_gate"] = smoke_gate()
                print(f"{now()} Smoke correctness and MMA gate passed", flush=True)
            save(state)
        state["status"] = "completed"
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        traceback.print_exc()
    finally:
        state["finished_at"] = now()
        save(state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    raise SystemExit(main())
