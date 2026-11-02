"""Seal a byte-LM experiment and admit one finite resident worker through mrun."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from mrun.client.submit import submit


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(
    source: Path, destination: Path, request: dict, checkpoints_from: Path | None = None
) -> None:
    workspace = source.parents[2]
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / "source"
    payload.mkdir()
    for path in sorted(source.glob("*.py")):
        if not path.name.startswith("test_"):
            shutil.copy2(path, payload / path.name)
    for name in (
        "README.md",
        "protocol.json",
        "algorithmic-protocol.json",
        "memory-futures-protocol.json",
        "transport-control-protocol.json",
    ):
        if (source / name).exists():
            shutil.copy2(source / name, payload / name)
    shutil.copytree(source / "data", payload / "data")
    if checkpoints_from is not None:
        checkpoint_dir = payload / "checkpoints"
        checkpoint_dir.mkdir()
        request["checkpoint_lineage"] = {}
        for seed in request["seeds"]:
            if request["stage"] == "memory_futures":
                parent = checkpoints_from / "memory" / f"seed-{seed}" / "phase_reference"
                summary = json.loads((parent / "summary.json").read_text())
                request["checkpoint_lineage"][str(seed)] = {}
                for cut in (128,) if request.get("transition_contrast") else (128, 160, 384):
                    curve = next(row for row in summary["curves"] if row["updates"] == cut)
                    archive = curve["receipt"]["candidate"]["evaluation"]["evidence"][
                        "candidate_archive"
                    ]
                    if (
                        archive["fingerprint"] != curve["receipt"]["post_state_fingerprint"]
                        or not curve["receipt"]["decision"]["promoted"]
                    ):
                        raise RuntimeError(
                            "declared memory parent is not the accepted source closure"
                        )
                    if request.get("transition_contrast") and (
                        request["reference_endpoints"][str(seed)]["common_parent_fingerprint"]
                        != archive["fingerprint"]
                    ):
                        raise RuntimeError(
                            "transport-control reference uses a different complete parent"
                        )
                    path = parent / "training/training_snapshots" / Path(archive["path"]).name
                    if sha256(path) != archive["sha256"]:
                        raise RuntimeError(
                            f"memory checkpoint custody mismatch seed{seed}/cut{cut}"
                        )
                    name = f"seed-{seed}-cut-{cut}.pt"
                    shutil.copy2(path, checkpoint_dir / name)
                    request["checkpoint_lineage"][str(seed)][str(cut)] = {
                        "path": f"checkpoints/{name}",
                        "sha256": archive["sha256"],
                        "closure_fingerprint": archive["fingerprint"],
                        "accepted_updates": cut,
                        "architecture": summary["architecture"],
                        "parent_run": str(checkpoints_from),
                    }
                continue
            parent = checkpoints_from / f"seed-{seed}" / "phase_reference"
            summary = json.loads((parent / "summary.json").read_text())
            archive = summary["frozen_snapshot"]
            path = parent / "training_snapshots" / Path(archive["path"]).name
            if sha256(path) != archive["sha256"]:
                raise RuntimeError(f"frozen checkpoint custody mismatch for seed {seed}")
            name = f"seed-{seed}.pt"
            shutil.copy2(path, checkpoint_dir / name)
            request["checkpoint_lineage"][str(seed)] = {
                "path": f"checkpoints/{name}",
                "sha256": archive["sha256"],
                "closure_fingerprint": archive["fingerprint"],
                "accepted_updates": summary["accepted_updates"],
                "architecture": summary["architecture"],
                "parent_run": str(checkpoints_from),
            }
    packages = {
        "saturn_pub": workspace / "saturn-pub/src/saturn_pub",
        "mdb": workspace / "saturn-pub/packages/mdb/src/mdb",
        "spf": workspace / "spectral-phase-format-research/src/spf",
    }
    for name, path in packages.items():
        shutil.copytree(
            path,
            payload / "vendor" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    (payload / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (payload / "preflight.v1.json").write_text(
        json.dumps(
            {
                "schema_version": "mrun-preflight-config-v1",
                "path_token_allowlist": ["outputs"],
                "hashed_roots": ["vendor", "data"] + (["checkpoints"] if checkpoints_from else []),
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "schema": "spf-byte-lm-source-v1",
        "packages": {name: str(path) for name, path in packages.items()},
        "files": {
            str(path.relative_to(payload)): sha256(path)
            for path in sorted(payload.rglob("*"))
            if path.is_file()
        },
    }
    (payload / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(payload / "source-manifest.json", destination / "source-manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("probe", "pilot", "diagnostic", "algorithmic", "memory_futures"),
        default="probe",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--interval", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--checkpoints-from", type=Path)
    parser.add_argument(
        "--studies", choices=("text", "memory"), nargs="+", default=["text", "memory"]
    )
    parser.add_argument("--memory-steps", type=int, default=384)
    parser.add_argument("--cold-audit", action="store_true")
    parser.add_argument("--future-parent-step", type=int, default=128)
    parser.add_argument("--no-futures", action="store_true")
    parser.add_argument("--future-steps", type=int, default=256)
    parser.add_argument("--transition-contrast", action="store_true")
    parser.add_argument("--reference-futures-from", type=Path)
    args = parser.parse_args()
    if not (1 <= args.steps <= 1024 and 1 <= args.interval <= 32):
        parser.error("first study is bounded to 1–1024 updates and 1–32 step futures")
    if not (1 <= len(args.seeds) <= 3 and 10 <= args.length <= 256):
        parser.error("first study is bounded to three seeds and 10–256 byte contexts")
    if not (1 <= args.cpu_threads <= 4):
        parser.error("first study uses 1–4 CPU threads")
    if args.stage in {"diagnostic", "memory_futures"} and args.checkpoints_from is None:
        parser.error("frozen diagnostics require --checkpoints-from the pilot outputs")
    if args.stage in {"algorithmic", "memory_futures"} and args.device != "cpu":
        parser.error("algorithmic pilot uses the measured CPU backend")
    if args.stage in {"algorithmic", "memory_futures"} and (
        args.width != 24
        or args.batch_size != 4
        or args.length != 64
        or args.interval != 16
        or args.cpu_threads != 1
    ):
        parser.error("algorithmic geometry is fixed: width24, textB4/T64, interval16, CPU1")
    if args.stage == "memory_futures" and args.future_steps != 256:
        parser.error("memory continuation protocol is fixed to 256 future updates per arm")
    if args.transition_contrast and (
        args.stage != "memory_futures" or args.reference_futures_from is None
    ):
        parser.error("transition contrast requires memory_futures and --reference-futures-from")
    if not (16 <= args.memory_steps <= 1024 and args.memory_steps % 16 == 0):
        parser.error("memory budget must be 16–1024 updates in sixteen-step intervals")
    request = {
        "schema": "spf-byte-lm-request-v1",
        "stage": args.stage,
        "device": args.device,
        "dtype": "float32",
        "seeds": args.seeds,
        "steps": args.steps,
        "interval_steps": args.interval,
        "batch_size": args.batch_size,
        "sequence_length": args.length,
        "d_model": args.width,
        "n_layers": 2,
        "state_size": 8,
        "cpu_threads": args.cpu_threads,
        "arm_specs": {
            "phase_reference": {"d_model": args.width},
            "mamba": {"d_model": max(2, round(args.width * 5 / 6))},
            "transformer": {"d_model": max(2, round(args.width * 3 / 8) * 2), "n_heads": 2},
        },
        "physical_geometry": {
            "feedback": {"batch": 8, "length": args.length},
            "autonomous_rollout": {"batch": 1, "generated_bytes": 8, "prefix_bytes_max": 24},
            "phase_native_feedback": {
                "batch": 1,
                "length": 16,
                "accumulators": ["packed", "clocked"],
            },
            "final_test": {"batches": 16, "batch": args.batch_size, "length": args.length},
            "phase_final_arithmetic": {"batch": 1, "lengths": [16, 64, 128]},
        },
        "learning_rate": 0.003,
        "one_resident_worker": True,
        "model_weight_downloads": 0,
        "heldout_used_for_selection": False,
        "discovery_stage": "exploratory",
    }
    if args.stage == "diagnostic":
        request["steps"] = 0
        request["physical_geometry"]["frozen_word_panels"] = {
            "prefix_length": 16,
            "suffix_length": 16,
            "batch": 1,
            "state_and_phase_panels": True,
        }
    if args.stage == "algorithmic":
        request.update(
            studies=args.studies,
            memory_steps=args.memory_steps,
            cold_audit=args.cold_audit,
            future_parent_step=args.future_parent_step,
            run_futures=not args.no_futures,
        )
        request["arm_specs"] = {
            "phase_reference": {"d_model": 24},
            "mamba": {"d_model": 20},
            "transformer_pos1": {"d_model": 18, "n_heads": 2, "positional_scale": 1.0},
            "transformer_pos002": {"d_model": 18, "n_heads": 2, "positional_scale": 0.02},
        }
        request["physical_geometry"] = {
            "text_train": {"batch": 4, "length": 64, "updates_per_condition": args.steps},
            "text_feedback": {"panels": 3, "batch": 8, "length": 64},
            "memory_train": {
                "batch": 8,
                "variable_length": [3, 7, 11, 19, 35],
                "updates_per_condition": args.memory_steps,
            },
            "memory_feedback": {"delays": [0, 8, 32], "batch": 8},
            "memory_final": {"delays": [0, 4, 8, 16, 32, 64], "examples_per_delay": 32},
            "future_siblings": {
                "enabled": not args.no_futures,
                "first_seed_only": True,
                "conditions": ["phase_reference", "transformer_pos1"],
                "parent_update": args.future_parent_step,
                "learning_rates": [0.003, 0.001],
                "horizon": 64,
                "observe_at": [16, 32, 64],
            },
            "native_mdb": {
                "delays": [0, 8, 32],
                "examples_per_delay": 8,
                "packed_prefix_state": True,
                "clocked_max_delay": 16,
            },
        }
    if args.stage == "memory_futures":
        request.update(
            future_steps=args.future_steps,
            steps=0,
            cold_audit=False,
            transition_contrast=args.transition_contrast,
        )
        request["arm_specs"] = {"phase_reference": {"d_model": 24}}
        request["physical_geometry"] = {
            "parent_cuts": [128, 160, 384],
            "future_parent": 128,
            "schedules": ["original384curriculum", "stationary0,4,8,16,32"],
            "learning_rates": [0.003, 0.001],
            "future_steps_per_arm": 256,
            "future_count": 4,
            "train": {"batch": 8, "max_length": 35, "query_targets": 8},
            "feedback": {
                "delays": [0, 4, 8, 16, 32, 64],
                "seeds": [81103, 81107, 81119],
                "batch": 8,
            },
            "native_mdb": {"cuts_and_endpoints": 7, "delays": [0, 8, 32], "batch": 8},
        }
        if args.transition_contrast:
            request["reference_endpoints"] = {}
            for seed in args.seeds:
                refroot = args.reference_futures_from / f"seed-{seed}"
                summary = json.loads((refroot / "summary.json").read_text())
                archive = summary["futures"]["stationary-lr-003"]["endpoint_archive"]
                path = refroot / "futures/cut-128/stationary-lr-003" / Path(archive["path"]).name
                if sha256(path) != archive["sha256"]:
                    raise RuntimeError("transport-control reference endpoint custody changed")
                request["reference_endpoints"][str(seed)] = {
                    **archive,
                    "parent_run": str(args.reference_futures_from),
                    "common_parent_fingerprint": summary["parent"]["archive"]["fingerprint"],
                }
            request["physical_geometry"].update(
                parent_cuts=[128],
                schedules=["stationary0,4,8,16,32"],
                learning_rates=[0.003],
                future_count=2,
                transition_frequency_factors={"control": "ones", "cut": "layer0 zero; layer1 one"},
                native_mdb={"cuts_and_endpoints": 4, "delays": [0, 8, 32], "batch": 8},
            )
    here = Path(__file__).resolve().parent
    run = args.run_dir.resolve()
    seal(here, run, request, args.checkpoints_from)
    task = "analysis" if args.stage == "diagnostic" else "training"
    config = {
        **request,
        "task_family": task,
        "source_manifest_sha256": sha256(run / "source-manifest.json"),
        "retry_on_kill": False,
        "seq_lens": [args.length],
    }
    note = (
        "Algorithmic SPF diagnosis: sequential fixed-budget text conditioning and byte-memory controls; "
        f"seeds={args.seeds}, text B4x64/{args.steps} updates, memory B8xvariableT/{args.memory_steps} updates; "
        "Saturn16-step inspected closures, same-parent LR64-step futures with16/32/64 observations; "
        "MDB native word-state clears/transplants and repair. No reopened text test; finite CPU1 lease, no retries."
    )
    if args.stage == "memory_futures":
        note = (
            f"SPF memory interference test seeds={args.seeds}: exact parent at update128, four 256-step futures "
            "crossing original and stationary delays with LR .003/.001. B8 maxT35, 8 query targets/update; "
            "observations16/32/64/128/256, retained endpoints, exact parent restore, original384 replay control. "
            "Frozen MDB native word state and component diagnostics at 3 old cuts and 4 endpoints. Cold closed; finite CPU1 lease."
        )
        if args.transition_contrast:
            note = (
                f"SPF transport cut seeds={args.seeds}: exact128 parent, two 256-step stationary-delay "
                "futures at LR.003, differing ONLY layer0 transition frequency omega1 vs0. B8 maxT35; "
                "zero-update cut observation, native MDB plus components, all closures retained; "
                "unchanged control must replay earlier stationary.003 endpoint. Cold closed, finite CPU1 lease."
            )
    job_id = submit(
        experiment="spf-algorithmic-diagnosis"
        if args.stage in {"algorithmic", "memory_futures"}
        else "spf-byte-language-pilot",
        cmd=["uv", "run", "--project", "{env}", "python", "bootstrap.py"],
        config=config,
        needs={"cuda": True} if args.device == "cuda" else {"cpu": True},
        prefer_host="beast",
        payload=run / "source",
        model=None,
        task=task,
        device=args.device,
        dtype="float32",
        timeout_s=args.timeout,
        retry_on_kill=False,
        preflight=True,
        guarded=True,
        note=note
        if args.stage in {"algorithmic", "memory_futures"}
        else (
            f"SPF raw-byte LM {args.stage}: real Mamba v1, causal Transformer and phase reference; "
            f"sequential arms, B={args.batch_size}, T={args.length}, {args.steps} attempted updates "
            f"per seed/arm, {args.interval}-step reversible public Saturn futures; "
            "MDB same-parent carrier panels and native packed/clocked replay. "
            "No test-guided tuning; stop at finite request. Sizing from mrun probe/history."
        ),
    )
    (run / "submission.json").write_text(
        json.dumps({"job_id": job_id, "request": request}, indent=2) + "\n"
    )
    print(json.dumps({"job_id": job_id, "run_dir": str(run)}), flush=True)


if __name__ == "__main__":
    main()
