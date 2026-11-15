"""Model-free synthetic StateCut/LocalStore overhead calibration.

This measures public closure construction, metadata reads and store operations.
It does not construct an adapter, execute a neural module, or fabricate receipts.
Production geometry is synthetic; timings do not include Session/native work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

LEGACY_WORKER_SHA256 = "d11a7be7385a6f83c775c023e07298693091d4233d9d6a5763405e9b9364b0d3"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clocks(cut, *, legacy):
    if legacy:
        # Exact five v1 before.payload reads used by native_trajectory.
        active = {"step": cut.payload["step"], "phase": cut.payload["phase"]}
        event = {"step": cut.payload["step"], "phase": cut.payload["phase"], "index": cut.payload["cursor"]}
    else:
        point = cut.execution_point
        active = {"step": point["logical_step"], "phase": point["phase"]}
        event = {"step": point["logical_step"], "phase": point["phase"], "index": point["local_clock"]}
    return active, event


def metadata(cut, *, legacy):
    # Exact record fields shared by v1/v2. Only clock extraction changed.
    clock = ({key: cut.payload[key] for key in ("step", "phase", "cursor")} if legacy else
             {"step": cut.execution_point["logical_step"], "phase": cut.execution_point["phase"], "cursor": cut.execution_point["local_clock"]})
    return {"cut": cut.fingerprint, "parent": cut.parent, "model_identity": cut.model_identity,
            "boundary": cut.boundary, "execution_point": dict(cut.execution_point),
            "store": "native-state-store", "clock": clock}


def geometry(profile):
    if profile == "production":
        return dict(image_tokens=1024, text_tokens=512, hidden=3072, conditioning_width=7680,
                    latent_width=128, rope_width=128, single_cursor=10, final_cursor=20)
    if profile == "tiny":
        return dict(image_tokens=4, text_tokens=3, hidden=8, conditioning_width=6,
                    latent_width=4, rope_width=4, single_cursor=1, final_cursor=2)
    raise ValueError("unknown synthetic geometry")


def synthetic_payload(spec, number, device):
    import torch

    phases = ("project", "joint", "single", "scheduler")
    phase = phases[number % 4]
    step = number // 4
    cursor = 0 if phase in ("project", "joint") else spec["single_cursor"] if phase == "single" else spec["final_cursor"]

    def filled(shape, value=1, dtype=torch.bfloat16):
        return torch.full(shape, value, dtype=dtype, device=device)

    n, t, h = spec["image_tokens"], spec["text_tokens"], spec["hidden"]
    context = None if phase == "project" else dict(
        temb=filled((1, h)), img_mod=filled((1, 6 * h)), txt_mod=filled((1, 6 * h)), single_mod=filled((1, 3 * h)),
        rope=(filled((n + t, spec["rope_width"]), dtype=torch.float32), filled((n + t, spec["rope_width"]), dtype=torch.float32)))
    return dict(latent=filled((1, n, spec["latent_width"])), conditioning=filled((1, t, spec["conditioning_width"])),
                img_ids=filled((1, n, 4), dtype=torch.int64), txt_ids=filled((1, t, 4), dtype=torch.int64),
                timesteps=filled((4,), dtype=torch.float32), sigmas=filled((5,), dtype=torch.float32),
                image=None if phase == "project" else filled((1, n, h), number + 1),
                text=None if phase == "project" else filled((1, t, h), number + 1), context=context,
                noise_prediction=filled((1, n, spec["latent_width"]), number + 1) if phase == "scheduler" else None,
                step=step, phase=phase, cursor=cursor)


def run(config):
    import torch
    from saturn_pub.core import StateCut
    from saturn_pub.store import LocalStore

    device = config.get("device", "cpu")
    profile, repeats = config.get("profile", "production"), config.get("repeats", 2)
    if device not in ("cpu", "cuda") or type(repeats) is not int or not 1 <= repeats <= 4:
        raise ValueError("bounded CPU/CUDA profile and1..4 repeats required")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("declared CUDA unavailable; no CPU fallback")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    spec = geometry(profile)
    torch.set_num_threads(config.get("cpu_threads", 4))

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    sync()
    started = time.perf_counter()
    cuts = []
    for number in range(4 * repeats + 1):
        payload = synthetic_payload(spec, number, device)
        point = dict(family="synthetic-flux-closure-custody", logical_step=payload["step"],
                     phase=payload["phase"], local_clock=payload["cursor"])
        cuts.append(StateCut("synthetic-no-checkpoint", {"synthetic": True, "device": device},
                             f"synthetic/{number}", cuts[-1].fingerprint if cuts else None, payload, point))
    del payload
    sync()
    build_seconds = time.perf_counter() - started
    result = {}
    for legacy in (True, False):
        mode = "v1-redundant" if legacy else "v2-cached"
        store = LocalStore(output / mode / "native-state-store")
        store.save(cuts[0])
        rows, events = [], []
        metadata_seconds = store_seconds = 0.
        for before, after in zip(cuts, cuts[1:]):
            sync()
            start = time.perf_counter()
            active, event = clocks(before, legacy=legacy)
            rows.append(metadata(after, legacy=legacy))
            events.append({"active": active, "event": event})
            sync()
            metadata_seconds += time.perf_counter() - start
            start = time.perf_counter()
            if legacy:
                store.save(before)
            store.save(after)
            sync()
            store_seconds += time.perf_counter() - start
        files = {p.relative_to(store.root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in store.root.rglob("*") if p.is_file()}
        result[mode] = dict(metadata_seconds=metadata_seconds, store_seconds=store_seconds,
                            store_calls=1 + (2 if legacy else 1) * (len(cuts) - 1),
                            source_rule_payload_reads=8 * (len(cuts) - 1) if legacy else 0,
                            rows=rows, events=events, files=files, bytes=sum(v["bytes"] for v in files.values()))
    equivalent = all(result["v1-redundant"][name] == result["v2-cached"][name] for name in ("rows", "events", "files"))
    if not equivalent:
        raise ValueError("public metadata/store equivalence failed")
    report = dict(schema="mdb-e18-custody-overhead-benchmark-v1", synthetic=True, profile=profile,
                  geometry=spec, device=device, python=platform.python_version(), torch=torch.__version__,
                  source_sha256=sha(__file__), legacy_worker_source_rule=LEGACY_WORKER_SHA256,
                  complete_cuts=len(cuts), timed_steps_per_mode=len(cuts) - 1, build_seconds=build_seconds,
                  observations=result, exact_metadata_and_store_equivalence=equivalent,
                  native_forwards=0, module_forwards=0, Session_Acts=0, Session_receipts=0,
                  limits=["Synthetic full geometry, not original activations or adapter execution",
                          "Session capture/guard/native arithmetic cost is not measured",
                          "v1 then v2 order; warm filesystem/cache can affect store timings",
                          "Payload read counts are source-rule counts, not allocator telemetry",
                          "CUDA copies/hashing occur only when device explicitly CUDA; no kernel parity claim"])
    (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run(json.loads(Path(args.config).read_text()))
    print(json.dumps({key: report[key] for key in ("schema", "device", "complete_cuts", "timed_steps_per_mode", "build_seconds", "exact_metadata_and_store_equivalence")}, sort_keys=True))


if __name__ == "__main__":
    main()
