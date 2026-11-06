"""Retrospective one-pass numerical replay from two actual native N1 states.

No model construction/native forwards. Q=G_1(N1) uses the frozen independent
equations and original checkpoint recipes. P/N targets are read only after both
Q passes are complete and their hash/landmark archive inventory is sealed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import runpy
import time
from pathlib import Path

import numpy as np

ANALYZER_SHA256 = "86084b8c751519dcb0ce4cbd5e8640a5cc84ccf12a986e6074c2e694002d40e5"
MEMOIZATION_SHA256 = "977a6df3717a714dec6d73b63b5cb3d2bfe35e10bd361aafecc3a12787cae214"
CELLS = ("c1-o0", "c0-o1")
LANDMARKS = (0, 5, 25, 26, 27)
RESULT = "flux-prompt-factorial-result"
SCOPE = ("Retrospective selected two-context pass1 replay, Q=G_1(N1). The cells differ in both prompt factors; "
    "they are not fresh prospective specimens. A=Q-N, B=P-Q, E=P-N are signed vector accounts, not causal shares "
    "or isolated kernel errors. Full comparisons exist only at five retained landmarks per context.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stats(delta, native):
    magnitude = np.abs(delta)
    energy = float(np.sum(delta * delta))
    reference = float(np.sum(native * native))
    return {"rms": float(np.sqrt(energy / delta.size)), "l2": float(np.sqrt(energy)), "squared_l2": energy,
        "signed_mean": float(delta.mean()), "max_abs": float(magnitude.max()), "mae": float(magnitude.mean()),
        "p95_abs": float(np.quantile(magnitude, .95)), "p99_abs": float(np.quantile(magnitude, .99)),
        "changed_value_count": int(np.count_nonzero(delta)), "elements": int(delta.size),
        "native_rms": float(np.sqrt(reference / delta.size)),
        "native_relative_rms": float(np.sqrt(energy / reference)) if reference else None}


def vector_account(q, p, n):
    """Full signed account, with a common native denominator for all terms."""
    require(q.shape == p.shape == n.shape and q.size and all(np.isfinite(x).all() for x in (q, p, n)),
        "vector account operands differ or are nonfinite")
    q, p, n = (np.asarray(x, dtype=np.float64) for x in (q, p, n))
    a, b, e = q - n, p - q, p - n
    dot = float(np.sum(a * b))
    aa, bb, ee = float(np.sum(a * a)), float(np.sum(b * b)), float(np.sum(e * e))
    return {"A_Q_minus_N": stats(a, n), "B_P_minus_Q": stats(b, n), "E_P_minus_N": stats(e, n),
        "dot_A_B": dot, "twice_dot_A_B": 2 * dot,
        "cosine_A_B": dot / np.sqrt(aa * bb) if aa and bb else None,
        "squared_norm_identity": {"E_squared": ee, "A_squared_plus_B_squared_plus_twice_dot": aa + bb + 2 * dot,
            "residual": ee - (aa + bb + 2 * dot)},
        "vector_identity_max_abs": float(np.max(np.abs(e - (a + b)))),
        "signs": {"A": "Q-N", "B": "P-Q", "E": "P-N"},
        "relative_denominator": "same full native N field RMS for all three terms; undefined if native RMS=0",
        "scope": "Exact signed accounting of these saved operands; no causal shares or isolated kernel attribution."}


class InputOnlyReader:
    """Permit only actual source/schedule records and checkpoint constants."""
    def __init__(self, evidence, allowed):
        self.evidence = evidence
        self.allowed = {identity(record) for record in allowed}
        self.reads = []

    def array(self, record):
        require(record.get("storage") == "checkpoint-recipe-v1" or identity(record) in self.allowed,
            "Q attempted to read a non-source/future activation")
        self.reads.append({"record_sha256": identity(record), "parameter": record.get("storage") == "checkpoint-recipe-v1"})
        return self.evidence.array(record)


def archive(output, value, descriptor):
    """Archive exact FP32 numerical output; immutable hash-addressed NPY."""
    require(value.dtype == np.float32 and np.isfinite(value).all(), "Q output must be finite FP32")
    path = output / "arrays" / (identity(descriptor) + ".npy")
    if not path.exists():
        with path.open("xb") as stream:
            np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return {"path": str(path.relative_to(output)), "sha256": sha(path), "bytes": path.stat().st_size,
        "dtype": "float32", "shape": list(value.shape), "descriptor": descriptor}


def load_q(output, record, describe):
    path = output / record["path"]
    require(path.is_relative_to(output) and not path.is_symlink() and sha(path) == record["sha256"], "Q archive identity differs")
    value = np.load(path, allow_pickle=False)
    require(describe(value) == record["descriptor"], "Q archived full field differs")
    return value


def replay_pass(reader, binding, origin, *, two, describe, context_fields, kind_at, output, progress=None):
    """Compute exactly28 events, from source operands only; no target reader."""
    values = {name: reader.array(record) for name, record in origin["inputs"].items()}
    latent, conditioning = values["latent"], values["conditioning"]
    records = binding["pass_schedules"]["1"]
    schedule = {name: reader.array(records[name]) for name in ("timestep", "sigma", "sigma_next")}
    require(float(schedule["timestep"]) == float(values["timesteps"][1])
        and float(schedule["sigma"]) == float(values["sigmas"][1])
        and float(schedule["sigma_next"]) == float(values["sigmas"][2])
        and records["next_clock"] == {"step": 2, "phase": "project", "cursor": 0}
        and records["reset"] == {key: None for key in ("context", "image", "text", "noise_prediction")},
        "pass1 source clock/schedule/reset differs")
    iterator = two["independent_pass_events"](latent, conditioning, values["img_ids"], values["txt_ids"], schedule,
        binding, lambda kind, spec: two["load_bank"](reader, kind, spec))
    transitions = []
    image = text = context_hash = previous = None
    for local, (kind, index, value) in enumerate(iterator):
        require(local < 28 and (kind, index) == kind_at(local), "Q pass operation sequence differs")
        if kind == "project":
            inputs = {"latent": describe(latent), "conditioning": origin["inputs"]["conditioning"],
                "img_ids": origin["inputs"]["img_ids"], "txt_ids": origin["inputs"]["txt_ids"], "schedule": records}
            image, text = value["image"], value["text"]
            fields = {"image": image, "text": text, **context_fields(value["context"])}
            context_hash = identity({name: describe(field) for name, field in context_fields(value["context"]).items()})
        elif kind in ("joint", "single"):
            inputs = {"image": describe(image), "text": describe(text), "own_project_context": context_hash}
            if kind == "single":
                inputs["residual"] = describe(np.concatenate((text, image), axis=1))
                text, image = value["output"][:, :conditioning.shape[1]], value["output"][:, conditioning.shape[1]:]
            else:
                image, text = value["image"], value["text"]
            fields = {"image": image, "text": text}
        elif kind == "readout":
            inputs = {"image": describe(image), "own_project_context": context_hash}
            fields = {"noise": value["noise"]}
        else:
            inputs = {"latent": describe(latent), "noise": describe(previous["noise"]),
                "sigma": records["sigma"], "sigma_next": records["sigma_next"]}
            fields = {"latent": value["latent"]}
        transitions.append({"logical_step": 1, "local_index": local, "kind": kind, "block_index": index,
            "inputs": inputs, "outputs": {name: {"descriptor": describe(field),
                "archive": archive(output, field, describe(field)) if local in LANDMARKS else None}
                for name, field in fields.items()}})
        if progress:
            progress({"phase": "retrospective-Q", "condition_id": origin["condition_id"], "logical_step": 1,
                "local_index": local, "kind": kind, "block_index": index, "native_forwards": 0})
        previous = fields
    require(len(transitions) == 28, "Q pass did not commit all28 own-state events")
    return {"condition_id": origin["condition_id"], "native_input_cut": origin["cut"], "origin": origin,
        "logical_step": 1, "chain_length": 28, "transitions": transitions,
        "input_read_proof": reader.reads, "scope": SCOPE}


def selected_native_origin(analyzer, old, evidence, binding, data, cell):
    """Only original carried operands and retained native step0 commit are inputs."""
    with (data / "native" / (cell + ".jsonl")).open() as stream:
        for index, line in enumerate(stream):
            if index == 27:
                native = json.loads(line)
                break
        else:
            raise ValueError("native N1 commit absent")
    capture = native["capture"]
    require(native["logical_step"] == 0 and native["local_index"] == 27
        and capture["retained_full_cut"] and native["outputs"]["latent"]["archive"] is not None
        and analyzer["identity"](capture["manifest"]) == capture["cut"] == native["receipt"]["result"]
        and native["receipt"] == evidence.receipts[native["receipt"]["fingerprint"]],
        "actual N1 source cut/receipt/array binding differs")
    original = binding["origins"][cell]
    old["check_origin"](evidence, original, 0)
    origin = copy.deepcopy(original)
    origin.update(cut=capture["cut"], point=capture["point"])
    origin["inputs"]["latent"] = native["outputs"]["latent"]["archive"]
    old["check_origin"](evidence, origin, 1)
    source_fields, start_fields = evidence.fields(original["cut"]), evidence.fields(origin["cut"])
    changed = {"latent", "step", "phase", "cursor"}
    require({k: v for k, v in start_fields.items() if k not in changed}
        == {k: v for k, v in source_fields.items() if k not in changed}, "N1 changed original carried operands")
    require(analyzer["describe_field"](evidence.array(origin["inputs"]["latent"])) == native["outputs"]["latent"]["descriptor"],
        "N1 full latent archive is not its observed descriptor")
    return origin, {"cut": capture["cut"], "point": capture["point"], "receipt": native["receipt"],
        "native_transition_index": 27, "source_manifest_sha256": identity(capture["manifest"])}


def run(data, *, analyzer_source, parameter_root, output, memoization_source, checkpoint=None, ssh=None,
        expected_analyzer_sha256=ANALYZER_SHA256, expected_memoization_sha256=MEMOIZATION_SHA256,
        max_decoded_bytes=18 << 30, max_retained_bytes=8 << 30, progress=None, custody=None, production_profile=True):
    """Numerical core. External wrapper must authenticate source collection.

    The optional custody JSON is bound verbatim, not independently certified by
    this core. Run only after the existing audit's lease ends, under mrun.
    """
    started = time.monotonic()
    data, output, analyzer_source, memoization_source = map(lambda p: Path(p).resolve(),
        (data, output, analyzer_source, memoization_source))
    require(not output.exists() and not output.is_relative_to(data), "output must be fresh and outside original producer tree")
    require(sha(analyzer_source) == expected_analyzer_sha256 == ANALYZER_SHA256
        and sha(memoization_source) == expected_memoization_sha256 == MEMOIZATION_SHA256, "frozen source binding differs")
    analyzer = runpy.run_path(str(analyzer_source))
    old = analyzer["independent"]()
    memo = runpy.run_path(str(memoization_source))["install"](analyzer, max_decoded_bytes,
        storage_codec="bf16-bits", max_retained_bytes=max_retained_bytes)
    try:
        source, summary, plan, binding = [read(data / name) for name in ("source.json", "summary.json", "plan.json", "binding.json")]
        analyzer["validate_plan"](plan, source, summary, production_profile=production_profile)
        require(source["binding_sha256"] == sha(data / "binding.json"), "producer source/binding identity differs")
        if production_profile:
            analyzer["source_abi"](data, source, plan)
            old["bank_profiles"](binding)
            require(source["checkpoint_files"][analyzer["CHECKPOINT_FILE"]] == analyzer["CHECKPOINT_SHA256"], "checkpoint identity differs")
        analyzer["parameter_ports"](binding)
        require(binding["parameter_artifacts"]["path"] == "parameter-artifacts.json"
            and binding["parameter_artifacts"]["sha256"] == sha(data / "parameter-artifacts.json")
            and sha(data / "retrieve_weights.py") == analyzer["PARAMETER_SHA256"],
            "checkpoint recipe manifest/retriever binding differs")
        recipes, hydration = old["parameter_cache"](data, parameter_root, checkpoint=checkpoint, ssh=ssh)
        evidence = old["Evidence"](data, parameter_root, recipes)
        output.mkdir(parents=True)
        (output / "arrays").mkdir()
        inputs, programs = {}, {}
        for cell in CELLS:
            origin, native_binding = selected_native_origin(analyzer, old, evidence, binding, data, cell)
            allowed = list(origin["inputs"].values()) + [binding["pass_schedules"]["1"][name] for name in ("timestep", "sigma", "sigma_next")]
            restricted = InputOnlyReader(evidence, allowed)
            programs[cell] = replay_pass(restricted, binding, origin, two=old["TWO"], describe=analyzer["describe_field"],
                context_fields=analyzer["context_fields"], kind_at=analyzer["kind_at"], output=output, progress=progress)
            inputs[cell] = native_binding
            save(output / (cell + ".Q.json"), programs[cell])
            gc.collect()
        seal = {"schema": "mdb-retrospective-matched-pass-Q-seal-v1", "kind": "retrospective-completed-Q-custody-seal",
            "source_job_id": source["job_id"], "source_request_sha256": source["request_sha256"],
            "sources": {"helper": sha(__file__), "analyzer": sha(analyzer_source), "memoization": sha(memoization_source)},
            "programs": {cell: {"path": cell + ".Q.json", "sha256": sha(output / (cell + ".Q.json"))} for cell in CELLS},
            "inputs": inputs, "math_transitions": 56, "target_arrays_read_before_Q_seal": 0, "native_forwards": 0,
            "scope": SCOPE, "custody": custody}
        seal["fingerprint"] = identity(seal)
        save(output / "Q-seal.json", seal)
        comparisons = {}
        from mdb.forecast_transaction import load_forecast_tree, verify_transaction
        for cell in CELLS:
            tx = data / "transactions" / cell
            verification = verify_transaction(tx, expected_seal_sha256=summary["seals"][cell]["seal_sha256"], max_bytes=plan["transaction_max_bytes"])
            p = analyzer["plain"](load_forecast_tree(tx, read(tx / "forecasts/program.json")["predictions"]))
            require(p["origin_cut"] == binding["origins"][cell]["cut"] and p["condition_id"] == cell,
                "P original origin/context differs")
            natives = [json.loads(line) for line in (data / "native" / (cell + ".jsonl")).read_text().splitlines()]
            rows = []
            for local in LANDMARKS:
                pred, native, qrow = p["transitions"][28 + local], natives[28 + local], programs[cell]["transitions"][local]
                require((pred["logical_step"], pred["local_index"], pred["kind"], pred["index"])
                    == (1, local, qrow["kind"], qrow["block_index"])
                    and (native["logical_step"], native["local_index"]) == (1, local)
                    and native["capture"]["retained_full_cut"]
                    and analyzer["identity"](native["capture"]["manifest"]) == native["capture"]["cut"] == native["receipt"]["result"]
                    and native["capture"]["manifest"] == evidence.store["cuts"][native["capture"]["cut"]]
                    and native["capture"]["point"] == evidence.point(native["capture"]["cut"])
                    and native["receipt"] == evidence.receipts[native["receipt"]["fingerprint"]]
                    and tuple(native["capture"]["point"][name] for name in ("logical_step", "phase", "index"))
                        == old["point_after"](28 + local)
                    and set(pred["outputs"]) == set(native["outputs"]) == set(qrow["outputs"]), "P/Q/N landmark clock/field join differs")
                fields = {}
                for name, qrecord in qrow["outputs"].items():
                    pa, na = pred["outputs"][name], native["outputs"][name]
                    require(pa["archive"] is not None and na["archive"] is not None, "P/N full landmark not retained")
                    declared = analyzer["native_field_descriptor"](native["capture"]["manifest"], name)
                    require(declared["shape"] == na["descriptor"]["shape"] and declared["sha256"] == na["descriptor"].get(
                        "bf16_le_sha256" if declared["dtype"] == "torch.bfloat16" else "fp32_le_sha256"),
                        "N field is not its retained typed manifest port")
                    qv = load_q(output, qrecord["archive"], analyzer["describe_field"])
                    pv, nv = evidence.array(pa["archive"]), evidence.array(na["archive"])
                    require(analyzer["describe_field"](pv) == pa["descriptor"] and analyzer["describe_field"](nv) == na["descriptor"],
                        "P/N archived landmark is not its sealed/native descriptor")
                    fields[name] = {"account": vector_account(qv, pv, nv), "Q": qrecord, "P": pa, "N": na}
                    del qv, pv, nv
                rows.append({"logical_step": 1, "local_index": local, "transition_index": 28 + local,
                    "kind": qrow["kind"], "block_index": qrow["block_index"], "native_cut": native["capture"]["cut"],
                    "native_point": native["capture"]["point"], "native_receipt": native["receipt"]["fingerprint"], "fields": fields})
            comparisons[cell] = {"rows": rows, "original_transaction_verification": verification}
        cache_report = memo.uninstall()
        result = {"schema": "mdb-retrospective-matched-pass-replay-v1", "mechanics_status": "numeric-core-valid-custody-unassessed",
            "source_job_id": source["job_id"], "source_request_sha256": source["request_sha256"],
            "source_binding_sha256": sha(data / "binding.json"), "source_json_sha256": sha(data / "source.json"),
            "Q_seal_sha256": sha(output / "Q-seal.json"), "sources": seal["sources"], "custody": custody,
            "math_transitions": 56, "logical_step": 1, "cells": list(CELLS), "comparisons": comparisons,
            "parameter_hydration": hydration, "parameter_read_cache": cache_report, "seconds": time.monotonic() - started,
            "native_forwards": 0, "source_data_mutated": False, "scope": SCOPE,
            "coverage": {"Q_complete_output_hashes": "All28 transitions per context", "Q_full_landmarks_per_context": 5,
                "P_N_full_comparisons_per_context": 5, "other_native_exits_not_inferred": 46}}
        save(output / "report.json", result)
        return result
    finally:
        if memo.stats()["active"]:
            memo.uninstall()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--analyzer-source", type=Path, required=True)
    parser.add_argument("--memoization-source", type=Path, required=True)
    parser.add_argument("--parameter-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ssh")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--custody", type=Path)
    args = parser.parse_args()
    report = run(args.data, analyzer_source=args.analyzer_source, memoization_source=args.memoization_source,
        parameter_root=args.parameter_root, output=args.out, checkpoint=args.checkpoint, ssh=args.ssh,
        custody=read(args.custody) if args.custody else None,
        progress=lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    print(json.dumps({"report_sha256": sha(args.out / "report.json"), "math_transitions": report["math_transitions"]}))
