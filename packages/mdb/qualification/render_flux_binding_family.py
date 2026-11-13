"""Normalize authenticated, saved computation-family evidence for the public viewer.

Only compact JSON, source scripts and retained PNG bytes are read. This does not
load arrays or weights, run a model, score semantics or extend applicability.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "mdb-computation-evidence-view-v1"
COLLECTION = "mdb-family-authenticated-compact-v1"
AUDIT = "mdb-flux-binding-family-saved-audit-v1"
LIMIT = 16 << 20
IMAGE_LIMIT = 4 << 20
MEDIA_LIMIT = 40 << 20
PORTS = {"latent": "latent4", "decoded": "decodedRaw", "pixels": "pngRGB"}
SCOPE = ("Source-assisted joint0 inputs and a fixed Q receiver. Original-weight equations preserve the native finite "
         "operation order and consume their own forecast states. Numerical comparisons, historical development "
         "effects and prospective frozen-rule outcomes remain separate; no semantic or universal claim is inferred.")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)

    path = Path(path)
    require(path.stat().st_size <= LIMIT, "metadata exceeds bound")
    value = json.loads(path.read_bytes(), parse_constant=reject)
    canonical(value)
    return value


def safe_member(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "unsafe compact member")
    path = root / name
    require(path.resolve().is_relative_to(root) and not any(part.is_symlink() for part in (path, *path.parents)),
            "compact symlink/escape refused")
    return path


def checked_compact(run, kind):
    run = Path(run).resolve()
    if run.name == "compact":
        run = run.parent
    collection_path, root = run / "collection.json", run / "compact"
    collection = read(collection_path)
    require(collection["schema"] == COLLECTION and collection["kind"] == kind, "authenticated family compact required")
    for name, record in collection["inventory"]["files"].items():
        path = safe_member(root, name)
        require(path.suffix in {".json", ".jsonl", ".py", ".png", ".md", ".txt"}, "non-metadata compact member refused")
        require(path.stat().st_size <= (IMAGE_LIMIT if path.suffix == ".png" else LIMIT), "compact member exceeds bound")
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "compact member hash differs: " + name)
    custody = collection["payload_custody"]
    receipts = [custody[key] for key in ("declared", "sealed", "executed")]
    require(custody["required"] is True and len({(row["sha256"], row["size_bytes"]) for row in receipts}) == 1,
            "collector payload custody disagrees")
    return root, collection, dict(path=str(collection_path), sha256=sha(collection_path),
        checked_members=len(collection["inventory"]["files"]), scope="Local collected metadata/PNG hashes; remote arrays remain under source custody")


def checked_json(root, collection, name):
    require(name in collection["inventory"]["files"], "required metadata absent from collection: " + name)
    return read(safe_member(root, name))


def metric(row, case, port, *, source):
    result = dict(row)
    reference, candidate = row.get("reference_rms"), row.get("candidate_rms")
    rms = row.get("rms")
    result.update(case=case, context=case.rpartition("/")[0], arm=case.rpartition("/")[2], method="equations", port=port,
        referenceRms=reference, candidateRms=candidate,
        maxAbs=row.get("max_abs", row.get("residual", {}).get("max_abs")),
        relativeRms=rms / reference if reference is not None and reference > 0 and rms is not None else None,
        source=source, comparison_scope="Original-weight full-field forecast versus native reference")
    return result


def png(root, collection, record):
    name = record["path"]
    path = safe_member(root, name)
    if not path.is_file():
        return dict(status="unavailable", path=name, reason="PNG is not retained in the local compact")
    expected = collection["inventory"]["files"].get(name)
    require(expected is not None and expected == {key: record[key] for key in ("bytes", "sha256")}, "image/collection identity differs")
    raw = path.read_bytes()
    require(len(raw) <= IMAGE_LIMIT and hashlib.sha256(raw).hexdigest() == record["sha256"]
            and raw.startswith(b"\x89PNG\r\n\x1a\n"), "retained PNG identity differs")
    return dict(status="verified", path=name, member=name, sha256=record["sha256"], bytes=len(raw),
        data_url="data:image/png;base64," + base64.b64encode(raw).decode("ascii"))


def bundle(native_run, audit_run, *, records=()):
    from mdb.evidence_viewer import load_records
    from mdb.forecast_transaction import verify_transaction

    native, nc, nreceipt = checked_compact(native_run, "native")
    audited, ac, areceipt = checked_compact(audit_run, "audit")
    request, plan, summary, roots, states, runtime, writers = (checked_json(native, nc, name) for name in (
        "request.json", "plan.json", "summary.json", "source-roots.json", "state-index.json", "runtime-contract.json", "native-source-writers.json"))
    audit = checked_json(audited, ac, "audit.json")
    raw_manifest = checked_json(native, nc, "files.json")
    request_sha = hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(request_sha == request["request_sha256"] == nc["request_sha256"] == nc["publication"]["request_sha256"]
        == summary["request_sha256"] == audit["native_request_sha256"], "native request identity differs")
    require(audit["schema"] == AUDIT and summary["schema"] == "mdb-flux-binding-family-v1"
        and summary["job_id"] == nc["job_id"] == audit["native_job_id"] == ac["publication"]["source_job"], "audit/native source job differs")
    require(sha(native / "files.json") == audit["native_manifest_sha256"] == nc["publication"]["files_manifest_sha256"]
        and raw_manifest["request_sha256"] == request_sha and raw_manifest["job_id"] == summary["job_id"], "audit/native raw manifest differs")
    require(sha(audited / "audit.json") == ac["publication"]["audit_sha256"]
        and sha(native / "summary.json") == nc["publication"]["summary_sha256"], "published report hash differs")
    require(audit["source_worker_sha256"] == request["source_hashes"]["flux_binding_family_campaign.py"], "audited worker differs")
    for name, record in nc["inventory"]["files"].items():
        if name != "files.json":
            require(raw_manifest["files"].get(name) == record, "compact/raw member identity differs: " + name)
    require(set(states) == set(audit["cases"]) == {cid + "/" + arm for cid in roots for arm in plan["arms"]}
        and audit["contexts"] == summary["contexts"] == list(roots) and audit["arms"] == summary["arms"] == plan["arms"], "case census differs")
    verified = verify_transaction(native / "forecast-transaction", expected_seal_sha256=summary["forecast_seal"],
        expected_event_head=summary["transaction"]["event_head"])
    require(verified == summary["transaction"] == audit["transaction"], "saved transaction receipts disagree")
    models = {record["model_identity"] for row in roots.values() for record in row["inputs"].values()}
    require(len(models) == 1, "mixed original model identities require separate views")
    model = next(iter(models))
    cases, futures, endpoints, media_bytes = [], [], [], 0
    for cid in roots:
        for arm in plan["arms"]:
            key = cid + "/" + arm
            state, source, account = states[key], roots[cid], audit["cases"][key]
            require(state["context"] == cid and state["arm"] == arm, "state occurrence identity differs")
            q = source["inputs"]["Q"]
            cases.append(dict(id=key, context=cid, arm=arm, input_cut=q["cut"], previous_cut=None, model_identity=model,
                execution_point=q.get("execution_point", q.get("point")), previous_velocity_role="Not used in this assay",
                inputs=dict(source_inputs=source["inputs"], support=source["support"], wrong_rows=source.get("wrong_rows"),
                    fixed_Q_receiver=writers[cid]["sources"]["Q"]["state"], source_writer_ports=writers[cid]["sources"],
                    base_conditioning=source["base_conditioning"], known=source.get("known", {}), final_states=state,
                    entry_and_consumer_accounts=account, construction=audit["construction_checks"][cid][arm],
                    native_effect_accounts=audit["observations"][cid])))
            metrics = {}
            for field, row in account["decoder"].items():
                if field in PORTS:
                    metrics[PORTS[field]] = metric(row, key, PORTS[field], source="independent saved full-array reader")
            equation = audit.get("equation_reports", {}).get(key, {}).get("continuation", {}).get("fields", {}).get("latent")
            if equation:
                residual = equation["residual"]
                metrics["latent4"] = metric(dict(rms=residual["l2"] / math.sqrt(residual["elements"])
                    if residual["elements"] else 0., residual=residual), key, "latent4", source="saved original-array FP64 equation report")
            endpoints.extend(metrics.values())
            for lane, method in (("native", "native"), ("forecast", "equations")):
                descriptor = audit["images"].get(key + "/" + lane)
                image = dict(status="unavailable", reason="No PNG descriptor retained")
                if descriptor:
                    member = state["root"] + "/decoder/" + ("native-reference" if lane == "native" else "forecast") + "/image.png"
                    require({name: descriptor[name] for name in ("sha256", "bytes")} == raw_manifest["files"].get(member), "audit/native PNG provenance differs")
                    image = png(audited, ac, descriptor)
                media_bytes += image.get("bytes", 0)
                require(media_bytes <= MEDIA_LIMIT, "portable image total exceeds bound")
                futures.append(dict(id=key + "/" + method, case_id=key, method=method, image=image,
                    field_refs=dict(final_state=state[lane + "_final"]), metrics=metrics if method == "equations" else {}))
    linked = []
    for item in load_records(records):
        record = item["record"]
        occurrences = {row["case"] for group in record.get("results", []) if isinstance(group, dict)
                       for row in group.get("endpoint_rows", []) if isinstance(row, dict) and isinstance(row.get("case"), str)}
        exact = record.get("request_sha256") == request_sha and record.get("model_identity") == model
        matches = sorted(occurrences & set(states)) if exact else []
        linked.append(dict(item, linked_cases=matches, link_status="exact-run-and-model" if matches else "unlinked",
            link_reason="Only explicit case occurrences under this exact request and model are linked.",
            authority="Supplied computation record; no symbol binding or broader applicability inferred"))
    first = roots[next(iter(roots))]["inputs"]["Q"]
    execution = first.get("execution", {})
    body = dict(schema=SCHEMA, binding=dict(model_identity=model, model_name="FLUX.2 Klein 4B distilled",
        request_sha256=request_sha, job_id=summary["job_id"], reference_job=None, observed_input_dtype=execution.get("dtype"),
        observed_input_device=execution.get("device", {}).get("type"), numerical_profile=runtime,
        population=("Historical component development; no held-out family claim" if plan["mode"] == "historical"
                    else "Frozen-rule fresh contexts with source-assisted observed native prefixes")),
        cases=cases, futures=futures, records=linked, comparisons=dict(endpoint=endpoints, response=[], interaction=[]),
        methods=[dict(id="native", label="Native reference", role="Original native finite execution"),
                 dict(id="equations", label="Original-weight equations", role="Own-state continuation; shared primitive kernels")],
        pipeline=[dict(id="inputs", label="Joint0 inputs", port=None), dict(id="writer", label="Selected text writer", port=None),
            dict(id="consumer", label="Joint1 and suffix", port=None), dict(id="latent4", label="Final latent", port="latent4"),
            dict(id="decodedRaw", label="VAE", port="decodedRaw"), dict(id="pngRGB", label="Image", port="pngRGB")],
        evidence=dict(report=dict(sha256=sha(audited / "audit.json"), collection_hash_checked=True),
            images=dict(verified=sum(row["image"]["status"] == "verified" for row in futures), bytes=media_bytes),
            transaction=dict(status="verified", result=verified, scope="Local public transaction metadata; external arrays verified by saved reader"),
            seal_sha256=summary["forecast_seal"], preseal_future_file_opens=None,
            producer_native_witnesses=[dict(case=case, **row) for case, row in audit.get("known_endpoint_checks", {}).items()],
            external_files=dict(status="saved-reader-reported", files=audit["external_sealed_files"], scope="Viewer does not reopen arrays"),
            mechanics_status=audit["mechanics_status"], instrument_status=audit["instrument_status"], terminal_status=audit["terminal_status"],
            source_audit=audit, native_collection=nreceipt, audit_collection=areceipt),
        work=dict(summary["counts"]), limitations=[*audit["limitations"], SCOPE,
            "Native arm-versus-support effects and lattice interactions retain their original orientation in source_audit/observations; they are not forecast-error rows.",
            "Only collected metadata and PNG bytes are checked locally; original arrays, CAS states and weights are not loaded."],
        provenance=dict(source_schema=AUDIT, native_root=str(native), audit_root=str(audited),
            normalizer_sha256=sha(__file__), model_executions=0, tensor_array_reads=0, live_controls=False, read_only=True))
    body["content_sha256"] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def render(native_run, audit_run, output, *, records=(), title="MDB computation family"):
    import mdb.evidence_viewer as viewer

    output = Path(output).resolve()
    for source in (native_run, audit_run, *records):
        root = Path(source).resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "view output overlaps source evidence")
    value = bundle(native_run, audit_run, records=records)
    html = viewer.render_html(value, title=title)
    # Adapt only generated bridge-specific text. The public UI source is intact.
    legacy = "Known development states. Hold reuses each branch’s own previous velocity; the frozen predictor adds a fitted correction. Both inputs already inherit earlier interventions. Approximate agreement does not establish a transferable law."
    require(html.count(legacy) == 1 and html.count('context=data.cases[0]?.context,arm="whole_Q"') == 1, "viewer template scope/initialization changed")
    html = html.replace(legacy, SCOPE).replace('context=data.cases[0]?.context,arm="whole_Q"', 'context=data.cases[0]?.context,arm=data.cases[0]?.arm')
    text = {
        "Observed state and the velocity actually consumed by the preceding scheduler step.": "Paired joint0 source inputs and the fixed Q receiver.",
        "Historical measured reference; no estimated error.": "Native measured reference; no estimated error.",
        "These contexts and arms are dependent development conditions.": "These contexts and arms are dependent recorded conditions.",
        '<option value="response">Change from baseline Q</option><option value="interaction">Full interaction: RW − R − W + Q</option>': "",
        '<option value="velocity3">Velocity proposal</option>': "",
    }
    for old, new in text.items():
        require(html.count(old) == 1, "viewer template family label changed: " + old)
        html = html.replace(old, new)
    inputs = ('function renderInputs(caseRow){replace("input-summary",'
        '[node("p","Q joint0 input cut  "+short(caseRow.inputs.source_inputs.Q.cut),"mono small"),'
        'node("p","P joint0 input cut  "+short(caseRow.inputs.source_inputs.P.cut),"mono small")]);'
        '$("input-details").textContent=JSON.stringify({source_inputs:caseRow.inputs.source_inputs,'
        'execution_point:caseRow.execution_point,tensors:caseRow.inputs},null,2);}')
    html, substitutions = re.subn(r"function renderInputs\(caseRow\)\{[^\n]+\}", lambda _match: inputs, html)
    require(substitutions == 1, "viewer template input renderer changed")
    labels = dict(functional_source_joint0="Original-weight joint0 calls", native_source_joint0="Native joint0 calls",
        functional_source_mlp="Recomputed original-weight MLPs", native_source_mlp="Native recomputed MLPs",
        functional_transitions="Original-weight suffix transitions", native_target_transitions="Native suffix transitions",
        functional_vae="Original-weight VAE decodes", native_vae="Native VAE decodes", native_blocks="Native blocks",
        acts="Typed Acts", rewinds="Parent rewinds")
    html, substitutions = re.subn(r"const workKeys=\{[^;]+\};", "const workKeys=" + json.dumps(labels) + ";", html)
    require(substitutions == 1, "viewer template work labels changed")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in (("view.json", json.dumps(value, sort_keys=True, indent=2, allow_nan=False)), ("index.html", html)):
        (output / name).write_text(raw)
    receipt = dict(schema="mdb-binding-family-view-receipt-v1", content_sha256=value["content_sha256"], cases=len(value["cases"]),
        futures=len(value["futures"]), records=len(value["records"]), html_sha256=sha(output / "index.html"),
        json_sha256=sha(output / "view.json"), viewer_source_sha256=sha(viewer.__file__), normalizer_sha256=sha(__file__),
        rendered_html_adaptations=["truthful source/equation scope", "first actual arm initialization", "actual family counter labels",
            "paired P/Q joint0 input cut display", "generic native reference label", "omit unsupported bridge comparison/velocity controls"],
        model_executions=0, tensor_array_reads=0, source_artifacts_modified=False)
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_run", type=Path)
    parser.add_argument("audit_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--records", action="append", default=[], type=Path)
    parser.add_argument("--title", default="MDB computation family")
    args = parser.parse_args()
    print(json.dumps(render(args.native_run, args.audit_run, args.output, records=args.records, title=args.title), sort_keys=True))
