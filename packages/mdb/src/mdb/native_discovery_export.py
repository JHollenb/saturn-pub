"""Typed context-bank export using MDB's existing knowledge interchange."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .knowledge_interchange import (
    CONTEXT,
    ArtifactBinding,
    KnowledgeBank,
    KnowledgeRecord,
    LogicalTuple,
    RecordProvenance,
)
from .native_discovery import (
    DiscoveryError,
    canonical,
    checked,
    digest,
    file_hash,
    save_json,
    sealed,
)


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def export_catalog(root, catalog, binding, report):
    """Export predictive contexts, without promoting anonymous components to facts."""
    revision = catalog["fingerprint"]
    abi = digest(binding)
    source = binding["model_identity"]
    bank_binding = ArtifactBinding(source, abi, revision, revision)
    records = []
    for component in catalog["components"]:
        payload = {**component, "binding": binding, "predictor": report["predictor"],
                   "predictor_locator": "predictor/model.json", "evidence_locator": "report.json",
                   "argument_roles": ["predicted source row", "receiver row", "attention head", "depth span"],
                   "applicability": "observed checkpoint/numerical profile; supplied-token depth forecast",
                   "semantic_status": "unresolved", "causal_status": "not-established"}
        records.append(KnowledgeRecord(
            LogicalTuple(component["id"], "predictive-component-context", payload,
                         {"execution": abi, "catalog": revision}),
            kind=CONTEXT, revision=revision,
            provenance=RecordProvenance(source, abi, revision, revision, "components.json",
                                        {"fit_split": "discovery", "heldout_evidence": "report.json"})))
    bank = KnowledgeBank("native-components-" + revision[:16], bank_binding, records,
        revision=revision, metadata={"scope": "anonymous model-bound contexts; not an exported factual ontology",
                                     "recipient_adapter": "unqualified"})
    save_json(Path(root) / "knowledge-bank.json", sealed(plain(bank.to_payload())))
    return bank


def load_catalog_bank(path):
    payload = json.loads(Path(path).read_text())
    checked(payload)
    records = []
    for row in payload["records"]:
        record = KnowledgeRecord(LogicalTuple(**row["logical_tuple"]), kind=row["kind"],
            revision=row["revision"], provenance=RecordProvenance(**row["provenance"]),
            record_id=row["record_id"], mass=row["mass"], metadata=row["metadata"])
        if record.kind != CONTEXT or record.value_content_digest != row["value_content_digest"]:
            raise DiscoveryError("context-bank record contract differs")
        records.append(record)
    bank = KnowledgeBank(payload["bank_id"], ArtifactBinding(**payload["binding"]), records,
                         revision=payload["revision"], metadata=payload["metadata"])
    if bank.bank_digest != payload["bank_digest"]:
        raise DiscoveryError("context-bank digest differs")
    return bank


def source_top1_metrics(prediction):
    """Bare ``source_top1_agreement`` plus its trivial baselines (or an explicit ``None``).

    Saved runs from before ``source_top1`` existed carry only the bare field; they still load and
    export ``source_top1_baselines: None`` rather than silently omitting the comparison.
    """
    record = prediction.get("source_top1")
    baselines = None
    if isinstance(record, dict) and record.get("baselines"):
        baselines = {"baselines": record["baselines"], "best_trivial": record.get("best_trivial"),
                     "margin_over_best_trivial": record.get("margin_over_best_trivial"),
                     "beats_every_trivial_baseline": record.get("beats_every_trivial_baseline"),
                     "rows": record.get("rows")}
    return {"source_top1_agreement": prediction["source_top1_agreement"],
            "source_top1_baselines": baselines}


def index_observations(root, out):
    """Project verified immutable results into existing WATCH/XREF input formats."""
    from .evidence_citations import make_citation
    from .native_discovery import verify_run

    root, out = Path(root).resolve(), Path(out).resolve()
    if out.is_relative_to(root):
        raise DiscoveryError("index projection must be outside immutable run evidence")
    verify_run(root)
    report = json.loads((root / "report.json").read_text())
    corpus = json.loads((root / "corpus.json").read_text())
    documents = {r["id"]: r for r in corpus["documents"]}
    model = report["binding"]["model_identity"]
    rows = []
    for document in report["heldout"]:
        def emit(arm, boundary, quantity, metrics):
            rows.append({"document": document["document"], "arm": arm, "boundary": boundary,
                "metrics": metrics, "quantity": quantity, "mechanics_status": "valid",
                "terminal_status": "not-requested", "forecast_seal": document["forecast_seal"],
                "citation": make_citation(f"ar://native/{model}/boundary{boundary}/observed-residual",
                    model_id=model, input_sha256=documents[document["document"]]["text_sha256"],
                    quantity=quantity, note="Predictive observation; not semantic or causal certification")})
        for kind, predictions in document["arms"].items():
            for p in predictions:
                emit(kind, report["boundaries"][p["span"] + 1], "own-depth-state/read-error",
                     {**p["state"], "route_kl": p["route_kl"],
                      **source_top1_metrics(p)})
        for kind in ("mean_write", "hold_state"):
            if kind not in document:
                continue
            for boundary, errors in zip(report["boundaries"][1:], document.get(kind, []), strict=True):
                emit(kind, boundary, "own-depth-state-error", errors)
        for kind, metrics in document.get("native_output_consumer", {}).items():
            emit(kind, report["boundaries"][-1], "paid-native-output-consumer", metrics)
        for kind in ("mean_write", "hold_state"):
            if kind + "_consumer" in document:
                emit(kind, report["boundaries"][-1], "paid-native-output-consumer", document[kind + "_consumer"])
    out.mkdir(parents=True, exist_ok=False)
    stream = out / "observations.jsonl"
    with stream.open("xb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")
    save_json(out / "observations.json", rows)
    index = sealed({"schema": "mdb-native-discovery-v1/evidence-projection", "source": str(root),
        "run_manifest_sha256": file_hash(root / "manifest.json"),
        "source_report_sha256": file_hash(root / "report.json"), "observations": len(rows),
        "files": {name: file_hash(out / name) for name in ("observations.jsonl", "observations.json")},
        "scope": "standard WATCH/XREF projection of measured forecasts; no gate receipt or scientific certification"})
    save_json(out / "projection.json", index)
    return index
