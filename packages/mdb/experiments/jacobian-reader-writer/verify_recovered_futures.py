#!/usr/bin/env python3
"""Pure-stdlib verification of recovered native/zero future parity.

This reader never opens the Bench runtime.  It verifies the publication chain
and content-addressed JSON directly, distinguishes repeated deterministic cuts
by their journal publications, and emits a reproducible evidence record.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ORIGIN_JOB = "job-4ba3cc18d86a"
RECOVERY_JOB = "job-d47b087e4124"
DEFAULT_RECOVERY = (
    HERE / "results" / f"recovery-{ORIGIN_JOB}" / RECOVERY_JOB / "artifacts"
)
DEFAULT_OUTPUT = (
    HERE
    / "results"
    / f"recovered-audit-{ORIGIN_JOB}-v2"
    / "future-parity.json"
)
EXPECTED_PROTOCOL_SHA256 = "2f9a406ef66d3096daca9f60b66bef485fa04b9ca910579eaa1f899d1a41b20c"
EXPECTED_WORKER_SHA256 = "82f1a5917327864d7c63cfdebf5883af9e485017faf6a00d0bb5c686c33a3d21"
EXPECTED_WHEEL_SHA256 = "1120c5299ffe096a76db84ceaa71cd44348db8f03a46e6bf17e72e433b96ea2e"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


class RecoveredStore:
    def __init__(self, artifacts: Path) -> None:
        self.artifacts = artifacts
        self.root = artifacts / "output" / "bench-export"
        self._json_cache: dict[str, dict[str, Any]] = {}

    def cas(self, reference: str) -> dict[str, Any]:
        prefix = "cas://sha256/"
        if not reference.startswith(prefix):
            raise RuntimeError(f"unsupported artifact reference: {reference}")
        digest = reference.removeprefix(prefix)
        if digest in self._json_cache:
            return self._json_cache[digest]
        path = self.root / "artifacts" / digest[:2] / f"{digest}.json"
        raw = path.read_bytes()
        if sha256_bytes(raw) != digest:
            raise RuntimeError(f"artifact content address mismatch: {digest}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"artifact is not an object: {digest}")
        self._json_cache[digest] = value
        return value

    def vm_checkpoint(self, reference: str) -> dict[str, Any]:
        digest = reference.removeprefix("cas://sha256/")
        path = self.root / "vm" / "artifacts" / digest[:2] / f"{digest}.json"
        raw = path.read_bytes()
        if sha256_bytes(raw) != digest:
            raise RuntimeError(f"VM checkpoint content address mismatch: {digest}")
        value = json.loads(raw)
        if value.get("schema") != "saturn-causal-context-vm-checkpoint-v3":
            raise RuntimeError("unexpected VM checkpoint schema")
        return value

    def publication_chain(self) -> list[dict[str, Any]]:
        selected = json.loads((self.root / "selected.json").read_text())
        digest = selected["journal"]
        reverse: list[dict[str, Any]] = []
        seen: set[str] = set()
        while digest is not None:
            if digest in seen:
                raise RuntimeError("publication chain cycle")
            seen.add(digest)
            raw = (self.root / "journal" / f"{digest}.pack").read_bytes()
            if sha256_bytes(raw) != digest:
                raise RuntimeError("publication pack content address mismatch")
            body = json.loads(gzip.decompress(raw))
            if body.get("schema") != "saturn-bench-publication-v1":
                raise RuntimeError("publication pack schema mismatch")
            for write in body.get("writes", []):
                payload = base64.b64decode(write["bytes"], validate=True)
                if sha256_bytes(payload) != write["sha256"]:
                    raise RuntimeError("publication member checksum mismatch")
            reverse.append(
                {
                    "journal_sha256": digest,
                    "manifest_ref": body["manifest"],
                    "parent_journal_sha256": body.get("parent"),
                    "writes": [row["path"] for row in body.get("writes", [])],
                }
            )
            digest = body.get("parent")
        chain = list(reversed(reverse))
        for index, publication in enumerate(chain):
            publication["index"] = index
            publication["manifest"] = self.cas(publication["manifest_ref"])
        if chain[-1]["journal_sha256"] != selected["journal"]:
            raise RuntimeError("selected journal is not the chain endpoint")
        if chain[-1]["manifest_ref"] != selected["manifest"]:
            raise RuntimeError("selected manifest differs from the journal endpoint")
        return chain


def candidate_rows(store: RecoveredStore, chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    publication_by_path = {
        path: publication
        for publication in chain
        for path in publication["writes"]
        if path.startswith("artifacts/")
    }
    rows: list[dict[str, Any]] = []
    for path in store.root.glob("artifacts/*/*.json"):
        value = json.loads(path.read_text())
        if value.get("schema") != "mdb-bench-jacobian-candidate-v1":
            continue
        recipe = value["write"]["recipe"]
        if recipe.get("dose") != 0:
            continue
        relative = path.relative_to(store.root).as_posix()
        publication = publication_by_path.get(relative)
        if publication is None:
            raise RuntimeError("zero candidate has no publication event")
        rows.append(
            {
                "candidate_ref": f"cas://sha256/{path.stem}",
                "candidate_publication_index": publication["index"],
                "candidate_publication_journal_sha256": publication["journal_sha256"],
                "attempt_id": value["attempt_id"],
                "branch": value["source"]["branch"],
                "source_cut_ref": value["source"]["cut_ref"],
                "source_activation_sha256": value["source"]["activation"]["sha256"],
                "candidate_payload_ref": value["candidate_ref"]["ref"],
                "candidate_sha256": value["candidate_ref"]["sha256"],
                "delta_payload_ref": value["delta_ref"]["ref"],
                "delta_sha256": value["delta_ref"]["sha256"],
                "dose": recipe["dose"],
                "actual_delta_norm": recipe["algebra"]["actual_delta_norm"],
            }
        )
    return sorted(rows, key=lambda row: row["candidate_publication_index"])


def branch_cut(publication: dict[str, Any], branch: str) -> dict[str, Any] | None:
    record = publication["manifest"].get("branches", {}).get(branch)
    if not isinstance(record, dict):
        return None
    return record


def segment_for_candidate(
    store: RecoveredStore,
    chain: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    branch = candidate["branch"]
    start = candidate["candidate_publication_index"]
    commit: dict[str, Any] | None = None
    trace: dict[int, dict[str, Any]] = {}
    endpoint: dict[str, Any] | None = None
    for publication in chain[start + 1 :]:
        branch_record = branch_cut(publication, branch)
        if branch_record is None:
            continue
        cut_ref = branch_record["cut"]
        cut = store.cas(cut_ref)
        if commit is None:
            if cut.get("label") != "act-committed":
                continue
            lineage = cut.get("intervention_lineage")
            if not isinstance(lineage, list) or len(lineage) != 1:
                raise RuntimeError("zero candidate commit has unexpected lineage")
            commit = {
                "publication_index": publication["index"],
                "journal_sha256": publication["journal_sha256"],
                "manifest_ref": publication["manifest_ref"],
                "cut_ref": cut_ref,
                "lineage": lineage,
            }
            continue
        if cut.get("intervention_lineage") != commit["lineage"]:
            continue
        if cut.get("label") == "token":
            count = len(cut["token_ids"])
            if 22 <= count <= 33:
                trace.setdefault(
                    count,
                    {
                        "publication_index": publication["index"],
                        "journal_sha256": publication["journal_sha256"],
                        "manifest_ref": publication["manifest_ref"],
                        "cut_ref": cut_ref,
                    },
                )
                if count == 33:
                    endpoint = trace[count]
                    break
    if commit is None or endpoint is None or set(trace) != set(range(22, 34)):
        raise RuntimeError("zero candidate publication segment is incomplete")
    return {**candidate, "commit": commit, "trace": trace, "endpoint": endpoint}


def native_publication(
    store: RecoveredStore,
    chain: list[dict[str, Any]],
    branch: str,
    cut_ref: str,
    before: int,
) -> dict[str, Any]:
    matches = []
    for publication in chain[:before]:
        record = branch_cut(publication, branch)
        if record is not None and record.get("cut") == cut_ref:
            cut = store.cas(cut_ref)
            if cut.get("label") == "token" and cut.get("intervention_lineage") == []:
                matches.append(publication)
    if not matches:
        raise RuntimeError("native comparison cut has no earlier publication")
    publication = matches[-1]
    return {
        "publication_index": publication["index"],
        "journal_sha256": publication["journal_sha256"],
        "manifest_ref": publication["manifest_ref"],
        "cut_ref": cut_ref,
    }


def cut_identity(store: RecoveredStore, record: dict[str, Any]) -> dict[str, Any]:
    cut = store.cas(record["cut_ref"])
    checkpoint_ref = cut["vm_checkpoint"]
    checkpoint = store.vm_checkpoint(checkpoint_ref)
    payloads = [row["payload_ref"] for row in checkpoint.get("delta_pages", [])]
    return {
        **record,
        "parent_cut_ref": cut["parent_cut"],
        "token_ids": cut["token_ids"],
        "pending_token_id": cut["pending"],
        "logits_payload_ref": cut["logits"]["ref"],
        "logits_sha256": cut["logits"]["sha256"],
        "vm_checkpoint_ref": checkpoint_ref,
        "vm_checkpoint_delta_payload_refs": [row["ref"] for row in payloads],
        "vm_checkpoint_delta_sha256s": [row["sha256"] for row in payloads],
        "vm_checkpoint_mount_sha256": checkpoint["mount_sha256"],
        "vm_checkpoint_statecut_ref": checkpoint["statecut"],
        "model_identity": cut["identity"],
        "lane": cut["lane"],
        "numerical_contract": cut["numerical_contract"],
        "intervention_lineage": cut["intervention_lineage"],
    }


def pair_record(
    store: RecoveredStore,
    native: dict[str, Any],
    zero: dict[str, Any],
    row: str,
    segment: str,
    token_count: int,
) -> dict[str, Any]:
    left = cut_identity(store, native)
    right = cut_identity(store, zero)
    checks = {
        "token_ids_exact": left["token_ids"] == right["token_ids"],
        "pending_token_exact": left["pending_token_id"] == right["pending_token_id"],
        "logits_payload_exact": left["logits_sha256"] == right["logits_sha256"],
        "vm_checkpoint_exact": left["vm_checkpoint_ref"] == right["vm_checkpoint_ref"],
        "vm_delta_payloads_exact": left["vm_checkpoint_delta_sha256s"]
        == right["vm_checkpoint_delta_sha256s"],
        "model_identity_exact": left["model_identity"] == right["model_identity"],
        "lane_exact": left["lane"] == right["lane"],
        "numerical_contract_exact": left["numerical_contract"]
        == right["numerical_contract"],
    }
    return {
        "row": row,
        "segment": segment,
        "token_count": token_count,
        "native": left,
        "zero": right,
        "checks": checks,
        "exact": all(checks.values()),
    }


def verify(recovery: Path) -> dict[str, Any]:
    store = RecoveredStore(recovery)
    chain = store.publication_chain()
    selected = json.loads((store.root / "selected.json").read_text())
    protocol_path = recovery / "source" / "protocol.json"
    protocol_raw = protocol_path.read_bytes()
    if sha256_bytes(protocol_raw) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("recovered protocol identity mismatch")
    protocol = json.loads(protocol_raw)
    recovery_manifest = json.loads((recovery / "recovery-manifest.json").read_text())
    if recovery_manifest["expected_source_sha256"]["real_bf16_worker.py"] != EXPECTED_WORKER_SHA256:
        raise RuntimeError("recovery manifest worker identity mismatch")
    if recovery_manifest["expected_source_sha256"]["protocol.json"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("recovery manifest protocol identity mismatch")
    if (
        recovery_manifest["expected_source_sha256"]["mdb_runtime-0.1.0-py3-none-any.whl"]
        != EXPECTED_WHEEL_SHA256
    ):
        raise RuntimeError("recovery manifest wheel identity mismatch")
    origin_record_path = HERE / "results" / ORIGIN_JOB / "job-record.json"
    origin_record_raw = origin_record_path.read_bytes()
    origin_record = json.loads(origin_record_raw)
    if origin_record["job_id"] != ORIGIN_JOB or origin_record["state"] != "timeout":
        raise RuntimeError("origin record is not the preserved timeout")
    if sha256_bytes(origin_record_raw) != recovery_manifest["origin_job_record_sha256"]:
        raise RuntimeError("origin record differs from recovery custody")
    expected_sources = recovery_manifest["expected_source_sha256"]
    if any(origin_record["config"]["source_sha256"].get(key) != value for key, value in expected_sources.items()):
        raise RuntimeError("origin request source identities differ from recovery custody")
    candidates = candidate_rows(store, chain)
    by_branch: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_branch.setdefault(candidate["branch"], []).append(candidate)
    if [len(by_branch.get(name, [])) for name in ("cat-to-fox-recipient", "bear-to-lion-recipient")] != [1, 2]:
        raise RuntimeError("unexpected zero-candidate attempt inventory")

    segment_specs = [
        ("cat-to-fox", "in-run-zero", by_branch["cat-to-fox-recipient"][0]),
        ("bear-to-lion", "in-run-zero", by_branch["bear-to-lion-recipient"][0]),
        ("bear-to-lion", "export-reopen-zero", by_branch["bear-to-lion-recipient"][1]),
    ]
    segments: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    native_cut_refs: dict[tuple[str, int, tuple[int, ...]], str] = {}
    for path in store.root.glob("artifacts/*/*.json"):
        value = json.loads(path.read_text())
        if (
            value.get("schema") == "saturn-open-bench-cut-v1"
            and value.get("label") == "token"
            and value.get("intervention_lineage") == []
        ):
            token_ids = tuple(value.get("token_ids", []))
            native_cut_refs[(value["branch"], len(token_ids), token_ids)] = (
                f"cas://sha256/{path.stem}"
            )
    for row, name, candidate in segment_specs:
        segment = segment_for_candidate(store, chain, candidate)
        segment_summary = {key: value for key, value in segment.items() if key != "trace"}
        segment_summary["trace_publications"] = [segment["trace"][count] for count in range(22, 34)]
        segments.append({"row": row, "segment": name, **segment_summary})
        for count in range(22, 34):
            zero_record = segment["trace"][count]
            zero_cut = store.cas(zero_record["cut_ref"])
            native_cut_ref = native_cut_refs.get(
                (candidate["branch"], count, tuple(zero_cut.get("token_ids", [])))
            )
            if native_cut_ref is None:
                raise RuntimeError("native trace cut is unavailable")
            native_before = by_branch[candidate["branch"]][0][
                "candidate_publication_index"
            ]
            native_record = native_publication(
                store,
                chain,
                candidate["branch"],
                native_cut_ref,
                native_before,
            )
            pairs.append(pair_record(store, native_record, zero_record, row, name, count))

    reopen = segments[-1]
    endpoint_ref = reopen["endpoint"]["cut_ref"]
    selected_manifest = store.cas(selected["manifest"])
    selected_identity = selected_manifest["identity"]
    identity_checks = {
        "protocol_model_id_exact": selected_identity["model_id"] == protocol["model_id"],
        "protocol_revision_exact": selected_identity["revision"] == protocol["revision"],
        "protocol_weights_exact": selected_identity["weights_sha256"]
        == protocol["model_safetensors_sha256"],
        "protocol_wheel_exact": protocol["wheel_sha256"] == EXPECTED_WHEEL_SHA256,
        "native_attention_backend": selected_manifest["attention_backend"] == "native-eager",
        "bf16_identity": selected_identity["dtype"] == "torch.bfloat16",
    }
    selected_branch = selected_manifest["branches"]["bear-to-lion-recipient"]
    close_ref = selected_branch["cut"]
    close_cut = store.cas(close_ref)
    endpoint_cut = store.cas(endpoint_ref)
    close_checks = {
        "selected_manifest_exact": selected["manifest"] == chain[-1]["manifest_ref"],
        "selected_journal_exact": selected["journal"] == chain[-1]["journal_sha256"],
        "label_close": close_cut.get("label") == "close",
        "parent_is_reopen_endpoint": close_cut.get("parent_cut") == endpoint_ref,
        "token_ids_exact": close_cut["token_ids"] == endpoint_cut["token_ids"],
        "pending_token_exact": close_cut["pending"] == endpoint_cut["pending"],
        "logits_payload_exact": close_cut["logits"]["sha256"]
        == endpoint_cut["logits"]["sha256"],
        "vm_checkpoint_exact": close_cut["vm_checkpoint"] == endpoint_cut["vm_checkpoint"],
    }
    all_pairs_exact = all(pair["exact"] for pair in pairs)
    all_candidates_noop = all(
        row["dose"] == 0
        and row["actual_delta_norm"] == 0
        and row["candidate_sha256"] == row["source_activation_sha256"]
        for row in candidates
    )
    status = (
        "valid"
        if all_pairs_exact
        and all_candidates_noop
        and all(close_checks.values())
        and all(identity_checks.values())
        else "invalid"
    )
    result = {
        "schema": "mdb-jspace-recovered-future-parity-v1",
        "mechanics_status": status,
        "origin_job": {
            "job_id": ORIGIN_JOB,
            "terminal_state": "timeout",
            "job_record_path": str(origin_record_path),
            "job_record_sha256": sha256_bytes(origin_record_raw),
            "submitted_request_sha256": origin_record["submitted_request_sha256"],
            "config_sha256": recovery_manifest["config_sha256"],
            "source_sha256": origin_record["config"]["source_sha256"],
        },
        "recovery_job_id": RECOVERY_JOB,
        "scope": "bounded source-bound native versus zero-write runtime mechanics",
        "terminal_claim": "original run remains timeout; no semantic J-edit utility claim",
        "method": {
            "runtime_imports": [],
            "model_loads": 0,
            "store_access": "read-only raw JSON, gzip journal, and CAS digest verification",
            "journal_publication_count": len(chain),
            "pairing": (
                "candidate publication -> distinct act-commit publication -> token cuts 22..33; "
                "native cuts selected by branch, empty lineage, identical tokens, and earlier publication"
            ),
        },
        "source_join": {
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_bytes(protocol_raw),
            "protocol_model_id": protocol["model_id"],
            "worker_sha256": EXPECTED_WORKER_SHA256,
            "recovery_manifest_expected_source_sha256": recovery_manifest["expected_source_sha256"],
            "selected_pointer": selected,
            "selected_manifest_identity": selected_identity,
            "selected_attention_backend": selected_manifest["attention_backend"],
            "identity_checks": identity_checks,
        },
        "zero_candidate_attempts": candidates,
        "publication_segments": segments,
        "pair_count": len(pairs),
        "pairs": pairs,
        "close": {
            "selected_close_cut_ref": close_ref,
            "reopen_endpoint_cut_ref": endpoint_ref,
            "checks": close_checks,
            "exact": all(close_checks.values()),
        },
        "summary": {
            "all_36_pairs_exact": all_pairs_exact and len(pairs) == 36,
            "all_three_zero_candidates_exact_noops": all_candidates_noop and len(candidates) == 3,
            "in_run_and_reopen_bear_attempts_distinct": (
                candidates[1]["attempt_id"] != candidates[2]["attempt_id"]
                and candidates[1]["candidate_publication_journal_sha256"]
                != candidates[2]["candidate_publication_journal_sha256"]
            ),
            "repeated_bear_cut_content_address_expected": (
                segments[1]["endpoint"]["cut_ref"] == segments[2]["endpoint"]["cut_ref"]
            ),
            "selected_close_exact": all(close_checks.values()),
        },
        "result_sha256": None,
    }
    unhashed = {**result, "result_sha256": None}
    result["result_sha256"] = canonical_sha256(unhashed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify(args.recovery.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(raw)
    print(
        json.dumps(
            {
                "mechanics_status": result["mechanics_status"],
                "output": str(args.output),
                "pair_count": result["pair_count"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
