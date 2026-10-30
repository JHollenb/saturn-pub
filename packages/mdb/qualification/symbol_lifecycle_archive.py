"""Verify a collected native project archive without model/framework imports."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import shutil
import sys
import tarfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PurePosixPath


class BlockRuntime(importlib.abc.MetaPathFinder):
    blocked = {"torch", "transformers", "mrun", "saturn", "common", "common_harness"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError("framework/private import during archive audit: " + fullname)
        return None


def read_bounded(path, limit):
    if path.stat().st_size > limit:
        raise ValueError("audit input exceeds byte budget: " + str(path))
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("audit input grew past byte budget: " + str(path))
    return raw


def extract(archive_path, destination, expected):
    raw = read_bounded(archive_path, 64 * 1024 * 1024)
    if len(raw) != expected["bytes"]:
        raise ValueError("archive byte budget or recorded size mismatch")
    if sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("archive seal mismatch")
    if destination.exists():
        raise ValueError("refusing to overwrite an extracted archive")
    with tarfile.open(archive_path, "r:gz") as archive:
        members, names, expanded = [], set(), 0
        for row in archive:
            expanded += row.size
            if len(members) >= 20000 or expanded > 512 * 1024 * 1024:
                raise ValueError("archive expansion exceeds finite audit budget")
            path = PurePosixPath(row.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] not in {"project", "export"}
                    or not (row.isfile() or row.isdir()) or str(path) in names):
                raise ValueError("unsupported archive member: " + row.name)
            names.add(str(path))
            members.append(row)
        destination.mkdir(parents=True)
        for row in members:
            target = destination.joinpath(*PurePosixPath(row.name).parts)
            if row.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(row) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def audit_project(root, report):
    from saturn_pub.values import digest

    from mdb.project import Project
    from mdb.spectral_address import SpectralQuery
    from mdb.symbol_lifecycle import SymbolLifecycle, _spectral_typed_payload

    project = Project(root)
    sid = report["config"]["session_id"]
    selected = project.session(sid)["selected"]

    def no_hydration(*args, **kwargs):
        raise RuntimeError("offline symbol navigation attempted tensor hydration")

    project.store.load = no_hydration
    cuts = project.cut_ids(sid)
    for identifier in cuts:
        project.cut_descriptor(sid, identifier)
    receipts = project.receipts(sid)
    record = project.symbol_lifecycle(sid, report["lifecycle"]["fingerprint"])
    lifecycle = SymbolLifecycle.from_dict(record["lifecycle"])
    if (lifecycle.binding.to_dict() != report["symbol_binding"]
            or [row.to_dict() for row in lifecycle.definitions] != [report["symbol_occurrence"]]
            or [row.to_dict() for row in lifecycle.uses] != report["symbol_uses"]
            or lifecycle.binding.model_identity != report["model_identity"]
            or lifecycle.binding.execution != digest(report["execution"])):
        raise ValueError("collected lifecycle differs from native report evidence")
    if len(lifecycle.uses) != len(report["interventions"]):
        raise ValueError("collected native uses differ from report")
    occurrence = lifecycle.definitions[0]
    source = report["source_context"]["source_parent_cut"]
    if occurrence.source_cut != source or selected != source:
        raise ValueError("collected source or selection differs from native report")
    location = occurrence.locations[0]
    reverse = project.symbols_at(sid, location.slot, cut_id=occurrence.source_cut,
                                selector=location.selector, branch=occurrence.branch)
    if not any(row["occurrence_id"] == occurrence.occurrence_id for row in reverse["matches"]):
        raise ValueError("native reverse lookup lost its definition")
    address = _spectral_typed_payload(occurrence.to_dict()["spectral_address"])
    query = SpectralQuery(
        "collected-identity-audit", address.phases, address.canonical_key.namespace,
        address.canonical_key.symbol_type, address.frame.frame_id,
        exact_symbol_id=lifecycle.symbol, exact_scene_id=address.instance_key.scene_id,
        exact_instance_id=address.instance_key.instance_id,
        exact_role_id=address.role_transform.role_id)
    good = project.resolve_symbol_address(sid, query, cut_id=occurrence.source_cut,
                                          branch=occurrence.branch)
    if good["selected"] is None:
        raise ValueError("native exact spectral lookup abstained")
    for wrong in (replace(query, exact_instance_id="audit-wrong-instance"),
                  replace(query, exact_role_id="audit-wrong-role")):
        if project.resolve_symbol_address(sid, wrong, cut_id=occurrence.source_cut,
                                          branch=occurrence.branch)["selected"] is not None:
            raise ValueError("wrong native identity did not abstain")
    if project.session(sid)["selected"] != selected:
        raise ValueError("offline lookup changed native selection")
    # Hash native page/blob bytes without decoding or importing a tensor runtime.
    byte_files, byte_count = 0, 0
    for directory, suffix in (("pages", ".json"), ("blobs", ".safetensors")):
        for path in sorted((project.store.root / directory).glob("*" + suffix)):
            with path.open("rb") as source_file:
                hashed = sha256()
                while chunk := source_file.read(1024 * 1024):
                    hashed.update(chunk)
            if hashed.hexdigest() != path.stem:
                raise ValueError("native content-addressed byte mismatch: " + path.name)
            byte_files += 1
            byte_count += path.stat().st_size

    def native_file(directory, identifier, suffix):
        if (not isinstance(identifier, str) or len(identifier) != 64
                or set(identifier) - set("0123456789abcdef")):
            raise ValueError("invalid native content address")
        path = project.store.root / directory / (identifier + suffix)
        if not path.is_file():
            raise ValueError("native payload reference is unavailable: " + identifier)
        return path

    for identifier in cuts:
        data = json.loads(read_bounded(native_file("cuts", identifier, ".json"), 4 * 1024 * 1024))
        if data.get("logical_parent") != data["manifest"]["parent"]:
            raise ValueError("native logical parent inventory differs")
        payload = {entry["key"]["value"]: entry["value"]
                   for entry in data["manifest"]["payload"]["items"]}
        if set(data["slots"]) != set(payload):
            raise ValueError("native slot inventory differs from manifest")
        blobs = set()
        for slot, page_id in data["slots"].items():
            page = json.loads(read_bounded(native_file("pages", page_id, ".json"), 4 * 1024 * 1024))
            if page.get("schema") != "saturn-pub-slot-page-v1" or page.get("value") != payload[slot]:
                raise ValueError("native slot page differs from its sealed descriptor")
            blob = page.get("tensor_blob")
            if blob is not None:
                native_file("blobs", blob, ".safetensors")
                blobs.add(blob)
        if sorted(blobs) != data.get("physical_blobs"):
            raise ValueError("native blob inventory does not close")
    return {"lifecycle_id": lifecycle.fingerprint, "cuts": len(cuts),
            "receipts": len(receipts), "definitions": len(lifecycle.definitions),
            "cut_ids": sorted(cuts), "receipt_ids": sorted(row.fingerprint for row in receipts),
            "selected_cut": selected, "payload_reference_closure": True,
            "uses": len(lifecycle.uses), "reverse_matches": len(reverse["matches"]),
            "sas_exact_match": True, "wrong_instance_abstains": True,
            "wrong_role_abstains": True, "selection_unchanged": True,
            "navigation_native_payload_reads": 0, "tensor_hydrations": 0,
            "byte_files_verified": byte_files, "native_bytes_hashed": byte_count,
            "terminal_status": "not-assessed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    report_path = args.result / "report.json"
    raw_report = read_bounded(report_path, 16 * 1024 * 1024)
    report = json.loads(raw_report)
    destination = args.result / "extracted"
    extract(args.result / "native-project-archive.tar.gz", destination,
            report["native_project_archive"])
    sys.meta_path.insert(0, BlockRuntime())
    import saturn_pub

    import mdb

    modules = {name: str(Path(module.__file__).resolve())
               for name, module in (("mdb", mdb), ("saturn_pub", saturn_pub))}
    if any(not Path(path).is_relative_to(Path(sys.prefix).resolve()) for path in modules.values()):
        raise RuntimeError("archive audit did not import installed wheels")
    audits = {name: audit_project(destination / name, report) for name in ("project", "export")}
    if audits["project"] != audits["export"]:
        raise ValueError("collected native project and export differ")
    loaded = sorted({name.split(".")[0] for name in sys.modules} & BlockRuntime.blocked)
    if loaded:
        raise RuntimeError("blocked runtime modules loaded")
    audit = {"schema": "mdb-symbol-lifecycle-collected-audit-v1",
             "source_report_sha256": sha256(raw_report).hexdigest(),
             "archive_sha256": report["native_project_archive"]["sha256"],
             "mechanics_status": "valid", "instrument_status": "valid",
             "trend_status": "observations-only", "terminal_status": "not-assessed",
             "frameworks_loaded": loaded, "installed_modules": modules, "projects": audits,
             "boundary": "Verified metadata and content-addressed bytes; no tensor decode or fresh native forward."}
    (args.result / "offline-audit.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
