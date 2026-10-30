"""Persist and navigate symbol evidence against verified native project cuts.

This is a metadata join. Loading a lifecycle never hydrates tensors, changes the
selected cut, evaluates a lifetime policy, or grants a native write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from saturn_pub.values import digest

from .project_slice import slot_descriptors
from .symbol_lifecycle import SymbolLifecycle, SymbolLifecycleError


def _table(project):
    with project._db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS symbol_lifecycles (
            session_id TEXT NOT NULL, id TEXT NOT NULL, symbol TEXT NOT NULL,
            body TEXT NOT NULL, joins TEXT NOT NULL,
            PRIMARY KEY(session_id, id))""")


def _identifier(value):
    if value is not None and not isinstance(value, str):
        raise SymbolLifecycleError("native reference must be a string")
    return value.removeprefix("cas://sha256/") if value else None


def _supported(binding, cut):
    binding.resolve_metadata(
        model_identity=cut["model_identity"], execution=cut["manifest"]["execution"],
        execution_point=cut["execution_point"], surface=cut["surface"],
        slots=slot_descriptors(cut), context=binding.context,
    )


def _joins(project, sid, lifecycle, *, slices=False):
    session = project.session(sid)
    binding = lifecycle.binding
    if (binding.model_identity != json.loads(session["identity"])
            or binding.execution != digest(json.loads(session["execution"]))):
        raise SymbolLifecycleError("lifecycle belongs to another model or execution")
    cuts, receipts, definitions, uses = {}, {}, [], []
    native_receipts = {r.fingerprint: r.to_dict() for r in project.receipts(sid)}

    def cut(identifier):
        if identifier not in cuts:
            cuts[identifier] = project.cut_descriptor(sid, identifier)
        return cuts[identifier]

    def receipt(row):
        identifier = _identifier(row.receipt)
        if identifier is None:
            return None
        if identifier not in native_receipts:
            raise SymbolLifecycleError("lifecycle receipt is not retained in this session")
        body = native_receipts[identifier]
        if (body.get("model_identity") != binding.model_identity
                or digest(body.get("execution")) != binding.execution):
            raise SymbolLifecycleError("native receipt belongs to another model or execution")
        if body.get("status") == "refused":
            raise SymbolLifecycleError("refused receipt cannot describe a completed symbol event")
        referenced = {body.get(k) for k in ("parent", "result", "candidate")} - {None}
        if row.result_cut and row.result_cut != body.get("result"):
            raise SymbolLifecycleError("symbol result cut differs from its native receipt")
        if getattr(row, "parent_cut", None) and row.parent_cut != body.get("parent"):
            raise SymbolLifecycleError("symbol parent cut differs from its native receipt")
        if row.source_cut and row.source_cut not in referenced:
            # A use may read an earlier ancestor through a declared source reference.
            # That is kept as an association, never synthesized as a receipt parent.
            if row not in lifecycle.uses:
                raise SymbolLifecycleError("definition cut is not referenced by its native receipt")
        receipts[identifier] = body
        return identifier

    for row in (*lifecycle.definitions, *lifecycle.uses):
        if row.session_id != sid:
            raise SymbolLifecycleError("lifecycle event belongs to another project session")
        if row.execution is not None and row.execution != binding.execution:
            raise SymbolLifecycleError("symbol event execution changed")
        for identifier in (row.source_cut, row.result_cut, getattr(row, "parent_cut", None)):
            if identifier:
                cut(identifier)
        if row.clock:
            clock_cut = _identifier(row.clock.metadata.get("source_cut"))
            event_cuts = {row.source_cut, row.result_cut, getattr(row, "parent_cut", None)}
            if clock_cut not in event_cuts | {None}:
                raise SymbolLifecycleError("clock witness refers to another source cut")
            execution = row.clock.metadata.get("model_execution")
            if execution is not None and execution != binding.execution:
                raise SymbolLifecycleError("clock witness belongs to another execution")
            if row.clock.kind == "execution-point" and row.clock.status in {"observed", "validated"}:
                clock_cut = clock_cut or row.source_cut
                if clock_cut is None or row.clock.to_dict()["reading"] != cut(clock_cut)["execution_point"]:
                    raise SymbolLifecycleError("clock reading differs from its native execution point")
            if (row.clock.kind in {"token-position", "token", "depth", "decoder-depth", "layer"}
                    and row.clock.status in {"observed", "validated"}):
                clock_cut = clock_cut or row.source_cut
                field = row.clock.metadata.get("native_point_field")
                point = {} if clock_cut is None else cut(clock_cut)["execution_point"]
                if (not isinstance(field, str) or field not in point
                        or row.clock.to_dict()["reading"] != point[field]):
                    raise SymbolLifecycleError("observed coordinate clock needs its exact native point field")

    for invalidation in lifecycle.to_dict()["invalidations"]:
        clock = invalidation.get("clock")
        if clock is None:
            continue
        clock_cut = _identifier(clock["metadata"].get("source_cut"))
        execution = clock["metadata"].get("model_execution")
        if execution is not None and execution != binding.execution:
            raise SymbolLifecycleError("invalidation clock belongs to another execution")
        coordinate = clock["kind"] in {
            "execution-point", "token-position", "token", "depth", "decoder-depth", "layer"}
        if coordinate and clock["status"] in {"observed", "validated"} and clock_cut is None:
            raise SymbolLifecycleError("observed invalidation coordinate needs a retained native cut")
        if clock_cut is not None:
            native = cut(clock_cut)
            if (clock["kind"] == "execution-point" and clock["status"] in {"observed", "validated"}
                    and clock["reading"] != native["execution_point"]):
                raise SymbolLifecycleError("invalidation clock reading differs from native execution point")
            if (coordinate and clock["kind"] != "execution-point"
                    and clock["status"] in {"observed", "validated"}):
                field = clock["metadata"].get("native_point_field")
                point = native["execution_point"]
                if (not isinstance(field, str) or field not in point or clock["reading"] != point[field]):
                    raise SymbolLifecycleError("invalidation coordinate needs its exact native point field")

    for row in lifecycle.definitions:
        native = cut(row.source_cut)
        _supported(binding, native)
        if not row.locations or any(loc not in binding.locations for loc in row.locations):
            raise SymbolLifecycleError("definition locations are outside the qualified binding")
        item = {"occurrence": row.occurrence_id, "source_cut": row.source_cut,
                "receipt": receipt(row), "support": "metadata-support-verified",
                "slice_basis": "full-root-binding-declared-dataflow"}
        if row.spectral_address:
            spectral = row.to_dict()["spectral_address"]
            if "canonical_key" in spectral:
                if spectral["canonical_key"]["symbol_id"] != lifecycle.symbol:
                    raise SymbolLifecycleError("spectral symbol differs from the registered binding")
                item["spectral_support"] = "typed-identity-verified; supplied-native-association"
            else:
                item["spectral_support"] = "opaque-unjoined-reference"
        if row.vm_bundle:
            bundle = row.to_dict()["vm_bundle"]
            if "schema" in bundle:
                for values, name, expected in (
                    (bundle["model_identity"], "id", binding.model_identity),
                    (bundle["abi_identity"], "execution", binding.execution),
                    (bundle["input_identity"], "source_cut", row.source_cut),
                    (bundle["input_identity"], "context", binding.context),
                ):
                    if name in values and values[name] != expected:
                        raise SymbolLifecycleError("VM bundle differs from the native source binding")
                exact = (bundle["model_identity"] == {"id": binding.model_identity}
                         and bundle["abi_identity"].get("execution") == binding.execution
                         and bundle["input_identity"].get("source_cut") == row.source_cut
                         and bundle["input_identity"].get("context") == binding.context)
                item["vm_support"] = ("native-source-metadata-matched" if exact
                                      else "bundle-seal-verified; source-association-unverified")
            else:
                item["vm_support"] = "opaque-unjoined-reference"
        if slices:
            sliced = project.slice(sid, symbol=binding.symbol, cut_id=row.source_cut,
                                   context=binding.context, binding_id=lifecycle.binding_fingerprint)
            item.update(slice_id=sliced["slice_id"], slice_complete=sliced["complete"],
                        slice_scope=sliced["scope"])
        definitions.append(item)
    for row in lifecycle.uses:
        if row.source_cut:
            _supported(binding, cut(row.source_cut))
        if any(loc not in binding.locations for loc in row.locations):
            raise SymbolLifecycleError("use locations are outside the qualified binding")
        receipt_id = receipt(row)
        closed = bool(receipt_id and row.parent_cut and row.result_cut)
        if receipt_id and not closed and row.status in {"observed", "validated"}:
            raise SymbolLifecycleError("observed symbol use requires native receipt parent/result closure")
        declared_consumers = set()
        for identifier in (row.parent_cut, row.result_cut):
            if identifier:
                consumer = cut(identifier)["surface"].get("consumer")
                if consumer:
                    declared_consumers.add(consumer)
        if closed:
            for step in receipts[receipt_id].get("transitions", []):
                consumer = step.get("native_operation", {}).get("consumer")
                if consumer:
                    declared_consumers.add(consumer)
        if declared_consumers and row.consumer not in declared_consumers:
            raise SymbolLifecycleError("symbol consumer differs from the native contract")
        uses.append({"use": row.use_id, "source_cut": row.source_cut,
                     "result_cut": row.result_cut, "receipt": receipt_id,
                     "consumer": row.consumer,
                     "consumer_support": ("native-contract-matched" if declared_consumers and closed
                                          else "unverified-label"),
                     "receipt_closure": ("native-parent-result-matched" if closed else "incomplete"),
                     "support": "recorded-consumer-association"})
    return {"definitions": definitions, "uses": uses, "cuts": sorted(cuts),
            "receipts": sorted(receipts), "native_payload_reads": 0,
            "external_references": "retained descriptors; external payload availability unverified",
            "authority": "read-only-evidence-index", "causal_effect": "not-established"}


def save_lifecycle(project, sid, lifecycle):
    """Verify the native metadata join, then retain one immutable evidence version."""
    lifecycle = SymbolLifecycle.from_dict(
        lifecycle.to_dict() if isinstance(lifecycle, SymbolLifecycle) else lifecycle)
    lifecycle.verify()
    if not lifecycle.definitions:
        raise SymbolLifecycleError("a project lifecycle needs a native source definition")
    _joins(project, sid, lifecycle)
    # Register the original binding using the project's existing sealed symbol index.
    with project._db() as db:
        project._save_symbols(db, sid, {"symbols": [lifecycle.binding.to_dict()]},
                              json.loads(project.session(sid)["identity"]),
                              json.loads(project.session(sid)["execution"]))
    joins = _joins(project, sid, lifecycle, slices=True)
    _table(project)
    body, joined = project._json(lifecycle.to_dict()), project._json(joins)
    with project._db() as db:
        db.execute("INSERT OR IGNORE INTO symbol_lifecycles VALUES (?, ?, ?, ?, ?)",
                   (sid, lifecycle.fingerprint, lifecycle.symbol, body, joined))
        saved = db.execute("SELECT body FROM symbol_lifecycles WHERE session_id=? AND id=?",
                           (sid, lifecycle.fingerprint)).fetchone()
        if saved[0] != body:
            raise SymbolLifecycleError("symbol lifecycle identity collision")
    return load_lifecycle(project, sid, lifecycle.fingerprint)


def load_lifecycle(project, sid, identifier):
    project.session(sid)
    _table(project)
    with project._db() as db:
        row = db.execute("SELECT body, joins FROM symbol_lifecycles WHERE session_id=? AND id=?",
                         (sid, identifier)).fetchone()
    if row is None:
        raise SymbolLifecycleError("symbol lifecycle is not retained in this session")
    lifecycle = SymbolLifecycle.from_dict(json.loads(row["body"]))
    if lifecycle.fingerprint != identifier:
        raise SymbolLifecycleError("symbol lifecycle index differs from its seal")
    joins = _joins(project, sid, lifecycle)
    recorded = json.loads(row["joins"])
    # Slice links are descriptive, but must still resolve to their original evidence.
    for definition in recorded["definitions"]:
        if definition.get("slice_id"):
            sliced = project.slice_record(sid, definition["slice_id"])
            if (sliced["cut_id"] != definition["source_cut"]
                    or sliced["symbol_binding"]["fingerprint"] != lifecycle.binding_fingerprint
                    or sliced["complete"] != definition.get("slice_complete")
                    or sliced["scope"] != definition.get("slice_scope")):
                raise SymbolLifecycleError("lifecycle slice differs from its native source")
    stripped = [{k: v for k, v in row.items()
                 if k not in {"slice_id", "slice_complete", "slice_scope"}}
                for row in recorded["definitions"]]
    if len(stripped) != len(joins["definitions"]):
        raise SymbolLifecycleError("lifecycle definition join changed")
    expected_definitions = []
    for saved, current in zip(stripped, joins["definitions"]):
        # ``slice_basis`` was added as descriptive metadata after v1 joins were
        # already retained.  A missing historical field remains missing in the
        # returned record, while a present value is still compared exactly.
        if "slice_basis" not in saved:
            current = {key: value for key, value in current.items() if key != "slice_basis"}
        expected_definitions.append(current)
    if stripped != expected_definitions:
        raise SymbolLifecycleError("lifecycle definition join changed")
    if recorded.keys() != joins.keys():
        raise SymbolLifecycleError("lifecycle join contains unsupported fields")
    if len(recorded["uses"]) != len(joins["uses"]):
        raise SymbolLifecycleError("lifecycle use join changed")
    expected_uses = []
    for saved, current in zip(recorded["uses"], joins["uses"]):
        # ``receipt_closure`` was added as descriptive metadata after v1 joins
        # were already retained.  A missing historical field remains missing
        # in the returned record, while a present value is compared exactly.
        if "receipt_closure" not in saved:
            current = {key: value for key, value in current.items()
                       if key != "receipt_closure"}
        expected_uses.append(current)
    if recorded["uses"] != expected_uses:
        raise SymbolLifecycleError("lifecycle use join changed")
    for field in joins.keys() - {"definitions"}:
        if field == "uses":
            continue
        if joins[field] != recorded[field]:
            raise SymbolLifecycleError("lifecycle native reference join changed")
    return {"lifecycle_id": identifier, "lifecycle": lifecycle.to_dict(), "joins": recorded}


def list_lifecycles(project, sid, *, symbol=None, cut_id=None, branch=None):
    project.session(sid)
    if cut_id is not None:
        project.cut_descriptor(sid, cut_id)
    _table(project)
    with project._db() as db:
        rows = db.execute("SELECT id FROM symbol_lifecycles WHERE session_id=? ORDER BY rowid",
                          (sid,)).fetchall()
    result = []
    for row in rows:
        record = load_lifecycle(project, sid, row["id"])
        body = record["lifecycle"]
        if symbol is not None and body["symbol"] != symbol:
            continue
        events = body["definitions"] + body["uses"]
        if cut_id is not None or branch is not None:
            events = [event for event in events
                      if (cut_id is None or event["source_cut"] == cut_id)
                      and (branch is None or event["branch"] == branch)]
            if not events:
                continue
        result.append(record)
    return result


def symbols_at(project, sid, port, *, cut_id=None, selector=None, branch=None):
    """Reverse exact recorded locations; no semantic inference from tensor bytes."""
    cut_id = cut_id or project.session(sid)["selected"]
    native = project.cut_descriptor(sid, cut_id)
    rows = list_lifecycles(project, sid, cut_id=cut_id, branch=branch)
    matches = []
    for record in rows:
        lifecycle = SymbolLifecycle.from_dict(record["lifecycle"])
        for occurrence in lifecycle.definitions:
            if occurrence.source_cut != cut_id or (branch is not None and occurrence.branch != branch):
                continue
            _supported(lifecycle.binding, native)
            for location in occurrence.locations:
                if location.slot != port or (selector is not None and tuple(selector) != location.selector):
                    continue
                matches.append({"symbol": lifecycle.symbol,
                    "binding_fingerprint": lifecycle.binding_fingerprint,
                    "lifecycle_id": lifecycle.fingerprint,
                    "occurrence_id": occurrence.occurrence_id, "branch": occurrence.branch,
                    "location": location.to_dict(), "source_cut": cut_id,
                    "spectral_address": occurrence.to_dict()["spectral_address"],
                    "clock": None if occurrence.clock is None else occurrence.clock.to_dict(),
                    "validity_status": occurrence.validity_status,
                    "invalidations": record["lifecycle"]["invalidations"],
                    "support": "metadata-support-verified"})
    return {"session_id": sid, "cut_id": cut_id, "port": port, "matches": matches,
            "coverage": "registered locations only; no match is not evidence of absence",
            "native_payload_reads": 0, "authority": "read-only-evidence-index"}


def resolve_address(project, sid, query, *, cut_id=None, branch=None):
    """Use the existing exact SAS kernel to select a registered native occurrence."""
    from dataclasses import asdict

    from .spectral_address import (
        SpectralBinding,
        SpectralQuery,
        SpectralReadKernelSpec,
        SupportResult,
        read_bindings,
    )
    from .symbol_lifecycle import _spectral_typed_payload

    query = query if isinstance(query, SpectralQuery) else SpectralQuery(**query)
    if not all((query.exact_symbol_id, query.exact_scene_id,
                query.exact_instance_id, query.exact_role_id)):
        raise SymbolLifecycleError("symbol address lookup requires complete exact identity and role")
    cut_id = cut_id or project.session(sid)["selected"]
    candidates, records, gaps = {}, {}, []
    for record in list_lifecycles(project, sid, cut_id=cut_id, branch=branch):
        for occurrence in record["lifecycle"]["definitions"]:
            if occurrence["source_cut"] != cut_id or (branch and occurrence["branch"] != branch):
                continue
            spectral = occurrence["spectral_address"]
            if not spectral or "canonical_key" not in spectral:
                continue
            address = _spectral_typed_payload(spectral)
            if address.role_transform.arity != 1 or address.role_transform.argument_index != 0:
                gaps.append({"occurrence": occurrence["occurrence_id"],
                             "reason": "compound-address-needs-ordered-binding"})
                continue
            reference = "occurrence://sha256/" + occurrence["fingerprint"]
            candidates[reference] = SpectralBinding(
                occurrence["fingerprint"], "mdb.symbol-native-location-v1",
                ((address.role_transform.role_id, address),), reference, address.frame)
            records[reference] = {"lifecycle_id": record["lifecycle_id"],
                                  "binding": record["lifecycle"]["binding"],
                                  "occurrence": occurrence,
                                  "statuses": record["lifecycle"]["statuses"],
                                  "invalidations": record["lifecycle"]["invalidations"],
                                  "validity_policy": "recorded-evidence-only; no automatic exclusion"}
    support = SupportResult(
        symbol_registered=bool(candidates), semantic_descriptor_in_support=True,
        role_signature_valid=True, instance_scope_valid=True,
        operator_available=False, spatial_transform_supported=False,
        coordinate_frame_supported=True, backend_lowering_supported=False,
        numerical_profile_supported=True, consumer_certified=False)
    result = read_bindings(query, tuple(candidates.values()),
                           SpectralReadKernelSpec("mdb-symbol-exact-identity-v1", metric="exact_identity"),
                           support)
    selected = (None if result.selected_binding is None
                else records[result.selected_binding.payload_reference])
    report = {"session_id": sid, "cut_id": cut_id, "query": asdict(query),
              "selected": selected, "abstain_reason": result.abstention_reason,
              "trace": asdict(result.trace), "support": asdict(support), "gaps": gaps,
              "native_payload_reads": 0, "authority": "read-only-evidence-index",
              "interpretation": "exact registered identity; no semantic inference or write admission"}
    return {"read_id": digest(report), **report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("action", choices=("add", "list", "show", "at", "resolve"))
    parser.add_argument("target", nargs="?")
    parser.add_argument("--session-id")
    parser.add_argument("--symbol")
    parser.add_argument("--cut")
    parser.add_argument("--branch")
    parser.add_argument("--selector", type=json.loads)
    args = parser.parse_args(argv)
    from .project import Project

    project = Project(args.root)
    sid = args.session_id
    if sid is None:
        sessions = project.sessions()
        if len(sessions) != 1:
            parser.error("select --session-id for a project with zero or multiple sessions")
        sid = sessions[0]["session_id"]
    if args.action == "add":
        if not args.target:
            parser.error("add requires a lifecycle JSON file")
        result = save_lifecycle(project, sid, SymbolLifecycle.load(args.target))
    elif args.action == "show":
        if not args.target:
            parser.error("show requires a lifecycle ID")
        result = load_lifecycle(project, sid, args.target)
    elif args.action == "resolve":
        if not args.target:
            parser.error("resolve requires a SpectralQuery JSON file")
        result = resolve_address(project, sid, json.loads(Path(args.target).read_text()),
                                 cut_id=args.cut, branch=args.branch)
    elif args.action == "at":
        if not args.target:
            parser.error("at requires a port")
        result = symbols_at(project, sid, args.target, cut_id=args.cut,
                            selector=args.selector, branch=args.branch)
    else:
        result = list_lifecycles(project, sid, symbol=args.symbol,
                                cut_id=args.cut, branch=args.branch)
    print(json.dumps(result, indent=2))
    return 0
