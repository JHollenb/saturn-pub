"""Persistent investigations over public cuts, MDB graphs and annotation sidecars.

Native execution and cut bytes remain owned by saturn_pub. SQLite is an index,
not an alternative checkpoint format; graph queries reuse StateGraph.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

from saturn_pub.core import Receipt, Session
from saturn_pub.store import LocalStore
from saturn_pub.symbols import SymbolBinding
from saturn_pub.values import canonical, describe, digest

from .state_annotations import StateAnnotationStore
from .state_graph import GraphSnapshot, NodeBinding, StateGraph


class Project:
    """A multi-session investigation, reopenable without a model or framework."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = LocalStore(self.root / "store")
        with self._db() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3):
                raise ValueError("unsupported MDB project schema")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, identity TEXT NOT NULL, execution TEXT NOT NULL,
                    selected TEXT NOT NULL, debug_info TEXT NOT NULL, revision INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS cuts (
                    session_id TEXT NOT NULL, id TEXT NOT NULL, manifest TEXT NOT NULL,
                    PRIMARY KEY (session_id, id));
                CREATE TABLE IF NOT EXISTS receipts (
                    session_id TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY (session_id, id));
                CREATE TABLE IF NOT EXISTS imports (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, analyzer TEXT NOT NULL,
                    version TEXT NOT NULL, input_digest TEXT NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS analysis_inputs
                    ON analyses(session_id, analyzer, version, input_digest);
                CREATE TABLE IF NOT EXISTS symbols (
                    session_id TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(session_id, id));
                CREATE TABLE IF NOT EXISTS slices (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, body TEXT NOT NULL);
                PRAGMA user_version = 3;
            """)
            for row in db.execute("SELECT * FROM sessions").fetchall():
                self._save_symbols(
                    db,
                    row["id"],
                    json.loads(row["debug_info"]),
                    json.loads(row["identity"]),
                    json.loads(row["execution"]),
                )
        for session in self.sessions():
            self.analyze(session["session_id"])

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.root / "project.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _json(value):
        return canonical(value).decode("utf-8")

    def save_session(
        self, session_id, session, cuts, *, debug_info=None, receipts=(), expected_revision=0
    ):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("project session needs a nonempty ID")
        selected = session.capture(retain=False)
        retained = {cut.fingerprint: cut for cut in [*cuts, selected]}
        # Immutable objects are published first; a failed index transaction can
        # leave harmless unindexed objects, never a reference to missing bytes.
        for cut in retained.values():
            if cut.model_identity != session.adapter.model_identity or dict(cut.execution) != dict(
                session.adapter.execution
            ):
                raise ValueError("project cut belongs to a different model or execution")
            self.store.save(cut)
        native_receipts = {
            r.fingerprint: r for r in [*session.receipts, *session.attempts, *receipts]
        }
        for receipt in native_receipts.values():
            self.store.receipt(receipt)
        identity = self._json(session.adapter.model_identity)
        execution = self._json(dict(session.adapter.execution))
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if old and (old["identity"] != identity or old["execution"] != execution):
                raise ValueError("project session ID is bound to another model or execution")
            if (old["revision"] if old else 0) != expected_revision:
                raise ValueError(
                    "project session changed in another writer; reopen before continuing"
                )
            revision = expected_revision + 1
            self._save_symbols(
                db,
                session_id,
                debug_info or {},
                session.adapter.model_identity,
                dict(session.adapter.execution),
            )
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET selected=excluded.selected, "
                "debug_info=excluded.debug_info, revision=excluded.revision",
                (
                    session_id,
                    identity,
                    execution,
                    selected.fingerprint,
                    self._json(debug_info or {}),
                    revision,
                ),
            )
            for cut in retained.values():
                body = self._json(cut.manifest())
                db.execute(
                    "INSERT OR IGNORE INTO cuts VALUES (?, ?, ?)",
                    (session_id, cut.fingerprint, body),
                )
                stored = db.execute(
                    "SELECT manifest FROM cuts WHERE session_id=? AND id=?",
                    (session_id, cut.fingerprint),
                ).fetchone()[0]
                if stored != body:
                    raise ValueError("project cut index identity collision")
            for identifier, receipt in native_receipts.items():
                body = self._json(receipt.to_dict())
                db.execute(
                    "INSERT OR IGNORE INTO receipts VALUES (?, ?, ?)",
                    (session_id, identifier, body),
                )
                stored = db.execute(
                    "SELECT body FROM receipts WHERE session_id=? AND id=?",
                    (session_id, identifier),
                ).fetchone()[0]
                if stored != body:
                    raise ValueError("project receipt index identity collision")
        return revision

    def _save_symbols(self, db, sid, info, identity, execution):
        for value in info.get("symbols", []):
            binding = SymbolBinding.from_dict(value)
            if binding.model_identity != identity or binding.execution != digest(execution):
                raise ValueError("project symbol belongs to another model or execution")
            body = self._json(binding.to_dict())
            db.execute(
                "INSERT OR IGNORE INTO symbols VALUES (?, ?, ?)", (sid, value["fingerprint"], body)
            )
            saved = db.execute(
                "SELECT body FROM symbols WHERE session_id=? AND id=?", (sid, value["fingerprint"])
            ).fetchone()[0]
            if saved != body:
                raise ValueError("project symbol identity collision")

    def symbols(self, session_id):
        session = self.session(session_id)
        with self._db() as db:
            rows = db.execute(
                "SELECT id, body FROM symbols WHERE session_id=? ORDER BY rowid", (session_id,)
            ).fetchall()
        result = []
        for row in rows:
            binding = SymbolBinding.from_dict(json.loads(row["body"]))
            if binding.to_dict()["fingerprint"] != row["id"]:
                raise ValueError("project symbol index differs from its seal")
            if binding.model_identity != json.loads(
                session["identity"]
            ) or binding.execution != digest(json.loads(session["execution"])):
                raise ValueError("project symbol differs from its session binding")
            result.append(binding.to_dict())
        return result

    def slice(
        self,
        session_id,
        *,
        port=None,
        symbol=None,
        cut_id=None,
        to_cut=None,
        context=None,
        binding_id=None,
        receipt_id=None,
    ):
        from .project_slice import slice_project

        return slice_project(
            self,
            session_id,
            port=port,
            symbol=symbol,
            cut_id=cut_id,
            to_cut=to_cut,
            context=context,
            binding_id=binding_id,
            receipt_id=receipt_id,
        )

    def slice_record(self, session_id, identifier):
        self.session(session_id)
        with self._db() as db:
            row = db.execute(
                "SELECT body FROM slices WHERE session_id=? AND id=?", (session_id, identifier)
            ).fetchone()
        if row is None:
            raise ValueError("slice is not retained in this project session")
        body = json.loads(row[0])
        if digest(body) != identifier or body["session_id"] != session_id:
            raise ValueError("project slice seal changed")
        GraphSnapshot.from_dict(body["graph"]).verify()
        return {"slice_id": identifier, **body}

    def slice_ids(self, session_id):
        self.session(session_id)
        with self._db() as db:
            return [
                r[0]
                for r in db.execute(
                    "SELECT id FROM slices WHERE session_id=? ORDER BY rowid", (session_id,)
                )
            ]

    def save_symbol_lifecycle(self, session_id, lifecycle):
        from .symbol_lifecycle_project import save_lifecycle

        return save_lifecycle(self, session_id, lifecycle)

    def symbol_lifecycle(self, session_id, identifier):
        from .symbol_lifecycle_project import load_lifecycle

        return load_lifecycle(self, session_id, identifier)

    def symbol_lifecycles(self, session_id, *, symbol=None, cut_id=None, branch=None):
        from .symbol_lifecycle_project import list_lifecycles

        return list_lifecycles(self, session_id, symbol=symbol, cut_id=cut_id, branch=branch)

    def symbols_at(self, session_id, port, *, cut_id=None, selector=None, branch=None):
        from .symbol_lifecycle_project import symbols_at

        return symbols_at(self, session_id, port, cut_id=cut_id, selector=selector, branch=branch)

    def resolve_symbol_address(self, session_id, query, *, cut_id=None, branch=None):
        from .symbol_lifecycle_project import resolve_address

        return resolve_address(self, session_id, query, cut_id=cut_id, branch=branch)

    def sessions(self):
        with self._db() as db:
            return [
                {
                    "session_id": r["id"],
                    "model_identity": json.loads(r["identity"]),
                    "execution": json.loads(r["execution"]),
                    "selected_cut": r["selected"],
                }
                for r in db.execute("SELECT * FROM sessions ORDER BY id")
            ]

    def session(self, session_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown project session: {session_id}")
        return dict(row)

    def cut_ids(self, session_id):
        self.session(session_id)
        with self._db() as db:
            return [
                r[0]
                for r in db.execute(
                    "SELECT id FROM cuts WHERE session_id=? ORDER BY rowid", (session_id,)
                )
            ]

    def load_cut(self, session_id, cut_id, *, device="cpu"):
        with self._db() as db:
            row = db.execute(
                "SELECT manifest FROM cuts WHERE session_id=? AND id=?", (session_id, cut_id)
            ).fetchone()
        if row is None:
            raise ValueError("cut is not retained in this project session")
        cut = self.store.load(cut_id, device=device)
        if self._json(cut.manifest()) != row[0]:
            raise ValueError("project cut index differs from native cut")
        return cut

    def cut_descriptor(self, session_id, cut_id):
        """Verify native cut metadata without hydrating tensor pages."""
        with self._db() as db:
            row = db.execute(
                "SELECT manifest FROM cuts WHERE session_id=? AND id=?", (session_id, cut_id)
            ).fetchone()
        if row is None:
            raise ValueError("cut is not retained in this project session")
        manifest = self.store.inspect(cut_id)
        if self._json(manifest) != row[0]:
            raise ValueError("project cut index differs from native cut")
        session = self.session(session_id)
        if manifest["model_identity"] != json.loads(session["identity"]) or manifest[
            "execution"
        ] != json.loads(session["execution"]):
            raise ValueError("native cut differs from the project session binding")
        metadata = json.loads((self.store.root / "cuts" / f"{cut_id}.json").read_bytes())
        point = metadata.get("execution_point")
        if describe(point) != manifest.get("execution_point"):
            raise ValueError("native execution point differs from its sealed manifest")
        surface = metadata.get("surface")
        if (None if surface is None else describe(surface)) != manifest.get("surface"):
            raise ValueError("native surface differs from its sealed manifest")
        return {
            "fingerprint": cut_id,
            "manifest": manifest,
            "model_identity": manifest["model_identity"],
            "parent": manifest["parent"],
            "boundary": manifest["boundary"],
            "execution_point": point or {},
            "surface": surface or {},
        }

    def components(self, session_id, *, query=None):
        """List recorded ports, structural operations and qualified symbols."""
        from .project_navigation import components

        return components(self, session_id, query=query)

    def xrefs(self, session_id, component, *, relation=None):
        """Query declared dependencies, never inferred causal effects."""
        from .project_navigation import xrefs

        return xrefs(self, session_id, component, relation=relation)

    def component(self, session_id, name):
        from .project_navigation import component

        return component(self, session_id, name)

    def analyze(self, session_id, *, analyzer="structure", version=None):
        from .project_analysis import analyze

        return analyze(self, session_id, analyzer=analyzer, version=version)

    def analyses(self, session_id):
        from .project_analysis import history

        return history(self, session_id)

    def trace(self, session_id, component, *, direction="upstream", depth=4):
        from .project_analysis import walk

        return walk(self, session_id, component, direction=direction, depth=depth)

    def backtrace(self, session_id, port, *, cut_id=None):
        from .project_analysis import native_backtrace

        return native_backtrace(self, session_id, port, cut_id=cut_id)

    def finding(self, session_id, component, title, **fields):
        from .project_analysis import finding

        return finding(self, session_id, component, title, **fields)

    def findings(self, session_id):
        from .project_analysis import findings

        return findings(self, session_id)

    def watch_events(self, session_id):
        events = json.loads(self.session(session_id)["debug_info"]).get("watch_events", [])
        for event in events:
            body = {
                key: value
                for key, value in event.items()
                if key not in {"fingerprint", "boundary", "authority"}
            }
            if digest(body) != event["fingerprint"]:
                raise ValueError("project watch event seal changed")
        return events

    def receipts(self, session_id):
        self.session(session_id)
        with self._db() as db:
            rows = list(
                db.execute(
                    "SELECT id, body FROM receipts WHERE session_id=? ORDER BY rowid", (session_id,)
                )
            )
        result = []
        for row in rows:
            data = json.loads(row["body"])
            fingerprint = data.pop("fingerprint")
            receipt = Receipt(canonical(data))
            if receipt.fingerprint != fingerprint or fingerprint != row["id"]:
                raise ValueError("project receipt seal changed")
            raw = (self.store.root / "receipts" / f"{fingerprint}.json").read_bytes()
            if raw != canonical(receipt.to_dict()):
                raise ValueError("native receipt bytes differ from the project index")
            result.append(receipt)
        return result

    def resume(self, session_id, adapter, *, max_steps=1000):
        """Rehydrate only through the public adapter's exact compatibility checks."""
        from .saturn_pub_adapter import SaturnPubDebuggerAdapter

        row = self.session(session_id)
        device = str(getattr(adapter, "device", "cpu"))
        cut = self.load_cut(session_id, row["selected"], device=device)
        native = Session.from_cut(adapter, cut)
        native.receipts = self.receipts(session_id) + native.receipts
        return SaturnPubDebuggerAdapter(
            native, project=self, session_id=session_id, max_steps=max_steps, resume=True
        )

    def _annotations(self, session_id, *, writable=False):
        row = self.session(session_id)
        identity = {
            "model_id": json.loads(row["identity"]),
            "execution": json.loads(row["execution"]),
        }
        registry = {identifier: {} for identifier in self.cut_ids(session_id)}

        def descriptor(identifier):
            # This is an annotation view, never an MDB-native checkpoint.
            cut = self.cut_descriptor(session_id, identifier)
            point = dict(cut["execution_point"])
            cursor = {
                **point,
                "index": point.get("index")
                if point.get("index") is not None
                else point.get("logical_step", 0),
            }
            return {
                "identity": identity,
                "cursor": cursor,
                "parent_cut": cut["parent"],
                "public_cut": cut["manifest"],
            }

        return StateAnnotationStore(
            self.root / "sessions" / digest(session_id),
            session_binding={
                "session_id": session_id,
                "family": "saturn-pub",
                "lane": "public-session",
                "model_identity_sha256": digest(identity),
                "publication_mode": "project",
            },
            cut_loader=descriptor,
            cut_registry=registry,
            write_enabled=writable,
        )

    def annotate(
        self, session_id, label, *, cut_id=None, role="observation", stage="exploration", **fields
    ):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.session(session_id)
            return self._annotations(session_id, writable=True).annotate(
                label, cut_id=cut_id or row["selected"], role=role, stage=stage, **fields
            )

    def annotations(self, session_id=None, **filters):
        sessions = [session_id] if session_id else [r["session_id"] for r in self.sessions()]
        return [
            {"session_id": sid, **row}
            for sid in sessions
            for row in self._annotations(sid).list(**filters)
        ]

    def import_graph(self, snapshot):
        """Index an existing graph/annotation export without changing its identities.

        This imports descriptive history, not the referenced native state bytes.
        A legacy Bench can supply StateGraph.from_bench(bench).seal().
        """
        if isinstance(snapshot, StateGraph):
            snapshot = snapshot.seal()
        if not isinstance(snapshot, GraphSnapshot):
            snapshot = GraphSnapshot.from_dict(snapshot)
        snapshot.verify()
        body = self._json(snapshot.to_dict())
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO imports VALUES (?, ?)", (snapshot.snapshot_id, body))
            stored = db.execute(
                "SELECT body FROM imports WHERE id=?", (snapshot.snapshot_id,)
            ).fetchone()[0]
            if stored != body:
                raise ValueError("imported graph identity collision")
        return snapshot.snapshot_id

    def graph(self, session_id=None):
        graph = StateGraph(
            metadata={"source": "mdb-project", "execution_authority": "native-runtime"}
        )
        if session_id is None:
            with self._db() as db:
                imports = list(db.execute("SELECT id, body FROM imports ORDER BY id"))
            for row in imports:
                snapshot = GraphSnapshot.from_dict(json.loads(row["body"]))
                if snapshot.snapshot_id != row["id"]:
                    raise ValueError("imported project graph seal changed")
                for node in snapshot.nodes:
                    graph.add_node(node)
                for edge in snapshot.edges:
                    graph.add_edge(
                        edge.source,
                        edge.target,
                        edge.relation,
                        graph=edge.graph,
                        evidence=edge.evidence,
                        binding=edge.binding,
                        metadata=edge.metadata,
                    )
                source = graph.add_source(
                    source_ref=snapshot.ref,
                    binding=snapshot.binding,
                    metadata={
                        "imported_snapshot": snapshot.snapshot_id,
                        "binding": snapshot.binding.to_dict(),
                    },
                )
                for node in snapshot.nodes:
                    graph.add_edge(source, node.node_id, "describes", graph="knowledge")
        sessions = [session_id] if session_id else [r["session_id"] for r in self.sessions()]
        for sid in sessions:
            row = self.session(sid)
            nodes = {}
            cuts = {}
            for identifier in self.cut_ids(sid):
                cut = self.cut_descriptor(sid, identifier)
                cuts[identifier] = cut
                nodes[identifier] = graph.add_state(
                    cut_ref=f"cas://sha256/{identifier}",
                    fingerprint=identifier,
                    binding=NodeBinding(
                        model=cut["model_identity"], session=sid, clock=dict(cut["execution_point"])
                    ),
                    metadata={"native_format": "saturn-pub", "boundary": cut["boundary"]},
                )
            for identifier, cut in cuts.items():
                if cut["parent"] in nodes:
                    graph.add_edge(
                        nodes[cut["parent"]], nodes[identifier], "continues", evidence="observed"
                    )
            for receipt in self.receipts(sid):
                data = receipt.to_dict()
                parent, result = data.get("parent"), data.get("result")
                # Restore/abort revisit retained state: record the transaction
                # without inventing a cyclic native execution dependency.
                output = (
                    nodes.get(result)
                    if data.get("operation") not in {"restore", "abort"} and result != parent
                    else None
                )
                inputs = [nodes[parent]] if parent in nodes else []
                if data.get("operation") in {"commit", "abort"}:
                    candidate = nodes.get(data.get("candidate"))
                    if candidate and candidate not in inputs:
                        inputs.append(candidate)
                action = graph.add_action(
                    act_ref=f"cas://sha256/{receipt.fingerprint}",
                    inputs=inputs,
                    output=output,
                    receipt_refs=(f"cas://sha256/{receipt.fingerprint}",),
                    binding=NodeBinding(model=json.loads(row["identity"]), session=sid),
                    metadata={"native_receipt": data},
                    evidence="observed",
                )
                if data.get("operation") == "restore" and result in nodes:
                    # Selection revisits an older cut. Preserve the reference
                    # outside computation lineage, where it would form a cycle.
                    graph.add_edge(
                        action,
                        nodes[result],
                        "describes",
                        graph="knowledge",
                        evidence="observed",
                        metadata={"role": "restore-target"},
                    )
            slice_annotations = []
            for entry in self._annotations(sid).list():
                annotation = self._annotations(sid).show(entry["annotation_id"])
                source = graph.add_source(
                    source_ref=entry["annotation_id"],
                    labels=(entry["label"],),
                    binding=NodeBinding(session=sid),
                    metadata=annotation,
                )
                graph.add_edge(source, nodes[entry["cut_id"]], "annotates", graph="knowledge")
                if annotation.get("details", {}).get("slice_id"):
                    slice_annotations.append((source, annotation["details"]["slice_id"]))
            symbol_nodes = {}
            for binding in self.symbols(sid):
                binding = SymbolBinding.from_dict(binding).to_dict()
                symbol_nodes[binding["fingerprint"]] = graph.add_source(
                    source_ref=f"symbol://sha256/{binding['fingerprint']}",
                    binding=NodeBinding(session=sid),
                    metadata=binding,
                    labels=(binding["symbol"],),
                )
            for identifier in self.slice_ids(sid):
                sliced = self.slice_record(sid, identifier)
                source = graph.add_source(
                    source_ref=f"slice://sha256/{identifier}",
                    binding=NodeBinding(session=sid),
                    labels=(sliced["component"],),
                    metadata={
                        "slice_id": identifier,
                        "graph_snapshot": sliced["graph"]["snapshot_id"],
                        "support": sliced["support"],
                        "complete": sliced["complete"],
                    },
                )
                graph.add_edge(source, nodes[sliced["cut_id"]], "describes", graph="knowledge")
                if sliced["symbol_binding"]:
                    symbol_node = symbol_nodes[sliced["symbol_binding"]["fingerprint"]]
                    graph.add_edge(symbol_node, source, "describes", graph="knowledge")
                for annotation_node, slice_id in slice_annotations:
                    if slice_id == identifier:
                        graph.add_edge(annotation_node, source, "related_to", graph="knowledge")
            for event in self.watch_events(sid):
                output = graph.add_output(
                    output_ref=f"watch://sha256/{event['fingerprint']}",
                    binding=NodeBinding(model=json.loads(row["identity"]), session=sid),
                    metadata=event,
                    evidence="observed",
                    labels=("diagnostic-watch",),
                )
                if event["cut"] in nodes:
                    graph.add_edge(
                        output,
                        nodes[event["cut"]],
                        "describes",
                        graph="knowledge",
                        evidence="observed",
                    )
        graph.verify()
        return graph


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a persistent MDB investigation")
    parser.add_argument("root", type=Path)
    from .project_navigation import add_arguments, execute

    add_arguments(parser, extra_actions=("sessions", "graph", "annotations"))
    parser.add_argument("--session-id")
    args = parser.parse_args(argv)
    try:
        project = Project(args.root)
        if args.action in {"sessions", "graph", "annotations"}:
            result = (
                project.sessions()
                if args.action == "sessions"
                else project.annotations(args.session_id)
                if args.action == "annotations"
                else project.graph(args.session_id).seal().to_dict()
            )
        else:
            options = vars(args).copy()
            options.pop("root")
            sid = options.pop("session_id")
            result = execute(project, sid, **options)
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"mdb project: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
