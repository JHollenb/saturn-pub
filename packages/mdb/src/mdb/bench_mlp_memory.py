"""Bench Acts that install and remove a linear MLP memory overlay (see ``mlp_memory``).

An overlay changes the effective model for every forward computed while it is live. Install
and remove are recorded as Acts in the active branch lineage; every cut taken while an overlay
is live records its digest, and a cut can only be restored under the overlay it was cut under.
History committed before an install keeps the K/V rows the native model wrote, so the overlay
takes effect from the install point onward. Removal is exact: the probe logits after removal
must equal the native probe logits recorded at install, bit for bit.
"""
from __future__ import annotations

import os
from pathlib import Path

from mdb.errors import BenchError
from mdb.mlp_memory import ACT_SCHEMA, probe_digest
from mdb.model_os_contracts import content_digest

PROBE_TOKENS = 16


def _boundary(bench):
    bench._check()
    if bench._edit:
        raise BenchError("commit or abort the pending Act first")
    if bench.branches and bench._backend().execution is not None:
        raise BenchError("MLP memory Acts require a committed token boundary")


def _probe_ids(bench):
    if bench.branches and bench.branch.token_ids:
        return [int(t) for t in bench.branch.token_ids[:PROBE_TOKENS]]
    return [i % bench.model.config.vocab_size for i in range(4)]


def _commit(bench, act, label):
    act["act_id"] = content_digest(act)
    act_ref = bench.artifacts.put(act)
    bench.events.append({"kind": f"mlp-memory-{act['kind']}", "act_id": act["act_id"],
                         "act_ref": act_ref, "memory_digest": act["memory_digest"]})
    cut = None
    if bench.branches:
        bench.branch.intervention_lineage.append(act["act_id"])
        cut = bench.checkpoint(label)
    else:
        bench._publish()
    return act_ref, cut


def install(bench, memory):
    _boundary(bench)
    if bench._mlp_memory is not None:
        raise BenchError("remove the live MLP memory overlay first")
    if memory.base_identity not in (None, bench.identity["weights_sha256"]):
        raise BenchError("MLP memory was fitted on different base weights")
    digest = memory.digest
    rel = f"mlp-memory/{digest}.safetensors"
    path = bench.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        memory.save(tmp)
        os.replace(tmp, path)
    probe = _probe_ids(bench)
    native = probe_digest(bench.model, probe)
    lineage = list(bench.branch.intervention_lineage) if bench.branches else None
    events, cuts = list(bench.events), dict(bench.cuts)
    installation = memory.install(bench.model)
    try:
        act = {"schema": ACT_SCHEMA, "kind": "install", "memory_digest": digest,
               "memory": memory.manifest(), "path": rel,
               "parent_cut": bench.branch.cut if bench.branches else None,
               "base_identity": bench.identity, "probe_ids": probe,
               "native_probe_sha256": native,
               "installed_probe_sha256": probe_digest(bench.model, probe)}
        act_id = content_digest(act)
        bench._mlp_memory = (memory, installation, rel, None)
        act_ref, cut = _commit_install(bench, act, act_id)
        return {**act, "act_id": act_id, "act_ref": act_ref, "cut_id": cut}
    except BaseException:
        installation.remove()
        bench._mlp_memory = None
        bench.events, bench.cuts = events, cuts
        if lineage is not None:
            bench.branch.intervention_lineage = lineage
        raise


def _commit_install(bench, act, act_id):
    act_ref = bench.artifacts.put({**act, "act_id": act_id})
    memory, installation, rel, _ = bench._mlp_memory
    bench._mlp_memory = (memory, installation, rel, act_ref)
    bench.events.append({"kind": "mlp-memory-install", "act_id": act_id, "act_ref": act_ref,
                         "memory_digest": act["memory_digest"]})
    if bench.branches:
        bench.branch.intervention_lineage.append(act_id)
        return act_ref, bench.checkpoint("mlp-memory-installed")
    bench._publish()
    return act_ref, None


def remove(bench):
    _boundary(bench)
    if bench._mlp_memory is None:
        raise BenchError("no MLP memory overlay is live")
    memory, installation, rel, install_ref = bench._mlp_memory
    installed = bench.artifacts.get(install_ref)
    probe = installed["probe_ids"]
    live = probe_digest(bench.model, probe)
    installation.remove()
    native = probe_digest(bench.model, probe)
    if native != installed["native_probe_sha256"]:
        raise BenchError("MLP memory removal did not restore the native probe logits; "
                         "the resident model changed while the overlay was live")
    lineage = list(bench.branch.intervention_lineage) if bench.branches else None
    events, cuts = list(bench.events), dict(bench.cuts)
    bench._mlp_memory = None
    act = {"schema": ACT_SCHEMA, "kind": "remove", "memory_digest": installation.digest,
           "install_act": installed["act_id"],
           "parent_cut": bench.branch.cut if bench.branches else None,
           "probe_ids": probe, "live_probe_sha256": live, "native_probe_sha256": native,
           "exact_undo": True}
    try:
        act_ref, cut = _commit(bench, act, "mlp-memory-removed")
        return {**act, "act_ref": act_ref, "cut_id": cut}
    except BaseException:
        bench._mlp_memory = (memory, memory.install(bench.model), rel, install_ref)
        bench.events, bench.cuts = events, cuts
        if lineage is not None:
            bench.branch.intervention_lineage = lineage
        raise


def reinstall(bench, info):
    """Resume support: reload the persisted overlay named by the session manifest."""
    from mdb.memit import load_overlay
    memory = load_overlay(Path(bench.root) / info["path"], expected_digest=info["digest"])
    bench._mlp_memory = (memory, memory.install(bench.model), info["path"], info["act_ref"])


def verify_act(act):
    """Check a frozen MLP memory Act's self-consistency."""
    if not isinstance(act, dict) or act.get("schema") != ACT_SCHEMA or act.get("kind") not in {
            "install", "remove"}:
        raise ValueError("MLP memory Act schema changed")
    body = {k: v for k, v in act.items() if k not in {"act_id", "act_ref", "cut_id"}}
    if act.get("act_id") != content_digest(body):
        raise ValueError("MLP memory Act identity changed")
    if act["kind"] == "install" and content_digest(act["memory"]) != act["memory_digest"]:
        raise ValueError("MLP memory manifest does not match its digest")
    if act["kind"] == "remove" and act.get("exact_undo") is not True:
        raise ValueError("MLP memory removal did not witness exact undo")
    return True
