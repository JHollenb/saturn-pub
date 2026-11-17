"""Cuts held by ``checkpoint-state`` records (``mdb.engine.checkpoint``), read for ``mdb serve``.

``Debugger.checkpoint`` stores the state of a cut as a sealed ``checkpoint-state`` record whose
document (a JSON blob) lists every kept trail mark's native cut (boundary, parent, execution
point, surface) and its payload, each tensor as parts that are ``put_tensor`` blobs of the
store. It stores no native cut beside it, so the recorded frame and timeline read these too: a
kept mark is a cut with the same id (its fingerprint) and the same descriptor fields as a native
one, and a slot tensor is its parts read verified (the record's seal, the document's sha256,
each part's payload sha256 against the document) and concatenated along the tensor's axis.

Nothing is recomputed: the fingerprint is the one the sealed record names (the engine recomputes
it from the content when it rehydrates the state; serve has no engine). Read-only, numpy only.
"""
from __future__ import annotations

from typing import Any

from . import blobs

KIND = "checkpoint-state"
SCHEMA = "mdb-checkpoint-v2"


class StateCuts:
    """The kept marks of every ``checkpoint-state`` record of one session, by fingerprint
    (first record wins), and ``cursor``: the newest record's cut."""

    def __init__(self, project: Any, sid: str):
        from ..core import index
        from ..core.bounded import load_json
        from ..core.kinds import CHECKPOINT_STATE  # noqa: F401 (declares the kind)

        self.project, self.store_root = project, project.store.root
        self.cuts: dict[str, dict[str, Any]] = {}
        self.cursor: str | None = None
        self.problems: list[str] = []
        for record in index.list_records(project, sid, kind=KIND):
            body = record.get("body") or {}
            if record.get("verified") is False or body.get("schema") != SCHEMA:
                continue
            try:
                doc = load_json(body["document"], store=project.store)
                marks = doc["marks"]
                if (marks[-1]["fingerprint"] != body["cut"]["fingerprint"]
                        or marks[0]["fingerprint"] != body["root"]["fingerprint"]):
                    raise ValueError("document marks differ from the record")
            except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
                self.problems.append(f"checkpoint {record['id'][:16]}: {error}")
                continue
            for i, mark in enumerate(marks):
                cut = mark.get("cut")
                if cut is None or mark["fingerprint"] in self.cuts:
                    continue
                self.cuts[mark["fingerprint"]] = {
                    "descriptor": {
                        "fingerprint": mark["fingerprint"], "manifest": None,
                        "model_identity": doc["model_identity"], "parent": cut["parent"],
                        "boundary": cut["boundary"],
                        "execution_point": cut["execution_point"] or {},
                        "surface": cut["surface"] or {},
                        "checkpoint": {"record": record["id"], "mark": i, "name": doc["name"]}},
                    "payload": cut["payload"], "tensors": doc["tensors"],
                    "parts": doc["parts"], "record": record["id"]}
            self.cursor = marks[-1]["fingerprint"]

    def __contains__(self, cut: str) -> bool:
        return cut in self.cuts

    def slot_names(self, cut: str) -> set[str]:
        payload = self.cuts[cut]["payload"]
        return set(payload.get("items") or {}) if payload.get("kind") == "mapping" else set()

    def _tensor(self, cut: str, slot: str, item: int | None) -> dict[str, Any]:
        entry = self.cuts[cut]
        encoded = (entry["payload"].get("items") or {}).get(slot)
        if encoded is None:
            raise blobs.BlobError(f"cut {cut[:12]} retains no slot {slot!r}")
        kind = encoded.get("kind")
        if kind in ("list", "tuple"):
            items = encoded.get("items") or []
            if item is None:
                raise blobs.BlobError(f"slot {slot!r} is a sequence of {len(items)}; give an "
                                      "item index")
            if not 0 <= item < len(items):
                raise blobs.BlobError(f"slot {slot!r} has no tensor at item {item}")
            encoded = items[item]
        elif item is not None and kind == "tensor":
            raise blobs.BlobError(f"slot {slot!r} holds one tensor; no item index")
        if encoded.get("kind") == "scalar":
            raise blobs.BlobError(f"slot {slot!r} holds no tensor at this cut (scalar "
                                  f"{encoded.get('value')!r})")
        if encoded.get("kind") != "tensor":
            raise blobs.BlobError(f"slot {slot!r} has unsupported kind {encoded.get('kind')!r}")
        return entry["tensors"][encoded["t"]]

    def slot_info(self, cut: str, slot: str, item: int | None = None) -> dict[str, Any]:
        desc = self._tensor(cut, slot, item)
        root = self.store_root
        for index in desc["parts"]:
            ref = self.cuts[cut]["parts"][index]["ref"]
            if not blobs._store_file(root / "tensors" / f"{blobs._hex(ref, 'ref')}.safetensors"):
                raise blobs.BlobError(f"slot {slot!r} part {ref[:12]} is not in the store")
        return {"shape": list(desc["shape"]), "dtype": desc["dtype"]}

    def cut_slot(self, cut: str, slot: str, item: int | None = None):
        """``(array, info)``: the slot tensor from its parts, each verified."""
        import numpy as np

        desc = self._tensor(cut, slot, item)
        pointers = [self.cuts[cut]["parts"][i] for i in desc["parts"]]
        arrays = []
        for pointer in pointers:
            array, info = blobs.tensor_blob(self.store_root, pointer["ref"])
            if info["sha256"] != pointer["sha256"] or list(info["shape"]) != pointer["shape"]:
                raise blobs.BlobError("checkpoint part differs from its document", refused=True)
            arrays.append(np.asarray(array))
        full = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=desc["axis"])
        if list(full.shape) != list(desc["shape"]):
            raise blobs.BlobError("checkpoint tensor parts differ from its shape", refused=True)
        record = self.cuts[cut]["record"]
        return full, {"shape": list(desc["shape"]), "dtype": desc["dtype"], "sha256": None,
                      "via": [f"cut {cut[:12]}", f"checkpoint {record[:12]}",
                              f"slot {slot}" + ("" if item is None else f"[{item}]"),
                              f"{len(pointers)} part(s)",
                              "each part's sha256 verified against the sealed document"]}
