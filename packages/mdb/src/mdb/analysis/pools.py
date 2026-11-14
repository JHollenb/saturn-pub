"""Pools: sealed value sets collected at one address over a prompt set (gap audit 2026-11-14, P6).

Leave-one-out means were coded four times and donor picking five times; TP1 hit the
self-inclusion bug (``next/AUDIT-RUN1-TP1.md:156``) and the hop-math ``mean_cell(exclude=...)``
argument was dead code (``AUDIT-HOPMATH.md`` section 1). A ``Pool`` is "values taken from other
runs" as one debugger object with provenance:

    pool = collect(dbg, items, "cache.L20.v[kv6,row=@carrier]", labels=("country",),
                   where=lambda item, run: item["correct"], name="carrier-v")
    pool.mean_for(item)                       # leave-one-out mean: the item is never in it
    pool.class_mean_for(item, by="country")   # same-label mean, item excluded
    pool.donor(item, rule="other:country")    # first other-label item in pool order
    pool.random(item, seed=0)                 # Gaussian, norm matched to the item's own value
    pool.seal()                               # a ``pool`` record: items digest, origins, payload

Every value a pool hands out is an ``engine.refs.ValueRef`` whose pointer names the pool (its
sealed record id once sealed), the rule, and exactly which items went in (``items`` and
``excluded``), so ``Run.set(addr, pool.mean_for(item))`` or a ``Branches`` arm records where an
edit value came from. Means are float64 over the stored native values (cast once to the model
dtype when an edit stages it).

Collection (``collect``): one run per item (``Run`` of the item's token ids), stepped until the
address is computed; ``@role`` in the address is the item's role row (``item["roles"][role]`` or
``item[role]``), ``@last`` the last prompt row. A port occurrence's row is stepped (prefill =
that row); a cache cell is read at the last row's first port (rows before it are the prefill's).
``where(item, run)`` (after the read) filters items; the skipped ids are recorded. Each item keeps
its origin pointer (address, value sha256, cut). Values are stored as the native dtype's exact
float values in a sha256-checked JSON payload (``core.bounded.put_json``; held on the pool with
no project store; ``save``/``load`` write and verify a file).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mdb.core.bounded import load_json, put_json
from mdb.core.kinds import POOL  # noqa: F401 (declares the kind)
from mdb.core.records import seal_record

from .finite import EXACT, MEASURED, f64, num
from .forecast import Refusal

SCHEMA = "mdb-pool-v1"
_ROLE = re.compile(r"@(\w+)")
_CAP = 64


def _torch():
    import torch

    return torch


def _role_row(item: dict[str, Any], role: str) -> int:
    if role == "last":
        return len(item["ids"]) - 1
    roles = item.get("roles") or {}
    value = roles.get(role, item.get(role))
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else None
    if not isinstance(value, int):
        raise Refusal(f"item {item.get('id')!r} has no row for role @{role}")
    return value


def bind(address: str, item: dict[str, Any]) -> str:
    """``address`` with every ``@role`` replaced by the item's row (``t@last`` -> ``t<last>``)."""
    return _ROLE.sub(lambda m: str(_role_row(item, m.group(1))), address)


class Pool:
    """Values at one address, one per item, with origins (module docstring)."""

    def __init__(self, name: str, address: str, items: list[dict[str, Any]], values: Any,
                 *, skipped: Sequence[Any] = (), model_identity: Any = None,
                 execution: dict[str, Any] | None = None, labels: Sequence[str] = ()):
        torch = _torch()
        self.name, self.address = name, address
        self.items = items                      # {id, address, origin, labels}
        self.values = values.detach().to("cpu") if isinstance(values, torch.Tensor) else values
        self.skipped = list(skipped)
        self.model_identity, self.execution = model_identity, dict(execution or {})
        self.labels = tuple(labels)
        self.record: dict[str, Any] | None = None
        self._index = {str(it["id"]): i for i, it in enumerate(items)}
        if len(self._index) != len(items):
            raise Refusal("pool item ids must be unique")

    # --- identity -------------------------------------------------------------------------

    @property
    def ids(self) -> list[str]:
        return [str(it["id"]) for it in self.items]

    def items_sha256(self, ids: Sequence[str] | None = None) -> str:
        chosen = self.ids if ids is None else list(ids)
        return hashlib.sha256(json.dumps(chosen, separators=(",", ":")).encode()).hexdigest()

    def payload(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "dtype": str(self.values.dtype).removeprefix("torch."),
                "shape": list(self.values.shape[1:]), "ids": self.ids,
                "labels": [it.get("labels") or {} for it in self.items],
                "tokens_sha256": [(it.get("origin") or {}).get("tokens_sha256")
                                  for it in self.items],
                "values": f64(self.values).reshape(len(self.items), -1).tolist()}

    def __len__(self) -> int:
        return len(self.items)

    def value(self, item: Any) -> Any:
        return self.values[self._position(item)]

    def _position(self, item: Any) -> int:
        key = str(item["id"] if isinstance(item, dict) else item)
        if key not in self._index:
            raise Refusal(f"item {key!r} is not in pool {self.name}")
        return self._index[key]

    def _label(self, item: Any, by: str) -> Any:
        if isinstance(item, dict) and by in ((item.get("labels") or {}).keys() | item.keys()):
            return (item.get("labels") or {}).get(by, item.get(by))   # nested labels win
        return self.items[self._position(item)]["labels"].get(by)

    @property
    def kernel_path(self) -> str:
        """``mdb-prefill-batched`` for cache-cell pools (read from the batched prefill; the
        per-item ``origin.prefill`` is recorded), else ``mdb-stepped``."""
        return "mdb-prefill-batched" if self.address.startswith("cache.") else "mdb-stepped"

    def duplicates(self) -> int:
        """Items whose prompt tokens equal an earlier item's (by ``tokens_sha256``)."""
        seen = [(it.get("origin") or {}).get("tokens_sha256") for it in self.items]
        return len([h for i, h in enumerate(seen) if h is not None and h in seen[:i]])

    def _excluded(self, item: Any, exclude: str) -> set[int]:
        """Positions that must not feed ``item``'s value: its own id and, when ``exclude`` is
        ``"tokens"`` (default), every item with the same prompt tokens under another id (a
        duplicated prompt would put the item's own value in its leave-one-out mean)."""
        if exclude not in ("id", "tokens"):
            raise Refusal("exclude is 'id' or 'tokens'")
        key = str(item["id"] if isinstance(item, dict) else item)
        out = {self._index[key]} if key in self._index else set()
        if exclude == "tokens":
            sha = None
            if isinstance(item, dict) and item.get("ids") is not None:
                sha = hashlib.sha256(json.dumps([int(t) for t in item["ids"]]).encode()
                                     ).hexdigest()
            elif key in self._index:
                sha = (self.items[self._index[key]].get("origin") or {}).get("tokens_sha256")
            if sha is not None:
                out |= {i for i, it in enumerate(self.items)
                        if (it.get("origin") or {}).get("tokens_sha256") == sha}
        return out

    # --- values handed out ------------------------------------------------------------------

    def _ref(self, value: Any, rule: str, used: list[str], excluded: str | None,
             **extra: Any) -> Any:
        from mdb.engine.ports import tensor_digest
        from mdb.engine.refs import ValueRef

        pointer = {"kind": "pool", "origin": f"pool:{self.name}", "pool": self.name,
                   "record": None if self.record is None else self.record["id"],
                   "address": self.address, "rule": rule, "n": len(used),
                   "items_sha256": self.items_sha256(used), "excluded": excluded,
                   "sha256": tensor_digest(value), "shape": list(value.shape), **extra}
        if len(used) <= 16:
            pointer["items"] = list(used)
        return ValueRef(value, pointer)

    def _mean(self, keep: list[int], rule: str, excluded: str | None, **extra: Any) -> Any:
        if not keep:
            raise Refusal(f"pool {self.name}: no item left for the {rule}")
        torch = _torch()
        index = torch.tensor(keep, dtype=torch.long)
        value = f64(self.values.index_select(0, index)).mean(0)
        return self._ref(value, rule, [self.ids[i] for i in keep], excluded, **extra)

    def mean(self) -> Any:
        """The mean over every item: the value for an item OUTSIDE the pool. For an item of the
        pool use ``mean_for`` (it is excluded by construction)."""
        return self._mean(list(range(len(self))), "mean of all items", None)

    def mean_for(self, item: Any, *, exclude: str = "tokens") -> Any:
        """Leave-one-out mean: every item but ``item`` (excluded by construction; an item not
        in the pool gets the full mean, ``excluded`` None). ``exclude="tokens"`` (default) also
        drops items with the same prompt tokens under another id; ``"id"`` drops only the id."""
        key = str(item["id"] if isinstance(item, dict) else item)
        gone = self._excluded(item, exclude)
        keep = [i for i in range(len(self)) if i not in gone]
        own = {self._index[key]} if key in self._index else set()
        return self._mean(keep, "leave-one-out mean", key if key in self._index else None,
                          exclude=exclude, duplicates_dropped=len(gone - own))

    def class_mean_for(self, item: Any, *, by: str, exclude: str = "tokens") -> Any:
        """Mean over the items with ``item``'s label ``by``, ``item`` excluded."""
        key = str(item["id"] if isinstance(item, dict) else item)
        label = self._label(item, by)
        gone = self._excluded(item, exclude)
        keep = [i for i, it in enumerate(self.items)
                if it["labels"].get(by) == label and i not in gone]
        return self._mean(keep, f"same-{by} mean, item excluded", key if key in self._index
                          else None, label={by: label}, exclude=exclude)

    def donor(self, item: Any, *, rule: str = "other", order: Sequence[str] | None = None,
              exclude: str = "tokens") -> Any:
        """One other item's value: ``rule`` ``any`` (any other item), ``same:<label>`` (same
        label) or ``other:<label>`` (another label); the first eligible item in ``order``
        (default: pool order) that is not ``item``."""
        key = str(item["id"] if isinstance(item, dict) else item)
        kind, _, by = rule.partition(":")
        if kind not in ("any", "same", "other") or (kind != "any" and not by):
            raise Refusal("donor rule is any, same:<label> or other:<label>")
        label = None if kind == "any" else self._label(item, by)
        gone = self._excluded(item, exclude)
        for donor_id in (order or self.ids):
            if str(donor_id) == key or str(donor_id) not in self._index \
                    or self._index[str(donor_id)] in gone:
                continue
            labels = self.items[self._index[str(donor_id)]]["labels"]
            if kind == "same" and labels.get(by) != label:
                continue
            if kind == "other" and labels.get(by) == label:
                continue
            position = self._index[str(donor_id)]
            ref = self._ref(self.values[position].clone(), f"donor ({rule})", [str(donor_id)],
                            key if key in self._index else None, donor=str(donor_id))
            ref.pointer["donor_origin"] = self.items[position]["origin"]
            return ref
        raise Refusal(f"pool {self.name}: no donor for {key!r} under rule {rule!r}")

    def random(self, item: Any = None, *, norm: float | None = None, seed: int = 0) -> Any:
        """A seeded Gaussian of the pool's value shape (float64), rescaled to ``norm`` or to the
        norm of ``item``'s own value (matched-norm random)."""
        torch = _torch()
        if norm is None:
            if item is None:
                raise Refusal("random needs norm= or an item whose norm it matches")
            norm = float(f64(self.value(item)).norm())
        g = torch.randn(tuple(self.values.shape[1:]),
                        generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64)
        g = g * (float(norm) / float(g.norm()))
        key = None if item is None else str(item["id"] if isinstance(item, dict) else item)
        return self._ref(g, "matched-norm Gaussian", [], key, seed=int(seed),
                         norm=num(float(norm), EXACT))

    # --- records ----------------------------------------------------------------------------

    def seal(self, *, session_id: str | None = None, store: Any = None) -> dict[str, Any]:
        """A ``pool`` record: address, items (ids, labels, origins; first 64), items digest,
        skipped ids, the values payload (sha256-checked JSON) and custody."""
        from mdb.core.custody import custody

        payload = put_json(self.payload(), store=store, run=self)
        shown = [{"id": str(it["id"]), "labels": it["labels"],
                  "origin": {k: it["origin"][k] for k in ("address", "sha256", "cut", "prefill")
                             if k in it["origin"]}}
                 for it in self.items[:_CAP]]
        body = {"schema": SCHEMA, "name": self.name, "address": self.address,
                "n": len(self.items), "shape": list(self.values.shape[1:]),
                "dtype": str(self.values.dtype).removeprefix("torch."),
                "items": shown, "items_omitted": max(0, len(self.items) - _CAP),
                "items_sha256": self.items_sha256(), "skipped": [str(s) for s in
                                                                 self.skipped[:_CAP]],
                "labels": list(self.labels), "payload": payload,
                "norms": {"mean": num(float(f64(self.values).reshape(len(self), -1).norm(dim=1)
                                            .mean()), MEASURED)} if len(self) else {},
                "duplicates": self.duplicates(),
                "custody": custody(self.kernel_path)}
        session = session_id or "mdb-pool:" + self.items_sha256()[:32]
        self.record = seal_record(session_id=session, kind="pool",
                                  model_identity=self.model_identity,
                                  execution=self.execution, authority=MEASURED, body=body)
        return self.record

    def save(self, path: str | Path) -> Path:
        """Write ``{record, payload, items}`` as JSON (seals first if needed)."""
        record = self.record or self.seal()
        path = Path(path)
        path.write_text(json.dumps({"record": record, "payload": self.payload(),
                                    "items": self.items}, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Pool:
        """Read a saved pool, checking the record seal and the payload sha256."""
        from mdb.core.bounded import json_bytes
        from mdb.core.records import verify_record

        torch = _torch()
        data = json.loads(Path(path).read_text())
        record = verify_record(data["record"])
        payload = data["payload"]
        if hashlib.sha256(json_bytes(payload)).hexdigest() != record["body"]["payload"]["sha256"]:
            raise Refusal("pool payload differs from its sealed sha256")
        items = data["items"]
        if [str(it["id"]) for it in items] != [str(i) for i in payload["ids"]]:
            raise Refusal("pool items differ from the payload ids (order or content): refused, "
                          "the leave-one-out mean would use the wrong rows")
        body = record["body"]
        if hashlib.sha256(json.dumps([str(i) for i in payload["ids"]], separators=(",", ":"))
                          .encode()).hexdigest() != body["items_sha256"]:
            raise Refusal("pool ids differ from the sealed items_sha256")
        for it, label in zip(items, payload.get("labels", [None] * len(items))):
            if label is not None and (it.get("labels") or {}) != label:
                raise Refusal(f"pool item {it['id']!r} labels differ from the sealed payload")
        for it, sha in zip(items, payload.get("tokens_sha256", [None] * len(items))):
            if sha is not None and (it.get("origin") or {}).get("tokens_sha256") != sha:
                raise Refusal(f"pool item {it['id']!r} tokens digest differs from the payload")
        dtype = getattr(torch, payload["dtype"])
        values = torch.tensor(payload["values"], dtype=torch.float64).reshape(
            len(payload["ids"]), *payload["shape"]).to(dtype)
        pool = cls(record["body"]["name"], record["body"]["address"], items, values,
                   skipped=record["body"]["skipped"], model_identity=record["model_identity"],
                   execution=record["execution"], labels=record["body"]["labels"])
        pool.record = record
        return pool

    def held_payload(self) -> Any:
        """The payload the record points at, read back and checked (``bounded.load_json``)."""
        if self.record is None:
            raise Refusal("seal the pool first")
        return load_json(self.record["body"]["payload"], run=self)


def collect(debugger: Any, items: Sequence[dict[str, Any]], address: str, *,
            name: str = "pool", labels: Sequence[str] = (),
            where: Callable[[dict[str, Any], Any], bool] | None = None) -> Pool:
    """Collect one value per item at ``address`` (module docstring). ``items``: dicts with
    ``id``, ``ids`` (token ids), optional ``roles`` and the ``labels`` keys."""
    from mdb.engine.run import Run

    torch = _torch()
    adapter = debugger.adapter if hasattr(debugger, "adapter") else debugger
    kept, values, skipped = [], [], []
    for item in items:
        bound = bind(address, item)
        ids = [int(t) for t in item["ids"]]
        last = len(ids) - 1
        if bound.startswith("cache."):
            run = Run(adapter, ids)
            run.until(f"t{last}.L0.{run.program.names[0]}")
        else:
            row = int(bound.split(".", 1)[0][1:]) if bound.startswith("t") else last
            run = Run(adapter, ids, prefill=min(row, last))
            run.until(bound.partition("[")[0])
        run.name = f"{name}:{item['id']}"
        ref = run.ref(bound)
        if where is not None and not where(item, run):
            skipped.append(item["id"])
            continue
        kept.append({"id": str(item["id"]), "address": bound,
                     "labels": {k: (item.get("labels") or {}).get(k, item.get(k)) for k in labels},
                     "origin": {**ref.pointer, "prefill": int(run.prefill), "tokens_sha256": hashlib.sha256(
                         json.dumps(ids).encode()).hexdigest()}})
        values.append(ref.value.detach().to("cpu"))
    if not kept:
        raise Refusal(f"pool {name}: no item kept (skipped {len(skipped)})")
    stacked = torch.stack(values)
    return Pool(name, address, kept, stacked, skipped=skipped,
                model_identity=adapter.model_identity, execution=dict(adapter.execution),
                labels=labels)


__all__ = ["Pool", "SCHEMA", "bind", "collect"]
