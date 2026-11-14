"""Core files: a run's prefix dumped once and reloaded as a steppable run on any device/dtype
(gap audit 2026-11-14, P2; the owner's "capture once, replay" rule as gdb's core file).

The scale study wrote its own capture format and three fp32 CPU replicas that rebuilt a
``DynamicCache`` and re-hooked HF modules by hand (``results/scale-20261114/capture.py``,
``hopmath.py:61-100``, ``forecast.py``, ``audit/a3_beast_cpu.py``); the hop-math audit had to find
a replica-null mismatch that produced a free "repair" by hand. Here:

* ``dump(run, path, row=K, capture=(...))``: the boundary state of ``t<K>.start`` (a row start
  the run holds: the prompt tokens, the key/value cache of rows ``0..K-1`` of every layer, the
  stepped-prompt slot) plus the values of the run's ``capture`` addresses (default: the last
  row's logits when computed), written once (``torch.save``) with a sealed ``core-file`` record:
  model identity, numeric scope, dtype/device, procedure, the committed edits on the path to the
  split (baked into the cache), every tensor's sha256, custody (P8).
* ``load(path)`` checks the record seal and every tensor digest (refusing a changed file).
* ``CoreFile.run(adapter)`` seeds a ``Session`` of ``adapter`` (same model; any device and dtype:
  the stored K/V are moved once, ``.to(device=, dtype=)``, never cast twice) with that state and
  returns a ``Run`` rooted there: every row from ``K`` on is stepped natively on the target
  adapter, with ports, edits, checkpoints, holds, branches and watches. On another dtype it is a
  replica (stored-dtype cache, target-dtype compute), labelled so.
* ``CoreFile.null_check(adapter)``: runs the replica to each captured address and records
  ``argmax_equal`` (logits) and ``max_abs_diff`` against the captured values (measured); every
  ``run()`` attaches it as ``run.core_check`` unless ``check=False``.

Structural refusals: another family, layer count, KV heads, head_dim or vocabulary than the dump's.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .ports import tensor_digest

SCHEMA = "mdb-core-file-v1"


def _shape_of(adapter: Any) -> dict[str, Any]:
    return {"family": adapter.spec.model_type, "layers": int(adapter.layers),
            "kv_heads": int(adapter.num_kv_heads), "head_dim": int(adapter.head_dim),
            "vocab": int(adapter.vocab_size)}


def dump(run: Any, path: str | Path, *, row: int | None = None,
         capture: tuple[str, ...] | None = None, session_id: str | None = None
         ) -> dict[str, Any]:
    """Write a core file of ``run`` split at ``t<row>.start`` (default: the run's first stepped
    row) and return its sealed record (module docstring)."""
    from mdb.core.custody import custody
    from mdb.core.kinds import CORE_FILE  # noqa: F401 (declares the kind)
    from mdb.core.records import num, seal_record

    row = run.first_row if row is None else int(row)
    found = [i for i, m in enumerate(run._trail) if m.row == row and m.label == "start"]
    if not found:
        raise ValueError(f"the run holds no t{row}.start boundary (first stepped row "
                         f"t{run.first_row}, cursor {run.where}); dump at a row it has started")
    index = found[-1]
    payload = run._cut_of(index).payload
    keys = [k.detach().to("cpu") for k in payload["keys"]]
    values = [v.detach().to("cpu") for v in payload["values"]]
    tokens = [int(t) for t in payload["tokens"][0].tolist()]
    last = len(run.tokens) - 1
    if capture is None:
        capture = (f"t{last}.logits",) if run.parse(f"t{last}.logits") <= run.parse(run.where) \
            else ()
    captured = {address: run.read(address).detach().to("cpu") for address in capture}
    tensors = {f"keys.{i}": k for i, k in enumerate(keys)}
    tensors.update({f"values.{i}": v for i, v in enumerate(values)})
    tensors.update({f"captured.{a}": t for a, t in captured.items()})
    manifest = {name: {"sha256": tensor_digest(t), "shape": list(t.shape),
                       "dtype": str(t.dtype).removeprefix("torch.")}
                for name, t in tensors.items()}
    adapter = run.adapter
    body = {
        "schema": SCHEMA, "split": {"row": row, "tokens": len(tokens)},
        "prompt": {"tokens": list(run.tokens)[:64], "length": len(run.tokens),
                   "sha256": hashlib.sha256(str(list(run.tokens)).encode()).hexdigest()},
        "shape": _shape_of(adapter),
        "source": {"dtype": str(adapter.dtype).removeprefix("torch."),
                   "device": str(adapter.device), "where": run.where,
                   "cut": run._trail[index].fingerprint},
        "procedure": {"prefill": int(run.prefill), "first_row": int(run.first_row),
                      "prompt_rows": len(run.tokens)},
        "path_edits": [{"transition": e["transition"], "layer": e["layer"],
                        "count": len(e["edits"])} for e in run._path_manifest(index)][:64],
        "edited": bool(run._path_manifest(index)),
        "captured": sorted(captured)[:64],
        "tensors": {"count": len(manifest),
                    "sha256": hashlib.sha256(str(sorted((k, v["sha256"]) for k, v in
                                                        manifest.items())).encode()).hexdigest()},
        "bytes": num(float(sum(t.numel() * t.element_size() for t in tensors.values())),
                     "measured"),
        "custody": custody("mdb-stepped"),
    }
    from mdb.core.pointers import seal_digest
    from saturn_pub.values import digest

    session = session_id or "mdb-run:" + seal_digest({"model": adapter.model_identity,
                                                      "prompt": digest(run.tokens)})[:32]
    record = seal_record(session_id=session, kind="core-file",
                         model_identity=adapter.model_identity,
                         execution=dict(adapter.execution), authority="measured", body=body)
    import json

    # the record and manifest as JSON text: plain data only, so ``weights_only`` loading works
    torch.save({"schema": SCHEMA, "record": json.dumps(record), "manifest": json.dumps(manifest),
                "prompt": list(run.tokens), "tokens": tokens,
                "keys": keys, "values": values, "captured": captured}, str(path))
    return record


class CoreFile:
    """A loaded, digest-checked core file (``load``)."""

    def __init__(self, data: dict[str, Any], path: str | None = None):
        self.path = path
        self.record = data["record"]
        self.manifest = data["manifest"]
        self.prompt = [int(t) for t in data["prompt"]]
        self.tokens = [int(t) for t in data["tokens"]]
        self.keys, self.values = data["keys"], data["values"]
        self.captured = data["captured"]
        self.split = int(self.record["body"]["split"]["row"])

    @property
    def id(self) -> str:
        return self.record["id"]

    def _check(self, adapter: Any) -> None:
        want, have = self.record["body"]["shape"], _shape_of(adapter)
        differ = [k for k in want if want[k] != have[k]]
        if differ:
            raise ValueError("core file of another model shape: " + ", ".join(
                f"{k} {want[k]} vs {have[k]}" for k in differ))

    def session(self, adapter: Any) -> Any:
        """A ``Session`` of ``adapter`` at the split, the stored K/V moved once to the adapter's
        device and dtype."""
        from saturn_pub.core import Session

        self._check(adapter)
        device, dtype = adapter.device, adapter.dtype

        def move(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, dtype=dtype) if (t.dtype != dtype or str(t.device)
                                                        != str(torch.device(device))) else t.clone()

        state = {"tokens": torch.tensor([self.tokens], dtype=torch.long, device=device),
                 "keys": [move(k) for k in self.keys], "values": [move(v) for v in self.values],
                 "layer": 0, "phase": "embed", "hidden": None, "logits": None,
                 "sampled_token": None}
        if self.split < len(self.prompt) - 1:
            state["prompt"] = torch.tensor([self.prompt], dtype=torch.long, device=device)
        return Session(adapter, state)

    def run(self, adapter: Any, *, lookahead: str = "none", check: bool = True,
            name: str | None = None) -> Any:
        """A ``Run`` of ``adapter`` rooted at the split (module docstring); ``run.core`` names the
        file and ``run.core_check`` holds the null check (``check``)."""
        from .run import Run

        cut = self.session(adapter).capture(retain=False)
        run = Run(adapter, self.prompt, root=cut, prefill=self.split, lookahead=lookahead)
        run.name = name or f"core@{self.id[:12]}"
        run.core = {"record": self.id, "path": self.path, "split": self.split,
                    "source_dtype": self.record["body"]["source"]["dtype"],
                    "dtype": str(adapter.dtype).removeprefix("torch."),
                    "replica": str(adapter.dtype).removeprefix("torch.")
                    != self.record["body"]["source"]["dtype"],
                    "same_identity": adapter.model_identity == self.record["model_identity"]}
        run.core_check = self.null_check(adapter, lookahead=lookahead) if check else None
        return run

    def null_check(self, adapter: Any, *, lookahead: str = "none") -> dict[str, Any]:
        """Run a fresh replica to every captured address and compare (measured): per address
        ``max_abs_diff`` (float64) and, for logits, ``argmax_equal``; ``ok`` when every argmax
        agrees (and every difference is 0 on the same dtype and identity)."""
        if not self.captured:
            return {"state": "unresolved", "reason": "the core file captured no value"}
        from .run import Run

        cut = self.session(adapter).capture(retain=False)
        replica = Run(adapter, self.prompt, root=cut, prefill=self.split, lookahead=lookahead)
        out: dict[str, Any] = {"checks": {}, "authority": "measured"}
        same = (str(adapter.dtype).removeprefix("torch.") == self.record["body"]["source"]["dtype"]
                and adapter.model_identity == self.record["model_identity"])
        ok = True
        for address in sorted(self.captured, key=replica.parse):
            replica.until(address)
            got = replica.read(address).detach().to("cpu")
            want = self.captured[address]
            diff = float((got.to(torch.float64) - want.to(torch.float64)).abs().max())
            entry: dict[str, Any] = {"max_abs_diff": diff}
            if address.endswith("logits"):
                entry["argmax_equal"] = int(got.reshape(-1).argmax()) == int(want.reshape(-1).argmax())
                entry["argmax"] = [int(want.reshape(-1).argmax()), int(got.reshape(-1).argmax())]
                ok = ok and entry["argmax_equal"]
            if same:
                ok = ok and diff == 0.0
            out["checks"][address] = entry
        out.update(ok=ok, same_dtype_and_identity=same,
                   rule="argmax equal at every captured logits" + (
                       " and bit-identical values (same dtype and identity)" if same else ""))
        return out


def load(path: str | Path) -> CoreFile:
    """Read a core file, verifying the record seal and every tensor's sha256."""
    from mdb.core.kinds import CORE_FILE  # noqa: F401 (declares the kind)
    from mdb.core.records import verify_record

    import json

    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not an mdb core file")
    data["record"], data["manifest"] = json.loads(data["record"]), json.loads(data["manifest"])
    verify_record(data["record"])
    manifest = data["manifest"]
    tensors = {f"keys.{i}": k for i, k in enumerate(data["keys"])}
    tensors.update({f"values.{i}": v for i, v in enumerate(data["values"])})
    tensors.update({f"captured.{a}": t for a, t in data["captured"].items()})
    if set(tensors) != set(manifest):
        raise ValueError("core file tensors differ from its manifest")
    for name, tensor in tensors.items():
        if tensor_digest(tensor) != manifest[name]["sha256"]:
            raise ValueError(f"core file tensor {name} differs from its sha256")
    summary = hashlib.sha256(str(sorted((k, v["sha256"]) for k, v in manifest.items()))
                             .encode()).hexdigest()
    if summary != data["record"]["body"]["tensors"]["sha256"]:
        raise ValueError("core file manifest differs from its sealed record")
    return CoreFile(data, str(path))


__all__ = ["CoreFile", "SCHEMA", "dump", "load"]
