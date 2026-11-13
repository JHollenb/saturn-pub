"""Sealed, model-agnostic weight-space experiments over MDB forecast storage.

Numerical backends are lazy. Static reading/writing can stop at seal; dynamic
arms use trusted callbacks for the native consumer and state/environment reset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .wsp_weights import (
    ParameterPatch,
    WeightSpaceError,
    digest,
    inspect_patch,
    parameter_inventory,
    weight_diff,
)

ROLES = {"native", "edited", "constructed", "sham", "dose", "ablation", "restoration",
         "wrong-site", "random", "static"}
STATE_POLICIES = {"restart", "compatible-parent", "hotpatch-declared-history", "not-applicable"}


def bind_patch_ports(patch, catalog):
    """Join original parameter bytes to an existing native trace and its uses.

    This is structural evidence. A native use of a parameter does not establish
    a proposed semantic role or a coordinate-specific causal dependency.
    """
    from .forward_symbols import verify_trace

    manifest = patch.manifest if isinstance(patch, ParameterPatch) else copy.deepcopy(patch)
    if (manifest.get("schema") != "mdb-parameter-patch-v1" or manifest.get("identity") !=
            digest({k: v for k, v in manifest.items() if k != "identity"})):
        raise WeightSpaceError("patch manifest identity differs")
    verify_trace(catalog.trace)
    if catalog.trace["model_identity"] != manifest["model_identity"]:
        raise WeightSpaceError("trace and patch model identities differ")
    rows = []
    for entry in manifest["parameters"]:
        selector = "parameter:" + entry["name"]
        port = catalog.resolve(selector)
        if "value_id" not in port:
            raise WeightSpaceError("patch parameter is absent in native trace")
        descriptor = catalog.trace["values"][port["value_id"]]
        if any(descriptor[k] != entry["before"][k] for k in ("shape", "dtype", "sha256")):
            raise WeightSpaceError("native parameter bytes differ from patch source")
        if set(port["declared_aliases"]) != set(entry["aliases"]):
            raise WeightSpaceError("native trace parameter aliases differ")
        rows.append({"parameter": entry["name"], "port": port,
                     "consumers": catalog.first_consumers(selector),
                     "role_status": "structural native use; semantic role unestablished"})
    return {"schema": "mdb-weight-patch-ports-v1", "patch_identity": manifest["identity"],
            "trace_fingerprint": catalog.trace_fingerprint,
            "model_identity": manifest["model_identity"], "execution": catalog.trace["execution"],
            "input_identity": catalog.trace["input_identity"], "parameters": rows}


def _file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _outcome_files(root, record):
    """Find numerical payloads in an encoded observation, without NumPy."""
    paths = set()

    def visit(value):
        if isinstance(value, dict):
            if value.get("kind") == "array":
                paths.add(value["path"])
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(record)
    return [{"path": path, "sha256": _file_sha(root / path)} for path in sorted(paths)]


def verify_weight_study(root, *, expected_manifest=None, expected_seal=None):
    """Verify both prospective evidence and all post-release native outcomes."""
    from .forecast_transaction import verify_transaction

    root = Path(root).resolve()
    final_path = root / "wsp-outcomes.json"
    if final_path.is_symlink():
        raise WeightSpaceError("outcome manifest is a symbolic link")
    final = json.loads(final_path.read_text())
    identity = final.get("identity")
    if (final.get("schema") != "mdb-weight-study-outcomes-v1"
            or identity != digest({k: v for k, v in final.items() if k != "identity"})
            or expected_manifest is not None and expected_manifest != identity):
        raise WeightSpaceError("outcome manifest identity differs")
    verified = verify_transaction(root, expected_seal_sha256=expected_seal,
                                  expected_event_head=final["event_head"])
    if verified["seal_sha256"] != final["seal_sha256"]:
        raise WeightSpaceError("outcomes bind a different forecast seal")
    persisted = []
    for path in sorted((root / "events").glob("*.json")):
        event = json.loads(path.read_text())
        if event["kind"] == "observation_persisted":
            persisted.append(event["detail"]["name"])
    if persisted != [row["name"] for row in final["outcomes"]]:
        raise WeightSpaceError("native outcome inventory differs")
    for row in final["outcomes"]:
        references = [{"path": "observations/" + row["name"] + ".json",
                       "sha256": row["sha256"]}, *row["arrays"]]
        for reference in references:
            relative = reference["path"]
            if not isinstance(relative, str) or Path(relative).is_absolute() or any(
                    part in {"", ".", ".."} for part in relative.split("/")):
                raise WeightSpaceError("unsafe outcome artifact path")
            path = root / relative
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise WeightSpaceError("outcome artifact contains a symbolic link")
            if _file_sha(path) != reference["sha256"]:
                raise WeightSpaceError("native outcome bytes differ: " + relative)
    return {"schema": "mdb-weight-study-verification-v1", "verified": True,
            "outcome_manifest": identity, "outcomes": len(final["outcomes"]),
            "unexecuted_arms": final["unexecuted_arms"],
            "identity_scope": "externally-pinned" if expected_manifest else "local-hash-consistency",
            "forecast": verified}


def _freeze_result(value):
    """Own result bytes before uninstall; preserve native BF16 as bit arrays."""
    from collections.abc import Mapping

    import numpy as np

    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        if str(tensor.dtype) == "torch.bfloat16":
            import torch
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("nonfinite BF16 native result")
            return {"encoding": "bf16-bits-v1", "bits": tensor.reshape(-1).view(
                torch.uint16).numpy().copy().reshape(tensor.shape)}
        return tensor.numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _freeze_result(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_freeze_result(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ArmContext:
    model: object
    declaration: dict
    prediction: object
    payloads: object
    frontiers: object


class WeightStudy:
    """Parameter edits, forecasts and per-arm outcomes with a global release seal.

    ``binding`` names model/checkpoint, numerical execution, consumer and horizon.
    Each arm declares its own state policy and contextual applicability. Model
    module hooks deny forwards before execution; direct .forward/functional calls
    and other objects remain outside that guard. This is not a Python sandbox.
    A parameter patch restores its writes; state policies are adapter declarations,
    not proof that caches, RNG, buffers or environments have been restored.
    """

    def __init__(self, root, *, binding, input_identity, source_identity,
                 model=None, budgets=None):
        from .forecast_transaction import ForecastTransaction

        required = {"model", "checkpoint", "execution", "consumer", "horizon"}
        if not isinstance(binding, dict) or not required <= set(binding) or any(
                not binding[key] for key in required):
            raise WeightSpaceError("model/checkpoint/execution/consumer/horizon binding required")
        self.binding = json.loads(json.dumps(binding, allow_nan=False))
        self.model, self.arms, self._hooks, self._violations = model, {}, [], []
        self._running = False
        self._owner_thread, self._started = threading.get_ident(), time.monotonic()
        self._root_calls, self._module_entries = 0, 0
        self._outcomes, self.outcome_manifest = [], None
        self._baseline = parameter_inventory(model) if model is not None else None
        self.tx = ForecastTransaction(root, model_identity=self.binding,
            input_identity=input_identity, source_identity=source_identity,
            frontier={"kind": "raw-input"}, budgets=budgets,
            execution_scope="WSP externally managed parameter edits; " + str(binding["execution"]))
        try:
            self.tx.frontier("wsp_contract", {
                "schema": "mdb-weight-study-v1", "binding": self.binding,
                "parameter_inventory": self._baseline,
                "guard_scope": "bound module __call__ hooks; trusted Python, no OS isolation",
                "restoration_scope": "declared parameter patches only; state reset is adapter-owned",
                "status_policy": "retain signed observations; no default scientific threshold"})
            if model is not None:
                for path, module in model.named_modules():
                    self._hooks.append(module.register_forward_pre_hook(self._guard(path)))
        except BaseException:
            self.close()
            raise

    def _guard(self, path):
        def guard(module, args):
            if (self._violations or not self._running or self.tx.phase != "execute"
                    or threading.get_ident() != self._owner_thread):
                self._violations.append({"module": path or "<root>", "phase": self.tx.phase})
                raise WeightSpaceError("native model entry outside released WSP arm")
            if (time.monotonic() - self._started) * 1000 >= self.tx.budgets["max_wall_ms"]:
                self._violations.append({"module": path or "<root>", "reason": "wall budget"})
                raise WeightSpaceError("native wall budget exhausted")
            self._module_entries += 1
            if not path:
                self._root_calls += 1
                if self._root_calls > self.tx.budgets["max_native_calls"]:
                    self._violations.append({"module": "<root>", "reason": "native call budget"})
                    raise WeightSpaceError("native call budget exhausted")
        return guard

    def _check(self):
        if self._violations:
            raise WeightSpaceError("latched premature native entry: " + str(self._violations))

    def add_arm(self, name, *, role, predictor, state_policy, patch=None,
                context=None, symbol_refs=()):
        """Freeze a prospective callback result and a parameter/state declaration.

        All arms must be added before seal. ``symbol_refs`` are supplied evidence
        links, not automatically established semantic roles. Constructed networks
        use a patch on an explicit initial (often zero) parameter baseline.
        """
        self._check()
        if self.tx.phase != "open" or name in self.arms:
            raise WeightSpaceError("new arms require an open study and unique name")
        if role not in ROLES or state_policy not in STATE_POLICIES or not callable(predictor):
            raise WeightSpaceError("declared role, state policy and bound predictor required")
        if patch is not None and (self.model is None or not isinstance(patch, ParameterPatch)
                                  or patch.model_identity != self.binding["model"]):
            raise WeightSpaceError("patch must bind this study model identity")
        declaration = {"name": name, "role": role, "state_policy": state_policy,
                       "context": context or {}, "symbol_refs": list(symbol_refs),
                       "patch": patch.manifest if patch else None,
                       "patch_artifact": "patches/" + name if patch else None}
        declaration = json.loads(json.dumps(declaration, allow_nan=False))
        self.tx.frontier(name + "_arm", declaration)
        if patch is not None:
            patch.save(self.tx.root / "patches" / name)

        def guarded_predictor(context):
            result = predictor(context)
            self._check()
            return result

        self.tx.predict(name, guarded_predictor)
        self.arms[name] = (declaration, patch)
        return copy.deepcopy(declaration)

    def seal(self):
        self._check()
        if self.model is not None and parameter_inventory(self.model) != self._baseline:
            raise WeightSpaceError("parameters changed during forecasting")
        return self.tx.seal()

    def run(self, name, callback):
        """Run one released arm. Ordinary callback failures remain per-arm rows.

        Parameter-integrity failures and a broken release seal stop the study.
        Successful callbacks return observations, work and state receipt mappings.
        Work units and state receipts are supplied by the adapter, not inferred.
        """
        self._check()
        if name not in self.arms or not callable(callback):
            raise WeightSpaceError("known arm and native callback required")
        declaration, patch = self.arms[name]
        if patch is not None and patch.manifest != declaration["patch"]:
            raise WeightSpaceError("patch declaration changed after freeze")

        def execute(context):
            self._check()
            calls_before, entries_before = self._root_calls, self._module_entries
            if self.model is not None and parameter_inventory(self.model) != self._baseline:
                raise WeightSpaceError("study baseline parameters changed")
            installed = patch.installed(self.model) if patch else nullcontext()
            if self.model is not None:
                import torch
                grad_context = torch.no_grad()
            else:
                grad_context = nullcontext()
            # Exceptions from installation/restoration remain integrity errors.
            with installed, grad_context:
                self._running = True
                try:
                    result = callback(ArmContext(self.model, copy.deepcopy(declaration),
                                                 context.predictions[name], context.payloads[name],
                                                 context.frontiers))
                    self._check()
                    if not isinstance(result, dict) or not {"observations", "work", "state"} <= set(result):
                        raise ValueError("arm callback must return observations, work and state")
                    if any(not isinstance(result[k], dict) for k in ("observations", "work", "state")):
                        raise ValueError("arm observations, work and state must be mappings")
                    result = _freeze_result(result)
                except WeightSpaceError:
                    raise
                except Exception as error:
                    result = {"observations": {}, "work": {}, "state": {},
                              "error": {"type": type(error).__name__, "message": str(error)}}
                finally:
                    self._running = False
            if self.model is not None and parameter_inventory(self.model) != self._baseline:
                raise WeightSpaceError("arm changed parameters outside the restored patch")
            return {"arm": declaration, "result": result, "mechanics_status": "valid",
                    "instrumented_work": {"root_forward_calls": self._root_calls - calls_before,
                                          "module_entries": self._module_entries - entries_before},
                    "integrity_work": {"parameter_bytes_hashed_for_arm":
                        2 * self._baseline["bytes_hashed"] if self._baseline else 0},
                    "instrument_status": "evidence-incomplete" if "error" in result else "observed",
                    "trend_status": "uninterpreted", "terminal_status": "not-evaluated"}

        record = self.tx.execute(name, execute)
        path = self.tx.root / "observations" / (name + ".json")
        self._outcomes.append({"name": name, "sha256": _file_sha(path),
                               "arrays": _outcome_files(self.tx.root, record)})
        return record

    def report(self):
        return {"schema": "mdb-weight-study-report-v1", "binding": self.binding,
                "arms": [copy.deepcopy(value[0]) for value in self.arms.values()],
                "premature_native_entries": copy.deepcopy(self._violations),
                "instrumented_work": {"root_forward_calls": self._root_calls,
                                      "module_entries": self._module_entries},
                "transaction": self.tx.report()}

    def close(self):
        self._remove_hooks()
        self.tx.close()
        self._finalize()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self._remove_hooks()
        try:
            return self.tx.__exit__(kind, error, traceback)
        finally:
            self._finalize()

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _finalize(self):
        if self.outcome_manifest is not None or self.tx.report()["seal_sha256"] is None:
            return
        observed = {row["name"] for row in self._outcomes}
        final = {"schema": "mdb-weight-study-outcomes-v1",
                 "seal_sha256": self.tx.report()["seal_sha256"],
                 "event_head": self.tx.report()["event_head"], "outcomes": self._outcomes,
                 "unexecuted_arms": [name for name, (declared, _) in self.arms.items()
                                      if name not in observed and declared["role"] != "static"]}
        final["identity"] = digest(final)
        with (self.tx.root / "wsp-outcomes.json").open("x") as stream:
            json.dump(final, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self.outcome_manifest = copy.deepcopy(final)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb wsp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="read patch metadata seals without a model")
    inspect.add_argument("root", type=Path)
    verify = sub.add_parser("verify", help="verify existing MDB forecast evidence offline")
    verify.add_argument("root", type=Path)
    verify.add_argument("--expected-manifest")
    verify.add_argument("--expected-seal")
    demo = sub.add_parser("demo", help="run a tiny CPU mechanics example, no downloaded model")
    demo.add_argument("root", type=Path)
    discovery = sub.add_parser("discover-demo", help="automatically find a blinded constructed-model edit")
    discovery.add_argument("root", type=Path)
    discovery.add_argument("--seed", type=int, default=731)
    diff = sub.add_parser("diff", help="stream a pair of single-file safetensors checkpoints")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_patch(args.root)
    elif args.command == "verify":
        result = verify_weight_study(args.root, expected_manifest=args.expected_manifest,
                                     expected_seal=args.expected_seal)
    elif args.command == "demo":
        from .wsp_demo import run_demo
        result = run_demo(args.root)
    elif args.command == "discover-demo":
        from .wsp_discovery_demo import run_discovery_demo
        result = run_discovery_demo(args.root, seed=args.seed)
    else:
        from collections.abc import Mapping

        from safetensors import safe_open

        class Checkpoint(Mapping):
            def __init__(self, handle):
                self.handle = handle

            def __getitem__(self, name):
                return self.handle.get_tensor(name)

            def __iter__(self):
                return iter(self.handle.keys())

            def __len__(self):
                return len(self.handle.keys())

        with safe_open(args.before, framework="pt", device="cpu") as a, safe_open(
                args.after, framework="pt", device="cpu") as b:
            result = weight_diff(Checkpoint(a), Checkpoint(b))
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.command == "diff" and args.output:
        with args.output.open("x") as stream:
            stream.write(body)
    else:
        print(body, end="")
    return 0
