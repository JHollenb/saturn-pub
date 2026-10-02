"""Bounded residual -> successor memory -> native future measurements.

This is an instrument, not a semantic reasoning detector. All arms start from
one cut, use admitted edits, and retain signed observations and controls. Native
execution and memory remain owned by the family runtime.
"""
import math
import re
import time

import torch

from .bench_mamba import MemoryInterchangeError
from .bench_qwen import BenchError, tensor_digest
from .model_os_contracts import content_digest

SCHEMA = "saturn-bench-circuit-measurement-v1"
OBSERVATION_SCHEMA = "saturn-bench-circuit-observation-v1"


def capabilities(family):
    return {"protocol": SCHEMA, "available": family in {"qwen2", "mamba"},
        "observable": "token_margin", "start_boundary": "committed-token",
        "memory_mediation": family == "mamba", "semantic_reasoning_detector": False,
        "memory_policy": "complete recurrent pages and flags; fixed recipient input" if family == "mamba"
            else "read-only KV observations" if family == "qwen2" else "not-qualified"}


def _family(bench):
    return getattr(bench, "family", "qwen2")


def _tokens(bench):
    return list(bench.branch.token_ids if _family(bench) == "qwen2" else bench.state["token_ids"])


def _layers(bench):
    return bench.model.config.num_hidden_layers


def _finite_scalar(value):
    try:
        return type(value) in {int, float} and math.isfinite(float(value))
    except OverflowError:
        return False


def _validate(bench, spec):
    if _family(bench) not in {"qwen2", "mamba"}:
        raise BenchError("circuit measurement is not qualified for diffusion")
    if bench.publication.mode != "journal":
        raise BenchError("circuit measurement requires journal mode for atomic panel publication")
    if not bench.branches or bench._edit or bench.cursor()["phase"] != "token":
        raise BenchError("circuit measurement requires a committed token boundary and no pending Act")
    if not isinstance(spec, dict): raise BenchError("circuit spec must be an object")
    # JSON normalization freezes the caller's mutable input and excludes tensors,
    # non-finite values and unspecified executable callbacks from this protocol.
    import json
    try: spec = json.loads(json.dumps(spec, allow_nan=False))
    except (TypeError, ValueError): raise BenchError("circuit spec must be finite JSON") from None
    allowed = {"name", "sites", "contrasts", "observable", "query_token_ids", "continuation_tokens",
               "memory_mediation", "max_native_operations", "max_capture_bytes", "seed"}
    if set(spec)-allowed: raise BenchError("unknown circuit spec fields")
    sites = spec.get("sites", [])
    if (not isinstance(sites, list) or not 1 <= len(sites) <= 8
        or any(type(s) is not int or not 0 <= s <= _layers(bench) for s in sites)
        or len(set(sites)) != len(sites)):
        raise BenchError("circuit sites must be 1..8 distinct residual layer boundaries")
    contrasts = spec.get("contrasts", [])
    if not isinstance(contrasts, list) or not 1 <= len(contrasts) <= 4:
        raise BenchError("circuit measurement requires 1..4 contrasts")
    names = set()
    for arm in contrasts:
        if (not isinstance(arm, dict) or set(arm)-{"name", "mode", "value"}
            or not isinstance(arm.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*", arm["name"])
            or arm["name"] in names or arm["name"] in {"native", "noop", "residual-restore"}
            or arm["name"].startswith(("random:", "memory-transfer:", "memory-restore:"))
            or arm.get("mode") not in {"add", "scale"} or not _finite_scalar(arm.get("value"))):
            raise BenchError("contrasts require unique names and finite scalar add/scale payloads")
        arm["value"] = float(arm["value"])
        scalar = torch.as_tensor(arm["value"], dtype=next(bench.model.parameters()).dtype)
        if not bool(torch.isfinite(scalar)):
            raise BenchError("contrast violates native finite scalar contract")
        names.add(arm["name"])
    observable = spec.get("observable", {})
    if (not isinstance(observable, dict) or set(observable) != {"kind", "token_id", "against_token_id"}
        or observable.get("kind") != "token_margin"
        or any(type(observable.get(k)) is not int or not 0 <= observable[k] < bench.model.config.vocab_size
               for k in ("token_id", "against_token_id"))
        or observable["token_id"] == observable["against_token_id"]):
        raise BenchError("declare a token_margin with two distinct native vocabulary IDs")
    query = spec.setdefault("query_token_ids", [])
    if (not isinstance(query, list) or len(query) > 32
        or any(type(t) is not int or not 0 <= t < bench.model.config.vocab_size for t in query)):
        raise BenchError("query_token_ids must contain at most 32 native vocabulary IDs")
    horizon = spec.setdefault("continuation_tokens", 1)
    if type(horizon) is not int or not 1 <= horizon <= 8: raise BenchError("continuation budget must be 1..8")
    mediation = spec.setdefault("memory_mediation", False)
    if type(mediation) is not bool: raise BenchError("memory_mediation must be boolean")
    if mediation and (_family(bench) != "mamba" or not query):
        raise BenchError("memory mediation requires Mamba complete pages and an explicit fixed query input")
    if (not isinstance(spec.setdefault("name", "state-path"), str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*", spec["name"])):
        raise BenchError("circuit name must be a safe symbol identifier")
    if type(spec.setdefault("seed", 0)) is not int or not 0 <= spec["seed"] < 2**63:
        raise BenchError("seed must be an integer in [0, 2**63)")
    # Count layer-token work conservatively, including prefix construction and
    # all controls. This is an execution budget, never an effect-size gate.
    arms = 3 + len(contrasts)*(4 if mediation else 2)
    operations = sum(sites) + len(sites)*_layers(bench)*arms*(horizon+len(query))
    budget = spec.setdefault("max_native_operations", 4096)
    if type(budget) is not int or not 1 <= budget <= 100000 or operations > budget:
        raise BenchError(f"native operation budget exceeded ({operations} layer-token upper bound)")
    consumed = horizon+len(query)
    generated = bench.cursor()["token"]
    if generated+consumed > bench.capacity: raise BenchError("circuit continuation exceeds session capacity")
    if _family(bench) == "qwen2":
        rows = len(_tokens(bench))+consumed
        if rows > bench.model.config.max_position_embeddings: raise BenchError("circuit continuation exceeds context capacity")
        memory_bytes = rows*bench.vm.runtime.codec.row_bytes
    else:
        memory_bytes = sum(t.numel()*t.element_size() for page in bench.state["pages"]
                           for t in (page["conv"], page["ssm"]))
    element_size = next(bench.model.parameters()).element_size()
    if _family(bench) == "mamba":
        element_size = min(element_size, *(t.element_size() for page in bench.state["pages"]
            for t in (page["conv"], page["ssm"])))
    residual_bytes = (_layers(bench)+1)*bench.model.config.hidden_size*element_size
    logit_bytes = bench.model.config.vocab_size*element_size
    # Includes frozen CPU snapshots and promoted comparison workspace; excludes
    # runtime-owned weights/workspace and durable cuts, which have their own budgets.
    # Six native-dtype snapshots plus four double-precision comparison buffers
    # is conservative even for BF16. Logit summary/consumer copies are included.
    capture_upper_bound = int((6 + 4*8/element_size)*(memory_bytes+residual_bytes+logit_bytes))
    limit = spec.setdefault("max_capture_bytes", 64*1024*1024)
    if type(limit) is not int or not 1 <= limit <= 1024**3 or capture_upper_bound > limit:
        raise BenchError(f"capture byte budget exceeded ({capture_upper_bound} upper bound)")
    return spec, operations, capture_upper_bound


def _summary(tensor):
    t = tensor.detach().cpu()
    return {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": tensor_digest(t),
            "bytes": t.numel()*t.element_size(), "norm": float(t.double().norm()),
            "mean": float(t.double().mean()) if t.numel() else None}


def _delta(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise BenchError("comparison requires aligned shape and dtype")
    a, b = actual.double(), reference.double()
    d = a-b
    norm, base = float(d.norm()), float(b.norm())
    product = float(a.norm()*b.norm())
    return {"exact": torch.equal(actual, reference), "l2": norm, "relative_l2": norm/base if base else None,
            "max_abs": float(d.abs().max()) if d.numel() else 0.0,
            "signed_mean": float(d.mean()) if d.numel() else None,
            "cosine": float((a*b).sum())/product if product else None,
            "changed_elements": int(torch.count_nonzero(d))}


def _memory(bench):
    if _family(bench) == "mamba":
        values = {f"{kind}.{i}": bench.resolve_value(f"{kind}.{i}").cpu()
                  for i in range(_layers(bench)) for kind in ("conv_state", "ssm_state")}
        from copy import deepcopy
        flags = deepcopy([page["flags"] for page in bench.state["pages"]])
    else:
        values = {f"kv_cache.layer_{i:03d}.{kind}": bench.resolve_value(f"kv_cache.layer_{i:03d}.{kind}").cpu()
                  for i in range(_layers(bench)) for kind in ("key", "value")}
        flags = None
    return values, flags


def _memory_summary(memory):
    values, flags = memory
    summaries = {path: _summary(t) for path, t in values.items()}
    return {"addresses": summaries, "cache_flags": flags, "bytes": sum(v["bytes"] for v in summaries.values()),
            "sha256": content_digest({"addresses": summaries, "cache_flags": flags})}


def _memory_delta(actual, reference, *, growth=False):
    av, af = actual; bv, bf = reference
    rows, squared = {}, 0.0
    for path, b in bv.items():
        a = av[path]
        if growth:
            # Qwen old rows and newly appended rows have different meanings.
            # Never pad a reference with zeros and call new storage a mutation.
            prefix = a[:, :, :b.shape[2]]
            delta = _delta(prefix, b)
            added = a[:, :, b.shape[2]:]
            rows[path] = {"retained_prefix": delta, "new_rows": added.shape[2], "new_payload": _summary(added)}
        else:
            delta = _delta(a, b); rows[path] = delta
        squared += delta["l2"]**2
    return {"addresses": rows, "aligned_l2": math.sqrt(squared), "cache_flags_exact": af == bf,
            "all_aligned_exact": all((v["retained_prefix"] if growth else v)["exact"] for v in rows.values())}


def _closure(bench, memory):
    value = {"cursor": dict(bench.cursor()), "token_ids": _tokens(bench),
             "logits": _summary(bench.resolve_value("logits")), "memory": _memory_summary(memory)}
    value["sha256"] = content_digest(value)
    return value


def _margin(bench, spec):
    logits = bench.resolve_value("logits").float()
    observable = spec["observable"]
    return float(logits[0, observable["token_id"]]-logits[0, observable["against_token_id"]])


def _future(bench, spec):
    receipts = []
    for token in spec["query_token_ids"]:
        bench.stage_input(token)
        receipts.append(bench.step("token"))
    for _ in range(spec["continuation_tokens"]-1): receipts.append(bench.step("token"))
    memory = _memory(bench)
    result = {"value": _margin(bench, spec), "closure": _closure(bench, memory),
              "endpoint_cut": bench.checkpoint("circuit-future"), "native_steps": receipts}
    return result


def _first(bench, spec, reference_trace=None):
    trace, tensors = [], {}
    while bench.cursor()["next_layer"] < _layers(bench):
        layer = bench.cursor()["next_layer"]
        hidden = bench.resolve_value("residual").cpu()
        row = {"next_layer": layer, "residual": _summary(hidden)}
        if reference_trace is not None: row["delta"] = _delta(hidden, reference_trace[layer])
        else: tensors[layer] = hidden
        trace.append(row)
        bench.step("layer")
    hidden = bench.resolve_value("residual").cpu()
    row = {"next_layer": _layers(bench), "residual": _summary(hidden)}
    if reference_trace is not None: row["delta"] = _delta(hidden, reference_trace[_layers(bench)])
    else: tensors[_layers(bench)] = hidden
    trace.append(row)
    native_step = bench.step("token")
    memory = _memory(bench)
    first = {"kind": "native-token-output", "value": _margin(bench, spec), "closure": _closure(bench, memory), "trace": trace,
             "successor_cut": bench.checkpoint("circuit-successor"), "native_step": native_step}
    return first, memory, tensors


def _effect(actual, baseline):
    measured = actual.get("kind") != "recipient-boundary-snapshot"
    return {"signed_effect": actual["value"]-baseline["value"] if measured else None,
            "native_output_measured": measured,
            "same_logits": actual["closure"]["logits"]["sha256"] == baseline["closure"]["logits"]["sha256"],
            "same_tokens": actual["closure"]["token_ids"] == baseline["closure"]["token_ids"],
            "closure_exact": actual["closure"]["sha256"] == baseline["closure"]["sha256"]}


def _edit(bench, value, mode):
    preview = bench.preview_edit("residual", value, mode=mode)
    return bench.commit_edit(preview["act_id"])


def atomic_measurement(bench, operation, *arguments):
    """Restore the authoritative registry as well as native state on failure."""
    # Refusing a command must not replace a live backend underneath an already
    # staged Act or discard an uncheckpointed in-flight token.
    if operation is measure_circuit:
        _validate(bench, arguments[0])
    elif operation is replay_circuit:
        _replay_evidence(bench, arguments[0])
    from copy import deepcopy
    original_cut = bench.branch.cut if bench.branches else None
    original_branch = bench.active
    original_cuts, original_events = deepcopy(bench.cuts), deepcopy(bench.events)
    try:
        with bench.publication.atomic_selection():
            return operation(bench, *arguments)
    except BaseException:
        bench.cuts, bench.events = original_cuts, original_events
        if original_cut is not None:
            bench.active = original_branch
            bench._restore(original_cut)
        raise


@torch.inference_mode()
def measure_circuit(bench, spec):
    spec, operations, capture_upper_bound = _validate(bench, spec)
    start = time.perf_counter()
    original_cut = bench.branch.cut
    parent = bench.checkpoint("circuit-measurement-parent")
    original_branch = bench.active
    from .bench_discovery import context
    binding_context = context(bench)
    spec_hash = content_digest(spec)
    from hashlib import sha256
    from pathlib import Path
    instrument_identity = {"module": "mdb.bench_circuit_measurement",
                           "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    parent_memory = _memory(bench)
    rows, references = [], []
    completed = False
    reference_closures = None

    def record(row):
        evidence = {"schema": OBSERVATION_SCHEMA, "session_id": bench.session_id,
            "family": _family(bench), "model_identity": bench.identity, "lane": bench.lane,
            "numerical_contract": bench.capabilities()["numerical_contract"], "spec": spec,
            "instrument_identity": instrument_identity,
            "spec_sha256": spec_hash, "parent_cut": parent, **row,
            "input_authority": {"initial_token": "native-pending-token",
                "query": "caller-supplied-fixed-input", "continuation": "native-greedy"},
            "query_token_ids": spec["query_token_ids"],
            "mechanics_status": "valid", "instrument_status": row.get("instrument_status", "native-consumer-executed"),
            "trend_status": "observed" if row.get("future") is not None else "evidence-incomplete",
            "terminal_status": "not-assessed"}
        ref = bench.artifacts.put(evidence)
        references.append(ref); rows.append({**evidence, "evidence_ref": ref})
        return ref

    try:
        for site in spec["sites"]:
            bench.restore(parent); bench.step("embedding")
            for _ in range(site): bench.step("layer")
            site_cut = bench.checkpoint(f"circuit-site-{site}")
            residual = bench.resolve_value("residual").cpu()
            baseline, baseline_memory, trace = _first(bench, spec)
            baseline_future = _future(bench, spec)
            closures = (baseline["closure"]["sha256"], baseline_future["closure"]["sha256"])
            if reference_closures is not None and closures != reference_closures:
                raise BenchError("same-parent native reference closure parity failed across sites")
            reference_closures = closures
            common = {"site": site, "site_cut": site_cut, "source_address": "residual",
                      "reference_first": baseline, "reference_future": baseline_future}
            record({**common, "name": "native", "role": "reference", "replay_cut": site_cut,
                "replay_stage": "site", "act": None, "first": baseline, "future": baseline_future,
                "metrics": {"residual": _delta(residual, residual),
                    "memory": _memory_delta(baseline_memory, parent_memory, growth=_family(bench) == "qwen2"),
                    "first_effect": _effect(baseline, baseline), "future_effect": _effect(baseline_future, baseline_future)}})
            arms = []
            for index, contrast in enumerate(spec["contrasts"]):
                scalar = torch.as_tensor(contrast["value"], dtype=residual.dtype)
                proposed = residual*scalar if contrast["mode"] == "scale" else residual+scalar
                if not bool(torch.isfinite(proposed).all()): raise BenchError("contrast violates finite residual contract")
                norm = float((proposed.double()-residual.double()).norm())
                keyed_seed = int(content_digest({"seed": spec["seed"], "spec": spec_hash, "parent": parent,
                    "site": site, "arm": index})[:16], 16) % (2**63)
                generator = torch.Generator(device="cpu").manual_seed(keyed_seed)
                noise = torch.randn(residual.shape, generator=generator, dtype=torch.float64)
                noise = (noise/noise.norm().clamp_min(1e-30)*norm).to(residual.dtype)
                arms.extend([(contrast["name"], "candidate", contrast["value"], contrast["mode"], None),
                    ("random:"+contrast["name"], "matched-random", noise, "add",
                     {"seed": keyed_seed, "target_delta_l2": norm, "payload": _summary(noise)})])
            arms.extend([("noop", "noop", 1.0, "scale", None),
                         ("residual-restore", "residual-restore", residual, "replace", None)])
            for name, role, payload, mode, random in arms:
                bench.restore(site_cut)
                discarded = None
                if role == "residual-restore":
                    # Restore the original workspace after a real admitted
                    # candidate; no native layer executes between these edits.
                    contrast = spec["contrasts"][0]
                    discarded = _edit(bench, contrast["value"], contrast["mode"])
                act = _edit(bench, payload, mode)
                actual_residual = bench.resolve_value("residual").cpu()
                first, memory, _ = _first(bench, spec, trace)
                future = _future(bench, spec)
                metrics = {"residual": _delta(actual_residual, residual), "memory": _memory_delta(memory, baseline_memory),
                           "first_effect": _effect(first, baseline), "future_effect": _effect(future, baseline_future)}
                if role in {"noop", "residual-restore"} and not (
                    metrics["first_effect"]["closure_exact"] and metrics["future_effect"]["closure_exact"]
                    and all(r["delta"]["exact"] for r in first["trace"])):
                    raise BenchError("circuit no-op/restoration closure parity failed")
                record({**common, "name": name, "role": role, "replay_cut": act["cut_id"], "replay_stage": "site",
                    "act": act, "discarded_act": discarded, "random_control": random,
                    "first": first, "future": future, "metrics": metrics})
                if spec["memory_mediation"] and role == "candidate":
                    for memory_role, recipient, donor, expected in (
                        ("memory-transfer", baseline["successor_cut"], first["successor_cut"], future),
                        ("memory-restore", first["successor_cut"], baseline["successor_cut"], baseline_future)):
                        bench.restore(recipient)
                        try:
                            memory_act = bench._circuit_memory_act(donor)
                        except MemoryInterchangeError as error:
                            bench.restore(recipient)
                            record({**common, "name": memory_role+":"+name, "role": memory_role,
                                "replay_cut": None, "replay_stage": "not-applicable", "act": None,
                                "donor_cut": donor, "recipient_cut": recipient, "first": None, "future": None,
                                "metrics": {"residual": None, "memory": None, "first_effect": None, "future_effect": None},
                                "instrument_status": "not-applicable", "reason": str(error)})
                            continue
                        transferred = _memory(bench)
                        transferred_first = {"kind": "recipient-boundary-snapshot", "value": _margin(bench, spec),
                            "logits_provenance": "retained-recipient-output; no native readout after interchange",
                            "closure": _closure(bench, transferred),
                            "successor_cut": memory_act["cut_id"], "trace": [], "native_step": None}
                        transferred_future = _future(bench, spec)
                        record({**common, "name": memory_role+":"+name, "role": memory_role,
                            "replay_cut": memory_act["cut_id"], "replay_stage": "successor", "act": memory_act,
                            "donor_cut": donor, "recipient_cut": recipient,
                            "recipient_policy": "preserve-history-pending-logits; supply-fixed-query-before-consumer",
                            "pending_token_match": first["closure"]["cursor"]["pending_token"] == baseline["closure"]["cursor"]["pending_token"],
                            "first": transferred_first, "future": transferred_future,
                            "metrics": {"residual": None, "memory": _memory_delta(transferred, baseline_memory),
                                "first_effect": _effect(transferred_first, baseline),
                                "future_effect": _effect(transferred_future, baseline_future),
                                "paired_future": _effect(transferred_future, expected)}})
        completed = True
    finally:
        # Candidate cuts/receipts remain evidence, while the selected native
        # closure and sibling branches are restored even on an instrument error.
        if bench._edit:
            edit = bench._edit
            bench.abort_edit(edit["act_id"] if isinstance(edit, dict) else edit[0])
        bench.active = original_branch
        bench.restore(parent if completed else original_cut)
    qualification = {"quality_status": "measured-causal-response", "semantic_binding": False,
        "mechanics_status": "valid", "instrument_status": "native-consumer-executed",
        "trend_status": "observed", "terminal_status": "not-assessed",
        "scope": "declared parent, residual sites, doses, numerical lane and native token-margin consumer",
        "limitations": ["no semantic reasoning label, uniqueness, minimality or portability claim",
            "memory distances are diagnostics, not measured task collateral",
            "memory interchange uses caller-supplied future inputs; it does not prove autonomous reasoning",
            "residual restoration occurs before the next native layer; it proves edit reversibility",
            "Qwen KV is measured read-only; diffusion successor mediation is not qualified"]}
    # Existing symbols receive actual observations, visibly scoped as a measured
    # response rather than an inferred universal semantic name.
    from .hyperion import HyperionObservation, compile_dynamic_circuit
    observations = [HyperionObservation(observation_id=f"row-{i}", cell_id="token-margin",
        arm_id=row["name"], phase="discovery" if row["role"] == "candidate" else "control",
        arm_kind="candidate" if row["role"] == "candidate" else row["role"],
        sites=(f"{_family(bench)}.residual.{row['site']}",), schedule=((row["site"],
            row["metrics"]["residual"]["l2"] if row["metrics"]["residual"] is not None else 0.0),),
        target_effect=row["metrics"]["future_effect"]["signed_effect"], collateral_effect=None,
        continuation_effect=float(not row["metrics"]["future_effect"]["same_tokens"]),
        consumer_id=f"bench.{_family(bench)}.token-margin", evidence_ref=row["evidence_ref"],
        metadata={"role": row["role"], "counter_trend": row["metrics"]["future_effect"]["signed_effect"] < 0,
                  "first_effect": row["metrics"]["first_effect"]["signed_effect"]}) for i, row in enumerate(rows)
        if row["future"] is not None]
    symbol_context = {**binding_context, "protocol": SCHEMA, "qualification": qualification}
    support = {coordinate for row in observations for coordinate, _ in row.schedule}
    symbol = None
    symbol_status = "not-applicable-insufficient-exposure-coordinates"
    dense_support = support == set(range(max(support) + 1)) if support else False
    if len(support) >= 2 and not dense_support:
        # Hyperion v1 indexes fitted weights on a dense zero-based lattice.
        # Sparse physical cuts are valid measurements. Keep their actual layer
        # coordinates and publish evidence without inventing unmeasured sites.
        symbol_status = "not-applicable-sparse-exposure-lattice"
    elif len(support) >= 2:
        symbol = compile_dynamic_circuit(observations, symbol_id=spec["name"], logical_symbol=spec["name"],
            model_id=bench.identity["model_id"], consumer_id=f"bench.{_family(bench)}.token-margin",
            time_axis="layer", address_space="bench.residual-and-successor-memory",
            residual_definition="same-parent native effects and successor-memory interchange under explicit future inputs",
            source_artifacts=[{"reference": ref} for ref in references], context=symbol_context)
        symbol_status = "measured-causal-response-binding"
    report = {"schema": SCHEMA, "session_id": bench.session_id, "family": _family(bench),
        "model_identity": bench.identity, "lane": bench.lane, "parent_cut": parent, "spec": spec,
        "spec_sha256": spec_hash, "rows": rows, **qualification,
        "numerical_contract": bench.capabilities()["numerical_contract"],
        "binding_context": binding_context,
        "instrument_identity": instrument_identity,
        "symbol_status": symbol_status,
        "inapplicable_rows": sum(row["instrument_status"] == "not-applicable" for row in rows),
        "memory_mediation": capabilities(_family(bench))["memory_mediation"] if spec["memory_mediation"] else False,
        "native_operation_upper_bound": operations, "capture_byte_upper_bound": capture_upper_bound,
        "operation_budget_unit": "native layer-token upper bound; embeddings, readout and publication reported in wall time",
        "wall_seconds": time.perf_counter()-start,
        "timing_scope": "instrument execution, native transitions, cuts and symbol compilation; excludes final report publication"}
    previous_symbols = dict(bench.branch.symbols)
    previous_events = list(bench.events)
    try:
        symbol_ref = bench.artifacts.put(symbol) if symbol is not None else None
        report["symbol_ref"] = symbol_ref
        report_ref = bench.artifacts.put(report)
        if symbol is not None: bench.bind_symbol(symbol)
        bench.events.append({"kind": "circuit-measured", "branch": bench.active, "parent_cut": parent,
                             "report_ref": report_ref, "symbol_ref": symbol_ref})
        bench.checkpoint("circuit-measurement-published")
    except BaseException:
        bench.branch.symbols = previous_symbols; bench.events = previous_events
        bench.restore(original_cut)
        raise
    return {"report_ref": report_ref, "report": report, **qualification,
            "session_id": bench.session_id, "cut_id": bench.branch.cut}


def inspect_measurements(bench):
    from .bench_discovery import context
    current = context(bench)
    current = current.get("binding_context", current)
    rows = []
    for event in bench.events:
        if event.get("kind") != "circuit-measured": continue
        report = bench.artifacts.get(event["report_ref"])
        symbol = bench.artifacts.get(event["symbol_ref"]) if event.get("symbol_ref") else None
        bound = symbol["symbol"]["context"] if symbol is not None else report["binding_context"]
        # Binding compatibility is the same contract used by Hyperion symbols.
        base = bound.get("binding_context", {k: v for k, v in bound.items() if k in current})
        rows.append({"report_ref": event["report_ref"], "symbol_ref": event["symbol_ref"],
                     "context_compatible": base == current, "report": report})
    return rows


def _replay_evidence(bench, evidence_ref):
    if (_family(bench) not in {"mamba", "qwen2"} or bench._edit or not bench.branches
        or bench.cursor()["phase"] != "token" or bench.publication.mode != "journal"):
        raise BenchError("circuit replay requires a qualified committed token boundary, journal mode and no pending Act")
    evidence = bench.artifacts.get(evidence_ref)
    if (evidence.get("schema") != OBSERVATION_SCHEMA or evidence["family"] != _family(bench)
        or evidence["model_identity"] != bench.identity or evidence["lane"] != bench.lane
        or evidence["numerical_contract"] != bench.capabilities()["numerical_contract"]
        or content_digest(evidence["spec"]) != evidence["spec_sha256"]):
        raise BenchError("incompatible circuit observation")
    return evidence


@torch.inference_mode()
def replay_circuit(bench, evidence_ref):
    evidence = _replay_evidence(bench, evidence_ref)
    original_cut = bench.branch.cut
    parent = bench.checkpoint("circuit-replay-parent")
    completed = False
    try:
        bench.restore(evidence["parent_cut"])
        _validate(bench, evidence["spec"])
        if evidence["replay_stage"] == "not-applicable":
            completed = True
            return {"evidence_ref": evidence_ref, "exact": None, "instrument_status": "not-applicable",
                    "reason": evidence["reason"], "session_id": bench.session_id, "cut_id": parent}
        replayed_act = None
        if evidence["replay_stage"] == "successor":
            _verify_act(evidence["act"], bench.artifacts)
            bench.restore(evidence["recipient_cut"])
            replayed_act = bench._circuit_memory_act(evidence["donor_cut"])
        else: bench.restore(evidence["replay_cut"])
        if evidence["replay_stage"] == "site": first, _, _ = _first(bench, evidence["spec"])
        elif evidence["replay_stage"] == "successor":
            first = {"kind": "recipient-boundary-snapshot", "value": _margin(bench, evidence["spec"]),
                     "closure": _closure(bench, _memory(bench))}
        else: raise BenchError("invalid circuit replay stage")
        future = _future(bench, evidence["spec"])
        first_comparison = _effect(first, evidence["first"])
        future_comparison = _effect(future, evidence["future"])
        trace_exact = None
        if evidence["replay_stage"] == "site":
            trace_exact = [(r["next_layer"], r["residual"]["sha256"]) for r in first["trace"]] == [
                (r["next_layer"], r["residual"]["sha256"]) for r in evidence["first"]["trace"]]
        completed = True
        return {"evidence_ref": evidence_ref,
            "exact": first_comparison["closure_exact"] and future_comparison["closure_exact"] and trace_exact is not False,
            "first": first_comparison, "future": future_comparison,
            "trace_exact": trace_exact,
            "replayed_act": replayed_act,
            "session_id": bench.session_id, "cut_id": parent, "endpoint_cut": bench.branch.cut}
    finally: bench.restore(parent if completed else original_cut)


def _verify_act(act, artifacts):
    cut = artifacts.get(act["cut_id"])
    receipt = artifacts.get(act["receipt_ref"])
    body = dict(receipt); digest = body.pop("content_sha256")
    if content_digest(body) != digest or receipt["decision"] != "commit":
        raise ValueError("circuit Act receipt is not a valid commit")
    if cut.get("receipt_ref") and cut["receipt_ref"] != act["receipt_ref"]:
        raise ValueError("circuit Act cut/receipt mismatch")
    if act.get("donor_cut"):
        output = receipt["continuation"]["output"]
        if output["act_id"] != act["act_id"] or output["donor_cut"] != act["donor_cut"]:
            raise ValueError("circuit memory Act donor/receipt mismatch")


def verify_artifact(value, artifacts, manifest, vm_artifacts=None):
    """Verify measurement cut/evidence references without executing a model."""
    if not isinstance(value, dict) or value.get("schema") not in {SCHEMA, OBSERVATION_SCHEMA}: return False
    if value["model_identity"] != manifest["identity"] or value["lane"] != manifest["lane"]:
        raise ValueError("circuit evidence model/lane identity differs from manifest")
    if content_digest(value["spec"]) != value["spec_sha256"]: raise ValueError("circuit spec digest mismatch")
    for key in ("parent_cut", "site_cut", "replay_cut", "donor_cut", "recipient_cut", "symbol_ref"):
        if value.get(key): artifacts.get(value[key])
    if value["schema"] == SCHEMA:
        for row in value["rows"]:
            recorded = artifacts.get(row["evidence_ref"])
            if recorded != {k: v for k, v in row.items() if k != "evidence_ref"}:
                raise ValueError("circuit report row differs from frozen observation")
            verify_artifact(recorded, artifacts, manifest, vm_artifacts)
    else:
        for key in ("first", "future", "reference_first", "reference_future"):
            observation = value[key]
            if observation is None: continue
            artifacts.get(observation.get("successor_cut", observation.get("endpoint_cut")))
            closure = dict(observation["closure"]); digest = closure.pop("sha256")
            if content_digest(closure) != digest: raise ValueError("circuit closure digest mismatch")
        for key in ("act", "discarded_act"):
            if value.get(key): _verify_act(value[key], artifacts)
        def refs(node):
            if isinstance(node, dict):
                for key, child in node.items():
                    if key in {"cut", "cut_id", "parent_cut", "site_cut", "replay_cut", "donor_cut",
                               "recipient_cut", "successor_cut", "endpoint_cut", "baseline_cut"} and isinstance(child, str):
                        referenced = artifacts.get(child)
                        if referenced["identity"] != manifest["identity"] or referenced["lane"] != manifest["lane"]:
                            raise ValueError("circuit nested cut identity mismatch")
                    elif key == "receipt_ref" and isinstance(child, str):
                        try: artifacts.get(child)
                        except Exception as error:
                            from .circuit_runtime import CircuitRuntimeError
                            if not isinstance(error, CircuitRuntimeError) or not isinstance(error.__cause__, FileNotFoundError): raise
                            if vm_artifacts is None: raise
                            vm_artifacts.get(child)
                    else: refs(child)
            elif isinstance(node, list):
                for child in node: refs(child)
        refs(value)
    return True
