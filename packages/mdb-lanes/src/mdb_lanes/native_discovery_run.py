"""One observation-only capture, fit, forecast and symbol-export pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .analysis_metrics import fit_mean_writes, mean_write_baselines, mean_write_forecast
from .attention_ranking import trivial_route_baselines
from .native_discovery import (
    SCHEMA,
    SHARED_PREFIX_REFUSAL,
    DiscoveryError,
    cross_split_prefixes,
    digest,
    error_metrics,
    file_hash,
    load_corpus,
    save_json,
    sealed,
)
from .native_discovery_capture import boundaries_for, capture_document
from .native_discovery_learning import (
    KINDS,
    fit_predictors,
    forecast_arrays,
    resolve_arms,
    score_forecast,
)
from .native_discovery_nominator import fit_contrast_nominator, nominate_sources, score_nominations
from .native_discovery_symbols import discover_components, register_observations


def local_basis_baseline(discovery, targets, *, ridge=0.01):
    """Paid native-neuron coordinates, distinct from first-frontier forecasts."""
    stages = discovery[0][1]["states"].shape[0] - 1
    result = {}
    for stage in range(stages):
        x = np.concatenate([a["neurons"][stage] for _, a in discovery]).astype(np.float64)
        y = np.concatenate([a["states"][stage + 1] - a["states"][stage]
                            for _, a in discovery]).astype(np.float64)
        center, scale = x.mean(0), max(float(x.std()), 1e-6)
        x = np.column_stack([np.ones(len(x)), (x - center) / scale])
        weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
        for record, arrays in targets:
            z = np.column_stack([np.ones(len(arrays["neurons"][stage])),
                                 (arrays["neurons"][stage] - center) / scale])
            prediction = arrays["states"][stage] + z @ weights
            result.setdefault(record["document_id"], []).append(
                {"stage": stage, **error_metrics(prediction, arrays["states"][stage + 1])})
    return {"scope": "local conditional readout after paid native neuron formation; not a sealed own-state forecast",
            "documents": result}


def run_on_model(model, tokenizer, *, corpus, out, binding, max_tokens=64, stages=3,
                 features=32, steps=60, interval_steps=10, seed=0, kinds=KINDS,
                 max_wall_seconds=600, refuse_shared_prefix=SHARED_PREFIX_REFUSAL,
                 arms=None, sealed_position_split=False, contrast_nominator=False):
    """Capture, fit, seal held-out forecasts, observe, score and export.

    Explicit, recorded options (defaults reproduce the original runs):
    ``arms``: learned-arm preset or list (``native_discovery_learning.ARM_PRESETS``; ``None`` fits
    the original configuration of every kind in ``kinds``). ``sealed_position_split``: seal a
    position-split mean-write baseline (position 0 vs rest, fit on discovery) with the forecasts.
    ``contrast_nominator``: seal a learning-free per-(layer, head) source nominator built by
    contrast ranking over the discovery documents' native attention.
    """
    from .forecast_transaction import ForecastTransaction

    if (max_tokens < 2 or stages < 1 or features < 1 or max_wall_seconds <= 0
            or not np.isfinite(max_wall_seconds)):
        raise DiscoveryError("invalid collection or fitting budget")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise DiscoveryError("analysis requires a frozen evaluation model")
    corpus = load_corpus(corpus)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    base = getattr(model, "model", model)
    boundaries = boundaries_for(len(base.layers), stages)
    device = str(next(model.parameters()).device)
    kv_groups = getattr(getattr(model, "config", None), "num_key_value_heads", None)
    arm_specs = resolve_arms(list(kinds) if arms is None else arms, kv_groups=kv_groups)
    options = {"arms": arm_specs, "sealed_position_split_mean_write": bool(sealed_position_split),
               "contrast_nominator": bool(contrast_nominator), "native_kv_groups": kv_groups}
    checked_binding = {"model_identity": binding["model_identity"], "execution": binding["execution"]}
    if not checked_binding["model_identity"] or not isinstance(checked_binding["execution"], dict):
        raise DiscoveryError("model and numerical execution identities are required")
    save_json(out / "corpus.json", corpus)
    save_json(out / "binding.json", checked_binding)
    visible, seen_inputs, windows = [], {}, []
    for row in corpus["documents"]:
        tokens = tokenizer(row["text"], add_special_tokens=False, truncation=True,
                           max_length=max_tokens)["input_ids"]
        if len(tokens) < 2:
            raise DiscoveryError("a visible document has fewer than two tokens")
        identity = digest(tokens)
        if seen_inputs.setdefault(identity, row["split"]) != row["split"]:
            raise DiscoveryError("identical visible token prefixes cross corpus splits")
        windows.append((row["split"], tokens))
        visible.append({"document": row["id"], "split": row["split"], "tokens": len(tokens),
                        "token_ids_sha256": identity})
    # Shared leading token runs give identical causal states; refuse long cross-split runs.
    for entry, shared in zip(visible, cross_split_prefixes(windows, refuse_at=refuse_shared_prefix),
                             strict=True):
        entry["longest_cross_split_shared_prefix"] = shared
        entry["shared_prefix_refusal_tokens"] = refuse_shared_prefix
    save_json(out / "visible-inputs.json", visible)
    source = {p.name: file_hash(p) for p in Path(__file__).parent.glob("native_discovery*.py")}
    source.update({p.name: file_hash(p) for p in (Path(__file__).parent / "analysis_metrics.py",
                                                 Path(__file__).parent / "attention_ranking.py")})
    print("DISCOVERY_SOURCE " + json.dumps(source, sort_keys=True), flush=True)
    save_json(out / "source.json", source)
    trace_root = out / "traces"
    trace_root.mkdir()
    captures = {"discovery": [], "development": [], "heldout": []}
    consumer_calls = 0

    def consume_last(state):
        """Paid frozen native final norm/head, applied only to the last token."""
        nonlocal consumer_calls
        with torch.inference_mode():
            x = torch.as_tensor(np.array(state[-1:], copy=True),
                                device=next(model.parameters()).device, dtype=torch.float32)
            logits = model.lm_head(base.norm(x))[0].detach().float().cpu().numpy().copy()
        consumer_calls += 1
        return logits

    def predict(predictor, context):
        result = forecast_arrays(predictor, context.frontiers["initial"])
        result["consumer_logits"] = consume_last(result["state"][-1])
        return {"predictions": result}

    def output_feedback(predicted, target):
        """Differentiate the predicted future through the frozen native consumer."""
        nonlocal consumer_calls
        logits = model.lm_head(base.norm(predicted[-1:])).float()
        with torch.no_grad():
            reference = model.lm_head(base.norm(target[-1:])).float()
            log_target = reference.log_softmax(-1)
        consumer_calls += 2
        return (log_target.exp() * (log_target - logits.log_softmax(-1))).sum(-1).mean()

    def consumer_metrics(logits, native):
        p = torch.from_numpy(np.array(logits, copy=True)).double()
        t = torch.from_numpy(np.array(native, copy=True)).double()
        lp, lt = p.log_softmax(-1), t.log_softmax(-1)
        return {"native_to_predicted_kl": float((lt.exp() * (lt - lp)).sum()),
                "top1_agreement": int(p.argmax() == t.argmax()),
                "native_top1": int(t.argmax()), "predicted_top1": int(p.argmax()),
                "vocabulary_size": len(native), "position": "last supplied token",
                "consumer": "paid frozen native final norm and LM head"}

    def budget():
        if time.monotonic() - started > max_wall_seconds:
            raise DiscoveryError("finite analysis wall budget exhausted")

    def observe(row):
        budget()
        result = capture_document(model, tokenizer, row, root=trace_root, binding=checked_binding,
                                  boundaries=boundaries, max_tokens=max_tokens,
                                  neuron_budget=features)
        captures[row["split"]].append(result)
        return result

    for split in ("discovery", "development"):
        for row in corpus["documents"]:
            if row["split"] == split:
                observe(row)
        print("DISCOVERY_CAPTURE " + json.dumps({"split": split, "documents": len(captures[split])}), flush=True)
    mean_write = np.stack([np.concatenate([a["states"][i + 1] - a["states"][i]
                        for _, a in captures["discovery"]]).mean(0) for i in range(stages)])
    # Post-hoc position-split mean writes (position 0 vs rest), fit on discovery only.
    split_writes = fit_mean_writes([a["states"] for _, a in captures["discovery"]])
    budget()
    metadata, predictors = fit_predictors(captures["discovery"], captures["development"],
        root=out / "predictor", binding=checked_binding, boundaries=boundaries,
        features=features, steps=steps, interval_steps=interval_steps, seed=seed, device=device,
        kinds=kinds, budget=budget, output_consumer=output_feedback, arms=arm_specs)
    nominator = None
    if contrast_nominator:
        nominator = fit_contrast_nominator([a["routes"] for _, a in captures["discovery"]],
                                           document_ids=[r["document_id"] for r, _ in captures["discovery"]])
        save_json(out / "contrast-nominator.json", sealed(nominator))
    catalog_arm = "sparse_consumer" if "sparse_consumer" in predictors else next(iter(predictors))
    catalog = discover_components(predictors[catalog_arm], captures["discovery"],
                                  root=out, predictor_identity=metadata["fingerprint"],
                                  accepted_updates=metadata["arms"][catalog_arm]["accepted_updates"])
    held_results, symbol_counts = [], []
    for row in corpus["documents"]:
        if row["split"] != "heldout":
            continue
        budget()
        encoded = tokenizer(row["text"], add_special_tokens=False, truncation=True,
                            max_length=max_tokens)
        ids = torch.tensor([encoded["input_ids"]], device=next(model.parameters()).device)
        with torch.inference_mode():
            initial = base.embed_tokens(ids)[0].detach().float().cpu().numpy().copy()
        # Only embedding-prefix state is exposed. No held trace exists yet.
        identity = digest([row["id"], row["text_sha256"], metadata["fingerprint"]])
        forecast_root = out / "forecasts" / identity
        predictions = {}
        with ForecastTransaction(forecast_root, model_identity=checked_binding,
            input_identity={"document": row["id"], "text_sha256": row["text_sha256"]},
            source_identity={"analysis": source, "predictor": metadata["fingerprint"]},
            frontier={"kind": "architectural", "source": "paid native token embedding prefix"},
            execution_scope="Externally managed native eager decoder; no bound-model guards. Predictions see only embedding frontier and learned predictor, not native future states.",
            budgets={"max_bytes": 128 * 1024 * 1024, "max_wall_ms": int(max_wall_seconds * 1000)}) as transaction:
            transaction.frontier("initial", initial)
            for kind, predictor in predictors.items():
                transaction.predict(kind, lambda context, net=predictor: predict(net, context))
                predictions[kind] = transaction.prediction(kind)
            baselines = [("hold_state", np.repeat(initial[None], stages, axis=0)),
                         ("mean_write", initial[None] + mean_write.cumsum(0)[:, None])]
            if sealed_position_split:
                baselines.append(("split_mean_write", mean_write_forecast(
                    initial, split_writes, split=True).astype(np.float32)))
            for name, states in baselines:
                transaction.predict(name, lambda context, states=states: {"predictions": {
                    "state": states, "consumer_logits": consume_last(states[-1])}})
            baseline_predictions = {name: transaction.prediction(name) for name, _ in baselines}
            if nominator is not None:
                transaction.predict("contrast_nominator", lambda context: {"predictions": {
                    "sources": nominate_sources(nominator, len(context.frontiers["initial"]))}})
                nominated = transaction.prediction("contrast_nominator")
            transaction.seal()
            observed = []

            def release(context):
                record, arrays = observe(row)
                if not np.array_equal(initial, arrays["states"][0]):
                    raise DiscoveryError("embedding frontier does not equal native initial boundary")
                observed.append((record, arrays, consume_last(arrays["states"][-1])))
                return {"trace": record, "native_states": arrays["states"],
                        "native_routes": arrays["routes"], "native_values": arrays["values"],
                        "consumer_logits": observed[-1][2]}

            transaction.execute("ordinary_native", release)
            transaction_report = transaction.report()
        record, arrays, native_logits = observed[0]
        arms = {kind: score_forecast(p, arrays) for kind, p in predictions.items()}
        consumers = {kind: consumer_metrics(p["consumer_logits"], native_logits)
                     for kind, p in predictions.items()}
        hold = [error_metrics(initial, target) for target in arrays["states"][1:]]
        mean_error = [error_metrics(p, t) for p, t in
                      zip(baseline_predictions["mean_write"]["state"], arrays["states"][1:], strict=True)]
        # A position-only source control needs neither labels nor native source values.
        uniform = np.tril(np.ones(arrays["routes"].shape[-2:], dtype=np.float32))
        uniform /= uniform.sum(-1, keepdims=True)
        baseline_kl = float((arrays["routes"] * (np.log(np.maximum(arrays["routes"], 1e-8))
                            - np.log(np.maximum(uniform, 1e-8)))).sum(-1).mean())
        position_baselines = mean_write_baselines(initial, arrays["states"], split_writes)
        route_baselines = [trivial_route_baselines(native) for native in arrays["routes"]]
        symbols = register_observations(record, arrays, predictions[catalog_arm],
                    root=out / "symbols" / identity, catalog=catalog)
        symbol_counts.append(symbols)
        optional = {}
        if sealed_position_split:
            optional["split_mean_write"] = [error_metrics(p, t) for p, t in zip(
                baseline_predictions["split_mean_write"]["state"], arrays["states"][1:], strict=True)]
            optional["split_mean_write_consumer"] = consumer_metrics(
                baseline_predictions["split_mean_write"]["consumer_logits"], native_logits)
            optional["split_mean_write_scope"] = "sealed with the forecasts; position-split mean write fit on discovery documents only"
        if nominator is not None:
            optional["contrast_nominator"] = score_nominations(nominated["sources"], arrays["routes"])
        held_results.append({**optional, "document": row["id"], "group": row["group"], "arms": arms,
                             "native_output_consumer": consumers,
                             "hold_state_consumer": consumer_metrics(baseline_predictions["hold_state"]["consumer_logits"], native_logits),
                             "mean_write": mean_error,
                             "mean_write_consumer": consumer_metrics(baseline_predictions["mean_write"]["consumer_logits"], native_logits),
                             "hold_state": hold, "uniform_route_kl": baseline_kl,
                             "position_split_mean_write": position_baselines["position_split_mean_write"],
                             "position_split_mean_write_scope": "post-hoc baseline fit on discovery documents only; not sealed; state metrics only",
                             "route_top1_trivial_baselines": route_baselines,
                             "forecast_seal": transaction_report["seal_sha256"],
                             "symbol_records": symbols})
        print("DISCOVERY_FORECAST " + json.dumps({"document": row["id"], "symbols": symbols["bindings"]}), flush=True)
    report = {"schema": SCHEMA + "/report", "binding": checked_binding,
        "corpus": corpus["fingerprint"], "predictor": metadata["fingerprint"],
        "boundaries": list(boundaries), "heldout": held_results,
        "local_native_neuron_baseline": local_basis_baseline(captures["discovery"], captures["heldout"]),
        **({"options": options} if (arms is not None or sealed_position_split or contrast_nominator) else {}),
        "components": len(catalog["components"]), "role_candidates": catalog["groups"],
        "role_candidates_nonempty": catalog.get("nonempty_groups"),
        "bindings": sum(s["bindings"] for s in symbol_counts),
        "work": {"model_loads": 1, "physical_document_forwards": sum(len(v) for v in captures.values()),
                 "physical_batch": 1, "embedding_prefix_calls": len(held_results),
                 "paid_last_token_norm_head_calls": consumer_calls,
                 "captured_bytes": sum(r["work"]["captured_bytes"] for v in captures.values() for r, _ in v),
                 "analysis_seconds": time.monotonic() - started, "interventions": 0,
                 "training": {kind: a["physical_work"] for kind, a in metadata["arms"].items()}},
        "mechanics_status": "valid", "instrument_status": "native state/read forecasts; anonymous components",
        "trend_status": "per-document and per-span observations retained",
        "terminal_status": "not-requested",
        "scope": {"continuation": "own predicted states across depth spans; supplied ordinary token history",
                  "semantics": "learned anonymous feature and role candidates; no automatic ontology certification",
                  "symbols": "public canonical symbols over passive native observations, not resumable decoder sessions",
                  "native_neurons": "separate paid-prefix local diagnostic; not a first-frontier forecast",
                  "remaining_native_work": "all teacher layers run for measurement; no serving speedup claim"}}
    save_json(out / "report.json", report)
    from .native_discovery_export import export_catalog
    export_catalog(out, catalog, checked_binding, report)
    files = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    save_json(out / "manifest.json", sealed({"schema": SCHEMA + "/run", "files": files,
                                           "corpus": corpus["fingerprint"],
                                           "predictor": metadata["fingerprint"]}))
    return {"root": str(out), "report": "report.json", "heldout_documents": len(held_results),
            "components": report["components"], "bindings": report["bindings"],
            "interventions": 0, "mechanics_status": "valid"}


def parameter_manifest(model_path, model, root):
    """Keep immutable original archive identities instead of copying pretrained tensors."""
    import struct
    tensors, archives = {}, {}
    for path in sorted(Path(model_path).glob("*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        sha = file_hash(path)
        archives[path.name] = {"sha256": sha, "header_sha256": digest(header), "bytes": path.stat().st_size}
        for name, value in header.items():
            if name != "__metadata__":
                tensors[name] = {"archive": path.name, **value}
    required = {name: {"shape": list(p.shape), "dtype": str(p.dtype),
                        "archive_entry": tensors.get(name)} for name, p in model.named_parameters()}
    if not archives or any(v["archive_entry"] is None for v in required.values()):
        raise DiscoveryError("original safetensors headers do not cover required named parameters")
    manifest = sealed({"schema": SCHEMA + "/parameter-artifacts", "checkpoint": str(model_path),
                       "archives": archives, "required_tensors": required,
                       "retention": "pretrained constants remain in original checkpoint"})
    save_json(root / "parameter-artifacts.json", manifest)
    script = '''"""Verify required original checkpoint archives without importing a model framework."""
import argparse, hashlib, json, pathlib, struct
p=argparse.ArgumentParser(); p.add_argument("checkpoint",type=pathlib.Path); a=p.parse_args()
m=json.loads((pathlib.Path(__file__).parent/"parameter-artifacts.json").read_text())
for name, spec in m["archives"].items():
    path=a.checkpoint/name; h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    if h.hexdigest()!=spec["sha256"]: raise ValueError("archive hash differs: "+name)
    with path.open("rb") as f:
        size=struct.unpack("<Q",f.read(8))[0]; header=json.loads(f.read(size))
    for tensor, entry in m["required_tensors"].items():
        if entry["archive_entry"]["archive"]==name:
            expected={k:v for k,v in entry["archive_entry"].items() if k!="archive"}
            if header.get(tensor)!=expected: raise ValueError("tensor header differs: "+tensor)
print(json.dumps({"verified_archives":len(m["archives"]),"required_tensors":len(m["required_tensors"])}))
'''
    (root / "retrieve_weights.py").write_text(script)
    return manifest


def run_pretrained(*, corpus, model, out, device="cuda", max_tokens=64, stages=3,
                   features=32, steps=60, interval_steps=10, seed=0, max_wall_seconds=600,
                   refuse_shared_prefix=SHARED_PREFIX_REFUSAL, arms=None,
                   sealed_position_split=False, contrast_nominator=False):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise DiscoveryError("CUDA requested but unavailable")
    checkpoint = Path(model).resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") not in {"qwen2", "qwen3"}:
        raise DiscoveryError("initial native capture qualification supports Qwen2/Qwen3")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    native = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
              dtype=torch.float32, attn_implementation="eager").to(device).eval().requires_grad_(False)
    archives = {p.name: file_hash(p) for p in sorted(checkpoint.glob("*.safetensors"))}
    tokenizer_files = {p.name: file_hash(p) for p in checkpoint.iterdir() if p.is_file()
        and p.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                       "vocab.json", "merges.txt", "added_tokens.json"}}
    binding = {"model_identity": digest({"config": config, "archives": archives}),
               "execution": {"backend": "hf-eager", "dtype": "float32", "device": device,
                             "model_type": config["model_type"], "torch": torch.__version__,
                             "transformers": transformers.__version__, "tokenizer_files": tokenizer_files,
                             "prompt_format": "raw corpus text; no chat template; no added special tokens",
                             "physical_batch": 1, "cache": False}}
    result = run_on_model(native, tokenizer, corpus=corpus, out=out, binding=binding,
                          max_tokens=max_tokens, stages=stages, features=features, steps=steps,
                          interval_steps=interval_steps, seed=seed, max_wall_seconds=max_wall_seconds,
                          refuse_shared_prefix=refuse_shared_prefix, arms=arms,
                          sealed_position_split=sealed_position_split,
                          contrast_nominator=contrast_nominator)
    # Publish retrieval metadata and extend custody without altering prior artifacts.
    root = Path(out)
    parameter_manifest(checkpoint, native, root)
    save_json(root / "parameter-custody.json", sealed({
        "schema": SCHEMA + "/parameter-custody", "run_manifest_sha256": file_hash(root / "manifest.json"),
        "files": {p: file_hash(root / p) for p in ("parameter-artifacts.json", "retrieve_weights.py")}}))
    return result
