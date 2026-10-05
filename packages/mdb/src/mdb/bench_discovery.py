"""Bounded native-consumer contrasts with independently prepared held contexts.

This protocol measures a declared response. It does not certify a named semantic
role, portability, a minimal circuit, or the fitted Hyperion kernel.
"""
import math
from pathlib import Path

import torch

from .bench_qwen import BenchError, tensor_digest
from .hyperion import HyperionObservation, compile_dynamic_circuit
from .model_os_contracts import content_digest

PROTOCOL = "native-consumer-v1"
SCHEMA = "saturn-bench-consumer-observation-v1"


def binding_status(context, valid):
    if not valid: return "context-invalidated"
    if context.get("protocol") == "saturn-bench-circuit-measurement-v1": return "measured-causal-response-binding"
    if context.get("protocol") != PROTOCOL: return "measured-binding"  # legacy API
    quality = context["qualification"]["quality_status"]
    return {"measured-answer-response": "measured-answer-binding",
        "measured-native-response": "measured-response-binding"}.get(quality, "candidate-binding")


def family(bench):
    return getattr(bench, "family", "qwen2")


def tokens(bench):
    return list(bench.branch.token_ids if family(bench) == "qwen2" else bench.state["token_ids"])


def context(bench):
    if family(bench) != "qwen2": return {"binding_context": bench.symbol_context()}
    return {"bench_model_abi": bench.model_abi, "lane": bench.lane,
        "source_tokens_sha256": content_digest(tokens(bench)),
        "intervention_lineage": list(bench.branch.intervention_lineage)}


def eos_ids(bench, observable):
    native = getattr(getattr(bench.model, "generation_config", None), "eos_token_id", None)
    value = observable.get("eos_token_id", native if native is not None else getattr(bench.tokenizer, "eos_token_id", None))
    ids = [] if value is None else value if isinstance(value, list) else [value]
    if any(type(i) is not int or not 0 <= i < bench.model.config.vocab_size for i in ids):
        raise BenchError("EOS must be native vocabulary IDs")
    return ids


def _validate(bench, spec):
    ar = family(bench) in {"qwen2", "mamba"}
    if spec.get("protocol") != PROTOCOL: raise BenchError("unknown discovery protocol")
    if bench._edit or (bench._backend().execution if family(bench) == "qwen2" else bench.state["execution"]):
        raise BenchError("consumer discovery starts at a committed boundary")
    sites = spec.get("sites", [])
    if len(sites) < 2 or any(type(s) is not int or s < 0 for s in sites) or len(set(sites)) != len(sites):
        raise BenchError("declare at least two distinct integer layer or denoise sites")
    width = bench.vm.runtime.codec.layers if family(bench) == "qwen2" else len(bench.adapter.layers) if ar else None
    if ar and max(sites) > width: raise BenchError("layer site outside native execution")
    if not ar and (min(sites) < bench.cursor()["step"] or max(sites) >= bench.cursor()["total_steps"]):
        raise BenchError("denoise site outside remaining timetable")
    arms = spec.get("contrasts", [])
    if not arms or len(arms) > 8: raise BenchError("declare 1..8 contrasts")
    for arm in arms:
        if (arm.get("mode") not in {"add", "scale"} or type(arm.get("value")) not in {int, float}
            or not math.isfinite(arm["value"]) or arm.get("name") in {"noop", "random"}):
            raise BenchError("contrasts require finite scalar add/scale doses and non-control names")
    held = spec.get("held_contexts", [])
    if len(held) > 8 or any(not isinstance(c, dict) or "prompt" not in c for c in held):
        raise BenchError("declare at most eight independently prepared held prompts")
    observable = spec.get("observable", {})
    kind = observable.get("kind")
    budget = spec.get("continuation_tokens", 1) if ar else bench.cursor()["total_steps"]
    if type(budget) is not int or not 1 <= budget <= 32: raise BenchError("consumer budget must be 1..32")
    if ar:
        vocab = bench.model.config.vocab_size
        if kind == "token_margin":
            ids = (observable.get("token_id"), observable.get("against_token_id"))
            if any(type(i) is not int or not 0 <= i < vocab for i in ids) or ids[0] == ids[1]:
                raise BenchError("token margin requires two different native vocabulary IDs")
        elif kind == "generated_answer":
            if bench.tokenizer is None or not isinstance(observable.get("expected"), str):
                raise BenchError("generated answer requires a tokenizer and an explicit exact expected span")
            eos = eos_ids(bench, observable)
            if not observable.get("delimiter") and not eos:
                raise BenchError("declare a delimiter or native EOS to close the answer span")
            if observable.get("delimiter") is not None and (not isinstance(observable["delimiter"], str) or not observable["delimiter"]):
                raise BenchError("answer delimiter must be a nonempty string")
            if any(not isinstance(c.get("expected", observable["expected"]), str) for c in held):
                raise BenchError("held expected answers must be exact strings")
        else: raise BenchError("AR consumer observable is token_margin or generated_answer")
        for c in [spec, *held]:
            if c is not spec and set(c)-{"prompt", "query_token_ids", "expected"}:
                raise BenchError("unknown held AR context field")
            query = c.get("query_token_ids", spec.get("query_token_ids", []))
            if not isinstance(query, list) or len(query) > 32 or any(type(i) is not int or not 0 <= i < vocab for i in query):
                raise BenchError("query inputs must be at most 32 explicit native token IDs")
            if kind == "generated_answer" and not query:
                raise BenchError("answer discovery requires explicit query inputs before the autonomous answer")
            position = spec.get("query_site_token_index", 0)
            if type(position) is not int or position < 0 or (query and position >= len(query)) or (not query and position != 0):
                raise BenchError("query site token lies outside the supplied query")
        query_length = max(len(c.get("query_token_ids", spec.get("query_token_ids", []))) for c in [spec, *held])
        held_prefill = 0
        for c in held:
            prompt = c["prompt"]
            if isinstance(prompt, str):
                if bench.tokenizer is None: raise BenchError("held text requires a tokenizer")
                held_prefill += len(bench.tokenizer.encode(prompt))
            elif isinstance(prompt, list) and prompt and all(type(i) is int and 0 <= i < vocab for i in prompt): held_prefill += len(prompt)
            else: raise BenchError("held AR prompt must be text or a nonempty token-ID list")
        operations = width * ((1+len(held)) * len(sites)*(len(arms)+3) * (budget+max(1, query_length)) + held_prefill)
    else:
        if kind != "image_mean": raise BenchError("diffusion consumer observable is image_mean of native VAE output")
        if any(set(c)-{"prompt", "seed", "expected"} for c in held): raise BenchError("held diffusion contexts use the parent's prepared geometry")
        # Charge complete denoiser traversals plus VAE and conditioning. These
        # units differ from AR layer invocations and are named in the receipt.
        operations = (1+len(held))*(1+len(sites)*(len(arms)+2))*(budget+2)
    cap = spec.get("max_native_operations")
    if type(cap) is not int or not 1 <= cap <= 100000 or operations > cap:
        raise BenchError(f"declared native operation budget too small: conservative bound {operations}")
    return ar, width, budget, operations


def _consume(bench, spec, query):
    ar = family(bench) in {"qwen2", "mamba"}
    observable = spec["observable"]
    if ar:
        # The current token is already embedded at every site. Complete it and
        # consume the rest of the caller's query through the native state machine.
        bench.step("token")
        for token in query[spec.get("query_site_token_index", 0)+1:]:
            bench.stage_input(token); bench.step("token")
        margin = None
        emitted = []
        complete = False
        span = None
        if observable["kind"] == "token_margin":
            logits = bench.last_logits
            margin = float(logits[0, observable["token_id"]].float()-logits[0, observable["against_token_id"]].float())
            for _ in range(spec.get("continuation_tokens", 1)-1): emitted.append(bench.step("token")["token_id"])
            scalar = margin
        else:
            eos = eos_ids(bench, observable)
            delimiter = observable.get("delimiter")
            for _ in range(spec.get("continuation_tokens", 1)):
                emitted.append(bench.step("token")["token_id"])
                text = bench.tokenizer.decode(emitted, skip_special_tokens=True)
                if delimiter and delimiter in text:
                    span = text[:text.index(delimiter)]; complete = True; break
                if emitted[-1] in eos:
                    span = bench.tokenizer.decode(emitted[:-1], skip_special_tokens=True); complete = True; break
            if span is None: span = bench.tokenizer.decode(emitted, skip_special_tokens=True)
            scalar = float(complete and span == observable["expected"])
        value = bench.last_logits
        return {"value": scalar, "margin": margin, "emitted_tokens": emitted, "answer_span": span,
            "answer_complete": complete, "completion_status": "complete" if complete else "budget-exhausted" if span is not None else "not-applicable",
            "consumer_sha256": tensor_digest(value), "tokens": tokens(bench),
            "collateral_status": "not-measured", "collateral_effect": None}
    bench.generate()
    image = bench.resolve_value("image")
    return {"value": float(image.float().mean()), "consumer_sha256": tensor_digest(image),
        "tokens": [], "emitted_tokens": [], "collateral_status": "not-measured", "collateral_effect": None}


def _child(bench, held, root):
    from .bench import BenchSession
    kwargs = dict(root=root, model_id=bench.identity["model_id"], revision=bench.identity["revision"],
        model_sha256=bench.identity["weights_sha256"], publication_mode="legacy", continuation_capacity=bench.capacity)
    if family(bench) == "qwen2":
        kwargs.update(lane=bench.lane, attention_backend=bench.vm.runtime.backend, resident_kv_bytes=bench.vm.resident_kv_bytes)
    else: kwargs.update(_adapter=bench.adapter, resident_state_bytes=bench.pages.budget)
    child = BenchSession.from_resident(bench.model, bench.tokenizer, **kwargs)
    try:
        if family(bench) in {"qwen2", "mamba"}: child.prefill(held["prompt"])
        else:
            geometry = bench.state["checkpoint"]
            prepared = bench.state["context"]
            child.prepare(held["prompt"], seed=held.get("seed", 0), height=prepared["height"], width=prepared["width"],
                num_inference_steps=geometry["total_steps"], guidance_scale=prepared["guidance_scale"])
    except BaseException:
        child.close()
        raise
    return child


@torch.inference_mode()
def discover(bench, spec):
    ar, width, budget, operations = _validate(bench, spec)
    if ar and spec["observable"]["kind"] == "generated_answer":
        # Freeze native completion policy in the evidence. Later caller changes
        # to generation_config must not redefine an already observed answer.
        spec = {**spec, "observable": {**spec["observable"], "eos_token_id": eos_ids(bench, spec["observable"])}}
    parent = bench.checkpoint("consumer-discovery-parent")
    binding_context = context(bench)
    rows, references, mechanics, competent = [], [], [], []
    contexts = [("discovery", None), *[("heldout", c) for c in spec.get("held_contexts", [])]]
    try:
        for context_index, (phase, held) in enumerate(contexts):
            bench.restore(parent)
            relative = None if held is None else f"discovery-contexts/{content_digest({'parent': parent, 'spec': spec, 'index': context_index})}"
            current = bench if held is None else _child(bench, held, bench.root/relative)
            local_spec = {**spec, "observable": {**spec["observable"]}}
            if held is not None and "expected" in held: local_spec["observable"]["expected"] = held["expected"]
            query = (held or {}).get("query_token_ids", spec.get("query_token_ids", []))
            local_parent = current.checkpoint("consumer-context-parent")
            try:
                for site in spec["sites"]:
                    current.restore(local_parent)
                    if ar:
                        if query:
                            position = spec.get("query_site_token_index", 0)
                            for token in query[:position]: current.stage_input(token); current.step("token")
                            current.stage_input(query[position])
                        else: current.step("embedding")
                        while current.cursor()["next_layer"] < site: current.step("layer")
                        path = "residual"
                    else:
                        while current.cursor()["step"] < site: current.step("denoise")
                        path = "latent"
                    site_cut = current.checkpoint(f"consumer-site-{site}")
                    baseline = _consume(current, local_spec, query)
                    baseline_cut = current.checkpoint("consumer-reference")
                    if baseline.get("answer_complete") and baseline["value"] == 1:
                        competent.append({"context": context_index, "phase": phase, "site": site})
                    arms = [*spec["contrasts"], {"name": "noop", "mode": "scale", "value": 1.0},
                        {"name": "random", "mode": "random", "value": 0.0}]
                    for arm_index, arm in enumerate(arms):
                        current.restore(site_cut)
                        before = current.resolve_value(path)
                        payload, mode = arm["value"], arm["mode"]
                        if mode == "random":
                            norm = max(float(before.float().norm())*abs(c["value"]-1) if c["mode"] == "scale"
                                else math.sqrt(before.numel())*abs(c["value"]) for c in spec["contrasts"])
                            noise = torch.randn(before.shape, generator=torch.Generator().manual_seed(spec.get("seed", 0)+site))
                            payload = (noise/noise.norm().clamp_min(1e-12)*norm).to(device=before.device, dtype=before.dtype); mode = "add"
                        preview = current.preview_edit(path, payload, mode=mode)
                        act = current.commit_edit(preview["act_id"])
                        actual = _consume(current, local_spec, query)
                        exact = actual == baseline
                        name = arm.get("name", f"contrast-{arm_index}")
                        if name == "noop" and not exact: raise BenchError("native consumer no-op parity failed")
                        endpoint = current.checkpoint("consumer-observed")
                        evidence = {"schema": SCHEMA, "protocol": PROTOCOL, "model_identity": bench.identity,
                            "session_id": current.session_id, "root_relative": relative, "parent_cut": local_parent,
                            "site_cut": site_cut, "baseline_cut": baseline_cut, "endpoint_cut": endpoint,
                            "act": act, "site": site, "path": path, "arm": arm, "context_phase": phase,
                            "query_site_token_index": spec.get("query_site_token_index", 0),
                            "context_index": context_index, "spec": local_spec, "query_token_ids": query,
                            "input_authority": "caller-supplied-query/native-autonomous-answer", "baseline": baseline,
                            "actual": actual, "signed_effect": actual["value"]-baseline["value"],
                            "mechanics_status": "valid", "instrument_status": "native-consumer-executed",
                            "terminal_status": "not-assessed", "no_op_exact": exact if name == "noop" else None,
                            "runtime_addresses": [path, "kv_cache" if family(bench) == "qwen2" else "ssm_state" if ar else "conditioner"],
                            "payload_policy": "current-context typed edit; no stale donor endpoint"}
                        ref = bench.artifacts.put(evidence); references.append(ref)
                        control = name in {"noop", "random"}
                        dose = arm["value"]-1 if arm["mode"] == "scale" else arm["value"]
                        if name == "random": dose = norm
                        rows.append(HyperionObservation(observation_id=f"context-{context_index}-site-{site}-arm-{arm_index}",
                            cell_id=spec["observable"]["kind"], arm_id=f"arm-{arm_index}", phase="control" if control else phase,
                            arm_kind=name if control else "candidate", sites=(f"{family(bench)}.{path}.{site}",),
                            schedule=((spec["sites"].index(site), float(dose)),), target_effect=evidence["signed_effect"],
                            collateral_effect=None, continuation_effect=float(actual["tokens"] != baseline["tokens"]),
                            consumer_id=f"bench.{family(bench)}.native-consumer", evidence_ref=ref,
                            metadata={"context_phase": phase, "context_index": context_index, "site": site,
                                "reference": baseline, "actual": actual, "contrast": arm, "counter_trend": evidence["signed_effect"] < 0}))
                        mechanics.append({"context": context_index, "site": site, "arm": name, "no_op_exact": evidence["no_op_exact"]})
            finally:
                current.restore(local_parent)
                if held is not None: current.close()
    finally: bench.restore(parent)
    measured = bool(spec.get("held_contexts"))
    semantic = any(c["phase"] == "heldout" for c in competent)
    qualification = {"quality_status": "measured-answer-response" if semantic else "measured-native-response" if measured else "candidate-trend",
        "semantic_binding": semantic, "competent_context_sites": competent,
        "scope": "declared contexts, sites, doses and native consumer only; semantic evidence limited to completed competent contexts",
        "mechanics_status": "valid", "instrument_status": "native-consumer-executed",
        "terminal_status": "not-assessed", "operation_budget_unit": "AR layer-token upper bound" if ar else "denoiser traversal/VAE/conditioner",
        "native_operation_upper_bound": operations, "mechanics": mechanics,
        "limitations": ["no measured collateral", "no minimality or portable semantic role", "Hyperion fit is diagnostic"]}
    binding_context.update(protocol=PROTOCOL, qualification=qualification)
    report = compile_dynamic_circuit(rows, symbol_id=spec.get("name", "bench-response"),
        logical_symbol=spec.get("name", "bench-response"), model_id=bench.identity["model_id"],
        consumer_id=f"bench.{family(bench)}.native-consumer", time_axis="layer" if ar else "denoise-step",
        address_space="bench.residual" if ar else "bench.latent",
        residual_definition="same-parent signed native consumer response, independently prepared held contexts",
        source_artifacts=[{"reference": ref} for ref in references], context=binding_context)
    report_ref = bench.bind_symbol(report)
    bench.checkpoint("consumer-discovery-bound")
    return {"report": report, "report_ref": report_ref, **qualification}


def replay_observation(bench, evidence_ref):
    evidence = bench.artifacts.get(evidence_ref)
    if evidence.get("schema") != SCHEMA or evidence["model_identity"] != bench.identity:
        raise BenchError("incompatible consumer observation")
    parent = bench.checkpoint("consumer-replay-parent")
    current = bench
    try:
        relative = evidence["root_relative"]
        if relative is not None:
            from .bench import BenchSession
            path = (bench.root/relative).resolve()
            if not path.is_relative_to((bench.root/"discovery-contexts").resolve()): raise BenchError("invalid observation context root")
            current = BenchSession.resume(bench.model, bench.tokenizer, root=path,
                model_sha256=bench.identity["weights_sha256"], _adapter=getattr(bench, "adapter", None))
        current.restore(evidence["act"]["cut_id"])
        actual = _consume(current, evidence["spec"], evidence["query_token_ids"])
        return {"evidence_ref": evidence_ref, "exact": actual == evidence["actual"],
            "same_logits": actual["consumer_sha256"] == evidence["actual"]["consumer_sha256"],
            "same_tokens": actual["tokens"] == evidence["actual"]["tokens"], "actual": actual,
            "session_id": current.session_id, "cut": current.branch.cut}
    finally:
        if current is not bench: current.close()
        bench.restore(parent)
