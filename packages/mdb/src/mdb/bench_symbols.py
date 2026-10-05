"""Bounded native-continuation experiments compiled into Hyperion artifacts."""
import math

import torch

from mdb.bench import BenchError, tensor_digest
from mdb.hyperion import HyperionObservation, compile_dynamic_circuit
from mdb.model_os_contracts import content_digest


@torch.inference_mode()
def discover(bench, spec):
    if spec.get("protocol") == "native-consumer-v1":
        from .bench_discovery import discover as consumer_discover
        return consumer_discover(bench, spec)
    with bench._lock:
        if bench._edit or bench._backend().execution:
            raise BenchError("discovery starts at a committed token boundary")
        sites = spec.get("sites", [])
        contrasts = spec.get("contrasts", [])
        horizon = spec.get("continuation_tokens", 1)
        observable = spec.get("observable", {})
        target = observable.get("token_id")
        if (len(set(sites)) < 2 or any(type(s) is not int or not 0 <= s <= bench.vm.runtime.codec.layers for s in sites)
                or not contrasts or type(horizon) is not int or not 1 <= horizon <= 4
                or observable.get("kind") != "logit" or type(target) is not int
                or not 0 <= target < bench.model.config.vocab_size):
            raise BenchError("discovery requires >=2 residual layer sites, explicit contrasts, a logit token and 1..4 continuation tokens")
        if (1 + len(sites)*(len(contrasts)+2))*horizon > spec.get("max_token_forwards", 64):
            raise BenchError("discovery continuation budget exceeded")
        for contrast in contrasts:
            if (contrast.get("mode") not in {"add", "scale"} or not isinstance(contrast.get("value"), (int, float))
                    or not math.isfinite(contrast["value"])):
                raise BenchError("contrasts require finite scalar add/scale payloads")
        parent = bench.checkpoint("discovery-parent")
        context = {"bench_model_abi": bench.model_abi, "lane": bench.lane,
            "intervention_lineage": list(bench.branch.intervention_lineage),
            "source_tokens_sha256": content_digest(bench.branch.token_ids), "session_id": bench.session_id,
            "parent_cut": parent, "spec": spec, "runtime_identity": bench.capabilities(),
            "binding_kind": "measured-candidate", "structural_names": [f"qwen.layer.{i}.residual" for i in sites],
            "coordinate_map": {str(i): site for i, site in enumerate(sorted(set(sites)))}}
        rows, cuts = [], {}
        try:
            bench._begin()
            for site in sorted(set(sites)):
                while bench.cursor()["next_layer"] < site:
                    bench.step("layer")
                cuts[site] = bench.checkpoint(f"discover-site-{site}")
            bench.generate(horizon)
            reference = bench.last_logits.detach().float().clone()
            reference_tokens = list(bench.branch.token_ids)
            baseline = bench.checkpoint("discovery-reference")
            for site in sites:
                for arm_index, arm in enumerate([*contrasts,
                        {"name": "noop", "mode": "scale", "value": 1.0},
                        {"name": "random", "mode": "random", "value": 0.01}]):
                    bench.restore(cuts[site])
                    if arm["mode"] == "random":
                        old = bench.resolve_value("residual")
                        generator = torch.Generator(device="cpu").manual_seed(spec.get("seed", 0) + site)
                        noise = torch.randn(old.shape, generator=generator).to(old.device)
                        payload = (noise/noise.norm().clamp_min(1e-12)*old.float().norm()*arm["value"]).to(old.dtype)
                        mode = "add"
                    else:
                        payload, mode = arm["value"], arm["mode"]
                    preview = bench.preview_edit("residual", payload, mode=mode)
                    act = bench.commit_edit(preview["act_id"])
                    output = bench.generate(horizon)
                    delta = bench.last_logits.float() - reference
                    effect = float(delta[0, target])
                    name = arm.get("name", f"contrast-{arm_index}")
                    dose = arm["value"]-1 if mode == "scale" else arm["value"]
                    evidence = {"schema": "saturn-bench-observation-v1", "session_id": bench.session_id,
                        "model_identity": bench.identity, "parent_cut": parent, "site_cut": cuts[site],
                        "baseline_cut": baseline, "act": act, "arm": arm, "site": site,
                        "target_effect": effect, "max_abs_logit_delta": float(delta.abs().max()),
                        "output": output, "baseline_tokens": reference_tokens,
                        "actual_tokens": list(bench.branch.token_ids), "consumer_logits_sha256": tensor_digest(bench.last_logits),
                        "memory_addresses": [f"kv_cache.layer_{max(0, site-1):03d}.{component}" for component in ("key", "value")],
                        "mechanics_status": "valid", "terminal_status": "not-assessed"}
                    evidence_ref = bench.artifacts.put(evidence)
                    control = name in {"noop", "random"}
                    rows.append(HyperionObservation(observation_id=f"site-{site}-arm-{arm_index}",
                        cell_id="logit-response", arm_id=f"arm-{arm_index}",
                        phase="control" if control else "discovery", arm_kind=name if control else "candidate",
                        sites=(f"qwen.layer.{site}.residual",), schedule=((sorted(set(sites)).index(site), float(dose)),),
                        target_effect=effect, collateral_effect=float(delta.abs().max()),
                        continuation_effect=float(reference_tokens != bench.branch.token_ids),
                        consumer_id="bench.qwen.greedy-continuation", evidence_ref=evidence_ref,
                        metadata={"target": str(target), "seed": str(spec.get("seed", 0)), "parent_cut": parent,
                                  "actual_layer": site, "contrast": arm}))
            report = compile_dynamic_circuit(rows, symbol_id=spec.get("name", "bench-candidate"),
                logical_symbol=spec.get("name", "bench-candidate"), model_id=bench.identity["model_id"],
                consumer_id="bench.qwen.greedy-continuation", time_axis="layer", address_space="bench.residual",
                residual_definition="same-parent residual intervention; signed change in declared native logit",
                source_artifacts=[{"reference": row.evidence_ref} for row in rows], context=context)
        finally:
            bench.restore(parent)
        ref = bench.bind_symbol(report)
        bench.events.append({"kind": "discovery", "parent_cut": parent, "report_ref": ref})
        bench.checkpoint("discovery-bound")
        return {"report_ref": ref, "report": report, "quality_status": "candidate-trend",
                "terminal_status": "not-assessed", "baseline_cut": baseline, "site_cuts": cuts}


def replay_observation(bench, evidence_ref):
    evidence = bench.artifacts.get(evidence_ref)
    if evidence.get("schema") == "saturn-bench-consumer-observation-v1":
        from .bench_discovery import replay_observation as consumer_replay
        return consumer_replay(bench, evidence_ref)
    if evidence["model_identity"] != bench.identity:
        raise BenchError("observation model mismatch")
    bench.restore(evidence["site_cut"])
    bench.restore(evidence["act"]["cut_id"])
    output = bench.generate(len(evidence["output"]))
    return {"same_logits": tensor_digest(bench.last_logits) == evidence["consumer_logits_sha256"],
            "same_tokens": bench.branch.token_ids == evidence["actual_tokens"], "output": output}
