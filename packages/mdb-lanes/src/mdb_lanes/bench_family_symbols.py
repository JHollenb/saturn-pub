"""Bounded same-parent native interventions for recurrent and image families."""
import math
import torch
from .bench_qwen import BenchError, tensor_digest
from .hyperion import HyperionObservation, compile_dynamic_circuit


def replay_observation(bench, evidence_ref):
    evidence = bench.artifacts.get(evidence_ref)
    if evidence.get("schema") == "saturn-bench-consumer-observation-v1":
        from .bench_discovery import replay_observation as consumer_replay
        return consumer_replay(bench, evidence_ref)
    if evidence.get("schema") != "saturn-bench-family-observation-v2" or evidence["model_identity"] != bench.identity:
        raise BenchError("incompatible family observation")
    parent = bench.checkpoint("observation-replay-parent")
    try:
        bench.restore(evidence["act"]["cut_id"])
        if bench.family == "mamba":
            bench.generate(evidence["horizon"]); value = bench.last_logits
        else:
            for _ in range(min(evidence["horizon"], bench.cursor()["total_steps"]-bench.cursor()["step"])):
                bench.step("denoise")
            image = evidence["observable"]["kind"] == "image_mean"
            if image: bench.generate()
            value = bench.resolve_value("image" if image else "latent")
        digest = tensor_digest(value)
        return {"evidence_ref": evidence_ref, "exact": digest == evidence["consumer_sha256"],
                "consumer_sha256": digest, "cut": bench.branch.cut}
    finally: bench.restore(parent)


def discover(bench, spec):
    if spec.get("protocol") == "native-consumer-v1":
        from .bench_discovery import discover as consumer_discover
        return consumer_discover(bench, spec)
    sites = spec.get("sites", [])
    contrasts = spec.get("contrasts", [])
    observable = spec.get("observable", {})
    horizon = spec.get("continuation_tokens", 1) if bench.family == "mamba" else spec.get("continuation_steps", 1)
    if (len(set(sites)) < 2 or any(type(s) is not int or s < 0 for s in sites)
        or not contrasts or type(horizon) is not int or not 1 <= horizon <= 4
        or (1+len(sites)*(len(contrasts)+2))*horizon > spec.get("max_native_forwards", 64)):
        raise BenchError("discovery requires explicit sites, contrasts and a bounded continuation budget")
    if any(c.get("mode") not in {"add", "scale"} or not isinstance(c.get("value"), (int, float))
           or not math.isfinite(c["value"]) for c in contrasts):
        raise BenchError("contrasts require finite scalar add/scale values")
    if bench.state["execution"] is not None: raise BenchError("discovery starts at a committed native boundary")
    kind = observable.get("kind")
    if bench.family == "mamba":
        target = observable.get("token_id")
        if kind != "logit" or type(target) is not int or not 0 <= target < bench.model.config.vocab_size:
            raise BenchError("Mamba discovery requires a valid native logit observable")
        if max(sites) >= len(bench.adapter.layers): raise BenchError("Mamba site outside layer range")
    elif kind not in {"latent_mean", "image_mean"}: raise BenchError("diffusion observable requires latent_mean or image_mean")
    if bench.family != "mamba":
        remaining = bench.cursor()["total_steps"]-bench.cursor()["step"]
        if remaining <= 0: raise BenchError("discovery requires an unfinished denoise schedule")
        native_horizon = remaining if kind == "image_mean" else min(horizon, remaining)
        if (1+len(sites)*(len(contrasts)+2))*native_horizon > spec.get("max_native_forwards", 64):
            raise BenchError("image observable exceeds declared native continuation budget")
    parent = bench.checkpoint("discovery-parent")
    context = bench.symbol_context(); cuts = {}; rows = []
    def consume():
        if bench.family == "mamba":
            output = bench.generate(horizon); value = bench.last_logits
            scalar = float(value[0, target]); tokens = bench.state["token_ids"][:]
        else:
            output = [bench.step("denoise") for _ in range(min(horizon,
                bench.cursor()["total_steps"]-bench.cursor()["step"]))]
            if kind == "image_mean": output += bench.generate()
            value = bench.resolve_value("image" if kind == "image_mean" else "latent")
            scalar = float(value.float().mean()); tokens = []
        return scalar, tensor_digest(value), tokens, output
    try:
        for site in sorted(sites):
            if bench.family == "mamba":
                if bench.state["execution"] is None: bench.step("embedding")
                while bench.cursor()["next_layer"] < site: bench.step("layer")
            else:
                while bench.cursor()["block"] is None or bench.cursor()["block"] < site+1:
                    bench.step("block")
                    if bench.cursor()["phase"] != "blocks": raise BenchError("site beyond denoiser block range")
            cuts[site] = bench.checkpoint(f"discover-site-{site}")
        reference, reference_hash, reference_tokens, _ = consume()
        baseline = bench.checkpoint("discovery-baseline")
        for site in sites:
            for index, arm in enumerate([*contrasts, {"name": "noop", "mode": "scale", "value": 1.0},
                                        {"name": "random", "mode": "random", "value": 0.01}]):
                bench.restore(cuts[site]); mode = arm["mode"]; payload = arm["value"]
                if mode == "random":
                    old = bench.resolve_value("residual")
                    generator = torch.Generator().manual_seed(spec.get("seed", 0)+site)
                    noise = torch.randn(old.shape, generator=generator).to(old.device)
                    payload = (noise/noise.norm().clamp_min(1e-12)*old.float().norm()*payload).to(old.dtype); mode = "add"
                act = bench.preview_edit("residual", payload, mode=mode)
                committed = bench.commit_edit(act["act_id"])
                scalar, digest, tokens, output = consume(); effect = scalar-reference
                name = arm.get("name", f"contrast-{index}")
                evidence = {"schema": "saturn-bench-family-observation-v2", "model_identity": bench.identity,
                    "session_id": bench.session_id, "parent_cut": parent, "site_cut": cuts[site], "act": committed,
                    "baseline_cut": baseline, "arm": arm, "site": site, "observable": observable,
                    "effect": effect, "reference": reference, "actual": scalar, "reference_hash": reference_hash,
                    "consumer_sha256": digest, "tokens": tokens, "output": output, "horizon": horizon,
                    "memory_links": ["conv_state", "ssm_state"] if bench.family == "mamba" else ["latent", "conditioner"],
                    "mechanics_status": "valid", "terminal_status": "not-assessed"}
                evidence_ref = bench.artifacts.put(evidence)
                control = name in {"noop", "random"}; dose = arm["value"]-1 if arm["mode"] == "scale" else arm["value"]
                rows.append(HyperionObservation(observation_id=f"site-{site}-arm-{index}", cell_id=kind,
                    arm_id=f"arm-{index}", phase="control" if control else "discovery", arm_kind=name if control else "candidate",
                    sites=(f"{bench.family}.block.{site}.residual",), schedule=((sorted(sites).index(site), float(dose)),),
                    target_effect=effect, collateral_effect=None, continuation_effect=float(tokens != reference_tokens),
                    consumer_id=f"bench.{bench.family}.native", evidence_ref=evidence_ref,
                    metadata={"parent_cut": parent, "site": site, "contrast": arm,
                              "collateral_status": "not-measured", "scalar": scalar}))
        report = compile_dynamic_circuit(rows, symbol_id=spec.get("name", "bench-candidate"),
            logical_symbol=spec.get("name", "bench-candidate"), model_id=bench.identity["model_id"],
            consumer_id=f"bench.{bench.family}.native", time_axis="layer" if bench.family == "mamba" else "block",
            address_space="bench.residual", residual_definition="same-parent intervention, signed native observable change",
            source_artifacts=[{"reference": row.evidence_ref} for row in rows], context=context)
    finally: bench.restore(parent)
    ref = bench.bind_symbol(report)
    bench.checkpoint("discovery-bound")
    return {"report": report, "report_ref": ref, "site_cuts": cuts, "baseline_cut": baseline,
            "quality_status": "candidate-trend", "terminal_status": "not-assessed"}
