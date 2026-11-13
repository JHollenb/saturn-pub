"""Raw-state attention -> image-MLP forecasts and routing/value accounting.

Uses only original checkpoint parameters and declared block-entry state. No
native intermediate return participates in a prospective forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil


def support():
    return runpy.run_path(str(Path(__file__).with_name("flux_mlp_state_law.py")))


def norm(x, eps=1e-6):
    import numpy as np
    B = support()["quantize"]
    x = np.asarray(x, np.float64)
    centered = x-x.mean(-1, keepdims=True)
    return B(centered / np.sqrt(np.mean(centered*centered, -1, keepdims=True)+eps))


def rms(x, weight, eps=1e-6):
    import numpy as np
    x = np.asarray(x, np.float64)
    return support()["quantize"](x / np.sqrt(np.mean(x*x, -1, keepdims=True)+eps) * weight)


def linear(x, weight):
    import numpy as np
    return support()["quantize"](np.asarray(x, np.float64) @ np.asarray(weight, np.float64).T)


def rotate(x, cos, sin):
    import numpy as np
    paired = np.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)
    left = np.multiply(x, cos[None, :, None], dtype=np.float32)
    right = np.multiply(paired, sin[None, :, None], dtype=np.float32)
    return support()["quantize"](np.add(left, right, dtype=np.float32))


def dense(q, k, v):
    """FP64 stable attention with B/H/query/head-feature geometry."""
    import numpy as np
    q, k, v = [np.asarray(a, np.float64) for a in (q, k, v)]
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("dense attention geometry differs")
    score = (q @ k.swapaxes(-1, -2))/np.sqrt(q.shape[-1])
    score -= score.max(-1, keepdims=True)
    probability = np.exp(score)
    probability /= probability.sum(-1, keepdims=True)
    return probability @ v, probability


def routing_components(q, k0, v0, k1, v1, text_rows):
    """Exact-arithmetic factorial; no semantic labels or inferred percentages."""
    import numpy as np
    out0, p0 = dense(q, k0, v0)
    out1, p1 = dense(q, k1, v1)
    dv = np.asarray(v1, np.float64)-np.asarray(v0, np.float64)
    dp = p1-p0
    route_text = dp[..., :text_rows] @ np.asarray(v0[..., :text_rows, :], np.float64)
    route_image = dp[..., text_rows:] @ np.asarray(v0[..., text_rows:, :], np.float64)
    values = p0 @ dv
    interaction = dp @ dv
    delta = out1-out0
    closure = route_text+route_image+values+interaction-delta
    return dict(route_text=route_text, route_image=route_image, value=values,
        interaction=interaction, delta=delta, closure=closure,
        text_mass_base=p0[..., :text_rows].sum(-1), text_mass_changed=p1[..., :text_rows].sum(-1))


def forecast(state, weights, indices, text_indices, heads=24):
    import numpy as np
    B = support()["quantize"]
    image, text = state["image"], state["text"]
    mod = lambda name: np.split(state[name].reshape(1, 1, -1), 6, axis=-1)
    si, ci, ga, sm, cm, gm = mod("img_mod")
    st, ct, _, _, _, _ = mod("txt_mod")
    xi = B(B(B(1+ci)*norm(image))+si)
    xt = B(B(B(1+ct)*norm(text))+st)
    width, text_rows = image.shape[-1], text.shape[1]
    dim = width//heads
    q_raw = linear(xi[:, indices], weights["attn.to_q.weight"]).reshape(1, len(indices), heads, dim)
    ki_raw = linear(xi, weights["attn.to_k.weight"]).reshape(1, image.shape[1], heads, dim)
    vi = linear(xi, weights["attn.to_v.weight"]).reshape(1, image.shape[1], heads, dim)
    kt_raw = linear(xt, weights["attn.add_k_proj.weight"]).reshape(1, text_rows, heads, dim)
    vt = linear(xt, weights["attn.add_v_proj.weight"]).reshape(1, text_rows, heads, dim)
    qi = rms(q_raw, weights["attn.norm_q.weight"])
    ki = rms(ki_raw, weights["attn.norm_k.weight"])
    kt = rms(kt_raw, weights["attn.norm_added_k.weight"])
    cos, sin = state["cos"], state["sin"]
    q = rotate(qi, cos[[text_rows+i for i in indices]], sin[[text_rows+i for i in indices]]).transpose(0, 2, 1, 3)
    k = rotate(np.concatenate([kt, ki], 1), cos, sin).transpose(0, 2, 1, 3)
    v = np.concatenate([vt, vi], 1).transpose(0, 2, 1, 3)
    head64, _ = dense(q, k, v)
    head = B(head64)
    merged = head.transpose(0, 2, 1, 3).reshape(1, len(indices), width)
    attention = linear(merged, weights["attn.to_out.0.weight"])
    U = B(image[:, indices]+B(ga*attention))
    mlp_input = B(B(norm(U)*B(1+cm))+sm)
    ff = support()["forecast"](mlp_input, weights["ff.linear_in.weight"], weights["ff.linear_out.weight"])
    final = B(U+B(gm*ff["output"]))
    return dict(q=q, k=k, v=v, head=head, attention=attention,
        norm_image=xi[:, indices], norm_text=xt[:, text_indices],
        q_raw=q_raw, image_k_raw=ki_raw[:, indices], text_k_raw=kt_raw[:, text_indices],
        q_norm=qi, image_k_norm=ki[:, indices], text_k_norm=kt[:, text_indices],
        mlp_input=mlp_input, projected=ff["projected"], activated=ff["activated"],
        product=ff["product"], mlp=ff["output"], image_exit=final)


def stage(run, source, previous):
    import numpy as np
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.parameter_artifacts import validate_manifest
    from mdb_lanes.job_support import one_lease_scheduling_contract
    here = Path(__file__).resolve().parent
    S = support()
    read, sha, save = S["read"], S["sha"], S["save"]
    src = source / "collection-01/extracted/binding-minimal-result"
    if read(source / "audit-v2.json")["mechanics_status"] != "verified" or read(previous / "audit-v1.json")["mechanics_status"] != "verified":
        raise ValueError("audited source states and MLP predecessor required")
    prior = read(src / "request.json")
    runpy.run_path(str(here / "flux_context_transfer.py"))["bind_environment"](prior["execution_environment"])
    reader = runpy.run_path(str(here / "analyze_flux_context_transfer.py"))
    rows = {r["id"]: r for r in read(src / "rows.json")}
    parents = {n: rows[old]["writer"]["entry"] for n, old in (("base", "control"), ("changed", "restore-I"))}
    cuts = {n: reader["load_cut"](src / "store", fp) for n, fp in parents.items()}
    from saturn_pub.values import describe
    if describe(cuts["base"].payload["image"]) != describe(cuts["changed"].payload["image"]) or describe(cuts["base"].payload["context"]) != describe(cuts["changed"].payload["context"]):
        raise ValueError("declared equal receiving image/context differs")
    run.mkdir(exist_ok=False, parents=True)
    base = cuts["base"].payload
    values = dict(image=base["image"], text_base=base["text"], text_changed=cuts["changed"].payload["text"],
        img_mod=base["context"]["img_mod"], txt_mod=base["context"]["txt_mod"],
        cos=base["context"]["rope"][0], sin=base["context"]["rope"][1])
    refs, extras = {}, []
    for name, tensor in values.items():
        path = run / (name + ".npy")
        array = tensor.detach().float().cpu().numpy().copy()
        np.save(path, array, allow_pickle=False)
        refs[name] = dict(sha256=sha(path), descriptor=describe(tensor))
        extras.append((path, path.name))
    parameters = validate_manifest(read(src / "parameter-artifacts.json"))
    selected = [r for r in parameters["artifacts"] if r["source"]["tensor"].startswith("transformer_blocks.1.")]
    if len(selected) != 16:
        raise ValueError("complete original joint1 parameter set required")
    save(run / "parameter-artifacts.json", {**parameters, "artifacts": selected})
    shutil.copyfile(src / "retrieve_weights.py", run / "retrieve_weights.py")
    config = dict(schema="mdb-flux-attention-state-law-plan-v1",
        model="component:" + prior["config"]["checkpoint_sha256"] + ":transformer_blocks.1:cpu-bf16-v1",
        source_model=prior["model"], checkpoint_sha256=prior["config"]["checkpoint_sha256"],
        model_path=prior["config"]["model_path"], output_scratch=prior["config"]["output_scratch"],
        checkpoint_file=selected[0]["source"]["file"], checkpoint_file_sha256=selected[0]["source"]["file_sha256"],
        parents=parents, input_files=refs, source_audit_sha256=sha(source / "audit-v2.json"),
        predecessor_audit_sha256=sha(previous / "audit-v1.json"),
        indices=read(previous / "collection-01/extracted/binding-minimal-result/plan.json")["selected_rows"],
        text_indices=[0, 1, 4, 11, 31, 63, 95, 127, 159, 191, 255, 319, 383, 447, 479, 511],
        cases=["base", "changed", "midpoint"], corners=["K0V0", "K1V0", "K0V1", "K1V1"],
        max_wall_seconds=600, cpu_threads=4, directory_custody=True,
        expected_counts=dict(component_loads=1, native_blocks=3, native_attention=3,
            native_image_ff=3, native_text_ff=3, native_linears=36,
            native_sdpa_in_block=3, native_sdpa_standalone=4, receipts=3,
            numpy_block_forecasts=3, numpy_corner_forecasts=4,
            full_pipeline_forwards=0, images=0, encodes=0, noise_draws=0),
        hypotheses=dict(structural="Equal incoming image/context gives exact equal image queries/keys/values across T cases; text K/V can change.",
            forecast="Predict selected image exits and attention/MLP stages from raw observed block-entry state plus original weights before any new native block.",
            corners="Four independent K/V factorial forecasts use predicted Q/K/V operands; validate native SDPA on those same supplied operands after seal.",
            scope="Base/changed are known CUDA source states; midpoint T is newly tested whole-block input. Native CPU numerical errors remain observations. No semantic labels, fitted coefficients, raw-prompt or text-exit prediction."))
    save(run / "plan.json", config)
    extras.extend((run / n, n) for n in ("plan.json", "parameter-artifacts.json", "retrieve_weights.py"))
    extras.extend((here / n, n) for n in ("flux_attention_state_law.py", "flux_mlp_state_law.py", "flux_context_transfer.py", "flux_family_custody.py"))
    pre = run / "preflight.v1.json"
    save(pre, dict(schema_version="mrun-preflight-config-v1",
        hashed_roots=["flux_attention_state_campaign.py", *[n for _, n in extras], pre.name],
        path_token_allowlist=[config["model_path"], config["output_scratch"], "binding-minimal-result"]))
    extras.append((pre, pre.name))
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    request = dict(schema="mdb-flux-attention-state-law-request-v1", config=config, model=config["model"],
        backend="diffusers", device="cpu", dtype="bfloat16", batch_size=1, needs={},
        execution_environment=prior["execution_environment"], config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes={n: sha(p) for p, n in [(here / "flux_attention_state_campaign.py", "flux_attention_state_campaign.py"), *extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=7,
            logical_branches=7, logical_checkpoint_replays=0, max_local_work_items=7,
            checkpoint_policy="One original joint1 module;3full native block steps through standalone MDB adapter plus4standalone SDPA kernel corners; all predictions before native; no encoder/VAE/retries."))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=here / "flux_attention_state_campaign.py", config=config,
        request=request, config_bytes=raw, additional_files=extras)
    return dict(request_sha256=request["request_sha256"], run=str(run))


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    S = support()
    read, sha, save = S["read"], S["sha"], S["save"]
    request = read(run / "request.json")
    runpy.run_path(str(run / "payload/flux_context_transfer.py"))["bind_environment"](request["execution_environment"])
    if (run / "submission.json").exists() or any(sha(run / "payload" / n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("already submitted or payload changed")
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request seal differs")
    job = guarded_mrun_launch(worker_command("flux_attention_state_campaign.py"), request=request,
        model=request["model"], task="forward", experiment="mdb-flux-attention-state-law",
        payload=run / "payload", pin="beast", gpu=False, needs={}, preflight=True,
        retry_on_kill=False, timeout_s=780,
        note="Computation-first raw-state QKV->attention->imageMLP: one checkpoint-local CPU BF16 joint1 component,3native full B1(512T+1024I) blocks plus4Q32/K1536 SDPA corners.3NumPy block forecasts+4corner forecasts sealed before targets;32image queries carried through MLP/residual. Port-only hooks+MDB StateCuts/3receipts;0fullpipeline/VAE/encodes/noise.600worker780outer,no retries; auto component probe.")
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--previous", type=Path)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    result = submit(a.run.resolve()) if a.submit else stage(a.run.resolve(), a.source.resolve(), a.previous.resolve())
    print(json.dumps(result, sort_keys=True))
