"""Complete joint image/text forecasts and a sealed two-block continuation assay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil


def helper(name):
    return runpy.run_path(str(Path(__file__).with_name(name)))


def states(arrays):
    import numpy as np
    B = helper("flux_mlp_state_law.py")["quantize"]
    common = {k: arrays[k] for k in ("image", "img_mod", "txt_mod", "cos", "sin")}
    base, changed = arrays["text_base"], arrays["text_changed"]
    return {name: dict(common, text=B((1-dose)*base.astype(np.float64)+dose*changed))
            for name, dose in (("base", 0.), ("quarter", .25), ("midpoint", .5))}


def project(x, weight, rows=64):
    """Chunk rows, with one FP64 parameter conversion per projection."""
    import numpy as np
    B = helper("flux_mlp_state_law.py")["quantize"]
    w = np.asarray(weight, np.float64)
    out = np.empty((*x.shape[:-1], w.shape[0]), np.float32)
    for start in range(0, x.shape[1], rows):
        out[:, start:start+rows] = B(np.asarray(x[:, start:start+rows], np.float64) @ w.T)
    return out


def attention(q, k, v, rows=64):
    """All queries/all keys, exact mathematical row independence; bounded workspace."""
    import numpy as np
    B = helper("flux_mlp_state_law.py")["quantize"]
    k, v = np.asarray(k, np.float64), np.asarray(v, np.float64)
    out = np.empty(q.shape, np.float32)
    for start in range(0, q.shape[2], rows):
        p = np.asarray(q[:, :, start:start+rows], np.float64) @ k.swapaxes(-1, -2)
        p /= np.sqrt(q.shape[-1])
        p -= p.max(-1, keepdims=True)
        np.exp(p, out=p)
        p /= p.sum(-1, keepdims=True)
        out[:, :, start:start+rows] = B(p @ v)
    return out


def mlp(x, incoming, outgoing, selected, rows):
    import numpy as np
    S = helper("flux_mlp_state_law.py")
    B = S["quantize"]
    z = project(x, incoming, rows)
    a, v = np.split(z, 2, axis=-1)
    s = B(S["silu"](a))
    product = B(s.astype(np.float64)*v)
    output = project(product, outgoing, rows)
    return output, dict(projected=z[:, selected].copy(), activated=s[:, selected].copy(),
                        product=product[:, selected].copy(), mlp=output[:, selected].copy())


def forecast(state, weights, image_indices, text_indices, *, heads=24, chunk_rows=64):
    import numpy as np
    L = helper("flux_attention_state_law.py")
    B = helper("flux_mlp_state_law.py")["quantize"]
    image, text = state["image"], state["text"]
    width, nt = image.shape[-1], text.shape[1]
    hd = width//heads
    mods = {n: np.split(state[n].reshape(1, 1, -1), 6, -1) for n in ("img_mod", "txt_mod")}
    xs = {n: B(B(B(1+mods[m][1])*L["norm"](x))+mods[m][0])
          for n, x, m in (("image", image, "img_mod"), ("text", text, "txt_mod"))}
    q, k, v = {}, {}, {}
    for name, prefix, normprefix in (("image", "attn.to_", "attn.norm_"),
                                      ("text", "attn.add_", "attn.norm_added_")):
        values = {}
        for port in "qkv":
            parameter = prefix+port+("_proj.weight" if name == "text" else ".weight")
            values[port] = project(xs[name], weights[parameter], chunk_rows).reshape(1, xs[name].shape[1], heads, hd)
        q[name] = L["rms"](values["q"], weights[normprefix+"q.weight"])
        k[name] = L["rms"](values["k"], weights[normprefix+"k.weight"])
        v[name] = values["v"]
    position = lambda d: L["rotate"](np.concatenate([d["text"], d["image"]], 1), state["cos"], state["sin"]).transpose(0, 2, 1, 3)
    Q, K = position(q), position(k)
    V = np.concatenate([v["text"], v["image"]], 1).transpose(0, 2, 1, 3)
    H = attention(Q, K, V, chunk_rows)
    merged = H.transpose(0, 2, 1, 3).reshape(1, nt+image.shape[1], width)
    fields = {}
    for name, x, selected, modname, start, end, outweight, ff in (
        ("text", text, text_indices, "txt_mod", 0, nt, "attn.to_add_out.weight", "ff_context"),
        ("image", image, image_indices, "img_mod", nt, nt+image.shape[1], "attn.to_out.0.weight", "ff")):
        shift, scale, ga, sm, cm, gm = mods[modname]
        a = project(merged[:, start:end], weights[outweight], chunk_rows)
        u = B(x+B(ga*a))
        xi = B(B(L["norm"](u)*B(1+cm))+sm)
        m, stages = mlp(xi, weights[ff+".linear_in.weight"], weights[ff+".linear_out.weight"], selected, chunk_rows)
        fields[name] = B(u+B(gm*m))
        fields.update({name+"."+port: value for port, value in stages.items()})
        fields.update({name+"."+port: value for port, value in dict(norm=xs[name][:, selected].copy(),
            attention=a[:, selected].copy(), mlp_input=xi[:, selected].copy(),
            q=Q[:, :, [start+i for i in selected]].copy(),
            k=K[:, :, [start+i for i in selected]].copy(),
            v=V[:, :, [start+i for i in selected]].copy(),
            head=H[:, :, [start+i for i in selected]].copy()).items()})
    if not all(np.isfinite(a).all() and np.array_equal(B(a), a) for a in fields.values()):
        raise ValueError("forecast closure must be finite exact widened BF16")
    return fields


def stage(run, previous, original):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.parameter_artifacts import validate_manifest
    from mdb_lanes.job_support import one_lease_scheduling_contract
    S = helper("flux_mlp_state_law.py")
    read, save, sha = S["read"], S["save"], S["sha"]
    prior = read(previous / "run-01/request.json")
    assert read(previous / "audit-v1.json")["mechanics_status"] == "verified"
    helper("flux_context_transfer.py")["bind_environment"](prior["execution_environment"])
    source = original / "collection-01/extracted/binding-minimal-result"
    recipes = validate_manifest(read(source / "parameter-artifacts.json"))
    selected = [r for r in recipes["artifacts"] if any(r["source"]["tensor"].startswith(f"transformer_blocks.{i}.") for i in (1, 2))]
    if len(selected) != 32:
        raise ValueError("two complete original joint parameter sets required")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "parameter-artifacts.json", {**recipes, "artifacts": selected})
    shutil.copyfile(source / "retrieve_weights.py", run / "retrieve_weights.py")
    extras = []
    for n, ref in prior["config"]["input_files"].items():
        src = previous / "run-01/payload" / (n+".npy")
        if sha(src) != ref["sha256"]:
            raise ValueError("accepted source input changed")
        extras.append((src, n+".npy"))
    config = dict(prior["config"], schema="mdb-flux-state-closure-plan-v1", blocks=[1, 2],
        model="component:"+prior["config"]["checkpoint_sha256"]+":transformer_blocks.1-2:cpu-bf16-v1",
        cases=["base", "quarter", "midpoint"], doses=dict(base=0., quarter=.25, midpoint=.5),
        chunk_rows=64, max_wall_seconds=720, max_member_bytes=128 << 20,
        expected_counts=dict(component_loads=1, native_blocks=9, native_attention=9, native_image_ff=9,
            native_text_ff=9, native_linears=108, native_sdpa=9, receipts=15, acts=6,
            forks=3, numpy_block_forecasts=6, full_pipeline_forwards=0, encodes=0, images=0, noise_draws=0),
        source_attention_audit_sha256=sha(previous / "audit-v1.json"),
        hypotheses=dict(closure="Predict every image and text exit element of joint1 and joint2, with predicted joint1 exit as joint2 forecast input.",
            consumer="After seal, fork native joint1 cut and replace BOTH image and text with sealed forecast, preserving clock/context; run unchanged native joint2.",
            decomposition="Separate own-input joint2 arithmetic error from native propagation of joint1 input error; retain exact signed sum closure.",
            novelty="Quarter text mix is new internal input; base and midpoint were previously observed. No new natural prompt, fitted coefficients or semantic labels."))
    config.pop("corners", None)
    save(run / "plan.json", config)
    here = Path(__file__).resolve().parent
    extras += [(run / n, n) for n in ("plan.json", "parameter-artifacts.json", "retrieve_weights.py")]
    extras += [(here / n, n) for n in ("flux_state_closure.py", "flux_attention_state_law.py", "flux_mlp_state_law.py", "flux_context_transfer.py", "flux_family_custody.py")]
    pre = run / "preflight.v1.json"
    save(pre, dict(schema_version="mrun-preflight-config-v1",
        hashed_roots=["flux_state_closure_campaign.py", *[n for _, n in extras], pre.name],
        path_token_allowlist=[config["model_path"], config["output_scratch"], "binding-minimal-result"]))
    extras.append((pre, pre.name))
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    request = dict(schema="mdb-flux-state-closure-request-v1", config=config, model=config["model"],
        backend="diffusers", device="cpu", dtype="bfloat16", needs={}, batch_size=1,
        execution_environment=prior["execution_environment"], config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes={n: sha(p) for p, n in [(here / "flux_state_closure_campaign.py", "flux_state_closure_campaign.py"), *extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=15,
            logical_branches=6, logical_checkpoint_replays=3, max_local_work_items=15,
            checkpoint_policy="One original two-block component load;6native chain steps+3predicted-input joint2 continuations,6typed replacement acts/3forks;6NumPy forecasts sealed before native;no model retries or encoders/VAE."))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=here / "flux_state_closure_campaign.py", config=config,
        request=request, config_bytes=raw, additional_files=extras)
    return dict(request_sha256=request["request_sha256"], run=str(run))


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    S = helper("flux_mlp_state_law.py")
    read, sha, save = S["read"], S["sha"], S["save"]
    request = read(run / "request.json")
    helper("flux_context_transfer.py")["bind_environment"](request["execution_environment"])
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    assert hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"]
    if (run / "submission.json").exists() or any(sha(run / "payload" / n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("payload changed or already submitted")
    job = guarded_mrun_launch(worker_command("flux_state_closure_campaign.py"), request=request,
        model=request["model"], task="forward", experiment="mdb-flux-state-closure", payload=run / "payload",
        pin="beast", gpu=False, needs={}, preflight=True, retry_on_kill=False, timeout_s=900,
        note="Complete joint1->joint2 CPU BF16 image+text state closure. One original2block component load;3rawstates base/newquarter/knownmidpoint.6chunked NumPy full-state forecasts sealed before targets;6native chain block steps+3native joint2 predicted-state continuations,3forks6Acts. All1536queries/keys,chunk64 math workspace;no predicted-state oracle refresh.0pipeline/VAE/encoder/noise.720worker900outer,no retries;auto planning.")
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--previous", type=Path)
    p.add_argument("--original", type=Path)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    result = submit(a.run.resolve()) if a.submit else stage(a.run.resolve(), a.previous.resolve(), a.original.resolve())
    print(json.dumps(result, sort_keys=True))
