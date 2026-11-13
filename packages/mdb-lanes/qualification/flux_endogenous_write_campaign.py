"""Native-write projection removal from two immutable observed joint0 exits.

The source writer has already been observed. Every NEW complete suffix and
image is forecast and sealed before its native replay. No prompt, writer or
semantic universality is claimed from this later source frontier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "mdb-flux-endogenous-write-v1"
RESULT = "flux-endogenous-write-result"
BASE = "flux_centered_write_campaign.py"
BASE_SHA256 = "bf4608c2c732d713c532a0edf81c4aef32d4d7dcc0a6ddf2867d3a9c1421e04a"
HELPERS = {
    BASE: BASE_SHA256,
    "flux_raw_to_image.py": "b04db3835522a007a15293ef177e4463cf05e9d01122d47d42531dc72bd9700a",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "7f13836ae94c84b7ed0f7df67b1f7b1d28a7e534a83d509fa7320a09a6d845df",
    "flux_image_tail_microscope.py": "b9efb61874005c8b11307c197a2ea5299078b42d5b4ab4b5cc719f7a4a4a32bb",
    "flux_source_receiver.py": "d483e41bfac018c8225662e8417bcef46f6916fa2a14d910f16e9a12beb5250c",
}
SOURCE_JOB = "job-445316274652"
SOURCE_REQUEST = "badd4880754550c73743836a7b35a00acfbcfd15a825ffb03641433d7ec09927"
SOURCE_MANIFEST = "c29177c6fbbf9217a84321bf948abc82db55b4f00abaae2078a591f1dcf77766"
SOURCE_ROOT = "/mnt/big/mdb/research/job-445316274652/flux-centered-write-result"
CONTEXT_IDS = ("fresh0s613", "fresh1s719")
ARMS = ("baseline", "sham", "top-half", "top-remove", "top-reflect", "bottom-remove", "random-remove", "random-matched")
DIRECTION_ARMS = dict(top="top-positive", bottom="bottom-positive", random="random-positive")
SCOPE = dict(kind="architectural", origin="previously-observed-native-joint0-exit-and-writer",
    raw_prompt_forecast=False, source_writer_forecast=False,
    scope="Two exact saved native joint0 exits; endogenous full-write projections onto previous even-derived finite output directions; all new suffix targets held until seal",
    independent_kernel_prediction=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def previous():
    for name, expected in HELPERS.items():
        require(sha(Path(__file__).with_name(name)) == expected, "frozen helper differs: " + name)
    return runpy.run_path(str(Path(__file__).with_name(BASE)))


def path_keys():
    return previous()["helpers"]()["_source_keys"]()


def contract():
    return dict(contexts=2, image_futures=16, reused_native_joint0_sources=2,
        native_qwen_source_calls=0, native_project_source_transitions=0,
        functional_transitions=1760, native_target_transitions=1760,
        functional_vae_calls=16, native_vae_reference_calls=16, native_vae_repeat_calls=2,
        sham_acts_per_lane=2, additive_endpoint_acts_per_lane=12, native_source_restores=2,
        first_consumer_port_sets_per_lane=16, component_hydrations=2,
        cuda_placements=4, cpu_offloads=4, functional_module_forwards=0,
        persistent_weight_banks=0, max_gpu_components=1)


def make_config(base):
    config = {k:v for k,v in base.items() if not k.startswith(("direction_", "spectral_")) and k not in (
        "job_id", "request_sha256", "worker_sha256", "output", "_request_record",
        "contexts", "primary_arms", "crossed_arms", "dose", "selection_protocol", "selection_protocol_sha256",
        "centered_envelope", "endpoint_order", "spectral_intervention", "spectral_coordinate")}
    config.update(schema=SCHEMA, context_ids=list(CONTEXT_IDS), arms=list(ARMS), output_name=RESULT,
        source_native_root=SOURCE_ROOT, source_native_job=SOURCE_JOB, source_native_request=SOURCE_REQUEST,
        source_native_manifest_sha256=SOURCE_MANIFEST, source_native_worker_sha256=BASE_SHA256,
        source_frontier="saved actual joint0 exit plus passive M_base/x/R_attention; source writer is observed",
        literal_write="BF16(gate_ff * native_M_base); effective write H_joint0 - R_attention reported separately",
        direction_geometry="normalize channel-centered previous finite BF16 top/bottom/random output payload per row; no eigensearch",
        payload_rule="BF16(-alpha*dot(P_channel(w_mul),u)*u); alpha=.5,1,2 for top;1 for bottom/random",
        random_matched_rule="-abs(beta_top)*sgn0(beta_random)*u_random; sgn0(0)=+1; energy control may overshoot native random component",
        endpoint_contract="Add identical presealed BF16 payload to each lane own observed joint0 exit; does not rerun an internally edited FF",
        first_consumer="step0/joint1 norm1, norm1_context, R_attention, x, M_base and gate_ff; unchanged original native modules",
        helper_source_sha256=dict(HELPERS), height=512, width=512, steps=4, batch_size=1,
        device="cuda", dtype="bfloat16", flux_mode="capture", capture_sites=True,
        component_hydrations=2, cuda_placements=4, offloads=4, max_gpu_components=1,
        max_wall_seconds=1800, forecast_transaction_bytes=3 << 30,
        transport_limits=dict(max_expanded_bytes=8 << 30, max_member_bytes=134221824),
        scientific_gates=[], retention="complete intervention/first-consumer/clock states, first consumer ports, full image arrays and all boundary hashes",
        decoder_contract="16 functional forecasts,16 actual native-final decodes,2 same-native-final repeats")
    return config


def validate_config(config):
    for key, expected in make_config({}).items():
        require(config.get(key) == expected, "frozen experiment field differs: " + key)
    for key in path_keys():
        require(config.get(key) and sha(config[key]) == config.get(key + "_sha256"), "source metadata differs: " + key)
    require(config.get("guidance_scale", 1.) == 1. and not config.get("reference_image", False)
        and not config.get("retry", False) and not config.get("nested_mrun", False), "unsupported execution profile")
    return contract()


def projection_payloads(gated_write, effective_write, prior_payloads):
    """Source-only FP64 geometry; all zero and finite-cast cases are retained."""
    import numpy as np
    import torch

    base = previous(); rows = base["scalar_rows"]
    def tensor(value):
        return value.detach().double().cpu().numpy()
    def record(value):
        return rows(torch.from_numpy(np.ascontiguousarray(value)))
    w, eff = tensor(gated_write), tensor(effective_write)
    z, zeff = w-w.mean(-1, keepdims=True), eff-eff.mean(-1, keepdims=True)
    units, betas, records = {}, {}, {}
    for name, payload in prior_payloads.items():
        raw = tensor(payload); centered = raw-raw.mean(-1, keepdims=True)
        lengths = np.sqrt(np.square(centered).sum(-1, keepdims=True))
        u = np.divide(centered, lengths, out=np.zeros_like(centered), where=lengths > 0)
        beta = (z*u).sum(-1, keepdims=True)
        units[name], betas[name] = u, beta
        total = np.square(z).sum(-1, keepdims=True)
        valid = lengths > 0; nvalid = int(np.count_nonzero(valid))
        mean = float(beta[valid].mean()) if nvalid else 0.; uniform_energy = nvalid*mean*mean
        records[name] = dict(beta=record(beta.squeeze(-1)), effective_beta=record((zeff*u).sum(-1)),
            zero_direction_rows=int(np.count_nonzero(lengths == 0)), native_projection_energy=record(np.square(beta.squeeze(-1))),
            native_centered_write_energy=record(total.squeeze(-1)),
            fraction_native_centered_energy=record(np.divide(beta*beta,total,out=np.zeros_like(beta),where=total>0).squeeze(-1)),
            beta_spatial_mean_on_valid_directions=mean, valid_direction_rows=nvalid, uniform_amplitude_energy=uniform_energy,
            varying_amplitude_energy=float(np.square(beta[valid]-mean).sum()),
            global_native_centered_energy_fraction=None if float(total.sum())==0 else float(np.square(beta).sum()/total.sum()),
            positive_rows=int(np.count_nonzero(beta>0)), negative_rows=int(np.count_nonzero(beta<0)), zero_beta_rows=int(np.count_nonzero(beta==0)))
    payloads, arms = {}, {}
    specifications = (("top-half","top",.5),("top-remove","top",1.),("top-reflect","top",2.),
                      ("bottom-remove","bottom",1.),("random-remove","random",1.))
    for arm, direction, alpha in specifications:
        ideal = -alpha*betas[direction]*units[direction]
        payload = torch.from_numpy(ideal.astype("float32")).to(torch.bfloat16)
        payloads[arm] = payload
        delta = tensor(payload); post = z+delta-delta.mean(-1, keepdims=True)
        predicted_energy = np.square(z).sum(-1)-alpha*(2-alpha)*np.square(betas[direction].squeeze(-1))
        arms[arm] = dict(direction=direction, alpha=alpha, kind="native-component-reflection" if alpha==2 else "native-component-attenuation",
            ideal_projection_after=record(((1-alpha)*betas[direction]).squeeze(-1)),
            ideal_centered_write_energy_after=record(predicted_energy),
            cast_projection_after=record((post*units[direction]).sum(-1)),
            cast_centered_write_energy_after=record(np.square(post).sum(-1)),
            ideal_payload_norm=record(np.sqrt(np.square(ideal).sum(-1))),
            cast_payload_norm=record(np.sqrt(np.square(delta).sum(-1))),
            cast_payload_row_mean=record(delta.mean(-1)),
            double_to_float32_to_bf16_cast_error_rms=float(np.sqrt(np.square(delta-ideal).mean())))
    beta_random = betas["random"]; amplitude = np.abs(betas["top"])
    signs = np.where(beta_random<0,-1.,1.)
    ideal = -amplitude*signs*units["random"]
    payload = torch.from_numpy(ideal.astype("float32")).to(torch.bfloat16)
    payloads["random-matched"] = payload
    factor = np.divide(amplitude,np.abs(beta_random),out=np.zeros_like(amplitude),where=beta_random!=0)
    arms["random-matched"] = dict(direction="random", kind="top-removal-energy-matched-control",
        ideal_payload_norm=record(np.sqrt(np.square(ideal).sum(-1))),
        cast_payload_norm=record(np.sqrt(np.square(tensor(payload)).sum(-1))),
        random_effective_attenuation_factor_where_defined=record(factor.squeeze(-1)),
        undefined_zero_random_beta_rows=int(np.count_nonzero(beta_random==0)),
        zero_random_direction_rows=records["random"]["zero_direction_rows"],
        overshoot_rows=int(np.count_nonzero((factor>1)&(beta_random!=0))),
        sign_at_zero=1, scope="equal nominal displacement energy, not necessarily equal native writer-energy change; undefined factors stored as zero and counted")
    cosines = {a+":"+b:record((units[a]*units[b]).sum(-1)) for a,b in (("top","bottom"),("top","random"),("bottom","random"))}
    report = dict(schema="mdb-endogenous-projection-geometry-v1", literal_writer="BF16(gate_ff*M_base)",
        effective_writer="joint0_image - R_attention widened before subtraction", directions=records, arms=arms,
        pairwise_output_direction_cosines=cosines,
        literal_vs_effective_rms=float(np.sqrt(np.square(w-eff).mean())),
        literal_centered_write_norm=record(np.sqrt(np.square(z).sum(-1))),
        effective_centered_write_norm=record(np.sqrt(np.square(zeff).sum(-1))),
        ideal_identity="z_alpha=z-alpha*(u^T z)u; component=(1-alpha)beta; energy=||z||^2-alpha(2-alpha)beta^2; alpha2 reflects at constant centered write energy",
        scope="Native full-write projection along even-derived finite directions. Endpoint additive Acts include separate BF16 rounding and are not internal FF recomputations.")
    return payloads, report


def source_inventory(config):
    """Authenticate all original evidence bytes near compute; run no model."""
    root = Path(config["source_native_root"]); wrapper = read(root/"files.json")
    require(sha(root/"files.json")==SOURCE_MANIFEST and wrapper["job_id"]==SOURCE_JOB
        and wrapper["request_sha256"]==SOURCE_REQUEST, "original native publication differs")
    names={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p!=root/"files.json"}
    require(names==set(wrapper["files"]), "original source inventory census differs")
    for name, record in wrapper["files"].items():
        path=root/name
        require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents)
            and path.stat().st_size==record["bytes"] and sha(path)==record["sha256"], "original source member differs: "+name)
    request=read(root/"request.json")
    identity=hashlib.sha256(json.dumps({k:v for k,v in request.items() if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(identity==SOURCE_REQUEST and request["source_hashes"].get(BASE)==BASE_SHA256, "original executed source differs")
    return root, wrapper


def consumer_equations(graph, model, state):
    """Original joint equations at index1, also retaining its writer ports."""
    import torch

    block = model.transformer_blocks[1]; image, text, context = (state[name] for name in ("image", "text", "context"))
    (shift, scale, gate), (shift_ff, scale_ff, gate_ff) = graph["_mods"](context["img_mod"], 2)
    (cshift, cscale, cgate), (cshift_ff, cscale_ff, cgate_ff) = graph["_mods"](context["txt_mod"], 2)
    image_ln, text_ln = graph["_norm"](block.norm1, image), graph["_norm"](block.norm1_context, text)
    x = (1 + scale) * image_ln + shift
    c = (1 + cscale) * text_ln + cshift
    attn = block.attn
    q, k, v = (graph["_linear"](getattr(attn, name), x).unflatten(-1, (-1, attn.head_dim)) for name in ("to_q", "to_k", "to_v"))
    cq, ck, cv = (graph["_linear"](getattr(attn, name), c).unflatten(-1, (-1, attn.head_dim)) for name in ("add_q_proj", "add_k_proj", "add_v_proj"))
    q = torch.cat([graph["_rms"](attn.norm_added_q, cq), graph["_rms"](attn.norm_q, q)], dim=1)
    k = torch.cat([graph["_rms"](attn.norm_added_k, ck), graph["_rms"](attn.norm_k, k)], dim=1)
    out = graph["_attention"](q, k, torch.cat([cv, v], dim=1), context["rope"])
    cout, out = out[:, :text.shape[1]], out[:, text.shape[1]:]
    residual = image + gate * graph["_linear"](attn.to_out[0], out)
    operand = graph["_norm"](block.norm2, residual) * (1 + scale_ff) + shift_ff
    write = graph["_ff"](block.ff, operand)
    image = residual + gate_ff * write
    text = text + cgate * graph["_linear"](attn.to_add_out, cout)
    text = text + cgate_ff * graph["_ff"](block.ff_context, graph["_norm"](block.norm2_context, text) * (1 + cscale_ff) + cshift_ff)
    return dict(state, image=image, text=text, phase="joint", cursor=2), dict(R_attention=residual, x=operand, M_base=write,
        gate_ff=gate_ff, image_ln=image_ln, text_ln=text_ln)


@contextmanager
def passive_consumer_ports(model, record):
    block=model.transformer_blocks[1]
    handles=[block.norm1.register_forward_hook(lambda _m,_a,v:record("image_ln",v)),
        block.norm1_context.register_forward_hook(lambda _m,_a,v:record("text_ln",v)),
        block.norm2.register_forward_pre_hook(lambda _m,a:record("R_attention",a[0])),
        block.ff.register_forward_pre_hook(lambda _m,a:record("x",a[0])),
        block.ff.register_forward_hook(lambda _m,_a,v:record("M_base",v))]
    try:
        yield
    finally:
        for handle in reversed(handles):handle.remove()


def fresh_adapter(model, scheduler_config, *, device):
    """Bind storage guards after placement; never carry them across offload."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    return Flux2KleinAdapter(model,FlowMatchEulerDiscreteScheduler.from_config(scheduler_config),granularity="operation",device=device)


def run(config, *, progress=None):
    declared=validate_config(config); base=previous(); old=base["helpers"]()
    import torch
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.core import Session, StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    output=Path(config["output"]);output.mkdir(parents=True,exist_ok=False)
    save(output/"plan.json",{k:v for k,v in config.items() if k!="_request_record"})
    if config.get("_request_record"):save(output/"request.json",config["_request_record"])
    started=time.monotonic(); events=[]; emissions=[]; placements=[]; observed={}
    counts={k:0 for k in declared if "per_lane" not in k and k!="max_gpu_components"}
    counts.update(contexts=2,image_futures=16)
    phase,released=dict(kind="source-loading"),dict(native=False)
    def event(kind,**details):
        row=dict(number=len(events),kind=kind,seconds=time.monotonic()-started,**details)
        require(row["seconds"]<=config["max_wall_seconds"],"finite wall bound exceeded")
        events.append(row)
        with (output/"timeline.jsonl").open("ab") as stream:stream.write(canonical(row)+b"\n")
        if progress:progress(row)
    def emit(kind,identifier,arm,lane,value):
        row=dict(number=len(emissions),kind=kind,context=identifier,arm=arm,lane=lane,receipt=value.to_dict());emissions.append(row)
        with (output/"receipt-emissions.jsonl").open("ab") as stream:stream.write(canonical(row)+b"\n")
    source_root,source_manifest=source_inventory(config)
    event("original_native_source_inventory_verified",files=len(source_manifest["files"]),bytes=source_manifest["expanded_bytes"])
    torch.set_num_threads(config.get("cpu_threads",4));torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    require(torch.cuda.is_available(),"original CUDA profile required")
    modules=old["_helper_modules"]({k:v for k,v in config.items() if k!="worker_sha256"})
    qw,graph,decoder_graph=(modules[old[k]] for k in ("QWEN_WORKER","DENOISER_HELPER","DECODER_HELPER"))
    runtime=qw["_runtime_contract"](config,torch);runtime["native_diffusers_sources"]=old["_native_source_contract"](config)["sources"]
    save(output/"runtime-contract.json",runtime)
    sources={}
    for key in path_keys():
        path=Path(config[key]);target=output/"source"/key/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        sources[key]=dict(path=str(target.relative_to(output)),sha256=sha(target),bytes=target.stat().st_size)
    flux,_,fm=old["_load_transformer"](config,runpy.run_path(config["flux_recipe_helper"]))
    vae,_,vm=old["_load_vae"](config,runpy.run_path(config["vae_recipe_helper"]))
    models=dict(flux=flux,vae=vae);prefixes=dict(flux=None,vae=("post_quant_conv.","decoder.","bn."))
    identities={name:old["_component_signature"](model,prefixes=prefixes[name]) for name,model in models.items()}
    original_identities=read(source_root/"source.json")["parameter_identities"]
    for name in models:old["_assert_signature_equal"](original_identities[name],identities[name],name)
    graph["validate_model"](flux);counts["component_hydrations"]=2
    def place(name,device):
        model=models[name]
        if name=="vae":model.post_quant_conv.to(device);model.decoder.to(device)
        else:model.to(device)
        signature=old["_component_signature"](model,prefixes=prefixes[name]);old["_assert_signature_equal"](identities[name],signature,name)
        placements.append(dict(component=name,device=device,signature=signature));counts["cuda_placements" if device=="cuda" else "cpu_offloads"]+=1
        event("placement",component=name,device=device)
        if device=="cpu":torch.cuda.empty_cache()
    store=LocalStore(output/"state-store");source_store=LocalStore(source_root/"state-store")
    def retain(cut):store.save(cut);return base["cut_record"](cut)
    def component(adapter,state,parent):
        cut=StateCut(adapter.model_identity,{**dict(adapter.execution),"composition":SCHEMA,"provenance":"component equations, not a native advance"},adapter.boundary(state),parent,state,adapter.point(state).to_dict(),adapter.surface(state).to_dict())
        return retain(cut)
    def patch(session,payload,identifier,arm,lane):
        before,after,value=base["apply_slot_edit"](session,"image",payload,mode="replace" if arm=="sham" else "add")
        unchanged=describe({k:v for k,v in before.payload.items() if k!="image"})==describe({k:v for k,v in after.payload.items() if k!="image"})
        require(unchanged,"Act changed undeclared slots");emit("act",identifier,arm,lane,value)
        key=("sham_acts_per_lane_" if arm=="sham" else "additive_endpoint_acts_per_lane_")+lane;counts[key]=counts.get(key,0)+1
        return dict(slot="image",mode="replace" if arm=="sham" else "add",before=retain(before),after=retain(after),receipt=value.to_dict(),
            all_other_slots_unchanged=unchanged,displacement=base["displacement_account"](before.payload["image"],after.payload["image"]))
    roots={};source_accounts={}
    prior_roots=read(source_root/"source-frontiers.json")["roots"]
    for identifier in CONTEXT_IDS:
        native_fork=read(source_root/f"cases/{identifier}/baseline/native-fork.json");cut=source_store.load(native_fork["common"]["cut"],device="cpu")
        require(cut.fingerprint==native_fork["common"]["cut"] and cut.payload["phase"]=="joint" and cut.payload["step"]==0 and cut.payload["cursor"]==1,"observed source clock differs")
        chain=[];ancestor=cut
        while True:
            chain.append(retain(ancestor))
            if not ancestor.parent:break
            parent_file=source_root/"state-store/cuts"/(ancestor.parent+".json")
            if not parent_file.exists():break
            require(len(chain)<16,"unexpected source ancestry depth");ancestor=source_store.load(ancestor.parent,device="cpu")
        records=read(source_root/f"writer/{identifier}/native-account.json")
        account={name:base["load_tensor"](source_root,record,"cpu") for name,record in records.items()}
        account["gate_ff"]=graph["_mods"](cut.payload["context"]["img_mod"],2)[1][2]
        literal=account["gate_ff"]*account["M_base"]
        effective=cut.payload["image"].double()-account["R_attention"].double()
        old_account=read(source_root/f"writer/{identifier}/account.json")
        priors={name:base["load_tensor"](source_root,old_account["payloads"][arm],"cpu") for name,arm in DIRECTION_ARMS.items()}
        payloads,geometry=projection_payloads(literal,effective,priors)
        original={name:old["_save_tensor"](output,f"writer/{identifier}/original/{name}.npy",value) for name,value in account.items()}
        original["literal_gated_write"]=old["_save_tensor"](output,f"writer/{identifier}/original/literal-gated-write.npy",literal)
        original["effective_write"]=old["_save_tensor"](output,f"writer/{identifier}/original/effective-write.npy",effective)
        prior_records={name:old["_save_tensor"](output,f"writer/{identifier}/prior-payloads/{name}.npy",value) for name,value in priors.items()}
        endpoints={arm:old["_save_tensor"](output,f"writer/{identifier}/payloads/{arm}.npy",value) for arm,value in payloads.items()}
        source_accounts[identifier]=dict(original=original,payloads=endpoints,prior_payloads=prior_records,
            prior_payload_sources={name:old_account["payloads"][arm] for name,arm in DIRECTION_ARMS.items()},native_account_sources=records,
            observed_native_common=native_fork["common"],source_ancestry=chain)
        save(output/f"writer/{identifier}/account.json",source_accounts[identifier]);save(output/f"writer/{identifier}/geometry.json",geometry)
        roots[identifier]=dict(prompt=prior_roots[identifier]["prompt"],seed=prior_roots[identifier]["seed"],arms=list(ARMS),source=retain(cut),source_ancestry=chain)
        counts["reused_native_joint0_sources"]+=1
    save(output/"source-frontiers.json",dict(scope=SCOPE,source_job=SOURCE_JOB,source_request=SOURCE_REQUEST,source_manifest_sha256=SOURCE_MANIFEST,roots=roots))
    processor=Flux2ImageProcessor(vae_scale_factor=2**(len(vae.config.block_out_channels)-1)*2)
    paths={};row_records={};port_records={"forecast":{},"native":{}}
    with base["instrument"](list(models.values()),phase,released,observed),torch.inference_mode():
        place("flux","cuda");phase["kind"]="forecast-denoiser"
        adapter=fresh_adapter(flux,read(config["flux_scheduler_config"]),device="cuda")
        for identifier,root in roots.items():
            parent=store.load(root["source"]["cut"],device="cuda")
            require(parent.model_identity==adapter.model_identity,"reused source model identity differs")
            for arm in ARMS:
                key=identifier+"/"+arm;branch_root=output/"cases"/identifier/arm;branch_root.mkdir(parents=True)
                state=parent.payload;start=component(adapter,state,parent.fingerprint);acts=[]
                if arm!="baseline":
                    branch=Session(adapter,state,parent=start["cut"])
                    payload=branch.read("image") if arm=="sham" else base["load_tensor"](output,source_accounts[identifier]["payloads"][arm],"cuda")
                    acts.append(patch(branch,payload,identifier,arm,"forecast"));state=branch.capture(retain=False).payload;start=acts[-1]["after"]
                save(branch_root/"forecast-acts.json",acts)
                rows=[dict(number=0,observed_joint0_source=True,retained=start,fields=base["tensor_tree"](state,old))];previous_cut=start["cut"]
                for number in range(1,111):
                    clock={k:state[k] for k in ("step","phase","cursor")}
                    if number==1:
                        state,ports=consumer_equations(graph,flux,state)
                        port_records["forecast"][key]={name:old["_save_tensor"](output,f"cases/{identifier}/{arm}/first-consumer/forecast/{name}.npy",value) for name,value in ports.items()}
                        counts["first_consumer_port_sets_per_lane_forecast"]=counts.get("first_consumer_port_sets_per_lane_forecast",0)+1
                    else:state=base["step_equations"](graph,flux,state)
                    counts["functional_transitions"]+=1;event_row=dict(step=clock["step"],phase=clock["phase"],index=clock["cursor"])
                    row=dict(number=number,event=event_row,fields=base["tensor_tree"](state,old))
                    if base["retain_transition"](event_row):row["retained"]=component(adapter,state,previous_cut);previous_cut=row["retained"]["cut"]
                    rows.append(row)
                require(state["step"]==4 and state["phase"]=="project" and state["cursor"]==0,"forecast suffix did not close")
                save(branch_root/"forecast-rows.json",rows);paths[key]=dict(root=str(branch_root.relative_to(output)),forecast_final=previous_cut);row_records[key]=rows
                event("forecast_suffix_completed",context=identifier,arm=arm,final_cut=previous_cut)
                del state,ports
        save(output/"first-consumer-forecasts.json",port_records["forecast"])
        place("flux","cpu");place("vae","cuda");phase["kind"]="forecast-vae"
        for key,path in paths.items():
            branch_root=output/path["root"];final=store.load(path["forecast_final"],device="cpu").payload
            argument=old["_decoder_input"](vae,final,config,{"Flux2KleinPipeline":Flux2KleinPipeline}).to("cuda")
            decoded=decoder_graph["functional_decode"](vae,argument,observe=None,counts={},profile="original")
            record=base["decoder_result"](old,branch_root,"forecast",decoded,processor)
            record["decoder_input"]=old["_save_tensor"](branch_root,"decoder/forecast/decoder-input.npy",argument)
            record.update(scope="own predicted latent through original shared-primitive functional VAE",macros_retained=False)
            save(branch_root/"decoder-forecast.json",record);counts["functional_vae_calls"]+=1
        responses={}
        for key,path in paths.items():
            identifier,arm=key.split("/")
            if arm=="baseline":continue
            basekey=identifier+"/baseline";selected=[]
            for row,base_row in zip(row_records[key],row_records[basekey]):
                if "retained" in row and "retained" in base_row:
                    left,right=(store.load(r["retained"]["cut"],device="cpu").payload for r in (row,base_row))
                    selected.append(dict(number=row["number"],left_cut=row["retained"]["cut"],right_cut=base_row["retained"]["cut"],fields=base["signed_fields"](left,right)))
            a,b=(read(output/paths[k]["root"]/"decoder-forecast.json") for k in (key,basekey))
            dec={name:base["signed_fields"](base["load_tensor"](output/path["root"],a[name],"cpu"),base["load_tensor"](output/paths[basekey]["root"],b[name],"cpu"))["/"] for name in ("decoded","pixels")}
            ports={name:base["signed_fields"](base["load_tensor"](output,rec,"cpu"),base["load_tensor"](output,port_records["forecast"][basekey][name],"cpu"))["/"] for name,rec in port_records["forecast"][key].items()}
            responses[key]=dict(orientation="arm-minus-own-baseline",full_retained_states=selected,decoder=dec,first_consumer=ports)
        save(output/"forecast-responses.json",responses);place("vae","cpu")
        inventory=old["_inventory"](output,exclude=("timeline.jsonl","receipt-emissions.jsonl"))
        tx=ForecastTransaction(output/"forecast-transaction",model_identity=SCHEMA,input_identity=hashlib.sha256(canonical(roots)).hexdigest(),
            source_identity=hashlib.sha256(canonical(HELPERS)).hexdigest(),frontier=dict(SCOPE,external_files=inventory),
            execution_scope="Observed previous native joint0 writes; all16 new full suffix and image forecasts sealed before targets",
            array_storage="files-v1",budgets=dict(max_predictions=1,max_executions=1,max_native_calls=3000,max_files=30000,max_bytes=config["forecast_transaction_bytes"],max_wall_ms=config["max_wall_seconds"]*1000))
        tx.frontier("observed-native-joint0",dict(roots=roots,scope=SCOPE))
        tx.predict("all-futures",lambda _ctx:dict(predictions=dict(external_files=inventory,paths=paths,targets_at_seal=0)))
        seal=tx.seal();event("all_forecasts_globally_sealed",seal_sha256=seal["seal_sha256"],native_targets_at_seal=0)
        def targets(_ctx):
            old["_verify_inventory"](output,inventory);released["native"]=True;event("native_targets_released",seal_sha256=seal["seal_sha256"])
            place("flux","cuda");phase["kind"]="target-denoiser"
            native_adapter=fresh_adapter(flux,read(config["flux_scheduler_config"]),device="cuda")
            for identifier,root in roots.items():
                parent=store.load(root["source"]["cut"],device="cuda");master=Session(native_adapter,parent.payload,parent=parent.parent)
                require(master.capture(retain=False).fingerprint==parent.fingerprint,"actual native source session differs")
                for arm in ARMS:
                    key=identifier+"/"+arm;branch_root=output/paths[key]["root"];branch=master.fork(parent);start=retain(branch.capture(retain=False));acts=[]
                    save(branch_root/"native-fork.json",dict(common=root["source"],fork=start,scope="actual public same-state child of reused observed native joint0 exit"))
                    if arm!="baseline":
                        payload=branch.read("image") if arm=="sham" else base["load_tensor"](output,source_accounts[identifier]["payloads"][arm],"cuda")
                        acts.append(patch(branch,payload,identifier,arm,"native"));start=acts[-1]["after"]
                    save(branch_root/"native-acts.json",acts);state=branch.capture(retain=False).payload
                    rows=[dict(number=0,observed_joint0_source=True,retained=start,fields=base["tensor_tree"](state,old))]
                    ports={}
                    def port(name,value):
                        ports[name]=old["_save_tensor"](output,f"cases/{identifier}/{arm}/first-consumer/native/{name}.npy",value)
                    for number in range(1,111):
                        clock={k:state[k] for k in ("step","phase","cursor")}
                        if number==1:
                            port("gate_ff",graph["_mods"](state["context"]["img_mod"],2)[1][2])
                            with passive_consumer_ports(flux,port):value=branch.continue_()
                            port_records["native"][key]=ports
                            counts["first_consumer_port_sets_per_lane_native"]=counts.get("first_consumer_port_sets_per_lane_native",0)+1
                        else:value=branch.continue_()
                        emit("native",identifier,arm,"native",value);counts["native_target_transitions"]+=1
                        after=branch.capture(retain=False);state=after.payload
                        event_row=dict(step=clock["step"],phase=clock["phase"],index=clock["cursor"])
                        row=dict(number=number,event=event_row,fields=base["tensor_tree"](state,old),receipt=value.fingerprint)
                        if base["retain_transition"](event_row):row["retained"]=retain(after)
                        rows.append(row)
                    require(state["step"]==4 and state["phase"]=="project" and state["cursor"]==0,"native suffix did not close")
                    save(branch_root/"native-rows.json",rows);paths[key]["native_final"]=after.fingerprint
                    event("native_suffix_completed",context=identifier,arm=arm,final_cut=after.fingerprint)
                    if arm=="baseline":
                        value=branch.restore(parent);emit("restore",identifier,arm,"native",value);restored=branch.capture(retain=False)
                        require(restored.fingerprint==parent.fingerprint,"actual complete native rewind differs")
                        save(branch_root/"native-rewind.json",dict(before=rows[-1]["retained"],after=retain(restored),receipt=value.to_dict()));counts["native_source_restores"]+=1
            save(output/"first-consumer-native.json",port_records["native"])
            place("flux","cpu");place("vae","cuda");phase["kind"]="target-vae"
            for key,path in paths.items():
                branch_root=output/path["root"];final=store.load(path["native_final"],device="cpu").payload
                record=old["_native_decoder"](branch_root,vae,final,config,"native_reference",{"Flux2KleinPipeline":Flux2KleinPipeline},processor,{},{})
                save(branch_root/"decoder-native-reference.json",record);counts["native_vae_reference_calls"]+=1
                if key.endswith("/baseline"):
                    repeat=base["native_repeat_decode"](old,branch_root,vae,final,config,{"Flux2KleinPipeline":Flux2KleinPipeline},processor,path["native_final"])
                    save(branch_root/"decoder-native-repeat.json",repeat);counts["native_vae_repeat_calls"]+=1
            place("vae","cpu");return dict(counts=counts,observed=observed,target_paths=len(paths))
        tx.execute("original-native-targets",targets);tx.close()
    final={name:old["_component_signature"](model,prefixes=prefixes[name]) for name,model in models.items()}
    for name in models:old["_assert_signature_equal"](identities[name],final[name],name)
    old["_verify_inventory"](output,inventory)
    for key,expected in declared.items():
        if "per_lane" in key:
            for lane in ("forecast","native"):require(counts[key+"_"+lane]==expected,"actual census differs: "+key+lane)
        elif key not in ("max_gpu_components",):require(counts[key]==expected,"actual census differs: "+key)
    save(output/"state-index.json",paths)
    save(output/"native-contract.json",dict(counts=counts,observed=observed,placements=placements,identities_before=identities,identities_after=final,conditional_scope=SCOPE,no_modified_native_block=True))
    save(output/"source.json",dict(schema=SCHEMA,job_id=config["job_id"],request_sha256=config["request_sha256"],sources=sources,
        helper_source_sha256=HELPERS,parameter_identities=final,parameter_recipe_counts=dict(flux=len(fm["artifacts"]),vae=len(vm["artifacts"])),
        inherited_native_source=dict(job_id=SOURCE_JOB,request_sha256=SOURCE_REQUEST,manifest_sha256=SOURCE_MANIFEST,output_root=SOURCE_ROOT),
        final_resident_parameter_checks="completed before publication",forecast_seal=seal["seal_sha256"]))
    result=dict(schema=SCHEMA,job_id=config["job_id"],request_sha256=config["request_sha256"],counts=counts,contract=declared,observed=observed,
        forecast_seal=seal["seal_sha256"],transaction=verify_transaction(output/"forecast-transaction",expected_seal_sha256=seal["seal_sha256"]),
        mechanics_status="completed",instrument_status="observed source writer; complete image/first-consumer forecasts and native replays",trend_status="uninterpreted observations retained",terminal_status="unassessed",seconds=time.monotonic()-started)
    save(output/"summary.json",result);return result


def worker(config_path):
    import faulthandler

    faulthandler.enable();faulthandler.dump_traceback_later(120,repeat=True);os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    config=read(config_path);validate_config(config);request=read(os.environ["SATURN_DEBUG_REQUEST"])
    identity=hashlib.sha256(json.dumps({k:v for k,v in request.items() if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(identity==request["request_sha256"] and request["config"]==config,"request/config differs")
    for name,expected in request["source_hashes"].items():require(sha(Path(config_path).parent/name)==expected,"executed source changed: "+name)
    job=os.environ["MRUN_JOB_ID"];require(job.startswith("job-") and "/" not in job,"invalid job identity")
    derived=dict(config,job_id=job,request_sha256=identity,_request_record=request,output=str(Path(config["output_scratch"])/job/RESULT))
    result=run(derived,progress=lambda row:print("MDB_ENDOGENOUS_PHASE "+canonical(row).decode(),flush=True))
    finalizer=runpy.run_path(str(Path(__file__).with_name("flux_image_tail_microscope.py")))
    ready=finalizer["finalize"](derived["output"],job_id=result["job_id"],request_sha256=identity,**config["transport_limits"])
    print("MDB_DIRECTORY_READY "+canonical(ready).decode(),flush=True);faulthandler.cancel_dump_traceback_later();return ready


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default=os.environ.get("SATURN_RUN_CONFIG"),required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
