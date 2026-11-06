"""Same-noise prompt factorial: four sealed own-state programs before denoising."""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import runpy
import time
from functools import lru_cache
from pathlib import Path

SEED = 127
RESULT = "flux-prompt-factorial-result"
TEMPLATE = "A {cone_color} cone to the {order} of a {cylinder_color} cylinder on a plain white background. Two separate objects, simple studio product photograph."
CONDITIONS = [{"id":f"c{color}-o{order}","color_assignment":color,"cone_right":order,"seed":SEED,
    "prompt":TEMPLATE.format(cone_color=("green","yellow")[color],cylinder_color=("yellow","green")[color],
        order=("left","right")[order])} for color in range(2) for order in range(2)]
COUNTS = {"model_loads":1,"prompt_encodings":4,"noise_draws":1,"adapter_steps":448,"acts":5,
    "candidate_calls":0,"receipt_emissions":453,"images":9,"vae_calls":9,"physical_batch":1}
LANDMARKS = [0,5,25,26,27]
PRIOR_INITIAL_SHA256 = "1ae1a639cb2407adadda597af48ef581d0129f6cf2b87df7e249274247b6129b"
PARAMETER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
CHECKPOINT_FILE = "transformer/diffusion_pytorch_model.safetensors"
CHECKPOINT_FILE_SHA256 = "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e"
OBSERVATION_PLAN = {"landmark_local_indices":LANDMARKS,"witness_policy":"64 evenly spaced flat coordinates per present field, shape-only selection",
    "native_transition_metadata":448,"predicted_transitions":448,"typed_cut_policy":"origins, landmarks and terminal consumers only; every other complete manifest/receipt retained as metadata, not hydratable replay",
    "comparison_scope":"Full-field native errors only at landmarks/finals; fixed-coordinate witness errors elsewhere. No full intermediate error norm or complete replay claim."}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4<<20),b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def save(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


@lru_cache(maxsize=1)
def prior():
    path = Path(__file__).with_name("flux_initial_noise_campaign.py")
    if sha(path) != PRIOR_INITIAL_SHA256:
        raise ValueError("qualified initial-noise helper differs")
    result = runpy.run_path(str(path))
    result["prior"]()  # Validate the full immutable transitive numerical cohort.
    return result


def contract(config=None):
    if config is not None and config.get("conditions",CONDITIONS) != CONDITIONS:
        raise ValueError("prompt factorial requires the four fixed seed127 cells")
    return json.loads(json.dumps({"conditions":CONDITIONS,"seed":SEED,"counts":COUNTS,"mathematical_steps":448,
        "forecast_order":[row["id"] for row in CONDITIONS],"native_order":[row["id"] for row in CONDITIONS],
        "sham_condition":"c0-o0","observation_plan":OBSERVATION_PLAN,"block_indices":list(range(20)),
        "logical_steps":[0,1,2,3],"target_calls":target_counts(16),"result":RESULT,
        "transaction_seals":["transactions/"+row["id"]+"/seal.json" for row in CONDITIONS],
        "cohort_seal":"cohort-seal.json"}))


def target_counts(multiplier=1):
    return {**{str(index):multiplier for index in range(20)},"norm_out":multiplier,"proj_out":multiplier,
        **{name:multiplier for name in ("x_embedder","context_embedder","time_guidance_embed",
            "double_stream_modulation_img","double_stream_modulation_txt","single_stream_modulation")},
        "pos_embed":2*multiplier,**{"joint"+str(index):multiplier for index in range(5)}}


def fixed_config():
    result = {"conditions":CONDITIONS,"seed":SEED,"height":512,"width":512,"steps":4,
        "block_indices":list(range(20)),"logical_steps":[0,1,2,3],"cpu_threads":4,
        "query_chunk":64,"projection_chunk":256,"neuron_chunk":256,"retain_fields":False,
        "sample_indices":[0,31,511,512,543,767,1023,1535],"image_sample_indices":[0,31,32,255,511,512,767,1023],
        "witness_count":64,"landmark_local_indices":LANDMARKS,"observation_plan":OBSERVATION_PLAN,
        "parameter_storage":"checkpoint-recipe-v1","array_storage":"bf16-bits","archive_integer_ids":"native-int64",
        "parameter_checkpoint_file":CHECKPOINT_FILE,"parameter_checkpoint_sha256":CHECKPOINT_FILE_SHA256,
        "expected_parameter_artifacts_module_sha256":PARAMETER_SHA256,
        "expected_closure_module_sha256":"45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23",
        "expected_joint_closure_module_sha256":"bb352ae4859335abdd751d33ccc2774fa4d37c53f5167c5012183c693b0d709c",
        "expected_tail_closure_module_sha256":"2788c984045098532009e105219e7c9cfcc38732a54d892bdc296149a979b586",
        "expected_final_closure_module_sha256":"b70f3d68f7fb0b1c0e2892bb7ab4efb8828a906ef80d9acb6f794608d673e579",
        "expected_derived_terminal_module_sha256":"7291e89a5b09823c5f595b3263626b23fb44951f46ef9fa5fd7c59002b8866c8",
        "transaction_max_bytes":64<<20,"transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":8<<30,"max_member_bytes":1<<30}}
    return json.loads(json.dumps(result))


def make_config(base_config=None):
    result = json.loads(json.dumps(base_config or {}))
    result.update(fixed_config())
    result.update({"max_"+key:COUNTS[key] for key in ("adapter_steps","acts","candidate_calls","images","vae_calls")})
    result.update(max_wall_seconds=3000,scope="One seed127 noise draw, four prompt cells; all448 own-state predictions and four actual VAE PNGs precede any native denoiser. Full comparison at landmarks; fixed witnesses elsewhere. No semantic or universal accuracy gate.",
        hypotheses={"scope":"Prospective factorial contrasts and numerical discrepancies are observations, including misses and large coordinates."})
    return result


def validate_config(config):
    expected = fixed_config()
    if any(config.get(key) != value for key,value in expected.items()):
        raise ValueError("factorial scientific/profile/storage geometry differs")
    for key in ("adapter_steps","acts","candidate_calls","images","vae_calls"):
        if type(config.get("max_"+key)) is not int or config["max_"+key] != COUNTS[key]:
            raise ValueError("factorial physical census differs")
    if type(config.get("max_wall_seconds")) is not int or not 1 <= config["max_wall_seconds"] <= 3600:
        raise ValueError("factorial requires finite reviewed wall budget")
    for key in ("scope","hypotheses"):
        if not config.get(key):
            raise ValueError("factorial declaration absent: "+key)


def value_record(value,*,archive=None,witness_count=64):
    """Full identity/moments plus shape-fixed witness; no large array in metadata."""
    import numpy as np

    value = np.ascontiguousarray(value)
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError("observed/predicted numerical field must be finite FP32")
    flat = value.reshape(-1)
    indices = np.unique(np.linspace(0,value.size-1,min(witness_count,value.size),dtype=np.int64))
    wide = value.astype(np.float64)
    descriptor = {"shape":list(value.shape),"dtype":"float32","fp32_le_sha256":hashlib.sha256(value.astype("<f4",copy=False).tobytes()).hexdigest()}
    if not np.any(value.view(np.uint32)&np.uint32(65535)):
        descriptor["bf16_le_sha256"] = hashlib.sha256((value.view(np.uint32)>>16).astype("<u2").tobytes()).hexdigest()
    result = {"descriptor":descriptor,"moments":{"elements":value.size,"rms":float(np.sqrt(np.mean(wide*wide))),
        "mean":float(wide.mean()),"min":float(wide.min()),"max":float(wide.max())},
        "witness":{"flat_indices":indices.tolist(),"values":flat[indices].tolist()},"archive":archive}
    return result


def stage_kind(local):
    if local == 0:
        return "project",None
    if local <= 5:
        return "joint",local-1
    if local <= 25:
        return "single",local-6
    return ("readout",None) if local == 26 else ("commit",None)


def descriptor(value):
    return value_record(value,witness_count=1)["descriptor"]


def context_fields(context):
    return {**{"context."+key:context[key] for key in ("temb","img_mod","txt_mod","single_mod")},
        "rope.cos":context["rope"][0],"rope.sin":context["rope"][1]}


def predict_events(binding,origin,checkpoint,config):
    """Stream qualified pure kernels; every next input is our preceding output."""
    import numpy as np

    from mdb.tensor_archive import load_tensor_archive

    root = binding["_root"]
    load = lambda record:load_tensor_archive(root,record)
    inputs = origin["inputs"]
    latent,conditioning,img_ids,txt_ids = (load(inputs[key]) for key in ("latent","conditioning","img_ids","txt_ids"))
    kernels = {name:runpy.run_path(str(Path(__file__).with_name(name))) for name in
        ("flux_reentry_reference.py","flux_transition_reference.py","flux_step_boundary_reference.py")}
    for logical in range(4):
        schedule,spec = binding["pass_schedules"][str(logical)],binding["project"]
        weights = {key:checkpoint.load(record) for key,record in spec["parameters"].items()}
        project = kernels["flux_reentry_reference.py"]["predict_project"](latent,conditioning,img_ids,txt_ids,
            load(schedule["timestep"]),weights,axes_dims_rope=spec["axes_dims_rope"],rope_theta=spec["rope_theta"],
            timestep_channels=spec["timestep_channels"],projection_chunk=config["projection_chunk"])
        del weights
        image,text,context = project["image"],project["text"],project["context"]
        context_id = identity({key:descriptor(value) for key,value in context_fields(context).items()})
        yield (logical,0,{"latent":descriptor(latent),"origin_conditioning":inputs["conditioning"]["descriptor"],
            "origin_txt_ids":inputs["txt_ids"]["descriptor"],"schedule":schedule},
            {"image":image,"text":text,**context_fields(context)})
        for index in range(5):
            spec = binding["joint"+str(index)]
            before = {"image":descriptor(image),"text":descriptor(text),"own_project_context":context_id}
            weights = {key:checkpoint.load(record) for key,record in spec["parameters"].items()}
            result = kernels["flux_reentry_reference.py"]["predict_joint"](image,text,context,weights,
                heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
                query_chunk=config["query_chunk"],projection_chunk=config["projection_chunk"],retain_fields=False)
            del weights
            image,text = result["image"],result["text"]
            yield logical,index+1,before,{"image":image,"text":text}
        current = np.concatenate((text,image),axis=1)
        for index in range(20):
            spec = binding["final_blocks"][str(index)]
            before = {"residual":descriptor(current),"image":descriptor(current[:,binding["text_rows"]:]),
                "text":descriptor(current[:,:binding["text_rows"]]),"own_project_context":context_id}
            weights = {key:checkpoint.load(record) for key,record in spec["parameters"].items()}
            result = kernels["flux_transition_reference.py"]["predict_transition"](current,context["single_mod"],**weights,
                heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
                rope_cos=context["rope"][0],rope_sin=context["rope"][1],query_chunk=config["query_chunk"],
                projection_chunk=config["projection_chunk"],neuron_chunk=config["neuron_chunk"],retain_fields=False,
                text_rows=None,sample_indices=config["sample_indices"])
            del weights
            current = result["output"]
            text,image = current[:,:binding["text_rows"]],current[:,binding["text_rows"]:]
            yield logical,index+6,before,{"image":image,"text":text}
        spec = binding["readout"]
        weights = {key:checkpoint.load(record) for key,record in spec["parameters"].items()}
        readout = kernels["flux_step_boundary_reference.py"]["predict_readout"](image,context["temb"],**weights,
            eps=spec["eps"],neuron_chunk=config["projection_chunk"],sample_indices=config["image_sample_indices"],retain_fields=False)
        del weights
        noise = readout["noise"]
        yield logical,26,{"image":descriptor(image),"own_project_context":context_id},{"noise":noise}
        committed = kernels["flux_step_boundary_reference.py"]["predict_flow_commit"](latent,noise,
            load(schedule["sigma"]),load(schedule["sigma_next"]))
        before = {"latent":descriptor(latent),"noise":descriptor(noise),"sigma":schedule["sigma"],"sigma_next":schedule["sigma_next"]}
        latent = committed["latent"]
        yield logical,27,before,{"latent":latent}


def predict_program(binding,origin,checkpoint,config,array,event):
    import numpy as np
    import torch

    records = []
    def archive(name,value):
        dtype = torch.float32 if name.startswith("rope.") else torch.bfloat16
        native = torch.from_numpy(np.array(value,copy=True)).to(dtype=dtype)
        if not np.array_equal(native.float().numpy().view(np.uint32),np.ascontiguousarray(value).view(np.uint32)):
            raise ValueError("landmark storage would round predicted values")
        return array(native)
    for logical,local,inputs,fields in predict_events(binding,origin,checkpoint,config):
        kind,index = stage_kind(local)
        retained = local in LANDMARKS
        record = {"logical_step":logical,"local_index":local,"transition_index":logical*28+local,"kind":kind,"index":index,
            "inputs":inputs,"outputs":{name:value_record(value,archive=archive(name,value) if retained else None,
                witness_count=config["witness_count"]) for name,value in fields.items()},"full_arrays_retained":retained}
        records.append(record)
        event("prediction_transition",condition=origin["condition_id"],logical_step=logical,local_index=local)
    if len(records) != 112:
        raise ValueError("forecast stopped before complete112 transitions")
    return {"schema":"mdb-prompt-factorial-program-v1","origin_cut":origin["cut"],"condition_id":origin["condition_id"],
        "transitions":records,"terminal_latent":records[-1]["outputs"]["latent"],"chain_length":112,
        "scope":OBSERVATION_PLAN["comparison_scope"]}


def comparison(actual,reference):
    import numpy as np

    actual,reference = np.asarray(actual),np.asarray(reference)
    if actual.shape != reference.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("comparison geometry/finite differs")
    delta = actual.astype(np.float64)-reference.astype(np.float64)
    denominator = float(np.square(reference.astype(np.float64)).sum())
    energy = float(np.square(delta).sum())
    return {"elements":actual.size,"rms":float(np.sqrt(energy/actual.size)),"mae":float(np.abs(delta).mean()),
        "signed_mean":float(delta.mean()),"max_abs":float(np.abs(delta).max()),"squared_l2":energy,
        "relative_rms":float(np.sqrt(energy/denominator)) if denominator else None,"sign":"native minus sealed prediction",
        "relative_rms_denominator":"sealed prediction RMS"}


def emit_parameters(checkpoint,output,request):
    from mdb import parameter_artifacts

    module = Path(inspect.getfile(parameter_artifacts))
    if sha(module) != PARAMETER_SHA256:
        raise ValueError("standalone parameter retriever differs")
    manifest = {"schema":"mdb-parameter-artifacts-v1","artifacts":list(checkpoint.entries.values()),
        "defaults":{"artifact_root":"parameter-cache","checkpoint_root":str(checkpoint.path.parent.parent),"ssh_host":"beast","python":"python3"},
        "provenance":{"job_id":request["job_id"],"request_sha256":request["request_sha256"],
            "producer_source_sha256":sha(Path(__file__)),"parameter_exporter_source_sha256":PRIOR_INITIAL_SHA256,
            "retrieval_module_sha256":PARAMETER_SHA256,"scope":"Shared169 checkpoint recipes; no saved parameter arrays. Hydrate outside immutable result."}}
    parameter_artifacts.validate_manifest(manifest)
    save(output/"parameter-artifacts.json",manifest)
    with (output/"retrieve_weights.py").open("xb") as stream:
        stream.write(module.read_bytes())
    return {"path":"parameter-artifacts.json","sha256":sha(output/"parameter-artifacts.json"),
        "script_sha256":PARAMETER_SHA256,"artifacts":len(manifest["artifacts"]),"parameter_files_written":0}


def validate_program(program,origin):
    rows = program["transitions"]
    if program["origin_cut"] != origin["cut"] or program["condition_id"] != origin["condition_id"] or len(rows) != 112:
        raise ValueError("sealed program origin/length differs")
    previous,latent,context_id = None,None,None
    for position,row in enumerate(rows):
        logical,local = divmod(position,28)
        if (row["transition_index"],row["logical_step"],row["local_index"],row["kind"],row["index"]) != (position,logical,local,*stage_kind(local)):
            raise ValueError("program order differs")
        outputs,inputs = row["outputs"],row["inputs"]
        if row["full_arrays_retained"] != (local in LANDMARKS) or any((value["archive"] is not None) != (local in LANDMARKS) for value in outputs.values()):
            raise ValueError("declared landmark coverage differs")
        if local == 0:
            source = origin["inputs"]
            if inputs["origin_conditioning"] != source["conditioning"]["descriptor"] or inputs["origin_txt_ids"] != source["txt_ids"]["descriptor"]:
                raise ValueError("project conditioning/integer IDs differ from own source")
            if logical == 0 and (inputs["latent"]["shape"] != source["latent"]["descriptor"]["shape"]
                    or inputs["latent"].get("bf16_le_sha256") != source["latent"]["descriptor"]["sha256"]):
                raise ValueError("first project did not consume actual observed Z0 bytes")
            if latent is not None and inputs["latent"] != latent:
                raise ValueError("project refreshed own preceding latent")
            context_id = identity({key:value["descriptor"] for key,value in outputs.items() if key.startswith(("context.","rope."))})
        elif local <= 25:
            if any(inputs[key] != previous[key]["descriptor"] for key in ("image","text")) or inputs["own_project_context"] != context_id:
                raise ValueError("block refreshed own image/text/context")
        elif local == 26:
            if inputs["image"] != previous["image"]["descriptor"] or inputs["own_project_context"] != context_id:
                raise ValueError("readout refreshed own image/context")
        else:
            if inputs["noise"] != previous["noise"]["descriptor"] or (latent is not None and inputs["latent"] != latent):
                raise ValueError("commit refreshed own latent/noise")
            latent = outputs["latent"]["descriptor"]
        previous = outputs
    if program["terminal_latent"] != rows[-1]["outputs"]["latent"]:
        raise ValueError("terminal consumer not bound to final prediction")


def run(*,model,adapter,sessions,checkpoint,output,config,request,source,counts,array=None,
        guard_components=None,event=lambda *args,**kwargs:None,wall=lambda:None,production_profile=True):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.flux_derived_terminal import derive_terminal
    from mdb.flux_final_latent_closure import closure_act
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.tensor_archive import load_tensor_archive

    output = Path(output)
    output.mkdir(exist_ok=True,parents=True)
    if (output/"plan.json").exists():
        raise ValueError("immutable factorial output already has a plan")
    (output/"native").mkdir()
    old = prior()
    array = array or old["step"]()["make_array_writer"](output,wall)
    store = LocalStore(output/"typed-store")
    save(output/"plan.json",config)
    ids = [row["id"] for row in CONDITIONS]
    if set(sessions) != set(ids) or adapter.execution.get("adapter") != "mdb-flux-final-latent-closure-v1":
        raise ValueError("factorial origins/closure adapter differ")
    cuts = {key:sessions[key].capture(retain=False) for key in ids}
    common = None
    origins = {}
    for key,cut in cuts.items():
        p = cut.payload
        for name in ("conditioning","txt_ids"):
            first = cuts[ids[0]].payload[name]
            if p[name].shape != first.shape or p[name].dtype != first.dtype:
                raise ValueError("prompt origins changed source geometry/dtype")
        invariant = describe({name:value for name,value in p.items() if name not in {"conditioning","txt_ids"}})
        if common is None:
            common = invariant
        if invariant != common or any(cut.execution_point[name] != value for name,value in
                {"logical_step":0,"phase":"project","index":None,"local_clock":0}.items()):
            raise ValueError("origins must share exact Z0/IDs/schedule/clock and differ only permitted text inputs")
        origins[key] = {**old["origin_record"](cut,array),"condition_id":key}
        store.save(cut)
    if any(counts[key] for key in ("adapter_steps","acts","images","vae_calls","receipt_emissions")):
        raise ValueError("factorial starts after physical work")
    binding = old["bind_inputs"](model=model,initial=cuts[ids[0]],array=array,counts=counts,config=config,
        production_profile=production_profile,checkpoint=checkpoint)
    binding.pop("origins")
    binding["origins"] = origins
    binding["parameter_artifacts"] = emit_parameters(checkpoint,output,request)
    if production_profile and binding["parameter_artifacts"]["artifacts"] != 169:
        raise ValueError("qualified parameter inventory differs")
    save(output/"binding.json",binding)
    save(output/"source.json",{**source,"job_id":request["job_id"],"request_sha256":request["request_sha256"],
        "source_hashes":request["source_hashes"],"conditions":CONDITIONS,"binding_sha256":sha(output/"binding.json"),
        "model_identity":adapter.model_identity,"execution":dict(adapter.execution),
        "parameter_artifacts":binding["parameter_artifacts"],"origin_identity":{key:identity(describe(cut.payload)) for key,cut in cuts.items()}})
    components = tuple(guard_components or (model,))
    parameters = old["prior"]()["parameter_census"](components)
    origin_identities = {key:identity(describe(cut.payload)) for key,cut in cuts.items()}
    state = {"phase":"forecast","condition":None,"blocked":0,"targets":target_counts(0),
        "by_condition":{key:target_counts(0) for key in ids}}
    targets = {model.single_transformer_blocks[index]:str(index) for index in range(20)}
    targets.update({model.norm_out:"norm_out",model.proj_out:"proj_out",model.pos_embed:"pos_embed",
        **{model.transformer_blocks[index]:"joint"+str(index) for index in range(5)},
        **{getattr(model,name):name for name in ("x_embedder","context_embedder","time_guidance_embed",
            "double_stream_modulation_img","double_stream_modulation_txt","single_stream_modulation")}})
    vae_modules = {id(module) for component in components if component is not model for module in component.modules()}
    def guard(module,args):
        if state["phase"] != "native" and not (state["phase"] == "vae" and id(module) in vae_modules):
            state["blocked"] += 1
            raise ValueError("native module forbidden before cohort seal/four predicted images")
        if module in targets:
            key = targets[module]
            state["targets"][key] += 1
            state["by_condition"][state["condition"]][key] += 1
    handles = [module.register_forward_pre_hook(guard) for component in components for module in component.modules()]
    transactions,programs,seals,rows,native_paths = {},{},{},{},{}
    def integrity():
        checkpoint._check()
        if state["blocked"] or old["prior"]()["parameter_census"](components) != parameters or any(
                identity(describe(cut.payload)) != origin_identities[key] for key,cut in cuts.items()):
            raise ValueError("source/parameter/guard integrity differs")
    def capture_metadata(cut,retained):
        manifest = cut.manifest()
        if identity(manifest) != cut.fingerprint:
            raise ValueError("public StateCut descriptor fingerprint differs")
        if retained:
            store.save(cut)
        return {"cut":cut.fingerprint,"manifest":manifest,"point":dict(cut.execution_point),
            "retained_full_cut":retained,"scope":"Hydratable snapshot" if retained else "Descriptor-only manifest; payload and complete replay not retained"}
    def receipt(value,branch):
        body = value.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps",1)
        counts["receipt_emissions"] += 1
        if counts["adapter_steps"] > 448 or counts["acts"] > 5:
            raise ValueError("factorial physical budget exceeded")
        store.receipt(value)
        with (output/"receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence":counts["receipt_emissions"]-1,"branch":branch,"receipt":value.fingerprint},sort_keys=True)+"\n")
    def render(name,session,acts,entry):
        cut = session.capture(retain=False)
        if cut.payload["step"] != 4 or any(cut.payload[key] is not None for key in ("image","text","context","noise_prediction")):
            raise ValueError("terminal reset incomplete")
        state["phase"] = "vae"
        counts["vae_calls"] += 1
        png = adapter.decode(session,height=512,width=512)
        path = output/(name+".png")
        png.save(path)
        counts["images"] += 1
        row = {"id":name,"entry":entry,"acts":acts,"final":capture_metadata(cut,True),
            "final_latent":array(cut.payload["latent"]),"image":{"path":path.name,"sha256":sha(path)},"denoiser_suffix_steps":0}
        save(output/"rows"/(name+".json"),row)
        rows[name] = row
        return row
    try:
        for key in ids:
            state["phase"] = "forecast"
            frontier = {**{name:value for name,value in binding.items() if name != "origins"},
                "origin":origins[key],"condition":next(row for row in CONDITIONS if row["id"] == key),"counts_before_seal":dict(counts)}
            tx = ForecastTransaction(output/"transactions"/key,model_identity=adapter.model_identity,
                input_identity=identity(frontier),source_identity=identity(request["source_hashes"]),
                frontier={"kind":"architectural","binding":frontier},execution_scope=config["scope"],
                budgets={"max_native_calls":1,"max_bytes":config["transaction_max_bytes"],"max_wall_ms":config["max_wall_seconds"]*1000})
            transactions[key] = tx
            tx.frontier("source_binding",frontier)
            tx.predict("program",lambda _,key=key: {"predictions":predict_program({**binding,"_root":str(output)},origins[key],checkpoint,config,array,event)})
            programs[key] = old["util"]()["json_value"](tx.prediction("program"))
            validate_program(programs[key],origins[key])
            integrity()
            seals[key] = tx.seal()
            event("condition_sealed",condition=key,seal=seals[key],counts=dict(counts))
        cohort = {"schema":"mdb-prompt-factorial-cohort-seal-v1","condition_ids":ids,"seals":seals,
            "origins":{key:origins[key]["cut"] for key in ids},"terminal_descriptors":{key:programs[key]["terminal_latent"]["descriptor"] for key in ids},
            "binding_sha256":sha(output/"binding.json"),"plan_sha256":sha(output/"plan.json"),"counts":dict(counts),"target_calls":state["targets"]}
        save(output/"cohort-seal.json",{**cohort,"fingerprint":identity(cohort)})
        event("cohort_sealed",cohort_sha256=sha(output/"cohort-seal.json"),counts=dict(counts))
        for position,key in enumerate(ids):
            tx = transactions[key]
            def consume(context,key=key):
                program = old["util"]()["json_value"](context.predictions["program"])
                native = load_tensor_archive(output,program["terminal_latent"]["archive"])
                if descriptor(native) != program["terminal_latent"]["descriptor"]:
                    raise ValueError("sealed terminal payload differs")
                reference = cuts[key].payload["latent"]
                payload = torch.empty_strided(reference.shape,reference.stride(),device=reference.device,dtype=reference.dtype)
                payload.copy_(torch.from_numpy(np.array(native,copy=True)).to(reference.device,reference.dtype))
                derived = derive_terminal(adapter,cuts[key],predicted_latent=payload,name=key+".terminal")
                save(output/"derivations"/(key+".json"),derived.derivation)
                session = Session.from_cut(adapter,derived.scaffold)
                entry = capture_metadata(derived.scaffold,True)
                act = session.apply(derived.action)
                receipt(act,key+".predicted")
                row = render(key+".predicted",session,[act.to_dict()],entry)
                integrity()
                if counts["adapter_steps"] != 0 or any(state["targets"].values()):
                    raise ValueError("native target before all predicted PNGs")
                first = {"condition_id":key,"cohort_seal_sha256":sha(output/"cohort-seal.json"),"image":row["image"],
                    "counts":dict(counts),"target_calls":state["targets"],"origin_cut":cuts[key].fingerprint}
                save(output/"first-images"/(key+".json"),first)
                return first
            tx.execute("predicted_terminal_vae",consume)
            tx.close()
            verify_transaction(tx.root,expected_seal_sha256=seals[key]["seal_sha256"],max_bytes=config["transaction_max_bytes"])
            save(output/"transactions"/key/"report.json",tx.report())
            if counts["images"] != position+1:
                raise ValueError("predicted image order differs")
        baseline = None
        for key in ids:
            session = Session.from_cut(adapter,cuts[key])
            state.update(phase="native",condition=key)
            records = []
            for position in range(112):
                wall()
                step_receipt = session.continue_()
                receipt(step_receipt,key+".native")
                cut = session.capture(retain=False)
                local = position%28
                p = cut.payload
                fields = ({"image":p["image"],"text":p["text"],**context_fields(p["context"])} if local == 0 else
                    {"image":p["image"],"text":p["text"]} if local <= 25 else
                    {"noise":p["noise_prediction"]} if local == 26 else {"latent":p["latent"]})
                retained = local in LANDMARKS
                values,errors = {},{}
                predicted = programs[key]["transitions"][position]["outputs"]
                for name,value in fields.items():
                    wide = value.detach().float().cpu().numpy()
                    values[name] = value_record(wide,archive=array(value) if retained else None,witness_count=config["witness_count"])
                    witness = values[name]["witness"]
                    if witness["flat_indices"] != predicted[name]["witness"]["flat_indices"]:
                        raise ValueError("witness geometry differs")
                    errors[name] = {"fixed_witness":comparison(witness["values"],predicted[name]["witness"]["values"]),"full_field":None}
                    if retained:
                        saved = load_tensor_archive(output,predicted[name]["archive"])
                        if descriptor(saved) != predicted[name]["descriptor"]:
                            raise ValueError("sealed landmark changed")
                        errors[name]["full_field"] = comparison(wide,saved)
                record = {"transition_index":position,"logical_step":position//28,"local_index":local,
                    "receipt":step_receipt.to_dict(),"capture":capture_metadata(cut,retained),"outputs":values,"errors":errors,
                    "comparison_scope":"Full-field" if retained else "Fixed coordinates only; no retained complete native payload"}
                records.append(record)
                with (output/"native"/(key+".jsonl")).open("a") as stream:
                    stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
                event("native_transition",condition=key,transition_index=position,counts=dict(counts))
            native_paths[key] = records
            final = session.capture(retain=False)
            render(key+".native",session,[],capture_metadata(final,True))
            if key == ids[0]:
                baseline = final
        session = Session.from_cut(adapter,baseline)
        expected = {key:baseline.payload[key] for key in ("latent","image","text","context")}
        action = session.apply(closure_act(baseline,**expected,name="factorial.terminal-sham"))
        receipt(action,ids[0]+".sham")
        render(ids[0]+".sham",session,[action.to_dict()],capture_metadata(baseline,True))
        integrity()
    finally:
        for tx in transactions.values():
            tx.close()
        for handle in handles:
            handle.remove()
    if any(counts[key] != COUNTS[key] for key in ("adapter_steps","acts","images","vae_calls","receipt_emissions")) or state["targets"] != target_counts(16):
        raise ValueError("completed physical/module census differs")
    native_row,sham_row = rows[ids[0]+".native"],rows[ids[0]+".sham"]
    summary = {"schema":"mdb-prompt-factorial-result-v1","job_id":request["job_id"],"request_sha256":request["request_sha256"],
        "counts":dict(counts),"conditions":CONDITIONS,"seals":seals,"cohort_seal_sha256":sha(output/"cohort-seal.json"),
        "rows":rows,"target_calls":state["targets"],"target_calls_by_condition":state["by_condition"],"blocked_entries":state["blocked"],
        "control_observations":{"sham_png_exact":native_row["image"]["sha256"] == sham_row["image"]["sha256"],
            "sham_latent_exact":native_row["final_latent"] == sham_row["final_latent"]},"observation_plan":OBSERVATION_PLAN,
        "mechanics_status":"pending-independent-audit","terminal_status":"unassessed"}
    save(output/"summary.json",summary)
    return summary


def worker():
    """One public installed pipeline, four encodes, one noise draw; no jobs here."""
    import faulthandler
    import importlib.metadata
    import shutil

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90,repeat=True)
    config,request = (json.loads(Path(os.environ[key]).read_text()) for key in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(path) != checksum for path,checksum in request["source_hashes"].items()):
        raise ValueError("factorial request/source cohort differs")
    request = {**request,"job_id":os.environ["MRUN_JOB_ID"]}
    output = Path(RESULT)
    output.mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb import flux_derived_terminal, parameter_artifacts
    from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("qualified production lane requires CUDA")
    old = prior()
    boundary,reentry = (runpy.run_path(str(Path(__file__).with_name(name))) for name in
        ("flux_step_boundary_reference.py","flux_reentry_reference.py"))
    abi = old["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=FluxFinalLatentClosureAdapter,config=config)
    save(output/"pre-model-source-abi.json",abi)
    save(output/"cuda-scalar-qualification.json",boundary["qualify_torch_scalar_promotion"](torch,device="cuda"))
    if shutil.disk_usage(output).free < config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("output filesystem lacks declared finite free-space bound")
    counts = dict.fromkeys(COUNTS,0)
    counts["physical_batch"] = 1
    def event(kind,**fields):
        body = old["util"]()["json_value"]({"kind":kind,"seconds":time.monotonic()-started,"counts":dict(counts),**fields})
        with (output/"timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body,sort_keys=True,allow_nan=False)+"\n")
        print("MDB_FLUX_PROMPT_FACTORIAL_PHASE "+json.dumps(body,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started > config["max_wall_seconds"]:
            raise RuntimeError("finite factorial wall budget exhausted")
    array = old["step"]()["make_array_writer"](output,wall)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if old["util"]()["identity"](inventory) != config["expected_checkpoint_sha256"] or inventory.get(CHECKPOINT_FILE) != CHECKPOINT_FILE_SHA256:
        raise ValueError("complete qualified checkpoint identity differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("qualified distilled native profile differs")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind):sha(inspect.getfile(kind)) for kind in (type(pipe),type(pipe.text_encoder),
        type(pipe.transformer),type(pipe.scheduler),type(pipe.vae),Flux2KleinAdapter,FluxFinalLatentClosureAdapter,
        type(pipe.transformer.transformer_blocks[0]),type(pipe.transformer.single_transformer_blocks[0]),
        type(pipe.transformer.single_transformer_blocks[0].attn.processor),type(pipe.transformer.norm_out),
        type(pipe.transformer.pos_embed),type(pipe.transformer.time_guidance_embed.timestep_embedder))}
    implementations.update({inspect.getfile(module):sha(inspect.getfile(module)) for module in (flux_derived_terminal,parameter_artifacts)})
    pipe.text_encoder.to("cuda")
    encodings,raw_encodings = {},{}
    for condition in CONDITIONS:
        with torch.inference_mode():
            conditioning,txt_ids = pipe.encode_prompt(condition["prompt"],device=torch.device("cuda"),num_images_per_prompt=1,max_sequence_length=512)
        counts["prompt_encodings"] += 1
        key = condition["id"]
        raw_encodings[key] = {"conditioning":array(conditioning),"txt_ids":array(txt_ids)}
        encodings[key] = conditioning.cpu(),txt_ids.cpu()
        event("prompt_encoded",condition=key)
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = FluxFinalLatentClosureAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    with torch.inference_mode():
        dtype = encodings[CONDITIONS[0]["id"]][0].dtype
        latent,img_ids = pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,dtype,
            torch.device("cuda"),torch.Generator(device="cuda").manual_seed(SEED))
        counts["noise_draws"] += 1
        sigmas = None if getattr(pipe.scheduler.config,"use_flow_sigmas",False) else np.linspace(1.,.25,4)
        timesteps,steps = retrieve_timesteps(pipe.scheduler,4,torch.device("cuda"),sigmas=sigmas,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
        if steps != 4:
            raise ValueError("native scheduler geometry differs")
        sessions = {key:adapter.session(latent=latent.clone(),conditioning=conditioning.to("cuda"),img_ids=img_ids.clone(),
            txt_ids=txt_ids.to("cuda"),timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
            for key,(conditioning,txt_ids) in encodings.items()}
    source = {"schema":"mdb-prompt-factorial-source-v1","encodings":raw_encodings,"checkpoint_files":inventory,
        "checkpoint_sha256":old["util"]()["identity"](inventory),"implementation_sources":implementations,
        "versions":{name:importlib.metadata.version(name) for name in ("torch","numpy","diffusers","transformers","saturn-pub","mdb-runtime")},
        "pre_model_source_abi":{"path":"pre-model-source-abi.json","sha256":sha(output/"pre-model-source-abi.json")},
        "cuda_scalar_qualification":{"path":"cuda-scalar-qualification.json","sha256":sha(output/"cuda-scalar-qualification.json")}}
    checkpoint = old["CheckpointParameters"](root,CHECKPOINT_FILE,CHECKPOINT_FILE_SHA256,pipe.transformer)
    with torch.inference_mode():
        summary = run(model=pipe.transformer,adapter=adapter,sessions=sessions,checkpoint=checkpoint,output=output,
            config=config,request=request,source=source,counts=counts,array=array,guard_components=(pipe.transformer,pipe.vae),
            event=event,wall=wall)
    if counts != COUNTS:
        raise ValueError("global load/encoding/noise/native census differs")
    event("complete")
    manifest = old["step"]()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"],
        max_expanded_bytes=config["transport_limits"]["max_expanded_bytes"],max_member_bytes=config["transport_limits"]["max_member_bytes"])
    faulthandler.cancel_dump_traceback_later()
    print("MDB_FLUX_PROMPT_FACTORIAL_COMPLETE "+json.dumps({"summary":summary,"manifest":manifest},sort_keys=True),flush=True)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path(RESULT)
        if output.is_dir() and not (output/"remote-manifest.json").exists():
            save(output/"failure.json",{"schema":"mdb-prompt-factorial-failure-v1","job_id":os.environ.get("MRUN_JOB_ID"),
                "error":type(error).__name__+": "+str(error),"scope":"Completed rows preserved; no retry or fabricated completed report."})
        raise
