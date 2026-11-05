"""Seal own-state full-step/project/joint0/joint1 predictions, then native futures.

Only byte-pinned pure predecessor functions are reused. This module owns its
finite execution, guards and atomic joint1/joint2 admission; predecessor worker
functions/globals are never changed or invoked as a different experiment.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import shutil
import time
from functools import lru_cache
from pathlib import Path

CONTEXT = {"id":"p1-s101","prompt_id":"p1","seed":101,
    "prompt":"A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
BLOCK_INDICES = list(range(6,20))
BRANCHES = [{"id":"native","mode":"native"},{"id":"sham","mode":"sham"},
    {"id":"inherited.predicted","mode":"inherited"},{"id":"joint1.predicted","mode":"joint1"}]
ORDER = [row["id"] for row in BRANCHES]
COUNTS = {"model_loads":1,"prompt_encodings":1,"noise_draws":1,"adapter_steps":188,"acts":3,
    "candidate_calls":0,"receipt_emissions":191,"images":4,"vae_calls":4,"physical_batch":1}
SAMPLE_INDICES = [0,31,511,512,543,767,1023,1535]
IMAGE_SAMPLE_INDICES = [0,31,32,255,511,512,767,1023]
PROJECT_TARGETS = ("x_embedder","context_embedder","time_guidance_embed","double_stream_modulation_img",
    "double_stream_modulation_txt","single_stream_modulation")
OBSERVATION_PLAN = {"reference_rows":["native"],"boundaries_per_branch":26,
    "slots":["image","text","latent","noise_prediction"],"witness_count":64,
    "reference_storage":"lossless-bf16-bits","candidate_storage":"descriptor-and-fixed-coordinate-witness",
    "scope":"Actual common step3/joint2 plus25 native exits; native reference arrays retained. Candidate intermediate full norms are worker observations with64 fixed witnesses, not independently auditable full arrays."}
OLD_CLOSURE_SHA256 = "45dd999fe54103ca7676248652798dd295fbc727d0fdf38cd912fa48b8011c23"
LEGACY_SOURCES = {
    "flux_reentry_campaign.py":"719312be66c017041106302c6acf5374299b17cba7aefe13e38d42f9478d8027",
    "flux_reentry_reference.py":"e7ba7e9b55d775386767a15d87af4f011d64831322a38ab7ee4c59537b31e108",
    "flux_step_boundary_campaign.py":"6a4e3f98131bcb8f3b6a379a4eda1680cba314e096013194935f7697a6bdf553",
    "flux_step_boundary_reference.py":"f890a62248f0ce8a1f30d29fe3db421326a196640c0fd578a8be7960dd315768",
    "flux_transition_reference.py":"f531e059635fe9940d55c38eaf5cc1d7ab9dd3a4cd1adeee5225ba9342f3bf83",
    "flux_mlp_replacement.py":"5ee9ba9d67c44e947bb6983a8bdec7b1e9335aaf28f0ffdd8ce71ab559767f8a",
    "flux_single_contexts.py":"3e1530216b56ef1609678b42e841730fc03b6e1deab1f197873e1e221c095bdf",
    "flux_attention_replacement_campaign.py":"f719dfc335bc6c588fbcbe3ef209aff2b2111037e95079e09daca92a601cf5a3"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def prior():
    for name,digest in LEGACY_SOURCES.items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("immutable predecessor source differs: "+name)
    return runpy.run_path(str(Path(__file__).with_name("flux_reentry_campaign.py")))


def step():
    return prior()["_step"]()


def util():
    return prior()["_util"]()


def contract(config=None):
    if config is not None and config.get("block_indices",BLOCK_INDICES) != BLOCK_INDICES:
        raise ValueError("joint-chain requires the declared14-block remaining step")
    return json.loads(json.dumps({"context":CONTEXT,"block_indices":BLOCK_INDICES,"branches":BRANCHES,
        "order":ORDER,"counts":COUNTS,"prefix_steps":68,"reference_steps":19,"bridge_steps":1,"suffix_steps":25,
        "panel_counts":{"adapter_steps":120,"acts":3,"receipt_emissions":123,"candidate_calls":0,"images":4,"vae_calls":4},
        "target_calls":{**{str(index):5 for index in BLOCK_INDICES},"norm_out":5,"proj_out":5,
            **{name:1 for name in PROJECT_TARGETS},"pos_embed":2,"joint0":1,"joint1":2},"observation_plan":OBSERVATION_PLAN}))


def validate_config(config):
    fixed = {"context":CONTEXT,"block_indices":BLOCK_INDICES,"branches":BRANCHES,"branch_execution_order":ORDER,
        "selector":{"logical_step":2,"phase":"single","index":6},"height":512,"width":512,"steps":4,
        "sample_indices":SAMPLE_INDICES,"image_sample_indices":IMAGE_SAMPLE_INDICES,"query_chunk":64,
        "projection_chunk":256,"neuron_chunk":256,"retain_fields":False,"array_storage":"bf16-bits",
        "archive_integer_ids":"native-int64","cpu_threads":4,"transaction_max_bytes":768<<20,
        "transport_method":"planned-authenticated-ssh-manifest",
        "transport_limits":{"max_expanded_bytes":8<<30,"max_member_bytes":1<<30},"observation_plan":OBSERVATION_PLAN,
        "expected_closure_module_sha256":OLD_CLOSURE_SHA256}
    if any(config.get(key) != value for key,value in fixed.items()):
        raise ValueError("joint-chain frozen scientific/closure geometry differs")
    for key in ("adapter_steps","acts","candidate_calls","images","vae_calls"):
        if type(config.get("max_"+key)) is not int or config["max_"+key] != COUNTS[key]:
            raise ValueError("joint-chain finite physical census differs")
    digest = config.get("expected_joint_closure_module_sha256")
    if not isinstance(digest,str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("joint-chain requires an explicit new joint closure module pin")
    if type(config.get("cpu_threads")) is not int or type(config.get("max_wall_seconds")) is not int or not 1<=config["max_wall_seconds"]<=900:
        raise ValueError("joint-chain finite execution budget differs")


def verify_source_abi(*,boundary,reentry,closure_type,expected_joint_closure_module_sha256):
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb.flux_forecast_closure import FluxForecastClosureAdapter

    old = prior()["verify_source_abi"](boundary=boundary,reentry=reentry,closure_type=FluxForecastClosureAdapter,
        expected_closure_module_sha256=OLD_CLOSURE_SHA256)
    if (not issubclass(closure_type,Flux2KleinAdapter) or closure_type.advance is not Flux2KleinAdapter.advance
            or sha(inspect.getfile(closure_type)) != expected_joint_closure_module_sha256):
        raise ValueError("joint-chain native/base/new closure source ABI differs")
    return {"schema":"mdb-flux-joint-chain-native-source-abi-v1","verified":True,"prior_native_and_closure":old,
        "joint_closure_module_sha256":expected_joint_closure_module_sha256,
        "joint_closure_type":closure_type.__module__+"."+closure_type.__qualname__,"native_advance_inherited":True,
        "scope":"Exact base2004, old closure45dd, project/joint numerical source pins and distinct newjoint1/2 surface. No CUDA kernel independence claim."}


def receipt(output,store,counts,branch,value):
    body = value.to_dict()
    counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps",1)
    counts["receipt_emissions"] += 1
    if counts["adapter_steps"] > COUNTS["adapter_steps"] or counts["acts"] > COUNTS["acts"]:
        raise RuntimeError("joint-chain finite physical budget exhausted")
    store.receipt(value)
    with (output/"receipt-emissions.jsonl").open("a") as stream:
        stream.write(json.dumps({"sequence":counts["receipt_emissions"]-1,"branch":branch,"receipt":value.fingerprint},sort_keys=True)+"\n")


def run_prefix(*,adapter,session,output,store,counts,wall):
    from mdb.native_navigation import NativeCutNavigator

    initial = session.capture()
    if initial.execution_point["logical_step"] != 0 or initial.execution_point["phase"] != "project":
        raise ValueError("joint-chain requires a clean newly encoded/seeded native initial state")
    store.save(initial)
    nav = NativeCutNavigator(session,max_steps=68,on_receipt=lambda value:receipt(output,store,counts,"prefix",value))
    path = []
    for _ in range(68):
        wall()
        path.append(nav.step().to_dict())
    pre6 = nav.capture()
    if any(pre6.execution_point[key] != value for key,value in {"logical_step":2,"phase":"single","index":6,"local_clock":6}.items()):
        raise ValueError("joint-chain prefix did not reach step2/pre6")
    store.save(pre6)
    record = {"schema":"mdb-flux-joint-chain-prefix-v1","context":CONTEXT,"initial":initial.fingerprint,
        "pre6":pre6.fingerprint,"point":dict(pre6.execution_point),"receipts":path,
        "counts":{"adapter_steps":68,"acts":0,"receipt_emissions":68},"scope":"One clean seed101 history; no imported/engineered state."}
    util()["save"](output/"prefix.json",record)
    return pre6,record


def bind_inputs(*,model,pre6,array,counts,prefix_ref,config,production_profile):
    import torch

    from mdb.forward_recording import tensor_descriptor

    binding = prior()["bind_inputs"](model=model,pre6=pre6,array=array,counts=counts,prefix_ref=prefix_ref,
        config=config,production_profile=production_profile)
    binding["schema"] = "mdb-flux-joint-chain-binding-v1"
    binding["target_entries_before_seal"] = {key:0 for key in contract()["target_calls"]}
    joint = model.transformer_blocks[1]
    if (any(getattr(joint,name).elementwise_affine or getattr(joint,name).eps != 1e-6
            for name in ("norm1","norm1_context","norm2","norm2_context"))
            or len({joint.attn.norm_q.eps,joint.attn.norm_k.eps,joint.attn.norm_added_q.eps,joint.attn.norm_added_k.eps}) != 1):
        raise ValueError("joint1 actual normalization profile differs")
    parameters = dict(joint.named_parameters())
    if any(value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()) for value in parameters.values()):
        raise ValueError("joint1 parameters must be finite native BF16")
    binding["joint1"] = {"parameters":{name:array(value) for name,value in parameters.items()},
        "native_parameters":{name:tensor_descriptor(value) for name,value in parameters.items()},
        "native_parameter_prefix":"transformer_blocks.1.","block_index":1,"heads":joint.attn.heads,
        "head_dim":joint.attn.inner_dim//joint.attn.heads,"layernorm_eps":joint.norm1.eps,"qk_eps":joint.attn.norm_q.eps,
        "scope":"Distinct actual joint1 parameters; serial per-stream attention residual followed by receiving MLP."}
    binding["joint0"]["block_index"] = 0
    binding["joint0"]["native_parameter_prefix"] = "transformer_blocks.0."
    return binding


def predict_chain(binding,*,archive_root,transition,boundary,reentry,config,event):
    from mdb.tensor_archive import load_tensor_archive

    prefix = prior()["predict_reentry"](binding,archive_root=archive_root,transition=transition,
        boundary=boundary,reentry=reentry,config=config,event=event)
    spec = binding["joint1"]
    parameters = {name:load_tensor_archive(archive_root,value) for name,value in spec["parameters"].items()}
    joint1 = reentry["predict_joint"](prefix["joint0"]["image"],prefix["joint0"]["text"],prefix["project"]["context"],parameters,
        heads=spec["heads"],layernorm_eps=spec["layernorm_eps"],qk_eps=spec["qk_eps"],
        query_chunk=config["query_chunk"],projection_chunk=config["projection_chunk"],retain_fields=False)
    del parameters
    gc.collect()
    event("joint1_forecast_complete")
    return {"prefix":prefix,"joint1":joint1,"joint1_input_source":"own complete predicted joint0 I/T and own predicted project context",
        "scope":"Nineteen-stage14single/readout/Euler/project/joint0/joint1 forecast; no native intermediate, context or Z3 refresh."}


def validate_prediction(pre6,prediction):
    prior()["validate_prediction"](pre6,prediction["prefix"])
    normal,descriptor = util()["json_value"],step()["_descriptor"]
    previous,current = prediction["prefix"]["joint0"],prediction["joint1"]
    if (prediction["joint1_input_source"] != "own complete predicted joint0 I/T and own predicted project context"
            or normal(current["input_image_descriptor"]) != normal(previous["image_descriptor"])
            or normal(current["input_text_descriptor"]) != normal(previous["text_descriptor"])
            or normal(current["image_descriptor"]) != descriptor(current["image"])
            or normal(current["text_descriptor"]) != descriptor(current["text"])):
        raise ValueError("joint1 complete own-state input/output join differs")


def run_panel(*,adapter,pre6,output,config,store,array,counts,wall,event,prediction):
    import numpy as np
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb.flux_joint_forecast_closure import closure_act
    from mdb.native_navigation import NativeCutNavigator

    helpers = util()
    def navigator(cut,branch,limit):
        return NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=limit,
            on_receipt=lambda value:receipt(output,store,counts,branch,value))
    def archived(value):
        if isinstance(value,torch.Tensor):
            return array(value)
        if isinstance(value,dict):
            return {key:archived(child) for key,child in value.items()}
        if isinstance(value,(tuple,list)):
            return [archived(child) for child in value]
        return value
    def boundary(cut):
        return {"cut":cut.fingerprint,"point":dict(cut.execution_point),
            "closure":{key:archived(cut.payload[key]) for key in ("latent","context","image","text","noise_prediction")},
            "clock":{key:cut.payload[key] for key in ("step","phase","cursor")}}
    nav = navigator(pre6,"reference",19)
    references,paths,cuts = [],[],[]
    for position in range(19):
        wall()
        entry = nav.step()
        current = nav.capture()
        store.save(current)
        references.append({"index":position,"receipt_id":entry.fingerprint,**boundary(current)})
        paths.append(entry.to_dict())
        cuts.append(current)
        event("reference_transition_complete",index=position,point=dict(current.execution_point))
    native_joint0,native_joint1 = cuts[17:19]
    for index,cut in ((1,native_joint0),(2,native_joint1)):
        if any(cut.execution_point[key] != value for key,value in {"logical_step":3,"phase":"joint","index":index,"local_clock":index}.items()):
            raise ValueError("native joint-chain clock scaffold differs")
    helpers["save"](output/"native-segment.json",{"input_cut":pre6.fingerprint,"boundaries":references,"receipts":paths,
        "native_Z3":references[15],"native_project":references[16],
        "native_post_joint0":boundary(native_joint0),"common_post_joint1":boundary(native_joint1),
        "counts":{"adapter_steps":19,"acts":0,"receipt_emissions":19}})
    validate_prediction(pre6,prediction)
    del cuts
    observer_class = runpy.run_path(str(Path(__file__).with_name("flux_attention_replacement_campaign.py")))["ContinuationObserver"]
    observer = observer_class(array=array,plan=config["observation_plan"])
    def tensor(value,reference):
        if not isinstance(value,np.ndarray) or value.dtype != np.float32 or tuple(value.shape) != tuple(reference.shape) or not np.isfinite(value).all():
            raise ValueError("complete joint-chain payload geometry/finite/type differs")
        if reference.dtype == torch.bfloat16:
            step()["_descriptor"](value)
        converted = torch.from_numpy(np.array(value,copy=True)).to(device=reference.device,dtype=reference.dtype)
        result = torch.empty_strided(reference.shape,reference.stride(),device=reference.device,dtype=reference.dtype)
        result.copy_(converted)
        if not np.array_equal(result.float().cpu().numpy().view(np.uint32),value.view(np.uint32)):
            raise ValueError("joint-chain payload cast would alter sealed values")
        return result
    def context(value,reference):
        if isinstance(reference,torch.Tensor):
            return tensor(value,reference)
        if isinstance(reference,dict) and set(value) == set(reference):
            return {key:context(value[key],reference[key]) for key in reference}
        if isinstance(reference,(tuple,list)) and len(value) == len(reference):
            return tuple(context(left,right) for left,right in zip(value,reference,strict=True))
        raise ValueError("joint-chain complete context structure differs")
    rows = []
    for spec in config["branches"]:
        name,mode = spec["id"],spec["mode"]
        parent = native_joint0 if mode == "inherited" else native_joint1
        nav = navigator(parent,name,26)
        row = {**spec,"parent":parent.fingerprint,"acts":[],"bridge_receipts":[],"bridge_boundaries":[],"bridge_records":[]}
        if mode != "native":
            before = nav.capture()
            if mode == "sham":
                payload = {key:before.payload[key] for key in ("latent","context","image","text")}
            else:
                prefix = prediction["prefix"]
                stage = prefix["joint0"] if mode == "inherited" else prediction["joint1"]
                payload = {"latent":tensor(prefix["remaining_step"]["commit"]["latent"],before.payload["latent"]),
                    "context":context(prefix["project"]["context"],before.payload["context"]),
                    "image":tensor(stage["image"],before.payload["image"]),"text":tensor(stage["text"],before.payload["text"])}
            store.save(before)
            act = closure_act(before,**payload,name=name+".complete-closure")
            entry = nav.apply(act)
            installed = nav.capture()
            store.save(installed)
            if describe(installed.payload) != describe({**before.payload,**payload}):
                raise ValueError("joint-chain admission changed undeclared state or sealed values")
            row["acts"].append({"slots":["latent","context","image","text"],"parent":before.fingerprint,
                "post_act":installed.fingerprint,"receipt":entry.to_dict(),"receipt_id":entry.fingerprint,
                "act":act.manifest(),"undeclared_slots_exact":True})
            row["installed"] = boundary(installed)
        if mode == "inherited":
            wall()
            bridge_before = nav.capture()
            entry = nav.step()
            bridged = nav.capture()
            store.save(bridged)
            row["bridge_receipts"].append(entry.to_dict())
            row["bridge_boundaries"].append(boundary(bridged))
            row["bridge_records"].append({"before":boundary(bridge_before),"after":boundary(bridged),
                "receipt_id":entry.fingerprint,"receipt":entry.to_dict()})
        common = nav.capture()
        if any(common.execution_point[key] != value for key,value in {"logical_step":3,"phase":"joint","index":2,"local_clock":2}.items()):
            raise ValueError("all joint-chain arms must meet at actual post-joint1")
        store.save(common)
        row["common_post_joint1"] = boundary(common)
        incoming = (row["bridge_receipts"][-1]["fingerprint"] if row["bridge_receipts"] else
            row["acts"][-1]["receipt_id"] if row["acts"] else paths[-1]["fingerprint"])
        def observe(current,index,incoming):
            value = observer.observe(name,current,index=index,incoming_receipt=incoming)
            value["complete_cut_retained"] = index in {0,25}
            for key,slot in value["slots"].items():
                if key in {"image","text"}:
                    active = current.execution_point["phase"] in {"joint","single","readout"}
                    slot["meaning"] = {"phase":"working-joint-or-single-stream" if active else "absent-or-stale-working-stream",
                        "meaningful_propagation_observable":active and slot["present"],"stale":not active}
            return value
        observe(common,0,incoming)
        path = []
        for position in range(25):
            wall()
            entry = nav.step()
            path.append(entry.to_dict())
            observe(nav.capture(),position+1,entry.fingerprint)
        final = nav.capture()
        if final.execution_point["logical_step"] != 4 or final.execution_point["phase"] != "project":
            raise ValueError("joint-chain suffix did not finish at committed Z4")
        store.save(final)
        counts["vae_calls"] += 1
        image = adapter.decode(nav.session,height=512,width=512)
        image_path = output/(name+".png")
        image.save(image_path)
        counts["images"] += 1
        row.update(final_cut=final.fingerprint,final_latent=array(final.payload["latent"]),
            image={"path":image_path.name,"sha256":sha(image_path)},continuation_receipts=path)
        helpers["save"](output/"rows"/(name+".json"),row)
        rows.append(row)
        event("branch_complete",row=name)
    if set(observer.rows) != set(ORDER) or any(len(value) != 26 for value in observer.rows.values()):
        raise ValueError("joint-chain continuation observation census differs")
    helpers["save"](output/"continuation-observations.json",{"schema":"mdb-flux-joint-chain-continuation-v1",
        "plan":config["observation_plan"],"rows":observer.rows,"branch_execution_order":ORDER,"model_calls_added":0})
    controls = {"sham_final_latent_exact_native":rows[0]["final_latent"] == rows[1]["final_latent"],
        "sham_png_exact_native":rows[0]["image"]["sha256"] == rows[1]["image"]["sha256"]}
    return {"rows":rows,"counts":dict(counts),"control_observations":controls,
        "native_segment_sha256":sha(output/"native-segment.json"),
        "continuation_observations_sha256":sha(output/"continuation-observations.json")}


def run(*,model,adapter,session,output,config,request,source,counts,array=None,guard_components=None,
        wall=lambda:None,event=lambda *args,**kwargs:None,started=None,production_profile=True):
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    started = time.monotonic() if started is None else started
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    helpers = util()
    save,identity = helpers["save"],helpers["identity"]
    if (output/"plan.json").exists():
        raise ValueError("immutable joint-chain output already contains a plan")
    array = array or step()["make_array_writer"](output,wall)
    save(output/"plan.json",config)
    save(output/"plan-frozen.json",{"plan_sha256":sha(output/"plan.json"),"job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_identity":identity(request["source_hashes"]),"counts":dict(counts),
        "scope":"Frozen before clean68transitionprefix; all19 own-state stages seal before native selected targets."})
    boundary = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    abi = verify_source_abi(boundary=boundary,reentry=reentry,closure_type=type(adapter),
        expected_joint_closure_module_sha256=config["expected_joint_closure_module_sha256"])
    if adapter.execution.get("adapter") != "mdb-flux-joint-forecast-closure-v1":
        raise ValueError("joint-chain requires the distinct joint1/2 closure execution contract")
    save(output/"native-source-abi.json",abi)
    store = LocalStore(output/"typed-store")
    pre6,prefix = run_prefix(adapter=adapter,session=session,output=output,store=store,counts=counts,wall=wall)
    prefix_ref = {"path":"prefix.json","sha256":sha(output/"prefix.json")}
    source = {**source,"prefix":prefix_ref,"native_source_abi":{"path":"native-source-abi.json","sha256":sha(output/"native-source-abi.json")}}
    save(output/"source.json",source)
    transition = runpy.run_path(str(Path(__file__).with_name("flux_transition_reference.py")))
    released = {"native":False,"blocked":0,"targets":{key:0 for key in contract()["target_calls"]}}
    targets = {model.single_transformer_blocks[index]:str(index) for index in BLOCK_INDICES}
    targets.update({model.norm_out:"norm_out",model.proj_out:"proj_out",model.pos_embed:"pos_embed",
        model.transformer_blocks[0]:"joint0",model.transformer_blocks[1]:"joint1",
        **{getattr(model,name):name for name in PROJECT_TARGETS}})
    def guard(module,args):
        if not released["native"]:
            released["blocked"] += 1
            raise ValueError("native component entered before whole joint-chain forecast seal")
        if module in targets:
            released["targets"][targets[module]] += 1
    handles = [module.register_forward_pre_hook(guard) for component in (guard_components or (model,)) for module in component.modules()]
    try:
        binding = bind_inputs(model=model,pre6=pre6,array=array,counts=counts,prefix_ref=prefix_ref,
            config=config,production_profile=production_profile)
        save(output/"binding.json",binding)
        tx = ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,
            input_identity=identity(binding),source_identity=identity(request["source_hashes"]),
            frontier={"kind":"architectural","binding":binding},execution_scope=config["scope"],
            budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000,"max_bytes":config["transaction_max_bytes"]})
        with tx:
            tx.frontier("source_binding",binding)
            event("prediction_start")
            tx.predict("joint_chain",lambda _:{"predictions":predict_chain(binding,archive_root=output,
                transition=transition,boundary=boundary,reentry=reentry,config=config,event=event)})
            tx.predict("hypotheses",lambda _:{"predictions":config["hypotheses"]})
            seal = tx.seal()
            event("joint_chain_sealed",seal=seal,target_entries_before_seal=dict(released["targets"]),blocked_entries=released["blocked"])
            released["native"] = True
            tx.execute("native",lambda context:run_panel(adapter=adapter,pre6=pre6,output=output,config=config,
                store=store,array=array,counts=counts,wall=wall,event=event,prediction=context.predictions["joint_chain"]))
    finally:
        for handle in handles:
            handle.remove()
    save(output/"transaction-report.json",tx.report())
    if (any(counts[name] != COUNTS[name] for name in ("adapter_steps","acts","receipt_emissions","images","vae_calls","candidate_calls"))
            or released["blocked"] != 0 or released["targets"] != contract()["target_calls"]):
        raise ValueError("joint-chain completed physical/target census differs")
    summary = {"schema":"mdb-flux-joint-chain-result-v1","job_id":request["job_id"],"request_sha256":request["request_sha256"],
        "counts":dict(counts),"prefix_counts":prefix["counts"],"panel_counts":contract()["panel_counts"],
        "target_calls":released["targets"],"blocked_entries":released["blocked"],"seal":seal,
        "model_identity":adapter.model_identity,"execution":dict(adapter.execution),"transaction_verification":verify_transaction(tx.root),
        "control_observations":tx.observation("native")["control_observations"],"worker_seconds":time.monotonic()-started,
        "mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary)
    return helpers["json_value"](summary)


def worker():
    """One admitted CUDA process; SSH custody is performed by the launcher."""
    import faulthandler

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90,repeat=True)
    helpers = util()
    save,sha,identity,json_value = (helpers[name] for name in ("save","sha","identity","json_value"))
    config,request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if (request["config"] != config or any(sha(path) != digest for path,digest in request["source_hashes"].items())
            or config.get("transport_method") != "planned-authenticated-ssh-manifest"):
        raise ValueError("joint-chain request/source/transport identity differs")
    request = {**request,"job_id":os.environ["MRUN_JOB_ID"]}
    output = Path("flux-joint-chain-result")
    output.mkdir(exist_ok=False)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print("MDB_FLUX_JOINT_CHAIN_BOOTSTRAP "+json.dumps({"job_id":request["job_id"],"seconds":time.monotonic()-started}),flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb.flux_forecast_closure import FluxForecastClosureAdapter
    from mdb.flux_joint_forecast_closure import FluxJointForecastClosureAdapter

    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("joint-chain requires admitted CUDA")
    reference = runpy.run_path(str(Path(__file__).with_name("flux_step_boundary_reference.py")))
    reentry = runpy.run_path(str(Path(__file__).with_name("flux_reentry_reference.py")))
    source_abi = verify_source_abi(boundary=reference,reentry=reentry,closure_type=FluxJointForecastClosureAdapter,
        expected_joint_closure_module_sha256=config["expected_joint_closure_module_sha256"])
    save(output / "pre-model-source-abi.json",source_abi)
    scalar_qualification = reference["qualify_torch_scalar_promotion"](torch,device="cuda")
    save(output / "cuda-scalar-qualification.json",scalar_qualification)
    disk = shutil.disk_usage(output)
    if disk.free < config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("joint-chain output filesystem lacks declared8GiB free disk")
    save(output / "disk-admission.json",{"output_filesystem":str(output.resolve()),"free_bytes":disk.free,
        "total_bytes":disk.total,"max_output_bytes":config["transport_limits"]["max_expanded_bytes"],
        "scope":"Pre-model free-space observation and finite output bound; no quota or guaranteed future free space."})
    counts = {key:0 for key in COUNTS}
    counts["physical_batch"] = 1
    heartbeat = {"last":time.monotonic(),"sequence":0}
    def event(kind,**fields):
        body = json_value({"sequence":heartbeat["sequence"],"kind":kind,"seconds":time.monotonic()-started,
            "counts":dict(counts),**fields})
        heartbeat["sequence"] += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body,sort_keys=True,allow_nan=False)+"\n")
        print("MDB_FLUX_JOINT_CHAIN_PHASE "+json.dumps(body,sort_keys=True),flush=True)
    def wall():
        now = time.monotonic()
        if now-started > config["max_wall_seconds"]:
            raise RuntimeError("finite joint-chain wall budget exhausted")
        if now-heartbeat["last"] >= 15:
            heartbeat["last"] = now
            event("heartbeat")
    array = step()["make_array_writer"](output,wall)
    root = Path(config["model_path"])
    inventory = {path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("joint-chain checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("joint-chain requires deterministic distilled five-joint/twenty-single Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind):sha(inspect.getfile(kind)) for kind in (type(pipe),type(pipe.text_encoder),
        type(pipe.transformer),type(pipe.transformer.single_transformer_blocks[6]),
        type(pipe.transformer.single_transformer_blocks[6].attn.processor),type(pipe.transformer.norm_out),
        type(pipe.vae),type(pipe.scheduler),Flux2KleinAdapter,FluxForecastClosureAdapter,FluxJointForecastClosureAdapter,
        type(pipe.transformer.transformer_blocks[0]),type(pipe.transformer.pos_embed),
        type(pipe.transformer.time_guidance_embed.timestep_embedder))}
    pipe.text_encoder.to("cuda")
    with torch.inference_mode():
        conditioning,txt_ids = pipe.encode_prompt(CONTEXT["prompt"],device=torch.device("cuda"),
            num_images_per_prompt=1,max_sequence_length=512)
    counts["prompt_encodings"] += 1
    encoding = {"conditioning":array(conditioning),"txt_ids":array(txt_ids)}
    conditioning,txt_ids = conditioning.cpu(),txt_ids.cpu()
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    event("encoder_released")
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = FluxJointForecastClosureAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    with torch.inference_mode():
        conditioning,txt_ids = conditioning.to("cuda"),txt_ids.to("cuda")
        latent,img_ids = pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,conditioning.dtype,
            torch.device("cuda"),torch.Generator(device="cuda").manual_seed(CONTEXT["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config,"use_flow_sigmas",False) else np.linspace(1.,.25,4)
        timesteps,steps = retrieve_timesteps(pipe.scheduler,4,torch.device("cuda"),sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
        if steps != 4:
            raise ValueError("joint-chain native schedule differs")
        session = adapter.session(latent=latent,conditioning=conditioning,img_ids=img_ids,txt_ids=txt_ids,
            timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
    source = {"schema":"mdb-flux-joint-chain-source-v1","job_id":request["job_id"],
        "request_sha256":request["request_sha256"],"source_hashes":request["source_hashes"],"context":CONTEXT,
        "encoding":encoding,"encodings":encoding,"model_identity":adapter.model_identity,"execution":dict(adapter.execution),
        "checkpoint_sha256":identity(inventory),"checkpoint_files":inventory,"implementation_sources":implementations,
        "versions":{name:importlib.metadata.version(name) for name in ("torch","numpy","diffusers","transformers","saturn-pub","mdb-runtime")},
        "counts_before_prefix":dict(counts),"cuda_scalar_qualification":{"path":"cuda-scalar-qualification.json",
            "sha256":sha(output / "cuda-scalar-qualification.json")},
        "pre_model_source_abi":{"path":"pre-model-source-abi.json","sha256":sha(output / "pre-model-source-abi.json")},
        "scope":config["scope"]}
    summary = run(model=pipe.transformer,adapter=adapter,session=session,output=output,config=config,request=request,
        source=source,counts=counts,array=array,guard_components=(pipe.transformer,pipe.vae),wall=wall,event=event,started=started)
    if counts != COUNTS:
        raise ValueError("joint-chain global load/encoding/noise census differs")
    event("complete")
    faulthandler.cancel_dump_traceback_later()
    manifest = step()["finalize_manifest"](output,job_id=request["job_id"],request_sha256=request["request_sha256"])
    print("MDB_FLUX_JOINT_CHAIN_COMPLETE "+json.dumps(json_value({"summary":summary,"manifest":manifest}),sort_keys=True),flush=True)
    return output,summary


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-joint-chain-result")
        if output.is_dir() and not (output/"remote-manifest.json").exists():
            util()["save"](output/"failure.json",{"schema":"mdb-flux-joint-chain-failure-v1",
                "job_id":os.environ.get("MRUN_JOB_ID"),"error":f"{type(error).__name__}: {error}",
                "mechanics_status":"errored","scope":"Completed evidence preserved; no model retry, archive or fabricated report."})
        raise
