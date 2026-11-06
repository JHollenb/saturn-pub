"""Framework-free, saved-only audit of four same-noise prompt conditions.

Rebuild every predicted field from actual origins and named checkpoint weights.
Native full-field comparisons are restricted to the declared landmarks; other
native comparisons use fixed-coordinate witnesses, not invented full arrays.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import runpy
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RESULT = "flux-prompt-factorial-result"
CELLS = ["c0-o0","c0-o1","c1-o0","c1-o1"]
LANDMARKS = [0,5,25,26,27]
COUNTS = {"model_loads":1,"prompt_encodings":4,"noise_draws":1,"adapter_steps":448,"acts":5,
    "candidate_calls":0,"receipt_emissions":453,"images":9,"vae_calls":9,"physical_batch":1}
CHECKPOINT_FILE = "transformer/diffusion_pytorch_model.safetensors"
CHECKPOINT_SHA256 = "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e"
PARAMETER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
INDEPENDENT_SOURCES = {
    "analyze_flux_attention_replacement.py":"e6a96ebd1939eea2979a483f855356e96656c5ef07d651c24e9d5035028fc1fd",
    "analyze_flux_concept_routes.py":"0fb3e0cb5e23bb98d9780af1b7c591bb1cf291db516fd967ad0951b0f0fd4bfe",
    "analyze_flux_final_latent.py":"7cea0fab5cd20ebc0610d95572cebf4d2e8bc4d3eda4392c322a99edd3ca9e0e",
    "analyze_flux_initial_noise.py":"93760e398fe844ca0c2000a902891ee8fd98a0ab9a7496a7f5654202ce04cb45",
    "analyze_flux_joint_chain.py":"641c847ffb3d9a3e14ae2402acf7b0257c4dcf8d6ac8d6319b885da704f05943",
    "analyze_flux_joint_tail.py":"20fb4cb18b4749ceb6ba47b00dfbdeae41944b5740fedc3605d83b9f6ddcedac",
    "analyze_flux_mlp_forecast.py":"fb5c11feff809f23f65b6c89dda73a589e2a46d11cede60aaf4e04b60d349252",
    "analyze_flux_mlp_replacement.py":"e43ac1568a85d0362b47c70a1460f8061d76fd4f6b6d7db0b815e5c51216b79c",
    "analyze_flux_reentry.py":"b4cdcc32ecbcfd18b8093b831f48ca2a7be664b73e161076d1bc9e4c1d9c8176",
    "analyze_flux_rescue_routes.py":"169f25a06833d559516d61e564d4fc70883e619cd909cd04026b0b13b293c0b9",
    "analyze_flux_single_contexts.py":"e90999990cadf3553a1d4b44bf9633cae579870444a8abe6c91e8ebf45a6f9b1",
    "analyze_flux_single_contraction.py":"57d313d7182a86ef37bf84fadf66da030b0198c382454076327ab826725bda7d",
    "analyze_flux_single_kv.py":"85418afbb0a652c5a41eb2dbc986261c46cd3ecd0fb24f46857ebb61dac76262",
    "analyze_flux_single_mlp.py":"ef5e19cc3eac2e0b2d50336ccb7844746fceac8194c8f9a6f9cba65824269dbd",
    "analyze_flux_step_boundary.py":"2fa799c9bae9fe7a08a3da727a617abd4b2de98eb9b0114abd82526482ecd93d",
    "analyze_flux_transition_chain.py":"2a0e7cf96381e3dc22a4c93f99656ab07f2554ec92d08a3196ad973cd1afd38a",
    "analyze_flux_two_pass.py":"499a25d18cbbdf6e7a6189159a44070e51057d2e24cb651fc37fe93e30631e81",
    "collect_flux_final_latent.py":"f1d07d347af933118390fa361a8b9dd5f7df1bddff2fcc686548bbf80f135427",
    "collect_flux_initial_noise.py":"ce3930b71b58dab832c09b7d4b0c4e3eefb8bc6435a72e4605992d415be38765",
    "collect_flux_joint_chain.py":"0b6e35c4799d22d6a9f2918bb0914835765564c19b4c0dabc2105f6425a23aea",
    "collect_flux_joint_tail.py":"2b2ff6581f4dc371ff13eae827d0aa9f066c3453ad7808df97c96099b549b1b5",
    "collect_flux_reentry.py":"f3743616b539d325a8a2450dc9ee53ee1898eb9c644dfe097455689dc34208ed",
    "collect_flux_step_boundary.py":"ba5014110dda4882c8652562f850ba31562a4dfe9f6834b8dbe362f4d9491edc",
    "collect_flux_two_pass.py":"24067973abd8426a3888d5a6f874e2e720e0acf6a00fae0a686124a21c83beec",
    "flux_family_custody.py":"e13c79dcb7e41d2583407f8adf2a5e3dae28cdbe6a51342fad1c78ee009ff12a",
    "flux_reentry_audit_math.py":"af4f1781836caabec387c7088538158ee5f285b45c46c1367f8a7f083cb4c081",
    "flux_step_boundary_audit_math.py":"7ded49c5e716ab6bedea05a725dbc3e373fc69543652eb06c2c5844a3f9ed713",
    "summarize_context_window.py":"691420937df6d5996defbfb9630e3a58c222ebecfdd82e5bac667220cb3405ae"}


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4<<20),b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    from collections.abc import Mapping
    if isinstance(value,Mapping):
        return {key:plain(child) for key,child in value.items()}
    if isinstance(value,(tuple,list)):
        return [plain(child) for child in value]
    return value


@lru_cache(maxsize=1)
def independent():
    for name,digest in INDEPENDENT_SOURCES.items():
        require(sha(HERE/name) == digest,"qualified independent source differs: "+name)
    return runpy.run_path(str(HERE/"analyze_flux_initial_noise.py"))


def describe_field(value):
    value = np.ascontiguousarray(value)
    require(value.dtype == np.float32 and value.size and np.isfinite(value).all(),"full field ABI differs")
    result = {"shape":list(value.shape),"dtype":"float32",
        "fp32_le_sha256":hashlib.sha256(value.astype("<f4",copy=False).tobytes()).hexdigest()}
    if not np.any(value.view(np.uint32)&np.uint32(65535)):
        result["bf16_le_sha256"] = hashlib.sha256((value.view(np.uint32)>>16).astype("<u2").tobytes()).hexdigest()
    return result


def compare(predicted,native):
    """Explicit P-N with native/reference RMS; no tolerance/quality gate."""
    p,n = np.asarray(predicted),np.asarray(native)
    require(p.shape == n.shape and p.size and np.isfinite(p).all() and np.isfinite(n).all(),"comparison operands differ")
    delta = p.astype(np.float64)-n.astype(np.float64)
    energy,reference = float(np.square(delta).sum()),float(np.square(n.astype(np.float64)).sum())
    predicted_energy = float(np.square(p.astype(np.float64)).sum())
    dot = float(np.sum(p.astype(np.float64)*n.astype(np.float64)))
    magnitude = np.abs(delta)
    return {"shape":list(p.shape),"elements":int(p.size),"l2":float(np.sqrt(energy)),"rms":float(np.sqrt(energy/p.size)),
        "mae":float(magnitude.mean()),"signed_mean":float(delta.mean()),"max_abs":float(magnitude.max()),
        "changed_value_count":int(np.count_nonzero(p != n)),"absolute_error_p95":float(np.quantile(magnitude,.95)),
        "absolute_error_p99":float(np.quantile(magnitude,.99)),"signed_dot":dot,
        "alignment_cosine":dot/float(np.sqrt(predicted_energy*reference)) if predicted_energy and reference else None,
        "squared_l2":energy,"reference_rms":float(np.sqrt(reference/p.size)),
        "relative_rms":float(np.sqrt(energy/reference)) if reference else None,
        "sign":"prediction minus native/reference","relative_rms_denominator":"native/reference RMS; zero denominator undefined"}


def moments(value):
    value = np.asarray(value,dtype=np.float64)
    return {"elements":int(value.size),"rms":float(np.sqrt(np.mean(value*value))),"mean":float(value.mean()),
        "min":float(value.min()),"max":float(value.max())}


def witness_indices(shape):
    size = int(np.prod(shape,dtype=np.int64))
    require(size > 0,"empty witness field")
    return np.unique(np.linspace(0,size-1,min(64,size),dtype=np.int64)).tolist()


def check_field(value,record,*,e=None,exact_required=False):
    require(record["descriptor"]["shape"] == list(value.shape) and record["descriptor"]["dtype"] == "float32",
        "reconstructed/declared field geometry differs")
    require(record["witness"]["flat_indices"] == witness_indices(value.shape),"adaptive/nonfixed witness coordinates")
    indexes = record["witness"]["flat_indices"]
    require(len(record["witness"]["values"]) == len(indexes),"witness count differs")
    same = describe_field(value) == record["descriptor"]
    if exact_required:
        require(same and moments(value) == record["moments"] and np.array_equal(
            value.reshape(-1)[indexes],record["witness"]["values"]),"retained full field identity/moments/witness differs")
    if record["archive"] is not None:
        require(e is not None,"archive reader absent")
        archived = e.array(record["archive"])
        require(describe_field(archived) == record["descriptor"] and moments(archived) == record["moments"]
            and np.array_equal(archived.reshape(-1)[indexes],record["witness"]["values"]),"sealed/native landmark archive differs")
    return {"full_descriptor_match":same,"fixed_witness_error":compare(value.reshape(-1)[indexes],
        np.asarray(record["witness"]["values"],dtype=np.float32)),
        "full_field_error":compare(value,e.array(record["archive"])) if record["archive"] is not None else None,
        "scope":"Complete independently rebuilt field hash; full sealed values available only at declared landmarks."}


def context_fields(context):
    return {**{"context."+key:context[key] for key in ("temb","img_mod","txt_mod","single_mod")},
        "rope.cos":context["rope"][0],"rope.sin":context["rope"][1]}


def kind_at(local):
    return ("project",None) if local == 0 else ("joint",local-1) if local <= 5 else (
        "single",local-6) if local <= 25 else ("readout",None) if local == 26 else ("commit",None)


def predictions(e,binding,origin):
    """Own-state independent recurrence; no producer/kernel/model imports."""
    old = independent()
    two = old["TWO"]
    values = {key:e.array(value) for key,value in origin["inputs"].items()}
    latent,conditioning = values["latent"],values["conditioning"]
    for logical in range(4):
        schedule = {name:e.array(binding["pass_schedules"][str(logical)][name]) for name in ("timestep","sigma","sigma_next")}
        iterator = two["independent_pass_events"](latent,conditioning,values["img_ids"],values["txt_ids"],schedule,
            binding,lambda kind,spec:two["load_bank"](e,kind,spec))
        previous,current_context,image,text = None,None,None,None
        for local,(kind,index,value) in enumerate(iterator):
            if kind == "project":
                inputs = {"latent":describe_field(latent),"origin_conditioning":origin["inputs"]["conditioning"]["descriptor"],
                    "origin_txt_ids":origin["inputs"]["txt_ids"]["descriptor"],"schedule":binding["pass_schedules"][str(logical)]}
                image,text = value["image"],value["text"]
                fields = {"image":image,"text":text,**context_fields(value["context"])}
                current_context = identity({name:describe_field(field) for name,field in context_fields(value["context"]).items()})
            elif kind in {"joint","single"}:
                inputs = {"image":describe_field(image),"text":describe_field(text),"own_project_context":current_context}
                if kind == "single":
                    inputs["residual"] = describe_field(np.concatenate((text,image),axis=1))
                    text,image = value["output"][:,:conditioning.shape[1]],value["output"][:,conditioning.shape[1]:]
                else:
                    image,text = value["image"],value["text"]
                fields = {"image":image,"text":text}
            elif kind == "readout":
                inputs = {"image":describe_field(image),"own_project_context":current_context}
                fields = {"noise":value["noise"]}
            else:
                inputs = {"latent":describe_field(latent),"noise":describe_field(previous["noise"]),
                    "sigma":binding["pass_schedules"][str(logical)]["sigma"],"sigma_next":binding["pass_schedules"][str(logical)]["sigma_next"]}
                latent = value["latent"]
                fields = {"latent":latent}
            yield logical,local,kind,index,inputs,fields
            previous = fields
        del iterator,previous,image,text,value
        gc.collect()


def targets(multiplier):
    return {**{str(i):multiplier for i in range(20)},"norm_out":multiplier,"proj_out":multiplier,
        **{key:multiplier for key in ("x_embedder","context_embedder","time_guidance_embed",
            "double_stream_modulation_img","double_stream_modulation_txt","single_stream_modulation")},
        "pos_embed":2*multiplier,**{"joint"+str(i):multiplier for i in range(5)}}


def parameter_ports(binding):
    """Join recipe tensor names to their actual full qualified module ports."""
    base = independent()["BASE"]
    names = []
    def check(spec,mapping,prefix=""):
        require(set(spec["parameters"]) == set(mapping),"parameter alias inventory differs")
        for alias,native in mapping.items():
            record = spec["parameters"][alias]
            if record is not None:
                expected = prefix+native
                require(record["storage"] == "checkpoint-recipe-v1" and record["parameter_tensor"] == expected,
                    "checkpoint recipe tensor is not the named native module parameter")
                names.append(expected)
    check(binding["project"],{key:key for key in binding["project"]["parameters"]})
    for i in range(5):
        spec = binding["joint"+str(i)]
        check(spec,{key:key for key in spec["parameters"]},f"transformer_blocks.{i}.")
    for i in range(20):
        check(binding["final_blocks"][str(i)],base["PARAMS"],f"single_transformer_blocks.{i}.")
    check(binding["readout"],{"modulation_weight":"norm_out.linear.weight","modulation_bias":"norm_out.linear.bias",
        "projection_weight":"proj_out.weight","projection_bias":"proj_out.bias"})
    require(len(names) == len(set(names)),"one native parameter port was rebound twice")
    return names


def source_abi(data,source,plan):
    reference = source["pre_model_source_abi"]
    abi = read(data/reference["path"])
    two = abi["prior_two_pass"]
    final = two["prior_final_latent"]
    tail = final["prior_joint_tail"]
    chain = tail["prior_joint_chain"]
    old = chain["prior_native_and_closure"]
    require(sha(data/reference["path"]) == reference["sha256"] and abi["verified"] is True
        and abi["native_advance_inherited"] is True and two["verified"] is True
        and old["reentry_actual"] == old["reentry_expected"] and old["base_step"]["verified"] is True
        and old["base_step"]["model_calls"] == 0 and old["base_step"]["actual"] == old["base_step"]["expected"]
        and old["closure_module_sha256"] == plan["expected_closure_module_sha256"]
        and chain["joint_closure_module_sha256"] == plan["expected_joint_closure_module_sha256"]
        and tail["tail_closure_module_sha256"] == plan["expected_tail_closure_module_sha256"]
        and final["final_closure_module_sha256"] == plan["expected_final_closure_module_sha256"]
        and two["derived_terminal_module_sha256"] == plan["expected_derived_terminal_module_sha256"]
        and abi["parameter_artifacts_module_sha256"] == PARAMETER_SHA256,"actual source/closure/derivation ABI differs")
    scalar_ref = source["cuda_scalar_qualification"]
    scalar = read(data/scalar_ref["path"])
    require(sha(data/scalar_ref["path"]) == scalar_ref["sha256"] and scalar["qualified"] is True
        and scalar["device"] == "cuda" and scalar["model_calls"] == 0 and len(scalar["cases"]) == 3
        and all(case["values"] == 2048 and case["native_dtype"] == "torch.bfloat16"
            and case["native_vs_bf16_dt_mismatches"] == 0 and case["native_vs_fp32_dt_mismatches"] > 0
            for case in scalar["cases"]),"actual scalar-promotion instrument differs")
    return {"source_abi_sha256":reference["sha256"],"cuda_scalar_qualification_sha256":scalar_ref["sha256"],"verified":True}


def conditions():
    template = "A {cone} cone to the {position} of a {cylinder} cylinder on a plain white background. Two separate objects, simple studio product photograph."
    return [{"id":f"c{c}-o{o}","color_assignment":c,"cone_right":o,"seed":127,
        "prompt":template.format(cone=("green","yellow")[c],cylinder=("yellow","green")[c],position=("left","right")[o])}
        for c in range(2) for o in range(2)]


def validate_plan(plan,source,summary,*,production_profile):
    require(plan["conditions"] == source["conditions"] == summary["conditions"] == conditions()
        and plan["seed"] == 127 and plan["logical_steps"] == [0,1,2,3] and plan["block_indices"] == list(range(20))
        and plan["landmark_local_indices"] == LANDMARKS and plan["witness_count"] == 64
        and plan["parameter_storage"] == "checkpoint-recipe-v1" and plan["transaction_max_bytes"] == 64<<20,
        "factorial source/profile/coverage plan differs")
    if production_profile:
        require(summary["counts"] == COUNTS and source["execution"]["device"]["type"] == "cuda"
            and source["execution"]["dtype"] == "torch.bfloat16" and plan["cpu_threads"] == 4
            and (plan["height"],plan["width"],plan["steps"]) == (512,512,4),"actual production execution/census differs")
    else:
        require(all(summary["counts"][key] == COUNTS[key] for key in ("adapter_steps","acts","receipt_emissions","images","vae_calls")),
            "tiny physical census differs")
    require(summary["blocked_entries"] == 0 and summary["target_calls"] == targets(16)
        and summary["target_calls_by_condition"] == {key:targets(4) for key in CELLS},"native target census differs")


def verify_cohort(data,binding,summary,programs,e):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    cohort = read(data/"cohort-seal.json")
    require(cohort["fingerprint"] == identity({key:value for key,value in cohort.items() if key != "fingerprint"})
        and cohort["condition_ids"] == CELLS and cohort["origins"] == {key:value["cut"] for key,value in binding["origins"].items()}
        and cohort["seals"] == summary["seals"] and cohort["binding_sha256"] == sha(data/"binding.json")
        and cohort["plan_sha256"] == sha(data/"plan.json") and cohort["target_calls"] == targets(0)
        and all(cohort["counts"][key] == 0 for key in ("adapter_steps","acts","receipt_emissions","images","vae_calls"))
        and summary["cohort_seal_sha256"] == sha(data/"cohort-seal.json"),"cohort seal/source/prospective counters differ")
    reports = {}
    for position,key in enumerate(CELLS):
        root = data/"transactions"/key
        verification = verify_transaction(root,expected_seal_sha256=summary["seals"][key]["seal_sha256"],max_bytes=64<<20)
        raw = read(root/"frontiers/source_binding.json")
        frontier = plain(load_forecast_tree(root,raw["value"]))
        expected = {**{name:value for name,value in binding.items() if name != "origins"},
            "origin":binding["origins"][key],"condition":conditions()[position],"counts_before_seal":cohort["counts"]}
        require(frontier == expected and "origins" not in frontier,"sealed own-origin frontier differs/contains another conditioning")
        require(programs[key]["origin_cut"] == frontier["origin"]["cut"] and programs[key]["condition_id"] == key
            and programs[key]["terminal_latent"]["descriptor"] == cohort["terminal_descriptors"][key],"program/consumer cohort join differs")
        first = read(data/"first-images"/(key+".json"))
        expected_counts = {"adapter_steps":0,"acts":position+1,"receipt_emissions":position+1,"images":position+1,"vae_calls":position+1}
        require(first["condition_id"] == key and first["origin_cut"] == binding["origins"][key]["cut"]
            and first["cohort_seal_sha256"] == sha(data/"cohort-seal.json") and first["target_calls"] == targets(0)
            and all(first["counts"][name] == value for name,value in expected_counts.items()),"actual pre-native PNG timing differs")
        row = read(data/"rows"/(key+".predicted.json"))
        require(first["image"] == row["image"],"first actual PNG not joined to consumer row")
        observation = read(root/"observations/predicted_terminal_vae.json")
        actual = plain(load_forecast_tree(root,observation["value"]))
        require(actual == first,"transaction actual consumer observation differs")
        reports[key] = {"verification":verification,"saved_report":read(root/"report.json"),"first_image":first,
            "seal_sha256":summary["seals"][key]["seal_sha256"]}
    return reports


def manifest_fields(manifest):
    return independent()["OLD"]["described_mapping"](manifest["payload"])


def native_field_descriptor(manifest,name):
    fields = manifest_fields(manifest)
    if name in {"image","text","latent"}:
        return fields[name]
    if name == "noise":
        return fields["noise_prediction"]
    context = independent()["OLD"]["described_mapping"](fields["context"])
    if name.startswith("context."):
        return context[name.split(".")[1]]
    return context["rope"]["items"][0 if name == "rope.cos" else 1]


def native_path(e,data,key,origin):
    rows = [json.loads(line) for line in (data/"native"/(key+".jsonl")).read_text().splitlines()]
    require(len(rows) == 112,"native path does not contain all112 transitions")
    before = origin["cut"]
    for position,row in enumerate(rows):
        logical,local = divmod(position,28)
        capture,r = row["capture"],row["receipt"]
        require((row["transition_index"],row["logical_step"],row["local_index"]) == (position,logical,local)
            and r == e.receipts[r["fingerprint"]] and r["operation"] == "continue" and r["steps"] == 1
            and r["parent"] == before and r["result"] == capture["cut"] and len(r["transitions"]) == 1,
            "native path receipt/parent/position differs")
        manifest = capture["manifest"]
        require(identity(manifest) == capture["cut"] and capture["point"] == e.typed["unpack"](manifest["execution_point"])
            and capture["point"] == r["transitions"][0]["after"]
            and tuple(capture["point"][name] for name in ("logical_step","phase","index")) == independent()["point_after"](position),
            "native manifest/cursor identity differs")
        retained = local in LANDMARKS
        require(capture["retained_full_cut"] == retained and (capture["cut"] in e.store["cuts"]) == retained,
            "declared hydratable coverage differs")
        if retained:
            require(e.store["cuts"][capture["cut"]] == manifest,"retained typed cut/metadata manifest differs")
        for name,value in row["outputs"].items():
            declared = native_field_descriptor(manifest,name)
            require(declared["shape"] == value["descriptor"]["shape"] and declared["sha256"] == value["descriptor"].get(
                "bf16_le_sha256" if declared["dtype"] == "torch.bfloat16" else "fp32_le_sha256")
                and (value["archive"] is not None) == retained and value["witness"]["flat_indices"] == witness_indices(declared["shape"]),
                "native field/witness is not the typed manifest port")
            if retained:
                check_field(e.array(value["archive"]),value,e=e,exact_required=True)
        before = capture["cut"]
    return rows


def audit_program(e,binding,origin,program,native,*,progress=None):
    require(program["chain_length"] == len(program["transitions"]) == 112,"program horizon differs")
    rows,all_hashes,field_count,scalar_count = [],True,0,0
    previous,latent_descriptor = None,describe_field(e.array(origin["inputs"]["latent"]))
    context_id = None
    for logical,local,kind,index,inputs,fields in predictions(e,binding,origin):
        position = logical*28+local
        sealed,n = program["transitions"][position],native[position]
        require((sealed["logical_step"],sealed["local_index"],sealed["transition_index"],sealed["kind"],sealed["index"])
            == (logical,local,position,kind,index) and set(sealed["outputs"]) == set(n["outputs"]) == set(fields)
            and sealed["full_arrays_retained"] == (local in LANDMARKS),"forecast/native field inventory/order differs")
        # Verify joins against the sealed descriptors separately from numerical reconstruction.
        if local == 0:
            require(sealed["inputs"]["latent"] == latent_descriptor and all(sealed["inputs"][name] == inputs[name]
                for name in ("origin_conditioning","origin_txt_ids","schedule")),"project borrowed a refreshed latent/conditioning/IDs/schedule")
            context_id = identity({name:record["descriptor"] for name,record in sealed["outputs"].items() if name.startswith(("context.","rope."))})
        elif local <= 25:
            require(all(sealed["inputs"][name] == previous[name]["descriptor"] for name in ("image","text"))
                and sealed["inputs"]["own_project_context"] == context_id,"block borrowed another I/T/context")
        elif local == 26:
            require(sealed["inputs"]["image"] == previous["image"]["descriptor"]
                and sealed["inputs"]["own_project_context"] == context_id,"readout borrowed another I/context")
        else:
            require(sealed["inputs"]["latent"] == latent_descriptor and sealed["inputs"]["noise"] == previous["noise"]["descriptor"]
                and sealed["inputs"]["sigma"] == inputs["sigma"] and sealed["inputs"]["sigma_next"] == inputs["sigma_next"],"commit borrowed latent/noise/schedule")
            latent_descriptor = sealed["outputs"]["latent"]["descriptor"]
        per_field = {}
        for name,value in fields.items():
            record,actual = sealed["outputs"][name],n["outputs"][name]
            require((record["archive"] is not None) == (local in LANDMARKS),"untruthful forecast archive coverage")
            rebuilt = check_field(value,record,e=e)
            all_hashes = all_hashes and rebuilt["full_descriptor_match"]
            field_count += 1
            scalar_count += value.size
            sample_p = np.asarray(record["witness"]["values"],dtype=np.float32)
            sample_n = np.asarray(actual["witness"]["values"],dtype=np.float32)
            require(record["witness"]["flat_indices"] == actual["witness"]["flat_indices"],"native/predicted witness addresses differ")
            full = compare(e.array(record["archive"]),e.array(actual["archive"])) if local in LANDMARKS else None
            observed = n["errors"][name]
            reverse = -sample_p.astype(np.float64)+sample_n.astype(np.float64)
            require(observed["fixed_witness"]["signed_mean"] == float(reverse.mean())
                and observed["fixed_witness"]["sign"] == "native minus sealed prediction"
                and observed["fixed_witness"]["relative_rms_denominator"] == "sealed prediction RMS"
                and (observed["full_field"] is not None) == (local in LANDMARKS),"worker-observed comparison sign/scope differs")
            per_field[name] = {"independent_reconstruction":rebuilt,"predicted_minus_native_witness":compare(sample_p,sample_n),
                "predicted_minus_native_full":full,"native_full_moments":actual["moments"],
                "native_moments_scope":"Recomputed from retained full values" if local in LANDMARKS else "Worker observed; full values not retained",
                "native_cut":n["capture"]["cut"],"full_native_array_retained":local in LANDMARKS}
        rows.append({"transition_index":position,"logical_step":logical,"local_index":local,"kind":kind,"index":index,
            "native_point":n["capture"]["point"],"native_receipt":n["receipt"]["fingerprint"],"fields":per_field})
        previous = sealed["outputs"]
        if progress:
            progress({"condition_id":origin["condition_id"],"logical_step":logical,"local_index":local,"reconstructed_fields":field_count})
    require(program["terminal_latent"] == program["transitions"][-1]["outputs"]["latent"],"terminal prediction is not final own state")
    return {"transitions":rows,"complete_field_hashes_match":all_hashes,"reconstructed_fields":field_count,
        "reconstructed_scalar_values":int(scalar_count),"full_native_comparison_transitions":20,
        "witness_only_native_comparison_transitions":92,"chain_length":112}


def factorial(values):
    """Signed same-noise paired contrasts; no semantic interpretation."""
    arrays = {key:np.asarray(values[key],dtype=np.float64) for key in CELLS}
    return {"color_at_o0":arrays["c1-o0"]-arrays["c0-o0"],"color_at_o1":arrays["c1-o1"]-arrays["c0-o1"],
        "order_at_c0":arrays["c0-o1"]-arrays["c0-o0"],"order_at_c1":arrays["c1-o1"]-arrays["c1-o0"],
        "color":((arrays["c1-o0"]-arrays["c0-o0"])+(arrays["c1-o1"]-arrays["c0-o1"]))/2,
        "order":((arrays["c0-o1"]-arrays["c0-o0"])+(arrays["c1-o1"]-arrays["c1-o0"]))/2,
        "interaction":arrays["c1-o1"]-arrays["c1-o0"]-arrays["c0-o1"]+arrays["c0-o0"]}


def endpoint_rows(e,data,summary,binding,programs,natives,*,production_profile):
    old = independent()
    rows = {path.stem:read(path) for path in (data/"rows").glob("*.json")}
    expected_ids = {key+suffix for key in CELLS for suffix in (".predicted",".native")}|{"c0-o0.sham"}
    require(set(rows) == expected_ids and summary["rows"] == rows,"actual endpoint census differs")
    latents,pixels,reports = {},{},{}
    for name,row in rows.items():
        require(name == row["id"] and row["denoiser_suffix_steps"] == 0 and row["final"]["retained_full_cut"] is True
            and row["final"]["manifest"] == e.store["cuts"][row["final"]["cut"]],"actual terminal row/cut differs")
        cut = row["final"]["cut"]
        latent = e.array(row["final_latent"])
        require(old["exact"](latent,e.slot(cut,"latent")) and tuple(e.point(cut)[key] for key in ("logical_step","phase","index"))
            == (4,"project",None) and all(e.typed["unpack"](e.fields(cut)[key]) is None for key in
                ("context","image","text","noise_prediction")),"terminal commit/consumer closure differs")
        image_path = old["H"]["file_path"](data,row["image"]["path"])
        require(sha(image_path) == row["image"]["sha256"],"actual native VAE PNG hash differs")
        with Image.open(image_path) as image:
            pixel = np.asarray(image).copy()
            require(image.mode == "RGB" and (not production_profile or image.size == (512,512)),"actual PNG geometry/mode differs")
        latents[name],pixels[name] = latent,pixel
        reports[name] = {"image":row["image"],"native_cut":cut,"final_latent":row["final_latent"],"actual_native_vae":True,
            "denoiser_suffix_steps":0,"scope":"Actual native VAE response to complete terminal latent; RGB itself was not mathematically predicted."}
        if name.endswith(".predicted"):
            key = name.removesuffix(".predicted")
            final = e.array(programs[key]["terminal_latent"]["archive"])
            action = row["acts"][0]
            wrapped = {"receipt":action,"receipt_id":action["fingerprint"],"act":action["act"],"parent":action["parent"],"post_act":action["result"]}
            require(len(row["acts"]) == 1 and row["entry"]["cut"] == action["parent"] and cut == action["result"],"predicted terminal Act/entry differs")
            old["TWO"]["PREV"]["atomic_act"](e,wrapped,action["parent"],{"latent":final,"image":None,"text":None,"context":None})
            old["TWO"]["check_derived_terminal"](e,read(data/"derivations"/(key+".json")),binding["origins"][key]["cut"],wrapped,
                final,read(data/"plan.json")["expected_derived_terminal_module_sha256"])
        elif name.endswith(".native"):
            key = name.removesuffix(".native")
            require(not row["acts"] and cut == natives[key][-1]["capture"]["cut"] == row["entry"]["cut"],
                "native terminal borrowed another path")
    sham = rows["c0-o0.sham"]
    action = sham["acts"][0]
    wrapped = {"receipt":action,"receipt_id":action["fingerprint"],"act":action["act"],"parent":action["parent"],"post_act":action["result"]}
    old["TWO"]["PREV"]["atomic_act"](e,wrapped,rows["c0-o0.native"]["final"]["cut"],
        {"latent":latents["c0-o0.native"],"image":None,"text":None,"context":None})
    controls = {"sham_png_exact":rows["c0-o0.native"]["image"]["sha256"] == sham["image"]["sha256"],
        "sham_latent_exact":old["exact"](latents["c0-o0.native"],latents["c0-o0.sham"])}
    require(summary["control_observations"] == controls,"retained sham observation differs")
    endpoints = {key:{"latent":compare(latents[key+".predicted"],latents[key+".native"]),
        "rgb":compare(pixels[key+".predicted"],pixels[key+".native"])} for key in CELLS}
    contrasts = {}
    for space,values in (("latent",latents),("rgb",pixels)):
        p = factorial({key:values[key+".predicted"] for key in CELLS})
        n = factorial({key:values[key+".native"] for key in CELLS})
        contrasts[space] = {name:{"prediction_minus_native_contrast":compare(p[name],n[name]),"predicted_contrast":moments(p[name]),
            "native_contrast":moments(n[name]),"sign":"color=c1-c0 averaged over order; order=o1-o0 averaged over color; interaction=c11-c10-c01+c00"} for name in p}
    return {"rows":reports,"endpoints":endpoints,"factorial_contrasts":contrasts,"controls":controls,
        "contrast_definitions":{"color_at_o0":"c10-c00","color_at_o1":"c11-c01","order_at_c0":"c01-c00",
            "order_at_c1":"c11-c10","color":"(color_at_o0+color_at_o1)/2","order":"(order_at_c0+order_at_c1)/2",
            "interaction":"c11-c10-c01+c00"},
        "scope":"Endpoint-only signed contrasts. Four dependent prompt conditions from one actual noise draw. Numeric/pixel contrasts do not classify colors, shape or spatial semantics."}


def analyze_data(data,*,parameter_root,expected_request=None,expected_job=None,checkpoint=None,ssh=None,
                 production_profile=True,progress=None):
    """Pure saved-evidence/math core; never grants authenticated custody by itself.

    expected_request is the complete sealed request dict when available. The
    external caller must authenticate inventories and source authority before
    promoting this report's custody-unassessed status. No pipeline/model calls.
    """
    started = time.monotonic()
    data = Path(data).resolve()
    old = independent()
    from mdb import parameter_artifacts
    require(sha(inspect.getfile(parameter_artifacts)) == PARAMETER_SHA256,"installed parameter hydration module differs")
    source,summary,plan,binding = [read(data/name) for name in ("source.json","summary.json","plan.json","binding.json")]
    validate_plan(plan,source,summary,production_profile=production_profile)
    require(source["job_id"] == summary["job_id"] and source["request_sha256"] == summary["request_sha256"]
        and source["binding_sha256"] == sha(data/"binding.json"),"saved source/request/binding differs")
    if expected_job is not None:
        require(source["job_id"] == expected_job,"actual parent job differs")
    if expected_request is not None:
        require(plan == expected_request["config"] and source["request_sha256"] == expected_request["request_sha256"]
            and source["source_hashes"] == expected_request["source_hashes"],"authenticated request/source differs")
    recipes,cache = old["parameter_cache"](data,parameter_root,checkpoint=checkpoint,ssh=ssh)
    e = old["Evidence"](data,parameter_root,recipes)
    require(set(binding["origins"]) == set(CELLS) and binding["parameter_artifacts"]["path"] == "parameter-artifacts.json"
        and binding["parameter_artifacts"]["sha256"] == sha(data/"parameter-artifacts.json")
        and sha(data/"retrieve_weights.py") == PARAMETER_SHA256,"recipe inventory/source differs")
    manifest = read(data/"parameter-artifacts.json")
    require(manifest["provenance"]["job_id"] == source["job_id"] and manifest["provenance"]["request_sha256"] == source["request_sha256"]
        and manifest["provenance"]["retrieval_module_sha256"] == PARAMETER_SHA256
        and all(not (data/path).exists() for path in recipes),"parameter recipe scope differs/saved weight copy present")
    if production_profile:
        require(len(recipes) == 169 and source["checkpoint_files"][CHECKPOINT_FILE] == CHECKPOINT_SHA256
            and all(record["source"]["file_sha256"] == CHECKPOINT_SHA256 for record in recipes.values())
            and source["checkpoint_sha256"] == identity(source["checkpoint_files"]) == plan["expected_checkpoint_sha256"],"qualified checkpoint identity differs")
        old["bank_profiles"](binding)
        source_abi(data,source,plan)
        require(manifest["provenance"]["producer_source_sha256"] == source["source_hashes"]["flux_prompt_factorial.py"],
            "recipe producer is not this request's worker")
    ports = parameter_ports(binding)
    require(all(record["model_identity"] == source["model_identity"] and record["execution"] == source["execution"]
        for record in e.store["cuts"].values()) and all(record["model_identity"] == source["model_identity"]
        and record["execution"] == source["execution"] for record in e.receipts.values()),"typed model/execution identity differs")
    origin_values = {key:old["check_origin"](e,origin,0) for key,origin in binding["origins"].items()}
    excluded = {"conditioning","txt_ids"}
    base = {key:value for key,value in e.fields(binding["origins"][CELLS[0]]["cut"]).items() if key not in excluded}
    for key,origin in binding["origins"].items():
        require({name:value for name,value in e.fields(origin["cut"]).items() if name not in excluded} == base,
            "four origins do not share exact Z0/IDs/clock/schedule")
        require(origin["condition_id"] == key and origin_values[key]["txt_ids"].dtype == np.int64,"integer source IDs lost exact dtype")
        if production_profile:
            for slot in ("conditioning","txt_ids"):
                require(old["exact"](e.array(source["encodings"][key][slot]),origin_values[key][slot]),"condition encoding and own origin differ")
    for logical in range(4):
        schedule = binding["pass_schedules"][str(logical)]
        require(float(e.array(schedule["timestep"])) == float(origin_values[CELLS[0]]["timesteps"][logical])
            and float(e.array(schedule["sigma"])) == float(origin_values[CELLS[0]]["sigmas"][logical])
            and float(e.array(schedule["sigma_next"])) == float(origin_values[CELLS[0]]["sigmas"][logical+1])
            and schedule["next_clock"] == {"step":logical+1,"phase":"project","cursor":0}
            and schedule["reset"] == {key:None for key in ("context","image","text","noise_prediction")},"pass schedule borrowed future/state")
    from mdb.forecast_transaction import load_forecast_tree
    programs = {key:plain(load_forecast_tree(data/"transactions"/key,read(data/"transactions"/key/"forecasts/program.json")["predictions"])) for key in CELLS}
    transactions = verify_cohort(data,binding,summary,programs,e)
    natives = {key:native_path(e,data,key,binding["origins"][key]) for key in CELLS}
    ledger = [json.loads(line) for line in (data/"receipt-emissions.jsonl").read_text().splitlines()]
    expected_ledger = [(key+".predicted",read(data/"rows"/(key+".predicted.json"))["acts"][0]["fingerprint"]) for key in CELLS]
    expected_ledger += [(key+".native",row["receipt"]["fingerprint"]) for key in CELLS for row in natives[key]]
    expected_ledger += [("c0-o0.sham",read(data/"rows/c0-o0.sham.json")["acts"][0]["fingerprint"])]
    require(len(e.receipts) == len(ledger) == 453 and all(row == {"sequence":i,"branch":branch,"receipt":receipt}
        for i,(row,(branch,receipt)) in enumerate(zip(ledger,expected_ledger,strict=True))),"physical receipt ledger order/census differs")
    numeric = {key:audit_program(e,binding,binding["origins"][key],programs[key],natives[key],progress=progress) for key in CELLS}
    endpoint = endpoint_rows(e,data,summary,binding,programs,natives,production_profile=production_profile)
    matched = all(row["complete_field_hashes_match"] for row in numeric.values())
    from mdb import tensor_archive
    return {"schema":"mdb-prompt-factorial-audit-v1","mechanics_status":"numeric-core-valid-custody-unassessed",
        "custody_status":"unassessed-by-pure-core","instrument_status":"independent-full-hashes-qualified" if matched else "independent-reconstruction-disagrees",
        "source":{"job_id":source["job_id"],"request_sha256":source["request_sha256"],"source_sha256":sha(data/"source.json"),
            "binding_sha256":sha(data/"binding.json"),"cohort_seal_sha256":sha(data/"cohort-seal.json"),"archive_sha256":None},
        "counts":summary["counts"],"mathematical_transitions":448,"conditions":conditions(),"transactions":transactions,
        "numeric":numeric,"endpoint_comparisons":endpoint,"parameter_custody":cache,"native_parameter_ports":ports,"seconds":time.monotonic()-started,
        "auditor_source_sha256":sha(Path(__file__)),"independent_sources":INDEPENDENT_SOURCES,
        "parameter_artifacts_source_sha256":sha(inspect.getfile(parameter_artifacts)),
        "tensor_archive_source_sha256":sha(inspect.getfile(tensor_archive)),
        "coverage":{"complete_predicted_fields_reconstructed":sum(row["reconstructed_fields"] for row in numeric.values()),
            "reconstructed_scalar_values":sum(row["reconstructed_scalar_values"] for row in numeric.values()),
            "full_native_landmark_transitions":80,"native_witness_only_transitions":368,"full_native_replay_retained":False},
        "scope":"One noise specimen, four dependent prompt conditions. cone_right code0 means cone-left and code1 cone-right. Text witness positions are tensor positions, not lexical correspondences. RGB factorial interaction is a signed pixel contrast, not semantic binding. Full native RMS only declared landmarks/finals; all other native errors fixed-witness only. No semantic success or universal bound; every numerical discrepancy retained.",
        "terminal_status":"unassessed"}


def analyze(run,*,parameter_root,checkpoint=None,ssh=None,progress=None):
    """Authenticate completed direct collection before invoking the pure core."""
    run = Path(run).resolve()
    collector_path = HERE/"collect_flux_prompt_factorial.py"
    collector = runpy.run_path(str(collector_path))
    request,job,collection = [read(run/name) for name in ("request.json","collection-job.json","collection.json")]
    require(request["request_sha256"] == hashlib.sha256(json.dumps({key:value for key,value in request.items()
        if key != "request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest(),"sealed request digest differs")
    collector["validate_job"](job,job["job_id"],request)
    require(collection["collector_source_sha256"] == sha(collector_path) and collection["job_id"] == job["job_id"]
        and collection["request_sha256"] == request["request_sha256"] and collection["scheduler_status"] == "succeeded"
        and collection["archive_sha256"] is None and all(sha(run/name) == value for name,value in collection["bindings"].items()),"authenticated collection/source binding differs")
    data = run/"extracted"/RESULT
    actual = collector["inventory"](data)
    require(actual == read(run/"collection-remote-before.json") == read(run/"collection-remote-after.json")
        == read(run/"collection-local-inventory.json"),"remote before/after/local inventory differs")
    collector["verify_manifest"](data,actual,job["job_id"],request)
    require(job["result"]["diagnostics"]["command"] == read(run/"admission-command.json")["command"],"actual admitted command differs")
    for name,digest in request["source_hashes"].items():
        require(sha(independent()["H"]["file_path"](run/"payload",name)) == digest,"executed source cohort differs")
    for name,digest in request["independent_math_sources"].items():
        require(sha(HERE/name) == sha(run/"independent-math-freeze"/name) == digest,"pre-outcome independent source differs")
    result = analyze_data(data,parameter_root=parameter_root,expected_request=request,expected_job=job["job_id"],
        checkpoint=checkpoint,ssh=ssh,progress=progress)
    result.update(mechanics_status="valid",custody_status="authenticated-before-after-local-inventory-verified")
    result["source"].update(collection_sha256=sha(run/"collection.json"),manifest_sha256=sha(data/"remote-manifest.json"),
        collection_mode=collection["collection_mode"],scheduler_status=job["state"],verified_artifact_files=actual["file_count"],
        expanded_bytes=actual["expanded_bytes"],executed_payload_sha256=collection["executed_payload_sha256"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,nargs="?")
    parser.add_argument("--data",type=Path,help="Pure numeric core only; custody remains unassessed")
    parser.add_argument("--parameter-root",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--ssh")
    parser.add_argument("--expected-request",type=Path)
    parser.add_argument("--expected-job")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    require((args.run is None) != (args.data is None),"choose authenticated run or direct data core")
    require(not args.out.exists(),"immutable audit output already exists")
    progress = lambda value:print(json.dumps({"phase":"independent-factorial",**value}),flush=True)
    if args.data:
        result = analyze_data(args.data,parameter_root=args.parameter_root,checkpoint=args.checkpoint,ssh=args.ssh,
            expected_request=read(args.expected_request) if args.expected_request else None,expected_job=args.expected_job,progress=progress)
    else:
        result = analyze(args.run,parameter_root=args.parameter_root,checkpoint=args.checkpoint,ssh=args.ssh,progress=progress)
    with args.out.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
