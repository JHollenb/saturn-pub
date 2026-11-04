"""Seal full image attention/MLP fields, lower them into MDB branches, consume native RGB."""
from __future__ import annotations

import gc
import importlib.metadata
import inspect
import json
import os
import runpy
import time
from pathlib import Path

CONTEXT = {"id": "p1-s73", "prompt_id": "p1", "seed": 73,
    "prompt": "A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph."}
QUERIES = [0, 31, 32, 255, 511, 512, 767, 1023]
NEURONS = list(range(0, 9216, 288))
COUNTS = {"model_loads": 1, "prompt_encodings": 1, "noise_draws": 1, "adapter_steps": 245,
    "acts": 11, "candidate_calls": 9, "receipt_emissions": 256, "images": 9, "vae_calls": 9, "physical_batch": 1}

BRANCHES = [
    {"id": "zero", "base": "zero", "feature": "native", "source": "zero", "selection": "native"},
    {"id": "text", "base": "text", "feature": "native", "source": "text", "selection": "native"},
    {"id": "sham", "base": "zero", "feature": "measured", "source": "zero", "selection": "AM"},
    {"id": "low.predicted.AM", "base": "zero", "feature": "predicted", "source": "zero", "selection": "AM"},
    {"id": "high.predicted.AM", "base": "text", "feature": "predicted", "source": "text", "selection": "AM"},
    {"id": "AM.measured", "base": "zero", "feature": "measured", "source": "text", "selection": "AM"},
    {"id": "AM.predicted", "base": "zero", "feature": "predicted", "source": "text", "selection": "AM"},
    {"id": "A.predicted", "base": "zero", "feature": "predicted", "source": "text", "selection": "A"},
    {"id": "M.predicted", "base": "zero", "feature": "predicted", "source": "text", "selection": "M"},
]
BRANCH_EXECUTION_ORDER = ["zero", "text", "AM.measured", "sham", "low.predicted.AM",
    "high.predicted.AM", "AM.predicted", "A.predicted", "M.predicted"]
OBSERVATION_PLAN = {"reference_rows": ["zero", "text", "AM.measured"], "boundaries_per_branch": 16,
    "slots": ["image", "text", "latent", "noise_prediction"],
    "reference_storage": "lossless-bf16-bits", "candidate_storage": "descriptor-and-fixed-coordinate-witness",
    "witness_count": 64, "witness_rule": "unique floor(linspace(0,numel-1,min(64,numel))) in C-value order",
    "comparison_rule": "all reference rows at or before this row in branch_execution_order, at identical cursor",
    "latent_rule": "stored Z3 is constant before native scheduler commit; only committed Z4 is updated latent",
    "image_text_rule": "working single outputs through readout entry; stored I/T after readout are stale",
    "noise_rule": "native readout output is meaningful only at scheduler entry"}


def validate_config(config):
    helper = {"BRANCHES": BRANCHES}
    if (config.get("context") != CONTEXT or config.get("height") != 512 or config.get("width") != 512
            or config.get("steps") != 4 or config.get("single_index") != 6
            or config.get("selector") != {"logical_step": 3, "phase": "single", "index": 6}
            or config.get("query_indices") != QUERIES or config.get("neuron_indices") != NEURONS
            or config.get("observation_plan") != OBSERVATION_PLAN
            or config.get("attention_query_chunk") != 64 or config.get("attention_projection_chunk") != 256
            or config.get("branch_execution_order") != BRANCH_EXECUTION_ORDER
            or config.get("neuron_chunk") != 256 or config.get("array_storage") != "bf16-bits"
            or config.get("archive_integer_ids") != "native-int64" or config.get("branches") != helper["BRANCHES"]):
        raise ValueError("full attention/MLP replacement scientific geometry differs")
    for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls"):
        if type(config.get("max_" + key)) is not int or config["max_" + key] != COUNTS[key]:
            raise ValueError("full attention/MLP replacement finite census differs: " + key)
    for key, lo, hi in (("max_wall_seconds", 1, 1500), ("cpu_threads", 4, 4), ("archive_compression_level", 1, 9)):
        if type(config.get(key)) is not int or not lo <= config[key] <= hi:
            raise ValueError("invalid full attention/MLP replacement budget: " + key)



def attention_profile(block):
    import torch

    processor=block.attn.processor
    if (type(processor).__module__!="diffusers.models.transformers.transformer_flux2"
            or type(processor).__name__!="Flux2ParallelSelfAttnProcessor"
            or processor._attention_backend is not None or processor._parallel_config is not None):
        raise ValueError("requires qualified default nonparallel native attention processor")
    return {"processor_type":type(processor).__module__+"."+type(processor).__name__,
        "processor_backend":processor._attention_backend,"parallel_config":processor._parallel_config,
        "attention_mask":None,"is_causal":False,"kv_cache":False,"joint_attention_kwargs":None,
        "sdpa_enabled_flags":{name:bool(getattr(torch.backends.cuda,name)()) for name in
            ("flash_sdp_enabled","mem_efficient_sdp_enabled","math_sdp_enabled","cudnn_sdp_enabled")},
        "scope":"Declared default dispatcher profile and backend-enabled flags, not evidence of the kernel actually selected."}


def bind_prediction_inputs(block, cuts, *, array, counts, builder_ref, preparation_ref, config):
    """Bind only current pre-entry state, actual parameters and RoPE."""
    import numpy as np
    import torch

    from mdb.forward_recording import tensor_descriptor

    attn, fused = block.attn, block.attn.to_qkv_mlp_proj
    h, i, heads, dim = attn.inner_dim, attn.mlp_hidden_dim, attn.heads, attn.head_dim
    if (tuple(fused.weight.shape) != (3*h+2*i,h) or h != heads*dim
            or block.norm.elementwise_affine or tuple(block.norm.normalized_shape) != (h,)
            or block.norm.eps != 1e-6 or attn.norm_q.eps != attn.norm_k.eps or attn.norm_q.eps!=block.norm.eps
            or not attn.norm_q.elementwise_affine or not attn.norm_k.elementwise_affine):
        raise ValueError("unsupported full attention/MLP native parameter profile")
    weight, bias = fused.weight, fused.bias
    parameters = {"qkv_weight": weight[:3*h], "qkv_bias": None if bias is None else bias[:3*h],
        "mlp_weight": weight[3*h:], "mlp_bias": None if bias is None else bias[3*h:],
        "norm_q_weight": attn.norm_q.weight, "norm_k_weight": attn.norm_k.weight}
    binding = {"schema": "mdb-flux-attention-replacement-binding-v1", "inputs": {},
        **{name+"_rows" if name in {"qkv_weight","qkv_bias","mlp_weight","mlp_bias"} else name:
            None if value is None else array(value) for name,value in parameters.items()},
        "qkv_weight_row_range": [0,3*h], "mlp_weight_row_range": [3*h,3*h+2*i],
        "hidden_width":h,"mlp_width":i,"heads":heads,"head_dim":dim,"eps":block.norm.eps,
        "qk_eps":attn.norm_q.eps,"elementwise_affine":False,"neuron_chunk":config["neuron_chunk"],
        "attention_query_chunk":config["attention_query_chunk"],"attention_projection_chunk":config["attention_projection_chunk"],
        "query_indices":QUERIES,"neuron_indices":NEURONS,
        "linear_path":"single_transformer_blocks.6.attn.to_qkv_mlp_proj",
        "native_parameters":{"fused":{name:tensor_descriptor(value) for name,value in fused.named_parameters()},
            "norm_q":{name:tensor_descriptor(value) for name,value in attn.norm_q.named_parameters()},
            "norm_k":{name:tensor_descriptor(value) for name,value in attn.norm_k.named_parameters()}},
        "counts_before_seal":dict(counts),"target_entries_before_seal":0,"builder":builder_ref,"preparation":preparation_ref,
        "attention_profile":attention_profile(block),
        "observation_plan":config["observation_plan"],"branch_execution_order":config["branch_execution_order"],
        "scope":"Incoming concatenated text/image, raw single modulation, all actual fused QKV/MLP rows, actual RMSNorm weights/eps and incoming RoPE. No current-target native activation input."}
    operands = {"parameters":{},"cases":{}}
    def immutable(value):
        if value.dtype not in {torch.bfloat16,torch.float32}:
            raise ValueError("unqualified attention prediction operand dtype")
        value = np.array(value.detach().float().cpu().numpy(),copy=True)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite attention prediction operand")
        value.setflags(write=False)
        return value
    for name,value in parameters.items():
        if value is not None and value.dtype!=torch.bfloat16:
            raise ValueError("native attention/MLP parameter dtype differs")
        operands["parameters"][name] = None if value is None else immutable(value)
    for role,cut in cuts.items():
        text,image = cut.payload["text"],cut.payload["image"]
        rope = cut.payload["context"]["rope"]
        if not isinstance(rope,(tuple,list)) or len(rope)!=2:
            raise ValueError("incoming full attention RoPE profile differs")
        # Keep shape validation separate from chained comparisons.
        if tuple(rope[0].shape)!=(text.shape[1]+image.shape[1],dim) or tuple(rope[1].shape)!=tuple(rope[0].shape):
            raise ValueError("incoming full attention RoPE extent differs")
        mod = cut.payload["context"]["single_mod"]
        if (image.dtype!=torch.bfloat16 or text.dtype!=torch.bfloat16 or mod.dtype!=torch.bfloat16
                or image.shape[0]!=1 or text.shape[0]!=1 or image.shape[-1]!=h or text.shape[-1]!=h
                or tuple(mod.shape) not in {(1,3*h),(1,1,3*h)}
                or rope[0].dtype!=torch.float32 or rope[1].dtype!=torch.float32):
            raise ValueError("incoming attention state dtype/shape differs")
        binding["inputs"][role]={"cut":cut.fingerprint,"image":array(image),"text":array(text),
            "single_mod":array(mod),"rope_cos":array(rope[0]),"rope_sin":array(rope[1])}
        operands["cases"][role]={"image":immutable(image),"residual":immutable(torch.cat((text,image),dim=1)),
            "mod":immutable(mod),"rope_cos":immutable(rope[0]),"rope_sin":immutable(rope[1]),"text_rows":text.shape[1]}
    return binding,operands


def predict_fields(case, parameters, binding, *, helper, attention_helper):
    return {"attention":attention_helper["predict_attention_field"](case["residual"],case["mod"],
        parameters["qkv_weight"],parameters["qkv_bias"],text_rows=case["text_rows"],heads=binding["heads"],
        norm_q_weight=parameters["norm_q_weight"],norm_k_weight=parameters["norm_k_weight"],
        layernorm_eps=binding["eps"],qk_eps=binding["qk_eps"],rope_cos=case["rope_cos"],rope_sin=case["rope_sin"],
        query_chunk=binding["attention_query_chunk"],projection_chunk=binding["attention_projection_chunk"],
        query_indices=binding["query_indices"]),
        "mlp":helper["predict_mlp_field"](case["image"],case["mod"],parameters["mlp_weight"],parameters["mlp_bias"],
            eps=binding["eps"],hidden_width=binding["hidden_width"],mlp_width=binding["mlp_width"],
            neuron_chunk=binding["neuron_chunk"],query_indices=binding["query_indices"],neuron_indices=binding["neuron_indices"])}


def bf16_distance(actual, reference, *, chunk=65536):
    """FP64 scalar reduction over immutable BF16 bit snapshots, bounded memory."""
    import numpy as np

    if actual.dtype!=np.uint16 or reference.dtype!=np.uint16 or actual.shape!=reference.shape or not actual.size:
        raise ValueError("continuation observation BF16 geometry differs")
    aa,bb=actual.reshape(-1),reference.reshape(-1)
    squared=signed=absolute=0.
    maximum=0.
    changed=0
    for start in range(0,aa.size,chunk):
        left=(aa[start:start+chunk].astype(np.uint32)<<16).view(np.float32).astype(np.float64)
        right=(bb[start:start+chunk].astype(np.uint32)<<16).view(np.float32).astype(np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("nonfinite continuation observation")
        delta=left-right
        squared+=float(np.sum(delta*delta));signed+=float(delta.sum());absolute+=float(np.abs(delta).sum())
        maximum=max(maximum,float(np.abs(delta).max()))
        changed+=int(np.count_nonzero(aa[start:start+chunk]!=bb[start:start+chunk]))
    return {"sample_count":int(aa.size),"changed_value_count":changed,"exact_bits":changed==0,
        "l2":float(np.sqrt(squared)),"rms":float(np.sqrt(squared/aa.size)),
        "mae":absolute/aa.size,"signed_mean":signed/aa.size,"max_abs":maximum}


def slot_meaning(point, name):
    if name=="latent":
        updated=point["logical_step"]==4 and point["phase"]=="project"
        return {"phase":"committed-Z4" if updated else "stored-Z3-before-commit",
            "meaningful_propagation_observable":updated,"constant_before_commit":not updated,"stale":not updated}
    if name=="noise_prediction":
        meaningful=point["logical_step"]==3 and point["phase"]=="scheduler"
        return {"phase":"native-readout-output" if meaningful else "stored-or-absent-noise",
            "meaningful_propagation_observable":meaningful,"stale":not meaningful}
    stale=point["phase"] not in {"single","readout"}
    return {"phase":"last-single-output-after-readout" if stale else "working-single-stream",
        "meaningful_propagation_observable":not stale,"stale":stale}


class ContinuationObserver:
    """Retain three BF16 reference traces; candidate full norms are observations."""
    def __init__(self, *, array, plan):
        self.array,self.plan=array,plan
        self.references={}
        self.rows={}
        self.buffers={}

    def observe(self, name, cut, *, index, incoming_receipt):
        import numpy as np
        import torch

        from mdb.forward_recording import tensor_descriptor

        reference=name in self.plan["reference_rows"]
        point=dict(cut.execution_point)
        entry={"index":index,"cut":cut.fingerprint,"point":point,"incoming_receipt":incoming_receipt,
            "complete_cut_retained":index in {0,15},"slots":{},"comparisons":{}}
        snapshots={}
        for key in self.plan["slots"]:
            value=cut.payload.get(key)
            if not isinstance(value,torch.Tensor):
                entry["slots"][key]={"present":False,"meaning":slot_meaning(point,key)}
                continue
            if value.dtype!=torch.bfloat16:
                raise ValueError("continuation observer requires actual native BF16 slots")
            descriptor=tensor_descriptor(value)
            cache_key=(descriptor["dtype"],tuple(descriptor["shape"]),descriptor["sha256"])
            if reference and cache_key in self.buffers:
                bits=self.buffers[cache_key]
            else:
                raw=value.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
                bits=np.frombuffer(raw,dtype=np.uint16).reshape(value.shape)
                if reference:
                    self.buffers[cache_key]=bits
            if bits.flags.writeable:
                raise ValueError("reference observation buffer is mutable")
            snapshots[key]=bits
            coordinates=np.unique(np.linspace(0,bits.size-1,min(self.plan["witness_count"],bits.size)).astype(np.int64))
            witness=(bits.reshape(-1)[coordinates].astype(np.uint32)<<16).view(np.float32)
            saved={"present":True,"descriptor":descriptor,"meaning":slot_meaning(point,key),
                "witness":{"flat_indices":coordinates.tolist(),"values_fp32":witness.tolist(),"value_order":"C",
                    "scope":"Fixed source-independent coordinates; permits sampled checks only, not independent full candidate norms."}}
            if reference:
                saved["array"]=self.array(value)
            entry["slots"][key]=saved
        if reference:
            self.references.setdefault(name,{})[index]={"point":point,"cut":cut.fingerprint,"snapshots":snapshots}
        for ref in self.plan["reference_rows"]:
            if ref not in self.references or index not in self.references[ref]:
                continue
            retained=self.references[ref][index]
            if retained["point"]!=point:
                raise ValueError("reference/candidate observation cursor differs")
            comparison={"reference_cut":retained["cut"],"slots":{}}
            for key in self.plan["slots"]:
                if key in snapshots and key in retained["snapshots"]:
                    comparison["slots"][key]=bf16_distance(snapshots[key],retained["snapshots"][key])
                else:
                    comparison["slots"][key]={"status":"not-present-in-both-current-states"}
            entry["comparisons"][ref]=comparison
        self.rows.setdefault(name,[]).append(entry)
        if len(self.rows[name])!=index+1:
            raise ValueError("continuation observation index duplicated or skipped")
        return entry

    def report(self):
        if set(self.rows)!=set(BRANCH_EXECUTION_ORDER) or any(len(rows)!=16 for rows in self.rows.values()):
            raise ValueError("continuation observation census differs")
        return {"schema":"mdb-flux-attention-continuation-observations-v1","plan":self.plan,
            "branch_execution_order":BRANCH_EXECUTION_ORDER,"rows":self.rows,
            "reference_rows":self.plan["reference_rows"],"model_calls_added":0,
            "scope":"All16 actual cursors observed. Reference I/T/latent/noise arrays retained losslessly when present. Intermediate candidate full arrays not retained; full norms are worker observations with native descriptors plus fixed-coordinate witnesses, not independently recomputed offline full norms. Stored Z3 and stale I/T are labeled separately from active propagation observables."}


def capture_attention_samples(block, *, text_rows, queries):
    """Actual native raw Q/K/V and norm Q/K witnesses at fixed image/key rows."""
    from contextlib import contextmanager

    @contextmanager
    def capture():
        state={"calls":0,"rows":{}}
        active={"index":None}
        handles=[]
        def before(module,args):
            active["index"]=state["calls"]
            state["calls"]+=1
            if active["index"]<2:
                total=args[0].shape[1] if args else 0
                # The block uses keyword-only hidden_states in the qualified path;
                # fused/norm hooks establish actual sequence geometry below.
                state["rows"][("zero","text")[active["index"]]]={"call_index":active["index"],"query":{},"keys":{},"sequence_hint":total}
        def put(name,value):
            index=active["index"]
            if index is None or index>=2:
                return
            row=state["rows"][("zero","text")[index]]
            q=[text_rows+item for item in queries]
            keys=sorted({0,text_rows-1,text_rows,value.shape[1]-1})
            row["query_indices"]=list(queries)
            row["key_indices"]=keys
            row["query"][name]=value[:,q].detach().clone()
            row["keys"][name]=value[:,keys].detach().clone()
        def fused(module,args,output):
            if active["index"] is not None and active["index"]<2:
                h=block.attn.inner_dim
                for name,value in zip(("q_raw","k_raw","v"),output[...,:3*h].split(h,dim=-1)):
                    put(name,value.unflatten(-1,(block.attn.heads,block.attn.head_dim)))
        def norm(name):
            return lambda module,args,output:put(name,output)
        try:
            handles=[block.register_forward_pre_hook(before),block.attn.to_qkv_mlp_proj.register_forward_hook(fused),
                block.attn.norm_q.register_forward_hook(norm("q_norm")),block.attn.norm_k.register_forward_hook(norm("k_norm"))]
            yield state
        finally:
            for handle in handles:
                handle.remove()
    return capture()


def run_panel(*, pipe, adapter, cuts, output, config, store, array, wall, event, counts, helper, capture, predictions):
    """Nine direct blocks, 2 native prefixes, 7 image Acts and 135 suffix steps."""
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb.forward_recording import tensor_descriptor
    from mdb.native_navigation import NativeCutNavigator

    legacy=runpy.run_path("flux_single_contraction.py")
    util=runpy.run_path("flux_single_mlp_campaign.py")
    save,sha=util["save"],util["sha"]
    active,candidates,native,images,prefixes,post,rows,controls={"branch":"direct"},{},{},{},{},{},{},{}
    observer=ContinuationObserver(array=array,plan=config["observation_plan"])
    def receipt(value):
        wall()
        body=value.to_dict()
        store.receipt(value)
        with (output/"receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence":counts["receipt_emissions"]-112,"branch":active["branch"],"receipt":value.fingerprint},sort_keys=True)+"\n")
        counts["receipt_emissions"]+=1
        counts["adapter_steps" if body["operation"]=="continue" else "acts"]+=body.get("steps",1)
        if counts["adapter_steps"]>COUNTS["adapter_steps"] or counts["acts"]>COUNTS["acts"]:
            raise RuntimeError("finite attention/MLP replacement receipt budget exhausted")
    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=137,on_receipt=receipt)
    low,high=cuts["zero"],cuts["text"]
    if (low.model_identity!=high.model_identity or low.execution!=high.execution or set(low.payload)!=set(high.payload)
            or any(describe(value)!=describe(high.payload[key]) for key,value in low.payload.items() if key not in {"image","text"})):
        raise ValueError("prepared sources differ outside working image/text")
    for cut in cuts.values():
        if any(cut.execution_point[key]!=value for key,value in {"logical_step":3,"phase":"single","index":6,"local_clock":6}.items()):
            raise ValueError("source is not current-step pre-single6")
        store.save(cut)
    controls.update({key+"_source_exact":torch.equal(low.payload[key],high.payload[key]) for key in ("image","text")})
    block=pipe.transformer.single_transformer_blocks[6]
    if not isinstance(block.attn.to_out,torch.nn.Linear):
        raise ValueError("replacement consumer is not native full linear port")
    with capture(block,text_rows=low.payload["text"].shape[1],query_indices=QUERIES,neuron_indices=NEURONS) as captured, capture_attention_samples(block,text_rows=low.payload["text"].shape[1],queries=QUERIES) as attention_samples:
        for spec in config["branches"]:
            wall()
            name,base,role=spec["id"],spec["base"],spec["source"]
            event("candidate_start",candidate=name)
            counts["candidate_calls"]+=1
            if spec["feature"]=="native":
                value=legacy["capture_candidate"](block,cuts[base])
                native[base]=value
                replacement=None
            else:
                recipient=native[base]
                donor=recipient["ports"]["original_concat"].detach().clone()
                t,h=recipient["text_rows"],recipient["attention_width"]
                payloads={}
                for field,sl in (("A",slice(None,h)),("M",slice(h,None))):
                    if field not in spec["selection"]:
                        continue
                    reference=recipient["ports"]["original_concat"][:,t:,sl]
                    if spec["feature"]=="predicted":
                        forecast=predictions[role]["attention"]["attention"] if field=="A" else predictions[role]["mlp"]["activation"]
                        payload=helper["prepare_mlp_payload"](forecast,reference)
                    else:
                        payload=native[role]["ports"]["original_concat"][:,t:,sl]
                    donor[:,t:,sl].copy_(payload)
                    payloads[field]=array(payload)
                value=legacy["capture_candidate"](block,cuts[base],selection=spec["selection"],
                    source={"parent":cuts[role].fingerprint,"concat":donor,"text_rows":t,"attention_width":h})
                footprint={"text_concat_exact_recipient":torch.equal(value["ports"]["used_concat"][:,:t],value["ports"]["original_concat"][:,:t])}
                for field,sl in (("A",slice(None,h)),("M",slice(h,None))):
                    expected=donor[:,t:,sl] if field in spec["selection"] else value["ports"]["original_concat"][:,t:,sl]
                    footprint[field+"_image_exact_declared"]=torch.equal(value["ports"]["used_concat"][:,t:,sl],expected)
                if not all(footprint.values()):
                    raise ValueError("attention/MLP consumer replacement violated declared footprint")
                replacement={"feature":spec["feature"],"source":role,"selection":spec["selection"],"payloads":payloads,"controls":footprint,
                    "diagnostics":{"native_original_concat_exact_recipient":torch.equal(value["ports"]["original_concat"],recipient["ports"]["original_concat"]),
                        "post_text_exact_recipient_native":torch.equal(value["text"],recipient["text"])}}
            candidates[name]={key:value[key] for key in ("input_parent","source_parent","selection","text_rows","attention_width","equation_exact","partition_controls")}
            candidates[name].update(ports={key:array(tensor) for key,tensor in value["ports"].items()},replacement=replacement)
            images[name]=value["image"]
            event("candidate_complete",candidate=name)
        for role in ("zero","text"):
            active["branch"]="prefix."+role
            n=navigator(cuts[role])
            entry=n.step()
            post[role]=current=n.capture()
            store.save(current)
            prefixes[role]={"input_cut":cuts[role].fingerprint,"post_single6":current.fingerprint,"receipt":entry.to_dict()}
            for slot in ("image","text"):
                controls[role+"_candidate_"+slot+"_exact_native"]=torch.equal(native[role][slot],current.payload[slot])
        save(output/"prefixes.json",{"prefixes":prefixes})
        save(output/"candidates.json",{"candidates":candidates})
        specs={row["id"]:row for row in config["branches"]}
        for name in config["branch_execution_order"]:
            spec=specs[name]
            active["branch"]=name
            n=navigator(post[spec["base"]])
            row={**spec,"parent":post[spec["base"]].fingerprint,"start_point":n.point,"acts":[],
                "continuation_policy":"recipient-stored-C-no-further-project"}
            if spec["feature"]!="native":
                row["acts"].append(legacy["install_port"](n,"image",images[name],name=name+".image",store=store))
            state=n.capture()
            store.save(state)
            observer.observe(name,state,index=0,incoming_receipt=row["acts"][-1]["receipt_id"] if row["acts"] else prefixes[spec["base"]]["receipt"]["fingerprint"])
            path=[]
            while n.point["logical_step"]<4:
                wall()
                entry=n.step()
                path.append(entry.to_dict())
                observer.observe(name,n.capture(),index=len(path),incoming_receipt=entry.fingerprint)
            if len(path)!=15:
                raise ValueError("attention replacement suffix geometry differs")
            final=n.capture()
            store.save(final)
            if describe(final.payload["conditioning"])!=describe(cuts[spec["base"]].payload["conditioning"]):
                raise ValueError("replacement suffix changed stored conditioning")
            counts["vae_calls"]+=1
            image=adapter.decode(n.session,height=512,width=512)
            file=output/(name+".png")
            image.save(file)
            counts["images"]+=1
            row.update(final_cut=final.fingerprint,final_latent=array(final.payload["latent"]),
                final_conditioning=tensor_descriptor(final.payload["conditioning"]),continuation_receipts=path,
                continuation_observation={"path":"continuation-observations.json","row":name},
                image={"path":file.name,"sha256":sha(file)})
            save(output/"rows"/(name+".json"),row)
            rows[name]=row
            event("branch_complete",row=name)
    observed={role:{"call_index":row["call_index"],**{key:array(row[key]) for key in
        ("layernorm","modulated_input","u","v","activation")}} for role,row in captured["rows"].items()}
    save(output/"entry-observations.json",{"rows":observed,"calls":captured["calls"],"target_entries_before_seal":0})
    attention_rows={role:{"call_index":row["call_index"],"query_indices":row["query_indices"],"key_indices":row["key_indices"],
        **{group:{key:array(value) for key,value in row[group].items()} for group in ("query","keys")}} for role,row in attention_samples["rows"].items()}
    save(output/"attention-stage-observations.json",{"rows":attention_rows,"calls":attention_samples["calls"],"target_entries_before_seal":0,
        "scope":"First two actual native calls only; actual raw fused Q/K/V and actual norm_q/norm_k hooks. No rotated native stage is claimed captured."})
    observations=observer.report()
    save(output/"continuation-observations.json",observations)
    controls["sham_png_equals_zero"]=rows["sham"]["image"]["sha256"]==rows["zero"]["image"]["sha256"]
    return {"rows":[rows[spec["id"]] for spec in config["branches"]],"counts":dict(counts),"control_observations":controls,
        "source_sha256":sha(output/"source.json"),"candidates_sha256":sha(output/"candidates.json"),
        "entry_observations_sha256":sha(output/"entry-observations.json"),
        "attention_stage_observations_sha256":sha(output/"attention-stage-observations.json"),
        "continuation_observations_sha256":sha(output/"continuation-observations.json"),
        "branch_execution_order":config["branch_execution_order"],"observation_plan":config["observation_plan"],
        "target_calls":captured["calls"]}


def worker():
    import faulthandler

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    print("MDB_FLUX_ATTENTION_REPLACEMENT_BOOTSTRAP " + json.dumps({"phase": "worker_enter", "unix_time": time.time()}), flush=True)
    util = runpy.run_path("flux_single_mlp_campaign.py")
    save, sha, identity, json_value = (util[key] for key in ("save", "sha", "identity", "json_value"))
    config, request = (json.loads(Path(os.environ[name]).read_text()) for name in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    validate_config(config)
    if request["config"] != config or any(sha(path) != digest for path, digest in request["source_hashes"].items()):
        raise ValueError("replacement request/source bytes differ")
    request = {**request, "job_id": os.environ["MRUN_JOB_ID"]}
    output = Path("flux-attention-replacement-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "plan-frozen.json", {"plan_sha256": sha(output / "plan.json"), "request_sha256": request["request_sha256"],
        "source_identity": identity(request["source_hashes"]), "job_id": request["job_id"],
        "work": {key: 0 for key in COUNTS if key != "physical_batch"}, "scope": config["scope"]})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    imports_started = time.monotonic()
    print("MDB_FLUX_ATTENTION_REPLACEMENT_BOOTSTRAP " + json.dumps({"phase": "runtime_imports_start", "worker_seconds": imports_started - started}), flush=True)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor
    from mdb.tensor_archive import write_tensor_archive

    helper = runpy.run_path("flux_mlp_replacement.py")
    attention_helper = runpy.run_path("flux_attention_reference.py")
    mlp = runpy.run_path("flux_single_mlp.py")
    engine = runpy.run_path("flux_single_contexts.py")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("attention/MLP replacement requires admitted CUDA")
    imports_finished, sequence = time.monotonic(), 0
    save(output / "startup.json", {"runtime_import_seconds": imports_finished - imports_started,
        "worker_preimport_seconds": imports_started - started, "clock_scope": "Worker entry through runtime imports; bootstrap environment resolution is outside this clock"})
    counts = {key: 0 for key in COUNTS}
    counts["physical_batch"] = 1
    progress = {"phase": "checkpoint_inventory", "wall_calls": 0, "last_heartbeat": imports_finished}
    def wall():
        now = time.monotonic()
        if now - started > config["max_wall_seconds"]:
            raise RuntimeError("finite full attention/MLP replacement wall budget exhausted")
        progress["wall_calls"] += 1
        if now - progress["last_heartbeat"] >= 15:
            progress["last_heartbeat"] = now
            event("heartbeat", phase=progress["phase"], wall_calls=progress["wall_calls"])
    def event(kind, **fields):
        nonlocal sequence
        body = json_value({"sequence": sequence, "kind": kind, "seconds": time.monotonic() - started, "counts": dict(counts), **fields})
        sequence += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_FLUX_ATTENTION_REPLACEMENT_PHASE " + json.dumps(body, sort_keys=True), flush=True)
    arrays = {}
    def array(value):
        wall()
        descriptor = tensor_descriptor(value)
        key = identity({name: descriptor[name] for name in ("dtype", "shape", "sha256")})
        if key not in arrays:
            arrays[key] = write_tensor_archive(output / "arrays" / (key + ".npy"), value, root=output)
        return arrays[key]
    root = Path(config["model_path"])
    event("checkpoint_inventory_start")
    inventory = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("replacement checkpoint inventory differs")
    progress["phase"] = "model_loading"
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    if (not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling
            or len(pipe.transformer.transformer_blocks) != 5 or len(pipe.transformer.single_transformer_blocks) != 20):
        raise ValueError("requires deterministic five-joint/twenty-single distilled Klein")
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    implementations = {inspect.getfile(kind): sha(inspect.getfile(kind)) for kind in
        (type(pipe), type(pipe.text_encoder), type(pipe.transformer), type(pipe.transformer.single_transformer_blocks[6]),
         type(pipe.transformer.single_transformer_blocks[6].attn.processor), type(pipe.vae), type(pipe.scheduler), Flux2KleinAdapter)}
    event("pipeline_loaded")
    progress["phase"] = "encoder_placement"
    pipe.text_encoder.to("cuda")
    event("encoder_resident")
    progress["phase"] = "prompt_encoding"
    wall()
    with torch.inference_mode():
        conditioning, txt_ids = pipe.encode_prompt(CONTEXT["prompt"], device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
    counts["prompt_encodings"] += 1
    event("encoding_complete")
    encodings = {"conditioning": array(conditioning), "txt_ids": array(txt_ids)}
    conditioning, txt_ids = conditioning.cpu(), txt_ids.cpu()
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    event("encoder_released")
    progress["phase"] = "denoiser_vae_placement"
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    event("denoiser_vae_resident")
    progress["phase"] = "adapter_construction"
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    source = {"schema": "mdb-flux-attention-replacement-source-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"], "model_identity": adapter.model_identity, "checkpoint_sha256": identity(inventory),
        "checkpoint_files": inventory, "execution": dict(adapter.execution), "implementation_sources": implementations,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "diffusers", "transformers", "saturn-pub", "mdb-runtime")},
        "context": CONTEXT, "encoding": encodings, "encodings": encodings, "counts_after_encoding": dict(counts),
        "cpu_thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "blas_configuration": np.__config__.show(mode="dicts"), "scope": config["scope"]}
    source["attention_dispatch_diagnostics"]=attention_profile(pipe.transformer.single_transformer_blocks[6])
    event("prompt_encoded")
    progress["phase"] = "noise_schedule"
    event("noise_schedule_start")
    with torch.inference_mode():
        conditioning, txt_ids = conditioning.to("cuda"), txt_ids.to("cuda")
        latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512, conditioning.dtype,
            torch.device("cuda"), torch.Generator(device="cuda").manual_seed(CONTEXT["seed"]))
        counts["noise_draws"] += 1
        schedule = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1., .25, 4)
        timesteps, steps = retrieve_timesteps(pipe.scheduler, 4, torch.device("cuda"), sigmas=schedule,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4))
        if steps != 4:
            raise ValueError("replacement native schedule differs")
        event("noise_schedule_ready")
        progress["phase"] = "session_construction"
        event("session_start")
        session = adapter.session(latent=latent, conditioning=conditioning, img_ids=img_ids, txt_ids=txt_ids,
            timesteps=timesteps.clone(), sigmas=pipe.scheduler.sigmas.clone())
    event("session_ready")
    progress["phase"] = "source_builder"
    event("source_builder_start")
    cuts, builder = engine["build_sources"](adapter=adapter, session=session, output=output / "builder", context=CONTEXT,
        job_id=request["job_id"], wall=wall)
    for key in ("adapter_steps", "acts", "receipt_emissions"):
        counts[key] = builder["counts"][key]
    event("source_builder_complete")
    builder_ref = {"path": "builder/builder.json", "sha256": sha(output / "builder/builder.json")}
    progress["phase"] = "source_preparation"
    event("source_preparation_start")
    cuts, preparation = util["prepare_sources"](adapter=adapter, cuts=cuts, output=output / "preparation", builder_ref=builder_ref, wall=wall)
    for key in ("adapter_steps", "acts", "receipt_emissions"):
        counts[key] = builder["counts"][key] + preparation["counts"][key]
    preparation_ref = {"path": "preparation/preparation.json", "sha256": sha(output / "preparation/preparation.json")}
    source.update(builder=builder_ref, preparation=preparation_ref)
    save(output / "source.json", source)
    event("sources_prepared")
    progress["phase"] = "source_binding"
    block = pipe.transformer.single_transformer_blocks[6]
    binding, operands = bind_prediction_inputs(block, cuts, array=array, counts=counts,
        builder_ref=builder_ref, preparation_ref=preparation_ref, config=config)
    if binding["hidden_width"] != 3072 or binding["mlp_width"] != 9216 or binding["heads"] != 24 or binding["head_dim"] != 128:
        raise ValueError("qualified full attention/MLP width/head profile differs")
    if any(tuple(row["image"].shape)!=(1,1024,3072) or tuple(row["residual"].shape)!=(1,1536,3072)
            or row["text_rows"]!=512 for row in operands["cases"].values()):
        raise ValueError("qualified production attention token geometry differs")
    save(output / "binding.json", binding)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(binding),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "binding": binding},
        execution_scope="Observed pre6 state to complete image attention and MLP predictions, then native consumer substitution and nine native RGB futures; trusted callback dataflow and external CUDA census",
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000, "max_bytes": 512 << 20})
    release = {"native": False, "blocked": 0, "target_calls": 0}
    def guard(module, args):
        if not release["native"]:
            release["blocked"] += 1
            raise ValueError("native model entered before full attention/MLP forecast seal")
        if module is block:
            release["target_calls"] += 1
    handles = [module.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for module in component.modules()]
    try:
        with tx:
            tx.frontier("source_binding", binding)
            progress["phase"] = "full_field_prediction"
            event("prediction_start")
            tx.predict("attention_mlp_fields", lambda _: {"predictions": {"cases": {
                role: predict_fields(row, operands["parameters"], binding, helper=helper, attention_helper=attention_helper)
                for role, row in operands["cases"].items()}}})
            tx.predict("replacement_hypotheses", lambda _: {"predictions": config["hypotheses"]})
            seal = tx.seal()
            event("full_field_forecast_sealed", seal=seal, target_entries_before_seal=release["blocked"])
            release["native"] = True
            progress["phase"] = "native_panel"
            tx.execute("native", lambda context: run_panel(pipe=pipe, adapter=adapter, cuts=cuts, output=output, config=config,
                store=LocalStore(output / "typed-store"), array=array, wall=wall, event=event, counts=counts, helper=helper,
                capture=mlp["capture_source_calls"], predictions=context.predictions["attention_mlp_fields"]["cases"]))
    finally:
        for handle in handles:
            handle.remove()
    save(output / "transaction-report.json", tx.report())
    if counts != COUNTS or release["target_calls"] != 11 or release["blocked"] != 0:
        raise ValueError("completed full attention/MLP replacement physical census differs")
    summary = {"schema": "mdb-flux-attention-replacement-result-v1", "job_id": request["job_id"], "request_sha256": request["request_sha256"],
        "counts": dict(counts), "builder_counts": {"adapter_steps": 108, "acts": 4, "receipt_emissions": 112},
        "panel_counts": {"adapter_steps": 137, "acts": 7, "receipt_emissions": 144, "candidate_calls": 9, "images": 9, "vae_calls": 9},
        "seal": seal, "transaction_verification": verify_transaction(tx.root), "model_identity": adapter.model_identity,
        "control_observations": tx.observation("native")["control_observations"], "target_calls": release["target_calls"],
        "branch_execution_order": BRANCH_EXECUTION_ORDER, "observation_plan": OBSERVATION_PLAN,
        "continuation_observations_sha256": sha(output / "continuation-observations.json"),
        "attention_stage_observations_sha256": sha(output / "attention-stage-observations.json"),
        "worker_seconds": time.monotonic() - started, "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed", "scope": config["scope"]}
    save(output / "summary.json", summary)
    event("complete")
    progress["phase"] = "archive_export"
    # Diagnostic stderr must not interleave with stdout custody transport.
    faulthandler.cancel_dump_traceback_later()
    raw_output_bytes=sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if raw_output_bytes+(1<<20)>config["transport_limits"]["max_expanded_bytes"]:
        raise ValueError("pre-export output exceeds finite expanded custody budget")
    print("MDB_FLUX_ATTENTION_REPLACEMENT_ARCHIVE_START " + json.dumps({"raw_output_bytes":raw_output_bytes, "counts": counts}), flush=True)
    util["emit"](output, summary, config)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
        output = Path("flux-attention-replacement-result")
        if output.is_dir():
            util = runpy.run_path("flux_single_mlp_campaign.py")
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-flux-attention-replacement-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                "request_sha256": request["request_sha256"], "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            util["save"](output / "failure.json", failure)
            util["emit"](output, failure, request["config"])
        raise
