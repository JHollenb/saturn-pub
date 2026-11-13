"""Prospective midpoint/native-profile span closure with branch-owned KV."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import second_order_support as base
import torch

from mdb_lanes.bench import BenchSession
from mdb_lanes.bench_circuit_discovery import _head_edit
from mdb_lanes.bench_circuit_measurement import _closure, _memory
from mdb_lanes.bench_hidden_memory import logical_address
from mdb_lanes.read_program import FrozenReadProgram
from mdb_lanes.read_weight_centered import (
    compile_centered_path,
    evaluate_centered_program,
    prepare_centered_path,
)
from mdb_lanes.read_weight_native import NativeQwenReadTransport
from mdb_lanes.read_weight_transport import SmoothQwenReadTransport

DOSES = (0.0, -0.25, 0.2, 0.6, 1.0)
METHODS = ("midpoint", "baseline-corrected", "zero-centered", "stale", "wrong-sign", "wrong-curvature")
CONSUMERS = ("midpoint", "baseline-corrected", "zero-centered", "stale", "functional-native")
save, error, refs = base.save, base.error, base.refs
previous = base.previous


class Runner(base.Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_layers = 0
        self.replaced_layers = 0

    def arm(self, name, parent, sites, last_token, target_ids, *, dose=1.0,
            edits=(), span=None, capture=True, count=3):
        bench = self.bench
        arm_started = time.perf_counter()
        bench.restore(parent)
        acts = []
        for port in edits:
            act, _, locality = _head_edit(bench, sites[port], 1.0-dose if port == "A" else 0.0)
            acts.append({"port": port, "act": act, "locality": locality})
        if name == "reinstall":
            for item in acts:
                site = sites[item["port"]]
                row = bench._tensor_load(item["locality"]["before_ref"])
                bench.memory_write({"mode": "replace", "selector": {
                    "layers": [site["layer"]], "components": ["value"],
                    "row_start": site["source_row"], "row_stop": site["source_row"]+1},
                    "values": {logical_address(site["layer"], "value", site["source_row"]): row.float().tolist()}})
        observer = base.Capture(bench, sites)
        trajectory, replacements = [], []
        runtime = bench.vm.runtime
        before_native, before_replaced = self.native_layers, self.replaced_layers
        with contextlib.ExitStack() as stack:
            if capture:
                stack.enter_context(observer.installed())
            advance = runtime.advance_token

            def bridge(execution):
                index = execution.next_layer
                if span is None or not observer.active or index not in (19, 20, 21, 22):
                    self.native_layers += 1
                    return advance(execution)
                binding, tape = span["binding"], span["tape"]
                if (binding["parent_cut"] != parent or binding["token_id"] != last_token
                        or execution.token_id != last_token or binding["sites"] != sites
                        or binding["length"] != execution.state.length or binding["layers"] != [19,20,21,22]):
                    raise RuntimeError("compiled span binding changed")
                declared_input = tape[f"input{index}"]
                if (execution.hidden.dtype != declared_input.dtype
                        or execution.hidden.device != declared_input.device
                        or not torch.equal(execution.hidden, declared_input)):
                    raise RuntimeError("compiled span consumed operand changed")
                observer.keep(f"consumed_input{index}", execution.hidden)
                for key, value in tape.items():
                    if key.endswith(str(index)):
                        observer.keep(key, value)
                key, value = tape[f"current_k{index}"], tape[f"current_v{index}"]
                delta_key, delta_value = execution.state.delta[index]
                row = execution.state.delta_rows
                for predicted, target in ((key,delta_key[:,:,row:row+1]),(value,delta_value[:,:,row:row+1])):
                    if (predicted.shape != target.shape or predicted.dtype != target.dtype
                            or predicted.device != target.device):
                        raise RuntimeError("compiled cache type or geometry changed")
                if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
                    raise RuntimeError("nonfinite compiled cache")
                delta_key[:, :, row:row+1].copy_(key)
                delta_value[:, :, row:row+1].copy_(value)
                execution.hidden = tape[f"exit{index}"].to(execution.hidden).clone()
                observer.keep(f"installed_exit{index}", execution.hidden)
                observer.keep(f"installed_current_k{index}", delta_key[:,:,row:row+1])
                observer.keep(f"installed_current_v{index}", delta_value[:,:,row:row+1])
                observer.keep(f"installed_k{index}", torch.cat((execution.state.ancestor[index][0],delta_key[:,:,:row+1]),dim=2))
                observer.keep(f"installed_v{index}", torch.cat((execution.state.ancestor[index][1],delta_value[:,:,:row+1]),dim=2))
                execution.next_layer += 1
                if execution.next_layer != index+1:
                    raise RuntimeError("compiled span cursor mismatch")
                self.replaced_layers += 1
                replacements.append({"layer": index, "position": execution.state.length,
                                     "token_id": execution.token_id, "next_layer": execution.next_layer})
                return tape[f"attention{index}"]

            runtime.advance_token = bridge
            stack.callback(setattr, runtime, "advance_token", advance)
            # Original modules in the replaced span must not execute at offset0.
            handles = []
            if span is not None:
                def refuse(_module, _args):
                    if observer.active:
                        raise RuntimeError("native module entered replaced span")
                for index in (19,20,21,22):
                    for module in bench.model.model.layers[index].modules():
                        handles.append(module.register_forward_pre_hook(refuse))
                for handle in handles:
                    stack.callback(handle.remove)
            for offset in range(count):
                result = self.step(last_token if offset == 0 else None)
                logits = bench.resolve_value("logits").float()[0]
                trajectory.append({"offset": offset, "authority": "caller-supplied" if offset == 0 else "native-greedy",
                    "consumed_token_id": result["token_id"], "next_token_id": result["next_token_id"],
                    "next_text": self.tokenizer.decode([result["next_token_id"]]), "cut": result["cut"],
                    "receipt_ref": result["receipt_ref"], "logits_sha256": result["logits_sha256"],
                    "target_margin": float(logits[target_ids[0]]-logits[target_ids[1]]),
                    "direct_object_token_match": result["next_token_id"] == target_ids[0] if offset == 0 else None})
                observer.active = False
        closure = _closure(bench, _memory(bench))["sha256"]
        sync(next(bench.model.parameters()).device)
        return {"name": name, "parent_cut": parent, "dose": dose, "acts": acts,
            "trajectory": trajectory, "capture": observer.refs, "end_cut": bench.branch.cut,
            "closure_sha256": closure,
            "wall_seconds_with_custody": time.perf_counter()-arm_started,
            "timing_scope": "restore, edits, native/compiled token execution, capture, receipts and final cache digest",
            "span_replacements": replacements, "native_layer_transitions": self.native_layers-before_native,
            "replaced_layer_transitions": self.replaced_layers-before_replaced}, observer.live


def prediction(program, dose, method, device):
    mapped = {"midpoint": "centered", "zero-centered": "centered"}.get(method, method)
    result = evaluate_centered_program(program, dose, method=mapped)
    result["method"] = method
    for key in ("pred_logpB", "logpB", "delta_logpB", "pB"):
        result[key] = torch.tensor(result[key], dtype=torch.float32, device=device).reshape(1,-1)
    return result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(path, center, device):
    # Both routes receive the SAME prepared immutable path. Direct includes
    # tensor -> CPU scalar conversion; compilation serializes its coefficients.
    def measure(fn):
        sync(device)
        started = time.perf_counter()
        fn()
        sync(device)
        return time.perf_counter()-started
    compiled = compile_centered_path(path, center=center)
    for dose in (0.2,0.6,1.):
        path(dose).detach().cpu().tolist()
    for dose in (0.2,0.6,1.):
        evaluate_centered_program(compiled,dose)
    compilation, rows = [], []
    for k in (1,5,32):
        doses = [(-.25 + 1.25*i/max(1,k-1)) if k > 1 else 1. for i in range(k)]
        direct, scalar, combined = [], [], []
        for repeat in range(3):
            def run_direct():
                return [path(d).detach().cpu().tolist() for d in doses]
            def run_scalar():
                return [evaluate_centered_program(compiled,d) for d in doses]
            def run_combined():
                started = time.perf_counter()
                program = compile_centered_path(path,center=center)
                compilation.append(time.perf_counter()-started)
                return [evaluate_centered_program(program,d) for d in doses]
            if repeat % 2:
                combined.append(measure(run_combined)); direct.append(measure(run_direct))
            else:
                direct.append(measure(run_direct)); combined.append(measure(run_combined))
            scalar.append(measure(run_scalar))
        cost = statistics.median(combined)
        rows.append({"K":k, "doses":doses, "direct_seconds":direct, "scalar_seconds":scalar,
            "compile_plus_scalar_seconds":combined,
            "compile_plus_scalar_median_seconds":cost, "direct_median_seconds":statistics.median(direct),
            "direct_over_compile_plus_scalar":statistics.median(direct)/cost})
    return {"compile_seconds":compilation, "rows":rows, "warmup":"one compile + 3 direct/scalar doses",
            "scope":"same prepared state and direction; CPU scalar results; no baseline preparation or consumer suffix",
            "native_model_calls":0}


def complete19(tape, zero, sites, dose):
    tape = dict(tape)
    for prefix in ("q","k","scores","weights","cos","sin"):
        tape[f"{prefix}19"] = zero[f"{prefix}19"]
    values = zero["v19"].clone()
    values[:, sites["A"]["kv_head"], sites["A"]["source_row"]] *= 1.0-dose
    tape["v19"] = values
    tape["current_k19"] = zero["k19"][:, :, -1:]
    tape["current_v19"] = zero["v19"][:, :, -1:]
    return tape


def make_span(parent, sites, last_token, tape):
    return {"binding":{"parent_cut":parent, "token_id":last_token,
                       "length":int(tape["k19"].shape[2])-1,
                       "layers":[19,20,21,22], "sites":sites}, "tape":tape}


@torch.inference_mode()
def bundle(runner,parent,sites,last_token,target_ids):
    bench, model = runner.bench, runner.bench.model
    arms, live = {}, {}
    for name, edits, capture in (("00",(),True),("sham",(),False),("reinstall",("A","B"),False)):
        arms[name], live[name] = runner.arm(name,parent,sites,last_token,target_ids,edits=edits,capture=capture)
    for control in ("sham","reinstall"):
        if (previous.signature(arms[control]) != previous.signature(arms["00"])
                or arms[control]["closure_sha256"] != arms["00"]["closure_sha256"]):
            raise RuntimeError(control+" failed native closure")
    zero = live["00"]
    device = zero["q22"].device
    source_started=time.perf_counter()
    with base.forbid_native(model) as guards:
        sources={dose:base.local_source_attention(model,zero,sites["A"],dose) for dose in DOSES}
        if not torch.equal(sources[0.],zero["attention19"]):
            raise RuntimeError("local preceding write baseline mismatch")
        sync(device); source_seconds=time.perf_counter()-source_started
        write=sources[1.].float()-zero["attention19"].float()
        setup_started=time.perf_counter()
        with torch.inference_mode(False), torch.enable_grad():
            smooth=SmoothQwenReadTransport(model,incoming=zero["input19"],baseline_attention=zero["attention19"],
                history={i:(zero[f"k{i}"][:,:,:-1],zero[f"v{i}"][:,:,:-1]) for i in (20,21)},
                cos=zero["cos22"],sin=zero["sin22"])
            path=prepare_centered_path(smooth,zero["q22"],zero["k22"],zero["scores22"],write,
                kv_head=sites["B"]["kv_head"],source_row=sites["B"]["source_row"],
                scaling=model.model.layers[22].self_attn.scaling)
            sync(device); prepare_seconds=time.perf_counter()-setup_started
            compile_started=time.perf_counter()
            zero_program=compile_centered_path(path,center=0.)
            midpoint_program=compile_centered_path(path,center=.5)
            sync(device); compile_seconds=time.perf_counter()-compile_started
        forecasts={str(d):{m:prediction(zero_program if m=="zero-centered" else midpoint_program,d,m,device)
                          for m in METHODS} for d in DOSES}
        native_started=time.perf_counter()
        native=NativeQwenReadTransport(model,incoming=zero["input19"],
            history={i:(zero[f"k{i}"][:,:,:-1],zero[f"v{i}"][:,:,:-1]) for i in (20,21,22)},
            cos=zero["cos22"],sin=zero["sin22"])
        sync(device); native_setup_seconds=time.perf_counter()-native_started
        native_started=time.perf_counter()
        predicted_A={str(d):complete19(native.evaluate(sources[d]),zero,sites,d) for d in DOSES}
        predicted_AB=complete19(native.evaluate(sources[1.],value_edit={
            "kv_head":sites["B"]["kv_head"],"source_row":sites["B"]["source_row"],"scale":0.}),zero,sites,1.)
        sync(device); native_seconds=time.perf_counter()-native_started
        # All consumer operands come from prospective functional A, never native A.
        a=predicted_A["1.0"]
        groups=model.config.num_attention_heads//model.config.num_key_value_heads
        first_head=sites["B"]["kv_head"]*groups
        row=sites["B"]["source_row"]
        value=zero["v22"][0,sites["B"]["kv_head"],row]
        projection=model.model.layers[22].self_attn.o_proj
        width=model.model.layers[22].self_attn.head_dim
        vectors=torch.stack([projection.weight[:,h*width:(h+1)*width].float()@value.float()
                             for h in range(first_head,first_head+groups)])
        spans={}
        for method in CONSUMERS:
            if method=="functional-native":
                tape=predicted_AB
            else:
                weights=forecasts["1.0"][method]["pB"].reshape(-1).to(a["weights22"].dtype).float()
                attention=(a["attention22"].float()-(weights[:,None]*vectors).sum(0)).to(a["attention22"].dtype)
                mlp,exit_state=previous.predict_exit(model.model.layers[22],a["input22"],attention)
                tape={**a,"attention22":attention,"mlp22":mlp,"exit22":exit_state}
            spans[method]=make_span(parent,sites,last_token,tape)
        zero_span=make_span(parent,sites,last_token,predicted_A["0.0"])
        with torch.inference_mode(False), torch.enable_grad():
            costs=benchmark(path,.5,device)
    record={"schema":"mdb-read-weight-closure-forecast-v1","parent_cut":parent,"sites":sites,
        "visibility":"native00 tape and parameters, local earlier write; no edited receiver outputs",
        "baseline_capture":arms["00"]["capture"],"source_attention":refs(bench,sources),
        "write_ref":bench._tensor_ref(write),"midpoint_program":midpoint_program,"zero_program":zero_program,
        "forecasts":refs(bench,forecasts),"native_predictions":refs(bench,predicted_A),
        "functional_AB":refs(bench,predicted_AB),"consumer_spans":refs(bench,spans),
        "zero_span":refs(bench,zero_span),"native_module_entries":guards["count"],
        "physical_steps_at_seal":runner.token_steps,
        "costs":{"source_seconds":source_seconds,"prepare_seconds":prepare_seconds,
            "two_compiles_seconds":compile_seconds,"native_profile_setup_seconds":native_setup_seconds,
            "six_native_profile_spans_seconds":native_seconds,
            "native_profile_linears_per_span":24,"native_profile_spans":6,
            "extra_consumer_mlp_linears":12,"same_state_benchmark":costs}}
    ref=bench.artifacts.put(record)
    runner.events.append({"kind":"all-predictions-sealed","ref":ref,"physical_steps":runner.token_steps})
    # Paid smooth diagnostic follows seal, and never supplies a candidate.
    with base.forbid_native(model),torch.inference_mode(False),torch.enable_grad():
        smooth_control={str(d):path(d).detach() for d in DOSES}
    arms["functional-zero"],live["functional-zero"]=runner.arm("functional-zero",parent,sites,last_token,target_ids,dose=0.,span=zero_span)
    for dose in DOSES[1:]:
        runner.events.append({"kind":"edited-receiver-started","forecast_ref":ref,"dose":dose,"physical_steps":runner.token_steps})
        arms[f"A-{dose}"],live[f"A-{dose}"]=runner.arm(f"A-{dose}",parent,sites,last_token,target_ids,
            dose=dose,edits=("A",),count=3 if dose==1. else 1)
    for method in CONSUMERS:
        arms[method],live[method]=runner.arm(method,parent,sites,last_token,target_ids,
            edits=("A","B"),span=spans[method])
    runner.events.append({"kind":"native-AB-started","forecast_ref":ref,"physical_steps":runner.token_steps})
    arms["AB"],live["AB"]=runner.arm("AB",parent,sites,last_token,target_ids,edits=("A","B"))
    groups=model.config.num_attention_heads//model.config.num_key_value_heads
    selected=slice(sites["B"]["kv_head"]*groups,(sites["B"]["kv_head"]+1)*groups)
    row=sites["B"]["source_row"]
    baseline_logp=torch.log_softmax(zero["scores22"].float(),-1)[0,selected,0,row]
    dose_rows=[]
    for dose in DOSES:
        actual=zero if dose==0. else live[f"A-{dose}"]
        native_logp=torch.log_softmax(actual["scores22"].float(),-1)[0,selected,0,row]
        control=smooth_control[str(dose)].reshape(1,-1)
        methods={}
        for method in METHODS:
            metrics=base.weight_error(forecasts[str(dose)][method],native_logp,baseline_logp)
            residual=forecasts[str(dose)][method]["logpB"].reshape(-1).double()-control.reshape(-1).double()
            profile=control.reshape(-1).double()-native_logp.double()
            metrics["smooth_remainder_rmse"]=float(residual.square().mean().sqrt())
            metrics["smooth_native_profile_rmse"]=float(profile.square().mean().sqrt())
            metrics["mse_cross_term"]=float(2*(residual*profile).mean())
            total=forecasts[str(dose)][method]["logpB"].reshape(-1).double()-native_logp.double()
            metrics["fp64_total_rmse"]=float(total.square().mean().sqrt())
            metrics["fp64_identity_max_abs"]=float((total-residual-profile).abs().max())
            metrics["fp64_mse_identity_residual"]=float(total.square().mean()-(residual.square().mean()+profile.square().mean()+2*(residual*profile).mean()))
            methods[method]=metrics
        profile_fields={}
        for key,predicted in predicted_A[str(dose)].items():
            if key in actual and isinstance(predicted,torch.Tensor) and predicted.shape==actual[key].shape:
                profile_fields[key]=error(predicted,actual[key])
        dense_logp=torch.log_softmax(predicted_A[str(dose)]["scores22"].float(),-1)[0,selected,0,row]
        dose_rows.append({"dose":dose,"methods":methods,"native_capture":arms["00" if dose==0. else f"A-{dose}"]["capture"],
            "functional_profile_errors":profile_fields,"functional_logp_rmse":float((dense_logp-native_logp).square().mean().sqrt()),
            "local_source_error":error(sources[dose],actual["attention19"]),
            "smooth_logp_ref":bench._tensor_ref(control)})
    local={}
    for method in CONSUMERS:
        tape=spans[method]["tape"]
        local[method]={key:error(tape[nativekey],live["AB"][nativekey]) for key,nativekey in (
            ("attention","attention22"),("mlp","mlp22"),("exit","exit22"),("input","input22"))}
        local[method]["injected_exit_exact"]=torch.equal(tape["exit22"],live[method]["installed_exit22"])
        local[method]["injected_current_k_exact"]=torch.equal(tape["current_k22"],live[method]["installed_current_k22"])
    zero_closure=(previous.signature(arms["functional-zero"])==previous.signature(arms["00"])
                  and arms["functional-zero"]["closure_sha256"]==arms["00"]["closure_sha256"])
    result={"parent_cut":parent,"sites":sites,"arms":arms,"forecast_ref":ref,"forecast":record,
        "dose_rows":dose_rows,"local_errors":local,
        "future_comparisons":{m:previous.compare_arms(bench,arms[m],arms["AB"]) for m in CONSUMERS},
        "functional_zero_closure_exact":zero_closure,
        "functional_AB_native_closure_exact":arms["functional-native"]["closure_sha256"]==arms["AB"]["closure_sha256"],
        "functional_AB_full_logit_signature_exact":previous.signature(arms["functional-native"])==previous.signature(arms["AB"]),
        "native_A_assisted":False,"mechanics_status":"valid","terminal_status":"not-assessed"}
    bench.restore(parent)
    return result


def fresh_cases():
    rows=[]
    for a,b,x,y,owner,split in (
        ("Nora","Ethan","pen","hat","Ethan","development"),
        ("Ivy","Owen","ball","watch","Ivy","development"),
        ("Clara","Leo","ring","box","Leo","fresh"),
        ("Maya","Theo","pen","cup","Maya","fresh"),
    ):
        rows.append({"id":f"{a.lower()}-{owner.lower()}","owners":[a,b],"objects":[x,y],
            "previous_owner":owner,"split":split,
            "prompt":f"{a} placed the {x} in the box. {b} placed the {y} in the bag. The object {owner} placed is the",
            "followups":[f" The object {a} placed is the",f" The object {b} placed is the"],
            "previous_current_owner_relation":"same" if owner==a else "other"})
    return rows


@torch.inference_mode()
def run_case(model,tokenizer,spec,weights_sha,case,root,budget):
    ids=tokenizer.encode(case["prompt"],add_special_tokens=False)
    copula=tokenizer.encode(" is",add_special_tokens=False)
    objects=[tokenizer.encode(" "+word,add_special_tokens=False) for word in case["objects"]]
    if len(copula)!=1 or any(len(x)!=1 for x in objects):
        raise RuntimeError("single-token locator/object instrument inapplicable")
    targets=[x[0] for x in objects]
    program=FrozenReadProgram.standard()
    bench=BenchSession.from_resident(model,tokenizer,root=root/case["id"],model_id=spec.model_id,
        revision=spec.revision,model_sha256=weights_sha,session_id=case["id"],
        attention_backend="native-eager",continuation_capacity=128)
    runner=Runner(bench,tokenizer,budget)
    try:
        bench.prefill(ids[:-1]); runner.step(ids[-1])
        answer_span=(len(bench.branch.token_ids),len(bench.branch.token_ids)+2)
        initial=[runner.step(),runner.step()]
        followup=runner.append(case["followups"][0])
        sites=previous.resolve_sites(program,bench,answer_span,followup,copula[0])
        first=bundle(runner,followup["parent_cut"],sites,followup["last_token"],targets)
        save(root/(case["id"]+"-first.json"),first)
        world=first["arms"]["midpoint"]
        bench.restore(world["end_cut"])
        span2=(len(bench.branch.token_ids)-2,len(bench.branch.token_ids))
        followup2=runner.append(case["followups"][1])
        sites2=previous.resolve_sites(program,bench,span2,followup2,copula[0])
        second=bundle(runner,followup2["parent_cut"],sites2,followup2["last_token"],targets[::-1])
        return {"case":case,"model_identity":bench.identity,"program":program.to_dict(),
            "initial_answer_span":answer_span,"initial_native":initial,"first":first,
            "second_candidate_world":second,"second_answer_span":span2,"second_world_source_cut":world["end_cut"],
            "physical_token_steps":runner.token_steps,
            "native_layer_transitions":runner.token_steps*model.config.num_hidden_layers-runner.replaced_layers,
            "observed_native_layer_transitions_in_arms":runner.native_layers,
            "replaced_layer_transitions":runner.replaced_layers,"prediction_events":runner.events,
            "semantic_cot_qualified":False}
    finally:
        bench.close()


def main():
    import qualification_support as support
    support.MAX_ARCHIVE_FILES=262144
    support.MAX_EXPANDED_BYTES=2<<30
    support.MAX_ARCHIVE_BYTES=512<<20
    support.REPORT_MARKER="MDB_READ_WEIGHT_CLOSURE_REPORT_V1 "
    support.ARCHIVE_MARKER="MDB_READ_WEIGHT_CLOSURE_ARCHIVE_V1 "
    raw=Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    request=support._read(Path(os.environ["SATURN_DEBUG_REQUEST"]))
    config=json.loads(raw)
    if (support._sha(raw)!=request["config_bytes_sha256"] or request["config"]!=config
            or support._request_identity(request)!=request["request_sha256"]):
        raise RuntimeError("request identity changed")
    if config["schema"]!="mdb-read-weight-closure-plan-v1" or config["cases"]!=fresh_cases() or config["doses"]!=list(DOSES):
        raise RuntimeError("frozen closure plan changed")
    job_id=os.environ["MRUN_JOB_ID"]; root=Path(config["output_root"])/job_id
    root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config["parameter_manifest"],root/"parameter-artifacts.json")
    shutil.copyfile(config["retrieval_script"],root/"retrieve_weights.py")
    parameter=support._parameter_identity(config,root/"parameter-artifacts.json")
    spec,model,tokenizer,weights,weight_files,content_sha=support._load_model(config,parameter)
    started=time.monotonic();cases=[];budget={"used":0,"limit":config["max_native_token_transitions"]}
    try:
        for case in config["cases"]:
            result=run_case(model,tokenizer,spec,weights,case,root,budget)
            save(root/(case["id"]+".json"),result);cases.append(result)
            print("MDB_READ_WEIGHT_CLOSURE_CASE "+json.dumps({"id":case["id"],"physical_token_steps":result["physical_token_steps"]}),flush=True)
    except BaseException as failure:
        support._emit_partial_custody(root,request,config,cases,failure,parameter=parameter,
            model_spec=spec,weight_files=weight_files,weight_content_sha=content_sha)
        raise
    report={"schema":"mdb-read-weight-closure-report-v1","job_id":job_id,
        "request_sha256":request["request_sha256"],"source_hashes":request["source_hashes"],
        "model":spec.to_dict(),"weights_sha256":weights,"weight_files":weight_files,
        "weight_content_sha256":content_sha,"parameter_identity":parameter,"cases":cases,
        "elapsed_seconds":time.monotonic()-started,"physical_token_steps":budget["used"],
        "native_layer_transitions":sum(c["native_layer_transitions"] for c in cases),
        "replaced_layer_transitions":sum(c["replaced_layer_transitions"] for c in cases),
        "prefill_calls":len(cases),"prefill_token_count":sum(len(tokenizer.encode(c["prompt"],add_special_tokens=False))-1 for c in config["cases"]),
        "receiver_count":2*len(cases),"native_A_output_assistance":False,
        "consumer_scope":"prospective predicted states/currentKV19–22 replace native span on offset0; prefix/suffix and later tokens native",
        "cost_scope":"per-state dense midpoint compilation and functional-native producer remain paid; same-state dose amortization only",
        "mechanics_status":"valid","trend_status":"all-rows-retained","terminal_status":"not-assessed",
        "economical_full_query_qualified":False,"semantic_cot_qualified":False}
    manifest=support._write_output_manifest(root)
    archive,sha,size=support._archive_results(root)
    report["artifact_custody"]={"schema":"mdb-read-weight-closure-custody-v1","output_root":str(root),
        "manifest":"output-manifest.json","manifest_sha256":support._file_sha(root/"output-manifest.json"),
        "archive":archive.name,"archive_sha256":sha,"archive_bytes":size,
        "expanded_bytes":manifest["expanded_bytes"],"file_count":manifest["file_count"],"pretrained_weight_copies":False}
    save(root/"report.json",report);support._emit_results(root,report)


if __name__=="__main__":
    main()

