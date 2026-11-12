"""Apply a frozen computation-edit recipe in a fresh native model process."""
from __future__ import annotations

import json
import os
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path

from wsp_auto_qwen import save, sha


def run(config, request):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from mdb.wsp import WeightStudy, verify_weight_study
    from mdb.wsp_discovery import _plain
    from mdb.wsp_selective import synthesize_low_rank_patch, virtual_low_rank_edit
    from mdb.wsp_weights import WeightSpaceError, parameter_inventory

    started = time.monotonic()
    root = Path(config["output_scratch"])/("wsp-program-replay-"+os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True,exist_ok=False)
    packet = Path("frozen-program.tar.gz")
    if sha(packet)!=config["frozen_program_sha256"]:
        raise WeightSpaceError("frozen source packet differs")
    with tarfile.open(packet) as stream:
        members = stream.getmembers()
        if len(members)>4096 or sum(m.size for m in members)>256*1024*1024:
            raise WeightSpaceError("source packet budget exceeded")
        if any(not (m.isfile() or m.isdir()) or Path(m.name).is_absolute() or ".." in Path(m.name).parts for m in members):
            raise WeightSpaceError("unsafe source packet")
        stream.extractall(root,filter="data")
    frozen = json.loads((root/"frozen-files.json").read_text())
    for name,ref in frozen.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha(root/path)!=ref["sha256"]:
            raise WeightSpaceError("frozen source file differs")
    save(root/"config.json",config)
    save(root/"request.json",request)
    read = lambda name:json.loads((root/name).read_text())
    inputs,selection,recipe,account = (read(name) for name in ("inputs.json","selection.json","recipe.json","computation-account.json"))
    for stage in ("capture","causal","discovery"):
        verify_weight_study(root/stage)
    torch.set_num_threads(config["cpu_threads"])
    checkpoint = Path(config["model_path"])
    if sha(checkpoint/"model.safetensors")!=config["checkpoint_sha256"]:
        raise WeightSpaceError("checkpoint differs")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2ForCausalLM.from_pretrained(checkpoint,local_files_only=True,dtype=torch.float32,
        attn_implementation="eager").cpu().eval().requires_grad_(False)
    original = parameter_inventory(model)
    save(root/"runtime.json",{"torch":str(torch.__version__),"numpy":np.__version__,
        "device":"cpu","dtype":"float32","generation":{"max_new_tokens":3,"do_sample":False,
            "repetition_penalty":1.,"use_cache":True},"execution":"fresh-process frozen recipe replay"})
    site = selection["site"]
    module = model.get_submodule(site)
    factors,patches = {},{}
    for family,c in selection["selected"].items():
        with np.load(root/"factors"/(c["key"]+".npz"),allow_pickle=False) as arrays:
            factors[family] = (arrays["left"].copy(),arrays["right"].copy())
        patch = synthesize_low_rank_patch(model,site,*factors[family],dose=c["dose"],
            model_identity={"checkpoint_sha256":config["checkpoint_sha256"]})
        expected = read("source-patch-manifests/"+family+".json")["manifest"]
        if patch.manifest!=expected:
            raise WeightSpaceError("recompiled weight bytes differ from the previously frozen candidate")
        patch.save(root/(family+"-patch"))
        patches[family] = patch
    old_weight = module.weight.detach().clone()
    old_bias = module.bias.detach().clone() if module.bias is not None else None
    capture = read("capture/observations/contrasts.json")
    from mdb.forecast_transaction import load_forecast_tree

    values = load_forecast_tree(root/"capture",capture["value"])
    direction = torch.tensor(np.asarray(values["result"]["observations"]["samples"][site]["desired"]).copy())
    task,controls,general = (inputs[k] for k in ("heldout","heldout_controls","heldout_preservation"))
    rows = task+controls
    encoded = tokenizer([r["prompt"]+r["original"][:2] for r in rows]+general,
        padding=True,return_tensors="pt",add_special_tokens=False)
    generation_inputs = tokenizer([r["prompt"] for r in rows],padding=True,return_tensors="pt",add_special_tokens=False)
    labels = torch.tensor([15+int(r["expected"][-1]) for r in rows])
    source_row = account["symbols"][recipe["symbol"]]["token_position"]
    anchor = None

    def evaluate():
        nonlocal anchor
        if time.monotonic()-started>config["max_wall_seconds"]:
            raise WeightSpaceError("replay wall budget exceeded")
        logits = model(**encoded,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        logp = logits.log_softmax(-1)
        n,nrows = len(task),len(rows)
        if anchor is None:
            anchor = logp[n:].detach().clone()
        kl = (anchor.exp()*(anchor-logp[n:])).sum(-1)
        target = logp[:nrows][torch.arange(nrows),labels]
        winners = logits[:nrows].argmax(-1)
        tokens = model.generate(**generation_inputs,max_new_tokens=3,do_sample=False,repetition_penalty=1.,
            use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        new = tokens[:,generation_inputs.input_ids.shape[1]:].tolist()
        expected = [[15+int(ch) for ch in r["expected"]] for r in rows]
        return {"observations":{"objective":float(target[:n].mean()-kl.mean()),
            "rewrite_correct":int((winners[:n]==labels[:n]).sum()),"task_rows":n,
            "control_correct":int((winners[n:]==labels[n:]).sum()),"control_rows":len(controls),
            "winner_ids":winners.tolist(),"target_ids":labels.tolist(),"target_logp":target.tolist(),
            "mean_rewrite_logp":float(target[:n].mean()),"preservation_kl":float(kl.mean()),
            "preservation_kl_rows":kl.tolist(),"logits":logits.detach().numpy().copy(),
            "generated_ids":new,"generated_text":tokenizer.batch_decode(new),"expected_ids":expected,
            "original_ids":[[15+int(ch) for ch in r["original"]] for r in rows],
            "generated_rewrite_correct":sum(x==y for x,y in zip(new[:n],expected[:n],strict=True)),
            "generated_control_correct":sum(x==y for x,y in zip(new[n:],expected[n:],strict=True))},
            "work":{"physical_forwards":1,"batch_size":len(logits),"generation_rows":nrows,"max_new_tokens":3},
            "state":{"policy":"restart","scoring":"two supplied unchanged answer digits",
                "generation":"fresh raw prefix; own tokens and native KV; no answer digits supplied"}}

    arms = ("native","semantic-program","compiled","factored","wrong-location","random","restored")
    binding = {"model":{"checkpoint_sha256":config["checkpoint_sha256"]},"checkpoint":config["checkpoint_sha256"],
        "execution":config["execution"],"consumer":"authored operand rewrite via native Qwen suffix",
        "horizon":"supplied-prefix final-digit score and own-state three-token answer"}
    outcomes = {}
    with WeightStudy(root/"heldout",model=model,binding=binding,input_identity={"task":task,"controls":controls,"general":general},
        source_identity={"frozen_program_sha256":config["frozen_program_sha256"],"recipe":recipe["identity"],"worker":sha(__file__)},
        budgets={"max_wall_ms":config["max_wall_seconds"]*1000,"max_bytes":512*1024*1024}) as study:
        for name in arms:
            patch = patches["semantic"] if name in {"compiled","restored"} else patches["random"] if name=="random" else None
            study.add_arm(name,role="native" if name=="native" else "restoration" if name=="restored" else "edited",
                patch=patch,state_policy="restart",symbol_refs=(recipe["identity"],),
                predictor=lambda ctx,name=name:{"predictions":{"arm":name,"request":config["program"],
                    "expected":recipe["predictions"],"candidate_frozen_before_all_heldout_release":True}})
        study.seal()
        for name in arms:
            def run_arm(ctx,name=name):
                handles = []
                if name=="restored":
                    handles.append(module.register_forward_hook(lambda layer,args,output:
                        torch.nn.functional.linear(args[0],old_weight,old_bias)))
                if name in {"semantic-program","wrong-location"}:
                    position = account["wrong_location"] if name=="wrong-location" else source_row
                    def hook(layer,args,output):
                        if output.shape[1]<=position:
                            return output
                        result = output.clone()
                        result[:len(task),position] += direction
                        return result
                    handles.append(module.register_forward_hook(hook))
                c = selection["selected"]["semantic"]
                manager = virtual_low_rank_edit(model,site,*factors["semantic"],dose=c["dose"]) if name=="factored" else nullcontext()
                try:
                    with manager:
                        return evaluate()
                finally:
                    for handle in handles:
                        handle.remove()
            study.run(name,run_arm)
            value = study.tx.observation(name)["result"]
            outcomes[name] = _plain({k:v for k,v in value.get("observations",{}).items() if k!="logits"})
            if "error" in value:
                outcomes[name]["error"] = _plain(value["error"])
            row = {"phase":"heldout","arm":name,"elapsed_s":time.monotonic()-started,
                "observations":outcomes[name]}
            with (root/"timeline.jsonl").open("a") as stream:
                stream.write(json.dumps(row,sort_keys=True)+"\n")
            print("WSP_PROGRAM_PROGRESS="+json.dumps(row,sort_keys=True),flush=True)
    summary = {"schema":"mdb-computation-program-qwen-replay-v1","job_id":os.environ["MRUN_JOB_ID"],
        "recipe":recipe,"selected":selection["selected"],"heldout":outcomes,
        "parameters_restored":parameter_inventory(model)==original,
        "heldout_verification":verify_weight_study(root/"heldout"),"elapsed_s":time.monotonic()-started,
        "source_job_id":config["source_job_id"],"frozen_program_sha256":config["frozen_program_sha256"],"terminal_status":"not-evaluated"}
    save(root/"summary.json",summary)
    save(root/"files.json",{p.relative_to(root).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size}
        for p in sorted(root.rglob("*")) if p.is_file()})
    archive = root.parent/(root.name+".tar.gz")
    with tarfile.open(archive,"w:gz") as stream:
        stream.add(root,arcname="evidence")
    print("WSP_PROGRAM_RESULT="+json.dumps({"summary":summary,"artifact":{"path":str(archive),"sha256":sha(archive),
        "bytes":archive.stat().st_size,"files_manifest_sha256":sha(root/"files.json"),"summary_sha256":sha(root/"summary.json")}},sort_keys=True),flush=True)


if __name__=="__main__":
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()),json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text()))
