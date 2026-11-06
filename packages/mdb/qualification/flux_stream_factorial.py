"""Same-parent image/text factorial at original FLUX joint2 entry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil


def helper(name):
    return runpy.run_path(str(Path(__file__).with_name(name)))


def corner_state(native, predicted, corner):
    if corner not in {"00", "10", "01", "11"}:
        raise ValueError("corner must select image/text with two binary digits")
    return dict(native, image=predicted["image"] if corner[0] == "1" else native["image"],
                text=predicted["text"] if corner[1] == "1" else native["text"])


def decompose(values):
    import numpy as np
    a, b, c, d = [np.asarray(values[k], np.float64) for k in ("00", "10", "01", "11")]
    image, text = b-a, c-a
    interaction = d-b-c+a
    total = d-a
    return dict(image=image, text=text, interaction=interaction, total=total,
                closure=image+text+interaction-total)


def stage(run, previous):
    import numpy as np
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.parameter_artifacts import validate_manifest
    from mdb.forecast_transaction import load_forecast_tree
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    S = helper("flux_mlp_state_law.py")
    read, save, sha = S["read"], S["save"], S["sha"]
    assert read(previous / "audit-v1.json")["mechanics_status"] == "verified"
    source = previous / "collection-01/extracted/binding-minimal-result"
    prior = read(source / "request.json")
    helper("flux_context_transfer.py")["bind_environment"](prior["execution_environment"])
    recipes = validate_manifest(read(source / "parameter-artifacts.json"))
    recipes["artifacts"] = [r for r in recipes["artifacts"] if r["source"]["tensor"].startswith("transformer_blocks.2.")]
    assert len(recipes["artifacts"]) == 16
    store = LocalStore(source / "store")
    rows = {(r["branch"], r["case"], r["block"]):r for r in read(source / "rows.json")}
    tx = source / "transaction"
    predicted = load_forecast_tree(tx, read(tx / "forecasts/joint1.json")["predictions"])
    bits = helper("analyze_flux_attention_state_law.py")["bits"]
    bindings = {}
    for case in prior["config"]["cases"]:
        row = rows[("native", case, 1)]
        cut = store.load(row["result"])
        binding = dict(source_cut=cut.fingerprint, source_model=cut.model_identity,
            source_execution=dict(cut.execution), source_point=dict(cut.execution_point),
            native_payload=describe(cut.payload),
            predicted={s:dict(sha256=bits(predicted[case][s]), shape=list(predicted[case][s].shape), dtype="torch.bfloat16") for s in ("image", "text")},
            calibration_outputs={corner:rows[(branch,case,2)]["ports"] for corner,branch in (("00","native"),("11","predicted-input"))})
        bindings[case] = binding
    run.mkdir(parents=True, exist_ok=False)
    save(run / "source-bindings.json", bindings)
    save(run / "parameter-artifacts.json", recipes)
    shutil.copyfile(source / "retrieve_weights.py", run / "retrieve_weights.py")
    config = dict(schema="mdb-flux-stream-factorial-plan-v1",
        model="component:"+prior["config"]["checkpoint_sha256"]+":transformer_blocks.2:cpu-bf16-v1",
        checkpoint_sha256=prior["config"]["checkpoint_sha256"], model_path=prior["config"]["model_path"],
        checkpoint_file=prior["config"]["checkpoint_file"], checkpoint_file_sha256=prior["config"]["checkpoint_file_sha256"],
        output_scratch=prior["config"]["output_scratch"],
        source_root=str(Path(prior["config"]["output_scratch"])/read(source / "summary.json")["job_id"]/"binding-minimal-result"),
        source_job=read(source / "summary.json")["job_id"], source_request_sha256=prior["request_sha256"],
        source_audit_sha256=sha(previous / "audit-v1.json"), source_files_sha256=sha(source / "files.json"),
        source_seal_sha256=read(source / "summary.json")["seal"]["seal_sha256"],
        cases=prior["config"]["cases"], corners=["00", "10", "01", "11"],
        indices=prior["config"]["indices"], text_indices=prior["config"]["text_indices"],
        chunk_rows=64, cpu_threads=4, max_wall_seconds=720, directory_custody=True,
        expected_counts=dict(component_loads=1,numpy_block_forecasts=12,native_blocks=12,native_attention=12,
            native_image_ff=12,native_text_ff=12,native_linears=144,native_sdpa=12,
            forks=12,acts=15,receipts=27,full_pipeline_forwards=0,encodes=0,images=0,noise_draws=0),
        hypotheses=dict(frontier="Prior observed complete native joint1 state plus prior sealed predicted joint1 image/text. Conditional assay, not raw-entry chain prediction.",
            structural="Changing text leaves image normalization and positioned QKV exact; changing image leaves text normalization and positioned QKV exact. Attention can transmit differences to both receiving MLPs.",
            numerical="Seal full original-weight mathematical forecasts for all12corners before new native block calls; report endpoint and small-effect error independently.",
            interaction="Retain both single-stream effects and nonadditive interaction at each witnessed port and full exits; no assumption that contributions add or that one stream dominates.",
            status="10/01 are six new mixed internal states;00/11 are known calibration endpoints rerun in the same native numerical geometry. No new natural prompt or image."))
    save(run / "plan.json", config)
    here = Path(__file__).resolve().parent
    files = ["flux_stream_factorial.py", "flux_native_joint_ports.py", "flux_state_closure.py", "flux_attention_state_law.py",
             "flux_mlp_state_law.py", "flux_context_transfer.py", "flux_family_custody.py"]
    extras = [(here/n,n) for n in files]+[(run/n,n) for n in ("plan.json","source-bindings.json","parameter-artifacts.json","retrieve_weights.py")]
    pre = run / "preflight.v1.json"
    save(pre, dict(schema_version="mrun-preflight-config-v1", hashed_roots=["flux_stream_factorial_campaign.py",*[n for _,n in extras],pre.name],
                   path_token_allowlist=[config["model_path"],config["output_scratch"],config["source_root"],"binding-minimal-result"]))
    extras.append((pre, pre.name))
    raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request=dict(schema="mdb-flux-stream-factorial-request-v1",config=config,model=config["model"],
        backend="diffusers",device="cpu",dtype="bfloat16",needs={},batch_size=1,execution_environment=prior["execution_environment"],
        config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(here/"flux_stream_factorial_campaign.py","flux_stream_factorial_campaign.py"),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=27,logical_branches=12,
            logical_checkpoint_replays=12,max_local_work_items=27,
            checkpoint_policy="One original joint2 component;12full-state NumPy forecasts sealed before12native block calls;12same-parent forks15typedActs including3no-opshams; no encoders/VAE/pipeline/retries. Source inputs read from exact previous accepted host-local cuts/forecast seal."))
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run / "request.json",request)
    stage_worker(run/"payload",worker=here/"flux_stream_factorial_campaign.py",config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request["request_sha256"])


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    S=helper("flux_mlp_state_law.py");read,save,sha=[S[n] for n in ("read","save","sha")]
    request=read(run/"request.json")
    helper("flux_context_transfer.py")["bind_environment"](request["execution_environment"])
    unsigned={k:v for k,v in request.items() if k!="request_sha256"}
    assert hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest()==request["request_sha256"]
    if (run/"submission.json").exists() or any(sha(run/"payload"/n)!=h for n,h in request["source_hashes"].items()):
        raise ValueError("payload changed or already submitted")
    job=guarded_mrun_launch(worker_command("flux_stream_factorial_campaign.py"),request=request,model=request["model"],task="forward",
        experiment="mdb-flux-stream-factorial",payload=run/"payload",pin="beast",gpu=False,needs={},preflight=True,retry_on_kill=False,timeout_s=900,
        note="CPU BF16 original joint2 four-corner image/text restoration,3knowncontexts.12NumPy complete-state forecasts sealed before12native all1536query/key blocks;6new mixed states plus6known calibration corners.12MDBforks15Acts(3no-opshams),27receipts. Exact prior completed host-local StateCuts+forecast seal supply declared inputs; no large payload transfer, no target returns supplied to forecaster.0fullpipeline/encoder/VAE/noise.720worker900outer,no retries;auto component planning.")
    save(run/"submission.json",dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--previous",type=Path);p.add_argument("--submit",action="store_true")
    a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.previous.resolve()),sort_keys=True))
