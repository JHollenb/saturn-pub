"""Installed public saved-array Qwen audit mechanics; synthetic inputs only."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import runpy
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/audit_flux_functional_qwen_prefix.py"))


def test_signed_complete_field_errors_tails_and_common_native_denominator():
    p = np.array([[[2., -4.], [1., 0.]]], np.float32)
    n = np.array([[[1., -2.], [1., 0.]]], np.float32)
    row = A["compare"](p, n)
    assert row["signed_mean"] == -.25
    assert row["native_signed_mean"] == 0
    assert row["changed_values"] == 2 and row["maxabs"] == 2
    assert row["worst"] == dict(coordinate=[0, 0, 1], predicted=-4., native=-2., signed_error=-2.)
    assert row["relative_rms"] == pytest.approx(np.sqrt(5 / 6))
    assert row["p99"] >= row["p95"]
    assert row["orientation"] == "functional-minus-native"


def test_signed_prompt_response_countertrend_and_zero_denominator():
    p, n = np.array([2., -2.], np.float32), np.array([-1., 1.], np.float32)
    zero = np.zeros(2, np.float32)
    row = A["compare"](p, n, predicted_baseline=zero, native_baseline=zero)
    assert row["cosine"] == -1 and row["dot"] == -4
    assert row["rms"] == 3 and row["native_rms"] == 1
    assert row["relative_rms"] == 3
    row = A["compare"](p, zero)
    assert row["relative_rms"] is None and row["cosine"] is None


def test_bitwise_signed_zero_and_invalid_geometry():
    row = A["compare"](np.array([0., -0.], np.float32), np.zeros(2, np.float32))
    assert row["rms"] == 0 and row["changed_values"] == 0 and row["bitwise_exact"] is False
    with pytest.raises(ValueError, match="geometry"):
        A["compare"](np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="nonfinite"):
        A["compare"](np.array([np.nan]), np.array([0.]))


def test_padded_queries_remain_in_full_account():
    native = np.ones((1, 4, 2), np.float32)
    predicted = native.copy()
    predicted[:, 2:] += 2
    row = A["row_accounts"](predicted, native, np.array([[1, 1, 0, 0]], np.int64))
    assert row["full"]["sample_count"] == 8 and row["full"]["changed_values"] == 4
    assert row["attended_query_rows"]["comparison"]["rms"] == 0
    assert row["padded_query_rows"]["comparison"]["rms"] == 2
    assert row["padded_query_rows"]["rows"] == 2


def test_original_unequal_query_width_and_reference_only_recipes():
    config = dict(hidden_size=12, intermediate_size=20, num_attention_heads=4, num_key_value_heads=2, head_dim=4, num_hidden_layers=4, vocab_size=19)
    suffixes = {"input_layernorm.weight": [12], "post_attention_layernorm.weight": [12],
        "self_attn.q_norm.weight": [4], "self_attn.k_norm.weight": [4], "self_attn.q_proj.weight": [16, 12],
        "self_attn.k_proj.weight": [8, 12], "self_attn.v_proj.weight": [8, 12], "self_attn.o_proj.weight": [12, 16],
        "mlp.gate_proj.weight": [20, 12], "mlp.up_proj.weight": [20, 12], "mlp.down_proj.weight": [12, 20]}
    shapes = {"model.embed_tokens.weight": [19, 12], "model.norm.weight": [12]}
    shapes.update({f"model.layers.{i}."+suffix: shape for i in range(4) for suffix, shape in suffixes.items()})
    manifest = dict(artifacts=[dict(source=dict(tensor=name, shape=shape, dtype="BF16")) for name, shape in shapes.items()])
    roles = A["parameter_roles"](manifest, config, 3)
    assert roles["hidden_width"] == 12 and roles["query_projection_width"] == 16
    assert roles["counts"] == dict(all=46, forecast=34, reference_only=12)
    assert "model.norm.weight" in roles["reference_only_parameters"]
    assert "model.layers.3.self_attn.q_proj.weight" not in roles["forecast_parameters"]
    manifest["artifacts"][2]["source"]["shape"] = [999]
    with pytest.raises(ValueError, match="shape"):
        A["parameter_roles"](manifest, config, 3)


def array_fixture():
    cases = {}
    for case, offset in (("prompt0", 0), ("prompt1", 2)):
        hidden = {i: np.arange(12, dtype=np.float32).reshape(1, 3, 4)+i+offset for i in range(4)}
        e = np.concatenate([hidden[i] for i in (1, 2, 3)], -1)
        projected = e[..., :2].copy()
        cases[case] = dict(forecast=dict(hidden_states={i: value.copy() for i, value in hidden.items()}, conditioning=e.copy(), projected=projected.copy()),
            native=dict(hidden_states=hidden, conditioning=e), consumers={name: dict(input=e.copy(), output=projected.copy()) for name in ("predicted", "reference")})
    return cases, {name: np.array([[1, 1, 0]], np.int64) for name in cases}


def test_full_own_concat_consumer_join_and_numerical_miss_remains_observation():
    cases, masks = array_fixture()
    p = cases["prompt1"]["forecast"]
    p["hidden_states"][2][0, 2, 3] += 1
    p["conditioning"] = np.concatenate([p["hidden_states"][i] for i in (1, 2, 3)], -1)
    cases["prompt1"]["consumers"]["predicted"]["input"] = p["conditioning"].copy()
    report = A["audit_arrays"](cases, masks, case_order=["prompt0", "prompt1"], prefix_layers=3, selected_hidden_states=[1, 2, 3])
    assert report["first_value_discrepancy"] == dict(prompt0=None, prompt1="hidden.2")
    assert report["cases"]["prompt1"]["fields"]["hidden.2"]["full"]["changed_values"] == 1
    assert report["cases"]["prompt1"]["fields"]["hidden.2"]["padded_query_rows"]["comparison"]["changed_values"] == 1
    assert report["signed_prompt_response"]["hidden.2"]["changed_values"] == 1
    cases["prompt1"]["consumers"]["predicted"]["input"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="actual context"):
        A["audit_arrays"](cases, masks, case_order=["prompt0", "prompt1"], prefix_layers=3, selected_hidden_states=[1, 2, 3])


def test_hidden_concat_refuses_final_norm_or_borrowed_state():
    cases, masks = array_fixture()
    cases["prompt1"]["forecast"]["conditioning"][..., -4:] *= .5
    with pytest.raises(ValueError, match="own interior"):
        A["audit_arrays"](cases, masks, case_order=["prompt0", "prompt1"], prefix_layers=3, selected_hidden_states=[1, 2, 3])


def checkpoint(root, name, shapes):
    import torch
    from safetensors.torch import save_file

    root.mkdir(exist_ok=True, parents=True)
    path = root/name
    values = {key: torch.full(shape, (i%7+1)/8, dtype=torch.bfloat16) for i,(key,shape) in enumerate(shapes.items())}
    save_file(values,path)
    with path.open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q",stream.read(8))[0]))
    rows = []
    for key,value in values.items():
        bits = value.view(torch.uint16).numpy().astype("<u2")
        raw, encoded = bits.tobytes(), io.BytesIO()
        np.save(encoded,bits,allow_pickle=False)
        archive = encoded.getvalue()
        rows.append(dict(path="parameters/"+hashlib.sha256(key.encode()+archive).hexdigest()+".npy",bytes=len(archive),sha256=hashlib.sha256(archive).hexdigest(),
            npy_prefix_b64=base64.b64encode(archive[:-len(raw)]).decode(),source=dict(file=name,file_sha256=A["sha"](path),tensor=key,dtype="BF16",shape=list(value.shape),
                data_offsets=header[key]["data_offsets"],payload_sha256=hashlib.sha256(raw).hexdigest())))
    return dict(schema="mdb-parameter-artifacts-v1",artifacts=rows)


def saved_fixture(root, *, numerical_miss=False, native_binding="runtime-observed-on-CUDA-host"):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoTokenizer, Qwen2TokenizerFast

    import mdb_lanes.parameter_artifacts as api
    from mdb_lanes.forecast_transaction import ForecastPayload, ForecastTransaction

    raw = root/"result"
    raw.mkdir(parents=True)
    tokenizer_dir = raw/"source/tokenizer"
    tokenizer_dir.mkdir(parents=True)
    vocab = {word:i for i,word in enumerate(["[PAD]","[UNK]","A","yellow","cone","left","green","cylinder","violet","torus","above","copper","pyramid","assistant"])}
    backend = Tokenizer(WordLevel(vocab,unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend,pad_token="[PAD]",unk_token="[UNK]")
    tokenizer.padding_side = "right"
    tokenizer.chat_template = "{{ messages[0]['content'] }}{% if add_generation_prompt %} assistant{% endif %}"
    tokenizer.save_pretrained(tokenizer_dir)
    # Use the persisted parser, exactly as the real worker and auditor do.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,padding_side="right")
    spec = dict(hidden_size=12,intermediate_size=20,num_attention_heads=4,num_key_value_heads=2,head_dim=4,num_hidden_layers=4,vocab_size=len(vocab))
    suffix = {"input_layernorm.weight":[12],"post_attention_layernorm.weight":[12],"self_attn.q_norm.weight":[4],"self_attn.k_norm.weight":[4],
        "self_attn.q_proj.weight":[16,12],"self_attn.k_proj.weight":[8,12],"self_attn.v_proj.weight":[8,12],"self_attn.o_proj.weight":[12,16],
        "mlp.gate_proj.weight":[20,12],"mlp.up_proj.weight":[20,12],"mlp.down_proj.weight":[12,20]}
    shapes = {"model.embed_tokens.weight":[len(vocab),12],"model.norm.weight":[12]}
    shapes.update({f"model.layers.{i}."+key:shape for i in range(4) for key,shape in suffix.items()})
    qm = checkpoint(root/"qwen-checkpoint","qwen.safetensors",shapes)
    fm = checkpoint(root/"flux-checkpoint","flux.safetensors",{"context_embedder.weight":[6,36]})
    write = lambda path,value: Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
    write(raw/"qwen-config.json",spec)
    write(raw/"qwen-parameter-artifacts.json",qm)
    write(raw/"flux-transformer-config.json",dict(joint_attention_dim=36,inner_dim=6))
    write(raw/"flux-parameter-artifacts.json",fm)
    shutil.copyfile(api.__file__,raw/"retrieve_qwen_weights.py")
    shutil.copyfile(api.__file__,raw/"retrieve_flux_weights.py")
    binding = dict(source_files={path.relative_to(raw/"source").as_posix():dict(path=str(path),sha256=A["sha"](path),bytes=path.stat().st_size) for path in tokenizer_dir.rglob("*") if path.is_file()})
    write(raw/"qwen-source-binding.json",binding)
    source_inventory = {"source/"+name:dict(sha256=row["sha256"],bytes=row["bytes"]) for name,row in binding["source_files"].items()}
    write(raw/"source-inventory.json",source_inventory)
    config = dict(schema="mdb-qwen-prefix-plan-v1",profile="tiny-cpu",cohort="prospective-new",prompts=list(A["PROMPTS"]),
        batch_size=1,padding="max_length",padding_side="right",truncation=True,all_512_rows_retained=True,prefix_layers=3,full_layers=4,selected_hidden_states=[1,2,3],max_length=512,
        qwen_parameter_root=str(root/"qwen-checkpoint"),flux_parameter_root=str(root/"flux-checkpoint"),
        attention_dispatch=dict(backend="sdpa",gqa_policy="repeat_kv",source="synthetic inspected source"),rope_buffer_policy=dict(init_device="cpu",init_dtype="float32",placement_dtype="bfloat16",source="transformers-qwen3-rotary-embedding-v1"))
    for key,name in {"qwen_config":"qwen-config.json","qwen_manifest":"qwen-parameter-artifacts.json","qwen_source_binding":"qwen-source-binding.json","qwen_retrieval_script":"retrieve_qwen_weights.py",
                     "flux_config":"flux-transformer-config.json","flux_manifest":"flux-parameter-artifacts.json","flux_retrieval_script":"retrieve_flux_weights.py"}.items():
        config[key],config[key+"_sha256"] = name,A["sha"](raw/name)
    retained = {key:dict(path=config[key],sha256=config[key+"_sha256"],bytes=(raw/config[key]).stat().st_size) for key in
        ("qwen_config","qwen_manifest","qwen_source_binding","qwen_retrieval_script","flux_config","flux_manifest","flux_retrieval_script")}
    config["tokenizers_native_source_sha256"] = "runtime-bound"
    runtime = dict(implementation_sources={"tokenizers_native_source_sha256":dict(sha256="a"*64,expected_sha256="runtime-bound",binding=native_binding)},attention_dispatch_source=config["attention_dispatch"],rope_buffer_policy=config["rope_buffer_policy"],
        numeric_flags_before_forecast=dict(float32_matmul_precision="highest",cuda_tf32=False,cuda_bf16_reduced_precision_reduction=True,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,deterministic_algorithms=True))
    write(raw/"runtime-contract.json",runtime)
    runtime_sha = A["sha"](raw/"runtime-contract.json")
    source_inventory.update({row["path"]:dict(sha256=row["sha256"],bytes=row["bytes"]) for row in retained.values()})
    source_inventory["runtime-contract.json"] = dict(sha256=runtime_sha,bytes=(raw/"runtime-contract.json").stat().st_size)
    write(raw/"source-inventory.json",source_inventory)
    request = dict(config=dict(config),source_hashes={"flux_functional_qwen_prefix_worker.py":A["WORKER_SHA256"],"flux_functional_qwen_prefix.py":A["QWEN_HELPER_SHA256"]})
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    config["request_sha256"] = request["request_sha256"]
    write(raw/"request.json",request)
    write(raw/"plan.json",config)
    def array(value,case,kind,index=None):
        value = np.ascontiguousarray(value,np.float32)
        assert not np.any(value.view(np.uint32)&65535)
        bits = (value.view(np.uint32)>>16).astype("<u2")
        relative = "arrays/"+case+"/"+kind+("-"+str(index) if index is not None else "")+".npy"
        path = raw/relative
        path.parent.mkdir(exist_ok=True,parents=True)
        np.save(path,bits,allow_pickle=False)
        return dict(path=relative,sha256=A["sha"](path),bytes=path.stat().st_size,content_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),shape=list(value.shape),
            stored_dtype="uint16",logical_dtype="torch.bfloat16",kind=kind,case=case,index=index,all_rows_retained=True)
    forecast,native,consumers,operands = {},{},{},[]
    base = np.arange(512*12,dtype=np.float32).reshape(1,512,12)%16
    for number,prompt in enumerate(A["PROMPTS"]):
        case = f"prompt-{number:02d}"
        messages = [dict(role="user",content=prompt)]
        formatted = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,padding="max_length",truncation=True,max_length=512,return_tensors="np",return_dict=True)
        ids,mask = [np.asarray(encoded[k],np.int64).reshape(1,-1) for k in ("input_ids","attention_mask")]
        p = {i:base+i+number for i in range(4)}
        n = {i:value.copy() for i,value in p.items()}
        if numerical_miss and number == 1:
            n[2][0,500,0] += 1
        ep,en = [np.concatenate([lane[i] for i in (1,2,3)],-1) for lane in (p,n)]
        cp,cn = ep[...,:6].copy(),en[...,:6].copy()
        profile = dict(native_module_forwards=0,prefix_layers=3,selected_hidden_states=[1,2,3],hidden_size=12,conditioning_width=36,attention_width=16,rope_buffer_policy=config["rope_buffer_policy"],attention_dispatch=config["attention_dispatch"])
        forecast[case] = dict(case=case,prompt=prompt,prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),formatted_chat=formatted,formatted_chat_sha256=hashlib.sha256(formatted.encode()).hexdigest(),
            input_ids=ids.tolist(),attention_mask=mask.tolist(),input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),attention_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            active_rows=int(mask.sum()),padded_rows=int(mask.size-mask.sum()),position_ids=[list(range(512))],native_module_forwards_during_forecast=0,
            forecast_hidden=[array(p[i],case,"forecast-hidden-state",i) for i in range(4)],forecast_conditioning=array(ep,case,"forecast-conditioning"),
            forecast_context_embedder=array(cp,case,"forecast-context-embedder"),profile=profile)
        native[case] = dict(case=case,retained_hidden=[array(n[i],case,"native-hidden-state",i) for i in range(4)],native_conditioning=array(en,case,"native-conditioning"),
            full_native_hidden_state_count=5,tail_layers_executed=1,output_hidden_states=True,use_cache=False,attention_backend=config["attention_dispatch"])
        consumers[case] = dict(predicted_input=forecast[case]["forecast_conditioning"],reference_input=native[case]["native_conditioning"],
            predicted_output=array(cp,case,"native-context-from-predicted"),reference_output=array(cn,case,"native-context-from-reference"),full_flux_transformer_calls=0)
        for name in ("predicted","reference"):
            operands.append(dict(ordinal=len(operands),shape=[1,512,36],dtype="torch.bfloat16",device="cpu",sha256=consumers[case][name+"_input"]["content_sha256"]))
    tx = ForecastTransaction(raw/"forecast-transaction",model_identity="synthetic-qwen",input_identity="two-prompt-ids",source_identity="synthetic-recipes",array_storage="files-v1",
        frontier=dict(kind="raw-input",source="synthetic saved fixture"),execution_scope="Synthetic saved arrays only; no real model calls.")
    tx.frontier("tokenized-inputs",{case:dict(input_ids=row["input_ids"],attention_mask=row["attention_mask"],prompt_utf8_sha256=row["prompt_utf8_sha256"]) for case,row in forecast.items()})
    payloads = {case+".conditioning":ForecastPayload(np.load(raw/row["forecast_conditioning"]["path"],allow_pickle=False),dtype=None) for case,row in forecast.items()}
    tx.predict("qwen-prefix-forecast",lambda _:dict(predictions=dict(cases=forecast,runtime_contract_sha256=runtime_sha),payloads=payloads))
    seal = tx.seal()
    tx.execute("native-targets-and-context-consumers",lambda _:dict(native=native,consumers=consumers))
    tx.close()
    summary = dict(request_sha256=config["request_sha256"],job_id="job-tiny-qwen-audit",cases=list(forecast),forecast_seal=seal,transaction=tx.report(),counts=dict(synthetic_saved_fixture=True))
    write(raw/"summary.json",summary)
    identity = {entry["source"]["tensor"]:entry["source"]["payload_sha256"] for entry in qm["artifacts"]}
    context_identity = fm["artifacts"][0]["source"]["payload_sha256"]
    write(raw/"source.json",dict(request_sha256=config["request_sha256"],job_id=summary["job_id"],worker_sha256=A["WORKER_SHA256"],helper_source_sha256={"flux_functional_qwen_prefix.py":A["QWEN_HELPER_SHA256"]},
        retained_metadata=retained,runtime_contract_sha256=runtime_sha,qwen_parameter_identity_before=identity,qwen_parameter_identity_after=identity,
        context_embedder_weight_identity_before=context_identity,context_embedder_weight_identity_after=context_identity,effective_plan_sha256=A["sha"](raw/"plan.json"),request_record_sha256=A["sha"](raw/"request.json"),synthetic_saved_fixture=True))
    for name,value in (("forecast-cases.json",forecast),("native-cases.json",native),("context-consumers.json",consumers),("context-consumer-operands.json",operands)):
        write(raw/name,value)
    write(raw/"native-contract.json",dict(full_native_layers=4,retained_states_per_case=4,native_encoder_calls=2,context_embedder_native_calls=4,forecast_native_module_forwards=0,
        context_embedder_shape=[6,36],context_embedder_weight_identity_before=context_identity,context_embedder_weight_identity_after=context_identity,
        context_embedder_bias=False,no_rgb_forecast=True,rope_buffer_policy=config["rope_buffer_policy"],attention_dispatch=config["attention_dispatch"]))
    events = [dict(kind="forecast_case_complete",case=case,native_forwards=0) for case in forecast]
    events += [dict(kind="forecast_globally_sealed",seal_sha256=seal["seal_sha256"],native_encoder_calls=0)]
    events += [dict(kind="native_case_complete",case=case,retained_hidden_states=4,full_native_hidden_state_count=5,context_calls=2) for case in forecast]
    (raw/"timeline.jsonl").write_text("\n".join(json.dumps(row) for row in events)+"\n")
    return raw,summary,repin(raw)


def repin(raw):
    files = {path.relative_to(raw).as_posix():dict(bytes=path.stat().st_size,sha256=A["sha"](path)) for path in raw.rglob("*") if path.is_file() and path.name!="files.json"}
    path = raw/"files.json"
    request_sha = json.loads((raw/"summary.json").read_text())["request_sha256"]
    path.write_text(json.dumps(dict(schema="mdb-reader-files-v1",job_id="job-tiny-qwen-audit",request_sha256=request_sha,files=files)))
    return A["sha"](path)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return saved_fixture(tmp_path_factory.mktemp("saved-qwen"),numerical_miss=True)


def test_saved_complete_audit_no_neural_calls(saved,tmp_path,monkeypatch):
    import torch

    def forbidden(*args,**kwargs):
        raise AssertionError("saved audit entered neural module")
    monkeypatch.setattr(torch.nn.Module,"_call_impl",forbidden)
    raw,summary,digest = saved
    report,_ = A["audit"](raw,tmp_path/"audit",expected_manifest=digest,expected_request=summary["request_sha256"])
    assert report["accounts"]["first_value_discrepancy"] == {"prompt-00":None,"prompt-01":"hidden.2"}
    assert report["accounts"]["cases"]["prompt-01"]["fields"]["hidden.2"]["padded_query_rows"]["comparison"]["changed_values"] == 1
    assert report["parameter_restoration"]["qwen"]["tensors"] == 46
    assert report["parameter_roles"]["query_projection_width"] == 16
    assert report["coverage"]["all_query_rows"] == 512
    provenance = json.loads((tmp_path/"audit/catalog/provenance.json").read_text())
    assert provenance["records"] == 2
    assert all(row["exact"] == 1 and row["changed_input"] == row["missing_mask"] == 0 for row in provenance["queries"])


@pytest.mark.parametrize("fault",["array","seal","consumer","prompt","chronology"])
def test_saved_tampering_refused(saved,tmp_path,fault):
    raw = tmp_path/"raw"
    shutil.copytree(saved[0],raw)
    def change(name,callback):
        value = json.loads((raw/name).read_text())
        callback(value)
        (raw/name).write_text(json.dumps(value))
    if fault == "array":
        record = json.loads((raw/"forecast-cases.json").read_text())["prompt-00"]["forecast_hidden"][0]
        with (raw/record["path"]).open("ab") as stream:
            stream.write(b"corrupt")
    elif fault == "seal":
        change("forecast-cases.json",lambda value:value["prompt-00"].update(native_module_forwards_during_forecast=1))
    elif fault == "consumer":
        change("context-consumer-operands.json",lambda value:value[0].update(sha256="0"*64))
    elif fault == "prompt":
        change("plan.json",lambda value:value["prompts"].__setitem__(0,"borrowed prompt"))
    else:
        path = raw/"timeline.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.insert(0,rows.pop())
        path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    digest = repin(raw)
    with pytest.raises(ValueError):
        A["audit"](raw,tmp_path/"audit",expected_manifest=digest,expected_request=saved[1]["request_sha256"])
    assert not (tmp_path/"audit").exists()


def test_actual_saved_only_compact_worker(saved,tmp_path):
    raw,summary,digest = saved
    path = HERE/"qualification/audit_flux_functional_qwen_prefix.py"
    config = dict(native_root=str(raw),native_manifest_sha256=digest,native_request_sha256=summary["request_sha256"],output_scratch=str(tmp_path/"outputs"))
    request = dict(config=config,request_sha256="2"*64,source_hashes={str(path):A["sha"](path)})
    request_path,config_path = tmp_path/"request.json",tmp_path/"config.json"
    request_path.write_text(json.dumps(request))
    config_path.write_text(json.dumps(config))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES="",SATURN_RESIDENT_WORKER="1",SATURN_DEBUG_REQUEST=str(request_path),SATURN_RUN_CONFIG=str(config_path),MRUN_JOB_ID="job-tiny-qwen-compact")
    result = subprocess.run([sys.executable,"-I","-B",str(path)],cwd=HERE/"qualification",env=env,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    ready = json.loads(next(line.split(" ",1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(ready["output_root"])
    assert A["sha"](output/"files.json") == ready["files_manifest_sha256"]
    assert json.loads((output/"summary.json").read_text())["records"] == 2
    assert not (output/"arrays").exists() and not (output/"parameters").exists()
    assert (output/"retrieve_evidence.py").is_file()


def test_native_extension_requires_actual_host_binding(tmp_path):
    raw,summary,digest = saved_fixture(tmp_path/"source",native_binding="staged-source-hash")
    with pytest.raises(ValueError,match="runtime binding"):
        A["audit"](raw,tmp_path/"audit",expected_manifest=digest,expected_request=summary["request_sha256"])


def test_catalog_budget_covers_original_complete_geometry_without_large_allocation():
    predicted = {"conditioning":np.broadcast_to(np.float32(0),(1,512,7680)),"projected":np.broadcast_to(np.float32(0),(1,512,3072))}
    calls = []
    def compared(p,n,*,max_elements):
        assert p is predicted and n is predicted
        calls.append(max_elements)
        return dict(mechanics_status="synthetic-budget-only")
    result = A["catalog_comparison"](predicted,predicted,compare_core=compared)
    assert calls == [5_505_024] and result["mechanics_status"] == "synthetic-budget-only"
