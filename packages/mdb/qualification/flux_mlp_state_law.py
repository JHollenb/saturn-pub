"""Own-input SwiGLU computation study; no image editing or fitted surrogate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def quantize(value):
    import numpy as np
    value = np.asarray(value, np.float32)
    if not np.isfinite(value).all():
        raise ValueError("finite BF16 operand required")
    bits = value.view(np.uint32)
    return ((bits + np.uint32(32767) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def silu(value):
    import numpy as np
    x = np.asarray(value, np.float64)
    p = x >= 0
    out = np.empty_like(x)
    out[p] = x[p] / (1 + np.exp(-x[p]))
    exp = np.exp(x[~p])
    out[~p] = x[~p] * exp / (1 + exp)
    return out


def forecast(x, incoming, outgoing):
    """Independent FP64 BLAS/SiLU, explicit BF16 operation boundaries."""
    import numpy as np
    x = np.asarray(x)
    if x.ndim != 3 or incoming.shape[1] != x.shape[-1] or incoming.shape[0] != 2 * outgoing.shape[1] or outgoing.shape[0] != x.shape[-1]:
        raise ValueError("SwiGLU geometry differs")
    projected = quantize(x.astype(np.float64) @ incoming.astype(np.float64).T)
    gate, value = np.split(projected, 2, axis=-1)
    activated = quantize(silu(gate))
    product = quantize(activated.astype(np.float64) * value)
    output = quantize(product.astype(np.float64) @ outgoing.astype(np.float64).T)
    return dict(projected=projected, activated=activated, product=product, output=output)


def decompose(low, high, outgoing):
    """Exact product-difference algebra on supplied rounded intermediates.

    Named components can cancel; their norms are not semantic percentages.
    Projection is diagnostic FP64 and excludes the native output rounding.
    """
    import numpy as np
    s0, s1 = [v["activated"].astype(np.float64) for v in (low, high)]
    v0, v1 = [np.split(v["projected"], 2, axis=-1)[1].astype(np.float64) for v in (low, high)]
    components = dict(value=s0 * (v1 - v0), gate=(s1 - s0) * v0,
                      interaction=(s1 - s0) * (v1 - v0),
                      product_rounding=(high["product"].astype(np.float64) - s1 * v1)
                                      - (low["product"].astype(np.float64) - s0 * v0))
    actual_delta = high["product"].astype(np.float64) - low["product"].astype(np.float64)
    residual = sum(components.values()) - actual_delta
    projected = {k: v @ outgoing.astype(np.float64).T for k, v in components.items()}
    return dict(components=components, projected=projected,
                closure_max_abs=float(np.abs(residual).max()),
                product_delta=actual_delta,
                projected_delta=actual_delta @ outgoing.astype(np.float64).T)


def statistics(value):
    import numpy as np
    x = np.asarray(value, np.float64)
    return dict(rms=float(np.sqrt(np.mean(x*x))), l2=float(np.linalg.norm(x)),
                max_abs=float(np.abs(x).max()), mean=float(x.mean()), scalars=int(x.size))


def difference(value, reference):
    import numpy as np
    d = np.asarray(value, np.float64) - np.asarray(reference, np.float64)
    r = statistics(d)
    r.update(exact=bool(np.array_equal(value, reference)), changed_scalars=int(np.count_nonzero(d)),
             relative_rms=r["rms"] / max(statistics(reference)["rms"], 1e-300))
    return r


def cases(low, high):
    import numpy as np
    mid = quantize((low.astype(np.float64) + high.astype(np.float64)) / 2)
    single = low.copy()
    single[:, 0] = high[:, 0]
    return dict(base=low, changed=high, midpoint=mid,
                permuted_midpoint=mid[:, ::-1].copy(), single_row=single)


def stage(run, source):
    import numpy as np
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.parameter_artifacts import validate_manifest

    here = Path(__file__).resolve().parent
    support = runpy.run_path(str(here / "flux_context_transfer.py"))
    audit = read(source / "audit-v2.json")
    if audit["mechanics_status"] != "verified":
        raise ValueError("accepted source audit required")
    original = source / "collection-01/extracted/binding-minimal-result"
    rows = {r["id"]: r for r in read(original / "rows.json")}
    prior = read(original / "request.json")
    support["bind_environment"](prior["execution_environment"])
    run.mkdir(exist_ok=False, parents=True)
    indices = sorted(np.random.default_rng(20261105).choice(1024, 32, replace=False).tolist())
    input_refs = {}
    extras = []
    for name, old in (("base", "control"), ("changed", "restore-I")):
        record = rows[old]["writer"]["fields"]["mlp_input"]
        path = original / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("native input witness differs")
        array = np.load(path, allow_pickle=False)[:, indices].copy()
        if array.shape != (1, 32, 3072) or not np.array_equal(array, quantize(array)):
            raise ValueError("source BF16 input geometry differs")
        target = run / (name + ".npy")
        np.save(target, array, allow_pickle=False)
        extras.append((target, target.name))
        input_refs[name] = dict(source_row=old, source_array=record, selected_sha256=sha(target),
                                source_writer_entry=rows[old]["writer"]["entry"])
    parameters = validate_manifest(read(original / "parameter-artifacts.json"))
    selected = [r for r in parameters["artifacts"] if r["source"]["tensor"] in {
        "transformer_blocks.1.ff.linear_in.weight", "transformer_blocks.1.ff.linear_out.weight"}]
    if len(selected) != 2:
        raise ValueError("exact named image-MLP parameter pair required")
    save(run / "parameter-artifacts.json", {**parameters, "artifacts": selected})
    shutil.copyfile(original / "retrieve_weights.py", run / "retrieve_weights.py")
    extras.extend((run / n, n) for n in ("parameter-artifacts.json", "retrieve_weights.py"))
    config = dict(schema="mdb-flux-mlp-state-law-plan-v1",
        model="component:" + prior["config"]["checkpoint_sha256"] + ":transformer_blocks.1.ff",
        source_model=prior["model"],
        model_path=prior["config"]["model_path"], output_scratch=prior["config"]["output_scratch"],
        checkpoint_sha256=prior["config"]["checkpoint_sha256"],
        checkpoint_file=selected[0]["source"]["file"], checkpoint_file_sha256=selected[0]["source"]["file_sha256"],
        source_audit_sha256=sha(source / "audit-v2.json"), source_job=prior.get("job_id", audit["job_id"]),
        inputs=input_refs, selected_rows=indices, cpu_threads=4, max_wall_seconds=300,
        cases=["base", "changed", "midpoint", "permuted_midpoint", "single_row"],
        expected_counts=dict(block_loads=1, native_mlp_calls=5, native_linear_calls=10, native_swiglu_calls=5,
            full_model_forwards=0, images=0, encodes=0, independent_numpy_forecasts=3),
        hypotheses=dict(row_permutation="Native midpoint output permutes with input rows; inspect numerical residual.",
            single_row="Only output row0 can differ from base; row0 should equal changed row0 under matched kernel geometry.",
            mathematical="Own-input checkpoint equations predict every retained native substep. Retain all numeric error; no quality gate.",
            nonlinear="Measure whether midpoint output equals endpoint average, without assuming affine or strong curvature.",
            labels="No object/color labels or fitted coefficients used. Inputs are observed normalized MLP states; upstream attention prediction is out of scope."),
        directory_custody=True)
    save(run / "plan.json", config)
    extras.extend(((run / "plan.json", "plan.json"), (here / "flux_context_transfer.py", "flux_context_transfer.py"),
                   (here / "flux_family_custody.py", "flux_family_custody.py")))
    pre = run / "preflight.v1.json"
    save(pre, dict(schema_version="mrun-preflight-config-v1",
        hashed_roots=[Path(__file__).name, *[n for _, n in extras], pre.name],
        path_token_allowlist=[config["model_path"], config["output_scratch"], "binding-minimal-result"]))
    extras.append((pre, pre.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = dict(schema="mdb-flux-mlp-state-law-request-v1", config=config, model=config["model"],
        backend="diffusers", device="cpu", dtype="bfloat16", batch_size=1,
        needs={}, execution_environment=prior["execution_environment"],
        config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes={n: sha(p) for p, n in [(Path(__file__), Path(__file__).name), *extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=5,
            logical_branches=5, logical_checkpoint_replays=0, max_local_work_items=5,
            checkpoint_policy="One isolated original-checkpoint image MLP; 5native32row calls after complete seal;3NumPy forecasts; no full pipeline/VAE/encoder; no retries."))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=Path(__file__), config=config, request=request,
                 config_bytes=raw, additional_files=extras)
    return dict(run=str(run), request_sha256=request["request_sha256"])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    request = read(run / "request.json")
    runpy.run_path(str(run / "payload/flux_context_transfer.py"))["bind_environment"](request["execution_environment"])
    if (run / "submission.json").exists() or any(sha(run / "payload" / n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("already submitted or stage changed")
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request seal differs")
    job = guarded_mrun_launch(worker_command(Path(__file__).name), request=request, model=request["model"], task="forward",
        experiment="mdb-flux-mlp-state-law", payload=run / "payload", pin="beast", gpu=False,
        needs=request["needs"], timeout_s=420, retry_on_kill=False, preflight=True,
        note="Computation-first: original joint1 image SwiGLU, CPU BF16 qualified MDB recorder.5native B1x32x3072 MLP calls,3pureNumPy predictions sealed first.2checkpoint tensors only,no pipeline/encoder/VAE/GPU/images. Source inputs were observed CUDA BF16; CPU kernel arithmetic separately measured. Beast pinned for checkpoint-local file; intent-first resources,300worker420outer limit,no retries.")
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


def worker():
    payload = Path.cwd()
    config, request = [read(os.environ[k]) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST")]
    if config != request["config"] or any(sha(payload / n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("source/config seal differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from safetensors import safe_open
    from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward
    from mdb.forward_recording import record_forward, tensor_descriptor
    from mdb.forward_ports import NativePortCatalog
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    scratch = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(exist_ok=False, parents=True)
    os.chdir(scratch)
    out = Path("binding-minimal-result")
    out.mkdir()
    save(out / "plan.json", config)
    save(out / "request.json", request)
    for name in ("parameter-artifacts.json", "retrieve_weights.py", "base.npy", "changed.npy"):
        shutil.copyfile(payload / name, out / name)
    checkpoint = Path(config["model_path"]) / config["checkpoint_file"]
    if sha(checkpoint) != config["checkpoint_file_sha256"]:
        raise ValueError("complete checkpoint file identity differs")
    records = read(out / "parameter-artifacts.json")["artifacts"]
    state, weights = {}, {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as bank:
        for record in records:
            source = record["source"]
            tensor = bank.get_tensor(source["tensor"])
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            if list(tensor.shape) != source["shape"] or tensor.dtype != torch.bfloat16 or hashlib.sha256(raw).hexdigest() != source["payload_sha256"]:
                raise ValueError("named checkpoint parameter/recipe join differs")
            name = source["tensor"].removeprefix("transformer_blocks.1.ff.")
            state[name] = tensor.clone().contiguous()
            weights[name] = tensor.float().numpy()
            weights[name].flags.writeable = False
    with torch.device("meta"):
        mlp = Flux2FeedForward(dim=3072, inner_dim=9216, bias=False)
    mlp.load_state_dict(state, strict=True, assign=True)
    mlp.eval().requires_grad_(False)
    counts = dict.fromkeys(config["expected_counts"], 0)
    counts["block_loads"] = 1
    def count_mlp(module, args):
        counts["native_mlp_calls"] += 1
    def count_linear(module, args):
        counts["native_linear_calls"] += 1
    def count_swiglu(module, args):
        counts["native_swiglu_calls"] += 1
    handles = [mlp.register_forward_pre_hook(count_mlp), mlp.linear_in.register_forward_pre_hook(count_linear),
               mlp.linear_out.register_forward_pre_hook(count_linear), mlp.act_fn.register_forward_pre_hook(count_swiglu)]
    inputs = cases(np.load(out / "base.npy", allow_pickle=False), np.load(out / "changed.npy", allow_pickle=False))
    model_identity = config["checkpoint_sha256"] + ":transformer_blocks.1.ff"
    tx = ForecastTransaction(out / "transaction", model_identity=model_identity,
        input_identity=config["source_audit_sha256"], source_identity=request["request_sha256"],
        frontier=dict(kind="architectural", boundary="observed-normalized-MLP-input", source_job=config["source_job"],
                      cases=config["cases"], selected_rows=config["selected_rows"]),
        execution_scope="Explicitly counted isolated native MLP; independent NumPy arithmetic uses original checkpoint parameters.",
        budgets=dict(max_native_calls=5, max_wall_ms=300000, max_bytes=128 << 20))
    predictions = {}
    def predict(_):
        for name in ("base", "changed", "midpoint"):
            predictions[name] = forecast(inputs[name], weights["linear_in.weight"], weights["linear_out.weight"])
            counts["independent_numpy_forecasts"] += 1
        predictions["permuted_midpoint"] = {k: v[:, ::-1].copy() for k, v in predictions["midpoint"].items()}
        predictions["single_row"] = {}
        for key, value in predictions["base"].items():
            mixed = value.copy()
            mixed[:, 0] = predictions["changed"][key][:, 0]
            predictions["single_row"][key] = mixed
        return dict(predictions=dict(fields=predictions, hypotheses=config["hypotheses"], counts_at_seal=dict(counts)))
    tx.predict("stage_outputs", predict)
    seal = tx.seal()
    save(out / "seal-census.json", dict(counts=counts.copy(), seal=seal))
    print("MDB_MLP_STATE_LAW " + json.dumps(dict(phase="sealed", counts=counts, seconds=time.monotonic()-started)), flush=True)
    observations, traces = {}, {}
    with tx:
        def execute(_):
            for index, name in enumerate(config["cases"]):
                if time.monotonic() - started > config["max_wall_seconds"]:
                    raise RuntimeError("finite wall budget")
                x = torch.from_numpy(inputs[name]).to(device="cpu", dtype=torch.bfloat16)
                capture = record_forward(mlp, (x,), model_identity=model_identity,
                    execution=dict(backend="cpu", device="cpu", dtype="bfloat16", threads=4),
                    input_identity=hashlib.sha256(inputs[name].tobytes()).hexdigest(), capture_ports=True,
                    metadata=dict(forward_call=index, case_id=name, checkpoint_sha256=config["checkpoint_sha256"]), max_operations=64)
                catalog = NativePortCatalog(capture.trace)
                fields, bindings = {}, {}
                for key, path in (("projected", "linear_in"), ("product", "act_fn"), ("output", "linear_out")):
                    binding = catalog.resolve("module:" + path + ":return")
                    fields[key] = capture.observed_values[binding["value_id"]].float().numpy()
                    bindings[key] = binding
                silus = [event for event in capture.trace["events"] if event["operator"] == "aten.silu.default"]
                if len(silus) != 1 or len(silus[0]["outputs"]) != 1:
                    raise ValueError("native SiLU operation absent/ambiguous")
                identifier = silus[0]["outputs"][0]
                fields["activated"] = capture.observed_values[identifier].float().numpy()
                bindings["activated"] = dict(value_id=identifier, operator="aten.silu.default", sequence=silus[0]["sequence"])
                observations[name] = fields
                traces[name] = capture.trace
                save(out / "traces" / (name + ".json"), capture.trace)
                save(out / "ports" / (name + ".json"), bindings)
                parameter_bindings = {n: catalog.resolve("parameter:" + n) for n in state}
                save(out / "parameters" / (name + ".json"), parameter_bindings)
                del capture
                print("MDB_MLP_STATE_LAW " + json.dumps(dict(phase="native_complete", case=name, counts=counts, seconds=time.monotonic()-started)), flush=True)
            return dict(fields=observations, counts=counts.copy())
        tx.execute("native", execute)
    for handle in handles:
        handle.remove()
    if counts != config["expected_counts"]:
        raise ValueError("declared physical census differs")
    delta = decompose(observations["base"], observations["changed"], weights["linear_out.weight"])
    curvature_predicted = predictions["midpoint"]["output"].astype(np.float64) - (predictions["base"]["output"].astype(np.float64) + predictions["changed"]["output"])/2
    curvature_actual = observations["midpoint"]["output"].astype(np.float64) - (observations["base"]["output"].astype(np.float64) + observations["changed"]["output"])/2
    metrics = dict(stage_errors={name: {key: difference(value, predictions[name][key]) for key, value in fields.items()} for name, fields in observations.items()},
        permutation=difference(observations["permuted_midpoint"]["output"], observations["midpoint"]["output"][:, ::-1]),
        untouched_rows=difference(observations["single_row"]["output"][:, 1:], observations["base"]["output"][:, 1:]),
        selected_row=difference(observations["single_row"]["output"][:, :1], observations["changed"]["output"][:, :1]),
        endpoint_delta=statistics(observations["changed"]["output"] - observations["base"]["output"]),
        curvature=dict(predicted=statistics(curvature_predicted), actual=statistics(curvature_actual), error=difference(curvature_actual, curvature_predicted)),
        product_difference_closure_max_abs=delta["closure_max_abs"],
        projected_components={k: statistics(v) for k, v in delta["projected"].items()},
        projected_delta=statistics(delta["projected_delta"]),
        native_output_delta_minus_FP64_projected=difference(observations["changed"]["output"].astype(np.float64)-observations["base"]["output"], delta["projected_delta"]))
    save(out / "metrics.json", metrics)
    summary = dict(schema="mdb-flux-mlp-state-law-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        counts=counts, seal=seal, transaction=verify_transaction(tx.root), worker_seconds=time.monotonic()-started,
        mechanics_status="pending-independent-audit", terminal_status="unassessed", scope="Isolated original-checkpoint image MLP; no attention or full-model/semantic forecast.")
    save(out / "summary.json", summary)
    save(out / "files.json", {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=summary["job_id"], request_sha256=request["request_sha256"],
        output_root=str(out.resolve()), files_manifest_sha256=sha(out / "files.json"), summary_sha256=sha(out / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--source", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        result = submit(args.run.resolve()) if args.submit else stage(args.run.resolve(), args.source.resolve())
        print(json.dumps(result, sort_keys=True))
