"""Test an unchanged external reader/writer against authenticated MDB futures.

The producer sees supplied current states and frozen coefficients. All candidate
images are sealed before this process opens historical future tensor values.
This is a trusted-code development assay, not an adversarial sandbox or a new
held-out benchmark. Original attention maps and the native VAE remain paid.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "mdb-explorer-binding-bridge-v1"
ARMS = ("whole_Q", "residual_only", "write_only", "both_RW")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def factorial(values):
    """Full finite interaction, with no unsigned subtraction."""
    import numpy as np

    q, r, w, rw = (np.asarray(values[name], dtype=np.float64) for name in ARMS)
    require(q.shape == r.shape == w.shape == rw.shape, "factorial shape differs")
    return (rw - r) - (w - q)


class Evidence:
    def __init__(self, bridge):
        from encoder_atoms import read, sha

        self.root = Path(bridge["source"]["root"])
        require(sha(self.root / "files.json") == bridge["source"]["manifest_sha256"],
                "original publication differs")
        self.files = read(self.root / "files.json")["files"]
        self.consumed = {}

    def member(self, name):
        from encoder_atoms import sha

        path = self.root / name
        require(path.resolve().is_relative_to(self.root.resolve()), "unsafe evidence member")
        expected = self.files[name]
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                "original evidence member differs: " + name)
        self.consumed[name] = expected
        return path

    def closure(self, ref):
        from encoder_atoms import read

        name = ref["member"]
        require(self.files[name] == ref["file"], "bridge cut inventory differs")
        metadata = read(self.member(name))
        require(metadata["schema"] == "saturn-pub-local-cut-v2", "authoritative v2 cut required")
        return [name, *("state-store/pages/" + page + ".json" for page in metadata["slots"].values()),
                *("state-store/blobs/" + blob + ".safetensors" for blob in metadata["physical_blobs"])]

    def cut(self, ref):
        from saturn_pub.store import LocalStore

        for name in self.closure(ref):
            self.member(name)
        cut = LocalStore(self.root / "state-store").load(ref["record"]["cut"], device="cpu")
        record = ref["record"]
        require(cut.fingerprint == record["cut"] and cut.parent == record["parent"]
                and cut.model_identity == record["model_identity"]
                and dict(cut.execution_point) == record["execution_point"], "StateCut identity differs")
        return cut

    def array(self, branch, record):
        import numpy as np
        from encoder_atoms import floats

        value = np.load(self.member(branch + "/" + record["path"]), allow_pickle=False)
        require(list(value.shape) == record["shape"], "native array shape differs")
        if record.get("encoding") == "bf16-bits-v1":
            require(value.dtype == np.uint16, "native BF16 representation differs")
            value = floats(value)
        require(np.isfinite(value).all(), "nonfinite native evidence")
        return value.astype(np.float64)


def checked_tensor(value, expected):
    from saturn_pub.values import describe

    actual = describe(value)
    require(all(actual[key] == expected[key] for key in ("sha256", "shape", "dtype", "stride")),
            "supplied tensor content/layout differs")
    return value


def dispatch_reader(registry, spec, frame):
    """Use the registered identity, preserving MDB's declared-read boundary."""
    return registry.run(spec.identity, "frozen-reader", frame)


def install_guard(evidence, bridge, source, fit, clock):
    """Deny future files until the globally sealed transaction releases them."""
    allowed = set()
    for case in bridge["cases"].values():
        for ref in (case["input_cut_after_conditioning_reset"], case["previous_velocity"]["cut"]):
            allowed.update(str((evidence.root / name).resolve()) for name in evidence.closure(ref))
    allowed.add(str((evidence.root / "files.json").resolve()))
    allowed.update(str((Path(source["modelRoot"]) / name).resolve()) for name in source["imageAssets"])
    allowed.update(str(Path(item["path"]).resolve()) for item in fit["fields"].values())
    allowed.update(str(Path(item["path"]).resolve()) for item in clock["fields"].values())
    phase = {"released": False, "denied": [], "future_opens": 0}
    roots = [evidence.root.resolve(), Path(source["modelRoot"]).resolve()]
    roots.extend(Path(item["path"]).parents[1] for item in fit["fields"].values())

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if str(path) in allowed or not any(path.is_relative_to(root) for root in roots):
            return
        if not phase["released"]:
            phase["denied"].append(str(path))
            raise PermissionError("Historical future unavailable before global seal")
        phase["future_opens"] += 1

    sys.addaudithook(audit)
    # Test a real target and a target cut, rather than trusting guard metadata.
    case = next(iter(bridge["cases"].values()))
    canaries = [evidence.root / case["evaluation_only"]["png"]["source_path"],
                evidence.root / case["evaluation_only"]["velocity3"]["cut"]["member"]]
    for path in canaries:
        try:
            path.open("rb")
        except PermissionError:
            continue
        raise ValueError("future quarantine canary was readable")
    phase["canary_denials"] = len(phase["denied"])
    return phase


def decoder_argument(vae, latent, ids, pipeline):
    """Match MDB's qualified CPU scatter/BN inverse before the GPU decoder."""
    import torch

    unpacked = pipeline._unpack_latents_with_ids(latent.detach().cpu(), ids.detach().cpu(), 32, 32)
    mean = vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(unpacked)
    std = (vae.bn.running_var.detach().cpu().view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(unpacked)
    result = pipeline._unpatchify_latents(unpacked * std + mean)
    require(result.dtype == torch.bfloat16 and result.device.type == "cpu"
            and tuple(result.shape) == (1, 32, 64, 64), "qualified decoder boundary differs")
    return result


def execute(root):
    from complete_masked_study import profile
    from encoder_atoms import identity, read, save, sha
    from flux_clock_study import assets_unchanged, source_checks, verify
    from paired_fields import load, metrics, payload, retain

    root = Path(root).resolve()
    request = read(root / "request.json")
    verify(root, request)
    bridge, source = read(root / "bridge.json"), read(root / "reader-source.json")
    fit, coeff_seal = read(root / "frozen-fit.json"), read(root / "frozen-coefficient-seal.json")
    require(sha(root / "frozen-fit.json") == coeff_seal["fitReceiptSHA256"]
            and fit["fields"] == coeff_seal["fields"] and coeff_seal["gain"] == 1, "frozen writer differs")
    clock = read(root / "clock-table.json")["rows"][3]
    evidence = Evidence(bridge)
    phase = install_guard(evidence, bridge, source, fit, clock)
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    import numpy as np
    import torch
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
    )
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from last_velocity_decode import scheduler
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.titan import ActRegistry, ActSpec, Frame, ProvidedFingerprint
    from source_native_reader import form
    from source_native_reader import load as load_reader

    started, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    require(numerical == source["numericalProfile"], "frozen numerical profile differs")
    model, assets = source_checks(source, diffusers)
    config = read(model / "transformer/config.json")
    require(config["eps"] == 1e-6 and config["num_attention_heads"] == 24, "reader geometry differs")
    weights, norms, position, loaded = load_reader(model / "transformer/diffusion_pytorch_model.safetensors", config)
    before = {name: (value.data_ptr(), value._version, payload(value)) for name, value in weights.items()}
    for item in fit["fields"].values():
        load(item)
    mods = {name: load(clock["fields"][name], "cuda") for name in ("double_img", "double_txt")}
    # Preserve fitted coefficients, while original pretrained tensors stay external.
    coefficient_copies = {}
    for name, item in fit["fields"].items():
        destination = root / "frozen-writer" / (name + ".npy")
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(item["path"], destination)
        require(sha(destination) == item["sha256"], "coefficient custody differs")
        coefficient_copies[name] = dict(original=item, custodyPath=str(destination.relative_to(root)), sha256=sha(destination))
    save(root / "frozen-writer-artifacts.json", dict(originalFitSHA256=sha(root / "frozen-fit.json"),
        originalSealSHA256=sha(root / "frozen-coefficient-seal.json"), coefficients=coefficient_copies,
        replayPolicy="Original receipts remain immutable and contain original absolute paths. A relocated replay must create a separately identified fit/seal path remapping, verify every original file and payload hash, and preserve coefficient bytes, normalization, bias, gain and rounding. This run served original source paths."))
    vae = AutoencoderKLFlux2.from_pretrained(str(model / "vae"), torch_dtype=torch.bfloat16,
                local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False)
    vae.post_quant_conv.to("cuda")
    vae.decoder.to("cuda")
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1) * 2)
    parameter_identity = identity({item["name"]: item["payloadSHA256"] for item in loaded})
    registry = ActRegistry()
    spec = ActSpec("mdb.explorer_first_joint_read", "v1",
        reads=("Z", "E", "imgIDs", "txtIDs", "modImage", "modText", "weights"), writes=("H0",),
        write_kinds={"H0": "tensor"}, write_abis={"H0": {"shape": [1, 1024, 3072], "dtype": "torch.bfloat16"}},
        numerical_contract="unchanged-source-native-image-query-reader; own recorded Z/E/IDs; compiled clock3")
    operations, attention, reads, servings, states = [], [], [], [], {}
    current = {}

    def operation(values):
        require(values["weights"] is weights, "parameter map binding differs")
        result = form(weights, values["Z"], values["E"], values["imgIDs"], values["txtIDs"],
            values["modImage"], values["modText"], norms, position, native, 24, operations, attention, current)
        require(list(result.shape) == [1, 1024, 3072] and result.dtype == torch.bfloat16,
                "actual reader output geometry differs from declared ABI")
        return {"H0": result}, {"originalAffineCalls": 7, "futureAccess": False}

    registry.register(spec, "frozen-reader", operation)
    with torch.inference_mode():
        for index, (case_id, case) in enumerate(bridge["cases"].items()):
            current.clear()
            current.update(id=case_id)
            cut = evidence.cut(case["input_cut_after_conditioning_reset"])
            previous = evidence.cut(case["previous_velocity"]["cut"])
            state = cut.payload
            require((state["step"], state["phase"], state["cursor"]) == (3, "project", 0), "input clock differs")
            require((previous.payload["step"], previous.payload["phase"]) == (2, "scheduler"), "previous velocity clock differs")
            require(state["image"] is None and state["text"] is None, "expected pre-projection state")
            z = checked_tensor(state["latent"], case["input_z3"]).to("cuda")
            e = checked_tensor(state["conditioning"], case["conditioning"]).to("cuda")
            n = checked_tensor(previous.payload["noise_prediction"], case["previous_velocity"]["descriptor"]).to("cuda")
            for name, descriptor in case["scheduler_inputs"].items():
                checked_tensor(state[name], descriptor)
            states[case_id] = {name: state[name] for name in ("latent", "img_ids", "txt_ids", "sigmas", "timesteps")}
            values = {"Z": z, "E": e, "imgIDs": state["img_ids"].to("cuda"), "txtIDs": state["txt_ids"].to("cuda"),
                      "modImage": mods["double_img"], "modText": mods["double_txt"], "weights": weights}
            fingerprints = {name: ProvidedFingerprint.content_sha256(parameter_identity if name == "weights" else payload(value))
                            for name, value in values.items()}
            abis = {name: ({"parameterMap": loaded} if name == "weights" else
                    {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride())}) for name, value in values.items()}
            frame = Frame.from_values(values, kinds={name: "parameter-map" if name == "weights" else "tensor" for name in values},
                abis=abis, slot_fingerprints=fingerprints, program_fingerprint=request["requestSHA256"],
                metadata={"case": case_id, "sourceCut": cut.fingerprint, "previousCut": previous.fingerprint})
            output, receipt = dispatch_reader(registry, spec, frame)
            directory = root / "arrays" / ("input-%02d" % index)
            fields = {name: retain(directory, name, value) for name, value in (("z", z), ("previous", n), ("H0", output.values["H0"]))}
            reads.append({"id": case_id, "fields": fields, "inputFrame": frame.to_dict(), "outputFrame": output.to_dict(), "receipt": receipt.to_dict()})
            servings.append(dict(id=case_id, population="existing-mdb-development", **fields,
                                 output=str(root / "arrays" / ("writer-%02d" % index))))
            print("BRIDGE_READER", case_id, flush=True)
    require(len(operations) == 56 and len(attention) == 8, "physical read budget differs")
    save(root / "reader-frames.json", reads)
    save(root / "serve-spec.json", dict(coefficientSeal=str(root / "frozen-coefficient-seal.json"),
        fitReceipt=str(root / "frozen-fit.json"), inputs=servings, receipt=str(root / "serving.json"),
        protectedRoots=[str(evidence.root), str(model), str(Path(fit["fields"]["weight"]["path"]).parents[1])],
        deniedCanaries=[str(evidence.root / next(iter(bridge["cases"].values()))["evaluation_only"]["png"]["source_path"])]))
    subprocess.run([sys.executable, "-B", "-u", str(root / "source_native_writer.py"), "serve", str(root / "serve-spec.json")],
        check=True, timeout=120, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    served = read(root / "serving.json")
    require(len(served["rows"]) == 8 and served["frameworksLoaded"] == [], "isolated writer census differs")
    fields, images, commits, vae_ops = {}, {}, [], []
    handles = []

    def hook(name):
        def count(module, args, result):
            vae_ops.append({**current, "module": name, "kind": type(module).__name__,
                "inputShape": list(args[0].shape), "outputShape": list(result.shape),
                "nominalMACs": result.numel() * (module.weight.numel() // module.weight.shape[0])})
        return count

    for name, module in vae.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))

    def decode(case_id, method, velocity):
        state = states[case_id]
        current.clear()
        current.update(id=case_id, method=method)
        sch = scheduler(source, torch, FlowMatchEulerDiscreteScheduler)
        z3 = state["latent"].to("cuda")
        z4 = sch.step(velocity.to("cuda"), sch.timesteps[3], z3, return_dict=False)[0]
        argument = decoder_argument(vae, z4, state["img_ids"], Flux2KleinPipeline)
        decoded = vae.decode(argument.to("cuda"), return_dict=False)[0]
        png = processor.postprocess(decoded, output_type="pil")[0]
        key = case_id + "/" + method
        directory = root / "arrays" / key
        fs = {name: retain(directory, name, value) for name, value in
              (("velocity3", velocity), ("latent4", z4), ("vaeInput", argument), ("decodedRaw", decoded), ("pngRGB", np.asarray(png)))}
        fields[key] = fs
        path = root / "images" / (case_id.replace("/", "--") + "--" + method + ".png")
        path.parent.mkdir(exist_ok=True)
        png.save(path)
        images[key] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        commits.append(dict(id=case_id, method=method, input=payload(z3), velocity=payload(velocity), output=payload(z4),
                            finalStepIndex=sch.step_index, decoderInput=payload(argument)))
        require(sch.step_index == 4, "finite scheduler did not commit")
        return fs

    with torch.inference_mode():
        for row in served["rows"]:
            require(row["method"] == "candidate", "unexpected writer method")
            expected = next(item for item in servings if item["id"] == row["id"])
            require(row["inputFields"] == {name: expected[name] for name in ("z", "previous", "H0")}, "writer input lineage differs")
            decode(row["id"], "candidate", load(row["velocity"], "cuda"))
            decode(row["id"], "hold", load(row["inputFields"]["previous"], "cuda"))
            print("BRIDGE_IMAGES", row["id"], flush=True)
    require(len(vae_ops) == 16 * 40, "candidate/hold native consumer work differs")
    require(len(phase["denied"]) == phase["canary_denials"] and not phase["future_opens"], "future touched before seal")
    save(root / "candidate-images.json", dict(fields=fields, images=images, commits=commits,
        readerFramesSHA256=sha(root / "reader-frames.json"), servingSHA256=sha(root / "serving.json")))
    external = {str(path.relative_to(root)): dict(sha256=sha(path), bytes=path.stat().st_size)
                for sub in ("arrays", "images", "frozen-writer") for path in (root / sub).rglob("*") if path.is_file()}
    for name in ("candidate-images.json", "reader-frames.json", "serving.json", "frozen-fit.json", "frozen-coefficient-seal.json"):
        path = root / name
        external[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    save(root / "sealed-external-files.json", external)
    tx = ForecastTransaction(root / "forecast-transaction", model_identity=bridge["cases"][next(iter(bridge["cases"]))]["input_cut_after_conditioning_reset"]["record"]["model_identity"],
        input_identity=identity(bridge), source_identity=identity(request["sourceHashes"]),
        frontier={"kind": "architectural", "scope": "Observed final-clock MDB states; original source assistance inherited"},
        execution_scope="Externally instrumented selective CUDA reader, isolated CPU frozen writer, exact MDB CPU normalization/native CUDA VAE",
        budgets=dict(max_predictions=1, max_executions=1, max_bytes=300 * 1024 * 1024, max_files=2000, max_wall_ms=600000))
    tx.frontier("source-cases", {"bridgeSHA256": sha(root / "bridge.json"), "existingDevelopmentEvidence": True})
    tx.predict("all-futures", lambda _ctx: {"predictions": {"external_files": external, "future_array_opens": phase["future_opens"]}})
    seal = tx.seal()
    print("BRIDGE_ALL_FUTURES_SEALED", seal["seal_sha256"], flush=True)
    accounts, responses, interactions, controls = [], [], [], []
    native_fields = {}

    def evaluate(_context):
        phase["released"] = True
        save(root / "release.json", dict(sealSHA256=seal["seal_sha256"], futureOpensBeforeRelease=phase["future_opens"],
            seconds=time.perf_counter() - started, canaryDenials=phase["canary_denials"], isolation="trusted Python audit hook, not an OS sandbox"))
        with torch.inference_mode():
            for case_id, case in bridge["cases"].items():
                velocity = evidence.cut(case["evaluation_only"]["velocity3"]["cut"]).payload["noise_prediction"]
                checked_tensor(velocity, case["evaluation_only"]["velocity3"]["descriptor"])
                final = evidence.cut(case["evaluation_only"]["final_state"]).payload
                branch = "cases/" + case_id
                decoder = read(evidence.member(branch + "/decoder-native-reference.json"))
                argument = decoder_argument(vae, final["latent"], final["img_ids"], Flux2KleinPipeline)
                expected_arg = evidence.array(branch, decoder["decoder_input"])
                require(np.array_equal(argument.float().numpy(), expected_arg), "native decoder normalization does not match MDB")
                sch = scheduler(source, torch, FlowMatchEulerDiscreteScheduler)
                committed = sch.step(velocity.to("cuda"), sch.timesteps[3], states[case_id]["latent"].to("cuda"), return_dict=False)[0]
                require(payload(committed) == payload(final["latent"]), "native scheduler reference does not close")
                native_fields[case_id] = {"velocity3": velocity.float().numpy().astype(np.float64),
                    "latent4": final["latent"].float().numpy().astype(np.float64), "vaeInput": expected_arg,
                    "decodedRaw": evidence.array(branch, decoder["decoded"]), "pngRGB": evidence.array(branch, decoder["pixels"])}
                png_path = evidence.member(branch + "/" + decoder["png"]["path"])
                output_png = root / "images" / (case_id.replace("/", "--") + "--native.png")
                shutil.copyfile(png_path, output_png)
                images[case_id + "/native"] = dict(path=str(output_png), sha256=sha(output_png), bytes=output_png.stat().st_size)
                for method in ("candidate", "hold"):
                    for port, record in fields[case_id + "/" + method].items():
                        accounts.append(dict(case=case_id, method=method, port=port, **metrics(load(record), native_fields[case_id][port])))
                # Two exact image witnesses qualify the current native consumer.
                if case_id.endswith("/whole_Q"):
                    observed = decode(case_id, "native-control", velocity.to("cuda"))
                    check = {port: metrics(load(record), native_fields[case_id][port]) for port, record in observed.items()}
                    require(all(value["maxAbs"] == 0 for value in check.values()), "native consumer witness differs")
                    controls.append(dict(case=case_id, comparisons=check, exact=True))
            for context in ("binding-anchor", "binding-wording"):
                for method in ("candidate", "hold"):
                    for port in ("velocity3", "latent4", "decodedRaw", "pngRGB"):
                        candidate = {arm: load(fields[context + "/" + arm + "/" + method][port]) for arm in ARMS}
                        target = {arm: native_fields[context + "/" + arm][port] for arm in ARMS}
                        for arm in ARMS[1:]:
                            responses.append(dict(context=context, method=method, arm=arm, port=port,
                                orientation="arm-minus-whole_Q", **metrics(candidate[arm] - candidate["whole_Q"], target[arm] - target["whole_Q"])))
                        interactions.append(dict(context=context, method=method, port=port, orientation="RW-R-W+Q",
                            **metrics(factorial(candidate), factorial(target))))
        return dict(accounts=len(accounts), signedResponses=len(responses), interactions=len(interactions), exactNativeControls=len(controls))

    tx.execute("historical-native-comparison", evaluate)
    tx.close()
    for handle in handles:
        handle.remove()
    for name, item in external.items():
        require(sha(root / name) == item["sha256"], "sealed prediction changed")
    require(before == {name: (value.data_ptr(), value._version, payload(value)) for name, value in weights.items()}, "reader parameters changed")
    require(len(vae_ops) == 18 * 40, "native VAE budget differs")
    assets_unchanged(assets)
    transaction = verify_transaction(root / "forecast-transaction")
    save(root / "external-verification.json", dict(files=len(external), bytes=sum(row["bytes"] for row in external.values()),
        inventorySHA256=sha(root / "sealed-external-files.json"), allFileHashesMatched=True,
        scope="Separate explicit verification of externally sealed prediction files; MDB transaction verification alone does not hash these files."))
    save(root / "report.json", dict(schema=SCHEMA, requestSHA256=request["requestSHA256"], sealSHA256=seal["seal_sha256"],
        accounts=accounts, signedResponses=responses, interactions=interactions, nativeControls=controls,
        images=images, fields=fields, work=dict(sourceNativeReads=8, originalAffineCalls=len(operations),
            originalAffineMACs=sum(row["nominalMACs"] for row in operations),
            attentionQKAVMACs=sum(2 * row["queryShape"][0] * row["queryShape"][1] * row["keyShape"][1] * row["queryShape"][2] * row["queryShape"][3] for row in attention),
            frozenWriterProducts=8, frozenWriterMACs=8 * 1024 * 3328 * 128, closedFormFits=0, newFullDenoisers=0,
            newEncoders=0, candidateVAEDecodes=8, holdVAEDecodes=8, nativeWitnessVAEDecodes=2,
            nativeVAEConvLinearCalls=len(vae_ops), nativeVAENominalMACs=sum(row["nominalMACs"] for row in vae_ops),
            originalReaderParameterBytes=sum(row["bytes"] for row in loaded)),
        numericalProfile=numerical, transaction=transaction, quarantine=phase,
        workerSecondsAfterImports=time.perf_counter() - started, cpuSecondsAfterImports=time.process_time() - cpu,
        peakCUDAAllocatedBytes=torch.cuda.max_memory_allocated(), mechanics_status="complete", instrument_status="executed",
        terminal_status="unassessed", limitations=["Existing observed development cases, two contexts at seed109; not held-out generalization.",
            "Original source assistance, historical prefixes and conditioning are inherited; not raw-prompt or zero-neural generation.",
            "Reader and frozen fitted writer unchanged; original learned attention and native VAE remain paid.",
            "Native CPU ID scatter/BN normalization matches MDB references; this differs from the earlier explorer GPU normalization.",
            "Image-query-only attention has its own dispatch geometry; no full-block native parity or measured speedup claim.",
            "Trusted-code Python file quarantine and MDB transaction are instrumentation, not an adversarial OS sandbox."]))
    save(root / "source-consumed.json", evidence.consumed)
    save(root / "physical-work.json", dict(originalReaderParameters=loaded, affines=operations, attention=attention, VAE=vae_ops))
    print("BRIDGE_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    execute(parser.parse_args().root)
