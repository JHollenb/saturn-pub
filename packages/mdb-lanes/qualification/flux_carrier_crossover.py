"""Sealed own-latent × own-previous-velocity continuations.

The inputs are authenticated, previously observed MDB states. Explicit finite
scheduler arithmetic and the qualified functional decoder predict all corners
before separate native scheduler/decoder execution. No transformer is loaded.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

CONTEXTS = ("binding-anchor", "binding-wording")
CORNERS = {"QQ": ("whole_Q", "whole_Q"), "PQ": ("both_RW", "whole_Q"),
           "QP": ("whole_Q", "both_RW"), "PP": ("both_RW", "both_RW")}


def require(value, message):
    if not value:
        raise ValueError(message)


def execute(root):
    from encoder_atoms import identity, read, save, sha
    from flux_clock_study import assets_unchanged, source_checks, verify
    from flux_frozen_reader_bridge import Evidence, checked_tensor, decoder_argument, install_guard
    from paired_fields import load, metrics, payload, retain

    root = Path(root).resolve()
    request = read(root / "request.json")
    verify(root, request)
    bridge, source = read(root / "bridge.json"), read(root / "reader-source.json")
    evidence = Evidence(bridge)
    phase = install_guard(evidence, bridge, source, {"fields": {}}, {"fields": {}})
    import diffusers
    import numpy as np
    import torch
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from complete_masked_study import profile
    from flux_functional_decoder import functional_decode
    from flux_functional_denoiser import functional_scheduler
    from last_velocity_decode import scheduler
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.titan import ActRegistry, ActSpec, Frame, ProvidedFingerprint

    started = time.perf_counter()
    numerical = profile(torch, 4)
    require(numerical == source["numericalProfile"], "original numerical profile differs")
    _, assets = source_checks(source, diffusers)
    vae = AutoencoderKLFlux2.from_pretrained(str(Path(source["modelRoot"]) / "vae"),
        torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False)
    vae.post_quant_conv.to("cuda")
    vae.decoder.to("cuda")
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1) * 2)
    states, velocities, sources = {}, {}, {}
    for context in CONTEXTS:
        for arm in ("whole_Q", "both_RW"):
            case = bridge["cases"][context + "/" + arm]
            cut = evidence.cut(case["input_cut_after_conditioning_reset"])
            previous = evidence.cut(case["previous_velocity"]["cut"])
            require((cut.payload["step"], cut.payload["phase"]) == (3, "project"), "source frontier differs")
            states[context, arm] = cut.payload
            checked_tensor(cut.payload["latent"], case["input_z3"])
            velocities[context, arm] = checked_tensor(previous.payload["noise_prediction"], case["previous_velocity"]["descriptor"])
            sources[context + "/" + arm] = dict(inputCut=cut.fingerprint, previousCut=previous.fingerprint)
        left, right = (states[context, arm] for arm in ("whole_Q", "both_RW"))
        require(all(payload(left[name]) == payload(right[name]) for name in ("conditioning", "img_ids", "txt_ids", "sigmas", "timesteps")),
                "crossing changes an undeclared source operand")

    registry = ActRegistry()
    spec = ActSpec("mdb.flux.final_hold_commit", "v1", reads=("latent", "velocity", "sigmas"),
        writes=("latent4",), write_kinds={"latent4": "tensor"},
        numerical_contract="Qualified original weak-scalar BF16 product added to FP32 latent, then BF16 cast; final clock3")

    def equation(values):
        sigmas = values["sigmas"].to("cuda")
        z, v = values["latent"], values["velocity"]
        result = functional_scheduler(z, v, sigmas, 3)
        return {"latent4": result}, {"newDenoisers": 0, "originalClock": 3}

    registry.register(spec, "carrier-crossover", equation)
    fields, images, frames, counts = {}, {}, [], {}
    native_calls = []
    handles = [module.register_forward_hook(lambda module, args, out: native_calls.append(type(module).__name__))
               for module in vae.modules() if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]

    def result(key, lane, z4, v, ids, *, functional):
        argument = decoder_argument(vae, z4, ids, Flux2KleinPipeline)
        decoded = functional_decode(vae, argument.to("cuda"), counts=counts) if functional else vae.decode(argument.to("cuda"), return_dict=False)[0]
        png = processor.postprocess(decoded, output_type="pil")[0]
        fs = {name: retain(root / "arrays" / key / lane, name, value) for name, value in
              (("velocity", v), ("latent4", z4), ("vaeInput", argument), ("decodedRaw", decoded), ("pngRGB", np.asarray(png)))}
        image = root / "images" / (key.replace("/", "--") + "--" + lane + ".png")
        image.parent.mkdir(exist_ok=True)
        png.save(image)
        fields[key + "/" + lane] = fs
        images[key + "/" + lane] = dict(path=str(image), sha256=sha(image), bytes=image.stat().st_size)
        print("CARRIER_FUTURE", key, lane, flush=True)

    with torch.inference_mode():
        for context in CONTEXTS:
            for corner, (za, va) in CORNERS.items():
                state = states[context, za]
                values = dict(latent=state["latent"].to("cuda"), velocity=velocities[context, va].to("cuda"), sigmas=state["sigmas"].to("cuda"))
                key = context + "/" + corner
                frame = Frame.from_values(values, kinds={name: "tensor" for name in values},
                    abis={name: {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride())} for name, value in values.items()},
                    slot_fingerprints={name: ProvidedFingerprint.content_sha256(payload(value)) for name, value in values.items()},
                    program_fingerprint=request["requestSHA256"], metadata={"case": key, "latentSource": sources[context + "/" + za], "velocitySource": sources[context + "/" + va]})
                output, receipt = registry.run(spec.identity, "carrier-crossover", frame)
                frames.append(dict(case=key, input=frame.to_dict(), output=output.to_dict(), receipt=receipt.to_dict()))
                result(key, "forecast", output.values["latent4"], values["velocity"], state["img_ids"], functional=True)
    require(not native_calls and phase["future_opens"] == 0, "native target or historical future used before seal")
    save(root / "carrier-frames.json", frames)
    save(root / "forecasts.json", dict(fields=fields, images=images, hypotheses=request["hypotheses"]))
    external = {str(path.relative_to(root)): dict(sha256=sha(path), bytes=path.stat().st_size)
                for folder in ("arrays", "images") for path in (root / folder).rglob("*") if path.is_file()}
    for name in ("forecasts.json", "carrier-frames.json"):
        path = root / name
        external[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    save(root / "sealed-files.json", external)
    tx = ForecastTransaction(root / "forecast-transaction", model_identity=next(iter(bridge["cases"].values()))["input_cut_after_conditioning_reset"]["record"]["model_identity"],
        input_identity=identity(sources), source_identity=identity(request["sourceHashes"]),
        frontier={"kind": "architectural", "scope": "Known supplied clock3 Z and own v2; synthetic crossover"},
        execution_scope="Original finite scheduler equation and qualified functional VAE before separate native scheduler/VAE targets",
        budgets=dict(max_predictions=1, max_executions=1, max_bytes=100 << 20, max_files=2000, max_wall_ms=600000))
    tx.frontier("source-operands", {"sources": sources, "noNewPrefix": True})
    tx.predict("all-corners", lambda _: {"predictions": {"files": external, "hypotheses": request["hypotheses"], "nativeCallsBeforeSeal": len(native_calls)}})
    seal = tx.seal()
    print("CARRIER_SEALED", seal["seal_sha256"], flush=True)
    comparisons = []

    def evaluate(_):
        phase["released"] = True
        save(root / "release.json", dict(sealSHA256=seal["seal_sha256"], futureOpensBeforeRelease=phase["future_opens"], nativeCallsBeforeRelease=len(native_calls)))
        with torch.inference_mode():
            for context in CONTEXTS:
                for corner, (za, va) in CORNERS.items():
                    state, v = states[context, za], velocities[context, va].to("cuda")
                    sch = scheduler(source, torch, FlowMatchEulerDiscreteScheduler)
                    z4 = sch.step(v, sch.timesteps[3], state["latent"].to("cuda"), return_dict=False)[0]
                    key = context + "/" + corner
                    result(key, "native", z4, v, state["img_ids"], functional=False)
                    comparisons.extend(dict(case=key, port=port, **metrics(load(record), load(fields[key + "/native"][port])))
                                       for port, record in fields[key + "/forecast"].items())
        return {"comparisons": len(comparisons)}

    tx.execute("native-corners", evaluate)
    tx.close()
    for handle in handles:
        handle.remove()
    for name, item in external.items():
        require(sha(root / name) == item["sha256"], "sealed forecast changed")
    accounts = []
    for context in CONTEXTS:
        for port in ("latent4", "decodedRaw", "pngRGB"):
            y = {corner: load(fields[context + "/" + corner + "/native"][port]) for corner in CORNERS}
            effect = y["PP"] - y["QQ"]
            norm2 = float(np.sum(effect * effect))
            for arm in ("PQ", "QP", "PP"):
                delta = y[arm] - y["QQ"]
                accounts.append(dict(context=context, port=port, corner=arm,
                    projection=float(np.sum(delta * effect) / norm2) if norm2 else None,
                    **metrics(delta, effect)))
            interaction = y["PP"] - y["PQ"] - y["QP"] + y["QQ"]
            accounts.append(dict(context=context, port=port, corner="interaction", **metrics(interaction, effect)))
    assets_unchanged(assets)
    save(root / "source-consumed.json", evidence.consumed)
    save(root / "report.json", dict(schema="mdb-flux-carrier-crossover-v1", requestSHA256=request["requestSHA256"],
        modelIdentity=next(iter(bridge["cases"].values()))["input_cut_after_conditioning_reset"]["record"]["model_identity"],
        sealSHA256=seal["seal_sha256"], transaction=verify_transaction(root / "forecast-transaction"),
        fields=fields, images=images, comparisons=comparisons, accounts=accounts, frames=frames, hypotheses=request["hypotheses"],
        numericalProfile=numerical, quarantine=phase, functionalPrimitiveCounts=counts,
        work={"functionalVAE": 8, "nativeVAE": 8, "nativeVAEConvLinear": len(native_calls), "newDenoisers": 0, "newEncoders": 0, "fits": 0},
        externalFilesVerified=len(external), workerSeconds=time.perf_counter()-started,
        mechanics_status="complete", instrument_status="executed", terminal_status="unassessed",
        limitations=["Two known dependent prompt contexts at seed109; four new crossed conditions and four diagonal replays.",
                     "Synthetic operand crossings do not establish natural reachability.",
                     "Exact original-weight shared-primitive forecast; no independent numerical kernels or compact predictor.",
                     "Visual semantic observations are unblinded and separate from numerical measurements."]))
    print("CARRIER_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    execute(parser.parse_args().root)
