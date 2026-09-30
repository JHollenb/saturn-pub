"""Replay a supplied rank-small FLUX.2 text writer through native images and saved cuts.

The package is supplied research data, not bundled or learned by this example. Requires a
distilled Klein checkpoint and NPZ mean/basis/weights/bias arrays; base weights stay frozen.
"""

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

from saturn_pub import Act
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.training import (
    EvaluationScores,
    PromotionObjective,
    PromotionPolicy,
    TrainingTransaction,
)
from saturn_pub.values import describe


class Writer(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.package_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as package:
            for name in ("mean", "basis", "weights", "bias"):
                self.register_buffer(name, torch.tensor(package[name].astype(np.float32)))
        width, rank = self.basis.shape
        if (
            self.mean.shape != (width,)
            or self.bias.shape != (width,)
            or self.weights.shape != (rank, width)
        ):
            raise ValueError("invalid low-rank writer geometry")
        if any(not torch.isfinite(value).all() for value in self.buffers()):
            raise ValueError("non-finite writer package")
        self.gain = torch.nn.Parameter(torch.tensor(0.0))

    def act(self, gain):
        # Seal the payload and dose when constructing the Act; no later training mutation leaks in.
        arrays = {name: value.detach().clone() for name, value in self.named_buffers()}

        def write(slots):
            state = slots["text"]
            mean, basis, weights, bias = (
                arrays[name].to(state.device) for name in ("mean", "basis", "weights", "bias")
            )
            delta = (((state.float() - mean) @ basis) @ weights + bias).to(state.dtype)
            return {
                "text": state + torch.tensor(gain, device=state.device, dtype=state.dtype) * delta
            }

        return Act(
            "flux2.text.low_rank_write",
            ("text",),
            ("text",),
            write,
            parameters={
                "package_sha256": self.package_sha256,
                "dose": gain,
                "payload": describe(arrays),
            },
        )


def count_red(image):
    """Full-resolution four-connected red components; same mask as the source demo.

    A count instrument, not a semantic judge: retained PNGs remain the visual authority.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (red >= 90) & (red - green >= 35) & (red - blue >= 35)
    seen = np.zeros_like(mask)
    areas = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue, area = deque([(int(y), int(x))]), 0
        seen[y, x] = True
        while queue:
            yy, xx = queue.popleft()
            area += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                y2, x2 = yy + dy, xx + dx
                if 0 <= y2 < height and 0 <= x2 < width and mask[y2, x2] and not seen[y2, x2]:
                    seen[y2, x2] = True
                    queue.append((y2, x2))
        if area >= 40:
            areas.append(area)
    return {"observed_count": len(areas), "component_areas": areas}


def load_pipeline(args):
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
        revision=args.revision,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def place_for_blocks(pipe):
    pipe.remove_all_hooks()
    pipe.text_encoder.to("cpu")
    pipe.transformer.to("cuda")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=611)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--expected-count", type=int, default=5)
    parser.add_argument("--output", default="outputs/flux2-hotfix")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--replay", help="saved cut digest; bypasses prompt encoding and feedback selection"
    )
    parser.add_argument("--dose", type=float, default=2.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this real-model example requests CUDA; tiny adapter tests use CPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    store = LocalStore(root / "state")
    writer = Writer(args.package).eval()
    print("phase: load recipient only", flush=True)
    pipe = load_pipeline(args)
    print(
        "weights: "
        + json.dumps(
            {
                name: {
                    "dtype": str(next(module.parameters()).dtype),
                    "bytes": sum(
                        value.numel() * value.element_size() for value in module.parameters()
                    ),
                }
                for name, module in (
                    ("encoder", pipe.text_encoder),
                    ("transformer", pipe.transformer),
                    ("vae", pipe.vae),
                )
            }
        ),
        flush=True,
    )
    reference_pixels = None
    inputs = None
    native_trace = {}
    if not args.replay:
        pipe.enable_model_cpu_offload()

        # Native model offload alone can retain the encoder's allocator blocks while
        # loading the denoiser. Release them at the actual encoder/denoiser boundary.
        def release_encoder(module, _inputs, output):
            module.to("cpu")
            torch.cuda.empty_cache()
            return output

        encoder_handle = pipe.text_encoder.register_forward_hook(release_encoder)

        def release_denoiser(_module, _inputs):
            pipe.transformer.to("cpu")
            torch.cuda.empty_cache()

        decoder_handle = pipe.vae.register_forward_pre_hook(release_denoiser)

        def memory_marker(name):
            def mark(_module, _inputs):
                print(
                    f"native phase: {name} allocated={torch.cuda.memory_allocated() / 2**20:.1f} MiB reserved={torch.cuda.memory_reserved() / 2**20:.1f} MiB",
                    flush=True,
                )

            return mark

        phase_handles = [
            module.register_forward_pre_hook(memory_marker(name))
            for name, module in (
                ("encoder", pipe.text_encoder),
                ("transformer", pipe.transformer),
                ("vae", pipe.vae),
            )
        ]
        native_step = {"value": -1}

        def trace_enter(_module, _inputs, kwargs):
            native_step["value"] += 1
            if native_step["value"] == 0:
                native_trace["inputs"] = describe(
                    {
                        name: kwargs[name]
                        for name in (
                            "hidden_states",
                            "encoder_hidden_states",
                            "timestep",
                            "img_ids",
                            "txt_ids",
                        )
                    }
                )

        trace_handles = [pipe.transformer.register_forward_pre_hook(trace_enter, with_kwargs=True)]

        def trace_output(name):
            def trace(_module, _inputs, output):
                if native_step["value"] == 0:
                    native_trace[name] = describe(output)

            return trace

        for index, block in enumerate(pipe.transformer.transformer_blocks):
            trace_handles.append(block.register_forward_hook(trace_output(f"joint.{index}")))
        for index, block in enumerate(pipe.transformer.single_transformer_blocks):
            trace_handles.append(block.register_forward_hook(trace_output(f"single.{index}")))
        print("phase: ordinary pipeline reference", flush=True)
        captured = {}

        def capture_conditioning(_pipeline, _step, _timestep, values):
            if not captured:
                captured["conditioning"] = values["prompt_embeds"].detach().cpu()
            captured["final_latent"] = values["latents"].detach().cpu()
            return values

        initial = torch.randn(
            (1, 128, args.size // 16, args.size // 16),
            generator=torch.Generator().manual_seed(args.seed),
            dtype=torch.float32,
        )
        reference = (
            pipe(
                prompt=args.prompt,
                height=args.size,
                width=args.size,
                num_inference_steps=4,
                guidance_scale=1.0,
                latents=initial,
                max_sequence_length=512,
                callback_on_step_end=capture_conditioning,
                callback_on_step_end_tensor_inputs=["prompt_embeds", "latents"],
            )
            .images[0]
            .convert("RGB")
        )
        reference.save(root / "reference.png")
        reference_pixels = reference.tobytes()
        pipe.maybe_free_model_hooks()
        for module in (pipe.text_encoder, pipe.transformer, pipe.vae):
            module.to("cpu")
        torch.cuda.empty_cache()
        # Reuse the actual reference's conditioner output; do not encode the same
        # prompt twice or acquire a second set of encoder workspaces.
        embeds = captured["conditioning"].to("cuda")
        txt_ids = pipe._prepare_text_ids(embeds).to("cuda")
        latent, img_ids = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels // 4,
            args.size,
            args.size,
            embeds.dtype,
            "cuda",
            None,
            latents=initial,
        )
        sigmas = (
            None
            if pipe.scheduler.config.get("use_flow_sigmas", False)
            else np.linspace(1.0, 0.25, 4)
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            4,
            "cuda",
            sigmas=sigmas,
            mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=4),
        )
        inputs = dict(
            latent=latent,
            conditioning=embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=timesteps,
            sigmas=pipe.scheduler.sigmas,
        )
        encoder_handle.remove()
        decoder_handle.remove()
        for handle in phase_handles:
            handle.remove()
        for handle in trace_handles:
            handle.remove()
    place_for_blocks(pipe)
    print("phase: bind native adapter identity", flush=True)
    adapter = Flux2KleinAdapter.from_pipeline(pipe)
    if args.replay:
        parent = store.load(args.replay, device="cuda")
        from saturn_pub import Session

        session = Session(adapter, parent.payload, parent=parent.parent)
        session.restore(parent)
    else:
        session = adapter.session(**inputs)
        state = session.capture(retain=False).payload
        input_match = (
            describe(
                {
                    "hidden_states": state["latent"],
                    "encoder_hidden_states": state["conditioning"],
                    "timestep": state["timesteps"][0].expand(1).to(adapter.dtype) / 1000,
                    "img_ids": state["img_ids"],
                    "txt_ids": state["txt_ids"],
                }
            )
            == native_trace["inputs"]
        )
        print(f"native input exact: {input_match}", flush=True)
        session.continue_()  # projection
        for index in range(3):
            session.continue_()
            same = (
                describe((session.read("text"), session.read("image")))
                == native_trace[f"joint.{index}"]
            )
            print(f"native block exact: joint.{index}={same}", flush=True)
        parent = session.capture()
        store.save(parent)
    records = {}
    payload = parent.payload
    print(f"cut: {parent.boundary} text={list(payload['text'].shape)}", flush=True)

    def render(name, gain=None, cut=parent):
        branch = session.fork(cut)
        if gain is not None:
            branch.apply(writer.act(float(gain)))
        adapter.finish(branch)
        if name == "native" and native_trace:
            probe = session.fork(cut)
            for index in range(3, adapter.joint_layers):
                probe.continue_()
                same = (
                    describe((probe.read("text"), probe.read("image")))
                    == native_trace[f"joint.{index}"]
                )
                print(f"native block exact: joint.{index}={same}", flush=True)
            for index in range(adapter.single_layers):
                probe.continue_()
                same = (
                    describe(torch.cat([probe.read("text"), probe.read("image")], dim=1))
                    == native_trace[f"single.{index}"]
                )
                print(f"native block exact: single.{index}={same}", flush=True)
        # Decode is a separate native phase. Keep only its weights resident while
        # convolution workspaces are live; model/state values are preserved exactly.
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        pipe.vae.to("cuda")
        image = adapter.decode(branch, height=args.size, width=args.size).convert("RGB")
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        pipe.transformer.to("cuda")
        image.save(root / f"{name}.png")
        record = {
            **count_red(image),
            "dose": gain,
            "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "final_latent": describe(branch.read("latent")),
            "cut": cut.fingerprint,
        }
        for receipt in branch.receipts:
            store.receipt(receipt)
        records[name] = record
        print(f"branch: {name} count={record['observed_count']} dose={gain}", flush=True)
        return branch, image, record

    native, native_image, native_record = render("native")
    parity = reference_pixels is None or native_image.tobytes() == reference_pixels
    if not parity:
        final = native.read("latent").cpu()
        pixels = np.asarray(native_image, dtype=np.int16)
        reference_array = np.asarray(reference, dtype=np.int16)
        diagnostics = {
            "latent_exact": torch.equal(final, captured["final_latent"]),
            "latent_max_error": (final.float() - captured["final_latent"].float())
            .abs()
            .max()
            .item(),
            "pixel_mad": float(np.abs(pixels - reference_array).mean()),
            "pixel_max_error": int(np.abs(pixels - reference_array).max()),
        }
        print("parity diagnostics: " + json.dumps(diagnostics), flush=True)
        raise RuntimeError("decomposed FLUX path differs from ordinary native pipeline RGB")
    training = []
    if not args.replay:
        cursor = {"proposal": 0.0}

        def restore_cursor(value):
            cursor.clear()
            cursor.update(value)

        controller = TrainingTransaction(
            controller_id="supplied-flux2-writer-dose",
            model=writer,
            optimizer=torch.optim.SGD(writer.parameters(), lr=1.0),
            cursor_capture=lambda: dict(cursor),
            cursor_restore=restore_cursor,
            policy=PromotionPolicy(
                "lexicographic",
                (
                    PromotionObjective("validation", "count_error", "maximize"),
                    PromotionObjective("autonomous_rollout", "count_exact", "maximize"),
                    PromotionObjective("trace_endpoints", "dose_nearness", "maximize"),
                ),
            ),
            max_interval_steps=1,
        )
        evaluations = 0

        def evaluate():
            nonlocal evaluations
            gain = float(writer.gain.detach())
            evaluations += 1
            _, _, measured = render(f"feedback-{evaluations}-dose-{gain:g}", gain)
            error = abs(measured["observed_count"] - args.expected_count)
            return EvaluationScores(
                validation={"count_error": -float(error)},
                autonomous_rollout={"count_exact": float(error == 0)},
                trace_endpoints={"dose_nearness": -abs(gain - 1.0)},
                evidence={"dose": gain, "observed_count": measured["observed_count"]},
            )

        training.append(controller.establish_baseline(evaluate).to_dict())
        for gain in (0.5, 1.0, 2.0, 4.0):

            def train_step(_step, target=gain):
                controller.optimizer.zero_grad()
                (0.5 * (writer.gain - target).square()).backward()
                controller.optimizer.step()
                cursor["proposal"] = target
                return {"proposed_dose": target}

            training.append(
                controller.run_interval(
                    interval_id=f"dose-{gain:g}",
                    steps=1,
                    train_step=train_step,
                    evaluator=evaluate,
                ).to_dict()
            )
        selected = float(writer.gain.detach())
    else:
        selected = args.dose
    candidate, _, _ = render("patched", selected)
    zero, _, _ = render("zero-dose", 0.0)
    if not native.compare(zero)["equal_payload"]:
        raise RuntimeError("zero-dose changed native continuation")
    if not args.replay:
        later = session.fork(parent)
        while later.inspect().boundary != "diffusion-step:1/after:joint.2":
            later.continue_()
        render("wrong-time", selected, later.capture())
    unchanged = session.capture(retain=False).fingerprint == parent.fingerprint
    session.commit(candidate)
    restoration = session.restore(parent).to_dict()
    uninstall, _, _ = render("uninstalled")
    if not unchanged or not native.compare(uninstall)["equal_payload"]:
        raise RuntimeError("parent or uninstall differs from native")
    report = {
        "model_revision": args.revision,
        "model_identity": adapter.model_identity,
        "package_sha256": writer.package_sha256,
        "parent": parent.fingerprint,
        "boundary": parent.boundary,
        "selected_dose": selected,
        "ordinary_pipeline_rgb_exact": None if args.replay else parity,
        "parent_unchanged": unchanged,
        "zero_dose_exact": True,
        "uninstall_exact": True,
        "restore": restoration,
        "records": records,
        "training": training,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "claim_boundary": "Supplied historical writer and target feedback; one discovery context. No generalization claim.",
        "mechanics_status": "verified",
        "terminal_status": "not-assessed",
        "patch_learning": "Payload supplied; controller selects only scalar dose using labelled RGB count feedback.",
    }
    (root / ("replay-report.json" if args.replay else "report.json")).write_text(
        json.dumps(report, indent=2)
    )
    print(f"complete: selected dose={selected:g}", flush=True)


if __name__ == "__main__":
    main()
