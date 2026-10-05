"""Native FLUX/SDXL phase and block continuation behind durable bench cuts."""
from dataclasses import fields
from collections.abc import Mapping
from copy import deepcopy
import inspect
import math
import torch

from .bench_qwen import BenchError
from .bench_qwen import tensor_digest
from .bench_native_forward import NativeForwardMachine


def plain(value):
    if isinstance(value, Mapping): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(plain(v) for v in value)
    if isinstance(value, list): return [plain(v) for v in value]
    return value


class DiffusionBenchAdapter:
    stages = ("conditioner", "block", "denoise", "scheduler", "vae", "commit")
    closure = ("conditioner", "latent", "denoiser_frame", "scheduler_state", "rng", "global_rng", "cursor", "image_candidate")
    writable = ("latent", "residual", "conditioner")
    default_step = "denoise"

    def __init__(self, pipeline, *, family):
        import diffusers
        from mrun.diffusion.phase import PhasePipeline
        self.model, self.family = pipeline, family
        self.pipeline_config = deepcopy(dict(pipeline.config))
        self.component_placements = {}
        self.component_buffers = {}
        for name, component in pipeline.components.items():
            if not isinstance(component, torch.nn.Module): continue
            values = list(component.parameters())
            devices = {v.device for v in values}
            dtypes = {v.dtype for v in values if v.is_floating_point()}
            if len(devices) > 1 or len(dtypes) > 1:
                raise BenchError("mixed-placement/precision parameters need separate qualification")
            if devices:
                self.component_placements[name] = (next(iter(devices)), next(iter(dtypes), None))
            # Native text encoders commonly keep FP32 rotary buffers beside
            # BF16 parameters. Preserve those exact caller-owned buffer bytes.
            self.component_buffers[name] = {key: (value.device, value.detach().cpu().clone())
                for key, value in component.named_buffers()}
        self.denoiser = pipeline.transformer if family == "flux" else pipeline.unet
        self.device = torch.device(getattr(pipeline, "_saturn_bench_device", getattr(pipeline, "_execution_device", self.denoiser.device)))
        self.machine = NativeForwardMachine(self.denoiser)
        if getattr(self.denoiser, "peft_config", None): raise BenchError("active denoiser adapters need separate block qualification")
        self.contract = f"{family}-native-block-scheduler-vae-v2"
        self.runtime_identity = {"diffusers": diffusers.__version__, "torch": torch.__version__,
            "caller_component_placement": {name: {"device": str(device), "dtype": str(dtype)}
                for name, (device, dtype) in self.component_placements.items()},
            "caller_component_buffers": {name: {key: {"device": str(device), "dtype": str(value.dtype),
                "sha256": tensor_digest(value)} for key, (device, value) in buffers.items()}
                for name, buffers in self.component_buffers.items()},
            "pipeline_call_source": inspect.getsource(inspect.unwrap(type(pipeline).__call__)),
            "vae_decode_source": inspect.getsource(inspect.unwrap(type(pipeline.vae).decode)),
            "pipeline_class": type(pipeline).__name__, "denoiser_source": self.machine.source_sha256,
            "scheduler_class": type(pipeline.scheduler).__name__, "scheduler_config": plain(dict(pipeline.scheduler.config)),
            "scheduler_step": inspect.getsource(type(pipeline.scheduler).step),
            "numerical_contract": self.contract}
        self.phase = PhasePipeline.wrap(pipeline, device=str(self.device))

    def checkpoint_tree(self, checkpoint):
        result = {f.name: plain(getattr(checkpoint, f.name)) for f in fields(checkpoint)}
        if self.family == "sdxl":
            result["state"] = {"family": checkpoint.state.family, "slots": plain(checkpoint.slots),
                               "metadata": plain(checkpoint.state.metadata)}
        return result

    def checkpoint_native(self, tree):
        if self.family == "flux":
            from mrun.diffusion.phase import TrajectoryCheckpoint
            return TrajectoryCheckpoint(**tree)
        from mrun.diffusion.nonflux import DiffusionTrajectoryCheckpoint, DiffusionTrajectoryState
        value = dict(tree); state = value["state"]
        value["state"] = DiffusionTrajectoryState(**state)
        return DiffusionTrajectoryCheckpoint(**value)

    def private_state(self, state, kind):
        from .bench_pages import private_tree
        return private_tree(state, readonly_ids=self.readonly_ids(state, kind))

    def readonly_ids(self, state, kind):
        from .bench_pages import tensors
        # Prepared conditioning, IDs, timetable and generator ancestor are native
        # inputs. Only latents and the step cursor change in this checkpoint tree.
        # Scheduler/RNG restoration reads a snapshot and installs a private copy.
        latent = self.latents(state)
        readonly = {id(t) for t in tensors(state["checkpoint"]) if t is not latent}
        readonly.update(id(t) for key in ("scheduler", "rng", "global_rng") for t in tensors(state[key]))
        if kind == "commit" and state["image_candidate"] is not None:
            readonly.add(id(state["image_candidate"]))
        return readonly

    def scheduler_capture(self):
        # Configuration is immutable and bound separately. Retain every live
        # scheduler attribute, including multistep history and begin/step index.
        result = {}
        for key, value in vars(self.model.scheduler).items():
            if key == "_internal_dict": continue
            value = plain(value)
            if value is None or isinstance(value, (int, float, bool, str, list, tuple, dict, torch.Tensor)):
                result[key] = deepcopy(value)
            else:
                raise BenchError(f"unsupported mutable scheduler field {key}: {type(value).__name__}")
        return result

    def scheduler_restore(self, state):
        scheduler = self.model.scheduler
        for key in list(vars(scheduler)):
            if key != "_internal_dict" and key not in state: delattr(scheduler, key)
        for key, value in state.items(): setattr(scheduler, key, deepcopy(value))

    def global_rng_capture(self):
        return {"cpu": torch.get_rng_state().clone(),
                "cuda": torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None}

    def global_rng_restore(self, state):
        torch.set_rng_state(state["global_rng"]["cpu"].cpu())
        if state["global_rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(state["global_rng"]["cuda"].cpu(), self.device)

    @torch.inference_mode()
    def prepare(self, prompt, *, seed, height, width, num_inference_steps, guidance_scale):
        if not isinstance(prompt, str) or not prompt or any(type(n) is not int or n <= 0 for n in (height, width, num_inference_steps)):
            raise BenchError("invalid diffusion prompt or geometry")
        if type(seed) is not int or not 0 <= seed < 2**63: raise BenchError("invalid seed")
        if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)) or not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise BenchError("guidance scale must be finite and nonnegative")
        if self.family == "flux" and guidance_scale > 1: raise BenchError("FLUX block lane currently requires distilled guidance <= 1")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        embeds = self.phase.encode(prompt)
        checkpoint = self.phase.capture_checkpoint(embeds, cut_step=0, height=height, width=width,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
        return {"checkpoint": self.checkpoint_tree(checkpoint), "scheduler": self.scheduler_capture(),
                "rng": generator.get_state(), "context": {"prompt": prompt, "seed": seed, "height": height,
                    "width": width, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale},
                "execution": None, "phase": "conditioner", "image_candidate": None, "image": None,
                "global_rng": self.global_rng_capture()}

    def latents(self, state):
        c = state["checkpoint"]
        return c["latents"] if self.family == "flux" else c["state"]["slots"]["latents"]

    def set_latents(self, state, tensor):
        c = state["checkpoint"]
        if self.family == "flux": c["latents"] = tensor
        else: c["state"]["slots"]["latents"] = tensor

    def cursor(self, state):
        c, e = state["checkpoint"], state["execution"]
        return {"step": c["step_index"], "total_steps": c["total_steps"], "phase": state["phase"],
                "block": e["frame"]["blocks"] if e else None}

    def _begin(self, state):
        if state["execution"] is not None: return
        self.phase._ensure_phase("denoise")
        c = self.checkpoint_native(state["checkpoint"])
        self.scheduler_restore(state["scheduler"])
        latents = self.latents(state)
        if self.family == "flux":
            t = c.timesteps[c.step_index].to(self.device)
            kwargs = {"hidden_states": latents.to(self.denoiser.dtype), "timestep": t.expand(latents.shape[0]).to(latents.dtype)/1000,
                "guidance": None, "encoder_hidden_states": c.prompt_embeds.to(self.device), "txt_ids": c.text_ids.to(self.device),
                "img_ids": c.latent_ids.to(self.device), "joint_attention_kwargs": dict(c.attention_kwargs), "return_dict": False}
            if type(self.model).__name__ == "FluxPipeline":
                kwargs["pooled_projections"] = c.pooled_prompt_embeds.to(self.device)
                if self.denoiser.config.guidance_embeds:
                    kwargs["guidance"] = torch.full((latents.shape[0],), c.guidance_scale, device=self.device, dtype=latents.dtype)
            cfg = False; eta = None
        else:
            from mrun.diffusion.nonflux import _restore_state_inputs, torch_cat
            _, cond, _ = _restore_state_inputs(self.phase, c, latent_override=None, prompt_embeds_override=None, generator=None)
            self.scheduler_restore(state["scheduler"])
            self.global_rng_restore(state)
            cfg = bool(c.state.metadata.get("cfg_enabled")); eta = c.state.metadata.get("eta")
            t = c.slot("schedule_timesteps")[c.step_index].to(self.device)
            model_input = torch_cat(latents, cfg=cfg)
            if callable(getattr(self.model.scheduler, "scale_model_input", None)):
                model_input = self.model.scheduler.scale_model_input(model_input, t)
            kwargs = {"sample": model_input, "timestep": t, "encoder_hidden_states": cond["condition_prompt_embeds"],
                "timestep_cond": cond.get("condition_timestep_cond"), "cross_attention_kwargs": c.attention_kwargs,
                "added_cond_kwargs": {"text_embeds": cond["condition_add_text_embeds"], "time_ids": cond["condition_add_time_ids"]},
                "return_dict": False}
            # The checkpoint helper restores its original generator. Our live
            # RNG cursor advances at each scheduler transaction and is separate
            # from that prepared ancestor; never reset it at a later denoise.
        state["execution"] = {"frame": self.machine.start(kwargs), "timestep": t,
                              "cfg": cfg, "eta": eta, "noise": None}
        state["scheduler"] = self.scheduler_capture()
        state["phase"] = "blocks"

    @torch.inference_mode()
    def _noise(self, state, *, block=False):
        # A restored in-flight frame skips _begin's input construction, but a
        # fresh caller-owned pipeline still needs its native denoiser resident.
        self.phase._ensure_phase("denoise")
        self._begin(state)
        e = state["execution"]
        if e["frame"]["done"]:
            if block: raise BenchError("denoiser blocks already completed")
            return
        self.machine.advance(e["frame"], finish=not block)
        if e["frame"]["done"]:
            noise = e["frame"]["output"][0]
            if e["cfg"]:
                from mrun.diffusion.nonflux import _rescale_noise_cfg
                uncond, text = noise.chunk(2); c = self.checkpoint_native(state["checkpoint"])
                noise = uncond+c.guidance_scale*(text-uncond)
                noise = _rescale_noise_cfg(noise, text, float(c.state.metadata.get("guidance_rescale", 0)))
            e["noise"] = noise
            state["phase"] = "scheduler"

    @torch.inference_mode()
    def advance(self, state, kind):
        self.restore_external(state)
        output = self._advance(state, kind)
        state["global_rng"] = self.global_rng_capture()
        return output

    def _advance(self, state, kind):
        c = state["checkpoint"]
        if state["phase"] == "complete": raise BenchError("image already committed")
        if kind == "conditioner":
            if c["step_index"] or state["execution"]: raise BenchError("conditioner boundary passed")
            return {"published": False}
        if kind == "block":
            if c["step_index"] >= c["total_steps"]: raise BenchError("denoise schedule complete")
            self._noise(state, block=True); return {"published": False}
        if kind == "scheduler":
            if state["phase"] != "scheduler": raise BenchError("scheduler requires completed noise prediction")
            self._scheduler(state); return {"published": False}
        if kind == "vae":
            if c["step_index"] != c["total_steps"] or state["execution"]: raise BenchError("VAE requires complete schedule")
            self._decode(state); return {"published": False}
        if kind == "commit":
            if state["phase"] != "decoded": raise BenchError("image commit requires VAE output")
            state["image"] = state["image_candidate"]; state["image_candidate"] = None; state["phase"] = "complete"
            return {"published": True, "image_committed": True}
        if kind != "denoise": raise BenchError("unsupported diffusion stage")
        if c["step_index"] < c["total_steps"]:
            self._noise(state); self._scheduler(state); return {"published": False}
        if state["phase"] != "decoded": self._decode(state)
        return self._advance(state, "commit")

    def _scheduler(self, state):
        e = state["execution"]
        self.scheduler_restore(state["scheduler"])
        latents = self.latents(state)
        if self.family == "flux":
            updated = self.model.scheduler.step(e["noise"], e["timestep"], latents, return_dict=False)[0].to(latents.dtype)
        else:
            from mrun.diffusion.nonflux import _step_scheduler
            generator = torch.Generator(device=self.device); generator.set_state(state["rng"].cpu())
            updated = _step_scheduler(self.model.scheduler, noise=e["noise"], timestep=e["timestep"],
                                      latents=latents, generator=generator, eta=e["eta"])
            state["rng"] = generator.get_state()
        self.set_latents(state, updated)
        state["checkpoint"]["step_index"] += 1
        state["scheduler"] = self.scheduler_capture(); state["execution"] = None
        state["phase"] = "vae" if state["checkpoint"]["step_index"] == state["checkpoint"]["total_steps"] else "denoise"

    def _decode(self, state):
        self.phase._ensure_phase("denoise")
        c = self.checkpoint_native(state["checkpoint"])
        if self.family == "sdxl":
            from mrun.diffusion.nonflux import _decode_sdxl
            image = _decode_sdxl(self.model, self.latents(state), c, "np")
        elif type(self.model).__name__ == "FluxPipeline":
            from mrun.diffusion.phase import _flux1_decode
            image = _flux1_decode(self.model, self.latents(state), height=c.height, width=c.width, output_type="np")
        else:
            from mrun.diffusion.phase import _flux2_decode
            image = _flux2_decode(self.model, self.latents(state), c.latent_ids.to(self.device),
                                  height=c.height, width=c.width, output_type="np")
        state["image_candidate"] = torch.from_numpy(image.copy()).to(self.device)
        state["phase"] = "decoded"

    def value(self, state, path):
        if path == "latent": return self.latents(state)
        if path == "image" and state["phase"] == "complete": return state["image"]
        if path == "image_preview" and state["phase"] == "decoded": return state["image_candidate"]
        if path == "noise" and state["execution"] and state["execution"]["noise"] is not None: return state["execution"]["noise"]
        if path == "residual" and state["phase"] == "blocks":
            local = state["execution"]["frame"]["locals"]
            return local["hidden_states" if self.family == "flux" else "sample"]
        if path == "conditioner":
            c = state["checkpoint"]
            return c["prompt_embeds"] if self.family == "flux" else c["state"]["slots"]["condition_positive_prompt_embeds"]
        raise BenchError(f"unavailable diffusion address {path}")

    def edit(self, state, path, tensor):
        self.value(state, path)
        if path == "latent":
            if state["execution"] or state["phase"] in {"decoded", "complete"}: raise BenchError("latent edits require denoise boundaries")
            self.set_latents(state, tensor)
        elif path == "residual":
            state["execution"]["frame"]["locals"]["hidden_states" if self.family == "flux" else "sample"] = tensor
        elif path == "conditioner":
            if state["execution"] or state["phase"] in {"decoded", "complete"}: raise BenchError("conditioner edits require denoise boundaries")
            c = state["checkpoint"]
            if self.family == "flux": c["prompt_embeds"] = tensor
            else: c["state"]["slots"]["condition_positive_prompt_embeds"] = tensor
        else: raise BenchError("undeclared diffusion write")

    def validate(self, state):
        c = self.checkpoint_native(state["checkpoint"])
        if c.pipeline_class != type(self.model).__name__: raise BenchError("diffusion checkpoint pipeline mismatch")
        if type(c.step_index) is not int or type(c.total_steps) is not int or not 0 <= c.step_index <= c.total_steps or c.total_steps <= 0:
            raise BenchError("invalid native schedule cursor")
        schedule = c.timesteps if self.family == "flux" else c.slot("schedule_timesteps")
        if len(schedule) != c.total_steps: raise BenchError("native timetable extent differs from schedule")
        latent = self.latents(state)
        if not isinstance(latent, torch.Tensor) or latent.ndim != (3 if self.family == "flux" else 4) or latent.shape[0] != 1:
            raise BenchError("invalid native latent geometry")
        if state["rng"].dtype != torch.uint8 or state["rng"].ndim != 1:
            raise BenchError("invalid native generator state")
        if state["phase"] not in {"conditioner", "blocks", "scheduler", "denoise", "vae", "decoded", "complete"}:
            raise BenchError("invalid diffusion phase")
        e = state["execution"]
        if (state["phase"] in {"blocks", "scheduler"}) != (e is not None):
            raise BenchError("diffusion phase differs from transient closure")
        if (state["phase"] in {"vae", "decoded", "complete"}) != (c.step_index == c.total_steps):
            raise BenchError("diffusion phase differs from schedule completion")
        if e and (e["frame"]["source_sha256"] != self.machine.source_sha256 or
                  not 0 <= e["frame"]["pc"] < len(self.machine.code)):
            raise BenchError("invalid native denoiser frame")
        if state["phase"] == "complete" and state["image"] is None: raise BenchError("committed image missing")
        if state["phase"] == "decoded" and state["image_candidate"] is None: raise BenchError("VAE candidate missing")
        if state["phase"] != "complete" and state["image"] is not None: raise BenchError("image published before commit")
        if e and (state["phase"] == "scheduler") != bool(e["frame"]["done"] and e["noise"] is not None):
            raise BenchError("scheduler requires completed native denoiser output")

    def restore_external(self, state):
        # Pipeline scheduler attributes are mutable native state, too. Rollback
        # restores them immediately, including after failed cut publication.
        self.scheduler_restore(state["scheduler"])
        self.global_rng_restore(state)

    def close(self):
        try:
            self.phase.close()
        finally:
            try:
                # Offload CPU owners before restoring any CUDA owner.
                ordered = sorted(self.component_placements.items(), key=lambda item: item[1][0].type != "cpu")
                for name, (device, dtype) in ordered:
                    getattr(self.model, name).to(device=device, dtype=dtype)
                for name, buffers in self.component_buffers.items():
                    component = getattr(self.model, name)
                    for key, (device, value) in buffers.items():
                        module_name, _, leaf = key.rpartition(".")
                        module = component.get_submodule(module_name) if module_name else component
                        module._buffers[leaf] = value.to(device).clone()
            finally:
                self.model.register_to_config(**self.pipeline_config)
