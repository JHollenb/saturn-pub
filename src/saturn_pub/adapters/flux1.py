"""Native FLUX.1 block execution, deterministic flow Euler, and VAE decode.

Text-only, batch-one inference over the native ``FluxTransformer2DModel`` stepped one
projection / joint (double) block / single block / readout at a time, followed by the
native ``FlowMatchEulerDiscreteScheduler`` update and the native ``AutoencoderKL`` decode.

FLUX.1 carries two text conditioners: a T5-XXL sequence (``encoder_hidden_states``) and a
CLIP pooled vector (``pooled_projections``). ``schnell`` has no guidance embedding;
guidance-distilled variants (``dev``) gate a guidance embedding on
``transformer.config.guidance_embeds``. Weights may execute resident on the device or, via
adapter-owned block-streamed residency, from host memory one native module at a time — see
:mod:`saturn_pub.adapters._residency`.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from typing import Any

import diffusers
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel

from ..contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
from ..core import Adapter, Session
from ..values import clone, digest, identity
from ._residency import BlockResidency


def _stable_configuration(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {name: _stable_configuration(item, key=name) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [_stable_configuration(item) for item in value]
        return sorted(items) if key == "_use_default_values" else items
    return value


def _tensor_guards(module: Any, *, prefix: str = "") -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            f"{prefix}{name}",
            id(value),
            value._version,
            value.data_ptr(),
            value.untyped_storage().data_ptr(),
        )
        for name, value in (*module.named_parameters(), *module.named_buffers())
    )


class Flux1Adapter(Adapter):
    def __init__(
        self,
        transformer,
        scheduler,
        *,
        pipeline=None,
        granularity="block",
        residency="resident",
        device=None,
        pin_host=False,
    ):
        if not isinstance(transformer, FluxTransformer2DModel):
            raise ValueError("expected a native FluxTransformer2DModel")
        if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError("expected flow Euler scheduler")
        if scheduler.config.stochastic_sampling:
            raise ValueError("stochastic flow is outside this deterministic closure")
        if granularity not in {"block", "operation"}:
            raise ValueError("granularity must be 'block' or 'operation'")
        self.model = transformer.eval()
        self.granularity = granularity
        self.pipeline = pipeline
        self.dtype = next(transformer.parameters()).dtype
        if residency == "streamed":
            target = (
                device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._residency = BlockResidency(target, mode="streamed", pin_host=pin_host)
        else:
            target = device if device is not None else next(transformer.parameters()).device
            self._residency = BlockResidency(target, mode="resident")
        self._residency.place(transformer)
        self.device = self._residency.device
        self.guidance_embeds = bool(transformer.config.guidance_embeds)
        self.rope_axes = len(transformer.config.axes_dims_rope)
        self.scheduler_config = dict(scheduler.config)
        self.joint_layers = len(transformer.transformer_blocks)
        self.single_layers = len(transformer.single_transformer_blocks)
        if not self.joint_layers or not self.single_layers:
            raise ValueError("both joint and single blocks are required")
        configuration = {"model": dict(transformer.config), "scheduler": self.scheduler_config}
        if pipeline is not None:
            configuration["vae"] = identity(pipeline.vae, dict(pipeline.vae.config))
        configuration_manifest = {
            "model": dict(transformer.config),
            "scheduler": self.scheduler_config,
            "vae": None if pipeline is None else dict(pipeline.vae.config),
        }
        self._configuration_fingerprint = digest(_stable_configuration(configuration_manifest))
        self.model_identity = identity(transformer, _stable_configuration(configuration))
        self._frozen_versions = _tensor_guards(transformer, prefix="transformer.") + (
            () if pipeline is None else _tensor_guards(pipeline.vae, prefix="vae.")
        )
        self.execution = {
            "family": "flux1",
            "adapter": "native-boundary-v2",
            "granularity": granularity,
            "dtype": str(self.dtype),
            "device": {"type": self.device.type, "index": self.device.index},
            "batch": {"size": 1, "reference_images": False},
            "environment_versions": {
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "python": platform.python_version(),
            },
            "model_program": (
                "project->joint_block[*]->single_block[*]->norm/proj_readout->flow_commit"
            ),
            "kernel": {
                "transformer": "FluxTransformer2DModel",
                "scheduler": "FlowMatchEulerDiscreteScheduler",
            },
            "numeric_scope": {
                "inference_mode": True,
                "autocast": False,
                "stochastic_sampling": False,
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "torch_num_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "tensor_layout": "state-slot-bound-stride",
            },
            "guidance_embeds": self.guidance_embeds,
            "reference_images": False,
            "residency": self._residency.to_dict(),
            "parity": "same-program-exact",
        }

    @classmethod
    def from_pipeline(
        cls,
        pipeline,
        *,
        granularity="block",
        residency="resident",
        device=None,
        pin_host=False,
    ):
        from diffusers import FluxPipeline

        if not isinstance(pipeline, FluxPipeline):
            raise ValueError("only the native FluxPipeline (FLUX.1) is supported")
        if any(hasattr(module, "_hf_hook") for module in pipeline.transformer.modules()):
            raise ValueError(
                "remove accelerate offload hooks before wrapping; block-streamed residency "
                "is adapter-owned (pass residency='streamed')"
            )
        return cls(
            pipeline.transformer,
            pipeline.scheduler,
            pipeline=pipeline,
            granularity=granularity,
            residency=residency,
            device=device,
            pin_host=pin_host,
        )

    @classmethod
    def tiny(
        cls,
        seed=7,
        *,
        granularity="block",
        guidance=False,
        residency="resident",
        device=None,
        pin_host=False,
    ):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = FluxTransformer2DModel(
                in_channels=8,
                num_layers=2,
                num_single_layers=2,
                attention_head_dim=16,
                num_attention_heads=2,
                joint_attention_dim=24,
                pooled_projection_dim=16,
                guidance_embeds=guidance,
                axes_dims_rope=(4, 6, 6),
            )
        return cls(
            model,
            FlowMatchEulerDiscreteScheduler(),
            granularity=granularity,
            residency=residency,
            device=device,
            pin_host=pin_host,
        )

    def _verify_frozen_model(self):
        current = _tensor_guards(self.model, prefix="transformer.") + (
            () if self.pipeline is None else _tensor_guards(self.pipeline.vae, prefix="vae.")
        )
        if current != self._frozen_versions:
            raise ValueError(
                "model parameters or buffers changed after adapter identity was frozen; "
                "rewrap the model in a new adapter"
            )

    def validate_execution(self):
        self._verify_frozen_model()
        configuration_manifest = {
            "model": dict(self.model.config),
            "scheduler": self.scheduler_config,
            "vae": None if self.pipeline is None else dict(self.pipeline.vae.config),
        }
        if digest(_stable_configuration(configuration_manifest)) != self._configuration_fingerprint:
            raise ValueError(
                "model configuration changed after adapter identity was frozen; rewrap"
            )
        expected = self.execution["numeric_scope"]
        current = {
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if any(expected[name] != value for name, value in current.items()):
            raise ValueError("live numerical environment drifted from the execution ABI; rewrap")

    def session(
        self, *, latent, conditioning, pooled, img_ids, txt_ids, timesteps, sigmas, guidance=None
    ):
        self.validate_execution()
        if self.guidance_embeds and guidance is None:
            raise ValueError("guidance-distilled FLUX.1 requires a guidance scale")
        if not self.guidance_embeds and guidance is not None:
            raise ValueError("this FLUX.1 transformer has no guidance embedding")
        return Session(
            self,
            {
                "latent": latent.to(self.device, self.dtype),
                "conditioning": conditioning.to(self.device, self.dtype),
                "pooled": pooled.to(self.device, self.dtype),
                "img_ids": img_ids.to(self.device),
                "txt_ids": txt_ids.to(self.device),
                "timesteps": timesteps.to(self.device),
                "sigmas": sigmas.to(self.device),
                "guidance": None if guidance is None else float(guidance),
                "step": 0,
                "phase": "project",
                "cursor": 0,
                "context": None,
                "text": None,
                "image": None,
                "noise_prediction": None,
            },
        )

    def boundary(self, state):
        prefix = f"diffusion-step:{state['step']}"
        phase, cursor = state["phase"], state["cursor"]
        if self.granularity == "operation":
            if phase == "joint":
                return f"{prefix}/joint:{cursor}"
            if phase == "single":
                return f"{prefix}/single:{cursor}"
            return f"{prefix}/{phase}"
        if phase == "joint" and cursor:
            return f"{prefix}/after:joint.{cursor - 1}"
        if phase == "single":
            return f"{prefix}/after:single.{cursor - 1}"
        return f"{prefix}/{phase}"

    def point(self, state):
        phase, cursor = state["phase"], state["cursor"]
        operator = phase
        index = cursor if phase in {"joint", "single"} else None
        next_operation = {
            "project": "native_input_projection",
            "readout": "norm_and_projection",
            "scheduler": "flow_scheduler_commit",
        }.get(phase, f"{phase}_block.{cursor}")
        return ExecutionPoint(
            family="flux1",
            logical_step=state["step"],
            phase=phase,
            operator=operator,
            index=index,
            edge="before",
            next_operation=next_operation,
            local_clock=cursor,
        )

    def transition(self, state):
        phase, cursor = state["phase"], state["cursor"]
        if phase == "project":
            reads = (
                "timesteps",
                "step",
                "img_ids",
                "txt_ids",
                "latent",
                "conditioning",
                "pooled",
                "guidance",
            )
            writes = ("context", "image", "text", "phase")
        elif phase in {"joint", "single"} and (phase == "joint" or cursor < self.single_layers):
            reads = ("context", "image", "text", "phase", "cursor")
            writes = ("image", "text", "phase", "cursor")
        elif self.granularity == "operation" and phase == "readout":
            reads, writes = ("image", "context"), ("noise_prediction", "phase")
        else:
            reads = (
                "noise_prediction",
                "image",
                "context",
                "latent",
                "timesteps",
                "sigmas",
                "step",
            )
            writes = (
                "latent",
                "step",
                "phase",
                "cursor",
                "text",
                "image",
                "context",
                "noise_prediction",
            )
        return TransitionSpec(
            self.point(state).next_operation,
            reads,
            writes,
            consumer="native-flux-suffix",
            footprint="declared",
        )

    def surface(self, state):
        def tensor_schema(value, *, optional=False):
            if value is None:
                return {"kind": "tensor", "optional": optional}
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "optional": optional,
            }

        slots = []
        definitions = (
            ("latent", "carrier", state["phase"] == "project", "flow_scheduler", "transformer"),
            ("conditioning", "conditioning", state["phase"] == "project", "caller", "transformer"),
            ("pooled", "conditioning", state["phase"] == "project", "caller", "transformer"),
            ("img_ids", "address", False, "caller", "rope"),
            ("txt_ids", "address", False, "caller", "rope"),
            ("timesteps", "schedule", False, "caller", "transformer_and_scheduler"),
            ("sigmas", "schedule", False, "caller", "scheduler"),
            (
                "text",
                "carrier",
                state["phase"] in {"joint", "single"},
                "native_block",
                "native_suffix",
            ),
            (
                "image",
                "carrier",
                state["phase"] in {"joint", "single"},
                "native_block",
                "native_suffix",
            ),
            ("noise_prediction", "derived", False, "norm_and_projection", "flow_scheduler"),
        )
        for name, role, writable, producer, consumer in definitions:
            if name in {"conditioning", "pooled"} and state["phase"] != "project":
                writable = True
            slots.append(
                SlotSpec(
                    name,
                    role=role,
                    writable=writable,
                    producer=producer,
                    consumer=consumer,
                    invalidates=(),
                    schema=tensor_schema(state[name], optional=state[name] is None),
                )
            )
        slots.extend(
            (
                SlotSpec(
                    "guidance",
                    role="schedule",
                    producer="caller",
                    consumer="transformer",
                    schema=(
                        {"kind": "scalar", "type": "none"}
                        if state["guidance"] is None
                        else {"kind": "scalar", "type": "float"}
                    ),
                ),
                SlotSpec(
                    "context",
                    role="closure",
                    producer="project",
                    consumer="native_suffix",
                    schema={"kind": "tensor_mapping", "optional": state["context"] is None},
                ),
                SlotSpec(
                    "step",
                    role="clock",
                    producer="flow_scheduler",
                    consumer="runtime",
                    schema={"kind": "integer", "minimum": 0, "maximum": len(state["timesteps"])},
                ),
                SlotSpec(
                    "phase",
                    role="clock",
                    producer="runtime",
                    consumer="runtime",
                    schema={
                        "kind": "enum",
                        "values": ["project", "joint", "single", "readout", "scheduler"],
                    },
                ),
                SlotSpec(
                    "cursor",
                    role="clock",
                    producer="native_block",
                    consumer="runtime",
                    schema={"kind": "integer", "minimum": 0},
                ),
            )
        )
        return SurfaceManifest(
            slots=tuple(slots),
            state_schema="saturn-pub-flux1-state-v1",
            consumer="native-flux1-suffix",
            horizon="native-diffusion-suffix",
        )

    def addresses(self, state):
        slots = ("latent", "conditioning", "pooled")
        return slots + (("text", "image") if state["text"] is not None else ())

    def validate(self, state: Mapping[str, Any]):
        self._verify_frozen_model()
        expected_keys = {
            "latent",
            "conditioning",
            "pooled",
            "img_ids",
            "txt_ids",
            "timesteps",
            "sigmas",
            "guidance",
            "step",
            "phase",
            "cursor",
            "context",
            "text",
            "image",
            "noise_prediction",
        }
        if set(state) != expected_keys:
            raise ValueError("FLUX.1 state schema keys differ from the execution contract")
        latent, conditioning, pooled = state["latent"], state["conditioning"], state["pooled"]
        for tensor, dimension in (
            (latent, self.model.config.in_channels),
            (conditioning, self.model.config.joint_attention_dim),
        ):
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != dimension:
                raise ValueError("invalid FLUX.1 input geometry")
            if tensor.dtype != self.dtype or tensor.device != self.device:
                raise ValueError("FLUX.1 input dtype/device mismatch")
        if (
            pooled.ndim != 2
            or pooled.shape[0] != 1
            or pooled.shape[-1] != self.model.config.pooled_projection_dim
            or pooled.dtype != self.dtype
            or pooled.device != self.device
        ):
            raise ValueError("invalid FLUX.1 pooled projection")
        for name, length in (("img_ids", latent.shape[1]), ("txt_ids", conditioning.shape[1])):
            if tuple(state[name].shape) != (1, length, self.rope_axes):
                raise ValueError("invalid FLUX.1 position IDs")
            if state[name].device != self.device or not torch.isfinite(state[name]).all():
                raise ValueError("invalid FLUX.1 position ID dtype/device/value")
        steps = len(state["timesteps"])
        if not steps or state["sigmas"].shape != (steps + 1,):
            raise ValueError("invalid flow schedule geometry")
        for name in ("timesteps", "sigmas"):
            if state[name].device != self.device or state[name].dtype != torch.float32:
                raise ValueError("flow schedule dtype/device mismatch")
        guidance = state["guidance"]
        if self.guidance_embeds:
            if type(guidance) is not float:
                raise ValueError("guidance-distilled FLUX.1 requires a float guidance scale")
        elif guidance is not None:
            raise ValueError("this FLUX.1 transformer has no guidance embedding")
        if type(state["step"]) is not int or not 0 <= state["step"] <= steps:
            raise ValueError("invalid flow step cursor")
        phase, cursor = state["phase"], state["cursor"]
        allowed_phases = {"project", "joint", "single"}
        if self.granularity == "operation":
            allowed_phases |= {"readout", "scheduler"}
        if phase not in allowed_phases or type(cursor) is not int:
            raise ValueError("invalid block cursor")
        if phase == "project":
            if cursor != 0 or any(state[name] is not None for name in ("text", "image", "context")):
                raise ValueError("invalid projection closure")
        else:
            limit = self.joint_layers if phase == "joint" else self.single_layers
            zero_single = phase == "single" and not cursor and self.granularity != "operation"
            if (
                not 0 <= cursor <= limit
                or zero_single
                or (phase in {"readout", "scheduler"} and not cursor)
            ):
                raise ValueError("invalid block cursor")
            if state["step"] == steps or not isinstance(state["context"], dict):
                raise ValueError("missing live transformer closure")
            if set(state["context"]) != {"temb", "rope"}:
                raise ValueError("incomplete transformer closure")
            width = self.model.config.num_attention_heads * self.model.config.attention_head_dim
            for name, length in (("text", conditioning.shape[1]), ("image", latent.shape[1])):
                value = state[name]
                if value is None or tuple(value.shape) != (1, length, width):
                    raise ValueError("invalid route geometry")
                if value.dtype != self.dtype or value.device != self.device:
                    raise ValueError("route dtype/device mismatch")
        prediction = state["noise_prediction"]
        if prediction is not None and (
            prediction.shape != latent.shape
            or prediction.dtype != self.dtype
            or prediction.device != self.device
            or not torch.isfinite(prediction).all()
        ):
            raise ValueError("invalid FLUX.1 readout prediction")
        if (phase == "scheduler") != (prediction is not None):
            raise ValueError("FLUX.1 prediction exists only at the scheduler boundary")

        def finite(value):
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("non-finite FLUX.1 state")
            if isinstance(value, Mapping):
                for item in value.values():
                    finite(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    finite(item)

        finite(state)

    def write(self, state, writes, invalidates):
        if invalidates:
            raise ValueError("FLUX.1 closure must remain complete")
        if state["phase"] in {"readout", "scheduler"}:
            raise ValueError("readout and scheduler boundaries are read-only")
        allowed = (
            {"latent", "conditioning", "pooled"}
            if state["phase"] == "project"
            else {"text", "image"}
        )
        if not set(writes) <= allowed:
            raise ValueError("write would invalidate the captured transformer closure")
        if state["step"] == len(state["timesteps"]):
            raise ValueError("completed trajectories are read-only")
        return super().write(state, writes, ())

    @torch.inference_mode()
    def advance(self, state):
        self.validate(state)
        if state["step"] == len(state["timesteps"]):
            raise ValueError("trajectory already complete")
        out, model, run = clone(state), self.model, self._residency.run
        phase, cursor = out["phase"], out["cursor"]
        if phase == "project":
            # Match the native forward exactly: the pipeline passes timestep/1000 in the
            # model dtype and the forward multiplies by 1000 again before embedding.
            timestep = out["timesteps"][out["step"]].expand(1).to(self.dtype) / 1000
            timestep = timestep.to(self.dtype) * 1000
            if self.guidance_embeds:
                guidance = torch.full([1], out["guidance"], device=self.device, dtype=torch.float32)
                guidance = guidance.to(self.dtype) * 1000
                temb = run(model.time_text_embed, timestep, guidance, out["pooled"])
            else:
                temb = run(model.time_text_embed, timestep, out["pooled"])
            ids = torch.cat((out["txt_ids"][0], out["img_ids"][0]), dim=0)
            out["context"] = {"temb": temb, "rope": run(model.pos_embed, ids)}
            out["image"] = run(model.x_embedder, out["latent"])
            out["text"] = run(model.context_embedder, out["conditioning"])
            out["phase"] = "joint"
        elif phase == "joint" and cursor < self.joint_layers:
            context = out["context"]
            out["text"], out["image"] = run(
                model.transformer_blocks[cursor],
                hidden_states=out["image"],
                encoder_hidden_states=out["text"],
                temb=context["temb"],
                image_rotary_emb=context["rope"],
                joint_attention_kwargs=None,
            )
            out["cursor"] += 1
            if self.granularity == "operation" and out["cursor"] == self.joint_layers:
                out["phase"], out["cursor"] = "single", 0
        elif phase == "joint" or cursor < self.single_layers:
            index = 0 if phase == "joint" else cursor
            context = out["context"]
            out["text"], out["image"] = run(
                model.single_transformer_blocks[index],
                hidden_states=out["image"],
                encoder_hidden_states=out["text"],
                temb=context["temb"],
                image_rotary_emb=context["rope"],
                joint_attention_kwargs=None,
            )
            out["phase"], out["cursor"] = "single", index + 1
            if self.granularity == "operation" and out["cursor"] == self.single_layers:
                out["phase"] = "readout"
        elif self.granularity == "operation" and phase == "readout":
            out["noise_prediction"] = run(
                model.proj_out, run(model.norm_out, out["image"], out["context"]["temb"])
            )
            out["phase"] = "scheduler"
        else:
            noise = out["noise_prediction"]
            if noise is None:
                noise = run(
                    model.proj_out, run(model.norm_out, out["image"], out["context"]["temb"])
                )
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
            scheduler.timesteps, scheduler.sigmas = out["timesteps"], out["sigmas"]
            scheduler.set_begin_index(out["step"])
            out["latent"] = scheduler.step(
                noise, out["timesteps"][out["step"]], out["latent"], return_dict=False
            )[0]
            out["step"] += 1
            out["phase"], out["cursor"] = "project", 0
            out["text"], out["image"], out["context"] = None, None, None
            out["noise_prediction"] = None
        return out

    def finish(self, session):
        if session.adapter is not self:
            raise ValueError("session belongs to another adapter")
        while session._state["step"] < len(session._state["timesteps"]):
            session.continue_()

    @torch.inference_mode()
    def decode(self, session, *, height, width):
        if self.pipeline is None or session.adapter is not self:
            raise ValueError("decode requires a native pipeline and its own session")
        if session._state["step"] != len(session._state["timesteps"]):
            raise ValueError("decode requires a completed trajectory")
        pipe = self.pipeline
        latent = pipe._unpack_latents(session.read("latent"), height, width, pipe.vae_scale_factor)
        latent = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        pixels = pipe.vae.decode(latent.to(next(pipe.vae.parameters()).device), return_dict=False)[
            0
        ]
        return pipe.image_processor.postprocess(pixels, output_type="pil")[0]
