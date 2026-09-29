"""Native FLUX.2 Klein block execution, deterministic flow Euler, and VAE decode.

Text-only, batch-one, guidance-one inference; no reference KV, CFG, LoRA, or offload hooks.
Uses the pinned diffusers modules directly, without replaying an executed block prefix.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel

from ..core import Adapter, Session
from ..values import clone, identity


class Flux2KleinAdapter(Adapter):
    def __init__(self, transformer, scheduler, *, pipeline=None):
        if not isinstance(transformer, Flux2Transformer2DModel):
            raise ValueError("expected a native Flux2Transformer2DModel")
        if transformer.config.guidance_embeds:
            raise ValueError("guidance-embedded transformers are outside this adapter")
        if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            raise ValueError("expected flow Euler scheduler")
        if scheduler.config.stochastic_sampling:
            raise ValueError("stochastic flow is outside this deterministic closure")
        self.model = transformer.eval()
        self.pipeline = pipeline
        self.device = next(transformer.parameters()).device
        self.dtype = next(transformer.parameters()).dtype
        self.scheduler_config = dict(scheduler.config)
        self.joint_layers = len(transformer.transformer_blocks)
        self.single_layers = len(transformer.single_transformer_blocks)
        if not self.joint_layers or not self.single_layers:
            raise ValueError("both joint and single blocks are required")
        configuration = {"model": dict(transformer.config), "scheduler": self.scheduler_config}
        if pipeline is not None:
            if pipeline.vae.bn.running_var.device.type != "cpu":
                raise ValueError("wrap with VAE on CPU to bind native phase-offloaded normalization")
            configuration["vae"] = identity(pipeline.vae, dict(pipeline.vae.config))
        self.model_identity = identity(transformer, configuration)
        self.execution = {
            "family": "flux2-klein",
            "adapter": "native-block-v1",
            "dtype": str(self.dtype),
            "device_type": self.device.type,
            "guidance": 1.0,
            "reference_images": False,
            "decoder_normalization_device": "cpu",
            "parity": "same-program-exact",
        }

    @classmethod
    def from_pipeline(cls, pipeline):
        from diffusers import Flux2KleinPipeline

        if not isinstance(pipeline, Flux2KleinPipeline) or not pipeline.config.is_distilled:
            raise ValueError("only distilled Flux2KleinPipeline is supported")
        if any(hasattr(module, "_hf_hook") for module in pipeline.transformer.modules()):
            raise ValueError("remove CPU-offload hooks and place the transformer before wrapping")
        return cls(pipeline.transformer, pipeline.scheduler, pipeline=pipeline)

    @classmethod
    def tiny(cls, seed=7):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = Flux2Transformer2DModel(
                in_channels=8,
                num_layers=3,
                num_single_layers=2,
                attention_head_dim=16,
                num_attention_heads=2,
                joint_attention_dim=24,
                timestep_guidance_channels=16,
                axes_dims_rope=(4, 4, 4, 4),
                guidance_embeds=False,
            )
        return cls(model, FlowMatchEulerDiscreteScheduler())

    def session(self, *, latent, conditioning, img_ids, txt_ids, timesteps, sigmas):
        return Session(
            self,
            {
                "latent": latent.to(self.device, self.dtype),
                "conditioning": conditioning.to(self.device, self.dtype),
                "img_ids": img_ids.to(self.device),
                "txt_ids": txt_ids.to(self.device),
                "timesteps": timesteps.to(self.device),
                "sigmas": sigmas.to(self.device),
                "step": 0,
                "phase": "project",
                "cursor": 0,
                "context": None,
                "text": None,
                "image": None,
            },
        )

    def boundary(self, state):
        prefix = f"diffusion-step:{state['step']}"
        phase, cursor = state["phase"], state["cursor"]
        if phase == "joint" and cursor:
            return f"{prefix}/after:joint.{cursor - 1}"
        if phase == "single":
            return f"{prefix}/after:single.{cursor - 1}"
        return f"{prefix}/{phase}"

    def addresses(self, state):
        slots = ("latent", "conditioning")
        return slots + (("text", "image") if state["text"] is not None else ())

    def validate(self, state: Mapping[str, Any]):
        latent, conditioning = state["latent"], state["conditioning"]
        for tensor, dimension in (
            (latent, self.model.config.in_channels),
            (conditioning, self.model.config.joint_attention_dim),
        ):
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != dimension:
                raise ValueError("invalid FLUX input geometry")
            if tensor.dtype != self.dtype or tensor.device != self.device:
                raise ValueError("FLUX input dtype/device mismatch")
        for name, length in (("img_ids", latent.shape[1]), ("txt_ids", conditioning.shape[1])):
            if tuple(state[name].shape) != (1, length, 4):
                raise ValueError("invalid FLUX position IDs")
        steps = len(state["timesteps"])
        if not steps or state["sigmas"].shape != (steps + 1,):
            raise ValueError("invalid flow schedule geometry")
        for name in ("timesteps", "sigmas"):
            if state[name].device != self.device or state[name].dtype != torch.float32:
                raise ValueError("flow schedule dtype/device mismatch")
        if type(state["step"]) is not int or not 0 <= state["step"] <= steps:
            raise ValueError("invalid flow step cursor")
        phase, cursor = state["phase"], state["cursor"]
        if phase not in ("project", "joint", "single") or type(cursor) is not int:
            raise ValueError("invalid block cursor")
        if phase == "project":
            if cursor != 0 or any(state[name] is not None for name in ("text", "image", "context")):
                raise ValueError("invalid projection closure")
        else:
            limit = self.joint_layers if phase == "joint" else self.single_layers
            if not 0 <= cursor <= limit or (phase == "single" and not cursor):
                raise ValueError("invalid block cursor")
            if state["step"] == steps or not isinstance(state["context"], dict):
                raise ValueError("missing live transformer closure")
            if set(state["context"]) != {"temb", "img_mod", "txt_mod", "single_mod", "rope"}:
                raise ValueError("incomplete transformer closure")
            width = self.model.config.num_attention_heads * self.model.config.attention_head_dim
            for name, length in (("text", conditioning.shape[1]), ("image", latent.shape[1])):
                value = state[name]
                if value is None or tuple(value.shape) != (1, length, width):
                    raise ValueError("invalid route geometry")
                if value.dtype != self.dtype or value.device != self.device:
                    raise ValueError("route dtype/device mismatch")

        def finite(value):
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("non-finite FLUX state")
            if isinstance(value, Mapping):
                for item in value.values():
                    finite(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    finite(item)

        finite(state)

    def write(self, state, writes, invalidates):
        if invalidates:
            raise ValueError("FLUX closure must remain complete")
        allowed = {"latent", "conditioning"} if state["phase"] == "project" else {"text", "image"}
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
        out, model = clone(state), self.model
        phase, cursor = out["phase"], out["cursor"]
        if phase == "project":
            timestep = out["timesteps"][out["step"]].expand(1).to(self.dtype)
            # Match the native forward's /1000 then *1000, including dtype rounding.
            temb = model.time_guidance_embed((timestep / 1000).to(self.dtype) * 1000, None)
            img_rope, txt_rope = (
                model.pos_embed(out["img_ids"][0]),
                model.pos_embed(out["txt_ids"][0]),
            )
            out["context"] = {
                "temb": temb,
                "img_mod": model.double_stream_modulation_img(temb),
                "txt_mod": model.double_stream_modulation_txt(temb),
                "single_mod": model.single_stream_modulation(temb),
                "rope": tuple(
                    torch.cat([text, image], dim=0) for text, image in zip(txt_rope, img_rope)
                ),
            }
            out["image"] = model.x_embedder(out["latent"])
            out["text"] = model.context_embedder(out["conditioning"])
            out["phase"] = "joint"
        elif phase == "joint" and cursor < self.joint_layers:
            context = out["context"]
            out["text"], out["image"] = model.transformer_blocks[cursor](
                hidden_states=out["image"],
                encoder_hidden_states=out["text"],
                temb_mod_img=context["img_mod"],
                temb_mod_txt=context["txt_mod"],
                image_rotary_emb=context["rope"],
                joint_attention_kwargs=None,
            )
            out["cursor"] += 1
        elif phase == "joint" or cursor < self.single_layers:
            index = 0 if phase == "joint" else cursor
            context = out["context"]
            text_length = out["text"].shape[1]
            combined = model.single_transformer_blocks[index](
                hidden_states=torch.cat([out["text"], out["image"]], dim=1),
                encoder_hidden_states=None,
                temb_mod=context["single_mod"],
                image_rotary_emb=context["rope"],
                joint_attention_kwargs=None,
            )
            out["text"], out["image"] = combined[:, :text_length], combined[:, text_length:]
            out["phase"], out["cursor"] = "single", index + 1
        else:
            noise = model.proj_out(model.norm_out(out["image"], out["context"]["temb"]))
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
            scheduler.timesteps, scheduler.sigmas = out["timesteps"], out["sigmas"]
            scheduler.set_begin_index(out["step"])
            out["latent"] = scheduler.step(
                noise, out["timesteps"][out["step"]], out["latent"], return_dict=False
            )[0]
            out["step"] += 1
            out["phase"], out["cursor"] = "project", 0
            out["text"], out["image"], out["context"] = None, None, None
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
        latent = pipe._unpack_latents_with_ids(
            session.read("latent"),
            session._state["img_ids"],
            height // (pipe.vae_scale_factor * 2),
            width // (pipe.vae_scale_factor * 2),
        )
        # Match native phase-offloaded decoding: CPU BN statistics are consumed
        # before the pipeline's decoder hook places the VAE weights.
        mean = pipe.vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(latent)
        std = (
            (
                pipe.vae.bn.running_var.detach().cpu().view(1, -1, 1, 1)
                + pipe.vae.config.batch_norm_eps
            )
            .sqrt()
            .to(latent)
        )
        latent = pipe._unpatchify_latents(latent * std + mean)
        pixels = pipe.vae.decode(latent.to(next(pipe.vae.parameters()).device), return_dict=False)[
            0
        ]
        return pipe.image_processor.postprocess(pixels, output_type="pil")[0]
