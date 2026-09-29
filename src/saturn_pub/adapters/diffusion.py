"""DDIM trajectories with explicit latent, scheduler, conditioning, and RNG closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from diffusers import DDIMScheduler, UNet2DModel

from ..core import Adapter, Session
from ..values import clone, identity


class DiffusionAdapter(Adapter):
    def __init__(
        self,
        unet: Any,
        scheduler: DDIMScheduler,
        *,
        decoder: Any = None,
        scaling_factor: float = 1.0,
    ):
        if not isinstance(scheduler, DDIMScheduler):
            raise ValueError(
                "v1 supports DDIMScheduler only; stateful solvers need another adapter"
            )
        self.model = unet.eval()
        self.scheduler_config = dict(scheduler.config)
        self.decoder = decoder
        self.scaling_factor = scaling_factor
        self.device = next(unet.parameters()).device
        self.dtype = next(unet.parameters()).dtype
        config = {
            "unet": dict(unet.config),
            "scheduler": self.scheduler_config,
            "scaling_factor": scaling_factor,
        }
        if decoder is not None:
            config["decoder"] = identity(decoder, dict(decoder.config))
        self.model_identity = identity(unet, config)
        self.execution = {
            "family": "diffusion-ddim",
            "adapter": "native-step-v1",
            "dtype": str(self.dtype),
            "device_type": self.device.type,
            "eta": 0.0,
            "parity": "same-program-exact",
        }

    @classmethod
    def tiny(cls, seed: int = 7) -> DiffusionAdapter:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            unet = UNet2DModel(
                sample_size=8,
                in_channels=3,
                out_channels=3,
                layers_per_block=1,
                block_out_channels=(8, 16),
                down_block_types=("DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D"),
                norm_num_groups=4,
            )
            return cls(unet, DDIMScheduler(num_train_timesteps=32, clip_sample=False))

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> DiffusionAdapter:
        from diffusers import StableDiffusionPipeline

        if not isinstance(pipeline, StableDiffusionPipeline):
            raise ValueError(
                "v1 supports StableDiffusionPipeline; SDXL/FLUX need separate adapters"
            )
        if not isinstance(pipeline.scheduler, DDIMScheduler):
            raise ValueError("install DDIMScheduler.from_config(pipeline.scheduler.config) first")
        return cls(
            pipeline.unet,
            pipeline.scheduler,
            decoder=pipeline.vae,
            scaling_factor=pipeline.vae.config.scaling_factor,
        )

    def session(
        self,
        *,
        steps: int = 4,
        seed: int = 11,
        latents: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> Session:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be positive")
        scheduler = DDIMScheduler.from_config(self.scheduler_config)
        scheduler.set_timesteps(steps, device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if latents is None:
            size = self.model.config.sample_size
            size = (size, size) if isinstance(size, int) else size
            latents = torch.randn((1, self.model.config.in_channels, *size), generator=generator)
        state = {
            "latent": latents.to(self.device, self.dtype),
            "conditioning": conditioning,
            "timesteps": scheduler.timesteps,
            "step": 0,
            "rng": generator.get_state(),
            "inference_steps": steps,
        }
        return Session(self, state)

    def boundary(self, state: Mapping[str, Any]) -> str:
        return f"diffusion-step:{state['step']}"

    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        return ("latent",) + (("conditioning",) if state["conditioning"] is not None else ())

    def validate(self, state: Mapping[str, Any]) -> None:
        latent = state["latent"]
        if (
            latent.ndim != 4
            or latent.shape[0] != 1
            or latent.shape[1] != self.model.config.in_channels
        ):
            raise ValueError("invalid diffusion latent geometry")
        if latent.dtype != self.dtype or not torch.isfinite(latent).all():
            raise ValueError("invalid diffusion latent dtype or values")
        scheduler = DDIMScheduler.from_config(self.scheduler_config)
        scheduler.set_timesteps(state["inference_steps"])
        if not torch.equal(state["timesteps"].cpu(), scheduler.timesteps):
            raise ValueError("scheduler timesteps differ from the execution contract")
        if not isinstance(state["step"], int) or not 0 <= state["step"] <= len(state["timesteps"]):
            raise ValueError("invalid scheduler cursor")
        conditional = hasattr(self.model.config, "cross_attention_dim")
        if conditional != (state["conditioning"] is not None):
            raise ValueError("conditioning must match the UNet family")
        if conditional and (
            state["conditioning"].ndim != 3
            or state["conditioning"].shape[0] != 1
            or state["conditioning"].shape[-1] != self.model.config.cross_attention_dim
            or state["conditioning"].dtype != self.dtype
            or not torch.isfinite(state["conditioning"]).all()
        ):
            raise ValueError("invalid conditioning")
        torch.Generator().set_state(state["rng"].cpu())

    def write(
        self, state: Mapping[str, Any], writes: Mapping[str, Any], invalidates: tuple[str, ...]
    ) -> Mapping[str, Any]:
        if invalidates:
            raise ValueError("each DDIM step recomputes its native dependencies")
        return super().write(state, writes, ())

    @torch.inference_mode()
    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate(state)
        if state["step"] == len(state["timesteps"]):
            raise ValueError("trajectory already complete")
        out = clone(state)
        scheduler = DDIMScheduler.from_config(self.scheduler_config)
        scheduler.set_timesteps(state["inference_steps"], device=self.device)
        latent = out["latent"].to(self.device)
        timestep = scheduler.timesteps[out["step"]]
        kwargs = {}
        if out["conditioning"] is not None:
            kwargs["encoder_hidden_states"] = out["conditioning"].to(self.device, self.dtype)
        noise = self.model(scheduler.scale_model_input(latent, timestep), timestep, **kwargs).sample
        out["latent"] = scheduler.step(noise, timestep, latent, eta=0.0).prev_sample
        out["step"] += 1
        return out

    @torch.inference_mode()
    def decode(self, session: Session) -> torch.Tensor:
        if session.adapter is not self or session._state["step"] != len(
            session._state["timesteps"]
        ):
            raise ValueError("decode requires this adapter's completed trajectory")
        latent = session.read("latent").to(self.device)
        if self.decoder is not None:
            latent = self.decoder.decode(latent / self.scaling_factor).sample
        return (latent / 2 + 0.5).clamp(0, 1)
