"""DDIM trajectories with explicit latent, scheduler, conditioning, and RNG closure."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from typing import Any

import diffusers
import torch
from diffusers import DDIMScheduler, UNet2DModel

from ..contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
from ..core import Adapter, Session
from ..values import clone, digest, identity


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


class DiffusionAdapter(Adapter):
    def __init__(
        self,
        unet: Any,
        scheduler: DDIMScheduler,
        *,
        decoder: Any = None,
        scaling_factor: float = 1.0,
        granularity: str = "step",
    ):
        if not isinstance(scheduler, DDIMScheduler):
            raise ValueError(
                "v1 supports DDIMScheduler only; stateful solvers need another adapter"
            )
        if granularity not in {"step", "operation"}:
            raise ValueError("granularity must be 'step' or 'operation'")
        self.model = unet.eval()
        self.granularity = granularity
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
        configuration_manifest = {
            "unet": dict(unet.config),
            "scheduler": self.scheduler_config,
            "scaling_factor": scaling_factor,
            "decoder": None if decoder is None else dict(decoder.config),
        }
        self._configuration_fingerprint = digest(_stable_configuration(configuration_manifest))
        self.model_identity = identity(unet, _stable_configuration(config))
        self._frozen_versions = _tensor_guards(unet, prefix="unet.") + (
            () if decoder is None else _tensor_guards(decoder, prefix="decoder.")
        )
        self.execution = {
            "family": "diffusion-ddim",
            "adapter": "native-boundary-v2",
            "granularity": granularity,
            "dtype": str(self.dtype),
            "device": {"type": self.device.type, "index": self.device.index},
            "batch": {"size": 1, "guidance": False},
            "environment_versions": {
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "python": platform.python_version(),
            },
            "model_program": "scale_model_input->unet_denoiser->ddim_scheduler_commit",
            "kernel": {"denoiser": type(unet).__name__, "scheduler": "DDIMScheduler"},
            "numeric_scope": {
                "inference_mode": True,
                "autocast": False,
                "scheduler_history": "eta-zero-stateless",
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "torch_num_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "tensor_layout": "state-slot-bound-stride",
            },
            "eta": 0.0,
            "parity": "same-program-exact",
        }

    @classmethod
    def tiny(cls, seed: int = 7, *, granularity: str = "step") -> DiffusionAdapter:
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
            return cls(
                unet,
                DDIMScheduler(num_train_timesteps=32, clip_sample=False),
                granularity=granularity,
            )

    @classmethod
    def from_pipeline(cls, pipeline: Any, *, granularity: str = "step") -> DiffusionAdapter:
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
            granularity=granularity,
        )

    def _verify_frozen_model(self) -> None:
        current = _tensor_guards(self.model, prefix="unet.") + (
            () if self.decoder is None else _tensor_guards(self.decoder, prefix="decoder.")
        )
        if current != self._frozen_versions:
            raise ValueError(
                "model parameters or buffers changed after adapter identity was frozen; "
                "rewrap the model in a new adapter"
            )

    def validate_execution(self) -> None:
        self._verify_frozen_model()
        configuration_manifest = {
            "unet": dict(self.model.config),
            "scheduler": self.scheduler_config,
            "scaling_factor": self.scaling_factor,
            "decoder": None if self.decoder is None else dict(self.decoder.config),
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
        self,
        *,
        steps: int = 4,
        seed: int = 11,
        latents: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> Session:
        self.validate_execution()
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
            "conditioning": (
                None if conditioning is None else conditioning.to(self.device, self.dtype)
            ),
            "timesteps": scheduler.timesteps,
            "step": 0,
            "phase": "denoiser",
            "noise_prediction": None,
            "rng": generator.get_state(),
            "inference_steps": steps,
        }
        return Session(self, state)

    def boundary(self, state: Mapping[str, Any]) -> str:
        if self.granularity == "operation":
            return f"diffusion-step:{state['step']}/{state['phase']}"
        return f"diffusion-step:{state['step']}"

    def point(self, state: Mapping[str, Any]) -> ExecutionPoint:
        return ExecutionPoint(
            family="diffusion-ddim",
            logical_step=state["step"],
            phase=state["phase"],
            operator="unet" if state["phase"] == "denoiser" else "ddim_scheduler",
            edge="before",
            next_operation=(
                "native_unet_denoiser" if state["phase"] == "denoiser" else "ddim_scheduler_commit"
            ),
            local_clock=state["step"],
        )

    def transition(self, state: Mapping[str, Any]) -> TransitionSpec:
        reads = ("latent", "step", "timesteps", "inference_steps")
        if state["phase"] == "denoiser":
            reads += ("conditioning",)
        else:
            reads += ("noise_prediction",)
        writes = (
            ("noise_prediction", "phase")
            if self.granularity == "operation" and state["phase"] == "denoiser"
            else ("latent", "noise_prediction", "phase", "step")
        )
        return TransitionSpec(
            self.point(state).next_operation,
            reads,
            writes,
            consumer="native-diffusion-suffix",
            footprint="declared",
        )

    def surface(self, state: Mapping[str, Any]) -> SurfaceManifest:
        def tensor_schema(value: torch.Tensor | None, *, optional: bool = False) -> dict[str, Any]:
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

        return SurfaceManifest(
            slots=(
                SlotSpec(
                    "latent",
                    role="carrier",
                    writable=state["phase"] == "denoiser",
                    producer="ddim_scheduler",
                    consumer="unet",
                    schema=tensor_schema(state["latent"]),
                ),
                SlotSpec(
                    "conditioning",
                    role="conditioning",
                    writable=state["phase"] == "denoiser" and state["conditioning"] is not None,
                    producer="caller",
                    consumer="unet",
                    schema=tensor_schema(state["conditioning"], optional=True),
                ),
                SlotSpec(
                    "noise_prediction",
                    role="derived",
                    producer="unet",
                    consumer="ddim_scheduler",
                    persistence="authoritative",
                    schema=tensor_schema(state["noise_prediction"], optional=True),
                ),
                SlotSpec(
                    "timesteps",
                    role="schedule",
                    producer="scheduler_setup",
                    consumer="unet_and_scheduler",
                    schema=tensor_schema(state["timesteps"]),
                ),
                SlotSpec(
                    "rng",
                    role="rng",
                    producer="session",
                    consumer="initial_latent",
                    schema=tensor_schema(state["rng"]),
                ),
                SlotSpec(
                    "step",
                    role="clock",
                    producer="ddim_scheduler",
                    consumer="runtime",
                    schema={"kind": "integer", "minimum": 0, "maximum": len(state["timesteps"])},
                ),
                SlotSpec(
                    "phase",
                    role="clock",
                    producer="runtime",
                    consumer="runtime",
                    schema={"kind": "enum", "values": ["denoiser", "scheduler"]},
                ),
                SlotSpec(
                    "inference_steps",
                    role="schedule",
                    producer="session",
                    consumer="scheduler_setup",
                    schema={"kind": "integer", "minimum": 1},
                ),
            ),
            state_schema="saturn-pub-diffusion-ddim-state-v2",
            consumer="native-ddim-suffix",
            horizon="native-diffusion-suffix",
        )

    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        return ("latent",) + (("conditioning",) if state["conditioning"] is not None else ())

    def validate(self, state: Mapping[str, Any]) -> None:
        self._verify_frozen_model()
        expected_keys = {
            "latent",
            "conditioning",
            "timesteps",
            "step",
            "phase",
            "noise_prediction",
            "rng",
            "inference_steps",
        }
        if set(state) != expected_keys:
            raise ValueError("diffusion state schema keys differ from the execution contract")
        latent = state["latent"]
        if (
            latent.ndim != 4
            or latent.shape[0] != 1
            or latent.shape[1] != self.model.config.in_channels
            or latent.device != self.device
        ):
            raise ValueError("invalid diffusion latent geometry")
        if latent.dtype != self.dtype or not torch.isfinite(latent).all():
            raise ValueError("invalid diffusion latent dtype or values")
        scheduler = DDIMScheduler.from_config(self.scheduler_config)
        scheduler.set_timesteps(state["inference_steps"])
        if not torch.equal(state["timesteps"].cpu(), scheduler.timesteps):
            raise ValueError("scheduler timesteps differ from the execution contract")
        if type(state["step"]) is not int or not 0 <= state["step"] <= len(state["timesteps"]):
            raise ValueError("invalid scheduler cursor")
        if type(state["inference_steps"]) is not int or state["inference_steps"] < 1:
            raise ValueError("invalid inference step count")
        if state["phase"] not in {"denoiser", "scheduler"}:
            raise ValueError("invalid diffusion operation phase")
        if self.granularity == "step" and state["phase"] != "denoiser":
            raise ValueError("step granularity cannot restore an operation boundary")
        conditional = hasattr(self.model.config, "cross_attention_dim")
        if conditional != (state["conditioning"] is not None):
            raise ValueError("conditioning must match the UNet family")
        if conditional and (
            state["conditioning"].ndim != 3
            or state["conditioning"].shape[0] != 1
            or state["conditioning"].shape[-1] != self.model.config.cross_attention_dim
            or state["conditioning"].dtype != self.dtype
            or state["conditioning"].device != self.device
            or not torch.isfinite(state["conditioning"]).all()
        ):
            raise ValueError("invalid conditioning")
        prediction = state["noise_prediction"]
        if prediction is not None and (
            prediction.shape != latent.shape
            or prediction.dtype != self.dtype
            or prediction.device != self.device
            or not torch.isfinite(prediction).all()
        ):
            raise ValueError("invalid denoiser prediction")
        if (state["phase"] == "scheduler") != (prediction is not None):
            raise ValueError("denoiser prediction exists only at the scheduler boundary")
        if state["step"] == len(state["timesteps"]) and state["phase"] != "denoiser":
            raise ValueError("completed trajectory cannot retain an uncommitted scheduler value")
        if (
            not isinstance(state["rng"], torch.Tensor)
            or state["rng"].device.type != "cpu"
            or state["rng"].dtype != torch.uint8
        ):
            raise ValueError("invalid RNG state")
        torch.Generator().set_state(state["rng"])

    def write(
        self, state: Mapping[str, Any], writes: Mapping[str, Any], invalidates: tuple[str, ...]
    ) -> Mapping[str, Any]:
        if invalidates:
            raise ValueError("each DDIM step recomputes its native dependencies")
        if state["phase"] != "denoiser":
            raise ValueError("scheduler boundaries are read-only; restore or commit first")
        if "noise_prediction" in writes:
            raise ValueError("denoiser predictions are derived native state")
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
        if out["phase"] == "denoiser":
            kwargs = {}
            if out["conditioning"] is not None:
                kwargs["encoder_hidden_states"] = out["conditioning"].to(self.device, self.dtype)
            noise = self.model(
                scheduler.scale_model_input(latent, timestep), timestep, **kwargs
            ).sample
        else:
            noise = out["noise_prediction"]
        if self.granularity == "operation" and out["phase"] == "denoiser":
            out["noise_prediction"] = noise
            out["phase"] = "scheduler"
        else:
            out["latent"] = scheduler.step(noise, timestep, latent, eta=0.0).prev_sample
            out["noise_prediction"] = None
            out["phase"] = "denoiser"
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
