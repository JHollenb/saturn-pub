"""Family compatibility contracts and bounded implementation status."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchAdapterContract:
    family: str
    boundaries: tuple[str, ...]
    mutable_closure: tuple[str, ...]
    status: str

    def qualify(self, capabilities):
        missing = set(self.mutable_closure)-set(capabilities.get("mutable_closure", ()))
        stages = capabilities.get("execution_stages", capabilities.get("boundaries", ()))
        missing_boundaries = set(self.boundaries)-set(stages)
        if capabilities.get("family") != self.family or missing or missing_boundaries:
            raise ValueError(f"incompatible {self.family} adapter; missing mutable closure: {sorted(missing)}; missing boundaries: {sorted(missing_boundaries)}")
        if not capabilities.get("numerical_contract") or not capabilities.get("checkpoint_identity"):
            raise ValueError("adapter must declare numerical contract and checkpoint identity")
        return {"family": self.family, "contract_qualified": True, "bench_status": self.status}


CONTRACTS = {
    "qwen2": BenchAdapterContract("qwen2", ("token", "embedding", "layer", "readout", "sample", "commit"),
        ("residual", "staged_kv", "pending_token", "committed_history", "cursor"), "implemented"),
    "flux": BenchAdapterContract("flux", ("conditioner", "block", "denoise", "scheduler", "vae", "commit"),
        ("conditioner", "latent", "denoiser_frame", "scheduler_state", "rng", "global_rng", "cursor", "image_candidate"), "implemented"),
    "sdxl": BenchAdapterContract("sdxl", ("conditioner", "block", "denoise", "scheduler", "vae", "commit"),
        ("conditioner", "latent", "denoiser_frame", "scheduler_state", "rng", "global_rng", "cursor", "image_candidate"), "implemented"),
    "mamba": BenchAdapterContract("mamba", ("token", "embedding", "layer", "readout", "sample", "commit"),
        ("residual", "conv_state", "ssm_state", "cache_flags", "pending_token", "committed_history", "cursor"), "implemented"),
}
