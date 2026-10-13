"""mrun registry resolution and one resident load, without nested admission."""
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import BenchError


@dataclass(frozen=True)
class BenchModelSpec:
    registry_name: str
    model_id: str
    revision: str
    family: str
    def to_dict(self): return asdict(self)


def resolve_bench_model(name):
    from mrun.models import resolve_model
    spec = resolve_model(name)
    family = str(spec.family)
    if family not in {"qwen2", "mamba", "flux"}: raise BenchError("model is outside qualified bench families")
    return BenchModelSpec(spec.name, spec.hf_id, "local-materialized", family)


def local_checkpoint_root(name):
    from mrun.models import (
        default_hub_root,
        hub_name,
        model_search_roots,
        models_root,
        resolve_model,
        snapshot_dir,
    )
    spec = resolve_model(name)
    if spec.family != "flux": return snapshot_dir(name)
    # AR snapshot_dir requires top-level safetensors. Diffusers checkpoints put
    # weights in component directories; resolve their model_index from the same
    # registry/cache authorities, with an explicit revision instead of mixing.
    candidates = [Path(spec.local_path)] if spec.local_path else []
    candidates += [base/spec.hf_id.rsplit("/", 1)[-1] for base in (models_root(), *model_search_roots())]
    candidates += [base/key for base in (models_root(), default_hub_root()) for key in (hub_name(spec), spec.name)]
    for root in candidates:
        if (root/"model_index.json").is_file(): return root
        reference = root/"refs/main"
        if reference.is_file():
            selected = root/"snapshots"/reference.read_text().strip()
            if (selected/"model_index.json").is_file(): return selected
        snapshots = list((root/"snapshots").glob("*/model_index.json"))
        if len(snapshots) == 1: return snapshots[0].parent
    raise BenchError(f"no single local Diffusers checkpoint resolved for {name}")


def load_bench_model(name, *, device="cuda"):
    import torch
    from mrun.models import load_hf_model, load_tokenizer, resolve_model
    spec = resolve_bench_model(name)
    if spec.family == "qwen2":
        from .ar.model import load_ar_model
        return load_ar_model(name, device=device, attn_implementation="eager")
    if spec.family == "mamba":
        model = load_hf_model(resolve_model(name), device_map={"": device}, dtype=torch.float32, local_files_only=True).eval()
        tokenizer = load_tokenizer(name, local_files_only=True)
        return spec, model, tokenizer
    from diffusers import DiffusionPipeline
    dtype = torch.float16 if name == "illustrious-xl" else torch.bfloat16
    pipeline = DiffusionPipeline.from_pretrained(str(local_checkpoint_root(name)), torch_dtype=dtype,
        low_cpu_mem_usage=True, local_files_only=True)
    from .bench import resident_family
    family = resident_family(pipeline)
    pipeline._saturn_bench_device = torch.device(device)
    pipeline.set_progress_bar_config(disable=True)
    return BenchModelSpec(spec.registry_name, spec.model_id, spec.revision, family), pipeline, None
