"""Family-qualified loading boundary. Resident models stay caller owned."""
from pathlib import Path
import json

from .errors import BenchError
from .circuit_runtime import DirectoryArtifactStore
from .bench_publication import recover_locked


def resident_family(resident, family=None):
    native = getattr(getattr(resident, "config", None), "model_type", None)
    name = type(resident).__name__
    detected = ("qwen2" if native == "qwen2" else "mamba" if native == "mamba" else
                "flux" if name in {"FluxPipeline", "Flux2KleinPipeline", "Flux2Pipeline"} else
                "sdxl" if name == "StableDiffusionXLPipeline" else None)
    if detected is None or (family is not None and detected != family):
        raise BenchError(f"unsupported or incompatible resident family: {family or native or name}")
    return detected


class BenchSession:
    """Factory preserving Qwen v1 callers and dispatching native family adapters."""

    @classmethod
    def from_saturn_pub(cls, session, *, project=None, session_id=None, max_steps=1000):
        """Use the public Session executor with MDB's existing debugger protocol."""
        from .saturn_pub_adapter import SaturnPubDebuggerAdapter
        adapter = SaturnPubDebuggerAdapter(session, project=project,
                                           session_id=session_id, max_steps=max_steps)
        return cls.from_backend(adapter, identity=adapter.identity)

    @classmethod
    def from_backend(cls, backend, *, identity, debugger_adapter=None, owns_backend=False,
                     **backend_options):
        """Use a caller runner, DebuggerAdapter, or registered backend factory.

        Native cuts stay owned by the supplied debugger adapter. Forward-only
        runners expose capture/intervention operations without claiming resume.
        """
        from .backend_session import BackendBenchSession
        return BackendBenchSession(backend, identity=identity, debugger_adapter=debugger_adapter,
                                   owns_backend=owns_backend, backend_options=backend_options)

    @classmethod
    def from_resident(cls, model, tokenizer=None, *, family=None, **kwargs):
        resolved = resident_family(model, family)
        if resolved == "qwen2":
            from .bench_qwen import QwenBenchSession
            return QwenBenchSession.from_resident(model, tokenizer, **kwargs)
        from .bench_family import FamilyBenchSession
        return FamilyBenchSession.from_resident(model, tokenizer, family=resolved, **kwargs)

    @classmethod
    def resume(cls, model, tokenizer=None, *, root, model_sha256=None, family=None,
               legacy_numerical_environment=None, _adapter=None):
        resolved = resident_family(model, family)
        path = Path(root)
        writer, selected = recover_locked(path)
        try:
            manifest = DirectoryArtifactStore(path / "artifacts").get(selected["manifest"])
            if manifest.get("schema") == "saturn-open-bench-v1":
                if legacy_numerical_environment is not None:
                    raise BenchError("legacy numerical qualification is Mamba-specific")
                if resolved != "qwen2": raise BenchError("Qwen v1 manifest requires a Qwen2 resident")
                from .bench_qwen import QwenBenchSession
                return QwenBenchSession.resume(model, tokenizer, root=root, model_sha256=model_sha256,
                    _writer=writer, _manifest=manifest)
            from .bench_family import FamilyBenchSession
            return FamilyBenchSession.resume(model, tokenizer, root=root,
                family=resolved, model_sha256=model_sha256, _writer=writer, _manifest=manifest,
                _adapter=_adapter, legacy_numerical_environment=legacy_numerical_environment)
        except BaseException:
            writer.close(); raise


def tensor_digest(tensor):
    from .bench_qwen import tensor_digest as digest
    return digest(tensor)
