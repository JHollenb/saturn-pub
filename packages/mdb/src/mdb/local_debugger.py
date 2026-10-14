"""Local shell assembly and finite native motion, without fleet dependencies."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .debugger_features import evaluate_predicate
from .debugger_protocol import DebuggerProtocolError, InMemoryDebugTransport
from .errors import BenchError


def load_checkpoint(path, *, device="cpu", dtype=None, token_ids=None):
    """Load an explicit local HF directory; never discover, download or unpickle."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise BenchError("--local needs a Hugging Face checkpoint directory containing config.json")
    raw = json.loads((path / "config.json").read_text())
    family = raw.get("model_type")
    if family not in {"qwen2", "mamba"}:
        raise BenchError(
            "local stateful debugging supports Qwen2/Qwen2.5 and Mamba v1; "
            "other executors can supply a DebuggerAdapter through from_backend"
        )
    dtype = dtype or ("bf16" if family == "qwen2" else "fp32")
    if family == "qwen2" and dtype != "bf16":
        raise BenchError("Qwen stateful debugging requires BF16; use --dtype bf16")
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise BenchError(
            "install MDB's autoregressive extra: pip install './packages/mdb[autoregressive]'"
        ) from exc
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    # Preserve the raw checkpoint's RoPE authority, as the mrun loader does.
    theta = raw.get("rope_theta")
    corrected = theta is not None and getattr(config, "rope_theta", None) != theta
    if corrected:
        config.rope_theta = float(theta)
    tokenizer = None
    if token_ids is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False
            )
        except (OSError, ValueError) as exc:
            raise BenchError(
                "local checkpoint needs tokenizer files, or supply --token-ids '1,2,3'"
            ) from exc
    selected_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[dtype]
    options = {
        "config": config,
        "dtype": selected_dtype,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if family == "qwen2":
        options["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(str(path), **options).eval().to(device)
    if corrected:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                if name.endswith("inv_freq"):
                    index = torch.arange(buffer.numel(), dtype=torch.float32, device=buffer.device)
                    buffer.copy_((float(theta) ** (-index / buffer.numel())).to(buffer.dtype))
    return model, tokenizer, family


def tiny_qwen(*, seed=113, dtype=None):
    if dtype not in {None, "bf16"}:
        raise BenchError("Qwen stateful debugging requires BF16; use --dtype bf16")
    try:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ImportError as exc:
        raise BenchError(
            "tiny Qwen needs pip install './packages/mdb[autoregressive]'; "
            "mdb debug --tiny counter needs no model libraries"
        ) from exc
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=256,
                attn_implementation="eager",
            )
        )
        .eval()
        .to(torch.bfloat16)
    )
    return model


def tiny_transformerlens(*, seed=113, dtype=None):
    if dtype not in {None, "fp32"}:
        raise BenchError("tiny TransformerLens uses CPU FP32; use --dtype fp32")
    try:
        import torch
        from transformer_lens.config import TransformerBridgeConfig
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError as exc:
        raise BenchError("install MDB's transformerlens extra for this demo") from exc
    torch.set_num_threads(1)
    return TransformerBridge.boot_native(
        TransformerBridgeConfig(
            n_layers=2,
            d_model=16,
            n_ctx=256,
            d_head=8,
            n_heads=2,
            d_mlp=32,
            d_vocab=32,
            act_fn="relu",
            device="cpu",
            seed=seed,
        )
    ).eval()


class LocalMotion:
    """Advance one native boundary at a time; the existing worker owns stops."""

    def __init__(self, adapter, *, bench=None, max_steps=1024, max_new_tokens=32):
        self.adapter, self.bench = adapter, bench
        self.max_steps, self.max_new_tokens = max_steps, max_new_tokens
        self.worker = None
        self.counter = bench is None
        self.default_kind = "step" if self.counter else "layer"
        self.native_step = (
            adapter._callbacks["step"] if self.counter else adapter._callbacks["step_kind"]
        )
        adapter._callbacks["step"] = lambda count: self.step(self.default_kind, count)
        adapter._callbacks["step_kind"] = self.step
        adapter._callbacks["continue_run"] = self.continue_run
        adapter._callbacks["reverse_continue"] = self.reverse_continue
        if bench is not None:
            adapter._callbacks["export"] = bench.export
        # CallbackDebuggerAdapter's standard optional catalog includes these names.

    def cursor(self):
        return self.adapter.status()["cursor"]

    def read_cut(self, ref):
        if self.counter:
            return self.adapter.read_cut(ref)
        return self.bench.artifacts.get(ref)

    def cut_info(self):
        status = dict(self.adapter.status())
        ref = status["cut_id"]
        return {
            "cut_id": ref,
            "parent_cut": self.read_cut(ref).get("parent_cut"),
            "branch": status.get("branch"),
            "cursor": status["cursor"],
        }

    def metadata(self):
        return {**self.cursor(), "branch": self.adapter.status().get("branch")}

    def current_point(self):
        cursor = self.cursor()
        if self.counter:
            return "step", cursor["step"]
        phase = cursor["phase"]
        return ("layer", cursor["next_layer"]) if phase == "layers" else (phase, cursor["token"])

    def safe_point(self, boundary=None, index=None):
        if boundary is None:
            boundary, index = self.current_point()
        return self.worker.safe_point(boundary, index, self.metadata())

    def _primitive(self):
        if self.counter:
            self.native_step(1)
            return "step", self.cursor()["step"]
        if self.bench.family == "transformerlens":
            kind = "embedding" if self.cursor()["phase"] == "input" else "layer"
            self.native_step(kind, 1)
            return kind, self.cursor()["next_layer"]
        cursor = self.cursor()
        phase = cursor["phase"]
        if phase == "token":
            kind = "embedding"
        elif phase == "layers":
            kind = (
                "layer"
                if cursor["next_layer"] < self.bench.model.config.num_hidden_layers
                else "readout"
            )
        elif phase == "readout":
            kind = "sample"
        elif phase == "sampled":
            kind = "commit"
        else:
            raise BenchError(f"unsupported local execution phase: {phase}")
        self.native_step(kind, 1)
        self.adapter.checkpoint("local-step")
        after = self.cursor()
        boundary = "token" if kind == "commit" else kind
        index = after["next_layer"] if kind == "layer" else after["token"]
        return boundary, index

    def _move(self, kind, count):
        if type(count) is not int or count < 1:
            raise DebuggerProtocolError("step count must be positive")
        if kind not in self.adapter.capabilities.boundaries:
            raise DebuggerProtocolError(f"unsupported local step boundary: {kind}")
        self.worker._paused = False
        self.worker._pause_reason = None
        completed = 0
        reason = "step-complete"
        try:
            for _ in range(self.max_steps):
                if (
                    self.bench is not None
                    and self.bench.family == "transformerlens"
                    and self.cursor()["phase"] == "complete"
                ):
                    reason = "complete"
                    break
                boundary, index = self._primitive()
                info = self.cut_info()
                self.worker._remember_cut(info)
                points = [(boundary, index)]
                if boundary == "embedding":
                    points.append(("layer", self.cursor()["next_layer"]))
                elif boundary == "token":
                    points.append(("commit", index))
                for point, position in points:
                    status = self.safe_point(point, position)
                    if status["state"] == "paused":
                        break
                if boundary == kind or (kind == "commit" and boundary == "token"):
                    completed += 1
                if status["state"] == "paused":
                    reason = "break-or-watch"
                    break
                if completed >= count:
                    break
            else:
                reason = "transition-budget"
        finally:
            self.worker._paused = True
        return {**self.cut_info(), "steps": completed, "stopped": reason}

    def step(self, kind, count=1):
        return self._move(kind, count)

    def continue_run(self):
        if self.bench is not None and self.bench.family == "transformerlens":
            return self._move(
                "layer", max(1, self.bench.adapter.n_layers - self.cursor()["next_layer"])
            )
        return self._move("step" if self.counter else "token", self.max_new_tokens)

    def reverse_continue(self, expression):
        path, separator, predicate = expression.partition(" ")
        if not separator:
            raise DebuggerProtocolError("reverse-continue needs PATH PREDICATE, e.g. token < 2")
        original = self.cut_info()["cut_id"]
        ref = self.read_cut(original).get("parent_cut")
        visited = set()
        try:
            for _ in range(self.max_steps):
                if not ref or ref in visited:
                    break
                visited.add(ref)
                self.adapter.restore(ref)
                context = self.metadata()
                value = context[path] if path in context else self.adapter.resolve_value(path)
                if evaluate_predicate(predicate, value, context):
                    return {**self.cut_info(), "exact": True, "predicate": expression}
                ref = self.read_cut(ref).get("parent_cut")
            raise DebuggerProtocolError("no matching retained parent cut within the history budget")
        except BaseException:
            self.adapter.restore(original)
            raise


class LocalDebugTransport(InMemoryDebugTransport):
    """Synchronous pause acknowledgement at an actual current native boundary."""

    def __init__(self, worker, motion):
        super().__init__(worker)
        self.motion = motion
        self.responses = {}

    def request(self, request):
        with self.worker._lock:
            if request.request_id in self.responses:
                return self.responses[request.request_id]
            response = super().request(request)
            if response.ok:
                result = dict(response.result)
                if request.operation == "pause":
                    result = self.motion.safe_point()
                elif request.operation in {
                    "step",
                    "continue",
                    "resume",
                    "restore",
                    "replay",
                    "act.commit",
                    "act.abort",
                    "fork",
                    "checkpoint",
                    "export",
                    "reverse-continue",
                }:
                    info = self.motion.cut_info()
                    self.worker._remember_cut(info)
                    result.update(info)
                if (
                    request.operation in {"status", "info", "pause"}
                    and self.motion.bench is not None
                ):
                    bench = self.motion.bench
                    ids = bench.inspect("token_ids")["value"]
                    result["runtime"]["token_count"] = len(ids)
                if request.operation in {"status", "info"}:
                    result.update(self.motion.cut_info())
                response = replace(response, result=result)
            self.responses[request.request_id] = response
            return response


def connect_local(adapter, *, identity, bench=None, max_steps=1024, max_new_tokens=32):
    from .bench import BenchSession

    if callable(getattr(adapter, "bind_debugger", None)):
        owner = BenchSession.from_backend(adapter, identity=identity)
        return owner, owner.debugger(), adapter.motion
    motion = LocalMotion(adapter, bench=bench, max_steps=max_steps, max_new_tokens=max_new_tokens)
    owner = BenchSession.from_backend(adapter, identity=identity)
    session = owner.debugger()
    motion.worker = session.transport.worker
    if bench is not None:
        motion.worker.source_frame_stack = bench.source_frames
    session.transport = LocalDebugTransport(motion.worker, motion)
    session.pause()
    return owner, session, motion
