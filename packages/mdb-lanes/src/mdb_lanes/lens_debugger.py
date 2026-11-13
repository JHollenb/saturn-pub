"""Cache-free TransformerLens layer execution over MDB's durable control plane.

The caller owns the bridge and weights. Only local, CPU, eval-mode, decoder-only
standard block stacks are admitted here. KV caches and arbitrary Python hooks
are deliberately outside this new numerical/state contract.
"""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from types import SimpleNamespace

import torch

from .backend_torch import TransformerLensRunner
from .bench_debugger import BenchDebuggerAdapter
from .bench_family import FamilyBenchSession, serialized
from .bench_pages import private_tree
from .bench_qwen import tensor_digest
from .circuit_runtime import content_digest
from .debugger_protocol import DebuggerProtocolError
from .errors import BenchError


class _Resident:
    def __init__(self, bridge, underlying):
        self.bridge, self.underlying = bridge, underlying
        self.config = SimpleNamespace(to_dict=lambda: bridge.cfg.to_dict())

    def named_parameters(self):
        return self.underlying.named_parameters()


class _LensExecution:
    family = "transformerlens"
    stages = ("embedding", "layer")
    closure = ("tokens", "attention_mask", "rng", "residual", "logits", "cursor", "hook_edits")
    writable = ("residual", "hooks")
    default_step = "layer"
    contract = "transformerlens-cpu-cachefree-layer-reentry-v1"

    def __init__(self, bridge):
        driver = getattr(bridge, "_driver", None)
        underlying = getattr(driver, "underlying_model", None)
        if underlying is None or not driver.supports("intervention_callbacks"):
            raise BenchError("stateful TransformerLens requires a local v4 PyTorch bridge")
        if type(underlying).__name__ not in {"NativeModel", "GPT2LMHeadModel", "Qwen2ForCausalLM"}:
            raise BenchError(
                "stateful TransformerLens currently supports native TL, GPT-2 and Qwen2/Qwen2.5; other architectures need layer-closure validation"
            )
        if not hasattr(bridge, "blocks") or not len(bridge.blocks):
            raise BenchError("stateful TransformerLens requires a standard decoder blocks stack")
        cfg = bridge.cfg
        if any(
            getattr(cfg, key, False)
            for key in ("is_audio_model", "is_visual_model", "is_multimodal", "is_stateful")
        ):
            raise BenchError(
                "stateful TransformerLens currently supports decoder-only text inference"
            )
        if getattr(getattr(underlying, "config", None), "is_encoder_decoder", False):
            raise BenchError("encoder-decoder execution needs a separate stateful contract")
        self.bridge, self.underlying = bridge, underlying
        self.model = _Resident(bridge, underlying)
        self.runner = TransformerLensRunner(bridge)
        self.device = next(underlying.parameters()).device
        if self.device.type != "cpu":
            raise BenchError(
                "stateful TransformerLens currently requires CPU; GPU is a separate lane"
            )
        self.n_layers = len(bridge.blocks)
        self.config_hash = content_digest(json.loads(json.dumps(cfg.to_dict(), default=str)))
        self.compatibility_mode = bool(getattr(bridge, "compatibility_mode", False))
        self.hook_catalog = tuple(self.runner.capabilities().intervention_points)
        self.environment = self._environment()
        self._torch_hook_signature()
        self.guard = self._versions()
        self._check()
        self.runtime_identity = {
            "contract": self.contract,
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "transformerlens": version("transformer-lens"),
            "bridge_class": type(bridge).__module__ + "." + type(bridge).__qualname__,
            "compatibility_mode": bool(getattr(bridge, "compatibility_mode", False)),
            "execution_source": sha256(inspect.getsource(type(self)).encode()).hexdigest(),
            "bridge_source": sha256(inspect.getsource(type(bridge).forward).encode()).hexdigest(),
            "hooks": list(self.runner.capabilities().intervention_points),
        }
        digest = sha256()
        for name, value in sorted(underlying.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(bytes.fromhex(tensor_digest(value)))
        self.weights_hash = digest.hexdigest()

    def _versions(self):
        values = [*self.underlying.named_parameters(), *self.underlying.named_buffers()]
        return tuple((name, id(t), t._version, str(t.device), str(t.dtype)) for name, t in values)

    @staticmethod
    def _environment():
        return (
            torch.get_num_threads(),
            torch.are_deterministic_algorithms_enabled(),
            torch.get_float32_matmul_precision(),
        )

    def _torch_hook_signature(self):
        result = []
        for name, module in self.underlying.named_modules():
            for slot in (
                "_forward_pre_hooks",
                "_forward_hooks",
                "_backward_pre_hooks",
                "_backward_hooks",
            ):
                for key, callback in getattr(module, slot, {}).items():
                    origin = getattr(callback, "__module__", type(callback).__module__)
                    if not origin.startswith("transformer_lens"):
                        raise BenchError(
                            "persistent Python torch hooks are outside the durable closure"
                        )
                    result.append((name, slot, key, id(callback)))
        return tuple(result)

    def _check(self):
        if self.bridge.training or any(m.training for m in self.underlying.modules()):
            raise BenchError("stateful TransformerLens requires bridge.eval() and fixed weights")
        if self._versions() != self.guard:
            raise BenchError("resident weights/buffers changed; open a new stateful session")
        if any(t.device.type != "cpu" for t in self.underlying.state_dict().values()):
            raise BenchError("stateful TransformerLens needs all parameters/buffers on CPU")
        if self._environment() != self.environment:
            raise BenchError("numerical environment changed; restore the original settings")
        # TL lazily installs its own normalization hooks on first execution.
        # Their lifecycle belongs to the bound library, not to user edits.
        self._torch_hook_signature()
        if (
            bool(getattr(self.bridge, "compatibility_mode", False)) != self.compatibility_mode
            or tuple(self.runner.capabilities().intervention_points) != self.hook_catalog
        ):
            raise BenchError("TransformerLens hook/compatibility configuration changed")
        if (
            content_digest(json.loads(json.dumps(self.bridge.cfg.to_dict(), default=str)))
            != self.config_hash
        ):
            raise BenchError("TransformerLens configuration changed; open a new session")
        if any(point.fwd_hooks or point.bwd_hooks for point in self.bridge.hook_dict.values()):
            raise BenchError(
                "persistent Python hooks are outside the durable closure; use staged hook specs"
            )

    def _call(self, state, inputs, **kwargs):
        self._check()
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.set_rng_state(state["checkpoint"]["rng"])
            result = self.runner.forward(inputs, **kwargs)
            state["checkpoint"]["rng"] = torch.get_rng_state().clone()
        self._check()
        return result

    def prefill(self, tokens, attention_mask=None):
        self._check()
        ids = torch.as_tensor(tokens)
        if ids.dtype not in {torch.int32, torch.int64}:
            raise BenchError("inputs must be integer token IDs")
        ids = ids.long().reshape(1, -1) if ids.ndim == 1 else ids.long()
        if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= self.bridge.cfg.n_ctx:
            raise BenchError("stateful TransformerLens needs one nonempty sequence within n_ctx")
        if bool((ids < 0).any()) or bool((ids >= self.bridge.cfg.d_vocab).any()):
            raise BenchError("token ID outside vocabulary")
        mask = torch.ones_like(ids) if attention_mask is None else torch.as_tensor(attention_mask)
        if ids.device.type != "cpu" or mask.device.type != "cpu":
            raise BenchError("stateful TransformerLens inputs must be CPU tensors")
        if (
            mask.shape != ids.shape
            or not bool(((mask == 0) | (mask == 1)).all())
            or not bool(mask[:, -1].all())
        ):
            raise BenchError(
                "attention_mask must match tokens, be binary and end in an active token"
            )
        return {
            "checkpoint": {
                "tokens": ids.clone(),
                "attention_mask": mask.long().clone(),
                "rng": torch.get_rng_state().clone(),
            },
            "residual": None,
            "logits": None,
            "next_layer": 0,
            "phase": "input",
            "hook_edits": {},
        }

    def cursor(self, state):
        return {
            "phase": state["phase"],
            "next_layer": state["next_layer"],
            "token": 0,
            "sequence_length": state["checkpoint"]["tokens"].shape[1],
        }

    def _hook_layer(self, point):
        if point not in self.runner.capabilities().intervention_points:
            raise BenchError(f"unsupported TransformerLens hook: {point}")
        parts = point.split(".")
        if len(parts) < 3 or parts[0] != "blocks" or not parts[1].isdigit():
            raise BenchError("durable hook edits currently require a decoder-block hook")
        return int(parts[1])

    def hook_spec(self, state, point, spec):
        self._check()
        layer = self._hook_layer(point)
        if state["phase"] == "complete" or layer < state["next_layer"]:
            raise BenchError("hook already passed; restore an earlier cut before editing it")
        spec = json.loads(json.dumps(spec, allow_nan=False))
        self.runner.capabilities().validate((), {point: spec})
        # Shape/device/finite validation occurs against the actual native tensor when fired.
        return spec

    def advance(self, state, kind):
        self._check()
        if state["phase"] == "complete":
            raise BenchError("sequence is complete; restore an earlier cut")
        context = state["checkpoint"]
        if state["phase"] == "input":
            state["residual"] = (
                self._call(
                    state,
                    context["tokens"],
                    stop_at_layer=0,
                    attention_mask=context["attention_mask"],
                )
                .output.detach()
                .clone()
            )
            state["phase"] = "layers"
            if kind == "embedding":
                return {"published": False}
        elif kind == "embedding":
            raise BenchError("embedding already passed")
        if kind != "layer":
            raise BenchError("stateful TransformerLens exposes embedding/layer boundaries")
        layer = state["next_layer"]
        edits = {
            point: spec
            for point, spec in state["hook_edits"].items()
            if self._hook_layer(point) == layer
        }
        options = {"start_at_layer": layer, "attention_mask": context["attention_mask"]}
        if layer + 1 < self.n_layers:
            options["stop_at_layer"] = layer + 1
            result = self._call(state, state["residual"], interventions=edits, **options)
            state["residual"] = result.output.detach().clone()
        else:
            result = self._call(state, state["residual"], interventions=edits, **options)
            state["logits"] = result.output.detach().clone()
            state["residual"] = None
            state["phase"] = "complete"
        state["next_layer"] += 1
        return {"published": state["phase"] == "complete", "hooks_applied": list(edits)}

    def private_state(self, state, _kind):
        return private_tree(state)

    def value(self, state, path):
        value = state.get(path)
        if path in {"token_ids", "attention_mask"}:
            value = state["checkpoint"]["tokens" if path == "token_ids" else path]
        if not isinstance(value, torch.Tensor):
            raise BenchError(f"unavailable TransformerLens address: {path}")
        return value

    def edit(self, state, path, tensor):
        if path != "residual" or state["phase"] != "layers":
            raise BenchError("residual writes require a retained layer boundary")
        state["residual"] = tensor

    def validate(self, state):
        if state["phase"] not in {"input", "layers", "complete"}:
            raise BenchError("invalid TransformerLens phase")
        n = state["next_layer"]
        if type(n) is not int or not 0 <= n <= self.n_layers:
            raise BenchError("invalid TransformerLens layer cursor")
        ids, mask = state["checkpoint"]["tokens"], state["checkpoint"]["attention_mask"]
        if (
            ids.dtype != torch.int64
            or ids.ndim != 2
            or ids.shape[0] != 1
            or not 0 < ids.shape[1] <= self.bridge.cfg.n_ctx
            or bool((ids < 0).any())
            or bool((ids >= self.bridge.cfg.d_vocab).any())
            or mask.shape != ids.shape
            or mask.dtype != torch.int64
            or not bool(((mask == 0) | (mask == 1)).all())
            or not bool(mask[:, -1].all())
            or ids.device.type != "cpu"
            or mask.device.type != "cpu"
        ):
            raise BenchError("invalid retained input geometry")
        rng = state["checkpoint"]["rng"]
        if (
            rng.dtype != torch.uint8
            or rng.shape != torch.get_rng_state().shape
            or rng.device.type != "cpu"
        ):
            raise BenchError("invalid retained RNG state")
        if state["phase"] == "input" and (
            n != 0 or state["residual"] is not None or state["logits"] is not None
        ):
            raise BenchError("invalid input-phase state")
        if state["phase"] == "layers":
            residual = state["residual"]
            if (
                not isinstance(residual, torch.Tensor)
                or n >= self.n_layers
                or residual.shape != (*ids.shape, self.bridge.cfg.d_model)
                or residual.device.type != "cpu"
                or state["logits"] is not None
            ):
                raise BenchError("residual/cursor geometry differs from native stack")
        logits = state["logits"]
        if state["phase"] == "complete" and (
            n != self.n_layers
            or not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != ids.shape
            or logits.device.type != "cpu"
            or state["residual"] is not None
        ):
            raise BenchError("complete sequence has invalid logits/cursor")
        for point, spec in state["hook_edits"].items():
            self._hook_layer(point)
            self.runner.capabilities().validate((), {point: spec})


class TransformerLensBenchSession(FamilyBenchSession):
    """Existing MDB pages, Model OS transactions, history and custody over TL."""

    @classmethod
    def from_resident(
        cls, bridge, tokens, *, root, attention_mask=None, model_id="local-lens", **kwargs
    ):
        adapter = _LensExecution(bridge)
        result = super().from_resident(
            adapter.model,
            root=root,
            family=adapter.family,
            _adapter=adapter,
            model_id=model_id,
            model_sha256=adapter.weights_hash,
            **kwargs,
        )
        try:
            result.prefill(tokens, attention_mask=attention_mask)
            return result
        except BaseException:
            result.close()
            raise

    @classmethod
    def resume(cls, bridge, *, root):
        adapter = _LensExecution(bridge)
        return super().resume(
            adapter.model,
            root=root,
            family=adapter.family,
            _adapter=adapter,
            model_sha256=adapter.weights_hash,
        )

    @serialized
    def prefill(self, tokens, *, attention_mask=None):
        return self._initialize(
            "main",
            lambda: self.adapter.prefill(tokens, attention_mask),
            {"prompt": "explicit-token-ids"},
        )

    @serialized
    def checkpoint(self, label=None, *, receipt_ref=None):
        self.adapter._check()
        return super().checkpoint(label, receipt_ref=receipt_ref)

    def close(self):
        with self._lock:
            if self.closed:
                return
            try:
                self.adapter._check()
            except BenchError:
                # Retain the last acknowledged cut; changed external state cannot
                # publish a new model-bound cut. Always release the session writer.
                self.state = None
                self.pages.pool.clear()
                self.closed = True
                self._writer.close()
                return
            super().close()

    def capabilities(self):
        result = super().capabilities()
        result["operations"] = [
            "prefill",
            "step",
            "generate",
            "inspect",
            "checkpoint",
            "fork",
            "checkout",
            "restore",
            "replay",
            "compare",
            "export",
            "close",
        ]
        result["limits"] = [
            "CPU eval-only",
            "cache-free single sequence",
            "layer boundaries",
            "no KV cache or incremental generation",
            "no persistent Python hooks",
        ]
        result["capture_points"] = list(self.adapter.runner.capabilities().capture_points)
        return result

    @property
    def last_logits(self):
        return self.state.get("logits") if self.state else None

    @serialized
    def inspect(self, path=None):
        if path == "hooks":
            return {"value": deepcopy(self.state["hook_edits"])}
        if path == "token_ids":
            return {"value": self.state["checkpoint"]["tokens"][0].tolist()}
        result = super().inspect(path)
        if path is None:
            result["token_count"] = self.state["checkpoint"]["tokens"].shape[1]
        return result

    @serialized
    def preview_edit(self, path, value, *, mode="replace"):
        if not path.startswith("hooks."):
            return super().preview_edit(path, value, mode=mode)
        if self._edit or mode != "replace":
            raise BenchError("hook specs require replace mode and no pending Act")
        point = path.removeprefix("hooks.")
        spec = self.adapter.hook_spec(self.state, point, value)
        parent = self.checkpoint("act-parent")
        private = private_tree(self.state)
        private["hook_edits"][point] = spec
        self._validate(private)
        payload = self.pages.encode(private)
        act_id = content_digest({"parent": parent, "path": path, "payload": payload, "mode": mode})
        self._edit = {
            "act_id": act_id,
            "parent": parent,
            "path": path,
            "payload": payload,
            "mode": mode,
        }
        return {
            "act_id": act_id,
            "parent_cut": parent,
            "plan": {"writes": [path], "authority": "model-os", "hook": point},
        }


class TransformerLensDebuggerAdapter(BenchDebuggerAdapter):
    """Stateful adapter usable directly with BenchSession.from_backend.

    Construct from a caller-owned bridge and tokens, or resume its durable root.
    The existing local motion controller supplies break/watch and history navigation.
    """

    def __init__(
        self,
        bridge,
        tokens=None,
        *,
        root,
        resume=False,
        attention_mask=None,
        model_id="local-lens",
        max_steps=1024,
        resident_state_bytes=64 << 20,
    ):
        if type(max_steps) is not int or not 1 <= max_steps <= 100000:
            raise BenchError("max_steps must be in 1..100000")
        if resume and (tokens is not None or attention_mask is not None):
            raise BenchError("resume uses retained tokens and mask; do not supply new inputs")
        if not resume and tokens is None:
            raise BenchError("a new stateful TransformerLens session requires token IDs")
        bench = (
            TransformerLensBenchSession.resume(bridge, root=root)
            if resume
            else TransformerLensBenchSession.from_resident(
                bridge,
                tokens,
                root=root,
                attention_mask=attention_mask,
                model_id=model_id,
                resident_state_bytes=resident_state_bytes,
            )
        )
        super().__init__(bench)
        self.identity = bench.identity
        from .local_debugger import LocalMotion

        self.motion = LocalMotion(self, bench=bench, max_steps=max_steps)

    def bind_debugger(self, controller):
        from .local_debugger import LocalDebugTransport

        self.motion.worker = controller
        controller.source_frame_stack = self.bench.source_frames
        for ref in self.bench.cuts:
            cut = self.bench.artifacts.get(ref)
            controller._remember_cut(
                {
                    "cut_id": ref,
                    "parent_cut": cut["parent_cut"],
                    "cursor": cut["cursor"],
                    "branch": cut["branch"],
                },
                name=cut.get("label"),
            )
        return LocalDebugTransport(controller, self.motion)

    def _call(self, name, *args):
        try:
            return super()._call(name, *args)
        except RuntimeError as exc:
            raise DebuggerProtocolError(str(exc)) from exc

    def optional_operations(self):
        return tuple(name for name in super().optional_operations() if name != "discover")

    def optional_operation(self, name, *args, **kwargs):
        try:
            return super().optional_operation(name, *args, **kwargs)
        except RuntimeError as exc:
            raise DebuggerProtocolError(str(exc)) from exc

    def close(self):
        self.bench.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
