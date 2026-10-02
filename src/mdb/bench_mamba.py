"""Mamba v1 native block recurrence, with a fully explicit provisional cache.

The adapter executes the installed native block, not a reimplementation of the
SSM equations. Cache state dtypes and per-layer previous-state flags survive cuts.
"""
import inspect
from copy import deepcopy
import torch

from .ar.mamba_state_program import _cache_state, mamba_execution_abi
from .bench_qwen import BenchError


class MemoryInterchangeError(BenchError):
    """Valid native cuts cannot participate in this bounded interchange policy."""


class MambaBenchAdapter:
    family = "mamba"
    stages = ("token", "embedding", "layer", "readout", "sample", "commit")
    closure = ("residual", "conv_state", "ssm_state", "cache_flags", "pending_token", "committed_history", "cursor")
    writable = ("residual", "conv_state", "ssm_state")
    default_step = "token"

    def __init__(self, model, tokenizer=None):
        if model.config.model_type != "mamba":
            raise BenchError("only native Mamba v1 is qualified")
        self.model, self.tokenizer = model.eval(), tokenizer
        self.device = next(model.parameters()).device
        self.layers = model.backbone.layers
        from transformers.models.mamba import modeling_mamba
        self.runtime_identity = mamba_execution_abi(model.config)
        self.runtime_identity["selected_scan_backend"] = "native-slow-forward"
        self.runtime_identity["block_source"] = inspect.getsource(modeling_mamba.MambaBlock.forward)
        self.runtime_identity["mixer_source"] = inspect.getsource(modeling_mamba.MambaMixer.slow_forward)
        self.contract = "mamba-native-single-token-slow-path-v2"

    def _cache(self, pages, *, indices=None):
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=self.model.config)
        selected = set(range(len(pages))) if indices is None else set(indices)
        for index, (layer, page) in enumerate(zip(cache.layers, pages, strict=True)):
            if index not in selected: continue
            if page is not None:
                # Native lazy initialization allocates private cache buffers and
                # copies the source. An extra source clone is unnecessary.
                layer.update_conv_state(page["conv"], state_idx=0,
                                        conv_kernel_size=page["conv"].shape[-1])
                layer.update_recurrent_state(page["ssm"], state_idx=0)
                # Native initialization and previous-state semantics are properties
                # of the cache layer, not inferred from token count.
                for key, value in page["flags"].items():
                    setattr(layer, key, deepcopy(value))
        return cache

    def _pages(self, cache, *, index=None):
        result = []
        selected = cache.layers if index is None else [cache.layers[index]]
        for layer in selected:
            flags = {k: v for k, v in vars(layer).items()
                     if isinstance(v, (bool, int, float, str)) or v is None
                     or isinstance(v, list) and all(isinstance(x, (bool, int, float, str)) or x is None for x in v)}
            # Transfer the private native cache allocation; it is never used for
            # another transition after this callback returns.
            result.append({"conv": _cache_state(layer.conv_states, label="conv").detach(),
                           "ssm": _cache_state(layer.recurrent_states, label="ssm").detach(),
                           "flags": deepcopy(flags)})
        return result if index is None else result[0]

    def private_state(self, state, kind):
        """Native slow_forward mutates only its newly allocated cache layer.

        Wrappers and transient residuals are private. Committed recurrent tensors
        are borrowed read-only; _cache creates the declared writable buffers.
        """
        private = {**state, "token_ids": state["token_ids"][:], "pages": state["pages"][:]}
        e = state["execution"]
        if e:
            private["execution"] = {**e, "pages": e["pages"][:], "hidden": e["hidden"].clone()}
        return private

    def readonly_ids(self, state, kind):
        from .bench_pages import tensors
        hidden = state["execution"]["hidden"] if state["execution"] else None
        return {id(t) for t in tensors(state) if t is not hidden}

    def _slow(self, cache, ids):
        return self._native_slow_call(lambda: self.model(ids, cache_params=cache, use_cache=True, return_dict=True))

    def _native_slow_call(self, callback):
        # Select the qualified lane locally; never alter the module-global kernel
        # availability or another caller's model. Native slow_forward is restored.
        forwards = [layer.mixer.forward for layer in self.layers]
        try:
            for layer in self.layers:
                layer.mixer.forward = layer.mixer.slow_forward
            return callback()
        finally:
            for layer, forward in zip(self.layers, forwards): layer.mixer.forward = forward

    @torch.inference_mode()
    def formed_consumer(self, state, *, model_abi, identity):
        """Read the current formed pages through the existing Model OS bridge.

        This comparison transaction has no authority over the bench's token
        cursor. It reads a committed page set and retains its native receipt.
        """
        if state["execution"] is not None: raise BenchError("formed-page bridge requires a committed token boundary")
        from .ar.mamba_state_runtime import MambaStateRuntime
        from .ar.mamba_state_program import MambaStateProgram, MambaLayerState
        from .model_os_mamba_state import build_mamba_state_package, MambaStateModelBackend, INSTALL_ADDRESS, OP_NOOP
        from .model_os_kernel import ModelOSKernel
        flags = [p.requires_grad for p in self.model.parameters()]
        try:
            runtime = MambaStateRuntime(self.model, model_abi_sha256=model_abi)
            program = MambaStateProgram(model_id=identity["model_id"], revision=identity["revision"],
                source_token_ids=tuple(state["token_ids"]), pending_token_id=state["pending"],
                execution_abi_sha256=runtime.execution_abi_sha256,
                layers=tuple(MambaLayerState(p["conv"].clone(), p["ssm"].clone()) for p in state["pages"]))
            package = build_mamba_state_package(runtime, model_id=identity["model_id"], tokenizer_identity=identity["tokenizer_sha256"])
            def read():
                backend = MambaStateModelBackend(runtime, program, package, suffix=[state["pending"]])
                kernel = ModelOSKernel.install(package, backend)
                result = kernel.execute(updates={INSTALL_ADDRESS: {"mode": OP_NOOP, "layers": []}}, horizon=1)
                return backend.readout_tensor(kernel.current_statecut).clone(), result.receipt.to_payload()
            return self._native_slow_call(read)
        finally:
            for parameter, flag in zip(self.model.parameters(), flags): parameter.requires_grad_(flag)

    @torch.inference_mode()
    def prefill(self, prompt):
        if isinstance(prompt, str):
            if self.tokenizer is None: raise BenchError("text prompt requires a tokenizer")
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        else:
            ids = torch.as_tensor(prompt, dtype=torch.long).reshape(1, -1)
        if ids.numel() == 0 or bool((ids < 0).any()) or bool((ids >= self.model.config.vocab_size).any()):
            raise BenchError("invalid Mamba prompt tokens")
        output = self._slow(None, ids.to(self.device))
        logits = output.logits[:, -1].detach()
        state = {"pages": self._pages(output.cache_params), "token_ids": ids[0].tolist(),
                 "prompt_length": ids.shape[1], "pending": int(logits.argmax(-1)),
                 "logits": logits, "execution": None}
        self.validate(state)
        return state

    def cursor(self, state):
        e = state["execution"]
        return {"token": len(state["token_ids"])-state["prompt_length"],
                "phase": e["phase"] if e else "token", "next_layer": e["next_layer"] if e else None,
                "pending_token": state["pending"]}

    @torch.inference_mode()
    def advance(self, state, kind):
        if kind not in self.stages: raise BenchError(f"unsupported Mamba stage {kind}")
        e = state["execution"]
        if e is None:
            hidden = self.model.backbone.embeddings(torch.tensor([[state["pending"]]], device=self.device))
            e = {"hidden": hidden, "pages": state["pages"], "next_layer": 0,
                 "phase": "layers", "logits": None, "sampled": None}
            state["execution"] = e
        if kind == "embedding":
            if e["next_layer"] != 0 or e["phase"] != "layers": raise BenchError("embedding already passed")
            return {"published": False}
        if kind in {"readout", "sample", "commit"} and e["next_layer"] != len(self.layers):
            raise BenchError("all layers must complete before readout")
        if kind == "commit" and e["sampled"] is None: raise BenchError("commit requires sampling")
        limit = e["next_layer"]+1 if kind == "layer" else len(self.layers)
        if kind == "layer" and e["next_layer"] == len(self.layers): raise BenchError("all layers completed")
        cache = self._cache(e["pages"], indices=range(e["next_layer"], limit))
        while e["next_layer"] < limit:
            i = e["next_layer"]
            block = self.layers[i]
            forward = block.mixer.forward
            try:
                block.mixer.forward = block.mixer.slow_forward
                e["hidden"] = block(e["hidden"], cache_params=cache)
            finally:
                block.mixer.forward = forward
            pages = list(e["pages"])
            # Only the updated layer acquires a new page identity.
            pages[i] = self._pages(cache, index=i)
            e["pages"], e["next_layer"] = pages, i+1
        if kind == "layer": return {"published": False}
        if e["logits"] is None:
            e["logits"] = self.model.lm_head(self.model.backbone.norm_f(e["hidden"]))[:, -1].float()
            e["phase"] = "readout"
        if kind == "readout": return {"published": False}
        e["sampled"] = int(e["logits"].argmax(-1))
        e["phase"] = "sampled"
        if kind == "sample": return {"published": False}
        emitted = state["pending"]
        state.update(pages=e["pages"], token_ids=[*state["token_ids"], emitted],
                     pending=e["sampled"], logits=e["logits"], execution=None)
        return {"published": True, "token_id": emitted, "next_token_id": state["pending"]}

    def value(self, state, path):
        e = state["execution"]
        if path == "residual" and e and e["phase"] == "layers": return e["hidden"]
        if path == "logits": return e["logits"] if e and e["logits"] is not None else state["logits"]
        parts = path.split(".")
        if len(parts) == 2 and parts[0] in {"conv_state", "ssm_state"}:
            try: i = int(parts[1])
            except ValueError: raise BenchError("invalid state-page layer") from None
            if not 0 <= i < len(self.layers): raise BenchError("state-page layer outside model")
            pages = e["pages"] if e else state["pages"]
            return pages[i]["conv" if parts[0] == "conv_state" else "ssm"]
        raise BenchError(f"unavailable Mamba address {path}")

    def edit(self, state, path, tensor):
        if path == "residual":
            self.value(state, path)
            state["execution"]["hidden"] = tensor
        elif path.split(".")[0] in {"conv_state", "ssm_state"}:
            self.value(state, path)
            e = state["execution"]
            if e and e["phase"] != "layers": raise BenchError("formed state edit after readout is invalid")
            pages = list(e["pages"] if e else state["pages"])
            i = int(path.split(".")[1]); page = dict(pages[i])
            page["conv" if path.startswith("conv_state") else "ssm"] = tensor
            pages[i] = page
            if e: e["pages"] = pages
            else: state["pages"] = pages
        else: raise BenchError("undeclared Mamba write")

    def transplant_memory(self, recipient, donor):
        """Private complete native memory with fixed recipient token accounting.

        v1 only interchanges same-position, same-history successors. Cache flags
        must match because they describe the native recurrence's geometry and
        initialization, not the hypothesized variable carried by the tensors.
        """
        self.validate_memory_transplant(recipient, donor)
        return {**recipient, "token_ids": recipient["token_ids"][:], "pages": [
            {"conv": source["conv"].clone(), "ssm": source["ssm"].clone(),
             "flags": deepcopy(source["flags"])} for source in donor["pages"]]}

    def validate_memory_transplant(self, recipient, donor):
        self.validate(recipient); self.validate(donor)
        if (recipient["execution"] is not None or donor["execution"] is not None
            or recipient["token_ids"] != donor["token_ids"]
            or recipient["prompt_length"] != donor["prompt_length"]):
            raise MemoryInterchangeError("successor memory requires matching committed position and history")
        for old, source in zip(recipient["pages"], donor["pages"], strict=True):
            if old is None or source is None or source["flags"] != old["flags"]:
                raise MemoryInterchangeError("incompatible successor cache flags")
            for key in ("conv", "ssm"):
                if (source[key].shape != old[key].shape or source[key].dtype != old[key].dtype
                    or source[key].device != old[key].device or not bool(torch.isfinite(source[key]).all())):
                    raise MemoryInterchangeError("incompatible successor memory tensor contract")

    def validate(self, state):
        if len(state["pages"]) != len(self.layers): raise BenchError("Mamba state layer count mismatch")
        if type(state["pending"]) is not int or not 0 <= state["pending"] < self.model.config.vocab_size:
            raise BenchError("invalid pending token")
        if not 0 < state["prompt_length"] <= len(state["token_ids"]): raise BenchError("invalid token accounting")
        if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in state["token_ids"]):
            raise BenchError("invalid committed token history")
        e = state["execution"]
        if e and (not 0 <= e["next_layer"] <= len(self.layers) or len(e["pages"]) != len(self.layers)):
            raise BenchError("invalid provisional Mamba cursor")
        if e:
            if e["phase"] not in {"layers", "readout", "sampled"} or e["hidden"].shape != (1, 1, self.model.config.hidden_size):
                raise BenchError("invalid provisional residual or stage")
            if e["phase"] != "layers" and e["next_layer"] != len(self.layers):
                raise BenchError("readout stage requires complete recurrent pages")
            if e["phase"] == "sampled" and (type(e["sampled"]) is not int or not 0 <= e["sampled"] < self.model.config.vocab_size):
                raise BenchError("invalid sampled pending token")
            if e["phase"] != "layers" and (e["logits"] is None or e["logits"].shape != (1, self.model.config.vocab_size)):
                raise BenchError("readout closure missing native logits")
        for pages in [state["pages"], *([e["pages"]] if e else [])]:
            for page in pages:
                if page is None: continue
                if page["conv"].shape != (1, self.model.config.intermediate_size, self.model.config.conv_kernel):
                    raise BenchError("invalid convolution state shape")
                if page["ssm"].shape != (1, self.model.config.intermediate_size, self.model.config.state_size):
                    raise BenchError("invalid SSM state shape")
