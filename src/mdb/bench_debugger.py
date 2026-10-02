"""The existing debugger protocol over the bench's authoritative cut registry."""
from collections.abc import Mapping

from mdb.debugger_adapters import AdapterCapabilities, CallbackDebuggerAdapter
from mdb.debugger_protocol import DebuggerProtocolError


class BenchDebuggerAdapter(CallbackDebuggerAdapter):
    def __init__(self, bench):
        self.bench = bench
        metadata = bench.capabilities()
        family = str(metadata["family"])
        self.native_family = family
        generic = hasattr(bench, "adapter")
        runtime = getattr(getattr(bench, "vm", None), "runtime", None)
        default_backend = getattr(runtime, "backend", "native")
        caps = AdapterCapabilities(family="autoregressive" if family in {"qwen2", "mamba"} else family,
            boundaries=tuple(metadata["execution_stages"]),
            inspectable_slots=("token_ids", "logits", "residual", "cursor", "kv_cache", "memory", "symbols", "circuits", "measurements", "frames") if family == "qwen2" else
                ("token_ids", "logits", "residual", "cursor", "conv_state", "ssm_state", "memory", "symbols", "circuits", "measurements", "frames") if family == "mamba" else
                ("latent", "conditioner", "residual", "noise", "image", "image_preview", "cursor", "memory", "symbols", "circuits", "frames"),
            writable_slots=tuple(metadata["writable"]), mutable_closure=tuple(metadata["mutable_closure"]),
            exact_replay=True, state_schema_id=metadata.get("state_schema_id", "saturn-open-bench-cut-v2" if generic else "saturn-open-bench-cut-v1"),
            backend=metadata.get("backend") or ("native" if generic else default_backend),
            consumer=f"{family}-native-continuation", checkpoint_requires_pause=True)
        def inspect(path=None):
            value = bench.inspect(path)
            # BenchSession exposes journal views such as ``measurements`` as
            # lists, while the debugger protocol intentionally returns an
            # object with a nested value for every inspect response.
            return value if isinstance(value, Mapping) else {"value": value}

        callbacks = dict(status=lambda: bench.inspect(), inspect=inspect,
            resolve_value=bench.resolve_value, checkpoint=self._checkpoint, restore=bench.restore,
            preview_act=self._preview, commit_act=bench.commit_edit, abort_act=bench.abort_edit,
            replay=self._replay, step=lambda count: self._step(metadata.get("default_step", "token"), count),
            step_kind=self._step, continue_run=lambda: {"session_id": bench.session_id, "state": "running"},
            diff=lambda a, b: bench.compare(a, b), reverse_continue=lambda cut: bench.restore(cut),
            fork_branch=self._fork, checkout=bench.checkout, discover=bench.discover)
        # Circuit measurement/replay are BenchSession-owned native operations.
        # Keep them out of diffusion adapters even if a future generic bench
        # happens to expose similarly named helpers.
        if family in {"qwen2", "mamba"} and metadata.get("circuit_measurement", {}).get("available", True):
            measure = getattr(bench, "measure_circuit", None)
            replay = getattr(bench, "replay_circuit", None)
            if callable(measure):
                callbacks["measure_circuit"] = measure
            if callable(replay):
                callbacks["replay_circuit"] = replay
        super().__init__(caps, **callbacks)

    def _checkpoint(self, name):
        ref = self.bench.checkpoint(name)
        return {"cut_id": ref, "session_id": self.bench.session_id,
                "branch": self.bench.active, "cursor": self.bench.cursor()}

    def _preview(self, act):
        if act.parent_cut != self.bench.branch.cut:
            raise DebuggerProtocolError("Act parent is not the selected bench cut")
        result = self.bench.preview_edit(act.path, act.value, mode=act.sign)
        # The protocol owns the caller's Act id; map it to the admitted bench id.
        self._act_ids = getattr(self, "_act_ids", {})
        self._act_ids[act.act_id] = result["act_id"]
        self._callbacks["commit_act"] = lambda key: self.bench.commit_edit(self._act_ids.get(key, key))
        self._callbacks["abort_act"] = lambda key: self.bench.abort_edit(self._act_ids.get(key, key))
        return {**result, "act_id": act.act_id}

    def _step(self, kind, count):
        result = self.bench.step(kind, count)
        return {"result": result, **self.bench.inspect()}

    def _replay(self, cut):
        result = self.bench.replay(cut)
        return {"result": result, "cut_id": self.bench.branch.cut, "exact": True,
                "session_id": self.bench.session_id}

    def _fork(self, name):
        cut = self.bench.fork(name)
        return {"cut_id": cut, "branch": name, "session_id": self.bench.session_id}

    def optional_operations(self):
        return (*super().optional_operations(), "fork_branch", "checkout", "discover")


def attach(bench, *, job_id, endpoint, **credentials):
    from mdb.debugger_runtime import attach_worker_debugger
    return attach_worker_debugger(job_id=job_id, session_id=bench.session_id,
        adapter=BenchDebuggerAdapter(bench), model_identity=bench.identity,
        endpoint=endpoint, source_frame_stack=bench.source_frames, **credentials)
