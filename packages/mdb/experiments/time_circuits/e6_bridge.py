"""Bound integer feature addresses and retain the tracer's exact intervention."""
from time import perf_counter

import torch

from mdb.research_session import NativeResearchSession, tensor_sha256


def graph_reference(graph, *, identity, prompt, replacement, completeness, influence):
    """Graph targets are records; its token-ID property is the stable tensor API."""
    return {"schema": "mdb-external-attribution-reference-v1", "identity": identity,
        "prompt": prompt, "input_tokens": graph.input_tokens.cpu().tolist(),
        "logit_target_token_ids": graph.logit_token_ids.cpu().tolist(),
        "replacement_score": float(replacement), "completeness_score": float(completeness),
        "feature_coordinates": graph.active_features[graph.selected_features].cpu().tolist(),
        "feature_influence": influence,
        "classification": "frozen instrument reference; semantic binding unassigned"}


def logits_distance(left, right):
    delta = (left.detach().float() - right.detach().float()).abs()
    last = delta.reshape(-1, delta.shape[-1])[-1]
    return {"all_logits_exact": bool(torch.equal(left, right)),
            "last_logits_exact": bool((last == 0).all()),
            "max_abs_all_logits": float(delta.max()), "max_abs_last_logits": float(last.max()),
            "mean_abs_last_logits": float(last.mean()),
            "last_top1_equal": bool(left.reshape(-1, left.shape[-1])[-1].argmax()
                                    == right.reshape(-1, right.shape[-1])[-1].argmax())}


class FeatureSessions:
    def __init__(self, *, root, identity, tokenizer, layers, features, operator="interchange", reference_repeats=1):
        self.root, self.identity, self.tokenizer = root, identity, tokenizer
        self.layers, self.features = layers, features
        self.sessions, self.records = {}, []
        self.reference_seconds = 0.
        self.operator = operator
        self.reference_repeats = reference_repeats

    def attach(self, model):
        cls = type(model)
        native = cls.feature_intervention
        self.native, self.model, self.cls = native, model, cls
        owner = self

        def bounded(self_, prompt, interventions, *, freeze_attention=True, **kwargs):
            if kwargs:
                raise ValueError("paper adapter only admits the frozen protocol's intervention arguments")
            if not isinstance(prompt, str):
                raise ValueError("paper adapter requires the declared text prompt")
            output, record = owner.execute(prompt, interventions, freeze_attention)
            started = perf_counter()
            reference = native(self_, prompt, interventions, freeze_attention=freeze_attention)
            repeated = native(self_, prompt, interventions, freeze_attention=freeze_attention) if owner.reference_repeats == 2 else None
            owner.reference_seconds += perf_counter() - started
            record["mdb_vs_native"] = logits_distance(output[0], reference[0])
            record["native_vs_native"] = logits_distance(reference[0], repeated[0]) if repeated is not None else None
            record["native_logits_exact"] = record["mdb_vs_native"]["all_logits_exact"]
            record["intervention_roundtrip_exact"] = tensor_sha256(
                torch.tensor(interventions, dtype=torch.float64).reshape(-1, 4)
            ) == record["observation"]["intervention_tensor_sha256"]
            if not record["intervention_roundtrip_exact"]:
                raise RuntimeError("feature intervention changed during serialization")
            owner.records.append(record)
            if owner.identity.get("deterministic_algorithms") and not record["native_logits_exact"]:
                raise RuntimeError("deterministic MDB/native intervention logits differ")
            return output
        cls.feature_intervention = bounded

    def execute(self, prompt, interventions, freeze_attention):
        if prompt not in self.sessions:
            seq = len(self.tokenizer(prompt)["input_ids"])
            def validate(state):
                if state["input.prompt"] != prompt:
                    raise ValueError("feature prompt context changed")
                edits = state["feature.edits"]
                if edits.dtype != torch.float64 or edits.ndim != 2 or edits.shape[1] != 4:
                    raise ValueError("feature edits require [N,4] FP64 address/value rows")
                if edits.numel():
                    coords = edits[:, :3]
                    if not torch.equal(coords, coords.round()) or bool((coords < 0).any()):
                        raise ValueError("feature coordinates must be nonnegative integers")
                    if bool((coords[:, 0] >= self.layers).any()) or bool((coords[:, 1] >= seq).any()) or bool((coords[:, 2] >= self.features).any()):
                        raise ValueError("feature address outside declared instrument geometry")
                flag = state["attention.frozen"]
                if flag.shape != () or flag.dtype != torch.bool:
                    raise ValueError("attention mode requires a scalar bool")
            def consume(state):
                rows = [(int(l), int(p), int(f), float(v)) for l, p, f, v in state["feature.edits"].tolist()]
                output = self.native(self.model, state["input.prompt"], rows,
                    freeze_attention=bool(state["attention.frozen"].item()))
                return output, {"last_logits_sha256": tensor_sha256(output[0]),
                    "intervention_tensor_sha256": tensor_sha256(state["feature.edits"]),
                    "finite": bool(torch.isfinite(output[0]).all()), "feature_writes": len(rows),
                    "freeze_attention": bool(state["attention.frozen"].item()),
                    "error_policy": "native circuit-tracer clean reconstruction error terms"}
            session = NativeResearchSession(root=self.root / f"prompt-{len(self.sessions)}",
                identity={**self.identity, "prompt": prompt, "sequence_length": seq,
                          "layers": self.layers, "feature_width": self.features},
                initial_state={"input.prompt": prompt, "feature.edits": torch.empty((0, 4), dtype=torch.float64),
                               "attention.frozen": torch.tensor(True)},
                writable=("feature.edits", "attention.frozen"), consumer=consume, validator=validate,
                resident_bytes=64 << 20)
            self.sessions[prompt] = session
        session = self.sessions[prompt]
        edits = torch.tensor(interventions, dtype=torch.float64).reshape(-1, 4)
        return session.run("call_" + str(len(session.receipts)), updates={"feature.edits": edits,
            "attention.frozen": torch.tensor(bool(freeze_attention))},
            metadata={"operator": "original-circuit-tracer-feature-" + self.operator})

    def finish(self):
        summaries = []
        for prompt, session in self.sessions.items():
            records = [record for record in self.records if record["session_id"] == session.session_id]
            replays = []
            for label, original in (("first", records[0]), ("last", records[-1])):
                parent = original["cut"]["handle"]
                name = next(name for name, cut in session.cuts.items() if cut.handle == parent)
                _, replay = session.run("replay_" + label, parent=name)
                exact = replay["observation"]["last_logits_sha256"] == original["observation"]["last_logits_sha256"]
                replays.append({"label": label, "original_cut": parent, "exact": exact, "record": replay})
            summaries.append({"session_id": session.session_id, "prompt": prompt, "root": str(session.root),
                "replay_first_exact": replays[0]["exact"], "replays": replays,
                "memory": session.inspect_memory()})
            session.close()
            # Preserve repeatability diagnostics even when the instrument's
            # floating execution is not bit reproducible; never erase its arms.
        return summaries

    def detach(self):
        self.cls.feature_intervention = self.native
        for session in self.sessions.values():
            session.close()
