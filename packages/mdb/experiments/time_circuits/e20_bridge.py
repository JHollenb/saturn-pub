"""Run the frozen E20 arm algebra through MDB's native research adapter."""
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mdb.research_session import NativeResearchSession, tensor_sha256


@dataclass
class Candidate:
    parent: str
    updates: dict = field(default_factory=dict)
    writes: list = field(default_factory=list)


def execute_context(paper, cr, *, root, identity, **settings):
    """Original native reference and MDB use the same model and arm definitions.

    Prefix formation retains the original native forward and residual hook. MDB
    takes custody at the completed subject prefix, then admits all formed-KV
    writes and runs the unchanged native suffix on a private hydrated closure.
    """
    original = {name: getattr(paper, name) for name in
                ("_prefix_cache", "_clone_cache", "_overwrite_rows", "_suffix_logits")}
    native = paper.run_context(cr, **settings)
    session, source_names, arm_records = None, {}, []
    fields = {}
    prefix_counts = {None: "parent", "clean": "seeded_prefix", "recipient": "residual_noop_prefix"}
    suffix = cr.filler_input.repeat(len(cr.subjects), 1)[:, cr.subj_pos + 1:]
    positions = torch.arange(cr.subj_pos + 1, cr.seq_len, device=cr.device)
    position_ids = positions.unsqueeze(0).expand(len(cr.subjects), -1)
    allowed = sorted(set(settings["mediator_layers"] + settings["outside_layers"]))

    def cache_state(cache):
        state = {"input.suffix": suffix, "input.cache_positions": positions,
                 "input.positions": position_ids}
        for layer, value in enumerate(cache.layers):
            state[f"kv.{layer}.keys"] = value.keys
            state[f"kv.{layer}.values"] = value.values
        return state

    def validate(state):
        for address, (shape, dtype, device) in fields.items():
            tensor = state[address]
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
                raise ValueError("native E20 geometry/dtype/device mismatch: " + address)

    def consume(state):
        cache = cr.PatchCache()
        for layer in range(cr.n_layers):
            cache.update(state[f"kv.{layer}.keys"], state[f"kv.{layer}.values"], layer)
        logits = cr.model(input_ids=state["input.suffix"], use_cache=True, past_key_values=cache,
            cache_position=state["input.cache_positions"], position_ids=state["input.positions"]).logits[:, -1, :].detach()
        return logits, {"last_logits_sha256": tensor_sha256(logits),
                        "finite": bool(torch.isfinite(logits).all()),
                        "shape": list(logits.shape), "subject_position": cr.subj_pos}

    def prefix(cr_, seed_layer, source):
        nonlocal session
        cache = original["_prefix_cache"](cr_, seed_layer, source)
        state = cache_state(cache)
        name = prefix_counts[source]
        if session is None:
            fields.update({a: (v.shape, v.dtype, v.device) for a, v in state.items()})
            session = NativeResearchSession(root=root, identity=identity, initial_state=state,
                writable=[f"kv.{layer}.{kind}" for layer in allowed for kind in ("keys", "values")],
                consumer=consume, validator=validate, device=cr.device, resident_bytes=64 << 20)
        else:
            session.capture(name, state, metadata={"formation": "original-native-prefix",
                "residual_source": source, "seed_layer": seed_layer, "subject_position": cr.subj_pos})
        source_names[id(cache)] = name
        return cache

    def clone(cache):
        return Candidate(source_names[id(cache)])

    def overwrite(candidate, rows, layers, subject_position, roll=0):
        encoded = session.backend._states[session.cuts[candidate.parent].handle]
        for layer in layers:
            for kind, source in zip(("keys", "values"), rows[layer]):
                address = f"kv.{layer}.{kind}"
                tensor = candidate.updates.get(address)
                if tensor is None:
                    tensor = session.pages.decode(encoded[address], private=True)
                donor = torch.roll(source, shifts=roll, dims=0) if roll else source
                if donor.shape != tensor[:, :, subject_position, :].shape:
                    raise ValueError("E20 mediator row geometry mismatch")
                tensor[:, :, subject_position, :] = donor.to(tensor.dtype)
                candidate.updates[address] = tensor
        candidate.writes.append({"layers": list(layers), "position": subject_position, "roll": roll})

    def suffix_logits(cr_, candidate):
        name = "arm_" + str(len(arm_records))
        logits, record = session.run(name, parent=candidate.parent, updates=candidate.updates,
            metadata={"formed_kv_writes": candidate.writes, "consumer": "original-native-suffix"})
        arm_records.append(record)
        return logits

    try:
        paper._prefix_cache, paper._clone_cache = prefix, clone
        paper._overwrite_rows, paper._suffix_logits = overwrite, suffix_logits
        result = paper.run_context(cr, **settings)
        # Exact equality is a within-lane runtime check, never a semantic gate.
        row_parity = result["rows"] == native["rows"]
        baseline, baseline_record = session.run("baseline_replay")
        replay = tensor_sha256(baseline) == arm_records[0]["observation"]["last_logits_sha256"]
        memory = session.inspect_memory()
        record = {"session_id": session.session_id, "root": str(Path(root)),
                  "context_index": settings["context_index"], "native_rows_exact": row_parity,
                  "baseline_replay_exact": replay, "arms": arm_records,
                  "baseline_replay": baseline_record, "memory": memory,
                  "boundary": "completed subject prefix -> formed-KV edits -> unchanged native suffix"}
        if not row_parity or not replay:
            raise RuntimeError("MDB E20 failed original-native within-lane parity")
        return result, record
    finally:
        for name, value in original.items():
            setattr(paper, name, value)
        if session:
            session.close()
