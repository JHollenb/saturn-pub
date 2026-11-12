"""Compile existing experiment metadata to a root/token/Act regeneration recipe."""
import argparse
import json
from pathlib import Path
from mdb.native_replay_recipe import SCHEMA, seal, validate


def read(p):
    return json.loads(p.read_text())


def prepare(raw, output):
    output.mkdir(exist_ok=False, parents=True)
    roots, events, observations, summaries = [], [], {}, []
    identity = execution = None
    def emit(session, name, kind, expected=None, **kwargs):
        event = {"id": session+"/"+name, "session": session, "kind": kind, **kwargs}
        if expected: event["expected_cut"] = expected
        events.append(event)
        return event["id"]
    for case_id in ("anchor", "intervening-note"):
        case = read(raw/"cases"/(case_id+".json"))
        parent_events = {}
        for side in ("recipient", "donor"):
            sid = case_id+"-"+side
            ids = case["token_ids"][side]
            parent = case["native_parents"][side]
            descriptor = read(raw/"project/store/cuts"/(parent+".json"))
            identity, execution = descriptor["manifest"]["model_identity"], descriptor["manifest"]["execution"]
            ancestry = []
            cursor = parent
            while cursor:
                ancestry.append(cursor)
                cursor = read(raw/"project/store/cuts"/(cursor+".json"))["manifest"]["parent"]
            ancestry.reverse()
            roots.append({"id": sid, "token_ids": ids[:-3]})
            emit(sid, "initial", "mark", ancestry[0])
            position = 1
            emit(sid, "initial-prefix", "continue", ancestry[position], steps=33)
            position += 1
            for i, token in enumerate(ids[-3:]):
                emit(sid, f"ingress-{i}", "prompt-token", ancestry[position], token_id=token)
                position += 1
                if i < 2:
                    emit(sid, f"consume-{i}", "continue", ancestry[position], steps=33)
                    position += 1
            if position != len(ancestry): raise ValueError("historical ingress is not supported length-only program")
            parent_events[side] = emit(sid, "parent", "mark", parent)
            resolution = case["resolutions"][side]
            event = emit(sid, "discovery", "continue", resolution["discovery_cut"], steps=22)
            observations[event] = {"traces": [resolution["trace"]]}
            emit(sid, "restore-after-discovery", "restore", parent, source=parent_events[side])
        for name in ("native-recipient", "native-donor", "auto-sham", "auto-pair"):
            record = read(raw/"arms"/(case_id+"--"+name+".json"))
            side = "donor" if name == "native-donor" else "recipient"
            sid = case_id+"-"+side
            emit(sid, name+"-restore", "restore", record["before_act_cut"], source=parent_events[side])
            if record["act"]:
                emit(sid, name+"-copy", "copy-values", record["trajectory"][0]["parent_cut"],
                    source=parent_events[record["payload_side"]], cells=record["arm"]["cells"],
                    target_row=record["target_row"], source_row=record["payload_row"])
            for row in record["trajectory"]:
                prefix = name+f"-token{row['step']}"
                event = emit(sid, prefix+"-readout", "continue", row["readout_cut"], steps=31)
                observations[event] = {"logits": row["logits"], "traces": row["traces"],
                    "next_token_id": row["next_token_id"]}
                event = emit(sid, prefix+"-commit", "continue", row["result_cut"], steps=2)
                observations[event] = {"committed_token_id": row["next_token_id"], "input_ids": row["input_ids"]}
            summaries.append({"case": case_id, "arm": name, "own_text": record["own_text"]})
    recipe = seal({"schema": SCHEMA, "model_identity": identity, "execution": execution, "roots": roots, "events": events})
    validate(recipe)
    checks = {"observations": observations, "historical_futures": summaries,
        "scope": "Comparison hashes/labels only; never installed as model state", "source_job": "job-d70d92ca2bce"}
    for name, value in (("recipe.json", recipe), ("expectations.json", checks)):
        (output/name).write_text(json.dumps(value, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"recipe_bytes": (output/"recipe.json").stat().st_size, "comparison_metadata_bytes": (output/"expectations.json").stat().st_size,
        "roots": len(roots), "events": len(events), "native_steps": sum(e.get("steps",0) for e in events),
        "expected_native_cuts": sum("expected_cut" in e for e in events), "activation_payloads": 0}))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    a=p.parse_args()
    prepare(a.raw.resolve(), a.output.resolve())
