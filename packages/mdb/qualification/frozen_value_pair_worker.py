"""Freeze the discovered pair before observing new wording/source positions."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import torch
from saturn_pub.core import Act

from coalition_support import Panel as CoalitionPanel
from mdb.frozen_value_pair import CAPTURE_LAYERS, arms
from mdb.semantic_scaffold import resolve_span
from mdb.value_coalition import LATE_CELLS, replacement
from scaffold_io import digest_file, save


class Panel(CoalitionPanel):
    def case(self, case):
        encoded = {side: self.tokenizer(case[side]["text"], add_special_tokens=False,
            return_offsets_mapping=True) for side in ("recipient", "donor")}
        ids = {side: encoded[side]["input_ids"] for side in encoded}
        positions = {side: resolve_span(encoded[side]["offset_mapping"], case[side]["source_span"]) for side in encoded}
        if positions["recipient"] != positions["donor"] or len(positions["recipient"]) != 1:
            save(self.root / "cases" / (case["id"] + "-inapplicable.json"),
                {"case": case, "source_rows": positions, "instrument_status": "inapplicable-multitoken-or-misaligned"})
            print("SCAFFOLD_PROGRESS " + json.dumps({"case": case["id"], "instrument_status": "inapplicable"}), flush=True)
            return []
        source = positions["recipient"][0]
        if ids["recipient"][:source] != ids["donor"][:source] or ids["recipient"][source+1:] != ids["donor"][source+1:]:
            raise ValueError("contrast changed beyond declared source")
        if max(map(len, ids.values())) > 128:
            raise ValueError("prompt length exceeded declared geometry")
        wrong = source - 1
        sessions, parents, values = {}, {}, {}
        for side in ("recipient", "donor"):
            sessions[side], parents[side] = self.opened("public-" + case["id"], side, ids[side], source)
            values[side] = sessions[side].session.read("values")
        outcomes = []
        for arm in arms():
            arm = {**arm, "arm": case["id"] + "--" + arm["arm"]}
            name, kind = arm["arm"], arm["kind"]
            side = "donor" if kind in {"donor-native", "reciprocal"} else "recipient"
            session = sessions[side]
            session.restore(parents[side])
            before = session.checkpoint()["cut_id"]
            target = wrong if kind == "wrong-target" else source
            payload_row = wrong if kind == "wrong-payload" else source
            payload_side = "recipient" if kind in {"sham", "reciprocal"} else "donor"
            act = None
            origin = [{"layer": l, "kv_head": kv, "side": side} for l, kv in LATE_CELLS]
            if kind not in {"native", "donor-native"}:
                installed = replacement(session.session.read("values"), values[payload_side],
                    arm["cells"], target, payload_row)
                act = session.session.apply(Act.replace("values", installed)).to_dict()
                session.checkpoint()
                if target == source:
                    for cell in origin:
                        if (cell["layer"], cell["kv_head"]) in arm["cells"]:
                            cell.update(side=payload_side, payload_row=payload_row)
            outcome = self.run_arm(session, ids[side], arm, source, origin,
                symbol_layers=CAPTURE_LAYERS, capture_layers=CAPTURE_LAYERS, operation_limit=8500)
            outcome.update({"case_id": case["id"], "session_id": session.session_id, "before_act_cut": before,
                "act": act, "source_row": source, "target_row": target, "payload_row": payload_row,
                "payload_side": payload_side, "cell_origins": origin})
            save(self.root / "arms" / (name + ".json"), outcome)
            outcomes.append({"case": case["id"], "arm": name, "own_text": outcome["own_text"]})
            print("SCAFFOLD_PROGRESS " + json.dumps({**outcomes[-1], "source_row": source,
                "operations": self.operations}), flush=True)
        save(self.root / "cases" / (case["id"] + ".json"), {"case": case, "source_row": source,
            "wrong_row": wrong, "token_ids": ids, "native_parents": parents,
            "token_labels": {side: [self.tokenizer.decode([t]) for t in ids[side]] for side in ids},
            "outcomes": outcomes})
        return outcomes


def main():
    import parameter_source
    config = json.loads(Path("saturn_run_config.json").read_text())
    request = json.loads(Path("saturn_debug_request.json").read_text())
    job = os.environ.get("MRUN_JOB_ID", os.environ.get("MRUN_RUN_ID", "unknown"))
    root = Path(config["output_root"]) / job
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest = parameter_source._write_parameter_manifest(Path(config["model_path"]), root / "parameter-artifacts.json")
    spec, model, tokenizer, weights, _, _ = parameter_source._load_model(config, manifest["provenance"])
    save(root / "request.json", request)
    panel = Panel(model, tokenizer, spec, weights, root)
    outcomes = [outcome for case in config["cases"] for outcome in panel.case(case)]
    summary = {"schema": "mdb-frozen-value-pair-v1", "job_id": job,
        "request_sha256": request["request_sha256"], "model_identity": weights,
        "worker_wall_s": time.monotonic() - started, "native_operations": panel.operations,
        "outcomes": outcomes, "mechanics_status": "collected; independent saved audit pending",
        "instrument_status": "prospective fixed-address pair and supplied source-span resolver; no retuning",
        "terminal_status": "not-requested; exploratory"}
    save(root / "report.json", summary)
    inventory = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
        "sha256": digest_file(p)} for p in sorted(root.rglob("*")) if p.is_file()]
    save(root / "output-manifest.json", {"files": inventory, "expanded_bytes": sum(p["bytes"] for p in inventory)})
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root).as_posix())
    raw = stream.getvalue()
    if len(raw) > request["workload_geometry"]["output_archive_max_bytes"]:
        raise RuntimeError("archive exceeds declared bound; native evidence retained")
    summary["archive"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(summary, sort_keys=True).encode())).decode(), flush=True)
    chunks = [raw[i:i+(512 << 10)] for i in range(0, len(raw), 512 << 10)]
    for i, chunk in enumerate(chunks):
        print(f"SCAFFOLD_ARCHIVE_V1 {i}/{len(chunks)} " + base64.b64encode(chunk).decode(), flush=True)


if __name__ == "__main__":
    main()
