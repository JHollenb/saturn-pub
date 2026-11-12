"""Finite native source-writer mediation with registered MDB symbol lifecycles."""
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
from saturn_pub import StateAddress
from saturn_pub.core import Act

from mdb.live_symbols import LiveSymbols
from mdb.project import Project
from mdb.prompt_qwen import PromptQwenAdapter
from mdb.saturn_pub_adapter import SaturnPubDebuggerAdapter
from mdb.semantic_scaffold import evaluate, resolve_span
from scaffold_io import Assay, digest_file, save


class Panel:
    def __init__(self, model, tokenizer, root):
        self.native = PromptQwenAdapter(model)
        self.tokenizer, self.root = tokenizer, root
        self.project = Project(root / "project")
        self.storage = Assay(model, tokenizer, root)
        self.steps = 0

    def open(self, name, ids):
        adapter = SaturnPubDebuggerAdapter(self.native.session(ids), project=self.project,
            session_id=name)
        return adapter, LiveSymbols(adapter)

    def advance(self, recorder, steps, **kwargs):
        self.steps += steps
        if self.steps > 7000:
            raise RuntimeError("declared native-operation budget exceeded")
        return recorder.advance(steps, **kwargs)

    def case(self, case, *, timing_only=False, support_only=False):
        encoded = {side: self.tokenizer(case[side]["text"], add_special_tokens=False,
            return_offsets_mapping=True) for side in ("recipient", "donor")}
        source = {side: resolve_span(encoded[side]["offset_mapping"], case[side]["source_span"])
            for side in encoded}
        ids = {side: encoded[side]["input_ids"] for side in encoded}
        row = source["recipient"][0]
        if source["recipient"] != [row] or source["donor"] != [row] or len(ids["recipient"]) != len(ids["donor"]):
            raise ValueError("this source-writer instrument requires matched one-token arguments")
        if ids["recipient"][:row] != ids["donor"][:row] or ids["recipient"][row + 1:] != ids["donor"][row + 1:]:
            raise ValueError("the declared contrast changed more than its argument")
        ordinary = {side: self.native.native_logits(ids[side]) for side in ids}
        ordinary_records = {side: self.storage.tensor(case["id"] + "-ordinary-" + side, value)
            for side, value in ordinary.items()}
        prep = {}
        for side in encoded:
            adapter, recorder = self.open(case["id"] + "-source-" + side, ids[side][:row + 1])
            self.advance(recorder, 20)  # embedding and layers 0..18
            incoming = recorder.observe("argument.writer-input", (StateAddress("hidden"),),
                context=case["id"], claim="Observed source-position residual before decoder layer 19; semantic label supplied.")
            hidden = adapter.session.read("hidden")
            cut = adapter.checkpoint()["cut_id"]
            receipt = self.advance(recorder, 9, symbols=(incoming,))
            values, keys = adapter.session.read("values"), adapter.session.read("keys")
            formed = recorder.observe("argument.formed-late-values",
                tuple(StateAddress("values", (layer, 0, head, row)) for layer in range(19, 28) for head in range(2)),
                context=case["id"], claim="Native source-position V rows formed by layers 19..27; no complete semantic algorithm claimed.",
                evidence=("cas://sha256/" + receipt.fingerprint,),
                metadata={"writer_input_definition": incoming.definitions[0].occurrence_id,
                    "writer_receipt": receipt.fingerprint, "source_token_position": row})
            prep[side] = {"adapter": adapter, "recorder": recorder, "cut": cut,
                "hidden": hidden, "values": values, "keys": keys,
                "formed_binding": formed.binding_fingerprint}
        outcomes = []
        arms = (("native-recipient", "native-donor", "donor-late-values", "delayed-donor-late-values")
            if timing_only else ("native-recipient", "native-donor", "sham-writer-input", "donor-writer-input",
                "donor-writer-input-restore-late-values", "donor-late-values"))
        if support_only:
            arms = ("native-recipient", "native-donor", "delayed-donor-late-values", "delayed-donor-all-values")
        for arm in arms:
            side = "donor" if arm == "native-donor" else "recipient"
            adapter, recorder = self.open(case["id"] + "-" + arm, ids[side][:row + 1])
            self.advance(recorder, 20)
            acts = []
            if arm in {"sham-writer-input", "donor-writer-input", "donor-writer-input-restore-late-values"}:
                value = prep["recipient" if arm == "sham-writer-input" else "donor"]["hidden"]
                acts.append(adapter.session.apply(Act.replace("hidden", value)).to_dict())
            writer_input = recorder.observe("argument.writer-input", (StateAddress("hidden"),),
                context=case["id"], claim="Observed source writer input after the declared arm, before native layers 19..27.")
            writer_receipt = self.advance(recorder, 9, symbols=(writer_input,))
            formed_values = adapter.session.read("values")
            payloads = []
            if arm in {"donor-writer-input-restore-late-values", "donor-late-values"}:
                replacement = prep["recipient" if "restore" in arm else "donor"]["values"]
                current = adapter.session.read("values")
                for layer in range(19, 28):
                    before = current[layer].clone()
                    current[layer][:, :, row] = replacement[layer][:, :, row]
                    payloads.append({"layer": layer,
                        "before": self.storage.tensor(case["id"] + arm + f"-L{layer}-before", before),
                        "after": self.storage.tensor(case["id"] + arm + f"-L{layer}-after", current[layer])})
                acts.append(adapter.session.apply(Act.replace("values", current)).to_dict())
            source_binding = recorder.observe("argument.source-values",
                tuple(StateAddress("values", (layer, 0, head, row)) for layer in range(19, 28) for head in range(2)),
                context=case["id"], claim="Observed late source V after declared writer/interchange arm.",
                evidence=("cas://sha256/" + writer_receipt.fingerprint,),
                metadata={"writer_input_definition": writer_input.definitions[0].occurrence_id,
                    "source_token_position": row, "arm": arm})
            formation = {"recipient_distance": float(torch.cat([(formed_values[l][:, :, row] - prep["recipient"]["values"][l][:, :, row]).float().flatten() for l in range(19, 28)]).norm()),
                "donor_distance": float(torch.cat([(formed_values[l][:, :, row] - prep["donor"]["values"][l][:, :, row]).float().flatten() for l in range(19, 28)]).norm())}
            self.advance(recorder, 4, symbols=(source_binding,))  # norm, readout, sample, commit
            staged = []
            # Caller-supplied remainder of the prompt. Model futures begin only after it.
            for index, token in enumerate(ids[side][row + 1:], start=row + 1):
                receipt = self.native.stage_prompt_token(adapter.session, token)
                adapter.checkpoint()
                staged.append({"prompt_position": index, "token_id": token, "receipt": receipt.fingerprint})
                if index < len(ids[side]) - 1:
                    self.advance(recorder, 33)
            if arm in {"delayed-donor-late-values", "delayed-donor-all-values"}:
                current = adapter.session.read("values")
                for layer in range(0 if arm == "delayed-donor-all-values" else 19, 28):
                    before = current[layer].clone()
                    current[layer][:, :, row] = prep["donor"]["values"][layer][:, :, row]
                    payloads.append({"layer": layer,
                        "before": self.storage.tensor(case["id"] + arm + f"-L{layer}-before", before),
                        "after": self.storage.tensor(case["id"] + arm + f"-L{layer}-after", current[layer])})
                acts.append(adapter.session.apply(Act.replace("values", current)).to_dict())
                source_binding = recorder.observe("argument.delayed-source-values",
                    tuple(StateAddress("values", (layer, 0, head, row)) for layer in range(19, 28) for head in range(2)),
                    context=case["id"], claim="Same donor late V installed after intervening prompt computation.",
                    metadata={"source_token_position": row, "arm": arm})
            future, trajectories = [], []
            first_logits = None
            for step in range(3):
                input_tokens = adapter.session.read("tokens").tolist()[0]
                binding = recorder.observe("argument.live-source-values",
                    tuple(StateAddress("values", (layer, 0, head, row)) for layer in range(19, 28) for head in range(2)),
                    context=case["id"], claim="Context-qualified original argument V at this native receiver; name supplied.",
                    metadata={"source_definition": source_binding.definitions[0].occurrence_id,
                        "generated_offset": step, "arm": arm})
                readout_receipt = self.advance(recorder, 31, symbols=(binding,))
                logits = adapter.session.read("logits")
                if step == 0:
                    first_logits = logits
                output_binding = recorder.observe("output.next-token-logits", (StateAddress("logits"),),
                    context=case["id"], claim="Actual native next-token readout before greedy sampling.",
                    metadata={"input_definition": binding.definitions[0].occurrence_id,
                        "readout_receipt": readout_receipt.fingerprint})
                token = int(logits.argmax())
                self.advance(recorder, 2, symbols=(output_binding,))
                if adapter.session.read("tokens").tolist()[0] != input_tokens + [token]:
                    raise ValueError("native readout/commit/own-feedback closure mismatch")
                future.append(token)
                trajectories.append({"input_ids": input_tokens, "next_token_id": token,
                    "logits": self.storage.tensor(case["id"] + arm + f"-step{step}-logits", logits),
                    "readout_receipt": readout_receipt.fingerprint,
                    "source_binding": binding.binding_fingerprint,
                    "output_binding": output_binding.binding_fingerprint})
            text = self.tokenizer.decode(future)
            def target_id(word, side):
                aliases = {self.tokenizer.encode(prefix + word, add_special_tokens=False)[0]
                    for prefix in ("", " ")}
                return max(aliases, key=lambda token: float(ordinary[side][0, token]))
            a = target_id(case["expected"], "recipient")
            b = target_id(case["donor_expected"], "donor")
            outcome = {"arm": arm, "own_text": text, "own_ids": future, "trajectory": trajectories,
                "recipient_score": evaluate(case, text), "donor_score": evaluate(case, text, donor=True),
                "formation": formation, "acts": acts, "value_payloads": payloads,
                "staged_prompt": staged, "session_id": adapter.session_id,
                "installed_source_binding": source_binding.binding_fingerprint,
                "donor_minus_recipient_margin": float(first_logits[0, b].float() - first_logits[0, a].float()),
                "ordinary_argmax_matches": int(ordinary[side].argmax()) == future[0],
                "ordinary_logit_max_abs": float((first_logits.float() - ordinary[side].float()).abs().max())}
            outcomes.append(outcome)
            print("SCAFFOLD_PROGRESS " + json.dumps({"case": case["id"], "arm": arm,
                "output": text, "operations": self.steps}), flush=True)
        result = {"case": case, "source_row": row, "outcomes": outcomes,
            "ordinary_logits": ordinary_records, "observable_tokens": {"recipient": a, "donor": b},
            "source_formation": {side: {"source_cut": p["cut"], "formed_binding": p["formed_binding"]}
                for side, p in prep.items()}}
        save(self.root / (case["id"] + "-result.json"), result)
        return result


def main():
    import parameter_source
    config = json.loads(Path("saturn_run_config.json").read_text())
    request = json.loads(Path("saturn_debug_request.json").read_text())
    job = os.environ.get("MRUN_JOB_ID", os.environ.get("MRUN_RUN_ID", "unknown"))
    root = Path(config["output_root"]) / job
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    manifest = parameter_source._write_parameter_manifest(Path(config["model_path"]), root / "parameter-artifacts.json")
    _, model, tokenizer, weights_sha, files, content_sha = parameter_source._load_model(config, manifest["provenance"])
    save(root / "request.json", request)
    panel = Panel(model, tokenizer, root)
    results = [panel.case(case, timing_only=config.get("timing_only", False),
        support_only=config.get("support_only", False)) for case in config["cases"]]
    symbols = sum(len(panel.project.symbols(s["session_id"])) for s in panel.project.sessions())
    summary = {"schema": "mdb-live-symbol-closure-assay-v1", "job_id": job,
        "request_sha256": request["request_sha256"], "model_identity": weights_sha,
        "weight_files": files, "weight_content_sha256": content_sha,
        "symbol_bindings": symbols, "native_operations": panel.steps,
        "worker_wall_s": time.monotonic() - started,
        "cases": [{"id": r["case"]["id"], "outcomes": [{k: a[k] for k in
            ("arm", "own_text", "formation", "recipient_score", "donor_score", "donor_minus_recipient_margin", "ordinary_argmax_matches", "ordinary_logit_max_abs")} for a in r["outcomes"]]} for r in results],
        "mechanics_status": "valid", "instrument_status": "native-source-formation-and-receipt-joined-symbols",
        "trend_status": "per-arm signed effects retained", "terminal_status": "not-requested; exploratory"}
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
        raise RuntimeError("archive exceeds declared bound; remote raw evidence retained")
    summary["archive"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(summary, sort_keys=True).encode())).decode(), flush=True)
    chunks = [raw[i:i + (512 << 10)] for i in range(0, len(raw), 512 << 10)]
    for i, chunk in enumerate(chunks):
        print(f"SCAFFOLD_ARCHIVE_V1 {i}/{len(chunks)} " + base64.b64encode(chunk).decode(), flush=True)


if __name__ == "__main__":
    main()
