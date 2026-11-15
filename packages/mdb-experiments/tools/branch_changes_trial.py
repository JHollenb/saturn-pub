"""Finite native Qwen branch-change trial, executed in one mrun lease.

Run with --tiny only for the local mechanics check. Real checkpoints use mrun.
The worker publishes each completed arm before attempting the next one.
"""
import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

import torch
from mrun.outbox import Outbox
from retrieve_weights import manifest, retrieve

from mdb.analysis.changes import explain_changes
from mdb.analysis.forecast import Margin
from mdb.analysis.propagate import Edit, whatif
from mdb.engine import Run
from mdb.ops.debugger import attach, open_tiny


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    # Explicit, recorded arithmetic policy; no flag is silently assumed by a bound.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    out = Outbox.from_env()
    for source_path in (Path(__file__), Path(__file__).with_name("retrieve_weights.py")):
        out.put_file(source_path, name=f"sources/{source_path.name}")
    if Path("sources.json").exists():
        out.put_file("sources.json")

    def save(name, value):
        out.put_json(name, value)
        print(name, json.dumps(value if name == "summary.json" else {"saved": True}), flush=True)

    if args.tiny:
        dbg = open_tiny("qwen2", "bf16")
        tokens, donor_tokens, site, source = [1, 5, 9, 3], [1, 7, 9, 3], 0, 1
        answer = (4, 9)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        root = Path(args.model)
        spec = manifest(root)
        save("parameter-artifacts.json", spec)
        # Prove exact original tensor-byte retrieval for one named parameter.
        name = next(iter(spec["tensors"]))
        retrieve(spec, name, root, "retrieval-check.raw")
        Path("retrieval-check.raw").unlink()
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16,
            attn_implementation="eager", local_files_only=True).eval().to("cuda")
        dbg = attach(model, tokenizer)
        prefix = "Description: icy\nTemperature: cold\nDescription: fiery\nTemperature: hot\nDescription: "
        tokens = tokenizer.encode(prefix + "freezing\nTemperature:")
        donor_tokens = tokenizer.encode(prefix + "boiling\nTemperature:")
        site, source = 20, 18
        answer = tuple(tokenizer.encode(t, add_special_tokens=False)[0] for t in (" hot", " cold"))
        save("inputs.json", {"tokens": tokens, "donor_tokens": donor_tokens,
            "decoded_source": tokenizer.decode([tokens[source]]),
            "answer_ids": answer, "site": site, "source": source})
        if len(tokens) != len(donor_tokens) or tokenizer.decode([tokens[source]]) != " freezing":
            raise RuntimeError("historical source alignment changed")

    row = len(tokens) - 1
    consumer = Margin(row, *answer)
    save("environment.json", {"job_id": os.environ.get("MRUN_JOB_ID"),
        "torch": torch.__version__, "python": platform.python_version(),
        "device": str(next(dbg.adapter.model.parameters()).device),
        "dtype": str(next(dbg.adapter.model.parameters()).dtype),
        "attention_implementation": "eager", "torch_num_threads": torch.get_num_threads(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "model_identity": dbg.adapter.model_identity})
    donor = dbg.run(donor_tokens, name="donor")
    donor.until(f"L{site}.h")
    donor_cells = {p: donor.read_cache(site, p)[:, :, source:source + 1, :].clone()
                   for p in ("k", "v")}
    initial = dbg.run(tokens, name="parent")
    initial.until(f"L{site}.h")
    parent = initial.session.capture(retain=False)
    own_cells = {p: initial.read_cache(site, p)[:, :, source:source + 1, :].clone()
                 for p in ("k", "v")}
    out.put_tensor("parent-state.pt", parent.payload)
    out.put_tensor("donor-cells.pt", donor_cells)

    def fork(ports=(), sham=False):
        run = Run(dbg.adapter, tokens, root=parent)
        for port in ports:
            run.set_cell(site, port, (own_cells if sham else donor_cells)[port], rows=[source])
        return run

    base = fork(); base.until("logits")
    base_logits = base.read("logits").detach().cpu()
    out.put_tensor("baseline-logits.pt", base_logits)
    base_margin = float(base_logits.double().reshape(-1)[answer[0]] -
                        base_logits.double().reshape(-1)[answer[1]])
    summaries, reports = {}, {}
    v_run = None
    for name, ports, sham in (("sham", ("k", "v"), True), ("v", ("v",), False),
                              ("k", ("k",), False), ("kv", ("k", "v"), False)):
        branch = fork(ports, sham); branch.until("logits")
        logits = branch.read("logits").detach().cpu()
        out.put_tensor(f"{name}/logits.pt", logits)
        margin = float(logits.double().reshape(-1)[answer[0]] - logits.double().reshape(-1)[answer[1]])
        summaries[name] = {"margin": margin, "delta": margin - base_margin,
                           "logits_equal_baseline": torch.equal(logits, base_logits)}
        reports[name] = {}
        for layer in (site, site + 1):
            report = explain_changes(base, branch, consumer, layer=layer).report
            report["common_parent"] = parent.fingerprint
            save(f"{name}/L{layer}.json", report)
            reports[name][layer] = report
            endpoints = {arm: {p: run.read(f"L{layer}.{p}").detach().cpu()
                              for p in ("h", "a", "p", "A", "o", "r", "n", "g", "u", "f", "m", "h_out")}
                         for arm, run in (("base", base), ("edited", branch))}
            endpoints["values"] = {arm: run.read_cache(layer, "v").detach().cpu()
                                    for arm, run in (("base", base), ("edited", branch))}
            out.put_tensor(f"{name}/L{layer}-endpoints.pt", endpoints)
        save(f"{name}/outcome.json", summaries[name])
        if name == "v":
            v_run = branch
        else:
            del branch
        gc.collect()

    # Nominate a neuron from the completed development explanation. Freeze a
    # mathematical suffix prediction before the unobserved repair arm executes.
    neuron = reports["v"][site]["mlp"]["neurons"]["rows"][0]["index"]
    mask = torch.zeros_like(base.read(f"L{site}.f"), dtype=torch.bool)
    mask[..., neuron] = True
    edit = Edit(f"t{row}.L{site}.f", base.read(f"L{site}.f"), "replace",
                mask=mask, selection=f"i={neuron}")
    try:
        def persist_prediction(record):
            save("selected-neuron-forecast.json", record)
            out.flush()

        prediction = whatif(v_run, edit, consumer, before_native=persist_prediction)
        save("selected-neuron-whatif.json", prediction.report)
    except ValueError as error:
        save("selected-neuron-whatif-refusal.json", {"reason": str(error)})
    # A direct native branch remains informative even if forecast admission refuses.
    intervention = {"selected_neuron": neuron, "selection": "largest direct signed MLP change",
                    "receiving_context": "development freezing context", "fresh_context": False,
                    "prediction_scope": "native arm below is not a fresh-context generalization test"}
    save("selected-neuron-plan.json", intervention)
    repair = fork(("v",))
    repair.set(f"L{site}.f", base.read(f"L{site}.f"), "replace", mask=mask,
               selection=f"i={neuron}")
    repair.until("logits")
    repaired_logits = repair.read("logits").double().reshape(-1)
    repair_margin = float(repaired_logits[answer[0]] - repaired_logits[answer[1]])
    summaries["selected_neuron_restore"] = {**intervention, "margin": repair_margin,
        "delta_from_v": repair_margin - summaries["v"]["margin"]}
    out.put_tensor("selected-neuron-logits.pt", repair.read("logits").detach().cpu())
    if not args.tiny:
        # Reorder the demonstrations while keeping the same source and shape.
        # Reuse the development-selected neuron without looking at a fresh repair.
        fresh_prefix = "Description: fiery\nTemperature: hot\nDescription: icy\nTemperature: cold\nDescription: "
        fresh_tokens = dbg.tokenizer.encode(fresh_prefix + "freezing\nTemperature:")
        fresh_donor_tokens = dbg.tokenizer.encode(fresh_prefix + "boiling\nTemperature:")
        if len(fresh_tokens) != len(tokens) or len(fresh_donor_tokens) != len(tokens):
            raise RuntimeError("fresh-context token alignment changed")
        fresh_donor = dbg.run(fresh_donor_tokens, name="fresh_donor")
        fresh_donor.until(f"L{site}.h")
        fresh_v = fresh_donor.read_cache(site, "v")[:, :, source:source + 1, :].clone()
        fresh_start = dbg.run(fresh_tokens, name="fresh_parent")
        fresh_start.until(f"L{site}.h")
        fresh_parent = fresh_start.session.capture(retain=False)
        fresh_base = Run(dbg.adapter, fresh_tokens, root=fresh_parent)
        fresh_base.until("logits")
        fresh_branch = Run(dbg.adapter, fresh_tokens, root=fresh_parent)
        fresh_branch.set_cell(site, "v", fresh_v, rows=[source]); fresh_branch.until("logits")
        fresh_edit = Edit(f"t{row}.L{site}.f", fresh_base.read(f"L{site}.f"), "replace",
                          mask=mask, selection=f"i={neuron}")

        def persist_fresh(record):
            save("fresh/forecast.json", {"selected_neuron_from_development": neuron,
                "parent": fresh_parent.fingerprint, "record": record,
                "scope": "conditional repair forecast after observing fresh V branch"})
            out.flush()

        fresh_prediction = whatif(fresh_branch, fresh_edit, consumer, before_native=persist_fresh)
        save("fresh/whatif.json", fresh_prediction.report)
        fresh_repair = Run(dbg.adapter, fresh_tokens, root=fresh_parent)
        fresh_repair.set_cell(site, "v", fresh_v, rows=[source])
        fresh_repair.set(f"L{site}.f", fresh_base.read(f"L{site}.f"), "replace", mask=mask,
                         selection=f"i={neuron}")
        fresh_repair.until("logits")
        fresh_logits = fresh_repair.read("logits").double().reshape(-1)
        fresh_margin = float(fresh_logits[answer[0]] - fresh_logits[answer[1]])
        out.put_tensor("fresh/repair-logits.pt", fresh_repair.read("logits").detach().cpu())
        summaries["fresh_selected_neuron_restore"] = {"neuron": neuron,
            "margin": fresh_margin,
            "independent_native_matches_whatif": fresh_margin == fresh_prediction.report["measured"]["edited"]["value"],
            "scope": "fresh demonstration order; observed prefix and V branch; repair forecast precedes native repair"}
    save("summary.json", {"schema": "mdb-branch-changes-trial-v1", "baseline_margin": base_margin,
        "arms": summaries, "elapsed_s": time.perf_counter() - started,
        "mechanics_status": "completed", "instrument_status": "native endpoint accounting",
        "trend_status": "inspect signed arm and component outcomes",
        "terminal_status": "no semantic circuit or cross-context generalization claim",
        "parent": parent.fingerprint})
    out.close()


if __name__ == "__main__":
    main()
