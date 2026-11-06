"""Saved-only audit of complete joint-state forecasts and predicted-input forks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

import numpy as np
from mdb.forecast_transaction import load_forecast_tree, verify_transaction
from saturn_pub.store import LocalStore


def audit(collection, output):
    here = Path(__file__).resolve().parent
    H = runpy.run_path(str(here / "analyze_flux_attention_state_law.py"))
    read, sha, tree, B, metric, error, bits = [H[n] for n in ("read", "sha", "tree", "B", "metric", "error", "bits")]
    custody = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](collection)
    data = collection / "extracted/binding-minimal-result"
    config, summary, rows, forks, receipts = [read(data / n) for n in ("plan.json", "summary.json", "rows.json", "forks.json", "receipts.json")]
    request = read(collection / "request.json")
    assert config == request["config"] and summary == read(collection / "report.json")["report"]
    assert summary["request_sha256"] == custody["request_sha256"] == request["request_sha256"]
    assert summary["counts"] == config["expected_counts"]
    assert read(collection / "collection-limits.json")["overrides"] == {}
    for n, fingerprint in request["source_hashes"].items():
        assert sha(collection / "payload" / n) == fingerprint
    tx = data / "transaction"
    verified = verify_transaction(tx, expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert verified["verified"] and verified["phase"] == "closed"
    predictions = {i:tree(load_forecast_tree(tx, read(tx / f"forecasts/joint{i}.json")["predictions"])) for i in config["blocks"]}
    census = tree(load_forecast_tree(tx, read(tx / "forecasts/census.json")["predictions"]))
    expected_at_seal = {k:v if k in {"component_loads", "numpy_block_forecasts"} else 0 for k,v in config["expected_counts"].items()}
    assert census["counts_at_seal"] == read(data / "seal-census.json")["counts"] == expected_at_seal
    assert census["hypotheses"] == config["hypotheses"]
    observed = tree(load_forecast_tree(tx, read(tx / "observations/native.json")["value"]))
    assert observed == dict(rows=rows, forks=forks, counts=summary["counts"])
    recipes = {r["source"]["tensor"]:r["source"] for r in read(data / "parameter-artifacts.json")["artifacts"]}
    native_source = read(data / "native-source.json")
    joined = 0
    for index, ports in native_source["parameters"].items():
        assert len(ports) == 16
        for n, descriptor in ports.items():
            src = recipes[f"transformer_blocks.{index}."+n]
            assert descriptor["sha256"] == src["payload_sha256"] and descriptor["shape"] == src["shape"]
            assert descriptor["dtype"] == "torch.bfloat16"
            joined += 1
    assert joined == len(recipes) == 32
    public_receipt = runpy.run_path(str(here / "summarize_context_window.py"))["public_receipt"]
    expected_receipts = [r["receipt"] for r in rows]+[a["receipt"] for f in forks for a in f["acts"]]
    assert len(receipts) == len(expected_receipts) == 15
    assert len({r["fingerprint"] for r in receipts}) == 15
    assert {r["fingerprint"] for r in receipts} == {r["fingerprint"] for r in expected_receipts}
    assert len(forks) == 3 and {f["case"] for f in forks} == set(config["cases"])
    for receipt in receipts:
        public_receipt(receipt)
        assert receipt == read(data / "store/receipts" / (receipt["fingerprint"]+".json"))
    store = LocalStore(data / "store")
    arrays = {n:np.load(collection / "payload" / (n+".npy"), allow_pickle=False) for n in config["input_files"]}
    roots = {name:B((1-dose)*arrays["text_base"].astype(np.float64)+dose*arrays["text_changed"])
             for name,dose in config["doses"].items()}
    def fields(fp):
        manifest = read(data / "store/cuts" / (fp+".json"))["manifest"]
        return {i["key"]["value"]:i["value"] for i in manifest["payload"]["items"]}
    def same_context(a, b):
        return all(a[n] == b[n] for n in ("img_mod", "txt_mod", "cos", "sin"))
    def expected_ports():
        ports = {}
        for stream in ("image", "text"):
            ff = "ff" if stream == "image" else "ff_context"
            for field, module in (("projected", ff+".linear_in"),("activated", ff+".act_fn.gate_fn"),("product",ff+".act_fn"),("mlp",ff)):
                ports[stream+"."+field] = module+":return[witness]"
            ports[stream+".mlp_input"] = ff+":input[witness]"
            ports[stream+".norm"] = "attn:input."+("hidden_states[image-witness]" if stream == "image" else "encoder_hidden_states[text-witness]")
            ports[stream+".attention"] = "attn:return."+stream+"[witness]"
            for field in "qkv":
                ports[stream+"."+field] = "native.dispatch_attention_fn:input."+field+"["+stream+"-witness]"
            ports[stream+".head"] = "native.dispatch_attention_fn:return["+stream+"-witness]"
            ports[stream] = "native.block:return."+stream+"[complete]"
        return ports
    named_ports = expected_ports()
    native, comparisons, joins, equations = {}, {}, [], {}
    rowmap = {(r["branch"],r["case"],r["block"]):r for r in rows}
    assert len(rowmap) == len(rows) == 9
    for row in rows:
        key = (row["branch"], row["case"], row["block"])
        name, index = row["case"], row["block"]
        r = row["receipt"]
        assert r["operation"] == "continue" and r["steps"] == 1
        assert (r["parent"],r["result"]) == (row["parent"],row["result"])
        before, after = store.load(row["parent"]), store.load(row["result"])
        assert before.model_identity == after.model_identity == config["model"]
        assert before.execution["device"] == after.execution["device"] == "cpu"
        assert before.execution_point["logical_step"] == after.execution_point["logical_step"] == 2
        assert before.execution_point["index"] == index and after.execution_point["index"] == index+1
        bp, ap = before.payload, after.payload
        assert bp["advanced"] == index-1 and ap["advanced"] == index
        pre, post = fields(row["parent"]), fields(row["result"])
        assert same_context(pre, post)
        if index == 1:
            assert row["branch"] == "native"
            for field in ("image","img_mod","txt_mod","cos","sin"):
                assert np.array_equal(bp[field].float().numpy(), arrays[field])
            assert np.array_equal(bp["text"].float().numpy(), roots[name])
        elif row["branch"] == "native":
            assert row["parent"] == rowmap[("native",name,1)]["result"]
        else:
            assert row["branch"] == "predicted-input"
            for stream in ("image","text"):
                assert np.array_equal(bp[stream].float().numpy(), predictions[1][name][stream])
        p = data / row["archive"]["path"]
        assert sha(p) == row["archive"]["sha256"] and row["complete_fields_from_state"] == ["image","text"]
        with np.load(p, allow_pickle=False) as z:
            values = {k:z[k] for k in z.files}
        assert not {"image","text"} & set(values)
        values.update({s:ap[s].float().numpy() for s in ("image","text")})
        assert set(values) == set(named_ports) == set(row["ports"]) == set(predictions[index][name])
        for field, value in values.items():
            port = row["ports"][field]
            assert port["port"] == named_ports[field]
            assert port["descriptor"]["sha256"] == bits(value)
            assert port["descriptor"]["shape"] == list(value.shape) and port["descriptor"]["dtype"] == "torch.bfloat16"
            assert np.isfinite(value).all()
        native[key] = values
        label = ":".join(map(str,key))
        comparisons[label] = {k:error(v,predictions[index][name][k]) for k,v in values.items()}
        # Independently compose observed image/text writes through their consumers.
        equations[label] = {}
        for stream, modname, selected in (("image","img_mod",config["indices"]),("text","txt_mod",config["text_indices"])):
            mod = np.split(arrays[modname].reshape(1,1,-1),6,-1)
            u = B(bp[stream].float().numpy()[:,selected]+B(mod[2]*values[stream+".attention"]))
            target = B(u+B(mod[5]*values[stream+".mlp"]))
            equations[label][stream] = error(values[stream][:,selected],target)
        joins.append(dict(key=list(key),ports=len(values),parent=row["parent"],result=row["result"],receipt=r["fingerprint"],context_preserved=True))
        del bp, ap, before, after
    fork_joins = []
    for fork in forks:
        name = fork["case"]
        assert fork["source"] == rowmap[("native",name,1)]["result"]
        source, parent = store.load(fork["source"]), store.load(fork["fork_parent"])
        assert parent.parent == source.fingerprint
        assert fields(source.fingerprint) == fields(parent.fingerprint)
        last = parent.fingerprint
        for stream, act in zip(("image","text"),fork["acts"],strict=True):
            assert act["stream"] == stream and act["before"] == last
            r = act["receipt"]
            assert r["operation"] == "apply" and r["parent"] == last and r["result"] == act["after"]
            assert r["act"] == act["act"] and act["act"]["writes"] == [stream]
            pre, post = fields(act["before"]),fields(act["after"])
            assert all(pre[k] == post[k] for k in pre if k != stream)
            cut = store.load(act["after"])
            assert np.array_equal(cut.payload[stream].float().numpy(),predictions[1][name][stream])
            last = act["after"]
        assert last == fork["predicted_input"] == rowmap[("predicted-input",name,2)]["parent"]
        assert fork["result"] == rowmap[("predicted-input",name,2)]["result"]
        assert same_context(fields(fork["source"]),fields(last))
        assert fields(fork["source"])["advanced"] == fields(last)["advanced"]
        fork_joins.append(dict(case=name,sealed_complete_input_exact=True,context_and_clock_preserved=True,typed_acts=2))
    closure, effects = {}, {}
    for name in config["cases"]:
        closure[name] = {}
        for stream in ("image","text"):
            n2 = native[("native",name,2)][stream].astype(np.float64)
            c2 = native[("predicted-input",name,2)][stream].astype(np.float64)
            p2 = predictions[2][name][stream].astype(np.float64)
            local, propagation, total = c2-p2,n2-c2,n2-p2
            closure[name][stream] = dict(total=metric(total),own_input_arithmetic=metric(local),
                native_propagation=metric(propagation),signed_additive_closure=metric(total-local-propagation),
                local_propagation_cosine=float(np.vdot(local,propagation)/max(np.linalg.norm(local)*np.linalg.norm(propagation),1e-300)))
    for index in config["blocks"]:
        effects[str(index)] = {}
        for name in ("quarter","midpoint"):
            effects[str(index)][name] = {}
            for stream in ("image","text"):
                actual = native[("native",name,index)][stream].astype(np.float64)-native[("native","base",index)][stream]
                pred = predictions[index][name][stream].astype(np.float64)-predictions[index]["base"][stream]
                e = error(actual,pred)
                magnitude = metric(actual)["rms"]
                e["relative_to_observed_effect"] = e["rms"]/magnitude if magnitude else None
                effects[str(index)][name][stream] = dict(observed=metric(actual),predicted=metric(pred),error=e)
    result = dict(schema="mdb-flux-state-closure-audit-v1",mechanics_status="verified",
        instrument_status="complete typed component cuts and actual native ports; no ATen replay",
        trend_status="all endpoint, effect and propagation errors retained",terminal_status="unassessed",
        job_id=summary["job_id"],request_sha256=request["request_sha256"],custody=custody,transaction=verified,
        counts=summary["counts"],parameter_recipe_joins=joined,state_port_joins=joins,predicted_fork_joins=fork_joins,
        errors=comparisons,actual_port_residual_equations=equations,continuation_error_accounting=closure,effects=effects,
        worker_seconds=summary["worker_seconds"],new_model_calls_in_audit=0,
        scope="All1024image and512text exit rows of original joint1 and2. P2consumes complete sealed P1. Native predicted-input forks consume sealed P1 exactly. Quarter newly tested internal state; no new natural prompts or full denoiser/scheduler/VAE continuation.",
        audit_sources={p.name:sha(p) for p in (Path(__file__),here/"analyze_flux_attention_state_law.py",here/"flux_family_custody.py",here/"summarize_context_window.py")})
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    return dict(mechanics_status=result["mechanics_status"],counts=result["counts"],
        exits={k:{s:v[s] for s in ("image","text")} for k,v in comparisons.items()},
        continuation_error_accounting=closure,effects=effects)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collection",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();print(json.dumps(audit(a.collection.resolve(),a.output.resolve()),sort_keys=True))
