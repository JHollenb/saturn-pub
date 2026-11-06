"""Saved-only effect directions and strict sign/zero confusion on complete state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mdb.forecast_transaction import load_forecast_tree
from saturn_pub.store import LocalStore


def analyze(root, output):
    read = lambda p:json.loads(p.read_text())
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    audit = read(root / "audit-v1.json")
    assert audit["mechanics_status"] == "verified"
    data = root / "collection-01/extracted/binding-minimal-result"
    assert read(data/"summary.json")["request_sha256"] == audit["request_sha256"]
    tx = data / "transaction"
    predicted = {i:load_forecast_tree(tx,read(tx/f"forecasts/joint{i}.json")["predictions"]) for i in (1,2)}
    store = LocalStore(data / "store")
    rows = {(r["case"],r["block"]):r for r in read(data/"rows.json") if r["branch"] == "native"}
    results = {}
    for index in (1,2):
        base = store.load(rows[("base",index)]["result"]).payload
        for case in ("quarter","midpoint"):
            value = store.load(rows[(case,index)]["result"]).payload
            for stream in ("image","text"):
                actual = value[stream].float().numpy().astype(np.float64)-base[stream].float().numpy()
                forecast = predicted[index][case][stream].astype(np.float64)-predicted[index]["base"][stream]
                signs = np.zeros((3,3),np.int64)
                np.add.at(signs,((np.sign(forecast).astype(np.int8)+1).ravel(),
                                 (np.sign(actual).astype(np.int8)+1).ravel()),1)
                results[f"j{index}:{case}:{stream}"] = dict(
                    cosine=float(np.vdot(actual,forecast)/np.linalg.norm(actual)/np.linalg.norm(forecast)),
                    signed_confusion_predicted_rows_observed_columns=signs.tolist(),
                    sign_order=[-1,0,1],scalars=int(actual.size),
                    strict_sign_or_zero_agreement=float(np.trace(signs)/actual.size))
    result = dict(schema="mdb-flux-state-effect-geometry-v1",scope="All case-minus-base coordinates; exact signs including0, no magnitude selection; numeric tensor diagnostic, no semantic labels.",
                  results=results,new_model_calls=0,audit_sha256=sha(root/"audit-v1.json"),
                  source_sha256=sha(Path(__file__)))
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write("\n")
    return {k:{n:v[n] for n in ("cosine","strict_sign_or_zero_agreement")} for k,v in results.items()}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.root.resolve(),a.output.resolve()),sort_keys=True))
