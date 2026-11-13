"""One model-free mrun lease audits pinned, already completed evidence trees."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from audit_wsp_binding import audit
from flux_family_custody import emit
from wsp_binding_protocol import save, sha


def run(config, request):
    start = time.monotonic()
    root = Path(config["output_scratch"])/("wsp-binding-audit-"+os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True, exist_ok=False)
    save(root/"config.json", config)
    save(root/"request.json", request)
    results = {}
    for name, record in config["studies"].items():
        source = Path(record["root"])
        if sha(source/"files.json") != record["files_manifest_sha256"] or sha(source/"summary.json") != record["summary_sha256"]:
            raise ValueError("completed source evidence differs from collected publication")
        result = audit(source)
        save(root/(name+"-audit.json"), result)
        results[name] = {"job_id": result["job_id"], "mechanics_status": result["mechanics_status"],
            "files_verified": result["files_verified"], "comparisons": result["comparisons"],
            "arms": {k: {q: v.get(q) for q in ("objective", "generated_target_correct", "generated_control_correct")}
                for k, v in result["arms"].items()}}
    comparison = None
    if "initial-evaluation" in config["studies"]:
        import numpy as np

        from mdb_lanes.forecast_transaction import load_forecast_tree

        first = Path(config["studies"]["initial-evaluation"]["root"])
        second = Path(config["studies"]["evaluation"]["root"])
        comparison = {"inputs_bytes_equal": sha(first/"inputs.json") == sha(second/"inputs.json"),
            "selected_factors_bytes_equal": sha(first/"selected.npz") == sha(second/"selected.npz"),
            "edited_parameter_manifests_equal": json.loads((first/"selected-patch-manifest.json").read_text()) ==
                json.loads((second/"selected-patch-manifest.json").read_text()), "arms": {},
            "scope": "same frozen candidate and inputs, public-runtime reproduction, not another independent split"}
        for arm in ("native", "broad", "role-compiled", "role-factored", "restored"):
            def observation(source):
                p = source/"evaluation"
                record = json.loads((p/"observations"/(arm+".json")).read_text())
                return load_forecast_tree(p, record["value"])["result"]["observations"]
            a, b = observation(first), observation(second)
            comparison["arms"][arm] = {"saved_full_logits_max_abs_difference":
                float(np.max(np.abs(np.asarray(a["logits"])-np.asarray(b["logits"])))),
                "generated_ids_equal": [list(r) for r in a["generated_ids"]] == [list(r) for r in b["generated_ids"]]}
        save(root/"replay-comparison.json", comparison)
    summary = {"schema": "mdb-role-selector-reader-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "studies": results, "model_loads": 0,
        "physical_forwards": 0, "elapsed_s": time.monotonic()-start, "terminal_status": "not-evaluated",
        "replay_comparison": comparison}
    save(root/"summary.json", summary)
    save(root/"files.json", {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()})
    emit(root, summary, compression_level=1)


if __name__ == "__main__":
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()),
        json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text()))
