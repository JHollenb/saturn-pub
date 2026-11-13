"""Recover completed native observations after report serialization failed.

No model imports or execution. Original files remain unchanged. The new output
retains the malformed report and declares its later custody boundary explicitly.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from worker import require, save, sha


def recover(config, output):
    from mdb.forecast_transaction import load_forecast_tree
    from mdb.wsp import verify_weight_study

    source = Path(config["recovery_root"])
    request = json.loads((source / "request.json").read_text())
    require(request["request_sha256"] == config["recovery_request_sha256"], "failed source request differs")
    inventory = {}
    for path in sorted(source.rglob("*")):
        require(not path.is_symlink(), "source recovery refuses symbolic links")
        if not path.is_file():
            continue
        name = str(path.relative_to(source))
        inventory[name] = sha(path)
        if name in {"request.json", "native-brief.json"}:
            destination = output / "failed-report" / name
        else:
            destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "recovery destination already exists")
        shutil.copyfile(path, destination)
        require(sha(destination) == inventory[name], "copied evidence differs")
    verification = verify_weight_study(output / "study")
    save(output / "wsp-verification.json", verification)
    brief, total_forwards = {}, 0
    for name in ("native", "edited", "substitute", "restore"):
        record = json.loads((output / "study/observations" / (name + ".json")).read_text())
        observed = load_forecast_tree(output / "study", record["value"])
        require(observed["instrument_status"] == "observed", "recovery requires completed source observations")
        total_forwards += observed["instrumented_work"]["root_forward_calls"]
        brief[name] = {cid: {key: value for key, value in row.items() if key not in {"first_logits", "token_ids"}}
            for cid, row in observed["result"]["observations"]["contexts"].items()}
    save(output / "native-brief.json", brief)
    receipt = {"schema": "mdb-wsp-report-recovery-v1", "source_job": config["recovery_job"],
        "source_request_sha256": config["recovery_request_sha256"], "source_files_sha256": inventory,
        "source_failure": "JSON serialization of read-only sampling metadata after all native arms finished",
        "original_files_modified": False, "new_model_loads": 0, "new_model_forwards": 0,
        "custody_scope": "first externally collected outcome pins occur in this recovery; original request was pinned before native execution",
        "forecast_and_outcome_verification": verification}
    save(output / "recovery-receipt.json", receipt)
    return {"phase": "recover", "instrument_status": "observed", "native_model_forwards": 0,
        "source_target_root_forwards": total_forwards, "source_job": config["recovery_job"],
        "source_request_sha256": config["recovery_request_sha256"],
        "scope": "saved-observation recovery after reporting failure; bounded think35 discovery, no benchmark or held-out claim"}
