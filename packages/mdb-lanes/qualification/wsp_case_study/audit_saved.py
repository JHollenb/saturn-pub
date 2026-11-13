"""Independent saved-array reader; no model loading, native forwards or editor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inventory(root, expected):
    root = Path(root)
    if sha(root / "files.json") != expected:
        raise ValueError("independent source file inventory differs")
    files = json.loads((root / "files.json").read_text())
    for name, fingerprint in files.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError("unsafe saved evidence path")
        if sha(path) != fingerprint:
            raise ValueError("saved source artifact differs")


def numerical_error(prediction, target):
    """Independent reader arithmetic, separate from the experiment's scorer."""
    import numpy as np

    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("invalid saved local projection geometry or values")
    residual = prediction - target
    rms, reference = float(np.sqrt(np.mean(residual ** 2))), float(np.sqrt(np.mean(target ** 2)))
    return {"scalars": int(target.size), "rms": rms, "relative_rms": rms / reference if reference else None,
            "maximum_absolute_error": float(np.max(np.abs(residual))), "signed_mean_error": float(residual.mean())}


def audit(roots, pins):
    import numpy as np

    from mdb_lanes.forecast_transaction import load_forecast_tree
    from mdb_lanes.wsp import verify_weight_study

    static, smoke = Path(roots["audit"]), Path(roots["native"])
    inventory(static, pins["audit_files_sha256"])
    inventory(smoke, pins["native_files_sha256"])
    diff = json.loads((static / "weight-diff.json").read_text())
    metadata = json.loads(Path("edit-manifest.json").read_text())
    if diff["changed_elements"] != metadata["changed_entries_total"]:
        raise ValueError("changed element count differs from pinned sparse manifest")
    vectors, static_rows = [], []
    with np.load(static / "derived-delta-columns.npz", allow_pickle=False) as data:
        for index, row in enumerate(diff["tensors"]):
            manifest = metadata["tensors"][index]
            delta, columns = data[f"t{index}_delta"], data[f"t{index}_columns"]
            if (row["name"] != manifest["name"] or row["after"]["sha256"] != manifest["edited_sha256"]
                    or row["before"]["sha256"] != manifest["base_sha256"]
                    or int(np.count_nonzero(delta)) != row["numerically_changed_elements"]
                    or list(columns) != row["columns"]["indices"]):
                raise ValueError("saved difference support or source identity differs")
            u, singular, _ = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
            vectors.append(u[:, 0])
            static_rows.append({"name": row["name"], "changed_elements": row["changed_elements"],
                "changed_columns": len(columns), "rank_one_remainder": float(np.linalg.norm(singular[1:]) / np.linalg.norm(singular))})
    verification = verify_weight_study(smoke / "study", expected_manifest=pins["native_outcome_manifest"],
                                        expected_seal=pins["native_forecast_seal"])
    observations, native_observations = {}, None
    for name in ("native", "edited", "substitute", "restore"):
        record = json.loads((smoke / "study/observations" / (name + ".json")).read_text())
        loaded = load_forecast_tree(smoke / "study", record["value"])["result"]["observations"]
        observations[name] = loaded["contexts"]
        if name == "native":
            native_observations = loaded
    comparisons = []
    for left, right in (("edited", "substitute"), ("native", "restore")):
        for cid, left_row in observations[left].items():
            right_row = observations[right][cid]
            a, b = left_row["first_logits"], right_row["first_logits"]
            comparisons.append({"left": left, "right": right, "context": cid,
                "different_first_logits": int(np.count_nonzero(a != b)),
                "maximum_first_logit_difference": float(np.max(np.abs(a.astype(np.float64) - b))),
                "own_token_ids_equal": bool(np.array_equal(left_row["token_ids"], right_row["token_ids"])),
                "emitted_tokens": len(left_row["token_ids"])})
    local_errors = []
    targets = native_observations.get("local_projection_targets", {})
    if targets:
        record = json.loads((smoke / "study/forecasts/native.json").read_text())
        predictions = load_forecast_tree(smoke / "study", record["predictions"])["local_forecasts"]
        for cid, layers in targets.items():
            for name, values in layers.items():
                local_errors.append({"context": cid, "parameter": name,
                    "endpoint": numerical_error(predictions[cid][name]["edited"], values["edited"]),
                    "edit_response": numerical_error(predictions[cid][name]["direct"], values["edited"] - values["original"])})
    behavior = {arm: {cid: {key: row.get(key) for key in (
        "generated_tokens", "completed_eos", "hit_cap", "final_response", "final_boxed_answer",
        "final_boxed_literal_matches_supplied_answer", "maximum_repeated_trigram_count", "sampling")}
        for cid, row in rows.items()} for arm, rows in observations.items()}
    return {"schema": "mdb-wsp-independent-saved-audit-v2", "static_rows": static_rows,
        "absolute_cosine_of_leading_output_directions": float(abs(np.dot(vectors[0], vectors[1]))),
        "native_verification": verification, "native_comparisons": comparisons, "behavior": behavior,
        "native_model_forwards": 0, "local_projection_errors": local_errors,
        "local_projection_error_audit": "recomputed from sealed forecasts and retained native arrays" if targets else
            "reported reductions only; native local target vectors were not retained in this initial run",
        "scope": "source/support verification and independent recomputation of saved continuation comparisons; no semantic or held-out claim"}
