"""Saved-only publication recovery for the completed Qwen spectral transaction.

The failed original job is immutable. No neural framework is imported and no
forecast, suffix or continuation is executed; only retained arrays are read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import re
import runpy
import shutil
import sys
from pathlib import Path

SCHEMA = "mdb-qwen-spectral-readout-recovery-v2"
RESULT = "qwen-spectral-readout-recovery-result"
SOURCE_JOB = "job-956049ba0eeb"
SOURCE_REQUEST = "4fd54f321531cc97aed4508c76c8899e43fc9ed507a4b675a7e9976848f71713"
HELPERS = {
    "qwen_spectral_readout_campaign.py": "a5a54ec9aa7c92a4c912c5703209278f03f0746980d0a92c59a0e12e09cafe97",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
}
PATH_KEYS = ("native_root", "source_inventory", "source_job", "source_worker_log", "output_scratch")
PIN_KEYS = ("source_inventory_sha256", "source_job_sha256", "source_worker_log_sha256")
BLOCKED = ("torch", "transformers", "diffusers", "safetensors")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def request_identity(request):
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_config(base):
    """Structural only: the launcher may supply relative metadata filenames."""
    keys = (*PATH_KEYS, *PIN_KEYS, "native_summary_sha256")
    config = {key: base[key] for key in keys if key in base}
    config.setdefault("native_root", "/mnt/big/mdb/research/"+SOURCE_JOB+"/qwen-spectral-readout-result")
    config.setdefault("source_inventory", "source-inventory.json")
    config.setdefault("source_job", "source-job.json")
    config.setdefault("source_worker_log", "source-worker.log")
    config.setdefault("output_scratch", "/mnt/big/mdb/research")
    config.update(schema=SCHEMA, output_name=RESULT, native_job_id=SOURCE_JOB, native_request_sha256=SOURCE_REQUEST,
        helper_source_sha256=dict(HELPERS), max_source_files=8192, max_source_bytes=1 << 30,
        transport_limits=dict(max_expanded_bytes=128 << 20, max_member_bytes=32 << 20), model_calls=0)
    return config


def validate_config(config):
    fixed = make_config(config)
    for key in fixed.keys() - set((*PATH_KEYS, *PIN_KEYS, "native_summary_sha256")):
        require(config.get(key) == fixed[key], "fixed recovery contract differs: " + key)
    for key in PATH_KEYS:
        require(type(config.get(key)) is str and bool(config[key]), "recovery path required: " + key)
    for key in PIN_KEYS:
        require(type(config.get(key)) is str and re.fullmatch("[0-9a-f]{64}", config[key]), "metadata SHA256 required: " + key)
    if "native_summary_sha256" in config:
        require(type(config["native_summary_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", config["native_summary_sha256"]), "invalid optional summary SHA256")
    return dict(model_calls=0, native_root=config["native_root"], result=RESULT)


def helpers():
    result = {}
    for name, expected in HELPERS.items():
        path = Path(__file__).with_name(name)
        require(sha(path) == expected, "frozen publication helper changed: " + name)
        result[name] = runpy.run_path(str(path))
    return result


def capture_inventory(root, *, job_id=SOURCE_JOB, request_sha256=SOURCE_REQUEST):
    """Near-data checksum metadata only; publish this outside the original root."""
    root = Path(root).resolve(); files = {}
    require(root.is_dir(), "completed source output root required")
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "source symlink forbidden")
        if path.is_file():
            before = path.stat(); value = dict(bytes=before.st_size, sha256=sha(path)); after = path.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "source file changed while hashing")
            files[path.relative_to(root).as_posix()] = value
    return dict(schema="mdb-qwen-spectral-source-inventory-v1", job_id=job_id, request_sha256=request_sha256,
        files=files, expanded_bytes=sum(value["bytes"] for value in files.values()))


def authenticate_inventory(root, inventory, *, job_id, request_sha256, max_files=8192, max_bytes=1 << 30):
    require(inventory.get("schema") == "mdb-qwen-spectral-source-inventory-v1" and inventory.get("job_id") == job_id
        and inventory.get("request_sha256") == request_sha256, "externally pinned source inventory identity differs")
    files = inventory.get("files")
    require(type(files) is dict and 0 < len(files) <= max_files and type(inventory.get("expanded_bytes")) is int
        and 0 < inventory["expanded_bytes"] <= max_bytes, "bounded complete source inventory required")
    for name, record in files.items():
        require(type(name) is str and name and not Path(name).is_absolute() and not any(part in (".", "..") for part in Path(name).parts)
            and type(record) is dict and set(record) == {"sha256", "bytes"} and type(record["bytes"]) is int and record["bytes"] >= 0
            and type(record["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", record["sha256"]), "unsafe source inventory record")
    require(capture_inventory(root, job_id=job_id, request_sha256=request_sha256) == inventory, "saved source inventory/content differs")
    return inventory


def source_accounts(config, qualified):
    """Authenticate failed-job custody and original metadata, without weights."""
    root = Path(config["native_root"])
    for key in ("source_inventory", "source_job", "source_worker_log"):
        require(sha(config[key]) == config[key+"_sha256"], "source metadata pin differs: " + key)
    inventory = authenticate_inventory(root, read(config["source_inventory"]), job_id=config["native_job_id"], request_sha256=config["native_request_sha256"],
        max_files=config["max_source_files"], max_bytes=config["max_source_bytes"])
    summary, source, runtime, request = (read(root / name) for name in ("summary.json", "source.json", "runtime.json", "request.json"))
    if "native_summary_sha256" in config:
        require(sha(root / "summary.json") == config["native_summary_sha256"], "explicit original summary pin differs")
    require(request_identity(request) == request["request_sha256"] == config["native_request_sha256"]
        and request["source_hashes"].get("qwen_spectral_readout_campaign.py") == HELPERS["qwen_spectral_readout_campaign.py"], "actual original request/worker differs")
    producer = qualified["qwen_spectral_readout_campaign.py"]; producer["validate_config"](request["config"])
    require(read(root / "declared-config.json") == request["config"] and source["job_id"] == summary["job_id"] == config["native_job_id"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"], "original source/job/request join differs")
    require(source["checkpoint_files"] == producer["CHECKPOINT"] and source["model_identity"] == summary["model_identity"]
        and source["execution"] == runtime["execution"] and source["genuine_record_prefix_capture"] is True and source["raw_input_prediction"] is False,
        "source checkpoint/profile/conditional scope differs")
    require(all(source["source_identity"].get(name) == value for name, value in request["source_hashes"].items()), "original source closure differs")
    from mdb.parameter_artifacts import validate_manifest

    manifest = validate_manifest(read(root / "parameter-artifacts.json")); bindings = read(root / "weight-binding.json"); headers = read(root / "checkpoint-headers.json")
    require(set(bindings) == {"postnorm", "gate", "up", "down", "finalnorm", "head"} and len(manifest["artifacts"]) == 6
        and sha(root / "retrieve_weights.py") == producer["RETRIEVER_SHA256"], "six exact original parameter recipes/retriever required")
    for row in manifest["artifacts"]:
        name = Path(row["path"]).stem; binding = bindings[name]; original = row["source"]
        require(row["path"] == "parameters/"+name+".npy" and original["file_sha256"] == source["checkpoint_files"][original["file"]]["sha256"]
            and original["tensor"] in headers[original["file"]]["selected"] and original["tensor"] == binding["original_tensor"]
            and original["payload_sha256"] == binding["original_payload_sha256"] and binding["native_dtype"] == "float32"
            and binding["parameter_archive_written"] is False and not (root / row["path"]).exists(), "original parameter metadata binding differs")
    job = read(config["source_job"]); log = Path(config["source_worker_log"]).read_bytes(); result = job.get("result") or {}
    require(job.get("job_id") == config["native_job_id"] and job.get("state") == "failed" and result.get("returncode") not in (None, 0), "actual original failed job required")
    for envelope in (job["config"], job["submitted_request"]["config"]):
        require(all(envelope.get(key) == value for key, value in request.items()), "actual failed-job request envelope differs")
    require(job["config"].get("retry_on_kill") is False and not job.get("needs", {}).get("cuda", False), "source CPU/no-retry admission differs")
    receipts = [job["payload_custody"][key] for key in ("declared", "sealed", "executed")] + [result["executed_payload"]]
    require(len({(value["sha256"], value["size_bytes"]) for value in receipts}) == 1, "failed-job executed source payload custody differs")
    expected_log = result.get("artifacts", {}).get("job_log")
    if expected_log is not None:
        require(dict(bytes=len(log), sha256=sha(config["source_worker_log"])) == {key: expected_log[key] for key in ("bytes", "sha256")}, "complete failed-job log differs")
    text = log.decode()
    require(all(value in text for value in ("publish_accounts", "binding.execution.torch", "TorchVersion")) and "MDB_DIRECTORY_READY " not in text,
        "failed-after-publication-entry diagnostic differs")
    return dict(inventory=inventory, summary=summary, source=source, runtime=runtime, request=request,
        failed_job=dict(job_id=job["job_id"], state=job["state"], returncode=result["returncode"], assigned_host=job.get("assigned_host"),
            job_sha256=config["source_job_sha256"], worker_log_sha256=config["source_worker_log_sha256"], executed_payload=result["executed_payload"],
            failure="Strict public JSON copier rejected TorchVersion at publication after the closed scientific transaction."))


def check_guard_journal(journal, prefixes, prediction, spectral, summary, seal_sha256):
    """Chronological seal exclusion and per-ID completeness, never map ordering."""
    count = len(prefixes)
    require([row["number"] for row in journal] == list(range(2*count+1))
        and [row["kind"] for row in journal] == ["genuine_prefix_complete"]*count+["global_forecasts_sealed"]+["prompt_targets_complete"]*count,
        "source guard chronology differs")
    captures, targets = journal[:count], journal[count+1:]
    capture_ids = [row["prompt"] for row in captures]; target_ids = [row["prompt"] for row in targets]
    require(len(set(capture_ids)) == len(set(target_ids)) == count and set(capture_ids) == set(target_ids) == set(prefixes),
        "source prompt receipts are not a unique complete cohort")
    require(all(row["final_suffix_entries"] == 0 for row in captures) and journal[count]["seal_sha256"] == seal_sha256
        and journal[count]["pretarget_counts"] == dict(spectral["pretarget_counts"]), "pretarget guard/seal receipts differ")
    pre = journal[count]["pretarget_counts"]
    require(pre["prefixes"] == pre["physical_entries"] == count and all(value == 0 for key, value in pre.items() if key not in ("prefixes", "physical_entries"))
        and journal[-1]["counts"] == summary["counts"] and summary["counts"]["denied_preseal_entries"] == 0, "actual suffix work preceded seal or final count differs")
    require(all(row["selected_future_rows"] == list(prediction["prompts"][row["prompt"]]["selection"]["rows"]) for row in targets),
        "source target own-forecast selection differs")
    return dict(guard_capture_order=capture_ids, target_completion_order=target_ids, pretarget_counts=pre)


def transaction_accounts(root, summary, *, source_config=None):
    """Verify saved public Tx/CAS, all decoded leaves and actual guard chronology."""
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root = Path(root); tx = root / "transaction"; seal = read(tx / "seal.json")
    verified = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"]["seal_sha256"], expected_event_head=summary["transaction"]["event_head"])
    require(verified == summary["transaction"] and verified["phase"] == "closed" and verified["events"] == 8
        and not seal["violations"] and seal["frontiers"] == ["prefixes", "spectral-source"] and seal["forecasts"] == ["cohort"], "closed original prospective transaction differs")
    events = [read(path) for path in sorted((tx / "events").glob("*.json"))]
    require([value["kind"] for value in events] == ["opened", "frontier_published", "frontier_published", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"], "source transaction release order differs")
    declaration = seal["declarations"]
    require(declaration["model_identity"] == summary["model_identity"] and declaration["input_identity"] == summary["input_identity"]
        and all(seal["pre_release_work"][name] == 0 for name in ("executions", "native_calls", "denied_attempts")), "preseal identity/native exclusion differs")
    forecast_record = read(tx / "forecasts/cohort.json")
    prediction = load_forecast_tree(tx, forecast_record["predictions"])
    observation_record = read(tx / "observations/native.json")
    require(observation_record["seal_sha256"] == verified["seal_sha256"], "posttarget observation seal join differs")
    observation = load_forecast_tree(tx, observation_record["value"])
    prefixes = load_forecast_tree(tx, read(tx / "frontiers/prefixes.json")["value"])
    spectral = load_forecast_tree(tx, read(tx / "frontiers/spectral-source.json")["value"])
    require(set(prefixes) == set(prediction["prompts"]) == set(observation["prompts"]) and prediction["directions_once"] is True
        and prediction["continuation_forecast"] is False and dict(observation["counts"]) == summary["counts"], "saved forecast/target cohort or actual census differs")
    random_payloads = {}
    for context in prefixes:
        descriptor = forecast_record["predictions"]
        for key in ("prompts", context, "parity", "random", "positive_input"):
            descriptor = descriptor["items"][key]
        name = context+".random.positive"; payload = forecast_record["payloads"][name]
        require(descriptor == payload["value"] == payload["original"], "random preseal prediction/named-payload binding differs")
        random_payloads[context] = dict(name=name, content_sha256=descriptor["content_sha256"], shape=descriptor["shape"], dtype=descriptor["dtype"])
    journal = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]; count = len(prefixes)
    chronology = check_guard_journal(journal, prefixes, prediction, spectral, summary, verified["seal_sha256"])
    callback_order = None
    if source_config is not None:
        callback_order = [row["id"] for row in source_config["prompts"]]
        require(callback_order == chronology["guard_capture_order"], "genuine prefix capture order differs from declared raw prompt order")
    expected = dict(prefixes=count, native_MLP_bases=count, component_readouts=0, parity_targets=0, ordinary_prefills=0, ordinary_shams=count, cached_futures=0)
    for key, row in observation["prompts"].items():
        prior = prediction["prompts"][key]
        require(set(row["component"]) == set(prior["grid"]["rows"]) and list(row["selection"]["rows"]) == list(prior["selection"]["rows"])
            and set(row["ordinary"]) == {*prior["selection"]["rows"], "sham"}, "complete native grid/forecast-selected ordinary cohort differs")
        expected["component_readouts"] += len(row["component"]); expected["parity_targets"] += len(row["parity"])
        expected["ordinary_prefills"] += len(row["ordinary"])-1
        expected["cached_futures"] += sum(len(value["continuation"]) for value in row["ordinary"].values())
    expected["physical_entries"] = sum(expected.values())
    suffix = expected["native_MLP_bases"]+expected["parity_targets"]+expected["ordinary_prefills"]+expected["ordinary_shams"]+expected["cached_futures"]
    tail = expected["component_readouts"]+expected["ordinary_prefills"]+expected["ordinary_shams"]+expected["cached_futures"]
    expected.update(final_postnorm_entries=suffix, final_MLP_entries=suffix, final_norm_entries=tail, head_entries=tail, denied_preseal_entries=0)
    require(expected == summary["counts"], "actual source physical/module census differs from complete saved cohorts")
    return dict(summary=summary, predictions=prediction, observations=observation), dict(verified=verified, guard_chronology=journal,
        serialized_frontier_order=list(prefixes), callback_decode_order=callback_order,
        random_positive_payloads=random_payloads,
        callback_order_provenance="Frozen producer constructs the frontier in source-config order; public Tx retains its in-memory encoded map for callbacks. Saved canonical JSON sorts map keys. Guard capture order independently joins the declaration.",
        **chronology, complete_scientific_counts=expected)


def random_seed_accounts(prediction, source_config, verification):
    """Compare complete preseal random inputs; mismatches are diagnostics only."""
    import numpy as np

    hypotheses = {"source_config_callback": verification["callback_decode_order"], "serialized_metadata_reload": verification["serialized_frontier_order"]}
    result = {}
    for name, order in hypotheses.items():
        require(order is not None and set(order) == set(prediction["prompts"]), "source-bound random seed cohort required")
        rows = {}
        for index, context in enumerate(order):
            prior = prediction["prompts"][context]["parity"]["random"]; actual = prior["positive_input"]
            require(actual.ndim == 1 and actual.dtype == np.float32 and actual.size > 0, "saved random antipode input geometry differs")
            seed = source_config["seed"]+index+1
            direction = np.random.default_rng(seed).standard_normal(actual.size); direction /= np.linalg.norm(direction)
            expected = (prior["requested_raw_norm"]*direction).astype(np.float32)
            changed = int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32)))
            shared_zero = (expected == 0) & (actual == 0)
            rows[context] = dict(seed=seed, elements=int(actual.size), FP32_bit_changed=changed,
                signed_zero_defects=int(np.count_nonzero(shared_zero & (np.signbit(expected) != np.signbit(actual)))),
                max_abs=float(np.max(np.abs(expected.astype(np.float64)-actual.astype(np.float64)))),
                expected_content_sha256=hashlib.sha256(expected.tobytes()).hexdigest(), recorded_content_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                status="bit-exact" if changed == 0 else "payload-mismatch-observed")
            if "random_positive_payloads" in verification:
                binding = verification["random_positive_payloads"][context]
                require(binding["content_sha256"] == rows[context]["recorded_content_sha256"], "decoded random input differs from its sealed named payload")
                rows[context]["sealed_payload"] = binding
        result[name] = dict(order=order, contexts=rows, bit_changed_total=sum(row["FP32_bit_changed"] for row in rows.values()))
    return dict(hypotheses=result, authority="Only complete preseal random-positive-input bytes and frozen Gaussian/normalize/rho/FP32 recipe; no target, weights or forward used.",
        instrument_status="source-config callback seed map matches complete saved inputs" if result["source_config_callback"]["bit_changed_total"] == 0 else "source-config seed-map mismatch observed; retained without rewriting or aborting",
        numerical_gate=False)


def publish_saved(root, output, *, execution, source_identity, run_id, qualified=None, source_config=None):
    """Core saved-fixture seam; production authenticates the external envelope first."""
    root, output = Path(root), Path(output); qualified = helpers() if qualified is None else qualified
    summary = read(root / "summary.json"); result, verification = transaction_accounts(root, summary, source_config=source_config)
    if source_config is not None:
        verification["random_control_seed_accounts"] = random_seed_accounts(result["predictions"], source_config, verification)
    account = output / "original-account"; (account / "transaction").mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "transaction/seal.json"):
        shutil.copyfile(root / name, account / name)
    # A JSON round-trip strips TorchVersion/other str subclasses without changing
    # represented values. Neither NumPy equations nor the original model rerun.
    execution = json.loads(json.dumps(execution, sort_keys=True, allow_nan=False))
    records = qualified["qwen_spectral_readout_campaign.py"]["publish_accounts"](result, account, execution=execution, source_identity=source_identity, run_id=run_id)
    return records, verification


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError("saved-only recovery forbids neural framework import: " + fullname)
        return None


def recover(config, output, *, job_id, request_sha256, own_request=None):
    validate_config(config); output = Path(output)
    require(not any(name in sys.modules for name in BLOCKED), "recovery process already imported a neural framework")
    guard = NoModelImports(); sys.meta_path.insert(0, guard)
    try:
        qualified = helpers(); authenticated = source_accounts(config, qualified); root = Path(config["native_root"]).resolve()
        require(not output.resolve().is_relative_to(root) and not root.is_relative_to(output.resolve()), "recovery output must be separate from original source")
        output.mkdir(parents=True, exist_ok=False)
        records, verification = publish_saved(root, output, execution=authenticated["runtime"]["execution"], source_identity=authenticated["source"]["source_identity"], run_id=job_id, qualified=qualified,
            source_config=authenticated["request"]["config"])
        authenticate_inventory(root, authenticated["inventory"], job_id=config["native_job_id"], request_sha256=config["native_request_sha256"], max_files=config["max_source_files"], max_bytes=config["max_source_bytes"])
        deviation = dict(status="protocol-deviation-retained", intended="One fixed norm-matched random direction reused across contexts.",
            executed="One source-seeded random direction per context, seed2718+context_index+1 in the original in-memory callback decode order, separately checked against all saved input bytes.",
            callback_decode_order=verification["callback_decode_order"], serialized_metadata_order=verification["serialized_frontier_order"],
            context_seeds={name: authenticated["request"]["config"]["seed"]+index+1 for index, name in enumerate(verification["callback_decode_order"])},
            seed_payload_verification=verification["random_control_seed_accounts"],
            main_direction_scope="The two weight-only spectral eigenvectors are shared across contexts; this random-control deviation does not change their source construction.",
            interpretation="Random antipodes are separate per-context controls; no shared-random-direction cross-context comparison is claimed.")
        report = dict(schema=SCHEMA, job_id=job_id, request_sha256=request_sha256, source_job_id=config["native_job_id"], source_request_sha256=config["native_request_sha256"],
            source_job_status="failed-after-completed-scientific-work; publication error preserved", failed_job=authenticated["failed_job"], model_calls=0,
            verification=verification, source_inventory_sha256=config["source_inventory_sha256"], native_root=str(root), immutable_source_preserved=True,
            original_account_base="original-account", metadata_only_seal_copy=True, relocated_complete_transaction=False,
            source_file_inventory=authenticated["inventory"], random_control_protocol_deviation=deviation,
            publication=dict(records=len(records["rows"]), catalog="original-account/catalog", atlas="original-account/atlas", accounts="original-account/accounts.json"),
            mechanics_status="authenticated saved-only recovery", terminal_status="unassessed")
        save(output / "report.json", report)
        save(output / "source.json", dict(schema=SCHEMA, job_id=job_id, request_sha256=request_sha256, helper_source_sha256=dict(HELPERS),
            worker_sha256=sha(__file__), source_job_id=config["native_job_id"], source_request_sha256=config["native_request_sha256"],
            source_metadata_sha256={key: config[key+"_sha256"] for key in ("source_inventory", "source_job", "source_worker_log")},
            execution=dict(backend="cpu", mode="saved-only NumPy/public Tx-CAS reader; no neural framework imported", model_calls=0),
            native_root=str(root), original_account_base="original-account"))
        if own_request is not None:
            save(output / "request.json", own_request); save(output / "declared-config.json", own_request["config"])
        save(output / "summary.json", dict(schema=SCHEMA, job_id=job_id, request_sha256=request_sha256, model_calls=0,
            source_job_id=config["native_job_id"], source_request_sha256=config["native_request_sha256"], source_job_status=report["source_job_status"],
            original_scientific_counts=verification["complete_scientific_counts"], recovery_counts=dict(model_calls=0, equation_records=len(records["rows"])),
            source_forecast_seal=verification["verified"]["seal_sha256"], random_control_protocol_deviation=deviation, original_account_base="original-account",
            mechanics_status=report["mechanics_status"], terminal_status="unassessed"))
        for key, name in (("source_inventory", "source-inventory.json"), ("source_job", "source-job.json"), ("source_worker_log", "source-worker.log")):
            shutil.copyfile(config[key], output / name)
        require(not any(name in sys.modules for name in BLOCKED) and not any(path.suffix in (".npy", ".npyb", ".safetensors") for path in output.rglob("*")), "recovery imported a model or copied a numerical payload")
        return qualified["flux_image_tail_microscope.py"]["finalize"](output, job_id=job_id, request_sha256=request_sha256, **config["transport_limits"])
    finally:
        sys.meta_path.remove(guard)


def worker(config_path):
    print("qwen-spectral saved-only recovery; no model calls; original failed job preserved", flush=True)
    config_path = Path(config_path).resolve(); config = read(config_path); validate_config(config)
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    require(request_identity(request) == request.get("request_sha256") and request.get("config") == config
        and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "recovery CPU request/config identity differs")
    for name, expected in request["source_hashes"].items():
        require(Path(name).name == name and sha(config_path.parent / name) == expected, "staged recovery source changed: " + name)
    require(request["source_hashes"].get(Path(__file__).name) == sha(__file__) and all(request["source_hashes"].get(name) == value for name, value in HELPERS.items()), "recovery worker/helper closure missing")
    job_id = os.environ["MRUN_JOB_ID"]; require(job_id.startswith("job-") and Path(job_id).name == job_id, "actual recovery job ID required")
    paths = {key: str((config_path.parent / config[key]).resolve()) if not Path(config[key]).is_absolute() else config[key] for key in PATH_KEYS}
    derived = dict(config, **paths); output = Path(derived["output_scratch"]) / job_id / RESULT
    receipt = recover(derived, output, job_id=job_id, request_sha256=request["request_sha256"], own_request=request)
    print("MDB_DIRECTORY_READY " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
