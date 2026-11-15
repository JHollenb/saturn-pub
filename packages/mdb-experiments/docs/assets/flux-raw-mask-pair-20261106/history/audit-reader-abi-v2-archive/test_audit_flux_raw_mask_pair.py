"""Complete synthetic saved pair and hostile joins; no neural forwards."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdb.forecast_transaction import ForecastTransactionError

HERE = Path(__file__).resolve().parents[1] / "qualification"
A = runpy.run_path(str(HERE / "audit_flux_raw_mask_pair.py"))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def save_arrays(old, new, value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            target = new / value["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(old / value["path"], target)
        for item in value.values():
            save_arrays(old, new, item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            save_arrays(old, new, item)


def write_responses(root, account):
    from saturn_pub.store import LocalStore
    cases = list(A["CASES"])
    qwen = {case: A["read"](root / "cases" / case / "forecast-qwen.json") for case in cases}
    rows = {case: A["read"](root / "cases" / case / "forecast-state-rows.json") for case in cases}
    stores = {case: LocalStore(root / "cases" / case / "forecast-state-store") for case in cases}
    array = lambda case, record: account["load_array"](root / "cases" / case, record)
    report = dict(schema="mdb-raw-mask-pair-responses-v1", recipe=A["RECIPE"], qwen={}, denoiser=[], decoder={})
    for name in [str(index) for index in range(len(qwen["short"]["hidden_states"]))] + ["conditioning"]:
        values = [array(case, qwen[case]["conditioning"] if name == "conditioning" else qwen[case]["hidden_states"][int(name)]) for case in cases]
        report["qwen"][name] = A["response_descriptor"](*values)
    for index in range(len(rows["short"])):
        leaves = {case: dict(A["tensor_leaves"](stores[case].load(rows[case][index]["cut"]).payload)) for case in cases}
        report["denoiser"].append(dict(number=index - 1, clock=rows["short"][index]["clock"],
            fields={name: A["response_descriptor"](leaves["short"][name], leaves["long"][name]) for name in leaves["short"]}))
    decoded = {case: A["read"](root / "cases" / case / "decoder-forecast.json") for case in cases}
    for name in ("decoded", "pixels"):
        report["decoder"][name] = A["response_descriptor"](*(array(case, decoded[case][name]) for case in cases))
    put(root / "forecast-responses.json", report)


def fixture(root):
    """Build actual tiny LocalStores/recipes/parser files, not model outputs."""
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    helper = runpy.run_path(str(Path(__file__).with_name("test_audit_flux_raw_to_image.py")))
    old_roots = {}
    from transformers import Qwen2TokenizerFast
    original_save = Qwen2TokenizerFast.save_pretrained

    def save_synthetic_parser(self, *args, **kwargs):
        # This controlled synthetic template guarantees20/600 known-vocabulary
        # rows from the unchanged short/long raw inputs. It is a fixture renderer,
        # never a claim about the original pretrained tokenizer or semantics.
        self.chat_template = "{{ 'A ' * (600 if messages[0]['content']|length > 512 else 20) }}"
        return original_save(self, *args, **kwargs)

    Qwen2TokenizerFast.save_pretrained = save_synthetic_parser
    try:
        for case, prompt in A["CASES"].items():
            helper["A"]["PROMPT"] = prompt
            old_roots[case], _ = helper["saved_fixture"](root / ("synthetic-" + case))
    finally:
        Qwen2TokenizerFast.save_pretrained = original_save
    raw = root / "pair"
    raw.mkdir()
    old_config, old_source, old_contract = [A["read"](old_roots["short"] / name) for name in ("plan.json", "source.json", "native-contract.json")]
    shutil.copytree(old_roots["short"] / "source", raw / "source")
    config = {**old_config, "schema": "mdb-flux-raw-mask-pair-plan-v1", "cases": A["CASES"], "case_order": list(A["CASES"]),
        "expected_mask_regimes": {"short": "partially-padded", "long": "all-attended"}, "response_recipe": A["RECIPE"]}
    for key in ("request_sha256", "job_id"):
        config.pop(key, None)
    config["helper_source_sha256"] = {**config["helper_source_sha256"], "flux_raw_to_image.py": helper["A"]["WORKER_SHA256"]}
    request = dict(config=copy.deepcopy(config), source_hashes={"flux_raw_mask_pair.py": A["WORKER_SHA256"], **config["helper_source_sha256"]})
    request["request_sha256"] = A["identity"](request)
    config.update(request_sha256=request["request_sha256"], output=str(raw))
    put(raw / "plan.json", config)
    put(raw / "request.json", request)
    shutil.copyfile(old_roots["short"] / "runtime-contract.json", raw / "runtime-contract.json")
    qspec = A["read"](raw / old_source["source_files"]["qwen_config"]["path"])
    forecasts, natives, masks, sdpa, ordinal = {}, {}, {}, {"forecast": [], "native": []}, 0

    def form_case(case, lane):
        nonlocal ordinal
        old, new = old_roots[case], raw / "cases" / case
        new.mkdir(parents=True, exist_ok=True)
        fq = A["read"](old / "forecast-qwen.json")
        qwen = A["read"](old / (lane + "-qwen.json"))
        mask = helper["A"]["load_array"](old, qwen["attention_mask"])
        all_attended = bool(mask.all())
        regime = "all-attended" if all_attended else "partially-padded"
        dispatch = {"backend": "sdpa", "gqa_policy": "enable_gqa" if all_attended else "repeat_kv", "source": config["attention_dispatch"]["source"]}
        profile = dict(attended_rows=int(mask.sum()), padded_rows=int(mask.size - mask.sum()), regime=regime,
            expected_regime=config["expected_mask_regimes"][case], expected_regime_observed=regime == config["expected_mask_regimes"][case])
        masks[case] = profile
        qwen.update(formatted_chat=fq["formatted_chat"], dispatch=dispatch, mask_observation=profile,
            forecast_prefix_layers=config["prefix_layers"] if lane == "forecast" else None,
            executed_full_layers=config["full_layers"] if lane == "native" else None)
        qwen["actual_sdpa_argument_observations"] = dict(calls=config["prefix_layers" if lane == "forecast" else "full_layers"],
            expected_regime=regime, all_argument_profiles_match_declared_regime=True, selected_kernel="unprofiled")
        put(new / (lane + "-qwen.json"), qwen)
        save_arrays(old, new, qwen)
        if lane == "forecast":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(raw / "source/tokenizer", local_files_only=True)
            uncropped = tokenizer.apply_chat_template([dict(role="user", content=A["CASES"][case])], tokenize=True,
                padding=False, truncation=False, add_generation_prompt=True, enable_thinking=False)
            put(new / "input.json", dict(prompt=A["CASES"][case], prompt_sha256=hashlib.sha256(A["CASES"][case].encode()).hexdigest(),
                formatted_chat=fq["formatted_chat"], dispatch=dispatch, mask=profile, uncropped_token_count=len(uncropped)))
        front = A["read"](old / ("frontend-" + lane + ".json"))
        # Frozen _frontend_record writes this nested metadata record; the
        # paired caller additionally writes the top-level frontend alias.
        put(new / "frontend" / (lane + ".json"), front)
        put(new / ("frontend-" + lane + ".json"), front)
        save_arrays(old, new, front)
        shutil.copyfile(old / (lane + "-state-rows.json"), new / (lane + "-state-rows.json"))
        shutil.copytree(old / (lane + "-state-store"), new / (lane + "-state-store"))
        decoders = {}
        for name in (["forecast"] if lane == "forecast" else ["native-own", "native-reference"]):
            record = A["read"](old / ("decoder-" + name + ".json"))
            put(new / ("decoder-" + name + ".json"), record)
            save_arrays(old, new, record)
            decoders[name] = record
        heads, head_dim = qspec["num_attention_heads"], qspec["head_dim"]
        kv = qspec["num_key_value_heads"] if all_attended else heads
        descriptor = lambda shape: dict(shape=shape, stride=[int(np.prod(shape[1:])), int(np.prod(shape[2:])), shape[-1], 1], dtype="torch.bfloat16", device="cpu")
        for _ in range(qwen["actual_sdpa_argument_observations"]["calls"]):
            sdpa[lane].append(dict(case=case, lane=lane, ordinal=ordinal, query=descriptor([1, heads, 512, head_dim]),
                key=descriptor([1, kv, 512, head_dim]), value=descriptor([1, kv, 512, head_dim]),
                mask=None if all_attended else dict(shape=[1, 1, 512, 512], stride=[512 * 512, 512 * 512, 512, 1], dtype="torch.bool", device="cpu"),
                is_causal=all_attended, enable_gqa=all_attended, configured_backend="sdpa", selected_kernel="unprofiled"))
            ordinal += 1
        return dict(qwen=qwen, flux=dict(rows=lane + "-state-rows.json", transitions=20),
            vae=decoders["forecast"] if lane == "forecast" else dict(own=decoders["native-own"], reference=decoders["native-reference"]))

    for case in A["CASES"]:
        forecasts[case] = form_case(case, "forecast")
    put(raw / "sdpa-forecast.json", sdpa["forecast"])
    front = A["read"](raw / "cases/short/frontend-forecast.json")
    put(raw / "common-frontend.json", {key: dict(exact_equal=True, descriptor=dict(shape=front[key]["shape"], stride=front[key]["stride"],
        dtype=front[key]["logical_dtype"], device="cpu", sha256=front[key]["content_sha256"]))
        for key in ("raw_noise", "packed_noise", "img_ids", "txt_ids", "timesteps", "sigmas", "generator_before", "generator_after")})
    write_responses(raw, helper["A"])
    external = {path.relative_to(raw).as_posix(): dict(bytes=path.stat().st_size, sha256=A["sha"](path)) for path in raw.rglob("*") if path.is_file()}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    tx = ForecastTransaction(raw / "forecast-transaction", model_identity="flux2-klein-4b-raw-mask-pair-original",
        input_identity=hashlib.sha256(canonical(dict(cases=A["CASES"], seed=277))).hexdigest(),
        source_identity=hashlib.sha256(canonical({key: config[key + "_sha256"] for key in old_source["source_files"]})).hexdigest(),
        frontier=dict(kind="raw-input", cases=A["CASES"], seed=277, source_inventory=external), budgets={"max_bytes": 1 << 24})
    tx.predict("paired-raw-composition", lambda _: dict(predictions=dict(cases=forecasts,
        responses=dict(path="forecast-responses.json", sha256=A["sha"](raw / "forecast-responses.json"), recipe=A["RECIPE"]), external_files=external, native_targets_at_seal=0)))
    seal = tx.seal()

    def observed(_):
        for case in A["CASES"]:
            natives[case] = form_case(case, "native")
        return natives

    tx.execute("both_native_qwen_flux_vae_chains", observed)
    tx.close()
    put(raw / "sdpa-native.json", sdpa["native"])
    counts = dict(component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=4, functional_transitions=40,
        native_transitions=40, functional_decode_calls=2, native_vae_calls=4, native_qwen_calls=2, guard_refusals=0, native_module_entries=1)
    contract = dict(counts=counts, component_identity_before={name: old_source[name + "_parameter_identity_before"] for name in ("qwen", "flux", "vae")},
        component_identity_after={name: old_source[name + "_parameter_identity_after"] for name in ("qwen", "flux", "vae")},
        qwen_tied_alias_before=True, qwen_tied_alias_after=True, forecast_module_entries=0, case_dispatch=masks,
        placement_records=[{key: value for key, value in row.items() if key in {"component", "direction", "before_signature", "after_signature"}} for row in old_contract["placement_records"]])
    source = dict(schema="mdb-flux-raw-mask-pair-source-v1", job_id="job-synthetic-mask-pair", request_sha256=request["request_sha256"], worker_sha256=A["WORKER_SHA256"],
        helper_source_sha256=config["helper_source_sha256"], source_files=old_source["source_files"], tokenizer_files=old_source["tokenizer_files"],
        runtime_contract_sha256=A["sha"](raw / "runtime-contract.json"), forecast_seal=seal)
    summary = dict(counts=counts, job_id=source["job_id"], request_sha256=request["request_sha256"], cases={case: dict(forecast=forecasts[case], native=natives[case], mask=masks[case]) for case in A["CASES"]},
        forecast_seal=seal, transaction=verify_transaction(raw / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]))
    for name, value in (("source.json", source), ("summary.json", summary), ("native-contract.json", contract)):
        put(raw / name, value)
    events = []
    for lane in ("forecast", "native"):
        if lane == "native":
            events.extend([dict(kind="forecast_globally_sealed", seal_sha256=seal["seal_sha256"], native_module_entries=0, case_count=2),
                dict(kind="native_targets_released", forecast_seal=seal["seal_sha256"])])
        for component in ("qwen", "flux", "vae"):
            events.append(dict(kind="component_placement", component=component, direction="cuda"))
            for case in A["CASES"]:
                events.append(dict(kind="phase_entry", case=case, component=component, lane=lane, native_module_entries=0 if lane == "forecast" else 1))
                if component == "vae":
                    events.append(dict(kind="phase_completed", case=case, component=component, lane=lane))
            events.append(dict(kind="component_placement", component=component, direction="cpu"))
    (raw / "timeline.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    repin(raw)
    return raw, summary


def repin(root):
    path = root / "files.json"
    if path.exists():
        path.unlink()
    finalizer = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))
    summary = A["read"](root / "summary.json")
    finalizer["finalize"](root, job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return A["sha"](path)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return fixture(tmp_path_factory.mktemp("saved-mask-pair"))


def test_signed_full_hash_orientation_tails_and_nested_leaves():
    short = np.array([[-0., 1.], [3., -5.]], np.float32).T
    long = short.copy()
    long[0, 1] = -7
    expected = (long.astype(np.float64) - short.astype(np.float64)).astype("<f8")
    record = A["response_descriptor"](short, long)
    assert record["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    q = runpy.run_path(str(HERE / "audit_flux_functional_qwen_prefix.py"))
    result = A["signed_field"](dict(forecast_short=short) | {"forecast-short": short, "forecast-long": long,
        "native-short": short, "native-long": short}, record, q)
    assert result["full"]["signed_mean"] < 0 and result["full"]["changed_values"] == 1
    assert set(dict(A["tensor_leaves"]({"context": {"nested": {"scale": short}, "rope": [long]}}))) == {"/context/nested/scale", "/context/rope/0"}
    bad = {**record, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="sealed prospective"):
        A["signed_field"]({"forecast-short": short, "forecast-long": long, "native-short": short, "native-long": long}, bad, q)


def test_actual_frozen_frontend_nested_metadata_is_future_and_hostile_inclusion_refused(tmp_path):
    import torch
    frozen = runpy.run_path(str(HERE / "flux_raw_to_image.py"))
    frontend = dict(raw_noise=torch.zeros(1, 2, 2, 2), packed_noise=torch.zeros(1, 4, 2),
        img_ids=torch.zeros(1, 4, 4, dtype=torch.int64), txt_ids=torch.zeros(1, 3, 4, dtype=torch.int64),
        timesteps=torch.tensor([1000.]), sigmas=torch.tensor([1., 0.]),
        generator_before=torch.zeros(4, dtype=torch.uint8), generator_after=torch.ones(4, dtype=torch.uint8),
        mu=0., base_sigmas=[1.], scheduler_config={})
    record = frozen["_frontend_record"](tmp_path, "native", frontend)
    put(tmp_path / "frontend-native.json", record)
    assert A["read"](tmp_path / "frontend/native.json") == A["read"](tmp_path / "frontend-native.json")
    names = {"cases/short/" + path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}
    names.add("cases/short/frontend/forecast.json")
    forecast = {"cases/short/frontend/forecast.json"}
    assert A["native_member"]("cases/short/frontend/native.json")
    assert not A["native_member"]("cases/short/frontend/forecast.json")
    A["check_frontier"](names, forecast)
    with pytest.raises(ValueError, match="complete pretarget"):
        A["check_frontier"](names, forecast | {"cases/short/frontend/native.json"})
    with pytest.raises(ValueError, match="complete pretarget"):
        A["check_frontier"](names, set())


def test_complete_saved_pair_preserves_numerical_misses_and_responses(saved, tmp_path, monkeypatch):
    import torch
    root, summary = saved
    monkeypatch.setattr(torch.nn.Module, "_call_impl", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("neural module called")))
    result, _ = A["audit"](root, tmp_path / "audit", expected_manifest=A["sha"](root / "files.json"), expected_request=summary["request_sha256"])
    assert result["new_neural_forward_calls"] == result["new_learned_recurrence_calls"] == 0
    assert len(result["signed_responses"]["denoiser"]) == 21
    assert result["cases"]["long"]["decoder"]["fields"]["decoded"]["terms"]["whole_functional_minus_native"]["rms"] > 0
    assert result["sdpa"]["forecast"]["long"]["calls"] == 3
    assert result["cases"]["short"]["parser"]["mask"]["regime"] == "partially-padded"
    assert result["cases"]["long"]["parser"]["mask"]["regime"] == "all-attended"
    assert result["cases"]["long"]["parser"]["producer_uncropped_token_count_recorded"] == 2
    assert result["cases"]["long"]["parser"]["uncropped_token_count"] > 512
    assert result["cases"]["long"]["parser"]["producer_uncropped_count_matches_actual"] is False
    assert "job_id" not in result["config"]
    assert result["signed_responses"]["decoder"]["decoded"]["full"]["rms"] == 0
    assert result["coverage"]["independent_seeds"] == 1
    assert A["read"](tmp_path / "audit/catalog/provenance.json")["records"] == 4


@pytest.mark.parametrize("fault", ["raw-byte", "response-hash", "response-clock", "native-in-frontier", "own-parent", "argument-ledger", "false-alias"])
def test_hostile_saved_pair_refused(saved, tmp_path, fault):
    original, summary = saved
    root = tmp_path / "raw"
    shutil.copytree(original, root)
    if fault == "raw-byte":
        (root / "cases/short/qwen/forecast/h0.npy").write_bytes(b"broken")
    elif fault in {"response-hash", "response-clock"}:
        path = root / "forecast-responses.json"
        value = A["read"](path)
        if fault == "response-hash":
            value["qwen"]["0"]["sha256"] = "0" * 64
        else:
            value["denoiser"][0]["clock"]["step"] = 99
        put(path, value)
        repin(root)
    elif fault == "native-in-frontier":
        path = root / "forecast-transaction/forecasts/paired-raw-composition.json"
        value = A["read"](path)
        value["tampered"] = "native evidence"
        put(path, value)
        repin(root)
    elif fault == "own-parent":
        path = root / "cases/long/native-state-rows.json"
        value = A["read"](path)
        value[4]["parent"] = A["read"](root / "cases/short/forecast-state-rows.json")[3]["cut"]
        put(path, value)
        repin(root)
    elif fault == "argument-ledger":
        path = root / "sdpa-native.json"
        value = A["read"](path)
        value.pop()
        put(path, value)
        repin(root)
    elif fault == "false-alias":
        path = root / "native-contract.json"
        value = A["read"](path)
        value["qwen_tied_alias_after"] = False
        put(path, value)
        repin(root)
    with pytest.raises((ValueError, ForecastTransactionError)):
        A["audit"](root, tmp_path / "refused", expected_manifest=A["sha"](root / "files.json"), expected_request=summary["request_sha256"])


def test_argument_profile_miss_is_preserved_without_scientific_gate():
    geometry = dict(prefix=1, full=1, rows=4, heads=4, kv_heads=2, head_dim=4)
    rows = []
    for ordinal, case in enumerate(A["CASES"]):
        desc = dict(shape=[1, 4, 4, 4], stride=[64, 16, 4, 1], dtype="torch.bfloat16", device="cpu")
        rows.append(dict(case=case, lane="forecast", ordinal=ordinal, query=desc, key=desc, value=desc,
            mask=None, is_causal=False, enable_gqa=False, configured_backend="sdpa", selected_kernel="unprofiled"))
    report = A["sdpa_account"](rows, cases={case: dict(mask=dict(regime="all-attended")) for case in A["CASES"]}, lane="forecast", geometry=geometry)
    assert report["long"]["all_argument_profiles_match_declared_regime"] is False


def test_actual_saved_pair_worker_cli_catalog_and_original_six_pngs(saved, tmp_path):
    root, summary = saved
    config = dict(native_root=str(root), native_manifest_sha256=A["sha"](root / "files.json"), native_request_sha256=summary["request_sha256"], output_scratch=str(tmp_path / "outputs"))
    request = dict(config=config, source_hashes={name: A["sha"](HERE / name) for name in (*A["HELPERS"], "audit_flux_raw_mask_pair.py")})
    request["request_sha256"] = A["identity"](request)
    put(tmp_path / "request.json", request)
    put(tmp_path / "config.json", config)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SATURN_RESIDENT_WORKER": "1", "MRUN_JOB_ID": "job-tiny-mask-pair-audit",
        "SATURN_DEBUG_REQUEST": str(tmp_path / "request.json"), "SATURN_RUN_CONFIG": str(tmp_path / "config.json")}
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "audit_flux_raw_mask_pair.py")], cwd=HERE, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    assert len(marker) == 1 and marker[0]["request_sha256"] == request["request_sha256"]
    output = Path(marker[0]["output_root"])
    assert A["sha"](output / "files.json") == marker[0]["files_manifest_sha256"]
    assert len(list(output.glob("*.png"))) == 6 and not list(output.rglob("*.npy")) and not list(output.rglob("*.safetensors"))
    assert A["read"](output / "summary.json")["model_calls"] == 0
    assert A["read"](output / "catalog/provenance.json")["automatic_family_induction"] is False
