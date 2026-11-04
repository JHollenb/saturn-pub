"""Saved-array regressions; no trained model, attention or native execution."""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1] / "qualification"
AUDIT = runpy.run_path(str(HERE / "analyze_flux_rescue_routes.py"))


def test_text_kv_selection_preserves_query_and_image_bank():
    shape, n = (1, 5, 2, 2), 2
    observed = {key: np.arange(20, dtype=np.float32).reshape(shape) + offset for key, offset in (("q", 0), ("k", 30), ("v", 60))}
    donor = {key: value + 200 for key, value in observed.items()}
    used = {key: value.copy() for key, value in observed.items()}
    for key in ("k", "v"):
        used[key][:, :n] = donor[key][:, :n]
    candidate = {"text_rows": n, "dispatch_calls": 1, "partition_controls": {
        "q_exact_recipient": True, "image_k_exact_recipient": True, "image_v_exact_recipient": True,
        "text_k_exact_selected": True, "text_v_exact_selected": True}}
    assert all(AUDIT["qkv_selection"](candidate, observed, used, donor).values())
    used["k"][:, n] += 1
    with pytest.raises(ValueError, match="undeclared operands"):
        AUDIT["qkv_selection"](candidate, observed, used, donor)


def test_wrong_text_donor_and_changed_query_are_not_valid_recipes():
    observed = {key: np.zeros((1, 3, 1, 1), dtype=np.float32) for key in ("q", "k", "v")}
    donor = {key: np.ones_like(value) for key, value in observed.items()}
    candidate = {"text_rows": 1, "dispatch_calls": 1, "partition_controls": {
        "q_exact_recipient": True, "image_k_exact_recipient": True, "image_v_exact_recipient": True,
        "text_k_exact_selected": True, "text_v_exact_selected": True}}
    with pytest.raises(ValueError, match="undeclared operands"):
        AUDIT["qkv_selection"](candidate, observed, observed, donor)
    used = {key: value.copy() for key, value in observed.items()}
    for key in ("k", "v"):
        used[key][:, :1] = 1
    used["q"][:, :1] = 1
    with pytest.raises(ValueError, match="undeclared operands"):
        AUDIT["qkv_selection"](candidate, observed, used, donor)


def test_signed_zero_recipe_and_scientific_mismatch_are_distinct():
    negative = np.array([-0.], dtype=np.float32)
    positive = np.array([0.], dtype=np.float32)
    assert not AUDIT["exact_bits"](negative, positive)
    report = AUDIT["distance"](np.array([4., -2.]), np.array([1., 3.]))
    assert report["base_l2"] == pytest.approx(34 ** .5)
    assert report["signed_mean"] == -1 and report["exact_base"] is False


def test_factorial_uses_signed_vector_cells_not_distance_algebra():
    z = np.array([10., 10.])
    i, t = np.array([12., 10.]), np.array([10., 7.])
    both = np.array([13., 5.])
    effect = AUDIT["factorial_interaction"](z, i, t, both)
    assert effect["l2"] == pytest.approx(5 ** .5)
    assert effect["signed_mean"] == -.5 and effect["max_abs"] == 2
    assert AUDIT["factorial_interaction"](z, i, t, i + t - z)["l2"] == 0


def test_bf16_write_rounding_order_and_saved_write_tamper():
    q = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    r = np.array([64., .125], dtype=np.float32)
    a, m = np.array([.3, 4.], dtype=np.float32), np.array([-.6, .01], dtype=np.float32)
    ga, gm = np.array([.5, .25], dtype=np.float32), np.array([.25, .5], dtype=np.float32)
    aw, mw = q(ga * a), q(gm * m)
    u = q(r + aw)
    ports = {"residual": r, "gate_attention": ga, "attention_used": a, "gate_mlp": gm,
             "mlp_used": m, "attention_write": aw, "mlp_write": mw, "intermediate": u, "image": q(u + mw)}
    assert AUDIT["sequential_equation"](ports, q)["image_exact"]
    ports["mlp_write"] = ports["mlp_write"] + 1
    with pytest.raises(ValueError, match="writer equation"):
        AUDIT["sequential_equation"](ports, q)


def test_array_hash_and_dtype_are_mechanics(tmp_path):
    e = AUDIT["Evidence"].__new__(AUDIT["Evidence"])
    e.data = tmp_path
    e.h = {"file_path": lambda root, name: root / name,
           "sha": lambda value: hashlib.sha256(value).hexdigest()}
    e.typed = {"matches": lambda value, descriptor, bf16: True}
    path = tmp_path / "value.npy"
    np.save(path, np.array([1.], dtype=np.float32), allow_pickle=False)
    record = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "stored_dtype": "float32",
              "stored_shape": [1], "descriptor": {"dtype": "torch.bfloat16"}}
    assert e.array(record).tolist() == [1.]
    np.save(path, np.array([1.], dtype=np.float64), allow_pickle=False)
    with pytest.raises(ValueError, match="hash"):
        e.array(record)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="dtype/shape"):
        e.array(record)


def test_receipt_walk_rejects_missing_and_cyclic_suffix():
    e = AUDIT["Evidence"].__new__(AUDIT["Evidence"])
    e.edges = {"parent": {"fingerprint": "r0", "steps": 1, "transitions": [{}], "result": "final"}}
    assert e.walk("parent", "final")["native_steps"] == 1
    with pytest.raises(ValueError, match="missing native"):
        e.walk("wrong-parent", "final")
    e.edges["parent"]["result"] = "parent"
    with pytest.raises(ValueError, match="cyclic"):
        e.walk("parent", "final")


def test_partial_full_state_retention_uses_receipt_points_without_inventing_cuts():
    before = {"logical_step": 3, "phase": "single", "next_operation": "single_block.0"}
    middle = {"logical_step": 3, "phase": "single", "next_operation": "single_block.1"}
    after = {"logical_step": 3, "phase": "single", "next_operation": "single_block.2"}
    receipts = {}
    for rid, parent, result, p0, p1 in (("r0", "parent", "not-retained", before, middle),
                                       ("r1", "not-retained", "end", middle, after)):
        receipts[rid] = {"operation": "continue", "parent": parent, "result": result, "steps": 1,
            "execution_point": p1, "transitions": [{"before": p0, "after": p1,
                "native_operation": {"operation": p0["next_operation"], "footprint": "declared",
                    "reads": ["image", "text", "context"], "writes": ["image", "text"], "invalidates": []}}]}
    points = AUDIT["receipt_points"](receipts.values())
    stored = {"parent": {"conditioning": "ownC"}, "end": {"conditioning": "ownC"}}
    path = {"visited_cuts": ["parent", "not-retained", "end"], "receipts": ["r0", "r1"]}
    report = AUDIT["conditioning_coverage"](path, receipts, stored, "ownC", points.__getitem__)
    assert report["retained_full_states_checked"] == 2
    assert report["intermediate_full_states_not_retained"] == 1
    assert "not-retained" not in stored
    receipts["r1"]["transitions"][0]["native_operation"]["reads"].append("conditioning")
    with pytest.raises(ValueError, match="conditioning read/write"):
        AUDIT["conditioning_coverage"](path, receipts, stored, "ownC", points.__getitem__)


def test_publisher_keeps_nine_rows_under_unresolved_controls(tmp_path, monkeypatch):
    module = runpy.run_path(str(HERE / "publish_flux_rescue_routes.py"))
    real_run_path = runpy.run_path
    common = real_run_path(str(HERE / "publish_flux_carriers.py"))
    root, output = tmp_path / "run", tmp_path / "published"
    data, tx = root / "extracted/flux-rescue-routes-result", root / "tx"
    data.mkdir(parents=True)
    tx.mkdir()
    (data / "rows").mkdir()
    (data / "typed-store/cuts").mkdir(parents=True)
    write = lambda path, value: path.write_text(json.dumps(value))
    for name in ("source.json", "prefixes.json", "candidates.json"):
        write(data / name, {})
    write(tx / "seal.json", {})
    write(tx / "observations.json", {})
    (tx / "observations").mkdir()
    write(tx / "observations/native.json", {"value": {}})
    names = ["zero", "own", "sham", "late.image", "late.text", "late.both", "early.kv.image", "early.kv.both", "early.coherent"]
    rows = []
    for i, name in enumerate(names):
        image = data / (name + ".png")
        Image.new("RGB", (512, 512), (i * 20, 17, 9)).save(image)
        latent = data / (name + ".npy")
        np.save(latent, np.array([i], dtype=np.float32), allow_pickle=False)
        for cut in ("parent", "final"):
            write(data / "typed-store/cuts" / (cut + ".json"), {"fingerprint": cut})
        row = {"id": name, "parent": "parent", "final_cut": "final", "acts": [],
               "image": {"path": image.name, "sha256": common["sha"](image)},
               "final_latent": {"path": latent.name, "sha256": common["sha"](latent)}}
        write(data / "rows" / (name + ".json"), row)
        rows.append(row)
    observed = {"rows": rows}
    run = {"data": data, "plan": {"branches": [{"id": n} for n in names]}, "transaction": tx,
           "source": {"archive_sha256": "pinned"}, "tree": type("Tree", (), {"decode": lambda self, _: observed})()}
    helper = HERE / "summarize_context_window.py"
    metric = {"base_l2": 2.5, "exact_base": False}
    audit = {"mechanics_status": "valid", "source": run["source"], "auditor_source_sha256": common["sha"](HERE / "analyze_flux_rescue_routes.py"),
             "audit_helpers": {helper.name: common["sha"](helper)}, "instrument_status": "replay-unresolved",
             "counts": {"images": 9, "vae_calls": 9}, "comparisons": {n: {b: {"latent": metric, "rgb": metric} for b in ("zero", "own")} for n in names},
             "native_paths": {n: {} for n in names}, "replay_qualification": {"controls": {"sham": False}},
             "control_observations": {"sham": False}, "candidate_checks": {}, "paired_source_checks": {}, "sealed_hypotheses": {}, "limits": []}
    audit["late_factorial_interactions"] = {key: {"l2": 1.} for key in ("Z4_latent", "final_RGB")}
    audit_path = root / "audit.json"
    write(audit_path, audit)
    monkeypatch.setattr(module["runpy"], "run_path", lambda path: {"source_run": lambda *a, **k: run} if Path(path).name == helper.name else real_run_path(path))
    result = module["publish"](root, audit_path, output)
    assert len(result["rows"]) == 9 and result["instrument_status"] == "replay-unresolved"
    assert result["control_observations"]["sham"] is False
    with Image.open(output / "rescue-routes-finals.png") as sheet:
        for row in rows:
            with Image.open(data / row["image"]["path"]) as original:
                assert sheet.crop(result["figures"]["finals"]["tiles"][row["id"]]["image_box"]).tobytes() == original.tobytes()
    with pytest.raises(ValueError, match="already exists"):
        module["publish"](root, audit_path, output)
