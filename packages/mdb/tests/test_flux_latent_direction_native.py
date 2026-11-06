"""Installed tiny six-future native cross; no real checkpoint or model jobs."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from PIL import Image
from safetensors.torch import load_file
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter

HERE = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    data, frozen = (os.environ.get(key) for key in ("MDB_FACTORIAL_TINY_DATA", "MDB_FACTORIAL_FROZEN_PAYLOAD"))
    if not data or not frozen:
        pytest.skip("requires explicit saved tiny source and immutable qualified helper payload")
    root = tmp_path_factory.mktemp("native-direction")
    for path in Path(frozen).glob("*.py"):
        shutil.copyfile(path, root / path.name)
    for name in ("flux_native_corner.py", "flux_latent_direction_native.py"):
        shutil.copyfile(HERE / "qualification" / name, root / name)
    m = runpy.run_path(str(root / "flux_latent_direction_native.py"))
    torch.set_num_threads(1)
    data = Path(data)
    recipes = json.loads((data / "parameter-artifacts.json").read_text())
    model = Flux2Transformer2DModel(in_channels=4, num_layers=5, num_single_layers=20, attention_head_dim=8,
        num_attention_heads=1, joint_attention_dim=6, timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(torch.bfloat16)
    model.load_state_dict(load_file(Path(recipes["defaults"]["checkpoint_root"]) / "transformer/diffusion_pytorch_model.safetensors"))
    adapter = FluxFinalLatentClosureAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    parents, references, native = {}, {}, {}
    store = LocalStore(data / "typed-store")
    for cell in m["CELLS"]:
        native[cell] = [json.loads(line) for line in (data / "native" / (cell + ".jsonl")).read_text().splitlines()]
        parents[cell] = store.load(native[cell][27]["capture"]["cut"])
        references[cell] = {"transitions": native[cell], "image": json.loads((data / "rows" / (cell + ".native.json")).read_text())["image"]}
    math_data = root / "CPU-math-fixture"
    math_data.mkdir()
    def archive(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        descriptor = m["support"]()["support"]()["descriptor"](value)
        path = math_data / "arrays" / (m["identity"](descriptor) + ".npy")
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            np.save(path, value)
        return {"path": path.relative_to(math_data).as_posix(), "sha256": m["sha"](path), "dtype": "float32",
            "shape": list(value.shape), "descriptor": descriptor}
    inputs = {"schema": "mdb-latent-direction-inputs-v1", "source": {"parent_job_id": "fixture-parent", "parent_request_sha256": "fixture-request"},
        "receivers": {cell: {"cut": native[cell][27]["capture"]} for cell in m["CELLS"]}, "branches": {}}
    programs = {}
    for branch in m["BRANCHES"]:
        cell = branch.split(".")[0]
        direction = None if branch.endswith(".sham") else branch.split(".d-")[1]
        value = parents[cell].payload["latent"].clone()
        if direction is not None:
            value.flatten()[0] += .0625 if direction == m["CELLS"][0] else -.0625
        spec = {"receiver": cell, "direction": direction, "kind": "sham" if direction is None else "direction",
            "input_latent": archive(value.float().numpy()), "requested_rms": 0.001, "effective_delta": {"scope": "synthetic input only"}}
        inputs["branches"][branch] = spec
        # Saved native reference values stand in for CPU forecasts in this
        # consumer-order fixture. No numerical accuracy claim or math occurs.
        terminal = {"descriptor": None, "archive": archive(parents[cell].payload["latent"].float().numpy())}
        terminal["descriptor"] = terminal["archive"]["descriptor"]
        transitions = [{"logical_step": 1 + i // 28, "local_index": i % 28, "kind": "fixture", "index": i,
            "inputs": {}, "outputs": {"latent": terminal} if i == 83 else {}} for i in range(84)]
        program = {"schema": "mdb-latent-direction-program-v1", "branch_id": branch, "receiver": cell, "direction": direction,
            "origin_cut": parents[cell].fingerprint, "inputs": spec, "chain_length": 84, "transitions": transitions, "terminal_latent": terminal}
        rel = "programs/" + branch + ".json"
        m["save"](math_data / rel, program)
        programs[branch] = {"path": rel, "sha256": m["sha"](math_data / rel)}
    m["save"](math_data / "inputs.json", inputs)
    m["save"](math_data / "expectations.json", {"scope": "synthetic mechanics only, no scientific quality gate"})
    cohort = {"schema": "mdb-latent-direction-cohort-seal-v1", "programs": programs,
        "inputs_sha256": m["sha"](math_data / "inputs.json"), "expectations_sha256": m["sha"](math_data / "expectations.json"), "math_transitions": 504, "native_forwards": 0}
    m["save"](math_data / "cohort-seal.json", {**cohort, "fingerprint": m["identity"](cohort)})
    m["save"](math_data / "summary.json", {**cohort, "cohort_seal_sha256": m["sha"](math_data / "cohort-seal.json")})
    config = m["make_config"]()
    config.update(parent_job_id="fixture-parent", parent_request_sha256="fixture-request")
    counts = dict.fromkeys(m["COUNTS"], 0)
    counts["physical_batch"] = 1
    vae, events, decoded = torch.nn.Identity(), [], []
    def decode(session, **kwargs):
        decoded.append(dict(counts))
        value = vae(session.read("latent"))
        digest = hashlib.sha256(value.float().numpy().tobytes()).digest()
        return Image.fromarray(np.full((2, 2, 3), list(digest[:3]), dtype=np.uint8))
    adapter.decode = decode
    output = root / "result"
    with torch.inference_mode():
        summary = m["run"](adapter=adapter, parents=parents, math_data=math_data, references=references, output=output,
            config=config, request={"job_id": "tiny-cross", "request_sha256": "fixture", "source_hashes": {"worker": m["sha"](root / "flux_latent_direction_native.py")}},
            source={"scope": "tiny native model and deterministic2x2 consumer, not real VAE or mathematical forecasts"}, counts=counts,
            parameter_manifest=recipes, retrieval_source=data / "retrieve_weights.py", guard_components=(model, vae), progress=events.append, production_profile=False)
    return {"m": m, "summary": summary, "output": output, "parents": parents, "math_data": math_data, "inputs": inputs,
        "decoded": decoded, "events": events, "config": config, "root": root}


def test_all_six_math_images_native0_before_504_native_and_12_acts(completed):
    x, m = completed, completed["m"]
    assert x["summary"]["counts"] == {**m["COUNTS"], "model_loads": 0}
    assert [v["adapter_steps"] for v in x["decoded"]] == [0] * 6 + [84, 168, 252, 336, 420, 504]
    kinds = [event["kind"] for event in x["events"]]
    assert kinds.count("receiver_sealed") == 2 and kinds.index("cohort_sealed") < kinds.index("predicted_image")
    assert kinds.index("all_predicted_images_complete") < kinds.index("native_transition")
    assert x["summary"]["target_calls"] == m["support"]()["target_counts"](18)
    assert all(v == m["support"]()["target_counts"](9) for v in x["summary"]["target_calls_per_condition"].values())


def test_complete_ledger_and_exact42_retained_exits(completed):
    x, m = completed, completed["m"]
    full = 0
    for branch in m["BRANCHES"]:
        rows = [json.loads(line) for line in (x["output"] / "native" / (branch + ".jsonl")).read_text().splitlines()]
        assert len(rows) == 84
        for i, row in enumerate(rows):
            assert (row["logical_step"], row["local_index"]) == (1 + i // 28, i % 28)
            keep = m["support"]()["retained"](1 + i // 28, i % 28)
            assert row["capture"]["retained_full_cut"] == keep
            assert all((v["archive"] is not None) == keep for v in row["outputs"].values())
            full += keep
    assert full == 42
    ledger = [json.loads(line) for line in (x["output"] / "receipt-emissions.jsonl").read_text().splitlines()]
    assert len(ledger) == 516 and [row["sequence"] for row in ledger] == list(range(516))
    assert sum(row["kind"] == "render_acts" for row in ledger) == sum(row["kind"] == "native_branch_acts" for row in ledger) == 6


def test_qualified_render_derivations_and_original_receiver_layout(completed):
    x, m = completed, completed["m"]
    for name in m["BRANCHES"]:
        row = x["summary"]["rows"][name]
        parent = x["parents"][row["receiver"]]
        derived = json.loads((x["output"] / row["predicted"]["derivation"]).read_text())
        assert derived["qualified_derivation"]["native_steps"] == derived["qualified_derivation"]["acts_applied"] == 0
        assert derived["source_N1"] == row["source_parent"] == parent.fingerprint
        assert row["predicted"]["native_steps_before_image"] == 0 and set(row["predicted"]["target_calls_after_image"].values()) == {0}
        entry = LocalStore(x["output"] / "typed-store").load(row["entry"]["cut"])
        assert entry.payload["latent"].stride() == parent.payload["latent"].stride()
        assert describe({k: v for k, v in entry.payload.items() if k != "latent"}) == describe({k: v for k, v in parent.payload.items() if k != "latent"})
        assert row["act"]["writes"] == ["latent"] and not row["act"]["invalidates"]


def test_sealed_math_alias_and_clock_tampering_refused(completed, tmp_path):
    x, m = completed, completed["m"]
    target = tmp_path / "math"
    shutil.copytree(x["math_data"], target)
    branch = m["BRANCHES"][0]
    path = target / "programs" / (branch + ".json")
    p = json.loads(path.read_text())
    p["transitions"][0]["logical_step"] = 2
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="program bytes"):
        m["load_math"](target, parent_job_id="fixture-parent", parent_request_sha256="fixture-request")
    ref = copy.deepcopy(x["inputs"]["branches"][branch]["input_latent"])
    ref["descriptor"]["shape"] = [1]
    with pytest.raises(ValueError, match="descriptor"):
        m["read_array"](x["math_data"], ref)


def test_baseline_diagnostics_and_recipe_only_storage(completed):
    x, m = completed, completed["m"]
    for name, row in x["summary"]["rows"].items():
        assert bool(row["native_replay_diagnostics"]) == name.endswith(".sham")
        if name.endswith(".sham"):
            assert all(all(v.values()) for v in row["native_replay_diagnostics"])
    recipe = json.loads((x["output"] / "parameter-artifacts.json").read_text())
    assert recipe["artifacts"] and all(not (x["output"] / r["path"]).exists() for r in recipe["artifacts"])
    bad = copy.deepcopy(x["config"])
    bad["counts"]["acts"] = 6
    with pytest.raises(ValueError, match="contract"):
        m["validate_config"](bad)


def test_completed_worker_export_uses_real_pinned_finalizer_and_serializes(completed, capsys):
    """Exercise the production export tail which run()-only checks omitted."""
    x, m = completed, completed["m"]
    finalizer = m["support"]()["support"]()["prior"]()["step"]()["finalize_manifest"]
    assert Path(inspect.getfile(finalizer)).name == "flux_step_boundary_campaign.py"
    assert inspect.signature(finalizer).parameters["job_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(finalizer).parameters["request_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (x["output"] / "summary.json").is_file()
    assert len(list(x["output"].glob("*.png"))) == 12
    for cell in m["CELLS"]:
        assert json.loads((x["output"] / "transactions" / cell / "verification.json").read_text())["verified"]
        assert (x["output"] / "transactions" / cell / "report.json").is_file()
    manifest = m["finalize_worker_output"](x["output"], summary=x["summary"],
        request={"job_id": "tiny-cross", "request_sha256": "fixture"}, config=x["config"])
    emitted = capsys.readouterr().out.strip()
    assert emitted.startswith("MDB_LATENT_DIRECTION_NATIVE_RESULT=")
    body = json.loads(emitted.split("=", 1)[1])
    assert body == {"summary": x["summary"], "manifest": manifest}
    remote = json.loads((x["output"] / "remote-manifest.json").read_text())
    assert remote["schema"] == "mdb-step-boundary-remote-manifest-v1"
    assert remote["job_id"] == "tiny-cross" and remote["request_sha256"] == "fixture"
    assert manifest["manifest_sha256"] == m["sha"](x["output"] / "remote-manifest.json")
    assert "summary.json" in remote["files"] and "files.json" in remote["files"]
    assert len([name for name in remote["files"] if name.endswith(".png")]) == 12
    for name, record in remote["files"].items():
        assert record == {"sha256": m["sha"](x["output"] / name), "bytes": (x["output"] / name).stat().st_size, "filetype": "regular"}
    with pytest.raises(ValueError, match="immutable final manifest"):
        m["finalize_worker_output"](x["output"], summary=x["summary"],
            request={"job_id": "tiny-cross", "request_sha256": "fixture"}, config=x["config"])
