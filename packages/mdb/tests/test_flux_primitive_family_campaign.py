"""Finite campaign contracts and actual archive-reload primitive construction."""
from __future__ import annotations

import copy
import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mdb.computation_protocol import compile_protocol

HERE = Path(__file__).resolve().parent
C = runpy.run_path(str(HERE.parent / "qualification/flux_primitive_family_campaign.py"))
T = runpy.run_path(str(HERE / "test_flux_primitive_family_writer.py"))
W = T["W"]


def configuration():
    spec = T["plan"]("sdpa", C["MODEL_ID"])["specification"]
    spec["continuation"]["transitions"] = 101
    plan = compile_protocol(spec)
    context = dict(id="explicit-fresh-context", seed=17, prompts=dict(P="left", Q="right"))
    assay = dict(schema="fixture-assay", phase="development", fresh_contexts=[context], design_sha256="a"*64,
        protocol_sha256=plan["plan_sha256"], effect_rule=None)
    return C["make_config"](dict(family="sdpa", compiled_protocol=plan, contexts=[context], arms=[a["id"] for a in plan["actions"]],
        family_rule=assay, family_rule_file_sha256="b"*64,
        fresh_source=dict(output_root="/unopened/source", files_manifest_sha256="c"*64, job_id="job-fixture", request_sha256="d"*64),
        height=512, width=512, steps=4, batch_size=1, device="cuda", dtype="bfloat16"))


def test_definitions_and_structural_staging_do_not_open_model_or_array_paths(monkeypatch):
    code = "import runpy,sys;runpy.run_path(sys.argv[1]);assert not {'torch','diffusers','saturn_pub','numpy','mdb'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(HERE.parent / "qualification/flux_primitive_family_campaign.py")], check=True)
    config = configuration()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("structural validator opened a path"))
    counts = C["validate_config"](config)
    assert counts["functional_transitions"] == counts["native_target_transitions"] == 606
    assert counts["native_blocks"] == 542 and counts["acts"] == 10 and counts["native_vae"] == 6
    assert config["retry"] is False and config["nested_mrun"] is False


def test_source_context_quarantine_and_generated_action_tampering_are_rejected():
    config = configuration()
    config["contexts"] = copy.deepcopy(config["contexts"])
    config["contexts"][0]["seed"] += 1
    with pytest.raises(ValueError, match="context subset"):
        C["validate_config"](config)
    config = configuration()
    config["compiled_protocol"]["actions"][3]["bindings"]["route"] = "P"
    with pytest.raises(ValueError, match="compiled plan"):
        C["validate_config"](config)
    config = configuration()
    config["family_rule"].update(phase="fresh", effect_rule=dict(plan_sha256=config["compiled_protocol"]["plan_sha256"],
        fresh_contexts=[config["contexts"][0]["id"]], development_contexts=[config["contexts"][0]["id"]]))
    with pytest.raises(ValueError, match="overlaps"):
        C["validate_config"](config)


@pytest.mark.parametrize("family", ["sdpa", "swiglu"])
def test_actual_source_archive_reload_rehydrates_live_qualified_ports(family, tmp_path):
    adapter, results, compiled = T["captures"](family)
    reloaded = []
    for role, row in zip(("Q", "P"), results, strict=True):
        records = {}
        for name, value in row["ports"].items():
            path = tmp_path / (role + "-" + name + ".npy")
            raw = value.detach().contiguous().view(torch.uint16).numpy()
            np.save(path, raw, allow_pickle=False)
            records[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        record = dict(ports=records, profile=row["profile"], descriptors=row["descriptors"], retained_descriptors=row["retained_descriptors"])

        def load(array):
            assert hashlib.sha256(Path(array["path"]).read_bytes()).hexdigest() == array["sha256"]
            return torch.from_numpy(np.load(array["path"], allow_pickle=False).copy()).view(torch.bfloat16)

        reloaded.append(C["rehydrate_source_ports"](W, record, adapter, copy.deepcopy(row["state"]), family, load))
    q, p = reloaded
    for action in compiled["actions"]:
        expected = W["construct_endpoint"](results[0]["ports"], results[1]["ports"], action, family=family, plan=compiled)
        actual = W["construct_endpoint"](q, p, action, family=family, plan=compiled, native=True)
        assert T["bits"](expected["image"], actual["image"])
        assert T["bits"](expected["text"], actual["text"])
    port = "v_consumed" if family == "sdpa" else "mlp_input"
    assert list(q[port].stride()) != q.descriptors[port]["stride"]
