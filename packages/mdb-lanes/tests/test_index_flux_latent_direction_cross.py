"""Public equation/atlas tests using saved synthetic fields, never a model."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdb_lanes.computation_atlas import ComputationAtlas
from mdb_lanes.equation_comparison import EquationCatalog

SOURCE = Path(__file__).parents[1] / "qualification/index_flux_latent_direction_cross.py"


@pytest.fixture
def module():
    return runpy.run_path(str(SOURCE))


@pytest.fixture
def fixture(module, tmp_path):
    artifact = tmp_path / "synthetic.json"
    artifact.write_text('{"scope":"synthetic saved arrays only"}\n')
    provenance = {"run_id": "synthetic-job-not-a-scheduler-result", "artifacts": [{"path": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
    contexts, arrays, occurrences = {}, {}, {}
    for index, branch in enumerate(module["BRANCHES"]):
        receiver = branch.split(".")[0]
        p = np.array([[1., -2., 3., -4.]], np.float32)
        n = p.copy() if index == 0 else p + np.array([[.25, -.5, -.75, 0]], np.float32) * index
        delta = p.astype(np.float64) - n
        latent = {"signed_mean": float(delta.mean()), "rms": float(np.sqrt(np.mean(delta * delta))),
            "max_abs": float(np.abs(delta).max()), "relative_rms": float(np.sqrt(np.mean(delta * delta)) / np.sqrt(np.mean(n.astype(float) ** 2)))}
        contexts[branch] = {"branch": branch, "receiver": receiver, "model_identity": {"checkpoint": "synthetic"},
            "source_identity": {"fixture": "1" * 64}, "execution": {"backend": "synthetic-NumPy"}, "horizon": 84,
            "input_cut": str(index) * 64, "receiving_cut": "a" * 64 if receiver == "c1-o0" else "b" * 64,
            "clock": {"step": 1, "phase": "project", "cursor": 0}, "installed_input": {"sha256": str(index + 1) * 64},
            "conditioning": {"receiver": receiver}, "prompt_context": {"seed": 127, "id": receiver}}
        arrays[branch] = (p, n)
        occurrences[branch] = {"endpoint": {"latent": latent, "RGB": {"signed_mean": -.25, "rms": 1.25}},
            "PNG_sources": {"predicted": {"path": "synthetic-P.png", "sha256": "c" * 64},
                "native": {"path": "synthetic-N.png", "sha256": "d" * 64}},
            "response_accounts": [{"cosine": -.75, "relative_rms": None}], "expectations": [{"holds": False}],
            "native_changed_minus_sham": None if branch.endswith(".sham") else
                {"latent": {"signed_mean": -.125}, "RGB": {"signed_mean": .75}}}
    return contexts, arrays, occurrences, provenance


def test_six_unique_occurrences_exact_queries_and_abstentions(module, fixture, tmp_path):
    contexts, arrays, occurrences, provenance = fixture
    output = tmp_path / "index"
    result = module["curate"](output, contexts=contexts, arrays=arrays.__getitem__, occurrences=occurrences, provenance=provenance)
    assert result["curated_records"] == result["exact_context_matches"] == 6
    assert len(set(result["atlas_observation_ids"])) == 6
    assert result["distinct_laws"] == result["distinct_routes"] == 1
    assert result["family_candidates_created"] == 0 and result["rule_induction"] == "not performed"
    assert all(row["missing_input_applicability"] == "unknown" and row["changed_clock_matches"] == 0
        and row["changed_receiver_matches"] == 0 and row["changed_installed_input_matches"] == 0 for row in result["queries"])
    assert json.loads((output / "result.json").read_text())["records"] == result["records"]
    with pytest.raises(ValueError, match="fresh"):
        module["curate"](output, contexts=contexts, arrays=arrays.__getitem__, occurrences=occurrences, provenance=provenance)


def test_equation_orientation_signed_means_and_VAE_consumer_separate(module, fixture, tmp_path):
    contexts, arrays, occurrences, provenance = fixture
    output = tmp_path / "index"
    result = module["curate"](output, contexts=contexts, arrays=arrays.__getitem__, occurrences=occurrences, provenance=provenance)
    catalog, atlas = EquationCatalog(output / "equations"), ComputationAtlas(output / "atlas")
    for branch, record, identifier in zip(module["BRANCHES"], result["records"], result["atlas_observation_ids"]):
        equation = catalog.show(record["equation_id"])
        assert set(equation["report"]["fields"]) == {"terminal_latent"}
        assert equation["report"]["residual_direction"] == "observed-minus-predicted"
        assert equation["report"]["fields"]["terminal_latent"]["residual"]["signed_mean"] == -occurrences[branch]["endpoint"]["latent"]["signed_mean"]
        observation = atlas.show(identifier)
        assert observation["signed_effects"][0]["value"] == occurrences[branch]["endpoint"]["latent"]["signed_mean"]
        assert observation["signed_effects"][0]["units"] == "latent-value"
        assert observation["signed_effects"][1]["units"] == "uint8-channel-value"
        assert "not a mathematical RGB equation" in observation["signed_effects"][1]["scope"]
        assert observation["equation"]["evidence"]["occurrence"]["expectations"] == [{"holds": False}]
        if not branch.endswith(".sham"):
            assert observation["signed_effects"][2]["contrast_id"] == "native-changed-minus-same-receiver-sham"
            assert observation["signed_effects"][2]["value"] == -.125


def test_nonzero_discrepancies_counterexamples_preserved_no_quality_gate(module, fixture, tmp_path):
    contexts, arrays, occurrences, provenance = fixture
    result = module["curate"](tmp_path / "index", contexts=contexts, arrays=arrays.__getitem__, occurrences=occurrences, provenance=provenance)
    atlas = ComputationAtlas(tmp_path / "index/atlas")
    rows = [atlas.show(identifier) for identifier in result["atlas_observation_ids"]]
    assert rows[0]["counterexamples"] == []
    assert all(row["counterexamples"][0]["status"] == "observed-counterexample" for row in rows[1:])
    assert all(row["rule_status"] == "observation; no validated rule" for row in rows)


def test_changed_saved_account_is_integrity_error(module, fixture, tmp_path):
    contexts, arrays, occurrences, provenance = fixture
    occurrences = copy.deepcopy(occurrences)
    occurrences[module["BRANCHES"][0]]["endpoint"]["latent"]["signed_mean"] = 99.
    with pytest.raises(ValueError, match="accepted endpoint"):
        module["curate"](tmp_path / "index", contexts=contexts, arrays=arrays.__getitem__, occurrences=occurrences, provenance=provenance)


def test_import_needs_no_model_framework(tmp_path):
    command = [sys.executable, "-I", "-c", "import runpy,sys;runpy.run_path(" + repr(str(SOURCE)) +
        ");assert 'torch' not in sys.modules;assert 'diffusers' not in sys.modules"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
