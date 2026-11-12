"""Prefix cardinality and input-only lineage plumbing without a model load."""
from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

S = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_primitive_family_sources.py"))


def rows():
    return [{"id": f"context-{i}", "seed": 7 if i < 2 else i,
             "prompts": {"P": "a left of b", "Q": "a right of b"}} for i in range(4)]


def test_explicit_contexts_drive_every_physical_count():
    data = rows()
    S["configure"]({"contexts": data})
    data[0]["seed"] = 999
    assert S["contexts"]()[0]["seed"] == 7
    assert S["counts"]() == {
        "contexts": 4, "prompt_encodes": 8, "noise_draws": 3, "native_transitions": 80,
        "native_blocks": 72, "selected_block_calls": 0, "decoder_calls": 0, "images": 0,
        "input_cuts": 8, "pipeline_loads": 1, "encoder_cuda_placements": 1,
        "transformer_cuda_placements": 1,
    }


def test_legacy_six_context_counts_and_new_subsets_remain_separate():
    old = rows() + [{"id": "extra-1", "seed": 91, "prompts": {"P": "P", "Q": "Q"}},
                    {"id": "extra-2", "seed": 91, "prompts": {"P": "P", "Q": "Q"}}]
    assert S["expected_counts"](old)["native_transitions"] == 120
    S["configure"]({"contexts": [copy.deepcopy(old[0])]})
    assert S["counts"]()["native_transitions"] == 20


def test_selected_writer_or_wrong_clock_is_never_a_prefix_operation():
    valid = {"logical_step": 0, "phase": "project", "index": None, "edge": "before"}
    assert S["before_source"](valid) == valid
    for change in ({"logical_step": 1}, {"phase": "single", "index": 4}, {"phase": "readout"},
                   {"edge": "after"}, {"phase": "project", "index": 0}):
        with pytest.raises(ValueError):
            S["before_source"]({**valid, **change})


def test_context_aliases_and_unbounded_source_panels_refuse():
    for data in ([], rows()*5, [rows()[0], rows()[0]]):
        with pytest.raises(ValueError):
            S["configure"]({"contexts": data})
