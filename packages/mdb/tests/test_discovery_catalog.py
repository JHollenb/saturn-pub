from __future__ import annotations

import pytest
import torch

from mdb.discovery_catalog import AliasRegistry, CatalogError, StructuralCatalog
from mdb.forward_recording import record_forward


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arbitrary = torch.nn.Sequential(torch.nn.LayerNorm(3),torch.nn.Linear(3,5),
                                             torch.nn.GELU(),torch.nn.Linear(5,3))
    def forward(self,x):
        return x+self.arbitrary(x)


def trace(model, value):
    return record_forward(model.eval(),(value,),model_identity="structural-test",
        execution={"backend":"cpu-fp32"},input_identity={"case":str(value.sum().item())}).trace


def test_generic_component_stage_and_residual_boundaries_are_structural():
    native = trace(Model(),torch.ones(2,4,3))
    catalog = StructuralCatalog(native)
    law = catalog.resolve("mlp_001")
    assert law["module"] == "arbitrary"
    assert catalog.target("law",law["entry"]) == law["exit"]
    update = next(s for s in catalog.sites if s["unit"] == "update")
    assert update["entry"] == law["exit"]
    assert catalog.target("update",law["exit"]) == update["exit"]
    assert catalog.target("op",0) == 1
    math = [s for s in catalog.sites if s["unit"] == "math"]
    assert [s["kind"] for s in math] == ["normalize","affine","gelu","affine"]
    assert math[-1]["exit"] == law["exit"]
    assert native["events"][math[-1]["exit"]-1]["operator"] in {"aten.view.default","aten._unsafe_view.default"}
    assert catalog.target("math",math[-1]["entry"]) == law["exit"]
    assert any(b["site_id"] == law["id"] and b["edge"] == "exit"
               for b in catalog.boundaries_at(law["exit"]))
    assert catalog.resolve("mlp_001:output")["value_id"] == law["outputs"][0]
    assert len(law["inputs"]) == 1
    assert catalog.resolve("module:arbitrary:input")["value_id"] == law["inputs"][0]
    assert len(law["parameters"]) == 6
    assert set(law["dependency_inputs"]) == set(law["inputs"]+law["parameters"])
    # A flatten at the beginning of a Linear belongs to that Linear; a
    # trailing reshape remains in its affine scope.
    for region in math:
        assert all(event["module_path"] == region["module"]
                   for event in native["events"][region["entry"]:region["exit"]])


def test_program_site_identity_is_independent_of_input_and_occurrence():
    model = Model()
    a,b = trace(model,torch.ones(2,3)),trace(model,torch.full((2,3),2.))
    first,second = StructuralCatalog(a),StructuralCatalog(b)
    assert a["fingerprint"] != b["fingerprint"]
    assert first.program_id == second.program_id
    assert first.resolve("mlp_001")["id"] == second.resolve("mlp_001")["id"]
    assert first.occurrences[0]["site_id"] == second.occurrences[0]["site_id"]
    assert first.occurrences[0]["id"] != second.occurrences[0]["id"]
    assert first.trace_fingerprint != second.trace_fingerprint
    changed = trace(model,torch.ones(4,3))
    assert StructuralCatalog(changed).program_id != first.program_id


def test_alias_rebinding_keeps_pinned_target_and_revision_without_semantic_gate():
    catalog = StructuralCatalog(trace(Model(),torch.ones(2,3)))
    aliases = AliasRegistry(catalog)
    first = aliases.bind("sum_of_thoughts","mlp_001",provenance={"author":"human"})
    pinned = aliases.pin("sum_of_thoughts")
    second = aliases.bind("sum_of_thoughts","gelu_001")
    assert second["revision"] == first["revision"]+1
    assert pinned["target_id"] == catalog.resolve("mlp_001")["id"]
    assert pinned["target_id"] != aliases.resolve("sum_of_thoughts")["target_id"]
    aliases.remove("sum_of_thoughts")
    third = aliases.bind("sum_of_thoughts","mlp_001")
    assert third["revision"] == 3
    with pytest.raises(CatalogError,match="unknown"):
        catalog.resolve("invented_semantic_component")
