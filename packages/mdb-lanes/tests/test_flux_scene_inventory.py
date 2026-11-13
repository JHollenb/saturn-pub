"""Mechanics of the scene instrument; no model or semantic capability claims."""
import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

HELPER = runpy.run_path(str(Path(__file__).parents[1]/"qualification"/"flux_scene_inventory.py"))


def test_factorial_recovers_known_conjunction_and_reconstructs():
    values = {(a,b,c): np.array([1+2*(2*a-1)+3*(2*b-1)*(2*c-1)])
              for a in (0,1) for b in (0,1) for c in (0,1)}
    report = HELPER["factorial_diagnostics"](values)
    assert report["max_reconstruction_abs"] == 0
    assert report["coefficients"]["shapeA"]["rms"] == 2
    assert report["coefficients"]["colorB*relation"]["rms"] == 3
    assert report["coefficients"]["shapeA*colorB"]["rms"] == 0


def test_pixel_inventory_observes_objects_without_prompt_or_shape_label():
    image = np.full((40,60,3), 255, dtype=np.uint8)
    image[8:25,4:19] = [240,20,10]
    image[12:32,38:55] = [10,20,240]
    report, _ = HELPER["pixel_inventory"](image)
    assert [x["color"] for x in report["components"]] == ["red", "blue"]
    assert report["relations"][0]["predicate"] == "centroid-left-of"
    assert all(x["shape_semantics"] == "unknown" for x in report["components"])
    assert HELPER["pixel_inventory"](np.full_like(image,255))[0]["components"] == []


def test_native_coordinates_control_support_instead_of_row_number():
    ids = np.array([[[0,h,w,0] for h in range(4) for w in range(4)]])
    base, source = np.zeros((1,16,2)), np.zeros((1,16,2))
    source[:,[0,4,8,12]] = 2
    support = HELPER["propose_support"](base, source, ids)
    assert support["top"] == [0,4,8,12]
    assert support["shifted"] == [2,6,10,14]
    assert support["top_energy_fraction"] == 1
    with pytest.raises(ValueError):
        HELPER["propose_support"](base, source, np.zeros_like(ids))


def test_experiment_is_complete_factorial_with_explicit_wrong_address_controls():
    plan = HELPER["make_plan"]()
    assert len(plan["histories"]) == 8
    assert len({tuple(h["factors"]) for h in plan["histories"]}) == 8
    assert len(plan["rows"]) == 18
    assert sum(r.get("mode") == "shifted" for r in plan["rows"]) == 3


def test_checkpoint_identity_preserves_existing_compact_encoding():
    worker = runpy.run_path(str(Path(__file__).parents[1]/"qualification"/"flux_scene_inventory_campaign.py"))
    assert worker["identity"]({"b":2,"a":1}) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()


def test_coefficient_scale_agrees_with_public_mdb_factorial_convention():
    from mdb_lanes.forward_algebra import factorial_contrast

    observations = {(a,b,c): 2*(2*a-1)+3*(2*b-1)*(2*c-1)
                    for a in (0,1) for b in (0,1) for c in (0,1)}
    diagnostic = HELPER["factorial_diagnostics"]({k:np.array([v]) for k,v in observations.items()})
    for selected in (("shapeA",),("colorB","relation")):
        reference = factorial_contrast(observations,factors=("shapeA","colorB","relation"),contrast_factors=selected)
        assert diagnostic["coefficients"]["*".join(selected)]["rms"] == abs(reference["value"])/(2**len(selected))


def test_state_coalition_panel_covers_all_seven_nonempty_coalitions_per_factor():
    worker = runpy.run_path(str(Path(__file__).parents[1]/"qualification"/"flux_scene_state_closure_campaign.py"))
    rows = worker["make_rows"]()
    assert len(rows)==25
    for factor in ("shapeA","colorB","relation"):
        bits = {tuple(r["bits"]) for r in rows if r.get("factor")==factor}
        assert len(bits)==7 and (0,0,0) not in bits
        assert (1,0,0) in bits and (0,0,1) in bits and (1,1,1) in bits
