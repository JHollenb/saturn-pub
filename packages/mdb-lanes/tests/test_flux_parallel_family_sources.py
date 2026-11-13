"""Model-free guard checks for the native input-only preparation frontier."""
import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def source():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/flux_parallel_family_sources.py"))


@pytest.mark.parametrize("point", [
    dict(logical_step=0, phase="single", index=4, edge="before"),
    dict(logical_step=1, phase="project", index=0, edge="before"),
    dict(logical_step=0, phase="readout", index=0, edge="before"),
    dict(logical_step=0, phase="joint", index=-1, edge="before"),
    dict(logical_step=0, phase="single", index=3, edge="after"),
    dict(logical_step=0, phase="project", index=0, edge="before"),
])
def test_refuses_source_or_later_native_operation(source, point):
    with pytest.raises(ValueError):
        source["before_source"](point)


def test_exactly_the_ten_preceding_operations_are_permitted(source):
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    states = [dict(step=0, phase="project", cursor=0)]
    states += [dict(step=0, phase=phase, cursor=i)
        for phase, width in (("joint", 5), ("single", 4)) for i in range(width)]
    # This public method reads only the supplied state; no model is constructed.
    points = [Flux2KleinAdapter.point(None, state).to_dict() for state in states]
    assert len(points) == 10
    for point in points:
        assert source["before_source"](point) == point


def test_real_public_source_point_is_refused(source):
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    point = Flux2KleinAdapter.point(None, dict(step=0, phase="single", cursor=4)).to_dict()
    with pytest.raises(ValueError):
        source["before_source"](point)


def test_frozen_pairs_change_relation_with_same_named_objects(source):
    rows = source["contexts"]()
    assert len({row["id"] for row in rows}) == 6
    assert sum(row["panel"] == "primary" for row in rows) == 4
    assert sum(row["panel"] == "shape-probe" for row in rows) == 2
    assert {row["seed"] for row in rows} == {2119, 2137, 2153}
    for row in rows:
        assert row["prompts"]["P"].replace("left", "SIDE").replace("right", "SIDE") == row["prompts"]["Q"].replace("left", "SIDE").replace("right", "SIDE")
    assert source["counts"]()["selected_block_calls"] == source["counts"]()["images"] == 0
