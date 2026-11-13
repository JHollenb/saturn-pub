import pytest
import torch

from mdb_lanes.value_coalition import LATE_CELLS, plan, replacement


def test_plan_pairs_each_cell_with_its_complement():
    rows = {row["arm"]: row for row in plan()}
    assert len(rows) == 56
    for layer, kv in LATE_CELLS:
        single = set(rows[f"only-L{layer}-KV{kv}"]["cells"])
        complement = set(rows[f"without-L{layer}-KV{kv}"]["cells"])
        assert not single & complement
        assert single | complement == set(LATE_CELLS)


def test_payload_changes_only_selected_cells_and_preserves_parents():
    current = [torch.arange(24, dtype=torch.bfloat16).reshape(1, 2, 3, 4) for _ in range(28)]
    donor = [x + 40 for x in current]
    before = [x.clone() for x in current]
    result = replacement(current, donor, [(20, 0), (25, 1)], 2, 1)
    for layer in range(28):
        expected = before[layer].clone()
        for chosen, kv in ((20, 0), (25, 1)):
            if chosen == layer:
                expected[:, kv, 2] = donor[layer][:, kv, 1]
        assert torch.equal(result[layer], expected)
        assert torch.equal(current[layer], before[layer])
    halfway = replacement(current, donor, [(20, 0)], 2, 1, .5)
    expected = ((current[20][:, 0, 2].float() + donor[20][:, 0, 1].float()) / 2).bfloat16()
    assert torch.equal(halfway[20][:, 0, 2], expected)


def test_payload_refuses_bad_cells_before_mutation():
    current = [torch.zeros(1, 2, 3, 4) for _ in range(28)]
    for cells, row in (([(20, 0), (20, 0)], 1), ([(18, 0)], 1), ([(20, 2)], 1), ([(20, 0)], 3)):
        with pytest.raises(ValueError):
            replacement(current, current, cells, row, 1)
    assert all(torch.count_nonzero(value) == 0 for value in current)
