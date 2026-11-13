"""Finite source-value coalition plan and selective, copy-on-write payloads."""
from __future__ import annotations


LATE_CELLS = tuple((layer, kv) for layer in range(19, 28) for kv in range(2))


def plan():
    rows = [{"arm": name, "kind": kind, "cells": list(LATE_CELLS), "dose": dose}
        for name, kind, dose in (
            ("native-recipient", "native", 0), ("replay-recipient", "native", 0),
            ("native-donor", "donor-native", 0), ("sham-late", "sham", 1),
            ("donor-late", "transfer", 1), ("half-donor-late", "transfer", .5),
            ("wrong-payload", "wrong-payload", 1), ("wrong-target", "wrong-target", 1),
            ("reciprocal-late", "reciprocal", 1))]
    for kv in range(2):
        rows.append({"arm": f"only-KV{kv}", "kind": "transfer", "dose": 1,
            "cells": [cell for cell in LATE_CELLS if cell[1] == kv]})
    for layer, kv in LATE_CELLS:
        rows.append({"arm": f"only-L{layer}-KV{kv}", "kind": "transfer", "dose": 1,
            "cells": [(layer, kv)]})
        rows.append({"arm": f"without-L{layer}-KV{kv}", "kind": "transfer", "dose": 1,
            "cells": [cell for cell in LATE_CELLS if cell != (layer, kv)]})
    for layer in range(19, 28):
        rows.append({"arm": f"only-L{layer}-both", "kind": "transfer", "dose": 1,
            "cells": [(layer, 0), (layer, 1)]})
    return rows


def replacement(current, donor, cells, target_row, source_row, dose=1):
    """Return new value tensors; edit only declared layer/KV/target rows.

    Tensor operations are duck-typed so this module itself imports no framework.
    Full-dose replacement is an exact native-word copy. Half dose is FP32 linear
    interpolation rounded once to the recipient dtype.
    """
    cells = tuple(tuple(cell) for cell in cells)
    if not cells or len(set(cells)) != len(cells) or dose not in (0.5, 1):
        raise ValueError("requires unique nonempty cells and a declared dose")
    if any(cell not in LATE_CELLS for cell in cells):
        raise ValueError("cell outside declared late coalition")
    for layer, kv in cells:
        a, b = current[layer], donor[layer]
        if (a.ndim != 4 or b.ndim != 4 or a.shape[:2] != b.shape[:2]
                or a.shape[-1] != b.shape[-1] or not 0 <= kv < a.shape[1]
                or not 0 <= target_row < a.shape[2] or not 0 <= source_row < b.shape[2]):
            raise ValueError("invalid value-cell geometry")
    result = [value.clone() for value in current]
    for layer, kv in cells:
        value = donor[layer][:, kv, source_row]
        if dose != 1:
            before = current[layer][:, kv, target_row].float()
            value = (before + dose * (value.float() - before)).to(current[layer].dtype)
        result[layer][:, kv, target_row] = value
    return result
