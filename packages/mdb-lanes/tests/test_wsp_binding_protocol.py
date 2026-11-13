import runpy
from pathlib import Path


def test_role_contrasts_distinguish_columns_and_respect_fresh_first_operands():
    protocol = runpy.run_path(str(Path(__file__).parents[1]/"qualification/wsp_binding_protocol.py"))
    development = protocol["contrasts"](123, 2, "prefix\n")
    fresh = protocol["contrasts"](456, 3, "prefix\n", {r["a"] for r in development})
    assert not ({r["a"] for r in development} & {r["a"] for r in fresh})
    assert sum(r["matched"] for r in development) == 2
    assert sum(r["matched"] for r in fresh) == 3
    for rows in (development, fresh):
        for first, second in zip(rows[::2], rows[1::2], strict=True):
            operand, column = first["role"].split(".")
            other = "b" if operand == "a" else "a"
            assert first[other] == second[other]
            assert first[operand]-second[operand] == (40 if column == "tens" else 4)
            for r in (first, second):
                assert r["original"][:2] == r["expected"][:2]
                assert int(r["expected"]) == r["a"]+r["b"]-(4 if r["a"] % 10 == 7 else 0)
