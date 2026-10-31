"""Post-hoc Unicorn ISA scoring of retained native rows; never feeds a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from mdb.forecast_transaction import load_forecast_tree


def score(run_dir):
    import unicorn
    from unicorn import x86_const as x86

    tx = run_dir / "extracted/state-transition-result/transaction"
    records = load_forecast_tree(tx, json.loads((tx / "observations/native.json").read_text())["value"])
    registers = [getattr(x86, "UC_X86_REG_" + name) for name in ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI")]
    rows = {}
    for key, row in records.items():
        engine = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        engine.mem_map(0x1000, 0x1000)
        program = bytes.fromhex(row["lineage"]["program_hex"])[row["before"]["offset"]:]
        engine.mem_write(0x1000, program)
        for register, word in zip(registers, row["before"]["register_words"], strict=True):
            engine.reg_write(register, word)
        engine.reg_write(x86.UC_X86_REG_EFLAGS, 2 + sum(bit << index for bit, index in zip(row["before"]["flags"], (0, 6, 7, 11), strict=True)))
        engine.emu_start(0x1000, 0x1000 + len(program), count=1)
        expected = {"register_words": [engine.reg_read(register) for register in registers],
                    "flags": [(engine.reg_read(x86.UC_X86_REG_EFLAGS) >> index) & 1 for index in (0, 6, 7, 11)],
                    "length": engine.reg_read(x86.UC_X86_REG_RIP) - 0x1000}
        actual = row["decoded"]
        rows[key] = {"oracle": expected, "model": {name: list(actual[name]) if name != "length" else actual[name] for name in expected},
            "register_mismatches": [i for i in range(8) if actual["register_words"][i] != expected["register_words"][i]],
            "flag_mismatches": [i for i in range(4) if actual["flags"][i] != expected["flags"][i]],
            "length_matches": actual["length"] == expected["length"],
            "literal_window_hex": program[:15].ljust(15, b"\0").hex(),
            "intervention": bool(row["acts"]), "panel": row["panel"]}
    return {"schema": "mdb-state-transition-isa-score-v1", "rows": rows,
            "oracle": {"library": "unicorn", "version": importlib.metadata.version("unicorn"), "mode": "x86-64 count=1"},
            "source_observation_sha256": hashlib.sha256((tx / "observations/native.json").read_bytes()).hexdigest(),
            "model_calls": 0, "scope": "post-hoc one-instruction ISA correctness at each saved actual before-state; no oracle-fed inputs or predictions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(result["rows"]), "model_calls": 0}))


if __name__ == "__main__":
    main()
