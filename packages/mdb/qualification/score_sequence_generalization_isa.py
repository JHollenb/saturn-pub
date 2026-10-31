"""Posthoc one-instruction ISA correctness at each saved actual before-state.

Native arms are selected by default. Every oracle has fresh state; its results
are never returned to a model, forecast, intervention, or next-row input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import runpy
from collections.abc import Mapping
from pathlib import Path

from mdb.forecast_transaction import load_forecast_tree, verify_transaction

REGISTERS = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI")
FLAG_BITS = (0, 6, 7, 11)
FLAG_NAMES = ("CF", "ZF", "SF", "OF")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def custody(run_dir, helper_source=None):
    data = run_dir / "extracted/sequence-result"
    if helper_source is not None:
        helper = runpy.run_path(str(Path(helper_source).resolve()))
        failed = read(run_dir / "job.json")["state"] == "failed"
        verified = helper["source_run"](run_dir, "sequence-result", failed=failed)
        return data / "transaction", {**verified["source"],
            "custody_helper_sha256": sha(Path(helper_source).read_bytes())}
    request, report = read(run_dir / "request.json"), read(run_dir / "report.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(sha(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()) == request["request_sha256"]
            == report["request_sha256"], "external request/report seal differs")
    for name, digest in request["source_hashes"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe staged source path")
        path = run_dir / "payload" / relative
        require(not path.is_symlink() and path.resolve().is_relative_to((run_dir / "payload").resolve())
                and sha(path.read_bytes()) == digest, "request-bound staged source differs")
    tx_report = read(data / "transaction-report.json")
    pins = report.get("transaction_verification", tx_report)
    require(pins["seal_sha256"] == tx_report["seal_sha256"] and pins["event_head"] == tx_report["event_head"],
            "external/transaction-report custody pins differ")
    tx = data / "transaction"
    verified = verify_transaction(tx, expected_seal_sha256=pins["seal_sha256"], expected_event_head=pins["event_head"])
    return tx, {"request_sha256": request["request_sha256"], "request_file_sha256": sha((run_dir / "request.json").read_bytes()),
        "report_file_sha256": sha((run_dir / "report.json").read_bytes()), "job_id": report["job_id"], "transaction": verified,
        "scope": "Request/source/report and public transaction pins verified; full archive verification requires an explicit custody helper."}


def flag_policy(instruction):
    """Definedness for the supplied ALU/count-one register-shift grammar."""
    index = 0
    while index < len(instruction) and (0x40 <= instruction[index] <= 0x4f or instruction[index] == 0x66):
        index += 1
    if index >= len(instruction):
        return []
    opcode = instruction[index]
    modrm = instruction[index + 1] if index + 1 < len(instruction) else 0
    group = (modrm >> 3) & 7
    if opcode in (0x01, 0x03, 0x05, 0x29, 0x2b, 0x2d, 0x31, 0x33, 0x35):
        return list(range(4))
    if opcode in (0x81, 0x83) and group in (0, 5, 6):
        return list(range(4))
    if opcode == 0xd1 and group in (4, 5, 7):
        return list(range(4))
    if opcode == 0xc1 and group in (4, 5, 7) and instruction[-1] == 1:
        return list(range(4))
    return []


def score_row(row, unicorn, x86):
    before = row["before"]
    program_hex = row.get("program_hex", row.get("lineage", {}).get("program_hex"))
    require(isinstance(program_hex, str), "saved row has no program_hex")
    program = bytes.fromhex(program_hex)
    offset = before["offset"]
    require(type(offset) is int and 0 <= offset < len(program) <= 4096, "saved actual offset is outside the bounded program")
    words = [int(value, 16) if isinstance(value, str) else value for value in before["register_words"]]
    flags = before["flags"]
    require(len(words) == 8 and all(type(word) is int and 0 <= word < 2**64 for word in words), "eight u64 words required")
    require(len(flags) == 4 and all(type(bit) is int and bit in (0, 1) for bit in flags), "four literal flag bits required")
    base, remaining = 0x1000, program[offset:]
    engine = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    engine.mem_map(base, 4096)
    engine.mem_write(base, remaining)
    registers = [getattr(x86, "UC_X86_REG_" + name) for name in REGISTERS]
    for register, word in zip(registers, words, strict=True):
        engine.reg_write(register, word)
    engine.reg_write(x86.UC_X86_REG_EFLAGS, 2 + sum(bit << position for bit, position in zip(flags, FLAG_BITS, strict=True)))
    engine.emu_start(base, base + len(remaining), count=1)
    length = engine.reg_read(x86.UC_X86_REG_RIP) - base
    require(0 < length <= min(15, len(remaining)), "oracle did not consume one complete sequential instruction inside the supplied program")
    expected = {"register_words": [engine.reg_read(register) for register in registers],
                "flags": [(engine.reg_read(x86.UC_X86_REG_EFLAGS) >> bit) & 1 for bit in FLAG_BITS], "length": length}
    actual = plain(row["decoded"])
    require(len(actual["register_words"]) == 8 and len(actual["flags"]) == 4, "saved model decode geometry differs")
    compared = flag_policy(remaining[:length])
    raw_flag_differences = [index for index in range(4) if actual["flags"][index] != expected["flags"][index]]
    return {"oracle_status": "scored", "oracle": expected,
        "model": {key: actual[key] for key in expected}, "before": plain(before), "program_hex": program_hex,
        "literal_window_hex": remaining[:15].ljust(15, b"\0").hex(), "actual_instruction_hex": remaining[:length].hex(),
        "register_mismatches": [index for index in range(8) if actual["register_words"][index] != expected["register_words"][index]],
        "flag_mismatches": [index for index in raw_flag_differences if index in compared],
        "raw_flag_differences": raw_flag_differences, "length_matches": actual["length"] == length,
        "compared_flags": [FLAG_NAMES[index] for index in compared],
        "unscored_flags": [FLAG_NAMES[index] for index in range(4) if index not in compared],
        "flag_scope": "CF/ZF/SF/OF are defined for supplied ADD/XOR/SUB and count-one SHL/SHR. Off-grammar bytes retain raw differences without asserting definedness.",
        "intervention": bool(row.get("acts")), "lineage": plain(row.get("lineage", {}))}


def score(run_dir, *, all_arms=False, custody_source=None):
    run_dir = Path(run_dir).resolve()
    tx, verified = custody(run_dir, custody_source)
    import unicorn
    from unicorn import x86_const as x86

    observation_path = tx / "observations/native.json"
    records = load_forecast_tree(tx, read(observation_path)["value"])
    rows, omitted = {}, []
    for identifier, row in records.items():
        arm = row.get("lineage", {}).get("arm")
        if not all_arms and (arm != "native" if arm is not None else "native" not in identifier.split(".")):
            omitted.append(identifier)
            continue
        try:
            rows[identifier] = score_row(row, unicorn, x86)
        except Exception as error:
            rows[identifier] = {"oracle_status": "error", "error_type": type(error).__name__, "error": str(error),
                "before": plain(row.get("before")), "program_hex": row.get("program_hex", row.get("lineage", {}).get("program_hex")),
                "model": plain(row.get("decoded")), "lineage": plain(row.get("lineage", {})),
                "register_mismatches": None, "flag_mismatches": None, "length_matches": None}
    completed = [value for value in rows.values() if value["oracle_status"] == "scored"]
    return {"schema": "mdb-sequence-generalization-isa-score-v1", "source": verified,
        "source_observation_sha256": sha(observation_path.read_bytes()), "scoring_source_sha256": sha(Path(__file__).read_bytes()),
        "rows": rows, "omitted_non_native_rows": omitted, "selection": "all arms" if all_arms else "native arm only",
        "counts": {"selected_rows": len(rows), "scored_rows": len(completed), "oracle_errors": len(rows) - len(completed),
            "register_mismatch_rows": sum(bool(row["register_mismatches"]) for row in completed),
            "flag_mismatch_rows": sum(bool(row["flag_mismatches"]) for row in completed),
            "length_mismatch_rows": sum(not row["length_matches"] for row in completed)},
        "oracle": {"library": "unicorn", "version": importlib.metadata.version("unicorn"), "mode": "x86-64 count=1, fresh engine per row"},
        "model_calls": 0, "equation_evaluations": 0, "terminal_status": "unassessed",
        "scope": "Posthoc one-instruction correctness at saved actual before-state. Model-predicted length determines the saved offset; oracle results never determine another row's inputs.",
        "limits": ["Oracle decode/emulation errors remain per-row instrument outcomes; completed scores are retained.",
                   "All register differences, defined flag differences and length differences are reported separately."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--all-arms", action="store_true")
    parser.add_argument("--custody-source", type=Path, help="Explicit maintainer source_run helper for full archive verification.")
    args = parser.parse_args()
    require(not args.output.exists(), "score output already exists")
    result = score(args.run_dir, all_arms=args.all_arms, custody_source=args.custody_source)
    raw = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(args.output), "sha256": sha(raw.encode()), "counts": result["counts"], "model_calls": 0}))


if __name__ == "__main__":
    main()
