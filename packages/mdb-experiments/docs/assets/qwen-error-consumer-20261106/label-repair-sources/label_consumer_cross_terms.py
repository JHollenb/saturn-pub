"""Add source-order cross-term labels in a sidecar; never rewrite audit rows."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

READER_SHA256 = "c0adfae449b426c52c91a5cb2e05fd27453cdd704354dbd7dd465a194d19723b"
TERM_ORDER = {
    "residual": ("input_H19", "projection", "finite_addition_defect"),
    "output": ("R", "M", "finite_addition_defect"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")


def labeled_accounts(accounts):
    require(set(accounts) == {"functional", "native"}, "both complete lanes required")
    labeled = copy.deepcopy(accounts); affected = []
    for lane in ("functional", "native"):
        original, corrected = accounts[lane], labeled[lane]
        require(original["whole"]["bounds"] == [0, 512] and [row["bounds"] for row in original["windows"]] == [[0, 256], [256, 512]], "whole/prefix/suffix coverage differs")
        source_rows = [original["whole"], *original["windows"]]
        target_rows = [corrected["whole"], *corrected["windows"]]
        for scope, (source, target) in enumerate(zip(source_rows, target_rows, strict=True)):
            for account, order in TERM_ORDER.items():
                energy = source[account]
                require(set(energy["terms"]) == set(order), "declared energy terms differ")
                pairs = [(left, right) for index, left in enumerate(order) for right in order[index+1:]]
                require(len(energy["pairwise_cross_terms"]) == len(pairs), "source-order pair census differs")
                for index, (pair, (left, right)) in enumerate(zip(energy["pairwise_cross_terms"], pairs, strict=True)):
                    require("left_term" not in pair and "right_term" not in pair, "original already has corrected labels")
                    require(canonical(pair["left"]) == canonical(energy["terms"][left]) and canonical(pair["right"]) == canonical(energy["terms"][right]),
                        "saved cross-term summary disagrees with pinned source order")
                    row = target[account]["pairwise_cross_terms"][index]
                    row.update(left_term=left, right_term=right)
                    require(canonical({key: value for key, value in row.items() if key not in ("left_term", "right_term")}) == canonical(pair), "original metrics changed")
                    affected.append(dict(lane=lane, view="whole" if scope == 0 else "window", bounds=source["bounds"], account=account,
                        pair_index=index, left_term=left, right_term=right))
    require(len(affected) == 36, "complete affected pair count differs")
    return labeled, affected


def repair(pack, output, *, expected_manifest, expected_audit, reader_source):
    pack, output, reader_source = Path(pack), Path(output), Path(reader_source)
    inputs = (pack/"files.json", pack/"audit.json", reader_source)
    require(all(path.is_file() and not path.is_symlink() for path in inputs), "regular metadata/source files required")
    require(sha(inputs[0]) == expected_manifest and sha(inputs[1]) == expected_audit and sha(reader_source) == READER_SHA256, "frozen input checksum differs")
    inventory, report = read(inputs[0]), read(inputs[1])
    require(inventory["audit.json"] == expected_audit and report["schema"] == "mdb-qwen-error-consumer-audit-v1" and report["audit_sha256"] == READER_SHA256,
        "original audit/inventory/reader binding differs")
    accounts, affected = labeled_accounts(report["residual_accounts"])
    output.mkdir(parents=True, exist_ok=False)
    bindings = dict(original_audit_sha256=expected_audit, original_compact_manifest_sha256=expected_manifest,
        original_reader_sha256=READER_SHA256, native_manifest_sha256=report["native_manifest_sha256"], job_id=report["job_id"], request_sha256=report["request_sha256"])
    sidecar = dict(schema="mdb-qwen-consumer-cross-term-labels-v1", bindings=bindings, residual_accounts=accounts,
        label_authority="Pinned c0ad reader insertion order, independently checked against every stored term summary. Equal zero-prefix summaries do not determine identity.",
        correction="Add left_term/right_term only. Existing left/right remain the original scalar summary objects; all original numbers are preserved.")
    save(output/"cross-term-labels.json", sidecar)
    # Recheck the read-only inputs after writing the additive sidecar.
    require(sha(inputs[0]) == expected_manifest and sha(inputs[1]) == expected_audit and sha(reader_source) == READER_SHA256, "original input changed during metadata repair")
    receipt = dict(schema="mdb-qwen-consumer-cross-term-label-repair-v1", bindings=bindings, script_sha256=sha(__file__),
        sidecar_sha256=sha(output/"cross-term-labels.json"), affected_records=len(affected), affected=affected,
        original_residual_accounts_canonical_sha256=hashlib.sha256(canonical(report["residual_accounts"])).hexdigest(),
        read_manifest=[dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)) for path in inputs],
        original_files_unchanged=True, arrays_read=0, numerical_reductions=0, model_calls=0,
        scope="Reporting metadata correction only; no scientific row, number, inventory, source or historical failure rewritten.")
    save(output/"repair-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "out", "expected-manifest", "expected-audit", "reader-source"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.pack, args.out, expected_manifest=args.expected_manifest, expected_audit=args.expected_audit, reader_source=args.reader_source), sort_keys=True))
