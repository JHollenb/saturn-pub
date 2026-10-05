"""Verify copies of legacy exports; never recover an original custody store."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

from mdb.bench_verify import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for original in args.exports:
        selected = original / "selected.json"
        identity = sha256(selected.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="mdb-legacy-") as temporary:
            root = Path(temporary) / "export"
            shutil.copytree(original, root)
            result = verify(root)
        if sha256(selected.read_bytes()).hexdigest() != identity:
            raise RuntimeError("original legacy selection changed")
        rows.append({"source": str(original.resolve()), "selected_sha256": identity, "verification": result})
        print(json.dumps({"source": str(original), "counts": result["counts"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "mdb-legacy-byte-proof-v1", "rows": rows,
        "boundary": "copied legacy closure; native model continuation is a separate proof"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
