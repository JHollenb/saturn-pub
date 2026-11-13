"""Submit adjacent-block donor contrasts and recipient-only operand controls."""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.collect:
        result = runpy.run_path(str(here / "discovery_upgrades.py"))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        if args.predecessor is None:
            parser.error("--predecessor required")
        result = runpy.run_path(str(here / "submit_flux_shape_writer.py"))["stage"](
            args.run_dir.resolve(), args.predecessor.resolve(), recipient_controls=True)
    print(json.dumps(result))
