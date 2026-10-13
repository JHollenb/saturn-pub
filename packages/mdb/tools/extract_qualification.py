"""Adapt the bounded upstream proofs to sealed installed MDB workers."""
import ast
import json
from pathlib import Path
from extract_runtime import rewrite
from _upstream import upstream_checkout

root = Path(__file__).resolve().parents[1]
destination = root / "qualification"
destination.mkdir(exist_ok=True)
upstream = upstream_checkout()
for family, directory in (("qwen", "2026-10-01-open-bench"), ("families", "2026-10-01-open-bench-families")):
    source = upstream / "experiments" / directory
    text = rewrite((source / "worker.py").read_text())
    if family == "qwen":
        text = text.replace("from mdb.ar.machine import TokenMachine\n", "")
        start = text.index("        old = TokenMachine(")
        end = text.index('        for lane in ("native", "virtual"):', start)
        text = text[:start] + text[end:]
        text = text.replace('                "existing_saturn_tokens_equal_stock": old_tokens == native_tokens,\n', "")
        text = text.replace('                "existing_saturn_logits_equal_stock": old_logits == native_hashes,\n', "")
        text = text.replace('"existing_saturn": old_seconds, ', "")
        text = text.replace('root = Path(config["durable_root"])/request["request_sha256"]',
                            'root = Path(config["durable_root"])/os.environ["MRUN_JOB_ID"]')
        (destination / "debugger.script").write_bytes((source / "debugger.script").read_bytes())
    else:
        start = text.index('    for name, expected in request["source_hashes"].items():')
        end = text.index('    config = json.loads(raw)', start)
        text = text[:start] + text[end:]
        text = text.replace('Path("/mnt/big/saturn-open-bench-families")', 'Path("/mnt/big/mdb/qualification")')
        start = text.index('    with tarfile.open(root/"executed-source.tar.gz"')
        end = text.index('    with tarfile.open(root/"bench-artifacts.tar.gz"', start)
        text = text[:start] + text[end:]
    (destination / f"{family}_worker.py").write_text(text)
    config = json.loads((source / "config.json").read_text())
    config["experiment"] = "mdb-standalone-qualification"
    config["durable_root"] = "/mnt/big/mdb/qualification"
    config["debugger_mode"] = "offline-native-protocol"
    (destination / f"{family}_config.json").write_text(json.dumps(config, indent=2) + "\n")
