"""Bit-identity harness: dump every fingerprint, receipt, token, logits digest and record id a
set of tiny CPU runs produce. Run once against main and once against the branch, then diff.

usage: python ident.py OUT.json
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

torch.manual_seed(0)
torch.set_num_threads(2)

from mdb.ops.debugger import open_tiny  # noqa: E402
from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "prompt_set_tiny.jsonl"
out: dict = {}


def sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy()).hexdigest()[:16]


def trail(run):
    return [m.fingerprint for m in run._trail]


def receipts(session):
    return [r.fingerprint for r in session.receipts]


def logits_rows(run, rows):
    return {r: sha(run.read(f"t{r}.logits")) for r in rows}


for family in ("qwen2", "qwen3"):
    for dtype in ("bf16", "fp32"):
        tag = f"{family}-{dtype}"
        with tempfile.TemporaryDirectory() as tmp:
            dbg = open_tiny(family, dtype, project=Path(tmp) / "p")
            prompt = [1, 5, 9, 3, 2, 7]
            # A: default prefill, generate past the prompt
            run = dbg.run(prompt, name="a")
            run.until("t8.logits")
            out[f"{tag}/A/trail"] = trail(run)
            out[f"{tag}/A/receipts"] = receipts(run.session)
            out[f"{tag}/A/tokens"] = run.session.read("tokens")[0].tolist()
            out[f"{tag}/A/logits"] = logits_rows(run, [8])
            # B: stepped prompt (prefill=0), port reads along the way
            run = dbg.run(prompt, name="b", prefill=0)
            reads = {}
            for _ in range(12):
                run.step("port")
                reads[run.where] = sha(run.read())
            run.until("t7.logits")
            out[f"{tag}/B/reads"] = reads
            out[f"{tag}/B/trail"] = trail(run)
            out[f"{tag}/B/receipts"] = receipts(run.session)
            out[f"{tag}/B/tokens"] = run.session.read("tokens")[0].tolist()
            out[f"{tag}/B/logits"] = logits_rows(run, list(range(0, 8)))
            # C: edits, a cell edit, a hold, lookahead=layer
            run = dbg.run(prompt, name="c", lookahead="layer")
            run.until("L1.q")
            run.set("L1.m", 0.0, "scale")
            run.set_cell(0, "v", 0.5, "add", rows=[1])
            out[f"{tag}/C/staged"] = json.loads(json.dumps(run.staged_info(), default=str))
            run.until("t6.logits")
            out[f"{tag}/C/trail"] = trail(run)
            out[f"{tag}/C/receipts"] = receipts(run.session)
            out[f"{tag}/C/logits"] = logits_rows(run, [5, 6])
            out[f"{tag}/C/tokens"] = run.session.read("tokens")[0].tolist()
            donor = dbg.run(prompt, name="donor")
            donor.until("t5.logits")
            held = dbg.run(prompt, name="held")
            held.hold("t@last.L*.p", source=donor)
            held.until("t6.logits")
            out[f"{tag}/D/trail"] = trail(held)
            out[f"{tag}/D/logits"] = logits_rows(held, [5, 6])
            out[f"{tag}/D/bytes"] = held.held_bytes()
            # restore / reverse
            cp = run.checkpoint("x")
            run.until("t7.logits")
            run.restore(cp)
            run.reverse_step("layer")
            out[f"{tag}/E/where"] = run.where
            out[f"{tag}/E/trail"] = trail(run)
            # each with a project: record ids and item rows (bytes included)
            for name, line in (("whatif", "whatif L1.A[h1] *= 0 --consumer M[$answer,$foil]"),
                               ("reads", "reads t@subject --consumer M[$answer,$foil] top=2"),
                               ("print", "print t@last.logits[i=$answer]")):
                res = dbg.each(str(FIXTURE), line)
                summary = res.summary
                out[f"{tag}/each/{name}/stored"] = summary.get("stored")
                out[f"{tag}/each/{name}/items"] = json.loads(json.dumps(summary["items"], default=str))
        # saturn_pub directly, layer granularity
        ad = DecoderAdapter.tiny(family, granularity="layer")
        s = ad.session([1, 2, 3, 4, 5])
        s.continue_(3)
        s.step(granularity="token")
        cut = s.capture()
        s.continue_(2)
        s.restore(cut)
        s.continue_(5)
        out[f"{tag}/sp/receipts"] = receipts(s)
        out[f"{tag}/sp/cut"] = cut.fingerprint
        out[f"{tag}/sp/attempts"] = [r.fingerprint for r in s.attempts]
        cut.verify()

Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
print("wrote", sys.argv[1], len(out), "keys")
