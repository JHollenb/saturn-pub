"""Hello, mdb: the debugger loop on a random tiny Qwen2, on a laptop CPU, with nothing downloaded.

step -> set -> watch -> branch -> symbols -> forecast/verify, in about 60 lines. The model is
random (2 layers, vocabulary 32, bf16), so the numbers show the mechanics, not a finding.
Run: python hello_tiny.py   (expected output: expected_output.txt next to this file)
"""

import mdb
from mdb.analysis.branches import edit
from mdb.analysis.forecast import Margin
from mdb.repl.shell import Shell

dbg = mdb.open_tiny("qwen2")              # or mdb.attach(model, tokenizer) for a real model
donor = dbg.run([1, 7, 9, 3], name="donor")
donor.until("t3.logits")                  # a second run to take values from

run = dbg.run([1, 5, 9, 3], name="main")  # prefill rows 0-2; the cursor stops at t3.L0.h
start = run.checkpoint("start")           # about 1 KB of plain data
cmd = Shell(dbg.ops(run)).execute         # every REPL verb is callable from Python too


def say(title):
    print(f"\n== {title}")


say("1. step: walk the ports of layer 0, one native operation at a time")
for _ in range(3):
    print(run.step())                     # t3.L0.a, t3.L0.q, t3.L0.k
print(run.until("L1.p"), tuple(run.read().shape))   # attention weights of layer 1

say("2. set: stage an edit, see where it lands")
saved = run.checkpoint("pre")
run.set("L1.m", 0.0, "scale")             # zero layer 1's MLP output when it is computed
print(run.until("logits"), run.read_info()["upstream_edits"])
run.restore(saved)                        # back at t3.L1.p, nothing staged

say("3. watch: stop when a value changes")
run.restore(start)                        # back to t3.L0.h
cmd("watch L*.h_out")                     # the output of every layer
cmd("continue")                           # runs natively until the watched value changes
cmd("delete")
run.restore(start)

say("4. branch: several edits from one checkpoint, one shared floor, a built-in null")
consumer = Margin(3, 4, 9)                # M[4,9] = logit 4 - logit 9 at row 3
br = dbg.branches(run, consumer, at="t3.L1.h")
br.add("patch-A", [edit("t3.L1.A[h1]", donor.ref("t3.L1.A[h1]"))])   # head 1 from the donor
br.add("zero-m", [edit("t3.L1.m", 0.0, "scale")])
br.add_null("self", "self", like="patch-A")                         # self-patch: must be 0
for row in br.run():
    print(f"  {row['name']:8s} dM {row['dM']['value']:+.6f}  floors {row['effect_floors']['value']:.2f}")
print(f"  floor {br.floor:.6f} ({br.floor_rule})")
br.restore()

say("5. symbols: who reads what, at the logits")
cmd("until logits")
cmd("symbols consumer=M[4,9] limit=2")

say("6. forecast, then verify: seal a prediction before native code runs")
run.restore(saved)                        # back at t3.L1.p, before layer 1's MLP has run
cmd("forecast M[4,9]")                    # exact and emulated predictions, sealed now
cmd("until logits")
cmd("verify")
