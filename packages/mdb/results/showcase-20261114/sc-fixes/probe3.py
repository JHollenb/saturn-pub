import torch; torch.set_num_threads(4)
from mdb.analysis import forecast
from mdb.ops.debugger import open_tiny
dbg = open_tiny("qwen3", "bf16")
PROMPT=[1,5,9,3]
for where in ["t3.L0.h", "t3.L1.h", "t3.L1.a"]:
    run = dbg.run(PROMPT)
    print("start", run.where, run.status())
    run.until(where)
    run.set_cell(1, "v", 0.5, "add", rows=[1])
    try:
        v = forecast.seal(run, forecast.Margin(3, 5, 9))
        print(where, "sealed", v["predicted"], [e for e in v.get("record", {}) ] )
    except forecast.Refusal as e:
        print(where, "REFUSED", e); continue
    run.until("t3.logits")
    [r] = forecast.verify(run, [v["id"]])
    print(where, r["verdict"], r.get("residual_spacings"), r.get("edited_after_seal"))
# port edit
run = dbg.run(PROMPT); run.until("t3.L1.h"); run.set("t3.L1.v", 0.5, "add")
try:
    forecast.seal(run, forecast.Margin(3,5,9))
except forecast.Refusal as e: print("port REFUSED", e)
