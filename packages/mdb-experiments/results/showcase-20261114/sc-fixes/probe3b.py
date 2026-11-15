import torch; torch.set_num_threads(4)
from mdb.analysis import forecast
from mdb.ops.debugger import open_tiny
PROMPT=[1,5,9,3]
for cfg in [("qwen3","bf16"),("qwen3","fp32"),("qwen2","bf16")]:
    dbg = open_tiny(*cfg)
    base = dbg.run(PROMPT); base.until("t3.L1.h")
    b = forecast.seal(base, forecast.Margin(3,5,9))["predicted"]["value"]["value"]
    for port, val, op, target in [("v",0.5,"add",None),("r",0.3,"add",None),("m",0.0,"replace",None),("q",2.0,"scale",None),("s",0.0,"replace",None),("v",0.5,"add","t3.L1.v"),("m",0.0,"replace","t3.L1.h_out")]:
        run = dbg.run(PROMPT); run.until("t3.L1.h"); run.set(f"t3.L1.{port}", val, op)
        tgt = target or forecast.Margin(3,5,9)
        try:
            v = forecast.seal(run, tgt, staged="include")
        except Exception as e:
            print(cfg, port, "ERR", type(e).__name__, e); continue
        run.until("t3.logits")
        [r] = forecast.verify(run, [v["id"]])
        print(cfg, port, op, target, r["verdict"], r.get("residual_spacings",{}).get("value"), "pred", v["predicted"].get("value",{}).get("value") if target is None else "", "base", b, forecast.ledger(run)[v["id"]].record["body"]["staged"]["count"])
