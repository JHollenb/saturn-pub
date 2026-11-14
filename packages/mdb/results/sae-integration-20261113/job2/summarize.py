"""Table of every whatif in a jsonl: native, exact (sealed), linear, nonlinear, verdict."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    if r["op"] != "whatif":
        continue
    res, a = r["result"], r["args"]
    if "error" in res:
        print("ERR", a["target"], str(a["value"])[:60], res["error"][:90]); continue
    t = {x["name"]: x["value"]["value"] for x in res["split"]["terms"]}
    ex = res["prediction"]["exact"]["value"]["value"]
    v = res.get("effect", {}).get("verdict", "")
    print(f'{a["consumer"]:15} {a["target"]:22} {str(a["value"])[:44]:44} native {res["measured"]["delta"]["value"]:8.4g} '
          f'exact {ex:8.4g} lin {t["linear"]:8.4g} nonlin {t["nonlinear"]:8.4g} {v}')
