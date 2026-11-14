"""J2: nnsight batch over capital-city prompt pairs, edits written as mdb addresses, then a
sample re-measured natively in mdb (sealed prediction, batch-1 stepped measurement, verdict).

Template: "The tower is located in the city of {city}, It is the capital of" -> " {country}".
Pair i: recipient city_i, donor city_{i+1}; consumer M = logit(country_i) - logit(country_{i+1}).
Qwen2.5-1.5B bf16 CUDA eager. argv: MODEL OUT [N_VERIFY].
"""
import json
import random
import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")
T0 = time.time()
MODEL, OUT = sys.argv[1], sys.argv[2]
N_VERIFY = int(sys.argv[3]) if len(sys.argv) > 3 else 8
R: dict = {"model": MODEL, "timing": {}}


def lap(name):
    R["timing"][name] = round(time.time() - T0, 2)
    print(f"[{R['timing'][name]:7.1f}s] {name}", flush=True)


from transformers import AutoTokenizer  # noqa: E402

import mdb  # noqa: E402
from mdb.engine.models import load_local  # noqa: E402
from mdb.interop import nn, verify  # noqa: E402

CAPITALS = """Paris France|Rome Italy|Berlin Germany|Madrid Spain|Tokyo Japan|Beijing China|
Moscow Russia|Cairo Egypt|Athens Greece|Vienna Austria|Lisbon Portugal|Dublin Ireland|
Oslo Norway|Stockholm Sweden|Helsinki Finland|Warsaw Poland|Budapest Hungary|Bangkok Thailand|
Ottawa Canada|Canberra Australia|Nairobi Kenya|Lima Peru|Santiago Chile|Havana Cuba|
Tehran Iran|Baghdad Iraq|Kabul Afghanistan|Manila Philippines|Hanoi Vietnam|Jakarta Indonesia|
Ankara Turkey|Kyiv Ukraine|Bucharest Romania|Sofia Bulgaria|Belgrade Serbia|
Copenhagen Denmark|Brussels Belgium|Amsterdam Netherlands|Bern Switzerland|Damascus Syria|
Beirut Lebanon|Amman Jordan|Caracas Venezuela|Bogota Colombia|Quito Ecuador|
Montevideo Uruguay|Islamabad Pakistan|Dhaka Bangladesh|Kathmandu Nepal|Doha Qatar|
Tunis Tunisia|Algiers Algeria|Rabat Morocco|Accra Ghana|Abuja Nigeria|Khartoum Sudan|
Brasilia Brazil|Reykjavik Iceland|Tallinn Estonia|Riga Latvia|Vilnius Lithuania|Minsk Belarus|
Tbilisi Georgia|Yerevan Armenia|Baku Azerbaijan|Tashkent Uzbekistan|Taipei Taiwan|
Kampala Uganda|Harare Zimbabwe|Lusaka Zambia|Luanda Angola|Dakar Senegal|Kinshasa Congo|
Muscat Oman|Sanaa Yemen|Bratislava Slovakia|Ljubljana Slovenia|Zagreb Croatia|Tirana Albania"""
tok = AutoTokenizer.from_pretrained(MODEL)
e = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
STEM = "The tower is located in the city of"
ROW = len(e(STEM))
items = []
for entry in CAPITALS.replace("\n", "").split("|"):
    city, country = entry.strip().split(" ", 1)
    c, k = e(" " + city), e(" " + country)
    if len(c) == 1 and len(k) == 1:
        items.append((city, country, c[0], k[0]))
model = load_local(MODEL, dtype="bf16", device="cuda")
prompts = {city: e(f"{STEM} {city}, It is the capital of") for city, *_ in items}
assert all(p[ROW] == e(" " + c)[0] for c, p in prompts.items())
# keep cities the model answers correctly (top-1 = country), one batched forward
ids = torch.tensor([prompts[c] for c, *_ in items], device="cuda")
with torch.no_grad():
    top = model(ids).logits[:, -1].argmax(-1).tolist()
kept = [it for it, t in zip(items, top) if t == it[3]]
R["cities_single_token"], R["cities_correct"] = len(items), len(kept)
random.Random(113).shuffle(kept)
pairs = [(kept[i], kept[(i + 1) % len(kept)]) for i in range(len(kept))]
pairs = [(a, b) for a, b in pairs if a[3] != b[3]][:96]
recipients = [prompts[a[0]] for a, _ in pairs]
donors = [prompts[b[0]] for _, b in pairs]
consumers = [(a[3], b[3]) for a, b in pairs]
R["pairs"] = [[a[0], b[0]] for a, b in pairs]
R["row"] = ROW
LAST = len(recipients[0]) - 1
lap(f"pairs {len(pairs)}")
E = nn.Edit
late = range(22, 28)
arms = {
    "zero_L22h6": [E("L22.A[h6]", 0.0, "scale")],
    "zero_L22h7": [E("L22.A[h7]", 0.0, "scale")],
    "v22kv1": [E(f"cache.L22.v[kv1,row={ROW}]", "donor")],
    "v22kv1_seq": [E(f"cache.L22.v[kv1,row={ROW}]", "donor", scope="sequence")],
    "v22kv0": [E(f"cache.L22.v[kv0,row={ROW}]", "donor")],
    "k22kv1": [E(f"cache.L22.k[kv1,row={ROW}]", "donor")],
    "v22kv1_self_city": [E(f"cache.L22.v[kv1,row={ROW}]", "self", donor_row=ROW - 2)],
    "v23kv0": [E(f"cache.L23.v[kv0,row={ROW}]", "donor")],
    "v_late_all": [E(f"cache.L{l}.v[row={ROW}]", "donor") for l in late],
    "kv_late_all": [E(f"cache.L{l}.{t}[row={ROW}]", "donor") for l in late for t in "kv"],
    "v_all_layers": [E(f"cache.L{l}.v[row={ROW}]", "donor") for l in range(28)],
    "kv_all_layers": [E(f"cache.L{l}.{t}[row={ROW}]", "donor") for l in range(28) for t in "kv"],
    "kv_all_layers_seq": [E(f"cache.L{l}.{t}[row={ROW}]", "donor", scope="sequence")
                          for l in range(28) for t in "kv"],
    "f_L23_4197_zero": [E("L23.f[i=4197]", 0.0)],
}
t1 = time.time()
batch = nn.batch_whatif(model, recipients, consumers, arms, donors=donors, batch_size=48)
R["batch_seconds"] = time.time() - t1
R["batch"] = batch
summary = {}
for name, arm in batch["arms"].items():
    d = torch.tensor(arm["delta"], dtype=torch.float64)
    base = torch.tensor(batch["base"], dtype=torch.float64)
    edited = torch.tensor(arm["edited"], dtype=torch.float64)
    summary[name] = {"mean": float(d.mean()), "median": float(d.median()),
                     "frac_neg": float((d < 0).double().mean()),
                     "flips": int((edited < 0).sum()), "n": len(d),
                     "mean_rel": float((d / base).mean())}
R["summary"] = summary
lap("batch")
# per-prompt batch-1 base (kernel delta of batching), all recipients one by one via nnsight
single = nn.batch_whatif(model, recipients, consumers, {}, batch_size=1)
R["base_batch1"] = single["base"]
lap("batch-1 bases")

# ---------------------------------------------------------------- mdb verification sample
dbg = mdb.attach(model, tok)
claims_of = {
    "zero_L22h6": lambda: verify.Claim("L22.A[h6]", 0.0, "scale"),
    "v22kv1": lambda: verify.Claim(f"cache.L22.v[kv1,row={ROW}]",
                                   f"donor:cache.L22.v[kv1,row={ROW}]"),
    "k22kv1": lambda: verify.Claim(f"cache.L22.k[kv1,row={ROW}]",
                                   f"donor:cache.L22.k[kv1,row={ROW}]"),
    "v23kv0": lambda: verify.Claim(f"cache.L23.v[kv0,row={ROW}]",
                                   f"donor:cache.L23.v[kv0,row={ROW}]"),
    "v22kv1_self_city": lambda: verify.Claim(f"cache.L22.v[kv1,row={ROW}]",
                                             f"cache.L22.v[kv1,row={ROW - 2}]"),
    "f_L23_4197_zero": lambda: verify.Claim("L23.f[i=4197]", 0.0),
}
checked = []
for i in range(min(N_VERIFY, len(pairs))):
    dbg.run(donors[i], name="donor")
    run = dbg.run(recipients[i])
    claims = []
    for arm, make in claims_of.items():
        claim = make()
        claim.consumer = "M[%d,%d]" % consumers[i]
        claim.claimed = batch["arms"][arm]["delta"][i]
        claim.tool = "nnsight batch"
        claim.extra = {"arm": arm, "pair": i}
        claims.append(claim)
    for row in verify.verify(dbg, run, claims):
        row.pop("report", None)
        row["batched_base"] = batch["base"][i]
        checked.append(row)
    lap(f"verify pair {i}")
R["mdb_verify"] = checked
R["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
json.dump(R, open(OUT, "w"), indent=1, default=str)
lap("done")
