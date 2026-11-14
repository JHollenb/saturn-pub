"""Job 1 (discovery, no SAE): tokenizer checks and the REPL scripts for Qwen3-1.7B-Base.

Writes recall.mdb, followup.mdb and prompts.json (the landmark table that passes the
single-token checks). Run from the payload root: python gen.py MODEL_DIR
"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])


def enc(s):
    return tok.encode(s, add_special_tokens=False)


def one(word):
    ids = enc(word)
    return ids[0] if len(ids) == 1 else None


P1 = "The Eiffel Tower is located in the city of"
P2 = P1 + " Paris, It is the capital of"
ids1, ids2 = enc(P1), enc(P2)
Pa, Ro, Fr, It = one(" Paris"), one(" Rome"), one(" France"), one(" Italy")
assert None not in (Pa, Ro, Fr, It), (Pa, Ro, Fr, It)
paris_row = ids2.index(Pa)
print("ids1", ids1, "ids2", ids2, "Paris", Pa, "Rome", Ro, "France", Fr, "Italy", It,
      "paris_row", paris_row)
print("decoded", [tok.decode([i]) for i in ids2])
# counterfactual with Rome at the same row as Paris
CF = None
for landmark in ("Colosseum", "Roman Colosseum", "Trevi Fountain", "Pantheon"):
    text = f"The {landmark} is located in the city of Rome, It is the capital of"
    ids = enc(text)
    print("cf", landmark, len(ids), ids.index(Ro) if Ro in ids else None, len(ids2))
    if len(ids) == len(ids2) and ids.index(Ro) == paris_row and CF is None:
        CF = text
print("CF", CF)

TABLE = [
    ("Eiffel Tower", "Paris", "France"), ("Colosseum", "Rome", "Italy"),
    ("Brandenburg Gate", "Berlin", "Germany"), ("Kremlin", "Moscow", "Russia"),
    ("Acropolis", "Athens", "Greece"), ("Forbidden City", "Beijing", "China"),
    ("Prado Museum", "Madrid", "Spain"), ("Big Ben", "London", "England"),
    ("Charles Bridge", "Prague", "Czech"), ("Schönbrunn Palace", "Vienna", "Austria"),
    ("Rijksmuseum", "Amsterdam", "Netherlands"), ("Belém Tower", "Lisbon", "Portugal"),
    ("Hungarian Parliament Building", "Budapest", "Hungary"),
    ("Tokyo Skytree", "Tokyo", "Japan"), ("Egyptian Museum", "Cairo", "Egypt"),
    ("Gyeongbokgung Palace", "Seoul", "Korea"), ("Little Mermaid statue", "Copenhagen", "Denmark"),
    ("Royal Palace of Stockholm", "Stockholm", "Sweden"), ("Wawel Castle", "Krakow", "Poland"),
    ("Atomium", "Brussels", "Belgium"),
]
ok = []
for landmark, city, country in TABLE:
    c, k = one(" " + city), one(" " + country)
    print("table", landmark, city, c, country, k)
    if c is not None and k is not None:
        ok.append({"landmark": landmark, "city": city, "country": country, "city_id": c,
                   "country_id": k})
json.dump({"ids1": ids1, "ids2": ids2, "paris_row": paris_row, "cf": CF, "table": ok,
           "ids": {"Paris": Pa, "Rome": Ro, "France": Fr, "Italy": It}},
          open("prompts.json", "w"), indent=1)

A = f"M[{Pa},{Ro}]"
B = f"M[{Fr},{It}]"
s1 = [f"start {' '.join(map(str, ids1))}", "where", "until logits", "print logits top=5",
      f"explain {A}", f"circuit {A} top=4 depth=2",
      *[f"reads t{r} consumer={A} top=5 causal=false" for r in range(1, 5)],
      f"symbols consumer={A} limit=10"]
s2 = [f"start {' '.join(map(str, ids2))}", "where", "checkpoint pre"]
if CF:
    s2 += [f"run name=rome {CF}", "print logits top=3", "frame main:", "restart pre"]
    for layer in range(28):
        s2.append(f"whatif cache.L{layer}.v[row={paris_row}] = rome:cache.L{layer}.v[row={paris_row}] "
                  f"consumer={B}")
    s2.append(f"whatif cache.L14.k[row={paris_row}] = rome:cache.L14.k[row={paris_row}] consumer={B}")
s2 += ["until logits", "print logits top=5", f"explain {B}", f"circuit {B} top=4 depth=2",
       f"reads t{paris_row} consumer={B} top=8 causal=false",
       f"reads t{paris_row} consumer={B} top=4"]
open("recall.mdb", "w").write("\n".join(s1) + "\n")
open("followup.mdb", "w").write("\n".join(s2) + "\n")
