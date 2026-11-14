"""Freeze the follow-up country vocabularies (tokenizer + das/out/bases.pt meta only; no model).
python gen_vocab.py TOKENIZER_JSON BASES_PT > vocab.json

full69 = the 11b readout-span vocabulary (das/out/bases.pt meta.country_tokens).
nontest41 = full69 minus the 28 test countries' tokens.
extras = single-token (' ' + name) sovereign-country names from the fixed candidate list below, not a test
country, not in full69, not a prefix-prompt country (France, Switzerland).
a69 = nontest41 + 28 extras drawn with random.Random(20261114).sample over the name-sorted extras.
aall = nontest41 + all extras.
"""
import json
import random
import sys

import torch
from tokenizers import Tokenizer

TEST = ["Japan", "Korea", "Russia", "China", "Thailand", "Poland", "Hungary", "Sweden", "Indonesia", "Israel",
        "Czech", "Britain", "Iran", "Afghanistan", "America", "Denmark", "Spain", "Italy", "Austria", "Greece",
        "Finland", "Norway", "Ireland", "Netherlands", "Egypt", "Australia", "Germany", "Portugal"]
PREFIX = {"France", "Switzerland"}
CAND = """Algeria Angola Armenia Bahamas Bahrain Belarus Belize Bolivia Cambodia Cameroon Chad Congo Cyprus Ecuador
Fiji Georgia Guatemala Guinea Haiti Honduras India Jordan Kuwait Laos Latvia Liberia Luxembourg Macedonia
Madagascar Mali Malta Mexico Monaco Mongolia Nicaragua Niger Oman Panama Qatar Rwanda Samoa Singapore Slovakia
Slovenia Sudan Tanzania Uruguay Vietnam Yemen Zambia Bhutan Botswana Brunei Burundi Comoros Djibouti Eritrea
Gabon Gambia Grenada Guyana Kiribati Kyrgyzstan Lesotho Malawi Maldives Mauritania Mauritius Moldova Montenegro
Namibia Nauru Palau Seychelles Suriname Tajikistan Togo Tonga Turkmenistan Tuvalu Vanuatu Benin""".split()

tok = Tokenizer.from_file(sys.argv[1])
one = lambda w: (lambda ids: ids[0] if len(ids) == 1 else None)(tok.encode(" " + w, add_special_tokens=False).ids)  # noqa: E731
full69 = torch.load(sys.argv[2], weights_only=False)["meta"]["country_tokens"]
test_ids = [one(c) for c in TEST]
assert None not in test_ids and set(test_ids) <= set(full69) and len(full69) == 69
nontest41 = sorted(set(full69) - set(test_ids))
extras = {}
for n in sorted(set(CAND) - set(TEST) - PREFIX):
    t = one(n)
    if t is not None and t not in full69 and t not in test_ids:
        extras[n] = t
names = sorted(extras)
pick = sorted(random.Random(20261114).sample(names, 28))
dec = lambda ids: [tok.decode([t]).strip() for t in ids]  # noqa: E731
out = {"test_countries": TEST, "test_ids": test_ids, "full69": full69, "full69_names": dec(full69),
       "nontest41": nontest41, "nontest41_names": dec(nontest41), "extras_all": extras,
       "a69_extra_names": pick, "a69": sorted(nontest41 + [extras[n] for n in pick]),
       "aall": sorted(nontest41 + list(extras.values()))}
out["a69_names"], out["aall_n"], out["a69_n"] = dec(out["a69"]), len(out["aall"]), len(out["a69"])
assert not set(out["a69"]) & set(test_ids) and not set(out["aall"]) & set(test_ids)
print(json.dumps(out, indent=1))
