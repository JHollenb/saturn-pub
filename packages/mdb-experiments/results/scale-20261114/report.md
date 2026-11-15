### (a) Carrier per model

| Model | Layers | Carrier (L, KV) | L/depth | Tail | Mode over correct base | Carrier names country, correct base | Write window ≤5 layers, all KV (sealed) | Same, carrier KV only (post hoc) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | L21 KV0 | 0.88 | 3 | 20/28 | 15/28 | 5/28 | 17/28 |
| Qwen3-0.6B-Base | 28 | L20 KV6 | 0.71 | 8 | 30/30 | 28/30 | 1/30 | 25/30 |
| Qwen2.5-1.5B | 28 | L22 KV1 | 0.79 | 6 | 38/40 | 37/40 | 4/40 | 13/40 |
| Qwen3-1.7B-Base | 28 | L20 KV6 | 0.71 | 8 | 41/41 | 39/41 | 22/41 | 38/41 |
| Qwen2.5-3B | 36 | L27 KV1 | 0.75 | 9 | 31/42 | 32/42 | 0/42 | 11/42 |
| Qwen3-4B-Base | 36 | L24 KV6 | 0.67 | 12 | 42/42 | 41/42 | 12/42 | 36/42 |

### (b) Accuracy, hop budget, readable errors

| Model | d2 | d3-lm | d3-per | d4 | d5 (supp) | d6 (supp) | Slack AUC d3+d4 | Median L_first, correct d2 / d3-lm / d3-per / d4 (chance) | Readable-error share of wrong items, d4 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 14/14 | 9/14 | 9/14 | 3/14 | 7/28 | 7/25 | 0.48 | 16.5 (9.0) / 17 (14) / 17 (17) / 11 (13) | 11/11 |
| Qwen3-0.6B-Base | 14/14 | 11/14 | 10/14 | 5/14 | 6/28 | 5/25 | 0.66 | 1.5 (6.0) / 2 (8) / 3.0 (7.5) / 6 (5) | 7/9 |
| Qwen2.5-1.5B | 14/14 | 11/14 | 10/14 | 5/14 | 12/28 | 8/25 | 0.77 | 3.5 (21.5) / 16 (13) / 15.5 (20.0) / 16 (10) | 6/9 |
| Qwen3-1.7B-Base | 14/14 | 10/14 | 11/14 | 6/14 | 11/28 | 6/25 | 0.65 | 0.0 (3.5) / 4.0 (3.5) / 1 (5) / 1.0 (6.5) | 5/8 |
| Qwen2.5-3B | 14/14 | 14/14 | 13/14 | 9/14 | 10/28 | 6/25 | 0.61 | 10.0 (26.5) / 7.5 (25.0) / 19 (20) / 19 (18) | 1/5 |
| Qwen3-4B-Base | 14/14 | 12/14 | 12/14 | 9/14 | 13/28 | 7/25 | 0.64 | 0.0 (6.0) / 4.0 (1.0) / 3.5 (5.0) / 6 (3) | 3/5 |

### (b) Repair and downstream necessity

| Model | Back-patch repairs, wrong d3/d4, country late | Same, country never | B6 median margin drop, language (depth) | B6, capital (base) | Projected language names truth, depth | Shuffled-country chance |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0/13 | 1/8 | 0.38 | 0.84 | 9/56 | 30/56 |
| Qwen3-0.6B-Base | 0/14 | 0/2 | 0.75 | 2.84 | 34/56 | 40/56 |
| Qwen2.5-1.5B | 0/9 | 1/7 | 0.75 | 3.28 | 30/56 | 42/56 |
| Qwen3-1.7B-Base | 0/14 | 0/1 | 1.00 | 1.69 | 31/56 | 39/56 |
| Qwen2.5-3B | 0/6 | – | 0.44 | 0.47 | 17/56 | 44/56 |
| Qwen3-4B-Base | 1/8 | 0/1 | 0.31 | 0.62 | 40/56 | 51/56 |

### (c) Hop-2 split of the carrier transplant (correct base items)

| Model | Tail | MLP share, dose 0.5 median (IQR) | MLP share, dose 1.0 median (IQR) | Direct share, dose 1.0 | Re-read, dose 1.0 | Median dose move of the MLP share |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 3 | -0.03 (-0.27–0.46) | 0.10 (-0.08–0.41) | 0.91 | -0.00 | 0.10 |
| Qwen3-0.6B-Base | 8 | 0.61 (0.47–0.75) | 0.61 (0.53–0.75) | 0.41 | 0.04 | 0.04 |
| Qwen2.5-1.5B | 6 | 0.50 (0.39–0.65) | 0.62 (0.53–0.70) | 0.44 | -0.03 | 0.08 |
| Qwen3-1.7B-Base | 8 | 0.46 (0.27–0.65) | 0.60 (0.45–0.71) | 0.40 | 0.01 | 0.10 |
| Qwen2.5-3B | 9 | 0.66 (0.40–0.81) | 0.70 (0.55–0.79) | 0.31 | -0.03 | 0.05 |
| Qwen3-4B-Base | 12 | 0.58 (0.47–0.76) | 0.52 (0.46–0.60) | 0.17 | 0.33 | 0.12 |

### (d) Concentration and family stability (correct items, per family)

| Model | Carrier agreement: currency / person / landmark / city | Top cell over V band, median by base family | Top-two share | Top two adjacent, same row |
|---|---|---|---|---|
| Qwen2.5-0.5B | 5/7 / 7/9 / 4/7 / 4/5 | 0.37 / 0.39 / 0.36 / 0.44 | 0.52–0.70 | 9/28 |
| Qwen3-0.6B-Base | 5/5 / 9/9 / 7/7 / 9/9 | 0.33 / 0.46 / 0.48 / 0.65 | 0.49–0.77 | 4/30 |
| Qwen2.5-1.5B | 8/8 / 11/11 / 8/9 / 11/12 | 0.44 / 0.34 / 0.27 / 0.55 | 0.51–0.79 | 36/40 |
| Qwen3-1.7B-Base | 10/10 / 10/10 / 9/9 / 12/12 | 0.62 / 0.62 / 0.47 / 0.70 | 0.51–0.76 | 6/41 |
| Qwen2.5-3B | 8/10 / 7/11 / 7/9 / 9/12 | 0.45 / 0.44 / 0.39 / 0.47 | 0.54–0.70 | 4/42 |
| Qwen3-4B-Base | 11/11 / 11/11 / 8/8 / 12/12 | 0.24 / 0.34 / 0.35 / 0.54 | 0.35–0.63 | 26/42 |

### (e) Hard prompts: where the error is

| Model | dist acc | dist wrong: true carrier names the true country | dist wrong: distractor cell names its country | dist wrong: reader mass on distractor > true | dist wrong: repair by true cell | obsc acc | obsc wrong: readable error |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 4/12 | 5/8 | 6/8 | 0/8 | 1/8 | 3/12 | 3/9 |
| Qwen3-0.6B-Base | 4/12 | 7/8 | 8/8 | 4/8 | 1/8 | 10/12 | 0/2 |
| Qwen2.5-1.5B | 10/12 | 1/2 | 2/2 | 0/2 | 1/2 | 12/12 | – |
| Qwen3-1.7B-Base | 9/12 | 3/3 | 3/3 | 0/3 | 3/3 | 9/12 | 0/3 |
| Qwen2.5-3B | 11/12 | 1/1 | 1/1 | 0/1 | 1/1 | 11/12 | 0/1 |
| Qwen3-4B-Base | 12/12 | – | – | – | – | 12/12 | – |

### Forecasts (per base family: range over the four families)

| Model | First-order Spearman vs native | Shuffled 95th pct | Passive proposal hit@1 | First-order / native at the carrier | Replica re-forward / native | Replica null error, max logit |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.88–1.00 | 0.67–0.80 | 36/48 | 0.55–0.73 | 0.98–1.00 | 0.12–0.17 |
| Qwen3-0.6B-Base | 0.89–0.96 | 0.48–0.56 | 44/48 | 0.44–0.62 | 0.99–1.02 | 0.29–0.44 |
| Qwen2.5-1.5B | 0.91–0.99 | 0.63–0.72 | 33/48 | 0.43–0.56 | 1.01–1.02 | 0.27–0.65 |
| Qwen3-1.7B-Base | 0.80–0.96 | 0.68–0.77 | 48/48 | 0.45–0.58 | 0.98–1.00 | 0.27–0.39 |
| Qwen2.5-3B | 0.80–1.00 | 0.79–0.81 | 24/48 | 0.41–0.63 | 1.00–1.00 | 0.13–0.17 |
| Qwen3-4B-Base | 0.80–1.00 | 0.71–0.80 | 41/48 | 0.28–0.54 | 1.00–1.04 | 0.21–0.29 |

### Co-readouts at the carrier (capital, all base items; language, all depth items)

| Model | Capital raw | Capital, country projected out | Lexical null (country direction alone) | Capital counts (projected and not lexical) | Language raw | Language projected | Language lexical null | Language counts |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 19/48 | 10/48 | 48/48 | 0/48 | 30/56 | 9/56 | 44/56 | 1/56 |
| Qwen3-0.6B-Base | 37/48 | 31/48 | 48/48 | 0/48 | 40/56 | 34/56 | 48/56 | 5/56 |
| Qwen2.5-1.5B | 43/48 | 30/48 | 46/48 | 1/48 | 43/56 | 30/56 | 48/56 | 3/56 |
| Qwen3-1.7B-Base | 42/48 | 32/48 | 48/48 | 0/48 | 41/56 | 31/56 | 48/56 | 4/56 |
| Qwen2.5-3B | 38/48 | 19/48 | 48/48 | 0/48 | 44/56 | 17/56 | 52/56 | 4/56 |
| Qwen3-4B-Base | 45/48 | 40/48 | 46/48 | 0/48 | 51/56 | 40/56 | 48/56 | 7/56 |

### Site nulls (all items of the base families; truth = the item's country)

| Model | Carrier | Fixed last span row | Mass row | Random span row (expectation) | Shuffled cell | Cross-item | Lens at L_c | Lens at L_c+4 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 21/48 | 17/48 | 20/48 | 13.6/48 | 22/48 | 0/45 | 28/48 | – |
| Qwen3-0.6B-Base | 39/48 | 35/48 | 39/48 | 26.7/48 | 35/48 | 0/46 | 4/48 | 37/48 |
| Qwen2.5-1.5B | 43/48 | 40/48 | 43/48 | 26.7/48 | 43/48 | 0/46 | 7/48 | 44/48 |
| Qwen3-1.7B-Base | 44/48 | 40/48 | 44/48 | 30.0/48 | 44/48 | 0/45 | 6/48 | 45/48 |
| Qwen2.5-3B | 36/48 | 37/48 | 36/48 | 26.3/48 | 36/48 | 0/46 | 2/48 | 35/48 |
| Qwen3-4B-Base | 46/48 | 46/48 | 46/48 | 33.1/48 | 46/48 | 0/45 | 32/48 | 45/48 |

### Closure witness (base families)

| Model | Same-country mean cell keeps the answer | Donor-country mean cell moves the answer to the donor's |
|---|---|---|
| Qwen2.5-0.5B | 28/42 | 8/48 |
| Qwen3-0.6B-Base | 31/42 | 17/48 |
| Qwen2.5-1.5B | 39/42 | 15/48 |
| Qwen3-1.7B-Base | 39/42 | 30/48 |
| Qwen2.5-3B | 40/42 | 15/48 |
| Qwen3-4B-Base | 40/42 | 15/48 |

### Supplement: d5/d6 and native-name cities (carrier frozen from the main panel)

| Model | d5 | d5-short | d6 | native-city | english-city | Readable-error share of wrong items: d5 / d6 / native | Carrier names country, correct: native / english | Median L_first, correct d5 / d6 (chance) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 7/28 | 7/16 | 7/25 | 10/28 | 9/28 | 16/21 / 15/18 / 3/18 | 7/10 / 7/9 | 20 (22) / 11 (22) |
| Qwen3-0.6B-Base | 6/28 | 4/16 | 5/25 | 18/28 | 18/28 | 16/22 / 16/20 / 2/10 | 18/18 / 17/18 | 2.0 (1.0) / 2 (7) |
| Qwen2.5-1.5B | 12/28 | 9/16 | 8/25 | 25/28 | 22/28 | 10/16 / 16/17 / 0/3 | 25/25 / 22/22 | 12.5 (22.0) / 16.0 (23.0) |
| Qwen3-1.7B-Base | 11/28 | 9/16 | 6/25 | 22/28 | 15/28 | 12/17 / 18/19 / 0/6 | 22/22 / 15/15 | 6 (2) / 2.5 (5.5) |
| Qwen2.5-3B | 10/28 | 6/16 | 6/25 | 24/28 | 25/28 | 12/18 / 15/19 / 0/4 | 20/24 / 17/25 | 23.0 (13.5) / 24.5 (24.5) |
| Qwen3-4B-Base | 13/28 | 7/16 | 7/25 | 24/28 | 24/28 | 11/15 / 16/18 / 0/4 | 22/24 / 23/24 | 7 (8) / 0 (2) |
