"""Compose prereg_scale.json: memo §1, §3 and §5 copied verbatim from the advisor memo, plus the
builder's operational definitions, deviations and custody. python make_prereg.py MEMO > prereg_scale.json"""
import hashlib
import json
import re
import sys

memo = open(sys.argv[1]).read()


def section(title_re, next_re):
    m = re.search(title_re, memo, re.M)
    n = re.search(next_re, memo[m.end():], re.M)
    return memo[m.start(): m.end() + (n.start() if n else len(memo))].strip()


panel_sha = hashlib.sha256(open("panel.json", "rb").read()).hexdigest()
doc = {
    "name": "Scaling the latent chain across sizes, hops and hardness (tracker entry 10)",
    "memo": "obsidian/model-math/ADVISOR-2026-11-14-scaling-hops-limits.md",
    "memo_sha256": hashlib.sha256(memo.encode()).hexdigest(),
    "sealed": "2026-11-14, before any capture of the panel on any model (Job C) is run",
    "panel": {"file": "panel.json", "sha256": panel_sha, "builder": "panel.py",
              "counts": {"currency": 12, "person": 12, "landmark": 12, "city": 12, "d2": 14, "d3-lm": 14,
                         "d3-per": 14, "d4-work": 14, "dist-city": 12, "obsc-city": 12},
              "answers": "base and hard: the capital's first token; depth: the official language's first token",
              "candidate_sets": "panel.json sets (countries 57, cities 43, languages 26, surnames 37, given 36), frozen; identical token ids for Qwen3 and Qwen2.5 (tokenizers agree 636/636, measured job-76c1b04dd5b1)"},
    "models": {"grid": ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"],
               "skipped": {"Qwen3-8B-Base": "bf16 weights 16.4 GB > 16,376 MiB card; mdb load_local has no paging, offload or device_map (measured: source read 2026-11-14)",
                           "Qwen2.5-7B": "bf16 weights 15.2 GB; same reason"},
               "dtype": "bf16 on cuda (Beast), forecasts fp32 on CPU",
               "note": "Qwen3-1.7B-Base is re-run on this panel with the same pipeline (the 1c captures hold only the depth items, and the methods differ); its earlier numbers stay as the reference"},
    "memo_section_1_verbatim": section(r"^## 1\. Hypotheses", r"^## 2\. "),
    "memo_section_3_verbatim": section(r"^## 3\. Metrics", r"^## 4\. "),
    "memo_section_5_verbatim": section(r"^## 5\. Falsifiers", r"^## 6\. "),
    "already_known_before_this_prereg": {
        "hop_budget_1.7B": "prereg_hopbudget.json, job-6e384b3f8297: B1 not judged (d3/d4 chance-dominated: real median 3/3 vs chance 3/4; d2 0), B2 locus gap 3 (pass) and slack AUC 0.5995 (falsifier < 0.6 fires); secondary L_first_stable AUC 0.80 (not judged)",
        "1.7B_accuracy_on_the_source_items": "d1 22/22, d2 22/22, d3-lm 39/43, d3-per 32/38, d4-work 12/29 (B1); base 64/64 and hard families from Y2",
        "accuracy_of_the_panel": "scored from Job C (answer argmax only) and appended as prereg_scale_accuracy.json before any readout or arm is opened"},
    "operational": {
        "M": "logit(item answer) - logit(donor answer) at the answer row; donor = panel.json item.donor",
        "floor": "2 bf16 spacings at the larger |logit| of the pair: 2 * 2^(floor(log2 max|logit|) - 7)",
        "single_cell_sweep": "Job A1: cell_sweep over every (span row, layer, kv) V cell, value replaced by the donor's V at the aligned donor row (aligned from the span end), dose 1; dM = M_arm - M_null of the same chunk",
        "item_carrier": "argmax over swept cells of -dM (the cell whose donor value most lowers M); reported only when |dM| > floor",
        "model_carrier": "modal (layer, kv) of item carriers over correct base two-hop items (48 base items; correct = argmax == answer); row stays item-specific (the item carrier's row)",
        "carrier_at_item": "for content statements on any item (correct or wrong): the model carrier (layer, kv) at row r* = argmax over span rows of -dM at that (layer, kv)",
        "rule_A1_A2": "judged on the Qwen3-4B-Base and Qwen2.5-3B model carriers' layers, as memo §1(a)",
        "mode_coverage": "share of correct base items whose item carrier equals the model carrier (layer, kv)",
        "write_window": "at the item carrier row: layers with any kv cell |dM| > floor; span = max - min + 1; share of correct base items with span <= 5",
        "accuracy_bands": "memo §1(b) table; d3 pools d3-lm and d3-per; a model's band is hit if the accuracy lies inside",
        "L_first": "as prereg_hopbudget.json with L_c = the model carrier layer and the carrier row = carrier_at_item row; chance guard as there; L_first_stable reported beside",
        "B3_backpatch": "wrong d3/d4 items: Run.set t<r>.L<L_c-4>.h := native t<r>.L<L_c+4>.h at r = carrier_at_item row (stepped run, prefill=span[0]); repaired = answer argmax becomes the answer; split late (L_first present) vs never (absent by L_c)",
        "B4": "readable error = the model carrier at carrier_at_item names a wrong country or ranks the true one > 3 of the 57 countries",
        "B6_mlp_ablation": "answer row m at layers L_c..L_c+2 replaced by the mean of the same port over the other panel items of the same block (self-excluded); margin = logit(answer) - max logit over the rest of its candidate set (languages for depth, cities for base); drop = native - ablated; prediction: median language drop >= 2 x median capital drop",
        "readout": "gamma_F * W_U over the candidate set on the cell write W_g v, v the cached value (for Qwen2.5 the cache already holds W_V x + b_V, so no extra term); the residual-port functional cell_readout is affine on Qwen2.5 (c . h + d . W_g b_V, commit 68080ff, tested) and the constant cancels in every hop_edge difference",
        "co_readouts": "capital (cities) and language at the carrier: raw; with u_c = gamma_F * W_U[country] projected out of the write AND out of every candidate direction (ch 23 rank-one whitening); lexical null = u_c alone as the write; a co-readout counts only on items where the projected readout names the truth AND the lexical null does not",
        "projected_language_chance": "the item's country replaced by another item's country (derangement, random.Random(0)) before projecting; the projected-language hit count is reported beside its shuffled count",
        "proposal": "passive P05 score per span cell from Job C: sum over the group's query heads h of p_h(r) * d . W_O,h (v_donor - v) at the answer row, d = gamma_F * (W_U[a] - W_U[b]); hit@1 / hit@3 against the item carrier",
        "row_rules": "R_argmax (reader heads = the model carrier group's query heads at L_c), R_last, R_first, S(r) = sum_h p_h(r) * country margin of the carrier cell at r (memo §4.1); each credited only on items where the rules differ; per family",
        "effect_size": "Spearman(forecast, native dM) over swept cells with |dM| > floor, per item then median, beside the 95th percentile of the same statistic with the forecast shuffled across cells (200 shuffles, Random(0)); ratio forecast/native median and IQR",
        "site_nulls": ["fixed site: the model carrier (layer, kv) at the last span row", "mass: the span-or-prefix row with the largest non-sink answer-row attention mass at L_c (t0 excluded, renormalised), model carrier kv", "random span row: exact expectation over span rows at the model carrier (layer, kv)", "shuffled cell: (layer, kv) drawn from the model's own item-carrier distribution (Random(0)) at the carrier_at_item row", "cross-item: the same (row index from span end, layer, kv) in another item of the family with another country must not name this item's country", "lens: answer row h at L_c and at L_c + 4 through the final norm and W_U over countries"],
        "concentration": "item carrier |dM| / |dM of the V-band arm| (all span rows, all layers, all kv replaced by the donor's aligned V); median and IQR; Qwen2.5 also top-two",
        "closure": "same-country mean cell (mean of the model carrier cell at carrier_at_item rows over other panel items with the same country, any block) at the item's carrier keeps the answer; another country's mean cell (the donor's country) moves the argmax to that country's answer token (capital for base)",
        "split_c": "hop_edge(neurons=False) of the donor transplant at the item carrier at doses 0.5 and 1.0 (v + alpha (v_d - v)), consumer M, correct base items; MLP share = totals.mlp / native_delta; median and IQR per model and dose",
        "e_routing": "dist-city wrong items: true carrier names the true country; distractor rows' cell at the model carrier (layer, kv) names the distractor's country; reader mass (non-sink, renormalised, carrier group heads at L_c) on the distractor rows > on carrier_at_item row; arm: the true carrier cell value written into the distractor rows' carrier cells (repair rate)",
        "t0": "all panel prompts begin with the same token, so t0's K and V are the same on every item: the position-mean replacement is the identity and its falsifier cannot fire here; every mass statistic is the non-sink share renormalised over rows >= 1",
        "forecasts_sealed_between_C_and_A": "Job F (CPU fp32 replica from the capture): (i) first-order dM = grad_V M . (v_d - v) for every span cell (one backward per item); (ii) replica re-forward of the answer row alone with the edited cell and every other K/V fixed, for the top 3 cells by the first-order term and for the passive proposal's top cell; plus the replica's null error against the captured logits",
        "per_family_only": "every statistic per (model, family); cells with n < 8 reported, not judged; no pooled verdict"},
    "deviations_from_memo": [
        "Panel hard block: dist-city 12 and obsc-city 12 (the memo's dist/obsc); para, long and lang are not in the 128-item panel, so their (e) bullets are not tested in this grid",
        "The dist family's distractor is a city of another country (gen_ext.py), not a clause naming the country; distractor country = that city's country",
        "Depth stops at d4 (memo panel). d5/d6 chains are being drafted and fact-checked; if verified they run as a separate supplement panel with its own prereg addendum, never mixed into this panel's verdicts",
        "§4.2 per-hop slots: the single-cell sweep over all span cells below L_c (donor = the panel donor) types cells by content (person / city / country, rank 1 in its set) and by causal |dM|; the cell-to-carrier consumer edge per cell is not run (budget) and is reported as not measured",
        "Mask cut of the distractor cell (memo §2 Job A vi) is replaced by the true-carrier write into the distractor rows (no mask edit exists in the engine API used here)",
        "hop_edge runs with neurons=False (layer terms only), which leaves the shares unchanged (test_hop_edge_cell_consumer_closes_on_the_cells_native_change)"],
    "custody": "prereg before Job C; accuracy addendum after Job C before any readout; forecasts sealed between Job C and Job A; job logs print ids and seconds only; per model one capture job and arm shards, each <= 10 min, priority 70"}
json.dump(doc, sys.stdout, indent=1)
