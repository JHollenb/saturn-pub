# Audit: Z1 (symbol-table chain readout) and Z2 (hop-2 ladder)

Auditor: independent Opus agent, 2026-11-13 ~23:10–23:40 PST. Branch `mdb-twohop` at a0bc966, local only.
Claims under audit: tracker entry 1, lines 81–170 of
`obsidian/research/2026-11-13-211248-mdb-experiment-tracker.md`.

Everything below was recomputed from the saved jsonl with my own scripts in `ext/audit/`. I did not
reuse `z1_summary.py` or `z2_summary.py`.

- `audit/recompute_z1.py` writes `audit/z1-recompute.json`.
- `audit/recompute_z2.py` writes `audit/z2-recompute.json`. It also checks the ladder arithmetic and
  re-runs the shuffle null with 2000 fresh shuffles plus a within-family null.
- `audit/first_subword.py` writes `audit/first-subword.json`. It covers the first-subword collisions
  the coordinator asked about.
- `audit/job/trivial_baselines.py` ran as one CPU mrun job, **job-05b1230d9ed7**: priority 70,
  1.35 s, 0.7 GB RSS, 0 VRAM, no forward pass. The log is in `audit/job-05b1230d9ed7.log`. It
  computes the trivial readouts asked for in item 4.

Labels: **measured** means I recomputed it from data or a job. **Code-read** means I established it
by reading the source. **Asserted** means it is my judgment.

## 1. Custody

**Verdict: holds.**

Times come from the mrun server (`/api/jobs` `created_ts`, converted to PST). Revisions come from
the `== revision` line of each job log. Commit times come from git (measured).

| event | time | revision run |
|---|---|---|
| prereg_chain.json committed (48c8b8c) | 22:37:24 | |
| z1_summary.py committed (68caa9d) | 22:42:35 | |
| Z1 smokes job-86b4…/job-f774… created | 22:38:36 / 22:43:32 | |
| aborted Z1 job-73fd30fbb451 created | 22:44:10 | 7f5c886 |
| driver fix eff7dd7 (r0 = gt.json row, as the prereg defines) | 22:45:46 | |
| **Z1 job-1ee4fe8e9656 created** | 22:45:46 | eff7dd7 |
| prereg_hop.json committed (3701cc5) | 22:50:38 | |
| Z2 smoke job-82f15013d0b6 created | 22:50:39 | |
| 28dfc65 (input_linearity + z2_summary.py) | 22:52:26 | |
| **Z2 job-b7a73e1349c4 created** | 22:52:27 | 28dfc65 |

- Each prereg file has exactly one commit. Both job revisions descend from their prereg commit
  (`git merge-base --is-ancestor`, measured).
- **28dfc65 is diagnostics only (code-read from the raw diff).**
  - It adds `dn_true`, `den`, `da_true` and `da_den`, plus an `input_linearity` dict.
  - It replaces `dn = port(edited, layer, "n") - port(run, layer, "n")` with `dn = dn_true`, which
    is the same expression.
  - It adds the `z2_summary.py` scorer, committed before the full run. Every other number is
    unchanged.
- One small custody note. The Z1 driver was patched twice after smokes had produced output: the
  record bound, then the row choice after job-73fd showed 5–6 items. The row fix makes the driver
  follow the prereg text, so it does not move the goalposts.

## 2. Recomputed headlines

### Z1 (measured, all 64 items)

| claim | recomputed | verdict |
|---|---|---|
| readable@1 countries 63/64, all 41/64 | 63/64, 41/64 | holds |
| union misses: capital 14, own descriptor 8 | capital 14 (all own capital), entity 8 (all own descriptor), country 1 | holds |
| capitals 61/64, entities 13/64 | 61/64, 13/64 | holds |
| B_lens L24 58/27, L20 38/25 | 58/27, 38/25 | holds |
| B_mass 53/34, site = bridge 51/64 | 53/34, 51/64 | holds |
| B_answer 60/8 | 60/8 | holds (the caveat is stated correctly) |
| content moves 61, answer 35, together 36 | 61/35/36; 2×2 (T,T)=34, (T,F)=27, (F,F)=2, (F,T)=1 | holds |
| 27 pairs: margin 9.0 → 1.75, 23 hold sign | 27 pairs, median M0 9.0 → M1 1.75, 23 with M1>0. The other 4 go to a third token (Edinburgh, Stockholm, Milan, Barcelona) | holds |
| T_city 12/12, 8/12, answer 2/12, together 6/12 | 12/12, 8/12, 2/12 (Osaka, Shanghai), 6/12 | holds |
| r0rest names country 49/64, first L median 7, ≤L20 49/49 | 49/64, median 7 (IQR 3–11), 49/49 | holds |
| hop native = job2 64/64, max diff 0 | 64/64, 0.0 | holds |
| direct share median 0.45, indirect 0.55 (IQR 0.46–0.65) | 0.453 / 0.547 (0.461–0.652) | holds |

### Z2 (measured)

| claim | recomputed | verdict |
|---|---|---|
| closure 64/64 donor, 64/64 ablate, 12/12 city | 64/64, 64/64, 12/12 | holds, but this is a bookkeeping identity (see §3) |
| median \|gap\| 0.053, residual-add 0.012 | 0.0526, 0.0120 (max 0.047) | holds |
| shares MLP 0.55 (0.48–0.66), direct 0.45 (0.35–0.54), reread −0.01, norm 0.00 | 0.553 (0.475–0.660), 0.453 (0.348–0.539), −0.010, −0.000 | holds |
| largest term MLP 44, direct 20 | 44 / 20 | holds |
| per-layer MLP L21–24 −1.67, −2.72, −1.65, −0.89; L26–27 +1.36 | −1.67, −2.71, −1.65, −0.89; +1.36, +1.36 | holds |
| 40 flips; direct+MLP 2/64 mismatches vs direct 37/64 | 40; 2 vs 37. MLP alone gives 29; "no change" gives 40 | numbers hold; the "predicts" framing is overstated (§3) |
| Spearman MLP 0.87, direct 0.82 | 0.875, 0.824 | holds |
| top-8 share median 0.23 (0.16–0.36); 45/64 items with a layer ≥0.5 | 0.234 (0.161–0.357); 45 | holds |
| recurrence 221 vs null 71 (ablate); 171 vs 54 (donor) | 221 vs 70.4 (2000 fresh shuffles; max null 115; 0 ≥ obs). Donor 171 vs 53.0 (max 94). Shuffling within family gives the same (69.9 / 52.1) | count holds; the interpretation is partial (§3) |
| linear share 0.25 donor, 0.49 ablate | 0.248 (IQR −0.07–0.46), 0.491 | number holds; the inference is overstated (§3) |
| ablate: direct 0.73, MLP 0.56, reread −0.19, 0 flips, against-unbound −0.34 | 0.728, 0.565, −0.192, 0, −0.341 | holds |
| channel-rfft 0.0780 vs 0.0779, paired diff 4e-5, energy 0.139/0.138 | 0.0780 / 0.0779, 4.0e-5, 0.1392 / 0.1384 | holds |
| neuron basis 0.61, 512/512, Parseval 7e-15 | 0.614, 512/512, 7.1e-15 | holds, with a caveat on comparability (§3) |
| table hop equals direct call 64/64 | 64/64 | holds |

Internal consistency checks (measured):

- The ladder rebuilt from its saved parts equals the saved gap on every record.
- `native_delta` equals (M1 − M0) from the saved logit pairs on 64/64, 64/64 and 12/12.
- On every listed neuron, `b0Δs + s0Δb + ΔsΔb == Δf` with relative error 0.
- The top-8 plus omitted effect equals the exact MLP term on every layer.
- `upstream` is exactly 0 everywhere.

## 3. Code correctness

All findings in this section are code-read and test-checked.

### `content` / `readout_basis`

**Verdict: correct, no leakage.**

- The write is `Σ_h p[h, r0] · W_O,h · v[κ(h), r0]`. `HeadMix.unit = einsum(o_proj block_h, v[κ(h)])`
  and `o_parts = p · unit`. It is summed only over reads whose value cell the binding holds; for
  the bridge that is L20 KV6, which means heads 12 and 13.
- The scores are `ρ · (W_U[t] ⊙ γ_final) · write` in float64. ρ is a positive scalar shared by all
  candidates, so it cannot change any rank.
- The direction comes entirely from the descriptor row's v. That row is causally before the answer
  row. The answer row enters only through the two scalars p[12, r0] and p[13, r0].
- No candidate country appears in the prompt: the few-shot lines name Bern and Paris, not their
  countries. There is no route from the consumer, the answer row or future rows.
- The unit test compares against a hand projection and passes.

### `hop_edge` closure

**Verdict: correct, and it closes by construction.**

- The ladder sums the captured Δo and Δm at the consumer row. These telescope to c·Δh_out.
- `residual_add_rounding` is defined as the plug `c·Δh_out − Σterms`. It is small: median 0.012,
  max 0.047.
- The norm term `(ρ1 − ρ0)/ρ0 · c·h1` is the exact finite RMS identity. So the final `gap` is only
  the bf16 final-norm and lm_head rounding of the two logits.
- The remainder does not absorb a real error. The plug and the gap are both at rounding scale, and
  the per-neuron identity and the top+omitted split are exact.
- But "closes on 64/64" is a bookkeeping check, as the prereg itself says for H1. It is not a
  finding about the model.

### SwiGLU identity

**Verdict: correct.**

- `(s0 + Δs)(b0 + Δb) − s0b0 = b0Δs + s0Δb + ΔsΔb`, computed on the captured `sg = silu(g)` and
  `u`.
- The effect is `(cᵀW_down)_i Δf_i`.
- `exact − captured` is the bf16 down_proj rounding: median 0.010 per item.

### Linear term

**Verdict: correct.**

- The RMSNorm Jacobian `γ(ρΔr − ρ³/d (r·Δr) r)` and `silu′(g0)·b0·GΔn + s0·UΔn` are right.
- `cos(Δn_lin, Δn_true)` has a median of 0.985–0.996 per layer (measured), so the norm is close to
  linear. The nonlinearity is in the SwiGLU.
- The tiny-edit test passes on qwen3-fp32 and is skipped on bf16, as designed.

### Shuffled null

**Verdict: fair as a label permutation; the interpretation is partial.**

- The shuffle keeps group sizes, and a within-family shuffle gives the same null. Family therefore
  does not explain the excess (measured).
- **There is a confound that cannot be removed on this panel.** Neurons are ranked by
  `|(cᵀW_down)_i Δf_i|`, and the consumer `c ∝ W_U[answer] − W_U[donor-0 answer]`. Items of the same
  country share the answer token, the capital, and country and capital are 1:1 here. Neurons with
  large `cᵀW_down_i` for that capital will therefore recur within a country even if Δf did not
  depend on the content.
- So "country-specific recurring neurons" is supported. "Neurons that carry the bridge content" is
  not separated from "neurons that write this capital's logit".
- The saved data cannot settle this, because Δf is stored only for the top 8 neurons. A ranking by
  |Δf| alone, or a panel with a shared capital across different bridges, would separate the two.

### Permuted-channel control

**Verdict: fair; the result was expected.**

- The same random permutation is applied to the gate rows and to Δn, which preserves `gate @ Δn`.
- Residual channel order has no meaning, so a null result was the expected outcome, as the prereg
  says.
- The neuron-versus-bin comparison is not like for like. Neuron share is the top-192 share of
  |effect|. Bin share is the top-32 share of Σ_i|effect_i|·|part_ik|, and taking the abs of each
  Parseval part spreads the mass.
- "Beats on 512/512" is true as computed. It is weaker evidence than it sounds that the neuron basis
  is more concentrated.

### Flip "prediction"

**Verdict: accounting, not a forecast.**

- direct + MLP are the measured terms from the edited run. Together with reread, norm and rounding
  they equal the native delta by construction. With all terms, the mismatch is 0 (measured).
- "2/64 mismatches" therefore means reread + norm rarely decide the sign.
- That makes the tracker's wording overstated in two places: "direct + MLP predicts which
  transplants flip the answer on 62/64", and in Z1, "Z2's flip prediction … shows the margin itself
  is tracked almost exactly". The number is right; the word "predicts" is not.

### Linear share 0.25 → "cannot be forecast passively at dose 1"

**Verdict: overstated.**

- 0.25 is the median of Σlinear / Σexact. The exact MLP sum has internal cancellation: L21–24 push
  one way and L26–27 push the other, so the ratio of sums is unstable (IQR −0.07 to 0.46).
- Per layer, the median linear/exact ratio on the donor arm is 0.47, 0.38, 0.52, 0.62, 0.65, 0.92,
  0.94, 0.86 for L20–27 (measured).
- Also, "linear" is evaluated on the actual measured Δr, so it is not a passive forecast in either
  case.
- Supported: the SwiGLU is materially nonlinear at L20–24 for a full-cell swap. Not supported: the
  hop "cannot be forecast passively". That was not tested.

## 4. Baseline fairness

- **The lens comparison is like for like (code-read).** Both use the same row r0 (a truncated run
  equals the full run at r0 by causality), the same four candidate sets, the same truth tokens and
  the same γ_final·W_U. The ρ difference does not affect ranks.
- **The cleanest comparison is L20.** `v = W_V·RMSNorm(h_L20[r0])`, so the view is the OV image of
  exactly the residual that B_lens L20 reads raw.
  - View 63/64 against lens L20 38/64 on countries; 41 against 25 on all. The OV map extracts the
    country far better than the same residual does.
  - Against lens L24 the view wins 6 discordant items to 1 on countries (sign test p≈0.13, not
    significant) and 25 to 11 on all (p≈0.03) (measured).
  - "F_lens did not fire" holds as worded. The countries-only margin over L24 is not significant.
- **Trivial readouts (job-05b1230d9ed7, measured).** Readable@1 out of 64, on countries / all:

  | readout | countries | all |
  |---|---|---|
  | descriptor-token embedding at r0 | 25 | 0 |
  | descriptor first token | 23 | 0 |
  | mean span embedding | 27 | 0 |
  | L20 KV6 OV of the r0 embedding (no context) | 20 | 6 |
  | the same OV on the span mean | 26 | 9 |

  - Chance on countries is 1/42 per item. The constant "China" guess gets 7/64.
  - "Readable@1 countries" is therefore not trivial (trivial readouts reach about 0.31–0.42). It is
    easy at later layers, though: lens L24 gets 58/64. Single-token cities are near-trivial; the
    embedding alone names the country for most of them.
  - The view's 63/64, and especially 41/64 on the union set, is well above every context-free
    readout. The cell content is computed by context, not a static token association.
- **Countries versus capitals.** The same cell ranks the true capital first within capitals on
  61/64, and the union's best token is the capital on 14/64.
  - The readout cannot claim the cell holds "the country, not the capital". Z10 failed, and the
    tracker reports that.
  - The summary line "the bridge cell's content (the right country on 63/64)" should carry that
    caveat.
- **T_diff content_moves 61/64 is largely by construction.** After the swap the cell is the donor's
  cell. Among the 53 donors that are themselves panel items, the donor's own native cell names its
  country on 52, and content moves on 51 of those.
  - So content_moves mostly re-measures readability on the donor's cell, read with the recipient's
    two p scalars. It is not independent causal evidence.
  - The same applies to T_city "country unchanged 12/12", because the donor is from the same country.
  - Also, the native cell names its own city on only 7 of the 12 T_city items, so the baseline for
    "entity moves 8/12" is weak.

## 5. Coordinator add-ons

### (1) First-subword confound

**Verdict: the headline holds on the clean subset (measured; tokenizer = the Qwen3-1.7B-Base
`tokenizer.json`).**

- All 42 countries and all 42 capitals are single tokens with unique ids, and no token is shared
  across sets.
- The only collisions are among entity first tokens:
  - `" the"` is shared by 16 landmarks.
  - `" r"` by rial, rouble and rupiah.
  - `" Mount"`, `" S"` (Socrates, Soren), `" Go"` (Goethe, Gothenburg), `" C"`, `" K"`, `" W"`,
    `" F"` and `" Pablo"` are each shared by two.
  - There is no `" New"`, `" San"` or `" South"` in these sets.
- Restricted to the 47 items whose country, answer and entity first tokens are all unique:

  | readout | countries | all |
  |---|---|---|
  | view | 46/47 (0.98) | 31/47 (0.66) |
  | lens L24 | 42 | 20 |
  | B_mass | 37 | 25 |
  | B_answer | 43 | 6 |

  - Entities stay at 13/47. All 13 entity hits were on unique items.
  - T_diff on these items: content 45, answer 28, together 28 of 47.
- The countries and all claims hold, and so does the entities weakness. Because collisions exist
  only on the entity side, they could only affect the entities set and the entity competitors in
  "all".

### (2) Sink row t0

**Verdict: no reported number depends on t0 as a site (code-read and measured).**

- Bridge, r0rest and desc are declared on span rows only. Z1 and Z2 call the table with
  `limit = #declared`, so token-row bindings, t0 included, are never listed. Their mass sits in
  Unbound, which no headline uses.
- B_mass searches `item["span"]` rows only, so t0 cannot be the "most-attended site". The 9 B_mass
  sites at L2 are span rows.
- The hop's direct term is the cell layer's Δo, and only the edited cell's heads change there.
- The string "t0" appears in neither `z1-full.jsonl` nor `z2-full.jsonl`, and y2_view also restricts
  to span rows. The "t0 The, mass 210, to_consumer 0.34" figures come from a full, unrestricted
  table, not from these jobs.
- Where t0 does enter, implicitly:
  - The B_answer and B_lens residuals, which contain everything, t0's reads included.
  - Z2's reread and MLP terms, since attention to t0 at L21–27 shifts when the query changes.
- t0 cannot be excluded from those terms without breaking the exact sum, and the saved records do
  not split it out. Excluding it is not a well-posed change to any reported number.

## 6. Tests

`tests/test_analysis_symbols.py`: **18 passed, 1 skipped** (measured). The skip is the bf16 variant
of the tiny-edit test, `needs fp32`; it passes on qwen3-fp32.

`uv run pytest` inside `packages/mdb` does not resolve, because saturn-pub is not on the registry,
and from the repo root it fails on the pyenv version. I used
`PYTHONPATH=packages/mdb/src:src common exec python -m pytest`. I removed the `.venv` directories
that the failed `uv run` attempts created.

## 7. Verdict per claim

**Holds:**
- Custody of both jobs, and 28dfc65 being diagnostics only.
- Z1 readable 63/64 and 41/64, including on the 47 first-token-unique items.
- All baselines as numbers. The lens comparison is like for like.
- Lifetime 49/64, median L7.
- The hop indirect share of 0.55.
- The T_diff and T_city counts.
- The 27-pair margin claim.
- Z1/Z2 verdict lists as computed: Z3, Z4, Z6, Z7 and Z10 failed; H3 and H8 failed; F_causal fired.
- Z2 shares, per-layer pattern, Spearman, concentration, ablate numbers and spectral numbers.

**Holds with correction:**
- *Closure 64/64.* True, but it is an identity check, not evidence about the model.
- *Recurrence 221 vs 71 (about 3×).* Reproduced. It is confounded with the shared answer token
  (the consumer), so content-specific neurons are not established.
- *Neuron basis beats channel bins 512/512.* The two quantities are not commensurate.
- *T_diff content-moves 61/64.* Largely the donor cell's own readability, not independent evidence.
- *"Right country on 63/64".* The same cell names the capital on 61/64.

**Overstated:**
- "direct + MLP predicts the flip set (62/64)" and "Z2's flip prediction shows the margin is
  tracked". This is accounting of measured terms; no forecast was made.
- "The hop cannot be forecast passively at dose 1". This rests on a ratio of sums with cancellation.
  The per-layer linear share is 0.38–0.94, and no passive forecast was tested.

**Wrong:** none found.

**Supported positives:**
- The OV readout of the declared cell names the bridge country well above the same-row lens at L20
  and above every context-free readout.
- Hop 2 splits about half direct, half L21–24 MLP, with L26–27 pushing back. This is exact and
  measured.
- The channel spectrum adds nothing over a permuted channel order.

**Partial evidence:**
- Neuron-level, content-specific recurrence.
- Whether the cell carries country rather than capital.
