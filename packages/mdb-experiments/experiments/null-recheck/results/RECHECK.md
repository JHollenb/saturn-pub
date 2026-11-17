# RECHECK: single-donor and zero-ablation nulls in the mdb record, re-run with 16 donors

2026-11-16. Model ID of the agent: claude-opus-5-5. Prereg `PREREG.md` (sealed guesses `GUESSES.txt`, sha256 `99029fc161ca80a13430f39c39ce9aae030d388e5a1816633be8f952e1fd87d6`, written before either job). Code: saturn-pub branch `null-recheck` (worktree `.claude/worktrees/null-recheck`), commits `69f6dba` (drivers, data, prereg; the jobs ran this revision) and `b4f74a6` (scorer, committed before any result was read), directory `packages/mdb-experiments/experiments/null-recheck/`. mdb is main b94fb8e, unchanged.

Labels: **MEASURED** = printed by a job below or by `score.py` on its outputs. **ASSERTED** = my reading.

## Jobs (all MEASURED)

| job | model | what | wall | peak VRAM / RSS |
|---|---|---|---|---|
| job-9809ef4f0c54 | Qwen3-1.7B-Base bf16 eager, `/mnt/big/llm-models/Qwen3-1.7B-Base` | C1–C6: 536,907 cell-sweep arms over 64 recipients × 16 donors, 320 TP1 lanes, 4 stepped cross-checks | 146.8 s | 5484 MB / 2333 MB (mrun) |
| job-343bbf1abad3 | Qwen2.5-1.5B bf16 eager, `/mnt/big/llm-models/Qwen2.5-1.5B` | C7–C11: 27 J2 recipients × 16 donors, 83 sweep arms + 48 residual lanes each, 4 stepped cross-checks | 22.7 s | 3942 MB / 2985 MB (mrun) |

Both priority 70, one model load, `--pin-reservation`, first attempt, no retries. Outputs: `out/69f6dba-q3/` (`q3_cells.npz`, `q3_summary.json`, job log), `out/69f6dba-q25/` (`q25_records.json`, job log), `out/score.json` (`score.py` on both).

Cross-checks (MEASURED). Q3: the batched K-cell arm equals stepped `Run.set_cell` exactly on 2/2 (0.0 and −0.25); the V-cell arm differs by 0.125 and 0.25 (−10.875 vs −10.75, −11.875 vs −11.625), one to two bf16 spacings, the known kernel-path gap. The largest batched null vs stepped native was 0.625 logit (Q3) and 0.19 (Q25); every dM is taken against its own chunk's null, so this offset cancels. Q25: mdb's multi-donor `donors.whatif` on the f4197 resample gave −0.125 and −0.25 (stepped, mdb label `effect` under mdb's own 1-spacing floor). Reproductions on the J2 panel: zeroing L22 h6 −3.87 (J2 −3.85), h7 −0.098 (J2 −0.07), f4197 −0.029 (J2 −0.04).

## Claim table (MEASURED, job ids above)

Unit share = units whose 16-donor pooled verdict (`mdb.analysis.donors.pooled`, effect if ≥ 1 donor exceeds the floor) is `effect`. Floor = 2 bf16 spacings at the larger \|logit\| + the null's spread across batches (≈ 0.25–0.375 logit here). Every donor patch in every claim was informative (bitwise different from the base), so no null below rests on an identity patch. "→ donor" = the answer becomes the donor's answer (the originals' flip metric).

| id | claim (tracker) | original | 16-donor result | floor units of effects: median / max; units with ≥ 4 floors | → donor | verdict |
|---|---|---|---|---|---|---|
| C1 | keys do not carry the bridge: K cell L20 KV6, carrier row (entry 1, T:43) | 0/123 flips, 1–2 donors | effect on 52/64 recipients; 368/1003 donor patches above floor; 30 of the 42 canonical-donor nulls overturned. Same-length donors only (no RoPE re-rotation): 47/64 | 2.3 / 10.7; 23/64 recipients | 0/1003 | **FLIPS**, mostly kernel-sensitive (median < 4 floors). Median \|dM\| 0.25 vs 8.75 for the V cell at the same site |
| C2 | K-only bands over the span from L0 / L8 / L16 / L20 do nothing (T:43) | 0/123 flips | effect on 64 / 64 / 64 / 63 of 64 | 3.0 / 35.7; 50, 52, 46, 42 of 64 | 0, 0, 1, 0 of 1024 | **FLIPS**: keys move M by up to 13.6 logits on some items (K cell alone: up to 4.0), but never transfer the answer |
| C3 | V band from L24 does nothing: "the read finishes between L20 and L23" (T:43) | 0/123 flips | effect on 32/64; 152/1024 patches | 1.7 / 4.5; 1/64 | 0/1024 | **FLIPS (kernel-sensitive)**. Median \|dM\| 0.125, against 13.6 for the band from L20 |
| C4 | the cell is written over a short L17–L21 window on the same row (T:236, T:305; entry 10 T:739) | 7/64 recipients have a live (\|dM\| > 0.5) gt-row V cell outside L17–L21 with the canonical donor (12/64 with either canonical donor) | 41/64 recipients. Layers: L16 26, L15 18, L22 7, L26 7, L10 6, L14 5, others ≤ 4. L15/L16 live cells are KV2 (89) and KV4 (34), not KV6 | outside the window: max \|dM\| 1.5 logits (6 floors); ≥ 4 floors on 2/64 recipients; outside L15–L22 none ≥ 4 floors. Inside the window up to 91 floors | – | **FLIPS** on the binary reading; the magnitude reading survives (ASSERTED: strong write L17–L21, weak live cells at L15/L16 on other KV heads) |
| C5 | 67% of span cells move M by exactly 0 (T:215, descriptive) | single donor | 644 of 38,753 canonical-donor null cells overturned (1.7%); any single donor as the reference: 1.8%. 867/38,976 cells have a pooled effect | – | – | **SURVIVES** (descriptive). Single V cells read by the answer row only are mostly real nulls, unlike the residual nulls of job-33fe7bcce697 (50%) |
| C6 | replacing t0 with its position mean costs nothing (entry 9, T:649, T:672) | mean ablation, median \|dM\| 0, 20/20 kept | 0/20 prompts, 0/320 donor patches above floor; 16/16 informative donors on each prompt (other first tokens); max 1.0 floor | – | – | **SURVIVES**. Position-mean arm reproduced (dM 0 on 20/20) |
| C7 | L22 KV1 key cell at the city row does nothing (entry 5 T:588; entry 4 T:562, post 2011 l.179/188) | −0.02 mean, 1 donor, 27 pairs | effect on 16/27; 35/432 patches; 14 of 25 canonical nulls overturned | 1.5 / 2.75; 0/27 | 0/432 | **FLIPS (kernel-sensitive)**; no patch reaches 4 floors |
| C8 | L22 KV0 value cell does nothing (T:588) | −0.00, 1 donor | 0/27, 0/432 patches | – | 0 | **SURVIVES** (16 donors, 16 informative on every recipient) |
| C9 | L22 h7 is not used (zeroed −0.07, T:588) | zero ablation | resampled from donors: effect on 27/27; 325/432 patches | 2.0 / 4.0; 0/27 | 0 | **FLIPS (kernel-sensitive)**: small (median \|dM\| 0.44) |
| C10 | "the L23 neuron [f4197] was specific to the Eiffel prompt" (T:588, T:592) | zero ablation −0.04 | resampled: effect on 18/27; 17/27 recipients with ≥ 4 floors. 16 of the 19 ≥ 4-floor patches come from the **Paris** donor: −1.9 to −3.7 logits (8–15 floors) on all 16 recipients that drew Paris. Zeroing it on the Paris recipient: −0.56 | 4.75 / 14.75; 17/27 | 0 | **FLIPS, real effect**. ASSERTED reading: f4197 is a Paris/France neuron that acts on any prompt carrying Paris; zero ablation only tested it where it is silent |
| C11 | city info has left its row by L26 (residual −7.3 at L22 → −0.06 at L26, T:584) | 1 donor, Eiffel prompt | J2 panel, h at the L26 input, city row: effect on 27/27; 189/432 patches. Median \|dM\| L22 15.9, L24 0.88, L26 0.31 | 1.8 / 7.75; 6/27 (up to −1.94 logits) | 0 | **FLIPS (kernel-sensitive)** as a null; "mostly gone by L24" holds as a magnitude statement (L26 ≈ 2% of L22) |

Counts: 11 claims re-run. **Survive 3** (C5, C6, C8). **Flip 8** (C1, C2, C3, C4, C7, C9, C10, C11): 1 with a large effect that changes a conclusion (C10), 2 with a sizeable minority of ≥ 4-floor effects but no answer transfer (C1, C2), 1 binary flip whose magnitude reading stands (C4), 4 within the kernel-sensitive band (C3, C7, C9, C11). **Not re-run 22**: 3 not testable by donors, 7 testable but deferred, 12 skipped (not a floor null, a control, already retracted, or already ≥ 8 donors). The full inventory is below.

Sealed guesses vs result (direction): C1 ✓ (.60 vs .81), C2 ✓, C3 ✗ (guessed survives), C4 ✓ (16-donor share .35 vs .64), C5 ✗ (guessed .30, measured .017), C6 ✓, C7 ✓ (.50 vs .59), C8 ✓, C9 ✗, C10 ✗, C11 ✗ (L24 info arm ✓).

## Per-entry verdicts

- **Entry 1 (latent two-hop bridge).** Headline "one value cell carries the bridge; values carry it, keys do not" **stands as a magnitude statement, not as a null**. Keys are not inert: the K cell moves M above the floor on 52/64 recipients (≥ 4 floors on 23), K bands on all 64 (up to ~36 floors). But across 5,099 key patches no key transferred the answer, and the K cell's median \|dM\| is ~1/35 of the V cell's. Replace "keys do nothing" / "the donor's keys, which do nothing alone" with "keys shift M by a few floors and never transfer the answer". "The read finishes between L20 and L23" stands in magnitude (V band from L24: max 4.5 floors, median \|dM\| 0.125 vs 13.6 from L20) but not as "0 effect". The "short L17–L21 write window" needs rewording: live cells (\|dM\| > 0.5) outside the window appear on 41/64 recipients with 16 donors (7/64 with the canonical donor), mostly L15/L16 on KV2/KV4, all ≤ 1.5 logits; the strong write is still L17–L21. Run 1 P1's zero-cell mass is real (C5: 1.7% overturned).
- **Entry 9 (token × position).** Headline "replacing t0 with its position mean costs nothing" **stands**, now with 16 informative donors per prompt (0/320 above floor).
- **Entry 10 (scale).** The 1.7B write window is C4 above: same rewording. The other five models' windows were not re-run (not yet known).
- **Entry 4 / entry 5 (recall circuit, interop).** "The entity variable is the L22 KV-head-1 value cell" **stands** (KV0 value survives with 16 donors; KV1 key cell effects stay < 4 floors). "**The L23 neuron was specific to the Eiffel prompt" does not stand**: resampled from the Paris prompt, f4197 moves the margin by 2–3.7 logits on every recipient that received Paris (MEASURED); it reads as a Paris/France unit (ASSERTED), and the 27-pair zero ablation was silent because the neuron is off for non-Paris cities. "City info leaves its row between L22 and L24" stands as a magnitude statement (median \|dM\| 15.9 → 0.88 → 0.31), not as a null at L26 (≥ 4 floors on 6/27). L22 h7: small, above-floor effects on all recipients (max 4.0 floors); "not used" should read "≤ 4 floors".
- **Entries 2, 3, 11, 12.** Not changed by this re-check (their nulls were not re-run or are not floor nulls; see the inventory).

## Inventory: null and "not used" claims in tracker entries 1–12 and their linked write-ups

T = `obsidian/research/2026-11-13-211248-mdb-experiment-tracker.md`. "Headline" = an entry's conclusion depends on the null. "Donors" = per recipient.

| # | claim (short) | file:line | model | intervention | donors | headline? | action |
|---|---|---|---|---|---|---|---|
| 1 | Keys do not carry the bridge: K-only single cell 0/123 flips, \|fraction\| 0.013 | T:43 | Qwen3-1.7B bf16 | single-cell donor transplant | 1–2 (123 pairs) | yes | re-run C1 |
| 2 | K-only bands from L0..L16 flip 0/123; "the donor's keys do nothing alone" | T:43 | same | band donor transplant | 1–2 | yes | re-run C2 |
| 3 | V bands from L24 flip 0/123 → "the read finishes between L20 and L23" | T:43 | same | band donor transplant | 1–2 | yes | re-run C3 |
| 4 | Cell written over a short L17–L21 window on the same row | T:236, T:305; entry 10 T:736–741, T:858 | same (entry 10: 6 models) | single-cell donor sweep, live = \|dM\| > 0.5 | 1–2 | yes | re-run C4 (1.7B) |
| 5 | 67% / 69% of span cells move M < 0.07, exactly 0 | T:215 | same | single-cell donor sweep | 1–2 | no (descriptive) | re-run C5 |
| 6 | Replacing t0 with its position mean costs nothing | T:649, T:672 | same | mean ablation at the L14 input | mean | yes (entry 9) | re-run C6 |
| 7 | L22 KV1 key cell −0.02; Eiffel K cell −0.06, "swapping the key does nothing measurable" | T:588, T:562; `obsidian/timestamps/2026/11/13/2011-mdb-review-against-mech-interp-tools-one-session-on-a-recall-prompt.md`:179, :188 | Qwen2.5-1.5B bf16 | single-cell donor transplant | 1 | yes | re-run C7 |
| 8 | L22 KV0 value cell −0.00 | T:588 | same | donor transplant | 1 | yes | re-run C8 |
| 9 | L22 h7 zeroed −0.07 | T:588 | same | zero ablation | 0 | supporting | re-run C9 (resample) |
| 10 | L23 f4197 zeroed −0.04 → "specific to the Eiffel prompt" | T:588, T:592 | same | zero ablation | 0 | yes | re-run C10 (resample) |
| 11 | City info leaves its row between L22 and L24 (−0.06 at L26) | T:584 | same, Eiffel prompt | residual donor patch (Rome) | 1, one prompt | yes | re-run C11 on the J2 panel |
| 12 | Run 1 P4 "no hidden relay 0/64" | T:245 | Qwen3-1.7B | relay-row transplant | 1 | already retracted by the audit | skip: every same-prefix donor is uninformative by construction |
| 13 | FT1: zero cut of t0 at the answer row leaves the default on 25/32; the sink is "not specially destructive" | T:184–187 | same | zero / mask cut | 0 | yes (FT1) | **not testable by donors**: every FT1 prompt starts with "The", so every donor's row 0 equals the base's; a donor with another first token changes the prompt |
| 14 | Z2 / Run 1: re-read share −0.01, never largest; norm 0.00 | T:115, T:256 | same | ladder accounting | – | supporting | skip: an accounting term, not an intervention null |
| 15 | Panel-mean cell flips 0/64 answers | T:126 | same | mean ablation | mean | no | skip: a flip count of a large measured effect, not a null |
| 16 | P8 random set and answer-token null median dM 0.0 | T:267 | same | control knock-outs | – | controls | skip: control arms |
| 17 | Entry 2: the swapped variable is in SAE features, error share 0.046 | T:403 | Qwen3-1.7B | residual-part swap | 1 (18 pairs) | yes | skip: a share of a large measured effect, not a floor null; the panel is not saved in the tree |
| 18 | Post 2112: key cell ≤ 0.375 "so the variable lives in the value cell" | `obsidian/timestamps/2026/11/13/2112-saes-meet-a-native-debugger-the-variable-is-in-the-features-the-computation-is-not.md`:82 | Qwen3-1.7B | single-cell donor | 1 (18 pairs) | yes (entry 2) | not re-run (panel not saved); nearest test C1. Not yet known |
| 19 | Entry 3: K alone takes direct reads off X in 6/67 ("values, not keys") | T:429 | R1-Distill-Qwen-1.5B FP32 | hint-row swap to the Y prompt | 1 | supporting (P9) | not re-run: at most 3 donors exist (the other letters) and it needs a third model load. Not yet known |
| 20 | Entry 3 B2: the accounting predicate P2 carries no signal | T:541 | same | AUROC of an accounting score | – | supporting | skip: not an intervention null |
| 21 | Entry 5 stack: 7/18 transcoder nodes no effect above floor | T:597, T:599 | Qwen3-1.7B | feature removal (zero) | 0 | no (headline is the 2× overstatement on above-floor nodes) | not re-run: needs the 18.8 GB transcoders. Not yet known |
| 22 | Entry 10: the band arm adds nothing over the single cell | T:991 | 0.6B / 1.7B / 4B fp32 replica | same-country mean cell | mean | supporting | skip: equal repair counts of two edits, not a no-effect verdict |
| 23 | Entry 10 B3: back-patch repairs 1/64 | T:867 | 6 models | self-patch | self | already narrowed | **not testable by donors** (a self-patch by design) |
| 24 | Entry 10 B6 / AUDIT-SCALE: "these MLPs are not necessary for the language answer" | T:874; `.worktrees/mdb-engine-refactor/packages/mdb/results/scale-20261114/AUDIT-SCALE.md`:164 | 6 models | answer-row MLP mean ablation | mean | yes (hop-3 location) | not re-run: a ratio of two non-zero drops, 6 model loads. Not yet known |
| 25 | Entry 10 write window on 0.5B, 0.6B, 1.5B, 3B, 4B | T:736–741 | 5 models | single-cell donor sweep | 1–2 | yes | not re-run (5 model loads). Not yet known; on 1.7B the binary window claim flipped (C4) |
| 26 | Entry 11 exp2 L4: component removal above floor only in L17–L21, "never at L15/16/19/21–23" | T:1097 | Qwen3-1.7B | component removal (zero) | 0 | yes (L4 holds) | not re-run as written (donor component values need the SAE). The whole KV6 cell at those layers at the carrier row is in C4's sweep; the live L15/L16 cells found there are KV2/KV4, not KV6 |
| 27 | Entry 11 exp4 G3: norm bracket below floor on 135/160; random nodes within floor 26/32 | T:1124, T:1126 | Qwen3-1.7B | frozen-arm brackets | – | supporting | **not testable by donors** (a difference of deterministic arms) |
| 28 | Entry 11 random-direction nulls (exp3 H3, exp5, exp7, 11b) | T:1114, T:1131, T:1151, T:1202 | Qwen3-1.7B | random controls | – | controls | skip: controls |
| 29 | Entry 11 exp1 W4: error-component swap below floor on 51/110 | T:1105 | Qwen3-1.7B | donor component swap | 1 | W4 already fails | skip: more donors can only strengthen the stated verdict |
| 30 | Entry 12: 1.7B flips "within noise", random push 4/9 | T:1229 | Qwen3-1.7B | dose + random push | – | yes | skip: a random-push control, not a donor null |
| 31 | Entry 12 arm B: true back-patch repairs rare, no arm raised the amplitude | T:1271–1273 | 3 models | back-patch, 16 (s, t) pairs + random | 16 | yes | skip: already ≥ 8 donors |
| 32 | Prior evidence cited by entry 1: K-only bands 0/120; "from L24 nothing"; "L22 KV0 V and KV1 K do nothing (≤ 0.005)" | `saturn/experiments/2026-10-14-latent-bridge-two-hop/FINDINGS.md`:34, :53, :63 | Qwen2.5-1.5B FP32 | donor transplants | 1 | prior evidence | not re-run (FP32, own panel); Qwen3 analogues C1–C3, bf16 Qwen2.5 analogues C7/C8 |
| 33 | 10-15 / 11-04: "hint rows not necessary"; zeroing hint rows 0/32 | `saturn/experiments/2026-10-15-cot-hint-read/FINDINGS.md`:69; `saturn/experiments/2026-11-04-memory-program-cascade/FINDINGS.md`:65 | R1-Distill | swap / zero | 1 / 0 | background for entry 3 | skip: both are already contradicted in their own FINDINGS (f = 0 caveat; 11-04 :147) |

No claim in entries 1–12 was found that used ≥ 8 donors except entry 12 arm B (#31).

## Deviations and limits

- C11 used the J2 27-pair panel, not the single Eiffel prompt (no same-length landmark donors exist for it). C1–C3 donors come from any length of the family, rows aligned from the end of the span; keys from another length are RoPE-rotated (one extra bf16 rounding). The same-length-only subset of C1 is reported (47/64) and agrees.
- Q25 donors are a fixed cyclic window of J2's order, so only 16 of 27 recipients drew Paris; C10's effect count depends on that composition (ASSERTED that a France-content donor is what matters; not tested with other French cities).
- The `answer_changed` field in `score.json` compares the arm's argmax with the answer token, not with the null lane's argmax; it is not used above.
- Single seed, one prompt set per model, bf16 kernel floor. The `analysis.donors.whatif` multi-donor path was used only for the f4197 cross-check; cell edits went through mdb's `cell_sweep` (`donors.whatif` refuses `cache.*` addresses: "unknown stop", MEASURED in the laptop smoke), with mdb's `informative` / `pooled` rules. That `donors.whatif` gap is worth a request (not filed here).
