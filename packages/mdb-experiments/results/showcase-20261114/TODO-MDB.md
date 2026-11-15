# Where mdb should be used

Generated from the `TODO(mdb)` / `TODO(mdb-feature P#)` comments added 2026-11-14 from `obsidian/reviews/2026-11-14-mdb-library-experiments-gap-audit.md`. Line numbers refer to the annotated files. Tag `mdb` = an existing mdb API replaces hand-rolled code; `mdb-feature P#` = a proposed mdb feature (P1 branches, P2 core files, P3 transcoder FeatureBasis, P4 holds, P5 selectors/origins, P6 pools, P7 Python each, P8 record hashes, P9 feature strips, P10 suspect lists, P11 thin accessors, Q2.x = stored columns).

## common.py (7)

- `common.py:23` [mdb-feature P7] prompt set with roles/vars/pairs (@carrier, $answer, $foil) loaded once — 14 of 14 drivers re-read items/sets/score JSON by hand and re-index them.
- `common.py:46` [mdb] use cm.slot_rows with a Slot whose layers/kv come from the map's model entry instead of a per-script carrier-row wrapper (gap audit §1 carrier-row row).
- `common.py:47` [mdb-feature P11] Slot taking layers/kv per model — exp6 re-implemented slot_rows as crow_of because Slot hard-codes layer and KV.
- `common.py:63` [mdb] use mdb.analysis.verdict.spacing (analysis/verdict.py:59) instead of a hand-rolled bf16 spacing (gap audit §1 bf16 spacing row).
- `common.py:73` [mdb] use supplied.judge(floor_spacings=, extra_floor=) / verdict.spacing instead of floor_of (gap audit §1 bf16 spacing row, P1 cheap part).
- `common.py:74` [mdb-feature P1] Branches record dM, floor in spacings and in floors per branch — every driver attaches the floor by hand.
- `common.py:82` [mdb-feature P8] dbg.records_to(path) / attached project store so every seal, verify and judge appends its full record, plus script_sha256 in procedure — results are dumped as a…

## exp1/exp1_depth.py (14)

- `exp1/exp1_depth.py:68` [mdb-feature P6] named value pool with pool.donor(item, rule='same country, family order') — donor choice and its provenance are hand-coded and not sealed.
- `exp1/exp1_depth.py:77` [mdb-feature P5] run.set(addr, other_run.ref('cache.L20.v[kv6,row=..]')) with ORIGIN:ADDR recorded — donor value is fetched by a second run and passed as a bare tensor, origin l…
- `exp1/exp1_depth.py:78` [mdb] use ops/sets.py ORIGIN:ADDR / @pair: cross-run values instead of a second dbg.run per donor (gap audit §1 donor-value row).
- `exp1/exp1_depth.py:93` [mdb-feature P1] run.branches(at=ck, consumer=Margin(last,a,b)).add('R1', edits=[...]).run(mode='stepped') — checkpoint -> edit -> until logits -> read -> restore loop keyed by …
- `exp1/exp1_depth.py:94` [mdb] use excursion/come_back or propagate.whatif instead of a hand-rolled branch loop (gap audit §1 branch-loop row).
- `exp1/exp1_depth.py:99` [mdb] use propagate.whatif / SC3 seal-including-staged-edits to seal the predicted margin before the edit instead of forecast.seal taken after set_cell commits (gap a…
- `exp1/exp1_depth.py:122` [mdb-feature P7] for item, run in pset.runs(at='t@last.logits', prefill='@span_first', shard=K/N) — item loop, PICK/shard slicing and JSONL are hand-rolled (14 of 14 drivers).
- `exp1/exp1_depth.py:145` [mdb] use symbols.table(consumer, sets=, content_layers=True) with declare('carrier', ...) instead of FeatureContext + ctx.best_feature/ctx.content one-off reads (gap…
- `exp1/exp1_depth.py:146` [mdb-feature P9] layer-indexed FeatureContext over an SAE set (L0..L20) — one basis and one layer today, so the 'country at carrier' column cannot be a time series.
- `exp1/exp1_depth.py:149` [mdb] use an edge-mode watch over t@carrier.L*.h plus backtrace/reverse-continue instead of a level watch and an unconditional break at the SAME stop t<last>.L20.h (g…
- `exp1/exp1_depth.py:150` [mdb] with a time axis 'fired' would mean 'first became true at layer l' and the wrong write could be named; today 30/83 false alarms cannot be split from late-correc…
- `exp1/exp1_depth.py:193` [mdb-feature P1] br.add_null('random', like='R1', norm='matched', pool=active_features, seed=...) — the matched-norm random-feature null is drawn and rescaled differently in 4 s…
- `exp1/exp1_depth.py:194` [mdb-feature P6] pool.random(matched_norm=..., seed=...) — random feature draw and norm rescale are hand-coded here.
- `exp1/exp1_depth.py:211` [mdb-feature P8] sealed per-item record via dbg.records_to — per-item output is a bare JSONL line; sealed ids are not backed by a stored record.

## exp2/exp2_base.py (11)

- `exp2/exp2_base.py:95` [mdb-feature P9] feature_lineage(f, layer, set) (decoder-cosine chain, labelled derived-local) as a binding column — feature lineage is hand-rolled over B[x].W_dec.
- `exp2/exp2_base.py:117` [mdb-feature P7] for item, run in pset.runs(...) — per-item run setup and the PICK/shard slicing below are hand-rolled.
- `exp2/exp2_base.py:147` [mdb] use symbols.table(consumer, sets=, content_layers=True) with declared carrier/desc/t0/relay/question instead of component_table only (gap audit §4 exp2 first ro…
- `exp2/exp2_base.py:148` [mdb] use MapTable permutation null (circuit_map.py:507) instead of dropped cross-item / shuffled-id nulls (gap audit §4 exp2).
- `exp2/exp2_base.py:170` [mdb-feature P9] projection strip + lineage as binding columns over a layer-indexed FeatureContext — exact projection strip over layers is hand-computed from raw reads.
- `exp2/exp2_base.py:179` [mdb] use symbols liveness() per layer with floors instead of remove-component arms per layer built by hand into cell_sweep (gap audit §4 exp2 'liveness() for L4').
- `exp2/exp2_base.py:180` [mdb-feature P1] br.add / br.add_null per (layer, arm) with floor attached — arm list + labels + floor bookkeeping are parallel hand-kept lists.
- `exp2/exp2_base.py:198` [mdb-feature P1] br.add_null('random', like='top', norm='matched', pool=active_nonfeature) — matched-norm random component arm drawn and rescaled by hand.
- `exp2/exp2_base.py:199` [mdb-feature P6] pool.random(matched_norm=..., seed=...) — same hand-coded random draw.
- `exp2/exp2_base.py:225` [mdb-feature P7] pset = dbg.prompt_set(...); pset.shard(K, N) — PICK / s0 / s1 slicing and JSONL loops below are hand-rolled.
- `exp2/exp2_base.py:249` [mdb-feature P8] record sink / attached project — provenance and per-item records are not persisted as sealed records.

## exp3/exp3_hop3.py (16)

- `exp3/exp3_hop3.py:43` [mdb-feature P3] sae.open_set('transcoder', TC_DIR, layers=...) with read_site='L<l>.n', write_site='L<l>.m' — safetensors transcoders are loaded and moved to device by hand; sa…
- `exp3/exp3_hop3.py:52` [mdb] use reads.readout(run, Margin).vector/.rho (analysis/reads.py:108-128) instead of manual final-norm gamma/rho and W_U rows (gap audit §1 final-norm readout row)…
- `exp3/exp3_hop3.py:58` [mdb] use interop.names.to_nnsight / core.foreign_names.ct_node instead of a hand-typed address map (gap audit §1 library address maps row).
- `exp3/exp3_hop3.py:61` [mdb-feature P7] pset.shard(K, N) with roles/vars — pool construction and SHARD/PICK slicing are hand-rolled.
- `exp3/exp3_hop3.py:79` [mdb-feature P3] FeatureBasis.encode for transcoders (bf16 GPU screen + float64 exact re-encode) — encode/decode of transcoders is hand-written here and again in exp4.
- `exp3/exp3_hop3.py:102` [mdb] use symbols.table + an edge-mode watch on the answer row's candidate-set argmax over L*, with margins (Q2 item 4), instead of one read at the logits (gap audit …
- `exp3/exp3_hop3.py:107` [mdb] use reads.readout(run, Margin(last, lang, rl)).vector/.rho instead of hand-rolled rho and cl/cc direct-effect vectors (gap audit §1 final-norm readout row).
- `exp3/exp3_hop3.py:125` [mdb-feature P3] MLP-write binding: component_binding(run, name, 't<r>.L<l>.m', transcoder_basis) with the Unbound row = error — recon, rel_err and error_de_lang are recomputed …
- `exp3/exp3_hop3.py:136` [mdb-feature P10] nominate(scores, rule, k) -> sealed SuspectList, then branches.measure(list, nulls=[...]) — frozen nomination rule is coded by hand (5 such rules across scripts…
- `exp3/exp3_hop3.py:163` [mdb-feature P7] each/pset.runs(..., shard=K/N) — item loop and shard slicing hand-rolled.
- `exp3/exp3_hop3.py:186` [mdb-feature P1] run.branches(at=ck)... br.add per nominated feature, 'joint', 'random', 'mean' — one checkpoint + go() loop is the Branches object; the joint arm below needs tw…
- `exp3/exp3_hop3.py:205` [mdb-feature P3] feature_value(f, act, basis) supplied vector from the transcoder binding — feature write -a*d_f is built by hand.
- `exp3/exp3_hop3.py:212` [mdb-feature P1] br.add_null('random', like='top', norm='matched', pool=active_non_nominated) — matched-norm random null drawn by hand.
- `exp3/exp3_hop3.py:213` [mdb-feature P6] pool.random(matched_norm=..., seed=...) — same.
- `exp3/exp3_hop3.py:222` [mdb-feature P6] pool.mean(by=..., leave_out=item) over the MLP write — leave-one-out class mean coded by hand (self-inclusion bug hit once in TP1).
- `exp3/exp3_hop3.py:225` [mdb] use branches from the one checkpoint (P1) so hop_edge attaches to any branch instead of two extra fresh runs (nat2, joint) (gap audit §4 exp3 / exp3:211-217).

## exp4/exp4_brackets.py (13)

- `exp4/exp4_brackets.py:47` [mdb-feature P3] transcoder FeatureBasis loaded via sae.open_set — same hand-written loader as exp3.
- `exp4/exp4_brackets.py:64` [mdb] use interop.names.to_nnsight / core.foreign_names.ct_node (SC8 circuit-tracer adapter) instead of a hand-typed address map (gap audit §1 library address maps ro…
- `exp4/exp4_brackets.py:74` [mdb-feature P4] run.hold('t@last.L>l.p', source=base_run) and run.hold('t@last.L>l.norm_scale', source=base_run) — frozen-pattern / frozen-norm gradient is a detached-q/k nnsig…
- `exp4/exp4_brackets.py:108` [mdb] use reads.readout(nat, Margin).vector/.rho instead of manual rF and c = gamma*(W_U[a]-W_U[b]) (gap audit §1 final-norm readout row).
- `exp4/exp4_brackets.py:119` [mdb-feature P3] transcoder binding on t<last>.L<l>.m giving feature writes and direct effects — encode/direct effect/decode redone by hand (exp3:43-85 again).
- `exp4/exp4_brackets.py:120` [mdb-feature P10] nominate(...) -> sealed SuspectList of top-5 by direct effect — nomination rule coded by hand.
- `exp4/exp4_brackets.py:136` [mdb-feature P1] br.add_null('random', like=node1, norm='matched', pool=same_layer_active) — matched-norm null drawn by hand.
- `exp4/exp4_brackets.py:150` [mdb-feature P4] hold('t@last.L>l.norm_scale', source=base_run) — per-layer base rho and scale dicts are rebuilt from an nnsight clean pass.
- `exp4/exp4_brackets.py:169` [mdb] use symbols.hop_edge per node as the exact split beside the held arms instead of only the three held arms (bf16-resolution brackets) (gap audit §4 exp4).
- `exp4/exp4_brackets.py:170` [mdb-feature P8] full seal_record kept via record sink — only sealed['id'] is stored (line below).
- `exp4/exp4_brackets.py:177` [mdb-feature P4] arm1 = run.hold(p), arm2 = run.hold(p).hold(norm_scale) on a P1 branch — frozen-p and frozen-norm arms are re-set at every later layer by hand (exp4:172-182).
- `exp4/exp4_brackets.py:178` [mdb-feature P1] br.add('arm0'|'arm1'|'arm2'|'null', edits=[...], holds=[...]) — branch loop by hand.
- `exp4/exp4_brackets.py:203` [mdb-feature P8] store the full bounded seal record, not just sealed['id'] — procedure, sealed_at and address map are not persisted.

## exp4/check_frozen_grad.py (1)

- `exp4/check_frozen_grad.py:1` [mdb-feature P4] run.hold('...p') / hold('...norm_scale') makes this bitwise check of the detach-based frozen gradient a debugger-mode test — hand-built nnsight freeze hooks liv…

## exp5/exp5_neurons.py (13)

- `exp5/exp5_neurons.py:47` [mdb] use interop.names.to_nnsight instead of a hand-typed address map (gap audit §1 library address maps row).
- `exp5/exp5_neurons.py:54` [mdb-feature Q2.2] linearity ratio per edit stored as a column (native dM over mdb's own linear forecast) — N2 ratio 0.83 / Spearman 0.94 was computed against nnsight only, so 'nn…
- `exp5/exp5_neurons.py:64` [mdb-feature P6] pool.mean(by=..., leave_out=item) of the f cell over the panel — 64-item mean computed by hand with no record of which items entered it.
- `exp5/exp5_neurons.py:68` [mdb-feature P7] pset.split(by='$bridge', alternate=A/B) with roles — A/B panel construction is hand-rolled.
- `exp5/exp5_neurons.py:69` [mdb-feature P10] nominate(scores, rule, k) -> sealed SuspectList — nomination rule (8 most negative, negative on every A item) coded by hand.
- `exp5/exp5_neurons.py:109` [mdb] use propagate.whatif linear term sealed as a second (derived) column next to the nnsight supplied forecast instead of only the nnsight forecast is sealed (gap a…
- `exp5/exp5_neurons.py:116` [mdb-feature P8] keep the full seal_record via record sink — only {'id','value'} is kept per record.
- `exp5/exp5_neurons.py:130` [mdb-feature P6] pool.random(matched_norm=..., seed=...) and a lexical/answer-token null pool — random set and answer-token-fixed null are drawn by hand.
- `exp5/exp5_neurons.py:131` [mdb] use analysis/candidates.py lexical null instead of the hand-rolled answer-token null score loop below (gap audit §1 answer-token null row).
- `exp5/exp5_neurons.py:184` [mdb-feature P1] run.branches(at=ck).add('set', edits=[...]).add(top_single).run(mode='stepped', recheck=2) — stepped authority over batched arms is the Branches 'recheck=k' pat…
- `exp5/exp5_neurons.py:192` [mdb-feature P5] run.set('t@last.L<x>.f[i=k1,k2]', mean_values) selector on the f port — value/mask tensors for the neuron set are built by hand.
- `exp5/exp5_neurons.py:205` [mdb] use verdict.spacing instead of common.spacing (gap audit §1 bf16 spacing row).
- `exp5/exp5_neurons.py:218` [mdb-feature P8] panel + named sets sealed into the project — panel membership is a loose JSON file, not custody-checked.

## exp6/exp6_tlnull.py (13)

- `exp6/exp6_tlnull.py:44` [mdb] use circuit_map.group_writers / slot per-model KV group instead of a hand-built query-head list for the KV group (gap audit §1 per-KV-group row).
- `exp6/exp6_tlnull.py:67` [mdb] use interop.names.to_tl (used here) and names.tl_hook (mdb-sc-das, SC7) for layout instead of per-hook layout handling in tl_arms/tl_donor_v (gap audit §6 tl_ho…
- `exp6/exp6_tlnull.py:78` [mdb] use cm.slot_rows with a per-model Slot (common.py:43-45) instead of crow_of re-implementing the carrier-row rule (gap audit §1 carrier-row row).
- `exp6/exp6_tlnull.py:79` [mdb-feature P11] Slot taking layers/kv from the map's model entry — exp6 needed the carrier per model, which Slot hard-codes.
- `exp6/exp6_tlnull.py:88` [mdb-feature P1] run.branches(at=ck): add one branch per head at L_c, L_c+1 plus the V interchange — checkpoint / zero head / until logits / restore loop written by hand.
- `exp6/exp6_tlnull.py:103` [mdb-feature P5] run.set(f't{last}.L{x}.A[h{h}]', 0) selector instead of a hand-built head mask — 4 lines of mask construction per arm; same class as SC7's 4-D TL slice bug.
- `exp6/exp6_tlnull.py:115` [mdb-feature P5] run.set(f't{crow}.L{LC}.v[kv{KVC}]', donor_ref) with the donor origin recorded — KV-group mask and donor tensor built by hand, origin lost.
- `exp6/exp6_tlnull.py:132` [mdb-feature P6] pool of donor values with origins (cache.L<LC>.v[kv,row=@carrier]) — donor value fetched by a second run and cached in a dict.
- `exp6/exp6_tlnull.py:153` [mdb] use names.tl_hook (mdb-sc-das, SC7) to hide the (batch,pos,heads,d_head) vs flat layout split instead of x.dim()==4 branches in zero(), vx(), tl_donor_v (gap au…
- `exp6/exp6_tlnull.py:154` [mdb-feature P5] shared selector-to-index helper used by both the mdb side and the TL side — same class of indexing error on the library side.
- `exp6/exp6_tlnull.py:211` [mdb-feature P7] for item, run in pset.runs(...) — item loop and PICK slicing hand-rolled.
- `exp6/exp6_tlnull.py:246` [mdb-feature Q2.7] kernel-path deltas (TL vs mdb no-intervention) recorded in the run's numeric_scope as measured entries — U16_b1 and pattern_maxdiff are re-measured and stored o…
- `exp6/exp6_tlnull.py:261` [mdb-feature P8] keep the full judged record via record sink — only record id is kept (exp6:249).

## exp7/exp7_iia.py (9)

- `exp7/exp7_iia.py:60` [mdb-feature P5] run.set('t<crow>.L20.v[kv6]', donor_run.ref(...)) with origin recorded — donor cell passed as a bare tensor, origin lost; second run per donor cached in DON.
- `exp7/exp7_iia.py:61` [mdb-feature P6] pool of donor cells with origins — same.
- `exp7/exp7_iia.py:68` [mdb] use symbols.table(consumer, sets=, content_layers=True) with a declared carrier instead of a one-off component_binding at the carrier (gap audit §0: symbol tabl…
- `exp7/exp7_iia.py:74` [mdb-feature P7] for item, run in pset.runs(...) — item loop and PICK slicing hand-rolled.
- `exp7/exp7_iia.py:89` [mdb] use SubspaceBinding (analysis/subspace.py on branch mdb-sc-das) with SAE pushforward basis and provenance instead of the hand-built QR basis Q of cell images an…
- `exp7/exp7_iia.py:90` [mdb-feature P3] one 'decomposition of a site's value' interface shared by Subspace, ComponentBinding and transcoder bindings — exp7's Q is the same object as a component basis.
- `exp7/exp7_iia.py:104` [mdb-feature P1] Subspace.interchange built on the P1 branch object so IIA and every other arm share one floor and one authority path — checkpoint + arms_v loop is a hand-rolled…
- `exp7/exp7_iia.py:111` [mdb] use set_cell(rows=, kv=) (or P5 selectors) instead of a masked port set with hand-built full/mask tensors (gap audit §4 exp7 row).
- `exp7/exp7_iia.py:129` [mdb-feature Q2.7] pyvene clean margin (kernel-path delta vs mdb stepped native) recorded in numeric_scope — M_clean is measured here and not attached to the run.

Total: 97 TODOs.
