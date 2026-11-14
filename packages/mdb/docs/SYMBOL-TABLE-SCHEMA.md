# Symbol table schema reference

This is the column reference for `mdb.analysis.symbols.table` (record kind `symbol-table`, schema `mdb-symbol-table-v1`), its long-frame export (`Table.to_frame`) and the related record kinds. It was written for branch `mdb-symtab` (2026-11-14), which builds the recommendations of `obsidian/reviews/2026-11-14-symbol-table-additions-review.md`.

The tables in the "Columns" section are generated from `mdb.analysis.frame.COLUMNS` / `READ_COLUMNS` / `HEAD_COLUMNS` (`frame.schema_markdown()`). `frame.json_schema(name)` builds a JSON Schema (draft 2020-12) for `mdb-symbol-table-v1`, `mdb-circuit-map-v1` and `mdb-trace-v1` from the same list and the record kinds' required fields. Edit `COLUMNS` and regenerate rather than editing those tables by hand.

## Authority labels

Every number in a table row is a claim `{value, authority}`. The labels are the record layer's (`core.records.AUTHORITIES`):

- `derived-exact`: float64 arithmetic on the run's own values (an identity, P06/P07/P19 and the like). Exact up to float64 summation order.
- `measured`: read from a native forward (a cache cell, a native consumer, a native arm).
- `supplied`: given by the caller (an edit's value, a label, a projection subspace, a Neuronpedia id mapping).
- `bound`: a rounding bound.
- `derived-local`: a local approximation (a float32 vocabulary ranking, a Jacobian term).

Structural fields (names, rows, token ids, counts) are plain values. In the frame, a claim column `c` comes with `c__auth` holding its label.

## Options of `symbols.table` (and of the `symbols` op line)

| Python | Op line | What it adds |
|---|---|---|
| `sets=[Candidates]` | `sets=A,B` (declared with `candidates`) | `content.<set>` per binding; with word-built sets also `truth` and `flags` |
| truth in `Candidates` | `truth=A:TOKEN\|WORD` (`$var` in `each`) | `truth_rank`, `names_truth` |
| `content_layers=True` | `content_layers=true` | `content_lifetime.<set>` strips |
| `hops={name: {native_delta, direct_delta}}` | `hops=NAME:NATIVE:DIRECT` | `hop` with the indirect share |
| `sink=(0,)` (default) | `sink=0` or `sink=none` | `mass_nonsink`, `sink_ratio` on t0, `report.sink` |
| `lexical_null=True` | `lexical_null=true` | `content.<set>.lexical_null` |
| `co=[CoReadout(target, project_out, k)]` | `co=TARGET/OTHER[:K]` | `content.<target>.co.<other>` (raw, projected, lexical_null of the removed part, counts) |
| `switch={target: on}` | `switch=TARGET:ON` | `content.<target>.switched` |
| `labels={k: v}` | `labels=K:V` | `labels.<k>` on every binding, `report.labels` |
| `derive={k: path}` | `derive=K:PATH` | derived `labels.<k>`, `report.derived_labels` |
| `foreign=True` | `foreign=true` | `foreign` per binding, `readers[*].foreign` |
| `causal=True\|"mask"\|"zero"` | `causal=true\|mask\|zero` | measured `causal` per listed binding (else `status: unknown`) |
| `components=[ComponentBinding]`, `neuronpedia={...}` | (Python only) | `report.components`: feature bindings |
| `store=` | (the project's store) | `full` blob pointer for bindings past 64 rows/definitions |

`contrast`, `mask_delta`, `share` and `share_of_abs` are added whenever the consumer is a scalar (a margin or a logit). `position`, `is_sink`, `token_class` and `pieces_of_word` are added to token rows (`token_class`/`pieces_of_word` need a tokenizer). `first_subword_of` is added to token rows when word-built candidate sets are read.

Declaring candidate sets on the REPL: `candidates countries = countries.json`, `candidates capitals = W1,W2,... paired=countries`, `candidates show NAME`, `candidates list`. A file is a JSON list of words, `{"words": [...], "pairs": {...}}`, or one word per line. Token ids are written `#123,#456`. The truth is never part of a set: it is given per table (`truth=`), so a reader never receives an expected answer (the rule ported from `mdb_lanes.symbol_sources`).

## Caveats that the columns encode

- `mass` and `to_consumer` stay exact and partition the head mass and the consumer; t0 cannot be removed from them without breaking the sum (AUDIT-Z1-Z2). The discounted forms (`contrast`, `mass_nonsink`) are added beside them.
- `mask_delta` is a direct-path quantity: the change of the consumer's direct term if that one read were masked with renormalisation, every other input held. On FT1 it was within 2 bf16 spacings of the native mask on 6/32 items (measured, `next/ft1-full-summary.json`). It is not the native mask effect.
- `share` flips sign with a negative margin; `share_of_abs` divides by `|consumer_total|` instead.
- A co-readout projects out the whole span of the other set (or its top-k principal directions with `:K`), not one direction. Projecting one direction leaves the true country in the write (AUDIT-SCALE.md:133-157). The removed part's own readout is reported as the lexical null, and `counts` is true only when the projected readout names the truth and the removed part alone does not.
- `causal` is measured with batch-1 stepped arms from a fresh prefix cache. The floor is the null arm's difference from the run's own stepped consumer. The `mask` and `zero` forms are the same number at the reader row (zeroing `p[h, j, s]` and zeroing `v_s` give the same row-j output). The column refuses a run that carries edits.
- Cross-item position means must be leave-one-out (`frame.loo_position_mean`): identical prefixes put copies of a row in its own mean (AUDIT-RUN1-TP1).

## Foreign names and their verification status

| Name | TransformerLens | nnsight | Status |
|---|---|---|---|
| value cell `cache.L<l>.v[kv<g>,row=<r>]` | `blocks.<l>.attn.v.hook_out[:, r, g, :]` | `model.model.layers[l].self_attn.v_proj.output[:, r, g*dh:(g+1)*dh]` | measured (`tests/test_interop_symtab.py`: TL == nnsight bitwise, mdb within fp32 rounding) |
| key cell `cache.L<l>.k[kv<g>,row=<r>]` | `blocks.<l>.attn.hook_rot_k[:, r, g, :]` | `...self_attn.source.attention_interface_1.inputs[0][2][:, g, r, :]` | TL measured here; the nnsight `kr` path and layout are measured by the port parity test in `test_interop.py` |
| token row `t<r>` | `blocks.*.attn.v.hook_out[:, r, :, :]` (+ `hook_rot_k`) | `model.model.layers[*].self_attn.v_proj.output[:, r, :]` | measured (per cell) |
| read `t<j>.L<l>.h<h>` | `blocks.<l>.attn.hook_pattern[:, h, j, s]` | `model.model.layers[l].self_attn.output[1][:, h, j, s]` | measured |
| residual `L<l>.h_out` | `blocks.<l>.hook_resid_post` | `model.model.layers[l].output` | measured (`test_interop.py`) |
| SAELens feature | `release`, `sae_id`, `hook_name`, feature index | | measured on a random dictionary (SAELens TopKSAE agrees on ids and activations) |
| pyvene spec | | | unverified: pyvene is not installed; the component string is not checked against its Qwen2 model map |
| circuit-tracer node | | | unverified: not installed (needs transformers <= 4.57) |
| Neuronpedia URL | | | supplied: built only from a caller-given `{release/sae_id: {model, source}}` mapping |

## Record kinds

- `symbol-table`: one page of at most 16 bindings. New optional body fields: `labels`, `derived_labels`, `candidate_sets` (name, size, rule, source, sha256, flag counts, truth), `components`.
- `trace` (schema `mdb-trace-v1`): a `rows.Trace` sealed with `Trace.seal(...)` or `sealed.seal_trace`. The per-step records are a sha256-checked JSON blob; the record keeps the trace kind, count, column names, a bounded `meta` and the sha256. `breakpoints.replay` accepts the `Trace` or the sealed record.
- `watch-hits` (schema `mdb-watch-hits-v1`): a replay result bound to its sealed trace's record id.
- `circuit-map` (schema `mdb-circuit-map-record-v1`): the map's exact JSON as a blob (so `Map.digest` recomputes from it), with its digest, slots, edges, candidate-set sizes and optionally a `MapTable` blob. `sealed.load_map` checks both the sha256 and the digest.
- `prompt-set` (the `each` summary): sets of any size. Past 64 items the first 64 rows stay inline and every row is in `items_blob`. `each ... shard=K/N` runs one contiguous part.

## Circuit maps

- `bind` marks each binding with its slot's `role`, `slot` and the map's `map_digest`; `item_report` lists `unbound_slots` (slots that bound no row on that item).
- Cross-item statistics key bindings by symbol name (`table.bindings.carrier.mass`), never by page position: `bind` skips a slot that binds no row, and positional paths then line up different slots across items (reproduced in `test_positional_paths_misalign_slots_when_a_slot_binds_no_row`).
- `freeze(..., ties="value", switch="labels.family.label", supplied={...})`: mode ties are broken by value and reported; `by_switch` holds the statistics per label value (the pooled summary is the fallback for an unseen value); a supplied constant must carry its source record (`supplied_constant`).
- `Map.symbols` declares `LabelSymbol(name, cardinality, visible_from)` (ported from `mdb_lanes.compiled_state.Symbol`): `freeze` refuses a label with more values than its cardinality, and a binding whose rows all lie before `visible_from` gets the label in `hidden_labels` instead.
- `MapTable.build(m, reports, switch=...)`: one row per slot with `n_bound`, statistics of `mass`, `to_consumer`, `contrast`, `mass_nonsink` and `mask_delta`, the modal site `(layer, kv, row - last)` with ties broken by value, a constant-site baseline, per set the best-token mode, `names_truth` count, the answer-permutation null and flag counts, and per-switch-value groups. `MapTable.frame()` gives the per-item rows behind it.

## Columns

## Binding rows (`report.bindings[*]`, frame level `binding`)

| Column | Type | Authority | Meaning |
|---|---|---|---|
| `symbol` | string | structural | the binding's name (a declared symbol, a slot, or t<row>) |
| `kind` | string | structural | declared, token, or component |
| `rows` | array | structural | source rows; half-open ranges a:b past 64 rows |
| `definitions` | array | structural | the addresses that define it (first 64, then a count) |
| `lifetime.strip` | string | derived | per reader row, one mark per layer: # read at or above min_mass, . below, - no value cell there, ? not computed |
| `lifetime.first` | string | derived | first t<j>.L<l> read at or above min_mass |
| `lifetime.last` | string | derived | last t<j>.L<l> read at or above min_mass |
| `mass` | claim | derived-exact (supplied under a p/s edit) | attention weight its reads receive, summed over heads and layers |
| `count` | integer | structural | number of (row, layer, head) reads |
| `mean_mass` | claim | derived-exact | mass per read |
| `readers` | array | mass exact or measured; effect derived-exact P19 | top reads {name t<j>.L<l>.h<h>, mass, effect, contrast, foreign} |
| `status` | string | derived | edited, read, live, unread (unresolved: pending symbol) |
| `token` | integer | structural | token rows: the token id |
| `text` | string | structural | token rows: the decoded text (with a tokenizer) |
| `position` | integer | structural | token rows: the row index |
| `is_sink` | boolean | structural | token rows: the row is a declared sink row |
| `token_class` | string | structural (tokenizer) | special, bos_or_t0, whitespace, punct, digit, word_initial or continuation |
| `pieces_of_word.index` | integer | structural (tokenizer) | index of the piece within its whitespace-delimited word |
| `pieces_of_word.count` | integer | structural (tokenizer) | pieces in that word |
| `first_subword_of` | array | derived-exact (tokenizer) | candidate sets in which this row's token is the first piece of a multi-piece word |
| `to_consumer` | claim | derived-exact P19 | exact direct contribution to the scalar consumer (share form: sum p c.W_O v_s) |
| `contrast` | claim | derived-exact P19 | sum over its reads of p (c.W_O v_s - c.W_O vbar_h): the effect against the head's own mean value (vbar_h = sum_s p v_s) |
| `mask_delta` | claim | derived-exact P19, direct path only | sum over its reads of -contrast/(1-p): each read masked alone with renormalisation, every other input held; not the native mask effect (FT1: within 2 bf16 spacings on 6/32 items) |
| `share` | claim | derived-exact | to_consumer / consumer_total (flips sign with the margin) |
| `share_of_abs` | claim | derived-exact | to_consumer / |consumer_total| |
| `mass_nonsink` | claim | derived-exact | mass renormalised per head over non-sink sources (sink rows: report.sink, default t0) |
| `sink_ratio` | claim | derived-exact | t0 only: its value norm over the median row's, median over layers |
| `role` | string | structural | the circuit-map role of the slot that bound it |
| `slot` | string | structural | the circuit-map slot that bound it |
| `map_digest` | string | structural | sha256 of the map whose slot bound it |
| `labels.<name>` | claim or {value, authority} | supplied, or derived (with from) | item labels (family, depth, country) and derived labels (decoded_country = content.countries.best) |
| `causal.status` | string | structural | unknown (observed only) or intervened |
| `causal.verdict` | string | measured | effect, null or below_floor |
| `causal.delta` | claim | measured | native consumer change with the consumer row's reads of the binding removed (p = 0, no renormalisation) at every layer and head |
| `causal.floor` | claim | measured | |null arm - the run's stepped consumer| |
| `content.<set>.best` | integer | derived-exact P19 | best token of the set for the binding's write |
| `content.<set>.score` | claim | derived-exact P19 | its readout rho gamma.W_U[t].w |
| `content.<set>.margin` | claim | derived-exact P19 | best minus runner-up |
| `content.<set>.truth` | integer | supplied | the declared true token |
| `content.<set>.truth_rank` | integer | derived-exact | 1-based rank of the truth |
| `content.<set>.names_truth` | boolean | derived-exact | best == truth |
| `content.<set>.flags.first_subword` | boolean | derived-exact (tokenizer) | best is the first piece of a multi-piece word of the set |
| `content.<set>.flags.collision` | boolean | derived-exact (tokenizer) | best stands for more than one word of the set |
| `content.<set>.flags.kind` | array | derived-exact (tokenizer) | other declared sets that hold the best token |
| `content.<set>.lexical_null.names_truth` | boolean | derived-exact P19 | the truth token's own unembedding read through the same basis names the truth |
| `content.<set>.co.<other>.projected.names_truth` | boolean | derived-exact P19; projection supplied | readout with the whole span of <other>'s gamma W_U rows removed from the write |
| `content.<set>.co.<other>.lexical_null.names_truth` | boolean | derived-exact P19 | readout of the removed part alone |
| `content.<set>.co.<other>.counts` | boolean | derived-exact | projected names the truth and the removed part alone does not |
| `content.<set>.switched.names_truth` | boolean | derived-exact | names the word paired with the decoded word of the switch set |
| `content_lifetime.<set>` | string | derived | per layer: C names the overall best, c in top 3, . names another, - no value cell |
| `hop.native_delta` | claim | measured | native consumer change under a supplied replacement of the binding |
| `hop.direct_delta` | claim | derived-exact P19 | its direct-path change |
| `hop.indirect_share` | claim | measured | 1 - direct/native |
| `foreign.cells` | array | measured (TL, nnsight) / unverified (pyvene) | per cell: TransformerLens hook + index, nnsight path + index, pyvene spec |
| `components.<i>.components.<j>` | object | derived-exact P19; SAE weights supplied | feature bindings: activation, to_consumer, content, saelens ids, neuronpedia link |

## Reads (frame level `read`)

| Column | Type | Authority | Meaning |
|---|---|---|---|
| `binding` | string | structural | the binding holding the value cells read |
| `reader_row` | integer | structural | the reading row j |
| `layer` | integer | structural | layer l |
| `head` | integer | structural | query head h |
| `mass` | number | derived-exact | sum over the binding's sources of p[h, j, s] |
| `effect` | number | derived-exact P19 | share-form contribution to the scalar consumer |
| `contrast` | number | derived-exact P19 | contrast-form contribution |
| `mask_delta` | number | derived-exact P19, direct path | -contrast/(1-p) per source, summed |
| `sources` | integer | structural | value cells of the binding the head reads |
| `supplied` | boolean | structural | a p/s edit supplied a weight |

## Heads (frame level `head`)

| Column | Type | Authority | Meaning |
|---|---|---|---|
| `reader_row` | integer | structural | row j |
| `layer` | integer | structural | l |
| `head` | integer | structural | h |
| `total_mass` | number | measured | sum_s p[h, j, s] |
| `unbound_mass` | number | derived-exact | mass of bindings not on this page |
