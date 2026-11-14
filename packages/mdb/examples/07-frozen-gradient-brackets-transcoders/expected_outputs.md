# Expected outputs: the attribution-graph gap in brackets, on transcoder nodes (Qwen3-1.7B-Base)

Source experiment: showcase exp4 (`results/showcase-20261114/exp4/`: `exp4_brackets.py`,
`prereg.json`, `score.py`, `summary.json`, `out/exp4-items.jsonl`), scored job
**job-09feeeb61cbe** (MEASURED, single seed). It was independently audited in
`results/showcase-20261114/AUDIT-SHOWCASE.md`: the exp4 row and correction 3. The owner approved it
on one condition: present it on the above-floor nodes. Model Qwen3-1.7B-Base, bf16, eager
attention, CUDA on Beast. Transcoders: mwhanna `qwen3-1.7b-transcoders-lowl0`, layers L20..L24.

## Items

All 32 items of exp4's panel: the first 8 base items by id of each two-hop family (city,
currency, landmark, person). The consumer is `M = logit(answer) - logit(donor answer)` at the
last row. `items.json` and `reference.json` are cut by `make_reference.py` from
`results/twohop-20261113/next/r1_items.json` and exp4's raw output. A full run takes 3.5 minutes,
so the example uses no subset. `MDB_ITEMS=0,1` runs a subset, and the comparison table then
covers only those items. exp4 drew its random node with a seeded rng over the whole panel, so
`items.json` stores each item's recorded pick and the example reuses it.

## Jobs

| Job | What | Wall | Peak VRAM | Peak RSS |
|---|---|---|---|---|
| job-2092d2852921 | smoke, item 0 (city:Barcelona). The script exited 0, but the job was marked failed because `run.sh` returned the exit status of a `pipefail` grep with nothing to print. Fixed: `run.sh` now exits with the script's return code. | 14 s | 3.9 GB | 8.5 GB |
| **job-5141d9c930ba** | full validation, 32 items (`validation-job-5141d9c930ba.log`) | 211 s | 3.9 GB | 9.6 GB |

Validation custody, from the sealed records: script sha256 `cf86f551...e5303` (the committed
`frozen_gradient_brackets.py`) and prereg sha256 `ee85a407...28e01` (`prereg.json`, exp4's
prereg copied verbatim). Kernel path `mdb-stepped`. Records written to the sink: 192
`supplied-forecast` (one per node-item edge, sealed before the arms) and 32 `branches` (one per
item, 22-24 arms each).

## Key numbers: validation vs showcase, same 32 items (MEASURED, job-5141d9c930ba vs job-09feeeb61cbe)

Both columns come from one rule (`summarise` in the script) applied to the nodes measured now and
to exp4's saved records. "Above floor" means `|arm0| > floor`, with floor = 2 bf16 spacings at the
larger `|logit|` of the pair (0.25 or 0.5 here).

| Number | Showcase | Now |
|---|---|---|
| node-items / with \|arm0\| > floor / > 4 floors | 160 / 103 / 31 | 160 / 103 / 31 |
| node-items with any bracket above floor | 44 | 44 |
| **G1 above floor**: median \|arm0 - arm1\| / \|arm0\| | 0.0952 | 0.0952 |
| G1 above 4 floors | 0.0870 | 0.0870 |
| G1 over all node-items (the prereg's "0.105 fail") | 0.1053 | 0.1053 |
| **native / frozen-gradient edge** (arm0 / -A), above floor | 0.9220 | 0.9220 |
| native / frozen-gradient edge, above 4 floors | 0.9742 | 0.9742 |
| **native / direct edge** (arm0 / -direct), above floor | 0.6108 | 0.6108 |
| native / direct edge, above 4 floors | 0.6388 | 0.6388 |
| frozen-gradient edge / direct edge, above floor | 0.7353 | 0.7353 |
| direct-edge overstatement on above-floor nodes (1 / 0.6108) | 1.637x | 1.637x |
| **G4**: Spearman(-A, arm0), all node-items | 0.9320 | 0.9320 |
| G3: norm bracket within floor | 135/160 | 135/160 |
| median \|bracket\| (rerouting, norm, nonlin) | 0.125, 0.095, 0.049 | 0.125, 0.095, 0.049 |
| machinery null (holds on, no edit) within floor | 192/192 | 192/192 |
| random-feature node, all brackets within floor | 26/32 | 26/32 |

These match the audit's full-panel figures: G1 0.095 above floor, edge share 0.92 / 0.97,
direct share 0.61-0.64, about 1.6x overstatement, Spearman 0.93, and 44/160 with any bracket
above floor.

**Reading (as the audit framed it).** Most brackets sit at about one bf16 step, so G1's 0.105
"fail" and G2's largest-bracket vote are at quantisation resolution. On the 103 node-items
where removing the node has an above-floor effect:

- The re-routing bracket is under a tenth of the native effect.
- The frozen-gradient edge accounts for 0.92 of the native effect, and 0.97 above 4 floors.
- The transcoder direct edge overstates the native effect by about 1.6x, not 2x.

The frozen-gradient edge explains the gap: it is later layers' linear response through their
MLPs (edge / direct = 0.74).

New in this example (not in exp4), MEASURED in job-5141d9c930ba:

- `supplied.judge` of the sealed edge against the native removal: the signs agree on 102/103
  above-floor node-items.
- The Gaussian matched-norm null (`Branches.add_null("random", like="n0.arm0")`, a seeded
  Gaussian add at node 1's MLP write with node 1's write norm) is within floor on only 17/32
  items. A random *direction* of that norm moves the margin more often than a random active
  *feature* of the same norm, which stays within floor on 26/32. The example reports this null.
  It does not enter any bracket or ratio above.

## Reproduction, node for node (MEASURED, job-5141d9c930ba)

| Check | Result |
|---|---|
| `Branches` base margin and floor equal to the native run's | 32/32 items |
| nodes (kind, layer, feature) equal to exp4's, random node included | 32/32 items |
| `arm0` (free removal, through `Branches`) bitwise equal to exp4 | 192/192 node-items |
| `arm1` (`hold t@last.L>l.p`) bitwise equal to exp4's hand-set `p` | 192/192 |
| machinery null bitwise equal | 192/192 |
| `arm2_f64` (holds `p` + `norm_scale`, float64 readout at the base final scale) vs exp4 `arm2` | max \|diff\| 1.1e-14 |
| `direct` (`component_table`) vs exp4 | max \|diff\| 2.7e-15 |
| frozen-gradient edge `-A` (nnsight via `names.to_nnsight`) vs exp4 | bitwise on 14/192, max \|diff\| 4.4e-15 |
| `arm2` read through `Branches(final_scale="held")` (bf16 logits) vs `arm2_f64` | max 0.70 floors |

The differences, explained:

- **direct, edge.** mdb's `TranscoderBasis.encode` recomputes the activations in float64 with one
  matmul over the active rows. exp4 did the same, but over its own active set, which it screened
  with a bf16 GPU matmul; mdb screens in float32. A different row set, or a different matmul
  shape, changes the BLAS blocking. That would make the activations, and hence `a_s d_s`,
  `direct` and `A = g . a_s d_s`, differ in the last float64 bits. This is the likely cause
  (asserted, not isolated). The gradient trace is the same nnsight computation.
  The bf16 cast of the edit absorbs these bits: `arm0` and `arm1` are bitwise equal on every
  node-item, so the written edits are identical.
- **arm2_f64.** The holds stage exactly exp4's `p`, `a` and `n` values (the bf16 arms are bitwise
  equal). The 1e-14 comes from float64 association order in the readout:
  `(rho_F c) . h` (`reads.readout(...).vector`) against exp4's `rho_F (c . h)`.
- **arm2 through Branches.** `final_held_logits` gives real bf16 logits,
  `lm_head(w_F (h_out rho_F,base).to(bf16))`. exp4 read a float64 linear readout instead. The two
  differ by bf16 rounding of the normed state and of the logits: up to 0.70 floors, so under one
  floor. The brackets use `arm2_f64`, as exp4 did. This is a choice of readout, not a disagreement
  between holds and exp4.

## mdb objects used, and the showcase code they replace

| exp4 hand code (TODO) | mdb object here |
|---|---|
| safetensors loader, bf16 encode, float64 re-encode, decode, direct effect (P3, exp4:47, :119) | `sae.open_transcoders`, `components.transcoder_binding`, `components.component_table` (the partition closes: `partition.holds` asserted) |
| manual `rF` and `c = gamma (W_U[a] - W_U[b])` | `reads.readout(run, Margin).vector` / `.rho` |
| hand-typed nnsight address map (exp4:64) | `interop.names.to_nnsight` for every module the gradient trace touches, and for the sealed `address_map` |
| hand-built rho dicts (P4, exp4:150); hand-stepped arm1/arm2 re-setting `p`, `a`, `n` (P4, exp4:177) | `run.hold("t@last.L>l.p", source=...)`, `run.hold("t@last.L>l.norm_scale", ...)`, `holds.final_held_logits` (via `final_scale="held"`), `holds.rms_scale` for the nnsight frozen scales |
| branch loop (P1, exp4:178); floor by hand | `dbg.branches(...)`, with arms `n<k>.arm0` / `arm1` (holds p) / `arm2` (holds p + norm_scale, final scale held), `null.L<l>`, `add_null("random")`; shared floor (`verdict.margin_floor`), `verdict.floors`, stability band |
| only `sealed['id']` stored (P8, exp4:170, :203) | `dbg.custody(script=, prereg=)`, `dbg.records_to(...)`: full supplied-forecast and branches records with custody |
| (none) | `supplied.seal` + `supplied.judge` for the library edge |

Kept as plain Python, because the API is not built: node nomination, top 5 by direct effect
(P10, suspect lists), the per-item loop (P7), and the recorded random-node pick. The nnsight
frozen-gradient trace stays a library computation (the edge is the library's number).
`symbols.hop_edge` was not used: it splits a *cache-cell* edit, and these edits are MLP-write
edits.

## Not verified

- The run used `nnsight._Frozen` (a private helper of `mdb.interop.nn`, as exp4 did).
- circuit-tracer itself is not run. The "attribution-graph edge" is nnsight's frozen gradient,
  the memo's allowed path, as in exp4's prereg.
- The transcoder-error bracket is deferred, as in the memo and the prereg.
- Single seed. The bracket verdicts inherit the audit's caveat: below about 4 floors, a verdict
  can flip under another kernel path (audit correction 1).
