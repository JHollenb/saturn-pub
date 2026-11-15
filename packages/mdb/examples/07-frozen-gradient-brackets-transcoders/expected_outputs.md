# Expected outputs: the attribution-graph gap in brackets, on transcoder nodes (Qwen3-1.7B-Base)

Source experiment: an earlier run of this experiment on the same 32 items (single seed), with a
sealed preregistration (`prereg.json`). It was independently reviewed, and the result is
presented on the above-floor nodes (see below). Model Qwen3-1.7B-Base, bf16, eager attention, one
CUDA GPU. Transcoders: mwhanna `qwen3-1.7b-transcoders-lowl0`, layers L20..L24.

## Items

All 32 items of the original panel: the first 8 base items by id of each two-hop family (city,
currency, landmark, person). The consumer is `M = logit(answer) - logit(donor answer)` at the
last row. `items.json` and `reference.json` are cut by `make_reference.py` from
the original run's item list and raw output. A full run takes 3.5 minutes,
so the example uses no subset. `MDB_ITEMS=0,1` runs a subset, and the comparison table then
covers only those items. The original run drew its random node with a seeded rng over the whole panel, so
`items.json` stores each item's recorded pick and the example reuses it.

## Runs

Measured on one CUDA GPU (RTX 4080) through our job scheduler.

| Run | What | Wall | Peak VRAM | Peak RSS |
|---|---|---|---|---|
| smoke | item 0 (city:Barcelona). The script exited 0, but the job was marked failed because `run.sh` returned the exit status of a `pipefail` grep with nothing to print. Fixed: `run.sh` now exits with the script's return code. | 14 s | 3.9 GB | 8.5 GB |
| **validation** | full run, 32 items | 211 s | 3.9 GB | 9.6 GB |

Validation log: `validation-job-5141d9c930ba.log` in this directory.

Validation custody, from the sealed records: script sha256 `cf86f551...e5303` (`frozen_gradient_brackets.py` as
validated; its comments were edited afterwards, the code is unchanged) and prereg sha256 `ee85a407...28e01` (`prereg.json`, the original
preregistration copied verbatim). Kernel path `mdb-stepped`. Records written to the sink: 192
`supplied-forecast` (one per node-item edge, sealed before the arms) and 32 `branches` (one per
item, 22-24 arms each).

## Key numbers: validation vs original run, same 32 items (measured)

Both columns come from one rule (`summarise` in the script) applied to the nodes measured now and
to the original run's saved records. "Above floor" means `|arm0| > floor`, with floor = 2 bf16 spacings at the
larger `|logit|` of the pair (0.25 or 0.5 here).

| Number | Original run | Now |
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

These match the original full-panel figures: G1 0.095 above floor, edge share 0.92 / 0.97,
direct share 0.61-0.64, about 1.6x overstatement, Spearman 0.93, and 44/160 with any bracket
above floor.

**Reading (as the independent review framed it).** Most brackets sit at about one bf16 step, so G1's 0.105
"fail" and G2's largest-bracket vote are at quantisation resolution. On the 103 node-items
where removing the node has an above-floor effect:

- The re-routing bracket is under a tenth of the native effect.
- The frozen-gradient edge accounts for 0.92 of the native effect, and 0.97 above 4 floors.
- The transcoder direct edge overstates the native effect by about 1.6x, not 2x.

The frozen-gradient edge explains the gap: it is later layers' linear response through their
MLPs (edge / direct = 0.74).

New in this example (not in the original run), measured in the validation run:

- `supplied.judge` of the sealed edge against the native removal: the signs agree on 102/103
  above-floor node-items.
- The Gaussian matched-norm null (`Branches.add_null("random", like="n0.arm0")`, a seeded
  Gaussian add at node 1's MLP write with node 1's write norm) is within floor on only 17/32
  items. A random *direction* of that norm moves the margin more often than a random active
  *feature* of the same norm, which stays within floor on 26/32. The example reports this null.
  It does not enter any bracket or ratio above.

## Reproduction, node for node (measured)

| Check | Result |
|---|---|
| `Branches` base margin and floor equal to the native run's | 32/32 items |
| nodes (kind, layer, feature) equal to the original run's, random node included | 32/32 items |
| `arm0` (free removal, through `Branches`) bitwise equal to the original run | 192/192 node-items |
| `arm1` (`hold t@last.L>l.p`) bitwise equal to the original run's hand-set `p` | 192/192 |
| machinery null bitwise equal | 192/192 |
| `arm2_f64` (holds `p` + `norm_scale`, float64 readout at the base final scale) vs the original run's `arm2` | max \|diff\| 1.1e-14 |
| `direct` (`component_table`) vs the original run | max \|diff\| 2.7e-15 |
| frozen-gradient edge `-A` (nnsight via `names.to_nnsight`) vs the original run | bitwise on 14/192, max \|diff\| 4.4e-15 |
| `arm2` read through `Branches(final_scale="held")` (bf16 logits) vs `arm2_f64` | max 0.70 floors |

The differences, explained:

- **direct, edge.** mdb's `TranscoderBasis.encode` recomputes the activations in float64 with one
  matmul over the active rows. The original run did the same, but over its own active set, which it screened
  with a bf16 GPU matmul; mdb screens in float32. A different row set, or a different matmul
  shape, changes the BLAS blocking. That would make the activations, and hence `a_s d_s`,
  `direct` and `A = g . a_s d_s`, differ in the last float64 bits. This is the likely cause
  (asserted, not isolated). The gradient trace is the same nnsight computation.
  The bf16 cast of the edit absorbs these bits: `arm0` and `arm1` are bitwise equal on every
  node-item, so the written edits are identical.
- **arm2_f64.** The holds stage exactly the original run's `p`, `a` and `n` values (the bf16 arms are bitwise
  equal). The 1e-14 comes from float64 association order in the readout:
  `(rho_F c) . h` (`reads.readout(...).vector`) against the original run's `rho_F (c . h)`.
- **arm2 through Branches.** `final_held_logits` gives real bf16 logits,
  `lm_head(w_F (h_out rho_F,base).to(bf16))`. The original run read a float64 linear readout instead. The two
  differ by bf16 rounding of the normed state and of the logits: up to 0.70 floors, so under one
  floor. The brackets use `arm2_f64`, as the original run did. This is a choice of readout, not a disagreement
  between holds and the original run.

## mdb objects used, and the original hand-written code they replace

| Original hand-written code | mdb object here |
|---|---|
| safetensors loader, bf16 encode, float64 re-encode, decode, direct effect | `sae.open_transcoders`, `components.transcoder_binding`, `components.component_table` (the partition closes: `partition.holds` asserted) |
| manual `rF` and `c = gamma (W_U[a] - W_U[b])` | `reads.readout(run, Margin).vector` / `.rho` |
| hand-typed nnsight address map | `interop.names.to_nnsight` for every module the gradient trace touches, and for the sealed `address_map` |
| hand-built rho dicts; hand-stepped arm1/arm2 re-setting `p`, `a`, `n` | `run.hold("t@last.L>l.p", source=...)`, `run.hold("t@last.L>l.norm_scale", ...)`, `holds.final_held_logits` (via `final_scale="held"`), `holds.rms_scale` for the nnsight frozen scales |
| branch loop; floor by hand | `dbg.branches(...)`, with arms `n<k>.arm0` / `arm1` (holds p) / `arm2` (holds p + norm_scale, final scale held), `null.L<l>`, `add_null("random")`; shared floor (`verdict.margin_floor`), `verdict.floors`, stability band |
| only `sealed['id']` stored | `dbg.custody(script=, prereg=)`, `dbg.records_to(...)`: full supplied-forecast and branches records with custody |
| (none) | `supplied.seal` + `supplied.judge` for the library edge |

Kept as plain Python, because the API is not built: node nomination, top 5 by direct effect,
the per-item loop, and the recorded random-node pick. The nnsight
frozen-gradient trace stays a library computation (the edge is the library's number).
`symbols.hop_edge` was not used: it splits a *cache-cell* edit, and these edits are MLP-write
edits.

## Not verified

- The run used `nnsight._Frozen` (a private helper of `mdb.interop.nn`, as the original run did).
- circuit-tracer itself is not run. The "attribution-graph edge" is nnsight's frozen gradient,
  the planned path, as in the original preregistration.
- The transcoder-error bracket is deferred, as in the preregistration.
- Single seed. Below about 4 floors, a bracket verdict can flip under another kernel path.
