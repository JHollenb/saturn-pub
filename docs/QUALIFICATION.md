# Qualification record

MDB's standalone extraction is committed. Model-backed qualification is ongoing;
the upstream Saturn assays certify their original source seals, not this package.

| Check | Observed result | Scope |
|---|---|---|
| Fresh installed runtime wheel | 188 passed, 5 skipped before pinned-client follow-up | Tiny deterministic mechanics, upstream imports blocked, outside the checkout |
| Package startup follow-up | 10 installed distribution/launch checks passed | Sealed configurable uv link mode; model execution unchanged |
| Clean pinned mrun dependency | 193 passed, 5 skipped in the final full run | Installed MDB plus the declared mrun Git revision; no workspace package on the import path; stale admission fallbacks refused |
| Dependency-free wheel | Root/session imports and offline CLI work without Torch or mrun | Metadata and evidence tools; model execution requires extras |
| Historical exports | Qwen, Mamba, FLUX and SDXL byte closures verify | CAS references, VM bundles, retained cuts and compatible symbols; continuation is separate |
| Qwen CLI open/resume | Both jobs succeeded, identical session identity | Real BF16/CUDA Qwen, layer cut, inspection, token generation, durable restart |
| Full Qwen parent assay | Timed out at 840 seconds during fresh continuation | All six parent lanes ran and seven resume exports remain; no final aggregate report |
| Retained Qwen cut audit | 1,158 committed cuts agree with expected token and logit bytes; seven closures valid | Offline retained evidence; native comparisons and fresh continuation are separate |
| Qwen portable follow-up | Seven continuations exact, three stock-native comparisons exact; successful exit | Source-deleted parent exports, native and virtual lanes each compared to their own reference |
| Historical Mamba native continuation | Original token-zero cut reproduced token 275 and its FP32/CUDA logits exactly | One-token continuation; continued and closed byte closures valid; outer job timed out after publishing the report |
| Historical FLUX native continuation | Exact image from the original conditioning cut; successful exit | BF16/CUDA Klein 4B, four denoise steps and native VAE; continued/closed stores and independent post-termination archive verify |
| Historical SDXL native continuation | Exact image from the original conditioning cut; successful exit | FP16/CUDA Illustrious XL, four denoise steps and native VAE; independent archive matches the closed report |
| Full Mamba first assay | Native, stepped, replay, fresh resume, debugger and intervention replay passed | Three prompts × 32 tokens; memory test failed to force eviction, so overall qualification remains incomplete |

The full parent assay is `job-697f7259acdb`. Its archive SHA256 is
`6166ab531a94b09d422eb53853996592d38e9f44fdbde9f2dee9ea03074b8896`.
The retained-cut audit SHA256 is
`7be5bea6eb96dab3dba73ec722bd043ea00d3d51f6ab060216848e6ae016dcc4`.
Both are in the local custody ledger. Large archives, model weights and mutable
result directories are excluded from Git.

The portable continuation follow-up copies the parent
exports, validates their closures, checks retained histories, then continues the
saved cuts with the same checkpoint and qualified numerical lane. It does not
repeat the full parent suite. Follow-up `job-4c79dfdd2e7f` succeeded: all seven
continuations and all three stock-native comparisons were exact. See its
[compact record](qualification/qwen-portable-continuation.json). Initial queued job
`job-804bf88da38a` was cancelled before execution to enforce sequential accelerator
assays. The original parent's missing aggregate timing remains unavailable.

The Qwen launcher jobs are `job-da82afae3ee0` (open) and `job-7011a334c1c9`
(resume). They share session `bench-job-da82afae3ee0`.

The retained 32-token references differ across Qwen lanes. Native versus virtual
has 8, 16 and 32 equal token positions across the three prompts; none of the
96 logit digests match across lanes. These are direct comparisons of the parent
resume specification, not a claim about error magnitude or model quality.
Both lanes reproduce their own references in the portable follow-up; cross-lane
equivalence is not established, and cuts cannot be exchanged without conversion.

## Remaining qualification

The original independent review in `qualification/review.json` bound the initial
proof source maps and settings. The final
`qualification/review-ext4-materialized.json` binds the repaired custody guard,
ext4 store and physically materialized cache. Its operational condition requires
the sealed inventory and selected roots to remain unchanged through execution.
The one-token Mamba legacy ABI check and sequential offline image pilots passed. Worker
logs must retain the resolved environment receipt. Long proofs use one lease,
one resident model, strict preflight and no automatic retries.

Remaining work includes full image-family assays, completion of Mamba's memory
recovery, their evidence custody and final reports.
Deployed broker validation and Saturn consumer migration remain separate work.

SDXL pilot `job-a87e433db106` completed in 104.28 seconds. The downloaded archive
independently verifies the same 159 cuts, 97 payload references and closed manifest;
all ten custody records verify. See the
[compact record](qualification/sdxl-legacy-continuation.json).

Mamba full assay `job-eb8c686d27ed` completed its execution panels and fresh-process
resume after source-store removal in 611.42 seconds. Its memory panel retained
sharing and exact earlier-cut restoration, but recorded zero evictions: the
9,450,720-byte test budget exceeded the 6,331,200-byte resident allocation.
The overall failed result is retained. The corrected panel measures each active
closure before shrinking residency, and continues all three siblings privately.

Revised Mamba `job-3148583403bb` reached independent export after its panels but
failed copying receipts on the FUSE/NTFS store with `ENOSPC`. Its intact source
closure independently verifies 427 cuts; its unselected partial export is
explicitly incomplete. The source and failure record are in verified custody.
The host reported 184GB available, so no byte-capacity exhaustion or mrun quota
claim is made. Remaining durable stores use `/mnt/hdd8tb/mdb/qualification` on
ext4, with roughly 3.2TB and 236 million inodes available at preflight.

Focused memory startup `job-8aea8286f1ef` encountered the same FUSE cache-write
error before model loading. `job-f783f97792e2` rejected linked archive roots
before model loading. Both are null model evidence. The corrected ext4 cache
contains physical copies of the successful runtime's selected dependency
archives, with the initial overlay and both materialization inventories retained.
Its inventory digest is bound into the new request and checked before execution.

Focused Mamba `job-d6aa0de67add` completed the corrected memory panel: all three
private suffixes matched, the parent remained unchanged, and the panel recorded
147 evictions and 245 hydrations. Each active 3,150,240-byte closure displaced
inactive state from a 6,300,480-byte union. Export completed, but source removal
refused the new volume because the deletion guard still named only the original
root. The repaired guard accepts exactly the two dedicated qualification roots
and retains job-identity, symlink and traversal rejection. Six isolated installed
guard tests and 21 custody/distribution/launcher checks pass. Continuation
`job-cd60e8ca599d` uses the retained export without repeating this panel.

FLUX pilot `job-b13b8853f65a` used the installed pinned mrun revision and completed
in 189.54 seconds, including a 40.31-second offline package install, store copy,
audits, loading and native continuation. Its post-termination archive matches the
closed report (195 cuts, 128 payload references). See the
[compact record](qualification/flux-legacy-continuation.json).

The SDXL pilot's initial queued request `job-09a793644e0f` was cancelled before
execution to replace its inherited 24GB RAM envelope with a single-case measured
reservation. The sizing record `job-282335c2f7d2` reports RSS 9,890.4; conservatively
normalizing that as MiB, adding 30% headroom plus two 512MiB state allowances, and
rounding to 512MB yields 14,848MB. The replacement `job-a87e433db106` retains the
11,264MB CUDA reservation, strict preflight and no retries. The full six-case
reservation remains unchanged. Both requests and the cancellation are retained.

The first Mamba legacy diagnostic (`job-cae29eb76eb6`) verified its copied byte
closure but refused native resume because the old manifest omitted its numerical
environment. That refusal is retained. The new explicit legacy qualification
requires an exact resident-environment match and preserves the old cut identities.
Queued replacements `job-0ba89163c9ca` and `job-a3890b08da70` were cancelled before
execution to seal the handoff and stage the repaired runtime, respectively. Each
new assay copies into its own job directory and verifies closure before and after
continuation. No automatic retries or parallel accelerator suites are enabled.

Probe `job-7d1f866607f8` was cancelled before bootstrap or model loading after
uv's I/O counters remained unchanged while a thread waited on the fleet cache
filesystem. The corrected finite probe is `job-6fdd2ba65253`, using an explicitly
sealed symlink install mode. No performance improvement is claimed until startup
and model execution are observed. Its package installation completed in 42.29
seconds and the sealed bootstrap checks passed. It subsequently timed out in
closure auditing without a final report. Its reference also incorrectly paired
a token-32 cut with a token-zero expected suffix; no parity claim is made from
that attempt. The corrected probe `job-17a160a09069` pins the original prompt-only
ancestor, checks coordinates before loading, and uses the scoped source audit.
It published a sealed report with exact next-token and logit agreement, valid
continued and closed closures (664 and 665 cuts), and an unchanged original
selection. The outer job timed out after the report; successful worker teardown
and the full 32-token family assay remain separate claims. The report SHA256 is
`adee08343bbe193beff7b8615f8d2d1c08a1a96351f1e294a8f7f3d22041dcfd`.
Its report, logs, configuration, request, scheduler record and resolved environment
are in verified local custody. An independently downloaded post-termination
archive verifies the same closed manifest, 665 cuts, 1,880 payload references and
665 VM bundles. See the [compact record](qualification/mamba-legacy-continuation.json).

The source audit now shares verified ancestry across per-cut source inspections
within one writer-serialized verification call. Independent installed-wheel
forest/source/Mamba/publication checks passed (56 tests). A copied historical
Mamba export retains the same 663 cuts, 1,828 payload references and 663 VM bundles;
later corruption is detected in a new audit. This changes verification work,
not model execution or serialized addresses.

Evidence sweeps over launcher logs currently report `target-streams-only`, with
no scientific receipts recognized. Their clean exit validates indexing/custody
mechanics; it establishes no circuit quality or semantic reasoning claim.
