# Live symbol definitions, uses and source-writer mediation

`mdb.live_symbols.LiveSymbols` explicitly records a persistent `SaturnPubDebuggerAdapter` in the existing canonical symbol/lifecycle indexes. It supplies no semantic discovery or predictor. `observe` qualifies a supplied name/claim against the actual Session ports, token context, native clock and cut. `advance` validates the bindings before executing and joins each use to the native consumer receipt's actual parent/result. A new observation is required at a later execution point. Immutable lifecycle versions retain earlier evidence.

```python
from mdb.live_symbols import LiveSymbols
from mdb.project import Project
from mdb.prompt_qwen import PromptQwenAdapter
from mdb.saturn_pub_adapter import SaturnPubDebuggerAdapter
from saturn_pub import StateAddress

native = PromptQwenAdapter(model).session(prompt_token_ids)
adapter = SaturnPubDebuggerAdapter(native, project=Project(output_dir), session_id="example")
symbols = LiveSymbols(adapter)
argument = symbols.observe("argument.value", (StateAddress("values", (19, 0, 0, source_row)),),
    context="declared-task", claim="Observed source V; role name supplied by this assay.")
receipt = symbols.advance(31, symbols=(argument,))  # Qwen2.5-1.5B: embed + 28 layers + norm + readout
output = symbols.observe("output.next-logits", (StateAddress("logits"),),
    context="declared-task", claim="Actual native logits before sampling.")
symbols.advance(2, symbols=(output,))  # native greedy sample and commit
```

`PromptQwenAdapter` reuses the public native model program. Its opt-in execution/state-schema identities expose KV read/write ports and constrained staging of the next caller-supplied prompt token. Only the unconsumed last token may change at an embedding boundary, with identical prior history. Legacy Qwen cuts are not interchangeable with this execution contract. Future generated tokens require no staging.

`Project(path, read_only=True)` requires an existing schema-3 project and uses SQLite mode=ro; it does not migrate or refresh analyses. Framework-free inspection is available through `symbols`, `symbol_lifecycles`, `symbol_lifecycle`, and `symbols_at`. Live registration defers optional recursive whole-slot slices, whose legacy materializer hits recursion depth on longer operation histories. All existing native metadata joins still run. Ordinary lifecycle saves retain their previous slice-materialization default.

The CLI supports `mdb symbols PROJECT list --session-id SESSION --read-only`, along with `show` and `at` navigation. Read-only mode rejects adding a lifecycle. The default writable constructor retains its historical migration/analysis behavior.

The finite qualification lives in `qualification/live_symbol_closure_worker.py`; stage/submit with `qualification/semantic_scaffold_launch.py --mode closure-smoke` for bump, `--mode closure` for the distinct advance context, or `--mode closure-semantics` for the additive categorization organism. Use explicit helper paths, installed wheels, strict mrun CUDA preflight and one finite resident lease per assay. Each paired context has six source-formation arms, three branch-owned generated tokens, <=240 macro transactions, <=7000 native substeps and 840 seconds. No retries. `semantic_scaffold_collect.py` verifies archive bounds/inventory; `live_symbol_closure_audit.py RESULTS NEW_ANALYSIS_DIR` verifies saved bytes, native argmax/own feedback, typed write scope, source formation/persistence, canonical metadata joins and reverse lookup without importing model frameworks.

Artifact roots: `artifacts/mdb-live-symbol-closure-20261111/`. Coding jobs `job-acf73ee0196b` and `job-03ee5485e2b1` each contain 52 canonical bindings, 494 locations, 102 lifecycle versions and 50 uses. Each context's four writer-background/value cells give donor-answer=donor-value presence. Donor writer input forms donor late V exactly; restoring recipient late V restores the recipient answer despite retained key changes. Direct donor V changes the answer. This is coalition-level conditional mediation, not individual layer necessity, full semantic compilation, general coding, arbitrary chat or speedup.

Observed native full-forward baseline maximum-logit differences range 0.15625–0.25; all four first tokens agree. Lane-local three-token shams are exact. The integer-only Python consumer admits just the first generated literal into a supplied function/assertion, not the rest of generated code. Failed staging/first submitted smoke and corrected saved-only audit diagnostics remain separate from successful native evidence.

Additive categorization `job-d43b9e2d53d9` also gives donor-category=donor-V over four cells, with native cold/hot, donor writer input -> hot, recipient V restoration -> cold, direct donor V -> hot. Matched semantic timing (`--mode closure-timing`) and depth support (`--mode closure-support`) panels each use four arms. Early and delayed late V both produce hot; delayed late and all-depth V both produce hot. This differs from historical Bench copying of boiling. The canonical summary keeps distinct execution contracts and unresolved portability explicit; high saved-value cosine does not establish intervention equivalence.

Five canonical panels contain 231 bindings, 2220 locations, 449 lifecycle versions and 218 uses; 78 native argmax decisions/78 source persistence comparisons/52 own-feedback links/109 payloads are audited. These are three distinct source pairs with two matched semantic replays, not five semantic contexts. Seventy-three focused tests pass. The complete blog, proposal, raw inventories, installed wheels, audits and evidence sweep are linked from the artifact root.

Historical Bench replay `job-376dc1962a22` reproduces boiling and the +8.75 targeted-margin effect; a separate saved audit checks 2285 files/30 decisions/20 feedback links/34 payloads and an exact sham. The semantic interchange disagreement is reproducible across the two execution contracts. It remains an unresolved portability/instrument question, not evidence that either task has no semantic mechanism. The next useful cut freezes the same input, cache and payload across consumers. No runtime or raw artifact was rewritten to remove the disagreement.
