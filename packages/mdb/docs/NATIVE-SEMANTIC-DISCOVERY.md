# Automatic discovery from ordinary native execution

`mdb analyze` implements the first bounded pass of the native semantic-discovery proposal. It collects ordinary documents, fits anonymous predictive coordinates and source nominators, seals forecasts before releasing held-out native computation, and exports public symbols and typed context records. Circuit Tracer is not a dependency. The default path performs zero interventions.

```bash
pip install 'mdb-runtime[autoregressive,mrun]'
mdb analyze prepare documents.jsonl --out corpus.json --seed 20261112
mdb analyze inspect /path/to/completed/run
mdb analyze verify /path/to/completed/run
mdb analyze index /path/to/completed/run --out /path/to/index-input
```

Each JSONL document accepts only `id`, `text`, optional `source`, and optional `group`. Semantic labels, source spans, answers and hand-selected heads are rejected. Exact and word-shingle duplicates are removed before source-group splitting into discovery, development and held-out sets. The frozen corpus records the allocation. Identical token prefixes crossing those splits are refused before collection, since truncation must not silently create input leakage.

`prepare`, `inspect`, `verify` and CLI help require no model frameworks. `run` requires an admitted mrun worker. Numerical execution uses the autoregressive extra; the initial supported capture contract is batch-one unpadded Qwen2/Qwen3, FP32 eager attention, no KV cache. Other numerical profiles need their own qualification.

`index` verifies the run and writes a separate, source-pinned numerical JSONL projection with model/input-qualified citations for existing `mdb evidence` WATCH/XREF ingestion. It leaves the immutable run tree untouched. The standard gate-receipt auto-ingester does not recognize the new forecast manifests directly; the projection supplies observations without fabricating scientific gate receipts. Run `mdb evidence run` and `mdb evidence verify` on the projection with their watch/registry/index paths outside the evidence target/results tree. A standard sweep qualifies that projection's numerical stream and index, while `mdb analyze verify` checks the actual forecast/corpus/artifact contracts.

## What runs

Capture boundaries are distributed evenly across decoder depth, starting at token embeddings and ending before final normalization. Passive hooks retain boundary residuals, actual native attention and value projections at the first block of each span, and a budgeted sample of that block's native MLP neurons. Tokenizer offsets supply physical coordinates. Hooks do not replace native outputs.

Three arms share the same captured evidence: linear features with consumer prediction, sparse features trained on span reconstruction, and sparse features with additional native route/value/output objectives. All fit local span writes and an own-depth state loss. Consumer arms also differentiate the last predicted residual through the frozen native final norm/head against discovery-only native output distributions. The sparse encoder uses positive top-k coordinates; head-specific low-rank queries and keys nominate source distributions. A learned value map and local/read write maps predict each subsequent residual. The read map currently averages predicted head attention, so head-specific routes have auxiliary supervision while the state update shares their mean.

The public `saturn_pub.training.TrainingTransaction` manages complete model, optimizer, gradient, RNG and cursor closure. Each bounded candidate is evaluated against development output distributions, local component behavior and own-depth trajectories. When a native output consumer is supplied, the continuation policy prioritizes its development KL, then own-depth error and local consumer loss. The artifact-only fitter can operate without an output consumer and prioritizes own-depth error in that case. Rejected candidates remain immutable restorable snapshots with their metrics and receipts. The next candidate halves its learning rate after rejection. Physical attempted work remains outside the rollback cursor. These decisions govern continuation; they do not decide whether an effect exists.

After fitting, formation/write/query/key signatures are clustered into anonymous role candidates. This is a statistical grouping, not semantic-role induction. Discovery-only activating examples are retained. A catalog whose selected arm accepted no updates is explicitly marked `initialization-only`. Inactive coordinates produce no proposed source binding.

## Forecast and consumer boundaries

On each held-out document, the only native state supplied to a learned predictor is the paid token-embedding frontier. Later span inputs are its own predicted states. All three arms plus hold-state and discovery-mean-write baselines are globally sealed through `ForecastTransaction` before the full native forward runs. A separate native-neuron ridge diagnostic uses paid local native coordinates and is explicitly excluded from that first-frontier comparison.

Each forecast also sends its predicted final residual through the frozen native final norm and language-model head at the last supplied token. Those logits are sealed before release and compared with the actual native endpoint through full-vocabulary KL and top-token agreement. This pays for original norm/head execution during both fitting and forecasting; it is not an independent model or an exported reasoner. State, route, value, output-consumer and baseline errors remain separate observations.

The forecast transaction uses externally managed eager execution, with no bound-model module guards. The tested predictor callback receives only the initial frontier, fitted student and explicitly paid final norm/head. This is an audited Python visibility boundary, not an adversarial sandbox.

Continuation here means depth spans under supplied token history. It does not mean generated-token continuation, a bidirectional reasoner, a trained 3D-attention architecture, or a measured serving speedup. Every teacher layer still executes for measurement. No output-head score influences checkpoint selection on held-out documents.

## Existing MDB records and knowledge export

The run produces real public `StateCut`, `SymbolBinding`, `SymbolTable`, `SymbolOccurrence`, `SymbolUse` and `SymbolLifecycle` records. The cuts are explicitly passive, non-resumable observation surfaces. They are not fabricated executable decoder sessions. A uniform head/receiver budget samples proposed uses; native attention mass and native top source are retained beside each learned nomination. Attention allocation is not causal necessity or semantic ground truth.

`knowledge-bank.json` uses the existing `mdb.knowledge_interchange.KnowledgeBank` and `KnowledgeRecord` types. Anonymous component records carry ordered argument roles, model/numerical identity, discovery provenance, predictor references, examples and unresolved semantic status. They are typed **contexts**. Asking for them as factual values returns `context-as-value`.

```python
from mdb.knowledge_interchange import CONTEXT, KnowledgeQuery
from mdb.native_discovery_export import load_catalog_bank

bank = load_catalog_bank("run/knowledge-bank.json")
address = bank.records[0].logical_tuple.address
result = bank.resolve(KnowledgeQuery(address, kind=CONTEXT))
```

This supplies an existing typed interchange surface for a later FieldLM adapter. It does not establish a factual ontology, recipient-independent latent coordinates, semantic naming, usable FieldLM retrieval, or independent reasoning over the exported records.

## Reproduce through one finite mrun lease

The repeatable launcher is in `experiments/native_discovery`. Stage it from installed MDB, Saturn-pub and mrun-pub wheels outside the source tree. It stages sealed installed package bytes and explicit input files, uses intent-first resource admission with CUDA declared, strict preflight, one resident model, a finite wall budget and no retries or scheduler children.

```bash
python experiments/native_discovery/submit.py stage /path/to/stage --corpus corpus.json
python experiments/native_discovery/submit.py submit /path/to/stage
python experiments/native_discovery/collect.py /path/to/stage
```

`corpus.py --source-root /explicit/public/tree --out corpus.json` is an optional maintainer extractor. It hash-ranks ordinary Markdown/Python files and takes deterministic character windows; it is not a runtime sibling-checkout dependency. Use disjoint documents for mechanics smoke and fresh exploration. Capture budget, source hashes, physical forwards, candidate attempts, output-head calls and resource measurements remain in the run evidence.

The worker verifies original named parameter/archive retrieval with `retrieve_weights.py` before publication. Pretrained constants remain in the pinned original safetensors archive. Per-run evidence retains new traces, learned predictors, every candidate closure, global seals, actual outcomes, symbols and an immutable hash inventory. A separate parameter-custody receipt extends that inventory without rewriting it. Retrieve the archive, check its external SHA-256, then run `mdb analyze verify` locally. A self-consistent hash chain alone is not external authentication.

## Remaining proposal work

The delivered pass makes the automatic capture → fitting → address nomination → fresh forecast → canonical symbol/context export loop executable. A meaningful positive result still requires predictive advantage over the simple baselines. Broader corpus coverage, recurring computational subgraphs, stable identity across refits, natural-context semantic descriptions, generated-token forecasts, a labeled interpretation audit, a recipient adapter and the 3D-reasoning comparison remain subsequent work. The current metrics preserve partial progress and disagreements without claiming those capabilities.
