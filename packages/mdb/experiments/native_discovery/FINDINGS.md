# First native-discovery implementation and observations

The automatic analysis loop runs end to end on Qwen2.5-1.5B base without semantic annotations or interventions. Learned sparse coordinates predict some early boundaries better than simple mean writes, but the tested learned accounts do not show a convincing final-consumer advantage over that baseline. This is an implemented research instrument and a mixed predictive result, not recovered semantic knowledge or a working independent reasoner.

## Specimen and execution

The model is `/mnt/big/llm-models/Qwen2.5-1.5B`, with model identity `6996fe7f165932ebad3c4a159abc7d4f23ea42431d21febfbc4f5a6e23fcf2a9`. Its original safetensors archive is pinned by the parameter manifests and independently verified retrieval scripts in each result.

The measured lane is eager FP32 CUDA, PyTorch 2.13.0+cu130, Transformers 5.14.1, Python 3.14.6, batch one, no cache, raw text without chat templates or added special tokens. Boundaries are layers 0, 9, 19 and 28, ending before final normalization. Each exploration uses 32 features per span, top-k eight, three arms and 60 attempted updates per arm, with development feedback every ten updates. All teacher parameters are frozen. Original tokenizer output IDs and physical offsets are retained; final source additionally records tokenizer asset hashes and prompt-format identity.

Corpus windows come from deterministic hash-ranked existing Markdown and Python files, including historical research scripts. They are ordinary source documents, not a representative broad-language benchmark. Six smoke documents and both twenty-document panels have disjoint source groups. Each exploration has 12 discovery, four development and four held-out documents, truncated to 64 tokens. The study uses supplied token histories, not generated-token continuation.

## Results, with panels kept separate

Mean last-position full-vocabulary native-to-predicted KL over four held-out documents; lower is better:

| Account | Initial panel | New-source panel with output-consumer feedback |
| --- | ---: | ---: |
| Hold embedding state | 108.018 | 130.670 |
| Discovery mean writes | **5.406** | 4.710 |
| Linear features plus consumer objectives | 8.449 | 5.545 |
| Sparse reconstruction | 6.333 | **4.703** |
| Sparse features plus consumer objectives | 6.866 | 5.160 |

Different corpora prevent attributing the between-column change to the feedback modification. The second panel is a fresh test of the revised workflow, not a matched causal comparison of objectives. Within that panel, sparse reconstruction and the consumer arm each beat mean writes on three of four documents, while both regress on the remaining document. The small reconstruction aggregate difference is not an established advantage. All learned arms match the native top token on zero of four documents in the second panel.

In the initial panel, sparse reconstruction's first-boundary relative MSE is 0.0400, 0.0781, 1.0183 and 0.0826, versus approximately 0.984 for mean writes on each document. At the final boundary its mean relative MSE is 0.9371 versus 0.5310 for mean writes; the sparse consumer arm reaches 2.5146. This preserves a substantial early-boundary trend alongside later drift. In the second panel, final relative MSE is 0.4867 for sparse reconstruction and 0.5043 for sparse consumers, versus 0.4811 for mean writes.

The initial continuation policy selects 40 linear updates and all 60 updates in each sparse arm. Its development-only diagnosis motivated adding differentiable feedback from predicted final state through the original frozen native norm/head, with development output KL taking first priority in continuation selection. The second panel accepts ten updates per arm and rewinds five of six candidate intervals per arm. Rejected states and their improvements/regressions remain archived. Each accepted arm improves its development output KL from the 8.0886 mean-write initialization, reaching 7.0621, 7.5730 and 7.2643 for linear, reconstruction and sparse consumer respectively. Those development improvements are feedback evidence, not fresh-evaluation claims.

Each exploration exports 96 feature-coordinate candidates, eight statistical role groups, 96 proposed source bindings and 96 observed uses. These are not counts of independent concepts or proven semantic roles. The canonical symbols bind learned source nominations to exact passive observation locations. Native attention mass and top source remain visible as comparison evidence; attention weights do not establish necessity. Native route KL often remains worse than uniform causal attention, so the source resolver is a particularly weak part of this first pass.

## Work, custody and verification

| Run | Job | Worker seconds | Scheduler seconds | Peak RSS MB | Peak VRAM MB |
| --- | --- | ---: | ---: | ---: | ---: |
| Six-document mechanics smoke | `job-3f6ccd1d56a3` | 49.74 | — | — | — |
| Initial twenty-document panel | `job-7a51d642a6dd` | 97.76 | 123.91 | 8,981.2 | 6,352 |
| Consumer-feedback panel | `job-950c44e99d19` | 95.87 | 126.97 | 9,641.6 | 6,360 |

Each exploration loads the teacher once and runs twenty physical document forwards. The initial panel pays 24 last-token norm/head calls; the revised panel pays 456 across fitting, development evaluation and held-out forecasts. Both fit 180 physical student updates and retain 21 candidate closures plus three accepted closures. All runs use one finite admitted lease, strict preflight, no retries and zero interventions. The first pre-model attempt, `job-d339d14fce34`, failed request-digest verification; its failure remains intact.

Directly collected archives match the job markers. The initial archive SHA-256 is `22a34b09d35e9d3c82c94e284f4922cf266c7963c8ef8c5c7328b5bc57eee04e`; the consumer-feedback archive is `ecf6ffc66180e873def8dee52b239010d82862c3a225bd26be2947a537e60575`. Original named parameter retrieval verifies all 338 required tensors from one original archive without persisting another pretrained weight bank.

Artifact-only audits independently recompute 200 sealed state/output metrics per exploration and verify all 24 saved candidate/accepted closures. They perform zero teacher loads or forwards. Each run's corpus allocation, artifact inventory, forecast seal and parameter-custody extension verify. Standard evidence run/verify succeeds on a separate 80-row numerical projection, with 80 model/input-qualified citations and three boundary addresses per panel. This standard sweep verifies the projection; dedicated analysis verification checks the actual run.

The focused Python 3.12 suite passes 100 checks, including knowledge-interchange coverage, and the new pipeline's eight checks also pass against local PyTorch 2.13. An older Python 3.10 environment fails an unchanged directory-fsync monkeypatch test; the same test passes on Python 3.12. Real-model execution uses the sealed installed wheels, not workspace imports. Subsequent source changes add the offline evidence projection, frozen-model admission guard and more explicit tokenizer/profile metadata; measured source/wheel bytes remain pinned in each payload.

Workspace evidence:

- [Initial panel and audit](../../../../../../artifacts/mdb-native-discovery-20261112/main/)
- [Consumer-feedback panel and audit](../../../../../../artifacts/mdb-native-discovery-20261112/consumer/)
- [Workflow and claim boundaries](../../docs/NATIVE-SEMANTIC-DISCOVERY.md)

## What this establishes and what remains open

Mechanics are valid for the named runs. The working inference is that useful intermediate structure can be learned from ordinary execution, while the current representation, shared read map, source resolver and training horizon do not yet retain the distinctions needed by the final consumer. Sparse reconstruction's early performance is a lead to investigate; a failed final comparison does not establish that sparse concepts cannot work.

The next useful comparison is a matched discovery/development study of formation versus late cancellation, with checkpoint curves, per-head read diagnostics and explicit positional information, followed by another quarantined source set. This does not require intervention panels. Semantic descriptions, recurring computational subgraphs, cross-refit identity, generated-token continuation, actual FieldLM adaptation and a trained 3D workspace remain open. Typed anonymous context export is implemented; an independent knowledge base is not established.
