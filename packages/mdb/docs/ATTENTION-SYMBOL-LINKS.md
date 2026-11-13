# Native attention linked to canonical symbols

The saved-evidence adapter in `mdb.symbol_attention` registers source bindings and observed query-head uses in an ordinary MDB Project. The offline view in `mdb.symbol_attention_view` navigates from source-attention cells to their symbol definitions and uses, and from source symbols back to their attention readers.

The temperature assay's derived Project and view are in `artifacts/mdb-attention-symbols-20261111/join-02` in the domains workspace. The original native evidence remains unchanged. This profile covers 493 bindings and all 5,040 observed public-Session head uses; Bench remains a separate comparison contract.

## Record path

1. `retained_binding(...)` qualifies supplied `StateAddress` locations against the retained native cut's model, execution, point, state schema and context-slot descriptors.
2. `definition_at(...)` creates a canonical lifecycle and native-point definition. Source-formation definitions reference the supplied input definition; receiver definitions reference the correct native K and V origins separately.
3. `attach_attention(...)` joins each query head to the correct grouped-query K/V locations and native decoder transition. It validates geometry, output-binding support and the original continue receipt's parent/result closure.
4. `project.save_symbol_lifecycle(..., include_slices=False)` persists the ordinary records without attempting the current recursive whole-slot static-slice expansion.
5. `Project(..., read_only=True)` supports reopening, `symbol_lifecycle(...)` and reverse `symbols_at(...)` lookup without model hydration.

Array digests alone do not prove a join. The qualification script independently verifies native K/V origins, captured reader row bytes, the actual attention probabilities, native logits and actual token commits. It does so with NumPy and retained safetensors/pages, without Torch, Transformers or private Saturn imports. The general adapter expects its caller to verify the external arrays.

## Reproduce the saved join

Use the pinned public core and MDB wheels from the artifact index, then run `qualification/link_attention_symbols.py RESULTS ATTENTION_ANALYSIS NEW_OUTPUT`. The output directory must be new. The script copies the original Project, checks every original inventory byte before and after, and adds canonical records only to the copy. Then call:

```python
from mdb.symbol_attention_view import write_view
write_view("NEW_OUTPUT")
```

The view has arm, generation-step, layer and head selectors. Gold source cells carry registered source bindings; other attention columns remain contextual observations. The writer-to-output tab exposes K and V lineage, readout receipts and the actual sample/commit receipt. Full canonical records and arrays remain accessible through their JSON descriptors.

## Boundaries

Semantic names, source rows and task grammar are supplied. An observed attention use does not establish individual-head necessity. The output reference records native readout provenance; it does not assign a separate causal output contribution to every head. This is an explicit integration API and a saved-evidence research view in an isolated worktree, rather than automatic recording and semantic discovery for arbitrary live chat. No inference speedup is claimed.

The same observations can be expressed as backend-neutral read records with an explicit causal status (`project_reads` in `mdb.symbol_reads`); see [SYMBOL-READS.md](SYMBOL-READS.md).
