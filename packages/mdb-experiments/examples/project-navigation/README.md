# Inspect, intervene, retain a finding, rewind

This installed-CLI walkthrough was executed on 2026-10-27 with a random tiny Qwen3,
CPU FP32. It demonstrates debugger mechanics rather than a learned mechanism.

Run `python run_demo.py` from an environment with MDB, Saturn-pub and the decoder
dependencies installed. It invokes the installed `mdb` executable beside Python,
retains the raw CLI responses, verifies that the edit changes the carrier and
restore recovers its exact descriptor, and reopens the saved finding in another
process. It creates `project/` beside the script; use a new directory to repeat.

The [exact debugger script](demo.mdb) does this:

```text
step 2
project callers port:hidden
project trace port:hidden --depth 2
project backtrace hidden
fork zero-carrier
set hidden 0
commit
project finding port:hidden --title "Zeroed carrier on private fork" --kind observation
restore before-edit
```

The recorded writers were `model.embed_tokens` and `decoder_layer.0`. The trace
connected `tokens`, `hidden`, `keys`, `values`, and `layer`. The original and
restored hidden digest was `bb3598d9fbbccc0be1fe61f0f972c792446e69b7a57bdc0478ca681a90066270`;
the zeroed digest was `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca`.
The observation survived a separate CLI process, and a further check reopened
analysis, findings and graph verification with model-framework imports blocked.

- [Checked summary](evidence/summary.json)
- [Raw CLI responses](evidence/raw-session.jsonl)
- [Finding after reopening](evidence/reopened-findings.json)
- [Framework-blocked offline check](evidence/offline-check.json)

To explore interactively after the script:

```sh
mdb public --tiny qwen3 --project ./project --session-id demo --resume
```

The separate pretrained qualification is in
[PROJECT-ANALYSIS-QUALIFICATION.md](../../docs/PROJECT-ANALYSIS-QUALIFICATION.md).

An offline DAG experiment is available as `python explore_dags.py PROJECT
--session-id demo --out ./dag-output`. It reuses `StateGraph` to compare the cyclic
component-name graph with native-cut and versioned-port DAGs, reports missing
provenance, and exports a sealed descriptive graph. It currently handles only
declared single-transition transactions and Act/copy records. See
[DAG-EXPLORATION.md](../../docs/DAG-EXPLORATION.md) for the findings and corrections.

## Qualified symbol slices

Run the new installed-CLI walkthrough with a new output directory:

```sh
python run_symbol_slice.py /tmp/mdb-symbol-slice-demo
```

It names the hidden carrier using native `project symbol-bind`, saves a
`project slice --symbol demo.carrier`, and joins a finding to that exact slice.
After a committed zero edit, the original binding refuses the changed input
context. A bounded downstream slice from the original cut identifies the declared
`replace` consumer. The original slice and finding reopen in fresh processes,
including a check with model frameworks blocked. Raw responses, native project
bytes and a summary remain in the supplied directory.

Executed on 2026-10-28 with random tiny Qwen3, CPU FP32, using installed
saturn-pub 0.4.1 and mdb-runtime 0.1.0 wheels. The nearest declared writer was
`decoder_layer.0`. All assertions passed; see the
[checked summary](evidence/symbol-slice-summary.json). The first two demo-driver
attempts exposed JSON formatting and session-field assumptions; the corrected
driver completed, then completed again with the framework-blocked check added.
Local raw attempts are retained under
`/Users/jakeholl/domains/artifacts/mdb-symbol-slices-2026-10-28*`.

This verifies CLI and custody mechanics. Slices describe whole-slot declared
dependencies, and symbol names remain supplied interpretations. No learned
mechanism or causal effect is established. See [SYMBOL-SLICES.md](../../docs/SYMBOL-SLICES.md)
for qualification, branch ancestry and frontier semantics.
