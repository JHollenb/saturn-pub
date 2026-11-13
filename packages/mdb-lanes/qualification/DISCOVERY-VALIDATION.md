# Discovery debugger validation, 2026-10-30

Sol agents implemented the structural runtime and resident controller in parallel.
Astra reviewed ownership, cuts, handler targeting, stops and the native audit.
Root integrated scripts, CLI, packaging and the executable research assay.

The initial installed-wheel native assay `job-6d0f94e0f628` completed in 9.104 s.
The expanded assay `job-1302a41e1848` completed in 8.440 s and launched the
installed CLI in a fresh Python process. Both used frozen R3 checkpoint
`9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e`,
Beast CPU FP32, one thread, and one finite mrun lease per assay.

After the error-reporting fix, the final installed payload was reconfirmed by
`job-eb41365effc0`: 10.705 s, 14 physical forwards, all eleven native audit
heads exact, the same four stops/continuation, and the same 14-command CLI
script. Process-tree peak RSS was 1744.5 MB. The
[final source-bound record](discovery-final-20261030.json) is the current
qualification; the earlier record remains immutable.

The expanded assay performed six native captures, six independent unchanged
controls, and two independent native audits: **14 physical forward calls**.
Four scalar lanes stopped at `post.1` exit PC369; residual completion was PC370.
The zero-output branch matched a fresh hooked eager execution across all eleven
heads, exactly. The restored branch matched a fresh unchanged native execution
across all eleven heads, exactly. Model weights were unchanged.

Zeroing the MLP output changed five continuous output heads, including register
logits (maximum absolute difference 8.661873817). The decoded branch retained
the same integer result in this context. This intervention establishes bounded
numerical behavior, not a semantic name for the MLP.

The native branch consumed its own predicted state over two instructions:
RDI 252 → 253 → 254, byte offset 0 → 7 → 14. The edited branch demonstrated
audit and one architectural commit. These are separate observations.

The executable CLI script ran 14 commands, fired one breakpoint handler,
forked and edited a branch, restored its parent and executed **773 ATen
primitives**. It made zero `model.forward` invocations; numerical replay still
performed real tensor computation. Its second suffix used 202 primitives from
the retained MLP cut instead of replaying the complete 571-event forward.
This is prefix reuse, not physical batching. Process-tree peak RSS for this
combined worker/CLI run was 1814.2 MB under the measured 2048 MB reservation.

Final installed-wheel verification runs outside the checkout using isolated
Python imports: **44 discovery checks** and **57 existing recording,
intervention, prepared-panel, forward-debugger and distribution checks** passed.
Ruff and `git diff --check` passed. One added regression injects an error after
two native primitives and verifies retained receipts/cut plus a nonzero CLI
exit. This fixes the error being reported as a successful stop.

An earlier launch, `job-51521370f6db`, failed before model loading due to an
incorrect request-digest encoding in the new launcher. Its log and request are
retained. A prior local staging attempt exceeded mrun's logical work-item cap;
it submitted no job. Corrected runs use finite logical work items and separate
primitive budgets. There were no automatic retries.

Run directories, payload wheels, source seals, logs, verified archives and
root recipes are retained in the domains workspace at
`artifacts/mdb-discovery-debugger-20261030/`. MinIO publication was unavailable;
the collector recovered the complete output archive from worker logs and
verified its SHA-256 before extraction. No native success was inferred from
missing remote objects.

The generic evidence-plane sweep reported `evidence-incomplete-empty-target`:
it does not yet ingest discovery journals as native receipt/stream files.
The discovery journal verifier and independent native comparisons are the
authority for this assay. No evidence-plane certification is claimed.

Regenerate the metadata summary with the saved analyzer:

```sh
python qualification/summarize_discovery.py /path/to/run-05 --output /tmp/summary.json
```

The summary joins analyzer, worker, request, report and payload-wheel hashes.
Raw outputs remain immutable; later corrections are separate records.
