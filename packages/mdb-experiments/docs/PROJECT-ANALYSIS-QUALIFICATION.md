# Project analysis workflow qualification

On 2026-10-27, mrun job `job-11030cf60aab` completed the finite project workflow
on Beast with Qwen2.5-0.5B-Instruct, CUDA FP32, eager attention, one resident
model, five prompt tokens, eight native transitions, and one explicit hidden-state
write. The worker took 17.92 seconds; total scheduler execution was 47.92 seconds.
The worker's peak allocated CUDA memory was 2,012,951,552 bytes.

The workflow exercised structural analysis, callers/consumers, a bounded dependency
trace, Saturn-pub's backtrace, an intervention fork, commit, a named observation,
and a fresh process reopening the project with Torch, Transformers and mrun imports
blocked. The repeated four-transition hidden state was bitwise equal. The saved
finding and graph survived reopening. This is a bounded workflow mechanics result;
it does not establish stock-model parity, semantic discovery, a causal mechanism,
or correctness of the intervention's downstream behavior.

The first run, `job-fe02ea3228f8`, failed after navigation because MDB had frozen
initial write permissions: `hidden` was absent at session construction, so the
later materialized carrier was rejected as read-only. Its raw failure is retained.
The correction exposes live native address permissions and adds a regression
covering the initially absent carrier, the later permitted write, and revocation
after restoring the pre-embedding cut. Native read-only ports remain protected.
The corrected run was a separately admitted specimen with automatic retries disabled.

Both runs used strict mrun preflight, intent-first resource planning, one finite
CUDA lease, sequential execution, and no automatic retries. The local public
distribution metadata was refreshed to 0.4.0 before staging the corrected payload.
The full project is retained on Beast at
`/mnt/big/mdb/project-analysis/job-11030cf60aab/project`. Scheduler object-store
artifact publication reported an unavailable MinIO endpoint; the collection below
preserves reports and guarded submission/execution receipts directly from the
requested jobs, without claiming successful object-store custody of project bytes.

- [Successful worker report](../results/project-analysis/job-11030cf60aab/report.json)
- [Successful guarded receipt](../results/project-analysis/job-11030cf60aab/custody.json)
- [Successful worker log](../results/project-analysis/job-11030cf60aab/worker.log)
- [Initial guarded receipt](../results/project-analysis/job-fe02ea3228f8/custody.json)
- [Initial failure log](../results/project-analysis/job-fe02ea3228f8/worker.log)

The reusable launcher is `qualification/submit_project_analysis.py`; set the
normal scheduler credentials and explicit worker cache/model roots for the target
host. `--collect JOB_ID` reads that exact job and saves its metadata, log, guarded
receipt and report. It neither follows retry children nor launches another job.

Local validation passed 102 targeted tests, Ruff lint, and installed wheel/sdist
checks. Installed tests exercise analysis and findings outside the checkout with
model frameworks and private workspace modules blocked.
