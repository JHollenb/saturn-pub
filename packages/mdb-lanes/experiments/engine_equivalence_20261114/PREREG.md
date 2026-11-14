# Qwen read panel: lane path vs engine path, FP32 (sealed 2026-11-14 before submission)

One question, one job (mrun, Beast, priority 70, one Qwen2.5-1.5B FP32 eager load, no retries).

**Question.** On the Qwen read-panel case (recipient prompt, receiver row 21, source rows 18 and 2,
first-position hot - cold margin), does the engine path (`mdb.attach` stepped `Run` with
`prefill=19`, `mdb.trace`, `propagate.whatif_many` zero cut `p[h,s] = 0`) give the same FP32 bits
as the lane path (`attention_port.AttentionPort(QwenBackend)` driven by the sealed FP32 read plan
8590b4f7...)?

Compared, each bit for bit (`torch.equal` on the FP32 tensors; margin as Python float):

1. base: the lane's native first-position logits vs the engine run's `t21.logits`;
2. attention: the lane's FP32 recompute (`Read.probs`) at row 21 vs `mdb.trace` native `L*.p` of
   the lane's own forward vs the stepped run's `t21.L*.p`;
3. cuts: the 24 single-layer zero-cut `floor_cut_recompute` arms of the plan (12 distinct
   (layer, head, source) cuts) measured by the engine vs the same arms on the lane path; and the
   engine vs the lane's default `delta` arms (`z:...` without `/cut_recompute`, 24 arms).

**Prediction (from the tiny-model test `tests/test_qwen_engine.py`, CPU):** 1 and 2 equal; 3 equal
for the recompute arms; delta arms differ in the last bits. Any other outcome is reported as
measured, with the largest difference. The lane margins are also checked offline against the
saved rows of jobs job-b33e6b98c328, job-4a433df3f30b, job-7c3232ad65bb (run-to-run determinism).

Out of scope: BF16, renormalising cuts, set cuts over several layers, keep arms (lane path only).
