# Predicted MLP writes and fixed-parent cached consumers

Job `job-64730d03279d` joins the earlier eye-color/shape prompt square to complete single6 MLP prediction and native image continuation. Model inventory SHA is `dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f`, distilled FLUX.2 Klein4B, CUDA BF16, batch1,512px,four steps,seed60665. Four observed native prefixes supply entry state. Four full M fields are sealed before the current step3/single6 computation. The joint incoming prefix is observed, so this is conditional state-to-write prediction.

The [results post](../../../../obsidian/blog/2026-11-04-123512-predicted-writes-and-more-futures-per-forward-pass.md) and [independent saved-only audit](../../../../artifacts/mdb-predicted-future-20261104/analysis-v1.json) separate three outcomes:

| Question | Result |
|---|---|
| Does predicted joint M reproduce measured-M transfer on the same recipient? | Close futures: final RGB MAE0.131655/255 and latent L2 2.22746. Full-field relative RMS errors0.0229–0.0275%. |
| Does that late M-only transfer perform the violet/narrow eye edit? | Eyes remain amber; full intended edit is not achieved. Native prompt square realizes the contrast. Coarse copper hair/plum clothing/framing/background remain, with fine collateral. |
| Can frozen-recipient operands avoid recomputing the full block? | All17 cached image/text endpoints equal independently executed full-block interventions exactly. Cached scope is one parent/time with fixed attention/text/R/gate. |
| Does the predictor save time on this17-case panel? | Cached endpoint timing0.02708s versus instrumented full-block0.17346s; joint prediction adds0.56566s, giving0.59274s total. Setup-inclusive result is unfavorable; no generation speedup. |

The [cached consumer](../qualification/flux_predicted_future.py) reuses observed recipient attention and text in the native fused output projection, replaces image M, and performs the native gated residual update. It enters neither normalization, QKV/MLP input projection, attention nor SwiGLU. It is a qualified research helper, not a general new debugger command or an independently predicted attention path. Five public image Acts connect cached outputs to13 native single blocks, readout, scheduler and VAE. The nine futures comprise four natural references, sham, measured joint M, predicted joint M, additive color+shape M and a fixed token roll.

All17 reference full blocks are counted validation work. The timed baseline retains existing capture/finite/footprint instrumentation; its6.41x ratio against cached evaluation is not a production kernel ratio. A66-candidate crossover is only an extrapolation of these timings. Individual cached endpoints are not predicted full-image futures.

The finite census is one load,four encodes,one noise draw,523 native transitions,five Acts,528 receipts,four source block calls,17 validation block calls,22 cached projections,nine decodes. Worker98.02s,runner140.61s,peak RSS4870.8MB/VRAM10068MB. One strict mrun lease,no retries. Package resolution uses installed public MDB/Saturn-pub/mrun-pub wheels. The [request](../../../../artifacts/mdb-predicted-future-20261104/run-01/request.json) pins source, prompt bank, hypotheses, numerical profile and budgets.

Use installed-wheel Python from outside the checkout. Staging and submission are separate:

```sh
python -I /absolute/MDB/qualification/submit_flux_predicted_future.py FRESH_RUN --prompt-bank EXACT_BANK.json
python -I /absolute/MDB/qualification/submit_flux_predicted_future.py FRESH_RUN --submit
python -I /absolute/MDB/qualification/submit_flux_predicted_future.py FRESH_RUN --collect JOB_ID
python -I /absolute/MDB/qualification/analyze_flux_predicted_future.py FRESH_RUN NEW_AUDIT.json
```

The launcher expects the documented historical bank schema, retains its SHA, and ships only explicit data and public qualification helpers. Use the shared `common exec` launcher when scheduler credentials are needed. Qualified environment choices are `MDB_WORKER_UV_CACHE_DIR=/mnt/big/llm-models/.uv-cache`, `MDB_WORKER_TORCH_HOME=/mnt/hdd8tb/mdb/torch`, and `MDB_WORKER_MODEL_ROOTS=/mnt/big/llm-models`. The worker uses declared SSD scratch and retains durable native store operations.

Audit coverage: complete custody/store/source hashes, all528 receipts, five image-only footprints, all full-field native error counts,17 source-bound native endpoint comparisons and all final latent/PNG bindings. Independent entry-equation reconstruction covers8positions×32neurons per source. This audit does not independently reexecute full M fields or trace GPU kernels. Visual review is explicitly unblinded. Three curated records retain ten model/parent-qualified addresses and28 numeric rows; the offline evidence run/verify/query succeeds. Automatic family induction remains open.

The next discriminating semantic cut holds the predicted field fixed while restoring context/image/history parents separately. The optimization target is cheaper field construction or sufficient reuse to repay its cost. These remain separate from the demonstrated local numerical compatibility.
