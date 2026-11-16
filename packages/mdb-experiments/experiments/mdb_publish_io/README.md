# mdb_publish_io (#23): durable publication and streamed residency transport

Measurement payloads from branch `perf/publish-io` (2026-11-13), ported onto main after the
mdb / mdb-experiments / mdb-lanes split. Job outputs were not committed on that branch; the
numbers below are copied from its commit messages (MEASURED on Beast then, not re-run here).

- `bench23a.py`: share of a durable `mdb_lanes.bench` Qwen2.5-0.5B decode spent in publication.
  job-408ecb653532, ext4 NVMe, 32 tokens: 2.60 s total, 0.79 s in 128 fsyncs (4 per token cut);
  1.71 s with fsync a no-op. Led to per-call publication in journal mode
  (`QwenBenchSession.from_resident(command_publication=...)`, mdb-lanes).
- `bench23b.py`: `saturn_pub` streamed residency decode rate, Qwen3-8B BF16 on the RTX 4080.
  Pageable streaming 0.685 tok/s; `pin_host + prefetch` 1.058 tok/s (339 hits / 40 misses),
  1.072 tok/s with `host_embedding`; logits identical to pageable streaming (job-3bc8af989482).
  Before small modules were kept out of the prefetch chain, prefetch was slower than none
  (0.512 vs 0.686 tok/s; job-de7a4e7184c2).

Both scripts expect a payload that ships `saturn_pub` as `./pub` (and, for 23a, `mdb` plus
`mdb_lanes` as `./mdbsrc`) and Beast paths under `/mnt/big`.
