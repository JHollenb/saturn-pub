# Addendum 1 to PREREG.md (2026-11-14T00:54:36-08:00): execution only, no rule changes

Written before any main or f3 panel arm has run, so no panel data had been seen.

1. Main-plan leases are repacked. `plans/main.leases.json` (cap 240) put five `night` all-cut branches into lease 1 as replicated copies. `Panel.leases` hashes `replicated` into `lease_sha256`, but `flux_read_panels.check_lease` recomputes the hash without it. The unchanged worker therefore refused lease 1 at startup (job-c1b2bf48ff0d, `PanelError: lease seal differs from its branches`, deterministic, no model load). The new file `plans/main.leases-r2.json` uses cap 156 on the same sealed plan (sha256 unchanged). It splits on word boundaries, red/black/ball and snow/night/rain, has no replicated branches, and has 163 and 164 work items. The arms, sets, floors, metric and forecasts do not change.
2. Main lease 0 (job-aded9ea015e1) and f3 (job-c301a12cb2ae) were `killed_vram` at model load. A wedged-queue clamp lowered their pinned 10,400 MB reservation to 6,309 MB while another session's job held the GPU (tree VRAM 8,314 MB > ceiling 6,940 MB). They are resubmitted unchanged in content, after the GPU goes idle, with an 11,000 MB reservation (measured load peak 8,367 MB × 1.3).
3. Prereg "Jobs" count: the main plan still runs as two jobs.
