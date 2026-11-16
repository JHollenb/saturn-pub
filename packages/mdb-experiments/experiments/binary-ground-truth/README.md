# binary-ground-truth: method side

Drivers for the experiment in `saturn/experiments/2026-11-16-binary-ground-truth/PREREG.md`. The method lives in
`mdb.machines.x86` (branch `mdb-x86`); these scripts only run it on Beast. Scoring is done by someone else, blind.

- `worker.py`: run-1 (`method.recover`: cells from write events, liveness by aligned-resample interchange over the
  32 donor inputs, merges by identical transfer, slots by (pc, operand)) and, with `--grid`, the run-2 patch grid
  (`grid.run_grid`) over the live cells. Writes `result/<program>/method/{cells.npz, symbol_table.jsonl,
  circuit_map.json, table.txt, summary.json}` and `result/<program>/grid/{columns.npz, meta.json}`.
- `demo.py`: the bsort-O2 demo transcript (`result/demo.txt`).
- `submit.py`: stages `src/mdb`, isa-bench's `torchx86` and `isa_bench/state.py`, the scripts and pinned wheels, and
  submits one GPU job (priority 70, `TORCHINDUCTOR_COMPILE_THREADS=1`, 600 s timeout) through isa-bench's launcher.

Inputs read per program: the reference store's `mem.bin`, `trace.npz` (initial state) and `prog.elf` (symbol table:
`gt_input` and entry), and from each binctr container's `meta.json` only `inputs.gt_input` and the steps / stdout /
exit code used as the lane check. No DWARF, webs or copy descriptors are opened.

Run-2 operationalisation choices (ASSERTED, not in the PREREG text): `zero` and `unaligned` do not depend on the
alignment axis (base = reference = the default input), so both alignments share one lane; `unaligned` draws a cell
of the same site from the donor run uniformly, seeded by (seed, cell, donor), independent of time.
