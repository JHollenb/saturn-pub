"""Census tables from the ROW lines of job3 logs (python3 census.py LOG [LOG ...])."""
import json
import statistics as st
import sys

rows = []
for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith("ROW "):
            rows.append(json.loads(line[4:]))
rows = list({r["pair"]: r for r in rows}.values())   # a retried job repeats pairs: keep last
print(f"{len(rows)} pairs")
med = lambda xs: st.median(xs) if xs else float("nan")  # noqa: E731

print("\n(a)+(A) reader side at the best V-cell layer of the entity row (native, measured)")
print(f"{'pair':22} {'base':>6} {'cf':>6} {'allKV':>7} {'L*':>3} {'cell':>7} {'K@L*':>6} "
      f"{'Delta':>7} {'Phi':>7} {'E':>7} {'sPhi':>6} {'sE':>6}")
sphi, se, a1 = [], [], 0
for r in rows:
    rd, b = r.get("reader") or {}, r.get("best_cell_layer")
    cell = r["cell"].get(str(b)) if b is not None else None
    d, p, e = rd.get("delta"), rd.get("phi"), rd.get("E")
    s1 = p / d if d else None
    s2 = e / d if d else None
    if d and abs(d) >= 1.0:
        sphi.append(s1), se.append(s2)
    if d is not None and cell is not None and abs(d - cell) <= 0.5:
        a1 += 1
    k = (r.get("cell_k") or {}).get(str(b))
    print(f"{r['pair']:22} {r['base']:6.2f} {r['cf_margin']:6.2f} {r['all_cells']:7.3f} {b!s:>3} "
          f"{cell!s:>7} {k!s:>6} {d!s:>7} {p!s:>7} {e!s:>7} "
          f"{(f'{s1:.2f}' if s1 is not None else '-'):>6} {(f'{s2:.2f}' if s2 is not None else '-'):>6}")
print(f"P-A1 |Delta - cell| <= 0.5: {a1}/{len(rows)};  median sigma_Phi {med(sphi):.3f}, "
      f"median sigma_E {med(se):.3f} over {len(sphi)} pairs with |Delta| >= 1")

print("\nsource side: entity-row residual at L.h_out carried as K/V of later layers "
      "(native, minus the null arm)")
for layer in ("4", "10", "16"):
    full, phi, err = [], [], []
    for r in rows:
        s = (r.get("source") or {}).get(layer) or {}
        if None in (s.get("null"), s.get("full"), s.get("phi"), s.get("E")):
            continue
        n = s["null"]
        full.append(s["full"] - n), phi.append(s["phi"] - n), err.append(s["E"] - n)
    shares = [(p / f, e / f) for f, p, e in zip(full, phi, err) if abs(f) >= 1.0]
    print(f"  L{layer}.h_out: n={len(full)} median full {med(full):.3f} phi {med(phi):.3f} "
          f"E {med(err):.3f};  median phi/full {med([a for a, _ in shares]):.3f} "
          f"E/full {med([b for _, b in shares]):.3f} (n={len(shares)} with |full| >= 1)")
    for r, f, p, e in zip(rows, full, phi, err):
        print(f"     {r['pair']:22} full {f:7.3f} phi {p:7.3f} E {e:7.3f}")

print("\n(B) causal error splice vs direct error share (A = native / -direct)")
for layer in ("8", "14", "18", "20", "22", "24"):
    nat = [r["splice"][layer]["native"] for r in rows if r["splice"].get(layer)]
    amp = [r["splice"][layer]["native"] / -r["splice"][layer]["error_direct"]
           for r in rows if r["splice"].get(layer)
           and abs(r["splice"][layer]["error_direct"] or 0) > 0.05]
    above = sum(abs(x) > 0.5 for x in nat)
    print(f"  L{layer}: median native {med(nat):.3f}  median |native| "
          f"{med([abs(x) for x in nat]):.3f}  |native| > 0.5 on {above}/{len(nat)}  "
          f"median |A| {med([abs(a) for a in amp]):.2f} (n={len(amp)})")
