"""Beast worker for run 6 (PREREG): torchx86's dispatch as a gated MLP, read with ``mdb.machines.x86.mlp``.

    python run6.py --ref REF --containers CDIR --programs bsort-O0 ... [--device cuda] --out result

Per program, 33 lanes (default input + in00..in31, the run-1 inputs):
  harness  plain dispatch vs hard gated MLP: pc per tick, T, y, exit equal on every lane
  (a)      blind attribution on f (contribution and ablation per class unit) vs the decoded class;
           soft gate tau in TAUS on 33 lanes each: per-step exact-output rate on the run's own states,
           attribution correct / wrong / none, and run-level equality with the hard run
  sae      top-k SAE, width 4K, k = 4, on f (bits, [K*134]): pooled over pcs, and one per pc
  (b)      usage per lane-step and per-run union (mlp.usage)
  whatif   two gate edits on lane 0 at the first add-class tick (sealed down prediction, then native)

Writes ``out/<program>/run6.json`` (and ``sae_pooled.pt``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parents[2] / "mdb/src"):
    if p.exists():
        sys.path.insert(0, str(p))

import torch  # noqa: E402

TAUS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.001)
EXTRA_TAUS = (0.5, 0.4, 0.35, 0.32)          # exploratory, not in the PREREG grid
SAE_K = 4


def inputs_of(cdir: Path, prog: str) -> list:
    out = []
    for r in ["default"] + [f"in{k:02d}" for k in range(32)]:
        gi = json.loads((cdir / prog / r / "meta.json").read_text())["inputs"].get("gt_input")
        out.append(bytes.fromhex(gi["hex"]) if gi else None)
    return out


def rates(named: torch.Tensor, cls: torch.Tensor) -> dict:
    n = max(len(cls), 1)
    return {"n": len(cls), "correct": int((named == cls).sum()) / n,
            "wrong_class": int(((named >= 0) & (named != cls)).sum()) / n, "none": int((named < 0).sum()) / n}


def by_class(named, cls, names) -> dict:
    out = {}
    for c in cls.unique().tolist():
        m = cls == c
        out[names[c]] = {"n": int(m.sum()), "correct": int((named[m] == c).sum())}
    return out


def forced_sweep(mlp, hard, cl, K, taus) -> list:
    """Soft gate applied to the hard run's own states at every lane-step (teacher-forced): the gate is
    softmax(onehot(class) / tau), exactly as torchx86's soft Dispatch builds it; f = gate * up,
    down = sum f > 0.5. Exact output = down equals the hard output; attribution is blind (f, down)."""
    live = hard.live
    up = hard.up[live]
    out = []
    for tau in taus:
        gate = torch.softmax(torch.nn.functional.one_hot(cl, K).float() / tau, -1)
        ex, nc, na = [], [], []
        for s in range(0, len(up), 8192):
            f = gate[s:s + 8192, :, None] * up[s:s + 8192].float()
            down = (f.sum(1) > 0.5).to(torch.uint8)
            ex.append((down == hard.down[live][s:s + 8192]).all(-1))
            r = mlp.attribute(f, down)
            nc.append(r["named_contribution"])
            na.append(r["named_ablation"])
        ex, nc, na = torch.cat(ex), torch.cat(nc), torch.cat(na)
        out.append({"tau": tau, "prereg_grid": tau in TAUS, "lane_steps": len(ex),
                    "exact_output_rate": float(ex.float().mean()),
                    "contribution": rates(nc, cl), "ablation": rates(na, cl),
                    "ablation_on_inexact": rates(na[~ex], cl[~ex]) if (~ex).any() else None,
                    "contribution_on_inexact": rates(nc[~ex], cl[~ex]) if (~ex).any() else None})
    return out


def sae_block(mlp, X, cls, pcs, K, dev, log) -> dict:
    t0 = time.time()
    N = len(X)
    pooled = mlp.train_topk_sae(X, [torch.arange(N, device=dev)], width=4 * K, k=SAE_K, steps=3000, batch=1024)
    a = torch.cat([mlp.sae_encode(pooled, 0, X[s:s + 8192]) for s in range(0, N, 8192)])
    feats = mlp.purity(a, cls, K)
    live = sorted([f for f in feats if f["fires"]], key=lambda f: -f["fires"])
    recon = torch.cat([a[s:s + 8192] @ pooled["W_dec"][0].T + pooled["b_dec"][0] for s in range(0, N, 8192)])
    Xf = X.float()
    fvu = float(((recon - Xf) ** 2).sum() / ((Xf - Xf.mean(0)) ** 2).sum())
    del recon, Xf
    w = sum(f["fires"] for f in live)
    out = {"pooled": {
        "N": N, "width": 4 * K, "k": SAE_K, "live_features": len(live), "fvu": fvu,
        "purity_top10_mean": sum(f["purity"] for f in live[:10]) / max(len(live[:10]), 1),
        "purity_fire_weighted": sum(f["purity"] * f["fires"] for f in live) / max(w, 1),
        "features_purity_ge_0.9": sum(f["purity"] >= 0.9 for f in live),
        "features_purity_lt_0.5": sum(f["purity"] < 0.5 for f in live),
        "top10": live[:10], "value_split": mlp.value_split(a, X, cls, feats, K)}}
    log(f"  sae pooled {time.time() - t0:.1f}s fvu {fvu:.3f} top10 purity {out['pooled']['purity_top10_mean']:.3f}")
    # one SAE per pc (pcs with >= 33 lane-steps), trained in chunks of groups
    upc, inv = pcs.unique(return_inverse=True)
    groups = [torch.nonzero(inv == i).flatten() for i in range(len(upc))]
    keep = [i for i, g in enumerate(groups) if len(g) >= 33]
    pw, pf, nlive, lifts, multi = 0.0, 0, 0, [], 0
    for s in range(0, len(keep), 64):
        ids = keep[s:s + 64]
        p = mlp.train_topk_sae(X, [groups[i] for i in ids], width=4 * K, k=SAE_K, steps=800, batch=128)
        for j, i in enumerate(ids):
            ai = mlp.sae_encode(p, j, X[groups[i]])
            fi = mlp.purity(ai, cls[groups[i]], K)
            li = [f for f in fi if f["fires"]]
            nlive += len(li)
            pw += sum(f["purity"] * f["fires"] for f in li)
            pf += sum(f["fires"] for f in li)
            vs = mlp.value_split(ai, X[groups[i]], cls[groups[i]], fi, K)
            if vs["mean_bit_agreement_lift"] is not None:
                lifts.append(vs["mean_bit_agreement_lift"])
            multi += len(li) >= 2
    out["per_pc"] = {"pcs": len(keep), "pcs_skipped_lt33": len(groups) - len(keep), "live_features": nlive,
                     "purity_fire_weighted": pw / max(pf, 1),
                     "pcs_with_2plus_live_features": multi,
                     "mean_bit_agreement_lift": sum(lifts) / len(lifts) if lifts else None}
    log(f"  sae per-pc {time.time() - t0:.1f}s pcs {len(keep)} purity {out['per_pc']['purity_fire_weighted']:.3f}")
    return out, pooled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--containers", type=Path, required=True)
    ap.add_argument("--programs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-sae", action="store_true")
    ap.add_argument("--forced-only", action="store_true", help="harness + hard attribution + forced sweep")
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    from mdb.machines.x86 import Binary
    from mdb.machines.x86 import mlp
    from mdb.machines.x86.lanes import Lanes

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"torch {torch.__version__} device {dev} {torch.cuda.get_device_name() if dev == 'cuda' else ''}")
    names = mlp.classes()
    K = len(names)
    for prog in a.programs:
        t0 = time.time()
        torch.manual_seed(0)
        b = Binary.load(a.ref / "programs" / prog)
        ins = inputs_of(a.containers, prog)
        B = len(ins)
        rec = {"program": prog, "lanes": B, "K": K, "classes": list(names), "device": dev, "torch": torch.__version__}
        plain = Lanes(b, ins, device=dev, cap=20000).run(record="pc")
        hard = mlp.capture(b, ins, device=dev, cap=20000)
        Tm = max(plain.pc.shape[1], hard.pc.shape[1])
        pad = lambda x: torch.nn.functional.pad(x, (0, Tm - x.shape[1]), value=-1)
        rec["harness_hard_eq_plain"] = {
            "lanes_equal": int(sum(bool(torch.equal(pad(plain.pc)[i], pad(hard.pc)[i])) and plain.y[i] == hard.result.y[i]
                                   and int(plain.exit_code[i]) == int(hard.result.exit_code[i])
                                   and int(plain.T[i]) == int(hard.result.T[i]) for i in range(B))),
            "lanes": B, "stops": sorted({hard.result.stop_kind(i) for i in range(B)}),
            "T": [int(x) for x in hard.result.T.tolist()]}
        log(f"{prog}: harness {rec['harness_hard_eq_plain']['lanes_equal']}/{B} ({time.time() - t0:.1f}s)")

        # (a) blind attribution under the hard gate
        cls = mlp.decoded_class(b, hard.pc)
        cl = cls[hard.live]
        att = mlp.attribute_capture(hard)
        zero = ~hard.down[hard.live].bool().any(-1)
        rec["a_hard"] = {k.replace("named_", ""): rates(v, cl) for k, v in att.items()}
        rec["a_hard"]["all_zero_output_steps"] = int(zero.sum())
        rec["a_hard"]["all_zero_output_by_class"] = {names[c]: int((cl[zero] == c).sum()) for c in cl[zero].unique().tolist()}
        rec["a_hard"]["by_class_ablation"] = by_class(att["named_ablation"], cl, names)
        rec["a_hard"]["gate_units_on_max"] = int((hard.gate[hard.live] != 0).sum(-1).max())
        log(f"  hard attribution {json.dumps({k: rec['a_hard'][k] for k in ('contribution', 'ablation')})}")
        rec["a_soft_forced"] = forced_sweep(mlp, hard, cl, K, sorted(TAUS + EXTRA_TAUS, reverse=True))
        for r in rec["a_soft_forced"]:
            log(f"  forced tau {r['tau']}: exact {r['exact_output_rate']:.4f} contrib wrong "
                f"{r['contribution']['wrong_class']:.4f} abl wrong {r['ablation']['wrong_class']:.4f}")
        fin = [r for r in rec["a_soft_forced"] if r["prereg_grid"]]
        rec["a_soft_forced_summary"] = {
            "largest_tau_exact_lt_1": max([r["tau"] for r in fin if r["exact_output_rate"] < 1] or [None], key=lambda x: x or 0),
            "largest_tau_attribution_wrong": max([r["tau"] for r in fin if r["ablation"]["wrong_class"] > 0
                                                  or r["contribution"]["wrong_class"] > 0] or [None], key=lambda x: x or 0)}
        if a.forced_only:
            out = a.out / prog
            out.mkdir(parents=True, exist_ok=True)
            (out / "run6_forced.json").write_text(json.dumps(rec, indent=1, default=str))
            log(f"{prog}: forced done {time.time() - t0:.1f}s")
            continue

        # (a) soft gate sweep: one batch, 33 lanes per tau
        taus = torch.tensor(TAUS).repeat_interleave(B)[:, None]
        Tmax = int(hard.result.T.max())
        soft = mlp.capture(b, ins * len(TAUS), mode="soft", tau=taus, device=dev, cap=2 * Tmax + 64)
        scls = mlp.decoded_class(b, soft.pc)
        rec["a_soft"] = []
        for j, tau in enumerate(TAUS):
            sl = slice(j * B, (j + 1) * B)
            live = soft.live[sl]
            up, down, c = soft.up[sl][live], soft.down[sl][live], scls[sl][live]
            exact = (down == up[torch.arange(len(c), device=up.device), c.clamp(min=0)]).all(-1)
            sa = mlp.attribute_capture(mlp.Capture(soft.pc[sl], soft.up[sl], soft.gate[sl], soft.down[sl], None))
            runs_eq = sum(bool(torch.equal(pad(soft.pc[sl][:, :Tm])[i], pad(hard.pc)[i]))
                          and soft.result.y[j * B + i] == hard.result.y[i]
                          and int(soft.result.T[j * B + i]) == int(hard.result.T[i])
                          and int(soft.result.exit_code[j * B + i]) == int(hard.result.exit_code[i]) for i in range(B))
            g_sel = soft.gate[sl][live][torch.arange(len(c), device=up.device), c.clamp(min=0)]
            row = {"tau": tau, "lane_steps": int(live.sum()), "exact_output_rate": float(exact.float().mean()),
                   "gate_selected_mean": float(g_sel.mean()),
                   "contribution": rates(sa["named_contribution"], c), "ablation": rates(sa["named_ablation"], c),
                   "ablation_wrong_on_inexact": rates(sa["named_ablation"][~exact], c[~exact]) if (~exact).any() else None,
                   "runs_equal_hard": runs_eq,
                   "stops": {k: sum(soft.result.stop_kind(j * B + i) == k for i in range(B))
                             for k in {soft.result.stop_kind(j * B + i) for i in range(B)}}}
            rec["a_soft"].append(row)
            log(f"  tau {tau}: exact {row['exact_output_rate']:.4f} contrib wrong {row['contribution']['wrong_class']:.4f}"
                f" abl wrong {row['ablation']['wrong_class']:.4f} runs= {runs_eq}/{B}")
        inexact = [r["tau"] for r in rec["a_soft"] if r["exact_output_rate"] < 1.0]
        wrong = [r["tau"] for r in rec["a_soft"] if r["ablation"]["wrong_class"] > 0 or r["contribution"]["wrong_class"] > 0]
        rec["a_soft_summary"] = {"largest_tau_exact_lt_1": max(inexact) if inexact else None,
                                 "largest_tau_attribution_wrong": max(wrong) if wrong else None}
        del soft

        # (b) usage
        u = mlp.usage(hard, cls, [lp.head for lp in b.loops])
        static = sorted({int(x) for x in b.rows[1:, b.F["cls"]].tolist()})
        rec["b_usage"] = {"lane0": u[0], "lanes": u, "classes_in_decoded_table": [names[c] for c in static]}
        log(f"  usage lane0 {json.dumps({k: v for k, v in u[0].items() if k != 'units'})}")

        # whatif on gate units, lane 0
        run = mlp.MlpRun(b, ins[0], device=dev)
        lane0 = mlp.decoded_class(b, run.cap_.pc[0, :run.T])
        add_ticks = torch.nonzero(lane0 == names.index("add")).flatten().tolist()
        if add_ticks:
            t = add_ticks[0]
            rec["whatif"] = {"tick": t, "attribute": {k: v for k, v in run.attribute(t).items() if k.startswith("named")},
                             "print_down": run.print(f"i{t}.mlp.down"), "print_gate": run.print(f"i{t}.mlp.gate[add]"),
                             "add_off": run.whatif(f"i{t}.mlp.gate[add]", 0),
                             "sub_on": run.whatif(f"i{t}.mlp.gate[sub]", 1)}
            log(f"  whatif {rec['whatif']['add_off']['down_prediction']} {rec['whatif']['add_off']['verdict']} / "
                f"{rec['whatif']['sub_on']['down_prediction']} {rec['whatif']['sub_on']['verdict']}")

        # SAE on f
        if not a.no_sae:
            live = hard.live
            X = (hard.gate[..., None] * hard.up.float())[live].reshape(int(live.sum()), K * mlp.NB).to(torch.uint8)
            rec["sae"], pooled = sae_block(mlp, X, cl, hard.pc[live], K, dev, log)
            out = a.out / prog
            out.mkdir(parents=True, exist_ok=True)
            torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in pooled.items()}, out / "sae_pooled.pt")
        rec["seconds"] = round(time.time() - t0, 1)
        out = a.out / prog
        out.mkdir(parents=True, exist_ok=True)
        (out / "run6.json").write_text(json.dumps(rec, indent=1, default=str))
        log(f"{prog}: done {rec['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
