"""Study S (controlled superposition) worker for Beast. PREREG: saturn/experiments/2026-11-16-cot-latent-binaries/S/.

    python study.py --collect --ref REF --containers CDIR --out result
    python study.py --score --capture DIR --conditions control rot ... [--pooled] [--per-pc] --out result

collect  per program (33 lanes: default + in00..in31), torchx86 hard gated-MLP dispatch: every live
         lane-step's f as bits [N, 24*134] (uint8), decoded class, pc, lane, the dispatch output bits
         and its value word. Saved to out/capture/<program>.pt (read in place by later jobs).
score    per condition (one mixing matrix shared by all programs): raw neurons, PCA, ICA, k-means and
         top-k SAEs, pooled over pcs (vs class indicators) and per pc (vs varying output-bit indicators).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parents[2] / "mdb/src"):
    if p.exists():
        sys.path.insert(0, str(p))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import sup  # noqa: E402

PROGRAMS = ("bsort-O0", "bsort-O2", "fsm-O0", "fsm-O2", "fnv-O0", "fnv-O2")
D = 24 * 134
N_POOLED = 8192            # rows per program for pooled fits (random subsample, seed 0)
MIN_CLASS_ROWS = 10        # class indicators scored when the class has >= this many pooled rows
PCA_M, ICA_M = 96, 24
SAE_PRIMARY = (96, 4)
SAE_GRID = ((24, 4), (96, 4), (384, 4), (1536, 4), (96, 1), (96, 16))
GRID_CONDITIONS = ("control", "rot", "rot_n0.3", "p256", "p64", "p16")
PER_PC_CONDITIONS = ("control", "rot", "p64", "p16")
PC_MIN_ROWS = 33
PC_SAE = (64, 4)
CONDITIONS = ("control", "rot", "rot_n0.1", "rot_n0.3", "rot_n1.0",
              "p3216", "p1024", "p512", "p256", "p128", "p64", "p32", "p16", "p8", "p4")


@functools.lru_cache(maxsize=2)
def mixing(cond: str):
    """(matrix or None, noise alpha)."""
    if cond == "control":
        return None, 0.0
    if cond.startswith("rot"):
        return sup.rotation(D, 1), float(cond.split("_n")[1]) if "_n" in cond else 0.0
    d = int(cond[1:])
    return sup.projection(d, D, 100 + d), 0.0


def inputs_of(cdir: Path, prog: str) -> list:
    out = []
    for r in ["default"] + [f"in{k:02d}" for k in range(32)]:
        gi = json.loads((cdir / prog / r / "meta.json").read_text())["inputs"].get("gt_input")
        out.append(bytes.fromhex(gi["hex"]) if gi else None)
    return out


def collect(a, dev, log):
    from mdb.machines.x86 import Binary, mlp
    names = list(mlp.classes())
    out = a.out / "capture"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"classes": names, "programs": {}}
    for prog in PROGRAMS:
        t0 = time.time()
        b = Binary.load(a.ref / "programs" / prog)
        hard = mlp.capture(b, inputs_of(a.containers, prog), device=dev, cap=20000)
        live = hard.live
        cls = mlp.decoded_class(b, hard.pc)[live]
        f = (hard.gate[..., None] * hard.up.float())[live].reshape(int(live.sum()), D).to(torch.uint8)
        down = hard.down[live]
        sh = torch.arange(64, device=down.device)
        value = (down[:, :64].long() << sh).sum(1)                     # int64 (wraps at bit 63)
        lane = torch.arange(live.shape[0], device=dev)[:, None].expand_as(live)[live]
        tick = torch.arange(live.shape[1], device=dev)[None].expand_as(live)[live]
        rec = {"f": f.cpu(), "cls": cls.cpu(), "pc": hard.pc[live].cpu(), "lane": lane.cpu(), "tick": tick.cpu(),
               "down": down.cpu(), "value": value.cpu()}
        torch.save(rec, out / f"{prog}.pt")
        # check: f's selected block equals down at every row (hard gate, one unit on)
        blk = f.reshape(-1, 24, 134)[torch.arange(len(cls), device=dev), cls.clamp(min=0)]
        meta["programs"][prog] = {"N": len(cls), "lanes": int(live.shape[0]),
                                  "class_counts": {names[c]: int((cls == c).sum()) for c in cls.unique().tolist()},
                                  "f_block_eq_down": bool(torch.equal(blk, down)),
                                  "zero_f_rows": int((f.sum(1) == 0).sum()), "pcs": int(hard.pc[live].unique().numel()),
                                  "seconds": round(time.time() - t0, 1)}
        log(f"{prog}: {json.dumps(meta['programs'][prog])}")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


# --------------------------------------------------------------------------- scoring


def load(cap: Path, dev):
    meta = json.loads((cap / "meta.json").read_text())
    data = {}
    for prog in PROGRAMS:
        r = torch.load(cap / f"{prog}.pt")
        r = {k: v.to(dev) for k, v in r.items()}
        g = torch.Generator().manual_seed(0)
        N = len(r["cls"])
        idx = torch.randperm(N, generator=g)[:N_POOLED].sort().values.to(dev)
        r["sub"] = idx
        data[prog] = r
    return meta, data


def pooled_condition(cond, data, names, dev, steps, log) -> dict:
    M, alpha = mixing(cond)
    progs = list(data)
    Ys, Ls, cnames = {}, {}, {}
    for j, prog in enumerate(progs):
        r = data[prog]
        Y = sup.mix(r["f"][r["sub"]], M)
        if alpha:
            Y = sup.add_noise(Y, alpha, 7 + j)
        Ys[prog] = Y
        cls = r["cls"][r["sub"]]
        keep = [c for c in cls.unique().tolist() if int((cls == c).sum()) >= MIN_CLASS_ROWS]
        Ls[prog] = torch.stack([cls == c for c in keep], 1)
        cnames[prog] = [names[c] for c in keep]
    res = {p: {"d": int(Ys[p].shape[1]), "N": len(Ys[p]), "classes": cnames[p], "methods": {}} for p in progs}
    t0 = time.time()
    for prog in progs:
        Y, L, m = Ys[prog], Ls[prog], res[prog]["methods"]
        cls = data[prog]["cls"][data[prog]["sub"]]
        m["raw"] = sup.summarize(sup.best_cut(Y, L), cnames[prog])
        if M is None and not alpha:              # unmixed bits: a neuron's own firing rule is bit > 0.5
            m["raw_native"] = sup.summarize(sup.native(Y > 0.5, L), cnames[prog])
        m["pca"] = sup.summarize(sup.best_cut(sup.pca(Y, PCA_M)[0], L), cnames[prog])
        m["ica"] = sup.summarize(sup.best_cut(sup.ica(Y, ICA_M, seed=0), L), cnames[prog])
        for k in sorted({L.shape[1], 24}):
            B = sup.kmeans(Y, k, seed=0)
            nat = sup.native(B.bool(), L)
            m[f"kmeans_k{k}" + ("_kc" if k == L.shape[1] else "")] = sup.summarize(nat, cnames[prog])
        res[prog]["zero_f_class_share"] = {n: float(((Y.abs().sum(1) == 0) & L[:, i]).sum() / L[:, i].sum())
                                    for i, n in enumerate(cnames[prog])}
    log(f"  {cond}: baselines {time.time() - t0:.1f}s")
    # SAEs: one group per program, trained together on the concatenation
    from mdb.machines.x86 import mlp
    Ycat = torch.cat([Ys[p] for p in progs])
    sizes = [len(Ys[p]) for p in progs]
    offs = [sum(sizes[:i]) for i in range(len(sizes))]
    groups = [torch.arange(o, o + s, device=dev) for o, s in zip(offs, sizes)]
    for w, k in (SAE_GRID if cond in GRID_CONDITIONS else (SAE_PRIMARY,)):
        t1 = time.time()
        p = mlp.train_topk_sae(Ycat, groups, width=w, k=k, steps=steps, batch=1024)
        for j, prog in enumerate(progs):
            A = mlp.sae_encode(p, j, Ys[prog])
            L = Ls[prog]
            nat = sup.native(A > 0, L)
            rec = sup.summarize(nat, cnames[prog])
            cut = sup.best_cut(A, L, signed=False)
            rec["oracle_cut"] = {"macro_iou": float(cut["iou"].mean()), "macro_precision": float(cut["precision"].mean())}
            Yc = Ys[prog]
            recon = A @ p["W_dec"][j].T + p["b_dec"][j]
            rec["fvu"] = float(((recon - Yc) ** 2).sum() / ((Yc - Yc.mean(0)) ** 2).sum())
            if cond in GRID_CONDITIONS:
                ci = torch.full_like(data[prog]["cls"][data[prog]["sub"]], -1)
                for i in range(L.shape[1]):
                    ci[L[:, i]] = i
                rec["value_split"] = sup.value_split(A > 0, ci, data[prog]["value"][data[prog]["sub"]], nat)
            res[prog]["methods"][f"sae_w{w}_k{k}"] = rec
        log(f"  {cond}: sae w{w} k{k} {time.time() - t1:.1f}s")
    return res


def per_pc_condition(cond, data, dev, steps, log) -> dict:
    from mdb.machines.x86 import mlp
    M, alpha = mixing(cond)
    out = {}
    for j, prog in enumerate(data):
        t0 = time.time()
        r = data[prog]
        Y = sup.mix(r["f"], M)
        if alpha:
            Y = sup.add_noise(Y, alpha, 70 + j)
        upc, inv = r["pc"].unique(return_inverse=True)
        groups = [torch.nonzero(inv == i).flatten() for i in range(len(upc))]
        groups = [g for g in groups if len(g) >= PC_MIN_ROWS]
        Lb, gs = [], []
        for g in groups:
            bits = r["down"][g].float()
            mu = bits.mean(0)
            var = (mu > 0) & (mu < 1)
            if var.any():
                Lb.append(r["down"][g][:, var].bool())
                gs.append(g)
        sums = {m: [] for m in ("raw", "pca", "ica", "kmeans_k8", f"sae_w{PC_SAE[0]}_k{PC_SAE[1]}")}
        class_purity_trivial = all(int(r["cls"][g].unique().numel()) == 1 for g in gs)
        for g, L in zip(gs, Lb):
            Yg = Y[g]
            sums["raw"].append(float(sup.best_cut(Yg, L)["iou"].mean()))
            sums["pca"].append(float(sup.best_cut(sup.pca(Yg, 16)[0], L)["iou"].mean()))
            sums["ica"].append(float(sup.best_cut(sup.ica(Yg, 8, seed=0), L)["iou"].mean()))
            sums["kmeans_k8"].append(float(sup.best_cut(sup.kmeans(Yg, min(8, len(g)), seed=0), L, signed=False)["iou"].mean()))
        key = f"sae_w{PC_SAE[0]}_k{PC_SAE[1]}"
        for s in range(0, len(gs), 64):
            ids = list(range(s, min(s + 64, len(gs))))
            p = mlp.train_topk_sae(Y, [gs[i] for i in ids], width=PC_SAE[0], k=PC_SAE[1], steps=steps, batch=128)
            for jj, i in enumerate(ids):
                A = mlp.sae_encode(p, jj, Y[gs[i]])
                sums[key].append(float(sup.best_cut(A, Lb[i], signed=False)["iou"].mean()))
        out[prog] = {"pcs": len(gs), "pcs_ge33": len(groups), "class_purity_trivially_1": class_purity_trivial,
                     "varying_bits_mean": sum(L.shape[1] for L in Lb) / max(len(Lb), 1),
                     "macro_bit_iou_mean_over_pcs": {m: sum(v) / max(len(v), 1) for m, v in sums.items()}}
        log(f"  per-pc {cond} {prog}: {len(gs)} pcs {time.time() - t0:.1f}s "
            f"{json.dumps({m: round(v, 3) for m, v in out[prog]['macro_bit_iou_mean_over_pcs'].items()})}")
    return out


def score(a, dev, log):
    meta, data = load(a.capture, dev)
    names = meta["classes"]
    a.out.mkdir(parents=True, exist_ok=True)
    rec = {"device": dev, "torch": torch.__version__, "sae_steps": a.sae_steps, "pc_sae_steps": a.pc_sae_steps,
           "capture_meta": meta, "pooled": {}, "per_pc": {}}
    t0 = time.time()
    for cond in a.conditions:
        if a.pooled:
            rec["pooled"][cond] = pooled_condition(cond, data, names, dev, a.sae_steps, log)
            line = {m: round(sum(rec["pooled"][cond][p]["methods"][m]["macro_iou"] for p in PROGRAMS) / len(PROGRAMS), 3)
                    for m in ("raw", "pca", "ica", f"sae_w{SAE_PRIMARY[0]}_k{SAE_PRIMARY[1]}")}
            log(f"{cond} pooled macro IoU mean: {json.dumps(line)} ({time.time() - t0:.0f}s)")
        if a.per_pc and cond in PER_PC_CONDITIONS:
            rec["per_pc"][cond] = per_pc_condition(cond, data, dev, a.pc_sae_steps, log)
        (a.out / "score.json").write_text(json.dumps(rec, indent=1, default=str))
    rec["seconds"] = round(time.time() - t0, 1)
    if dev == "cuda":
        rec["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    (a.out / "score.json").write_text(json.dumps(rec, indent=1, default=str))
    log(f"done {rec['seconds']}s peak {rec.get('peak_vram_mb')} MB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--containers", type=Path)
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--per-pc", action="store_true")
    ap.add_argument("--sae-steps", type=int, default=3000)
    ap.add_argument("--pc-sae-steps", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"torch {torch.__version__} device {dev} {torch.cuda.get_device_name() if dev == 'cuda' else ''}")
    if a.collect:
        collect(a, dev, log)
        if a.capture is None:
            a.capture = a.out / "capture"
    if a.score:
        score(a, dev, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
