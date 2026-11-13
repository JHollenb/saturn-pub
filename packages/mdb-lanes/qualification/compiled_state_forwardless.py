"""Qualify true resume, K/V injection, registered timing and forwardless serving on Qwen2.5-1.5B.

One finite CUDA worker, two prompt families, replayed from the 2026-11-02
forecast-cost experiment:

addition (Parts E/F): the compiled_state_addition.py family. Compiler fit on the
whole Part E/F train half, scoring on the Part D/E 32,768-row sample. ``measure``
runs full resumes at L20 and L28 for F4, F4|c and F5 plus Part F's T3c partial arms
at L20 (answer positions run L20..27, operand K/V compiled), a full-ladder parity
gate, the registered timing protocol at batch 1 and 256 against the prefix-cached
native forward, the zero-layer F5 chain that feeds its own r0/r1 back, timed
against KV-cached native decode, and the refresh serving loop over a tau sweep.

G1 (Part G): "The capital of {X} is {Y}. The capital of {Z} is", with stage5's RNG
order for the holdout and split. The 4-token zero-layer chain is fit on the train
rows and scored on SEEN (and HO) rows, timed against KV-cached native decode, and
served over the whole 2,256-row stream.

Native tokens are the model's own greedy tokens through mdb's manual layer loop;
ground-truth answers are evaluator references only.
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import time
from pathlib import Path

OUTPUT = Path("forwardless-result")
G1_PREFIX = ("The capital of Nepal is Kathmandu. The capital of Ghana is Accra.\n"
             "The capital of Uganda is Kampala. The capital of Vietnam is Hanoi.\n")
G1_PIECES = ["The capital of", 0, " is", 1, ". The capital of", 2, " is"]
COUNTRIES = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
             ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Kenya", "Nairobi"),
             ("Peru", "Lima"), ("Chile", "Santiago"), ("Cuba", "Havana"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
             ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Israel", "Jerusalem"), ("Turkey", "Ankara"),
             ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Hungary", "Budapest"), ("Portugal", "Lisbon"),
             ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
             ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Switzerland", "Bern"),
             ("Canada", "Ottawa"), ("Argentina", "Buenos"), ("Pakistan", "Islamabad"), ("Afghanistan", "Kabul"),
             ("Thailand", "Bangkok"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("Korea", "Seoul"),
             ("Australia", "Canberra"), ("Tunisia", "Tunis"), ("Bulgaria", "Sofia"), ("Jamaica", "Kingston"),
             ("Malaysia", "Kuala"), ("Taiwan", "Taipei"), ("Wales", "Cardiff"), ("Scotland", "Edinburgh"),
             ("England", "London")]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def support():
    return runpy.run_path(str(Path(__file__).with_name("campaign_support.py")))


def log(*parts):
    print("MDB_FORWARDLESS", *parts, flush=True)


def worker():
    config_raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    config = json.loads(config_raw)
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if request.get("config") != config:
        raise ValueError("worker configuration differs from sealed request")
    for name, expected in request["source_hashes"].items():
        if Path(name).exists() and file_sha(name) != expected:
            raise ValueError("staged source differs: " + name)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import mdb
    from mdb_lanes.compiled_state import (
        FeatureSet,
        PromptFamily,
        Symbol,
        compile_state,
        visibility,
        weights_sha256,
    )
    from mdb_lanes.compiled_state_runner import (
        DecoderSplice,
        Rows,
        _DeviceMoments,
        device_features,
        kv_unrows,
        measure,
        parity,
    )
    from mdb_lanes.compiled_state_serving import ZeroLayerChain, chain_agreement, serve
    from mdb_lanes.payload import seal_package
    from mdb_lanes.timing import time_arms

    add = runpy.run_path(str(Path(__file__).with_name("compiled_state_addition.py")))
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    seed, bs = config["seed"], config["batch"]
    torch.manual_seed(seed)
    root = Path(config["model_path"])
    weights = {p.name: file_sha(p) for p in sorted(root.glob("*.safetensors"))}
    weights_id = sha(json.dumps(weights, sort_keys=True).encode())
    if weights_id != config["expected_weights_sha256"]:
        raise ValueError("checkpoint bytes differ from the pinned identity")
    tok = AutoTokenizer.from_pretrained(root)
    model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, attn_implementation="sdpa").to(config["device"]).eval()
    dev = next(model.parameters()).device
    on_cuda = dev.type == "cuda"
    nL, V = model.config.num_hidden_layers, model.config.vocab_size
    identity = {"name": config["model"], "path": str(root), "weights_sha256": weights_id, "safetensors": weights}
    out = {"schema": "mdb-forwardless-qualification-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
           "request_sha256": request["request_sha256"], "config": config}

    def gate_of(*gates):
        return {"passed": all(g["passed"] for g in gates), "exact": all(g["exact"] for g in gates),
                "rule": gates[0]["rule"], "parts": len(gates)}

    with torch.no_grad():
        # ================================================================ addition (Parts E/F)
        cfg = config["addition"]
        SHOTS, dig, symbol_values = add["SHOTS"], add["dig"], add["symbol_values"]
        family = add["addition_family"]()
        digit = np.array([tok.convert_tokens_to_ids(str(i)) for i in range(10)])
        plus, eq = tok.convert_tokens_to_ids("+"), tok.convert_tokens_to_ids("=")
        prefix = tok(SHOTS, add_special_tokens=False)["input_ids"]
        cls = np.full(V, 10, np.int64)
        cls[digit] = np.arange(10)
        A, B = add["domain"]()
        N = len(A)
        perm = np.random.default_rng(seed).permutation(N)
        train, test = perm[: N // 2], np.sort(perm[N // 2:])
        fit_idx = np.random.default_rng(seed + 5).permutation(train)[: cfg["n_fit"]]
        ev_idx = np.random.default_rng(seed + 1).permutation(test)[: cfg["n_eval"]]
        run = DecoderSplice(model, prefix)

        def suffix8(idx):
            a, b = A[idx], B[idx]
            return np.stack([digit[dig(a, 2)], digit[dig(a, 1)], digit[dig(a, 0)], np.full(len(a), plus),
                             digit[dig(b, 2)], digit[dig(b, 1)], digit[dig(b, 0)], np.full(len(a), eq)], 1)

        def decode_rows(idx, steps):
            ids8 = suffix8(idx)
            toks = [run.decode(ids8[st:st + bs], steps)[0].cpu().numpy() for st in range(0, len(ids8), bs)]
            return ids8, np.concatenate(toks)

        def rows_of(idx, toks, ids8):
            S = A[idx] + B[idx]
            truth = digit[np.stack([dig(S, 2), dig(S, 1), dig(S, 0)], 1)]
            return Rows(np.concatenate([ids8, toks[:, :2]], 1), symbol_values(A[idx], B[idx], cls[toks[:, :2]]), truth)

        t = time.monotonic()
        fit_ids8, fit_toks = decode_rows(fit_idx, 2)
        ev_ids8, ev_toks = decode_rows(ev_idx, 3)
        fit, evaluate = rows_of(fit_idx, fit_toks, fit_ids8), rows_of(ev_idx, ev_toks, ev_ids8)
        log(f"addition native decode {len(fit_idx)} + {len(ev_idx)} rows {time.monotonic() - t:.1f}s")
        F5 = "F5-+totals"  # the zero-layer chain and serving feature set
        report, maps = measure(model, family, prefix, fit, evaluate, layers=cfg["layers"], feature_sets=cfg["feature_sets"],
                               lam=config["lam"], threshold=0.95, batch=bs, r2_rows=cfg["r2_rows"],
                               partial_layers=cfg["partial_layers"], parity_rows=config["parity_rows"],
                               model_identity=identity, log=log)
        save(OUTPUT / "report.json", report)
        ladder = parity(run, family, evaluate.input_ids[:config["parity_rows"]], range(nL + 1),
                        partial_layers=range(nL))
        gate = gate_of(report["parity"]["gate"], ladder["gate"])
        out["addition"] = {"rows": {"fit": len(fit), "eval": len(evaluate), "train_half": len(train),
                                    "test_half": len(test)},
                           "report_fingerprint": report["fingerprint"], "report_sha256": file_sha(OUTPUT / "report.json"),
                           "parity_ladder": ladder, "gate": gate}
        log("addition parity", json.dumps(gate))

        W = {k: torch.as_tensor(m.weights, device=dev) for k, m in maps.items()}
        vis = torch.as_tensor(visibility(family), device=dev, dtype=torch.float32)
        names = [s.name for s in family.symbols]
        score, kv_pos = list(family.score_positions), [p for p in range(len(family.positions))
                                                         if p not in family.score_positions]
        digits = digit.tolist()
        cls_t = torch.as_tensor(cls, device=dev)

        def compiled(P, vals, f, L, positions):
            """Compiled states [B, len(positions), D] at ``positions`` only."""
            sw = family.feature_sets[f].switch
            w = W[(f, L)]
            if not sw:
                return compile_state(P[:, positions], w[positions]).to(run.dtype)
            s0 = family.symbol(sw).visible_from
            return compile_state(P[:, positions], w[:, positions], vals[:, names.index(sw)],
                                 sum(p < s0 for p in positions)).to(run.dtype)

        def chain_of(f, vocab=None):
            return ZeroLayerChain(run, maps[(f, nL)], feedback={"r0": (0, cls_t), "r1": (1, cls_t)}, vocab=vocab)

        def arms_for(bsz):
            r = test[:bsz]
            ids8, toks = decode_rows(r, 3)
            ids8_t = torch.as_tensor(ids8, device=dev)
            ids10 = torch.as_tensor(np.concatenate([ids8, toks[:, :2]], 1), device=dev)
            vals = torch.as_tensor(symbol_values(A[r], B[r], cls[toks[:, :2]]), device=dev)

            def native(vocab=None):
                x, _, _ = run.native(ids10)
                return run.tokens(run.readout(x[:, score], vocab), vocab)

            def full(f, L, vocab=None):
                P = device_features(family, vals, vis)
                if L == nL:
                    return run.tokens(run.readout(compiled(P, vals, f, L, score), vocab), vocab)
                x, _, _ = run.run(compiled(P, vals, f, L, list(range(len(family.positions)))), L)
                return run.tokens(run.readout(x[:, score], vocab), vocab)

            def t3c(f, L, vocab=None):
                P = device_features(family, vals, vis)
                kv = {layer: kv_unrows(torch.einsum("bpf,pfd->bpd", P[:, kv_pos], W[(f, layer, "kv")][kv_pos])
                                       .to(run.dtype), run.kv_heads, run.head_dim) for layer in range(L, nL)}
                x, _, _ = run.run(compiled(P, vals, f, L, score), L, positions=score, kv=kv, kv_positions=kv_pos)
                return run.tokens(run.readout(x, vocab), vocab)

            # headline arms only: [feature set, layer, "digit" | "full" readout]
            head = {"digit": ("+digit_head", digits), "full": ("", None)}
            forward = {"native_prefix_cached": native}
            for f, L, h in cfg["timing_full"]:
                forward[f"{f}({L}){head[h][0]}"] = lambda f=f, L=L, h=h: full(f, L, head[h][1])
            for f, L, h in cfg["timing_t3c"]:
                forward[f"T3c-{f}({L}){head[h][0]}"] = lambda f=f, L=L, h=h: t3c(f, L, head[h][1])
            chain, chain_d = chain_of(F5), chain_of(F5, digits)
            serving = {"native_kv_decode_3": lambda: run.decode(ids8_t, 3)[0],
                       "chain_F5_L28": lambda: chain(vals, margins=False)[0],
                       "chain_F5_L28+digit_head": lambda: chain_d(vals, margins=False)[0]}
            return forward, serving

        timing, chain_timing = {}, config.get("chain_timing", True)
        for bsz in config["timing_batches"]:
            forward, serving = arms_for(bsz)
            timing[str(bsz)] = {
                "forward": time_arms(forward, baseline="native_prefix_cached", gate=gate, reps=config["reps"],
                                     warmup=config["warmup"], cuda=on_cuda, batch=bsz, device=str(dev))}
            if chain_timing:
                timing[str(bsz)]["serving"] = time_arms(serving, baseline="native_kv_decode_3", gate=gate,
                                                        reps=config["reps"], warmup=config["warmup"], cuda=on_cuda,
                                                        batch=bsz, device=str(dev))
            log(f"addition timing b{bsz}", json.dumps({k: round(v["median_ms"], 3) for part in timing[str(bsz)].values()
                                                       for k, v in part["arms"].items()}))
        out["addition"]["timing"] = timing

        ev_vals = evaluate.values
        out["addition"]["chain"] = {
            "feature_set": F5, "layer": nL, "steps": 3, "feedback": {"r0": "step 0 digit class", "r1": "step 1 digit class"},
            "full_vocab": chain_agreement(chain_of(F5), ev_vals, ev_toks, batch=bs),
            "digit_head": chain_agreement(chain_of(F5, digits), ev_vals, ev_toks, batch=bs),
            "native_answer_accuracy": float((ev_toks == evaluate.reference).all(1).mean())}
        log("addition chain", json.dumps(out["addition"]["chain"]["full_vocab"]["exact_through"]))
        out["addition"]["serving"] = [
            serve(run, family, F5, ev_ids8, ev_vals, steps=3, feedback={"r0": (0, cls_t), "r1": (1, cls_t)}, tau=tau,
                  warmup=cfg["serve_warmup"], refit_every=cfg["serve_refit_every"], batch=bs, lam=config["lam"],
                  audit_tokens=ev_toks, log=log)
            for tau in config["taus"]]

        # ================================================================ G1 (Part G)
        g = config["g1"]
        rs = np.random.default_rng(seed)
        nC = len(COUNTRIES)
        X, Z = np.meshgrid(np.arange(nC), np.arange(nC), indexing="ij")
        keep = X != Z
        fill = np.stack([X[keep], X[keep], Z[keep]], 1)
        holdout = sorted(int(x) for x in rs.choice(nC, nC // 5, replace=False))
        if holdout != g["expected_holdout"]:
            raise ValueError(f"G1 holdout replay differs from Part G: {holdout}")
        N1 = len(fill)
        split = rs.permutation(N1) < N1 // 2
        has_ho = np.isin(fill, holdout).any(1)
        train1 = np.flatnonzero(split & ~has_ho)
        seen = rs.permutation(np.flatnonzero(~split & ~has_ho))[:4096]
        held = rs.permutation(np.flatnonzero(~split & has_ho))[:4096]
        country = np.array([tok(" " + c, add_special_tokens=False)["input_ids"] for c, _ in COUNTRIES])
        capital = np.array([tok(" " + k, add_special_tokens=False)["input_ids"] for _, k in COUNTRIES])
        if country.shape != (nC, 1) or capital.shape != (nC, 1):
            raise ValueError("G1 fillers must be single tokens")
        country, capital = country[:, 0], capital[:, 0]
        cols, slot_pos = [], {}
        for piece in G1_PIECES:
            if isinstance(piece, str):
                cols += [np.full(N1, i) for i in tok(piece, add_special_tokens=False)["input_ids"]]
            else:
                slot_pos[piece] = len(cols)
                cols.append((country if piece != 1 else capital)[fill[:, piece]])
        S1 = np.stack(cols, 1)
        T1 = S1.shape[1]
        prefix1 = tok(G1_PREFIX, add_special_tokens=False)["input_ids"]
        texts = [G1_PREFIX + "".join(p if isinstance(p, str) else " " + COUNTRIES[fill[r, p]][1 if p == 1 else 0]
                                     for p in G1_PIECES) for r in range(N1)]
        bad = sum(enc != prefix1 + S1[r].tolist()
                  for r, enc in enumerate(tok(texts, add_special_tokens=False)["input_ids"]))
        setup = {"rows": N1, "T": T1, "prefix_tokens": len(prefix1), "holdout": holdout, "n_train": len(train1),
                 "n_seen": len(seen), "n_ho": len(held), "slot_positions": {"XYZ"[s]: p for s, p in slot_pos.items()},
                 "tokenization_mismatch": int(bad)}
        log("G1 setup", json.dumps(setup))
        if bad or [T1, len(prefix1), len(train1), len(seen), len(held)] != g["expected_shape"]:
            raise ValueError("G1 replay differs from Part G's setup")
        run1 = DecoderSplice(model, prefix1)
        K = g["steps"]
        toks1 = np.concatenate([run1.decode(S1[st:st + bs], K)[0].cpu().numpy() for st in range(0, N1, bs)])
        vocab_steps = [np.unique(toks1[train1, j]) for j in range(K)]
        tables = []
        for j in range(K - 1):
            tab = np.full(V, len(vocab_steps[j]), np.int64)
            tab[vocab_steps[j]] = np.arange(len(vocab_steps[j]))
            tables.append(tab)
        literal = {p: tok.decode([int(S1[0, p])]) for p in range(T1) if p not in slot_pos.values()}
        slot_at = {p: "XYZ"[s] for s, p in slot_pos.items()}
        positions = tuple(slot_at.get(p, f"lit{p}") for p in range(T1)) + tuple(f"gen{j + 1}" for j in range(K - 1))
        fam1 = PromptFamily(
            name="qwen-g1-capital-distractor", prefix=G1_PREFIX, positions=positions,
            symbols=(Symbol("X", nC, slot_pos[0]), Symbol("Y", nC, slot_pos[1]), Symbol("Z", nC, slot_pos[2]),
                     *(Symbol(f"t{j + 1}", len(vocab_steps[j]) + 1, T1 + j) for j in range(K - 1))),
            feature_sets={"slots+prev": FeatureSet(("X", "Y", "Z", *(f"t{j + 1}" for j in range(K - 1))))},
            score_positions=tuple(range(T1 - 1, T1 + K - 1)),
            notes={"literals": json.dumps(literal), "t": "class of the native greedy token at that step among the "
                   "train rows' tokens; the last class is other", "source": "saturn 2026-11-02 payload/stage5.py G1"})
        vals1 = np.concatenate([fill, np.stack([tables[j][toks1[:, j]] for j in range(K - 1)], 1)], 1)
        ids1 = np.concatenate([S1, toks1[:, :K - 1]], 1)
        vis1 = torch.as_tensor(visibility(fam1), device=dev, dtype=torch.float32)
        moments = _DeviceMoments(fam1, [nL], model.config.hidden_size, ["slots+prev"], device=dev)
        for st in range(0, len(train1), bs):
            r = train1[st:st + bs]
            _, finals = run1.decode(S1[r], K)
            v = torch.as_tensor(vals1[r], device=dev)
            moments.add(device_features(fam1, v, vis1), {nL: finals}, v)
        cmap = moments.fit().solve("slots+prev", nL, config["lam"])
        del moments
        feedback = {f"t{j + 1}": (j, torch.as_tensor(tables[j], device=dev)) for j in range(K - 1)}
        subset = sorted(set(np.concatenate(vocab_steps).tolist()))
        chain1, chain1_s = (ZeroLayerChain(run1, cmap, feedback=feedback),
                            ZeroLayerChain(run1, cmap, feedback=feedback, vocab=subset))
        truth1 = capital[fill[:, 2]]
        g1 = {"setup": setup, "family": fam1.fingerprint, "vocab_sizes": [len(x) for x in vocab_steps],
              "readout_subset": len(subset),
              "map": cmap.manifest(weights_sha256(cmap.weights)),
              "parity": parity(run1, fam1, ids1[seen[:config["parity_rows"]]], range(nL + 1),
                               partial_layers=range(nL)),
              "chain": {}}
        cap = g.get("chain_rows") or None  # chain scoring rows per split (all when None)
        for sname, rows in (("SEEN", seen[:cap]), ("HO", held[:cap])):
            for label, ch in (("full_vocab", chain1), ("readout_subset", chain1_s)):
                a = chain_agreement(ch, vals1[rows], toks1[rows], batch=bs)
                pred = np.concatenate([ch(vals1[rows[st:st + bs]], margins=False)[0][:, 0].cpu().numpy()
                                       for st in range(0, len(rows), bs)])
                a.update({"answer_acc_pred": float((pred == truth1[rows]).mean()),
                          "answer_acc_native": float((toks1[rows, 0] == truth1[rows]).mean())})
                g1["chain"][f"{sname}:{label}"] = a
        log("G1 chain", json.dumps({k: v["exact_through"] for k, v in g1["chain"].items()}))
        gate1 = g1["parity"]["gate"]
        timing1 = {}
        for bsz in config["timing_batches"] if chain_timing else ():
            r = seen[:bsz] if len(seen) >= bsz else np.resize(seen, bsz)
            ids_t = torch.as_tensor(S1[r], device=dev)
            v_t = torch.as_tensor(vals1[r], device=dev)
            timing1[str(bsz)] = {
                "serving": time_arms({"native_kv_decode_4": lambda ids_t=ids_t: run1.decode(ids_t, K)[0],
                                      "chain_L28": lambda v_t=v_t: chain1(v_t, margins=False)[0],
                                      "chain_L28+readout_subset": lambda v_t=v_t: chain1_s(v_t, margins=False)[0]},
                                     baseline="native_kv_decode_4", gate=gate1, reps=config["reps"],
                                     warmup=config["warmup"], cuda=on_cuda, batch=bsz, device=str(dev))}
            log(f"G1 timing b{bsz}", json.dumps({k: round(v["median_ms"], 3) for part in timing1[str(bsz)].values()
                                                 for k, v in part["arms"].items()}))
        g1["timing"] = timing1
        order = np.random.default_rng(seed + 7).permutation(N1)
        g1["serving_stream"] = {"order": "default_rng(seed + 7).permutation(rows)", "rows": N1,
                                "rows_with_heldout_fillers": int(has_ho.sum())}
        g1["serving"] = [serve(run1, fam1, "slots+prev", S1[order], vals1[order], steps=K, feedback=feedback, tau=tau,
                               warmup=g["serve_warmup"], refit_every=g["serve_refit_every"], batch=g["serve_batch"],
                               lam=config["lam"], audit_tokens=toks1[order], window=g["serve_window"], log=log)
                         for tau in config["taus"]]
        out["g1"] = g1

    out["environment"] = {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0) if on_cuda else None,
                          "peak_vram_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if on_cuda else None,
                          "mdb_package_sha256": sha(json.dumps({_package: seal_package(_package) for _package in ("mdb", "mdb_lanes")}, sort_keys=True).encode()),
                          "mdb_file": mdb.__file__}
    out["elapsed_s"] = round(time.monotonic() - started, 1)
    save(OUTPUT / "qualification.json", out)
    summary = {"schema": "mdb-forwardless-qualification-summary-v1", "job_id": out["job_id"],
               "request_sha256": request["request_sha256"], "report_fingerprint": report["fingerprint"],
               "qualification_sha256": file_sha(OUTPUT / "qualification.json"),
               "gates": {"addition": out["addition"]["gate"]["passed"], "g1": gate1["passed"]},
               "elapsed_s": out["elapsed_s"], "peak_vram_mb": out["environment"]["peak_vram_mb"]}
    save(OUTPUT / "summary.json", summary)
    log("done", json.dumps(summary))
    support()["emit"](OUTPUT, summary, max_compressed_bytes=32 << 20)


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        if OUTPUT.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-forwardless-qualification-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                       "request_sha256": request["request_sha256"], "error": f"{type(error).__name__}: {error}"}
            save(OUTPUT / "failure.json", failure)
            support()["emit"](OUTPUT, failure, max_compressed_bytes=32 << 20)
        raise
