"""Qualify the compiled-state residual view on Qwen2.5-1.5B three-digit addition.

One finite CUDA worker. The addition prompt family reproduces the 2026-11-02
forecast-cost Parts D/E setup: the same 4-shot prefix, ten suffix positions
(a2 a1 a0 + b2 b1 b0 = r0 r1), the same domain, seed and train/test split, and
the same per-position feature layout and causal visibility. Fit rows are the
first ``n_fit`` of Part H's nested train order; evaluation rows are the first
``n_eval`` of the Part D/E random test sample. r0 and r1 are the model's own
greedy answer tokens. Ground-truth sums are evaluator references only.
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import time
from pathlib import Path

SHOTS = "12+35=47\n481+307=788\n256+139=395\n604+218=822\n"
POSITIONS = ("a2", "a1", "a0", "plus", "b2", "b1", "b0", "eq", "r0", "r1")
OUTPUT = Path("compiled-state-result")


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


def addition_family():
    """Part D/E feature layout as declared symbols (the cg switch is c1 * 2 + c2)."""
    from mdb.compiled_state import FeatureSet, PromptFamily, Symbol

    pos_a, pos_b = {2: 0, 1: 1, 0: 2}, {2: 4, 1: 5, 0: 6}
    symbols = [*(Symbol(f"a{i}", 10, pos_a[i]) for i in (2, 1, 0)),
               *(Symbol(f"b{i}", 10, pos_b[i]) for i in (2, 1, 0)),
               *(Symbol(f"s{i}", 19, pos_b[i]) for i in (2, 1, 0)),
               Symbol("c1", 2, 6), Symbol("c2", 2, 6),
               *(Symbol(f"p{i}", 100, pos_b[i]) for i in (2, 1, 0)),
               Symbol("r0", 11, 8), Symbol("r1", 11, 9),
               Symbol("t1", 20, 6), Symbol("t2", 20, 6), Symbol("cg", 4, 6)]
    digits = ("a2", "a1", "a0", "b2", "b1", "b0", "r0", "r1")
    f4 = (*digits, "s2", "s1", "s0", "c1", "c2", "p2", "p1", "p0")
    return PromptFamily(
        name="qwen-three-digit-addition-4shot", prefix=SHOTS, positions=POSITIONS, symbols=tuple(symbols),
        feature_sets={"F1-digits": FeatureSet(digits), "F4-+pairs": FeatureSet(f4),
                      "F4-+pairs|c": FeatureSet(f4, switch="cg"), "F5-+totals": FeatureSet((*f4, "t1", "t2"))},
        score_positions=(7, 8, 9),
        notes={"r": "r0/r1 are digit classes 0-9 of the native greedy answer tokens, 10 = other",
               "s": "column sums a_i + b_i", "c": "c1 = a0+b0 >= 10, c2 = a1+b1+c1 >= 10",
               "p": "column pairs 10 a_i + b_i", "t": "column totals s1 + c1, s2 + c2",
               "source": "saturn experiments/2026-11-02-forecast-cost-and-compressed-addition payload/stage3.py"})


def dig(x, p):
    return (x // 10 ** p) % 10


def domain():
    import numpy as np

    a = np.arange(100, 1000)
    A, B = np.meshgrid(a, a, indexing="ij")
    A, B = A.ravel(), B.ravel()
    keep = ((A // 100 + B // 100) <= 9) & ((A + B) < 1000)
    return A[keep], B[keep]


def symbol_values(A, B, r_cls):
    import numpy as np

    a = {i: dig(A, i) for i in range(3)}
    b = {i: dig(B, i) for i in range(3)}
    c1 = (a[0] + b[0] >= 10).astype(np.int64)
    c2 = (a[1] + b[1] + c1 >= 10).astype(np.int64)
    cols = [a[2], a[1], a[0], b[2], b[1], b[0], a[2] + b[2], a[1] + b[1], a[0] + b[0], c1, c2,
            10 * a[2] + b[2], 10 * a[1] + b[1], 10 * a[0] + b[0], r_cls[:, 0], r_cls[:, 1],
            a[1] + b[1] + c1, a[2] + b[2] + c2, 2 * c1 + c2]
    return np.stack(cols, 1).astype(np.int64)


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
    from mdb.compiled_state_runner import DecoderSplice, Rows, measure
    from mdb.payload import seal_package

    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(config["seed"])
    root = Path(config["model_path"])
    weights = {p.name: file_sha(p) for p in sorted(root.glob("*.safetensors"))}
    if not weights:
        raise ValueError("a local safetensors checkpoint is required")
    weights_sha256 = sha(json.dumps(weights, sort_keys=True).encode())
    if config.get("expected_weights_sha256") and weights_sha256 != config["expected_weights_sha256"]:
        raise ValueError("checkpoint bytes differ from the pinned identity")
    tok = AutoTokenizer.from_pretrained(root)
    model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    digit = np.array([tok.convert_tokens_to_ids(str(i)) for i in range(10)])
    plus, eq = tok.convert_tokens_to_ids("+"), tok.convert_tokens_to_ids("=")
    prefix = tok(SHOTS, add_special_tokens=False)["input_ids"]
    cls = np.full(model.config.vocab_size, 10, np.int64)
    cls[digit] = np.arange(10)

    # Part D/E split: rs = default_rng(seed); perm = rs.permutation(N); train = first half
    A, B = domain()
    N = len(A)
    perm = np.random.default_rng(config["seed"]).permutation(N)
    train, test = perm[: N // 2], np.sort(perm[N // 2:])
    fit_idx = np.random.default_rng(config["seed"] + 5).permutation(train)[: config["n_fit"]]  # Part H order
    ev_idx = np.random.default_rng(config["seed"] + 1).permutation(test)[: config["n_eval"]]  # Part D/E sample

    run = DecoderSplice(model, prefix)

    def suffix8(idx):
        a, b = A[idx], B[idx]
        return np.stack([digit[dig(a, 2)], digit[dig(a, 1)], digit[dig(a, 0)], np.full(len(a), plus),
                         digit[dig(b, 2)], digit[dig(b, 1)], digit[dig(b, 0)], np.full(len(a), eq)], 1)

    def greedy_rows(idx):
        ids = suffix8(idx)
        out = []
        for st in range(0, len(ids), config["batch"]):
            x = ids[st:st + config["batch"]]
            for _ in range(2):
                logits, _ = run.forward(x, keep=1)
                x = np.concatenate([x, logits[:, -1].argmax(-1).cpu().numpy()[:, None]], 1)
            out.append(x)
        ids10 = np.concatenate(out)
        S = A[idx] + B[idx]
        truth = digit[np.stack([dig(S, 2), dig(S, 1), dig(S, 0)], 1)]
        return Rows(ids10, symbol_values(A[idx], B[idx], cls[ids10[:, 8:10]]), truth)

    fit, evaluate = greedy_rows(fit_idx), greedy_rows(ev_idx)
    # S1: digit-built suffix ids equal the tokenizer's ids for the full prompt text
    probe = range(min(64, len(evaluate)))
    mismatch = sum(tok(SHOTS + f"{A[ev_idx[i]]}+{B[ev_idx[i]]}=" + tok.decode([int(evaluate.input_ids[i, 8])])
                       + tok.decode([int(evaluate.input_ids[i, 9])]), add_special_tokens=False)["input_ids"]
                   != prefix + evaluate.input_ids[i].tolist() for i in probe)
    family = addition_family()
    report, maps = measure(model, family, prefix, fit, evaluate, layers=config["layers"],
                           feature_sets=config["feature_sets"], lam=config["lam"], threshold=config["threshold"],
                           batch=config["batch"], r2_rows=config["r2_rows"],
                           model_identity={"name": config["model"], "path": str(root),
                                           "weights_sha256": weights_sha256, "safetensors": weights},
                           log=lambda m: print("MDB_COMPILED_STATE", m, flush=True))
    save(OUTPUT / "report.json", report)
    summary = {"schema": "mdb-compiled-state-qualification-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
               "request_sha256": request["request_sha256"], "config": config,
               "report_fingerprint": report["fingerprint"], "report_sha256": file_sha(OUTPUT / "report.json"),
               "tokenization_probe": {"n": len(probe), "mismatch": int(mismatch), "prefix_tokens": len(prefix)},
               "native_accuracy_eval": report["native"].get("native_accuracy"),
               "horizons": {f: v["horizon"] for f, v in report["feature_sets"].items()},
               "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                               "mdb_package_sha256": sha(json.dumps(seal_package("mdb"), sort_keys=True).encode()),
                               "mdb_file": mdb.__file__},
               "maps_retained": False,
               "maps_policy": "weights are not archived; every map manifest with its float32 weight digest is in the report",
               "elapsed_s": round(time.monotonic() - started, 1)}
    save(OUTPUT / "summary.json", summary)
    print("MDB_COMPILED_STATE horizons", json.dumps(summary["horizons"]), flush=True)
    support()["emit"](OUTPUT, summary, max_compressed_bytes=32 << 20)
    del maps


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        if OUTPUT.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-compiled-state-qualification-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                       "request_sha256": request["request_sha256"], "error": f"{type(error).__name__}: {error}"}
            save(OUTPUT / "failure.json", failure)
            support()["emit"](OUTPUT, failure, max_compressed_bytes=32 << 20)
        raise
