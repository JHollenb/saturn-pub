"""SAELens and TransformerLens find a feature; Saturn ships a replayable ablation.

Division of labor, not reimplementation:

* TransformerLens runs GPT-2 small and exposes the residual stream; SAELens decodes it
  into features. They are good at this, so they own step 1: find the top-active SAE
  feature at the last position of a prompt.
* Saturn loads the *same* Hugging Face weights, proves that its ``layer:L`` boundary
  carrier equals TransformerLens ``blocks.L.hook_resid_pre`` at the last position
  (measured, not assumed), ablates that one feature on the carrier, runs the native
  remaining model, and seals parent/candidate cuts so the intervention replays from a
  receipt in a fresh process.

Run from the checkout (GPT-2 small, CPU, offline; needs the ``interop`` extra)::

    pip install -e '.[interop]'
    python examples/interop_saelens.py

The SAE release ``gpt2-small-res-jb`` was trained on ``center_writing_weights=True``
activations, but this example uses ``from_pretrained_no_processing`` so the residual
stream is comparable to Saturn's raw native weights. That makes feature *magnitudes*
approximate; it does not touch the alignment, ablation-agreement, or replay claims,
which use the same no-processing residual, activation, and decoder direction on both
sides. Weights come from the local Hugging Face cache; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

from saturn_pub import Act, Session  # noqa: E402
from saturn_pub.adapters import load as load_adapter  # noqa: E402
from saturn_pub.investigation import Investigation  # noqa: E402
from saturn_pub.store import LocalStore  # noqa: E402

# The prompt/layer below were chosen honestly from a small sweep (see the module-level
# PROMPTS_TRIED note). GPT-2 small actually predicts " London" here; removing a single
# SAE feature flips the top token to " Paris".
DEFAULT_MODEL = "gpt2"
DEFAULT_LAYER = 6
DEFAULT_PROMPT = "The Eiffel Tower is located in the city of"
GENERATION = 8

# Other prompt/layer pairs tried while choosing the headline (all measured, none hidden):
PROMPTS_TRIED = (
    "The Eiffel Tower is located in the city of (layers 6, 8)",
    "When John and Mary went to the store, John gave a drink to (layers 6, 8)",
    "The capital of France is (layers 6, 8)",
    "The quick brown fox jumps over the lazy (layers 6, 8)",
    "My favorite color is (layers 6, 8)",
    "The doctor picked up the scalpel and (layers 6, 8)",
)


def hook_name(layer: int) -> str:
    """The TransformerLens hook Saturn's ``layer:L`` boundary is aligned against."""
    return f"blocks.{layer}.hook_resid_pre"


def saturn_step_to_layer(adapter, token_ids, layer):
    """Step a native session to the ``layer:L`` boundary (carrier == resid_pre[L])."""
    session = adapter.session(list(token_ids))
    session.continue_(1 + layer)  # one embed transition, then L decoder layers
    return session


def ablation_act(direction, activation) -> Act:
    """Subtract ``activation * direction`` from the residual carrier at ``layer:L``."""
    vector = direction.reshape(1, 1, -1).to(dtype=torch.float32)
    return Act.add("hidden", vector, dose=-float(activation), name="sae-feature-ablation")


def _tl_ablation_hook(layer, direction, activation):
    def hook(resid, hook):  # noqa: A002 - TransformerLens passes hook= by keyword
        resid[:, -1, :] = resid[:, -1, :] - float(activation) * direction
        return resid

    return hook_name(layer), hook


def _tail(session, prompt_len):
    return session.read("tokens")[0].tolist()[prompt_len:]


def replay(directory, parent_id, candidate_id):
    """Fresh-process: rebuild the adapter, load both cuts, run the native suffix."""
    torch.set_num_threads(1)  # match the execution ABI recorded when the cut was saved
    torch.set_grad_enabled(False)
    adapter = load_adapter(DEFAULT_MODEL)
    store = LocalStore(directory)
    out = {}
    for name, identifier in (("native", parent_id), ("candidate", candidate_id)):
        cut = store.load(identifier)
        session = Session.from_cut(adapter, cut)
        prompt_len = session.read("tokens").shape[1]
        adapter.generate(session, GENERATION)
        out[name] = session.read("tokens")[0].tolist()[prompt_len:]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/interop-saelens")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--replay", nargs=3, metavar=("DIR", "PARENT", "CANDIDATE"))
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(*args.replay)))
        return

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    layer = args.layer

    # --- Step 1: TransformerLens + SAELens own feature discovery -----------------------
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    tl = HookedTransformer.from_pretrained_no_processing(DEFAULT_MODEL).eval()
    loaded = SAE.from_pretrained("gpt2-small-res-jb", hook_name(layer), device="cpu")
    sae = loaded[0] if isinstance(loaded, tuple) else loaded

    tokens = tl.to_tokens(args.prompt)  # BOS-prepended, matching the SAE's training
    token_ids = tokens[0].tolist()
    _, cache = tl.run_with_cache(tokens, names_filter=hook_name(layer))
    resid_pre = cache[hook_name(layer)]  # (1, seq, d_model)
    feature_acts = sae.encode(resid_pre)[0, -1]
    feature = int(feature_acts.argmax())
    activation = float(feature_acts[feature])
    direction = sae.W_dec[feature].detach().clone()  # (d_model,)

    # --- Step 2: Saturn boundary equals the TransformerLens hook (measured) ------------
    adapter = load_adapter(DEFAULT_MODEL)
    session = saturn_step_to_layer(adapter, token_ids, layer)
    boundary = session.inspect().boundary
    saturn_carrier = session.read("hidden")[0, 0]
    tl_carrier = resid_pre[0, -1]
    alignment_max_abs = float((saturn_carrier - tl_carrier).abs().max())

    # --- Step 3: fork native vs feature-ablated candidate, run the native suffix -------
    parent = session.capture()
    native = session.fork(parent)
    candidate = session.fork(parent)
    act = ablation_act(direction, activation)
    ablate_receipt = candidate.apply(act)
    candidate_cut = candidate.capture()  # ablated carrier, still at layer:L

    adapter.generate(native, 1)
    adapter.generate(candidate, 1)
    native_logits = native.read("logits")[0]
    candidate_logits = candidate.read("logits")[0]

    # Cross-check: a TransformerLens hook doing the same ablation should agree.
    tl_clean = tl(tokens)[0, -1]
    tl_ablated = tl.run_with_hooks(
        tokens, fwd_hooks=[_tl_ablation_hook(layer, direction, activation)]
    )[0, -1]
    clean_gap = float((native_logits - tl_clean).abs().max())
    ablation_gap = float((candidate_logits - tl_ablated).abs().max())

    def topk(logits, k=5):
        values, indices = logits.topk(k)
        return [
            {"token": int(i), "text": tl.to_string([int(i)]), "logit": float(v)}
            for v, i in zip(values, indices)
        ]

    top_native = topk(native_logits)
    top_candidate = topk(candidate_logits)

    # Optional dose grid (native == dose 0, full ablation == dose -1).
    grid = Investigation.doses(
        f"How does feature {feature} at {hook_name(layer)} move the next token?",
        "hidden",
        (activation * direction).reshape(1, 1, -1).to(torch.float32),
        [0.0, -0.5, -1.0, -1.5],
        steps=adapter.layers - layer + 1,  # remaining layers + fused readout
    )
    dose_panel = grid.run(
        session,
        lambda branch: {
            "next_token": int(branch.read("tokens")[0, -1]),
            "next_text": tl.to_string([int(branch.read("tokens")[0, -1])]),
            "top_logit": float(branch.read("logits").max()),
        },
        parent=parent,
    )

    # Full continuations for a human-readable diff.
    native_tail_session = session.fork(parent)
    adapter.generate(native_tail_session, GENERATION)
    candidate_tail_session = session.fork(parent)
    candidate_tail_session.apply(act)
    adapter.generate(candidate_tail_session, GENERATION)
    native_tail = _tail(native_tail_session, len(token_ids))
    candidate_tail = _tail(candidate_tail_session, len(token_ids))

    # --- Step 4: custody + fresh-process replay ----------------------------------------
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    store = LocalStore(directory)
    parent_admission = store.save_report(parent)
    candidate_admission = store.save_report(candidate_cut)
    parent_id = parent_admission["cut"]
    candidate_id = candidate_admission["cut"]
    store.receipt(ablate_receipt)
    for receipt in (*native_tail_session.receipts, *candidate_tail_session.receipts):
        store.receipt(receipt)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--replay",
            str(directory),
            parent_id,
            candidate_id,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    fresh = json.loads(completed.stdout)
    replay_native_equal = fresh["native"] == native_tail
    replay_candidate_equal = fresh["candidate"] == candidate_tail
    replay_ok = replay_native_equal and replay_candidate_equal
    if not replay_ok:
        raise RuntimeError("fresh-process replay did not reproduce the continuations")

    runtime_s = time.perf_counter() - started
    report = {
        "specimen": "GPT-2 small, CPU float32, TransformerLens no_processing + HF native",
        "prompt": args.prompt,
        "tokens": token_ids,
        "sae": {
            "release": "gpt2-small-res-jb",
            "hook_point": hook_name(layer),
            "d_sae": int(sae.W_dec.shape[0]),
            "trained_processing": "center_writing_weights=True (magnitudes approximate here)",
        },
        "feature": {"id": feature, "activation": activation},
        "alignment": {
            "saturn_boundary": boundary,
            "transformer_lens_hook": hook_name(layer),
            "claim": f"Saturn '{boundary}' carrier[-1] == TransformerLens {hook_name(layer)}[:, -1]",
            "max_abs_diff": alignment_max_abs,
        },
        "next_token": {
            "native_top": top_native,
            "candidate_top": top_candidate,
            "native_argmax": top_native[0]["token"],
            "candidate_argmax": top_candidate[0]["token"],
            "flipped": top_native[0]["token"] != top_candidate[0]["token"],
        },
        "cross_check": {
            "clean_tl_vs_saturn_max_abs": clean_gap,
            "ablated_tl_vs_saturn_max_abs": ablation_gap,
            "tolerance": 5e-3,
            "within_tolerance": ablation_gap < 5e-3,
        },
        "continuation": {
            "native_tokens": native_tail,
            "candidate_tokens": candidate_tail,
            "native_text": tl.to_string(native_tail),
            "candidate_text": tl.to_string(candidate_tail),
            "changed": native_tail != candidate_tail,
        },
        "dose_grid": [
            {
                "arm": row["name"],
                "role": row["role"],
                "next_text": row.get("metrics", {}).get("next_text"),
                "status": row["status"],
            }
            for row in dose_panel["rows"]
        ],
        "custody": {
            "parent_cut": parent_id,
            "candidate_cut": candidate_id,
            "parent_admission": parent_admission,
            "candidate_admission": candidate_admission,
        },
        "fresh_process_replay": {
            "native_byte_equal": replay_native_equal,
            "candidate_byte_equal": replay_candidate_equal,
            "verdict": "exact" if replay_ok else "mismatch",
        },
        "prompts_tried": list(PROMPTS_TRIED),
        "runtime_seconds": round(runtime_s, 2),
        "environment": dict(adapter.execution["environment_versions"]),
        "limitations": (
            "Feature magnitudes use no_processing residuals (SAE was trained on centered "
            "activations); the alignment, ablation-agreement, and replay claims do not depend "
            "on that. TransformerLens and Saturn use different block kernels, so cross-check "
            "gaps are floating-point, not zero."
        ),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    nat0, can0 = top_native[0], top_candidate[0]
    print(
        f"SAE {hook_name(layer)} feature {feature} (activation {activation:.2f}) at the "
        f"last position.\n"
        f"Address alignment: Saturn '{boundary}' carrier == TL {hook_name(layer)}[:, -1], "
        f"max abs diff {alignment_max_abs:.2e}.\n"
        f"Next token: native {nat0['text']!r} (logit {nat0['logit']:.2f}) -> "
        f"ablated {can0['text']!r} (logit {can0['logit']:.2f}); "
        f"top-1 flipped={report['next_token']['flipped']}.\n"
        f"TL-vs-Saturn next-token logits: clean {clean_gap:.2e}, ablated {ablation_gap:.2e} "
        f"(tolerance 5e-3).\n"
        f"Native continuation : {tl.to_string(native_tail)!r}\n"
        f"Ablated continuation: {tl.to_string(candidate_tail)!r}\n"
        f"Fresh-process replay byte-equal: native={replay_native_equal}, "
        f"candidate={replay_candidate_equal}.\n"
        f"Report: {directory / 'report.json'} ({runtime_s:.1f}s)"
    )


if __name__ == "__main__":
    main()
