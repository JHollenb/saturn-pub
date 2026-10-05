"""Small experiments over an admitted caller-owned BenchSession.

The --tiny command uses random CPU models only to teach and check mechanics.
Real models belong in an mrun worker; these functions never acquire a lease.
"""

import argparse
import json
from pathlib import Path


def _signature(rows):
    return [(row["token_id"], row["logits_sha256"]) for row in rows]


def _exact(comparison):
    # A named runtime replay assertion; never a scientific effect-size gate.
    return all(value for key, value in comparison.items() if key.startswith("same_"))


def rewind_and_replay(session, tokens=4):
    """Resume an in-flight token at a layer boundary and compare its suffix."""
    original = session.checkpoint("lab-history-parent")
    session.step("embedding")
    session.step("layer")
    partial = session.checkpoint("lab-partial-layer")
    first = session.generate(tokens)
    left = session.checkpoint("lab-uninterrupted-end")
    second = session.replay(partial, max_new_tokens=tokens)
    right = session.checkpoint("lab-replayed-end")
    comparison = session.compare(left, right)
    if _signature(first) != _signature(second) or not _exact(comparison):
        raise RuntimeError("same-lane partial-cut replay differs")
    session.restore(original)
    return {"partial_cut": partial, "left_cut": left, "right_cut": right,
            "rows": first, "comparison": comparison, "exact": True}


def residual_intervention(session, scale=0.9, tokens=4):
    """Retain baseline, aborted preview and committed candidate from one cut."""
    import torch

    original_branch = session.inspect()["branch"]
    original = session.checkpoint("lab-intervention-parent")
    session.step("embedding")
    session.step("layer")
    partial = session.checkpoint("lab-residual-site")
    tag = partial.rsplit("/", 1)[-1][:12]
    session.fork(f"lab-baseline-{tag}", cut=partial)
    baseline = session.generate(tokens)
    baseline_logits = session.resolve_value("logits").float().cpu()
    baseline_cut = session.checkpoint("lab-baseline-end")

    session.fork(f"lab-candidate-{tag}", cut=partial)
    preview = session.preview_edit("residual", scale, mode="scale")
    session.abort_edit(preview["act_id"])
    aborted_cut = session.checkpoint("lab-aborted-preview")
    abort_comparison = session.compare(partial, aborted_cut)
    if not _exact(abort_comparison):
        raise RuntimeError("aborted residual preview changed executable state")
    preview = session.preview_edit("residual", scale, mode="scale")
    committed = session.commit_edit(preview["act_id"])
    candidate = session.generate(tokens)
    candidate_logits = session.resolve_value("logits").float().cpu()
    candidate_cut = session.checkpoint("lab-candidate-end")
    delta = candidate_logits - baseline_logits
    session.checkout(original_branch)
    session.restore(original)
    return {"common_parent": partial, "scale": scale,
            "baseline_cut": baseline_cut, "candidate_cut": candidate_cut,
            "aborted_cut": aborted_cut, "abort_comparison": abort_comparison,
            "commit": committed, "baseline_rows": baseline, "candidate_rows": candidate,
            "equal_token_positions": sum(a["token_id"] == b["token_id"]
                                         for a, b in zip(baseline, candidate)),
            "final_logit_delta": {"l2": float(torch.linalg.vector_norm(delta)),
                                  "signed_mean": float(delta.mean()),
                                  "max_abs": float(delta.abs().max())},
            "interpretation": "Observed response; zero effect and unchanged tokens remain evidence."}


def three_siblings(session, tokens=2):
    """Observe shared backing and private continuations, without forcing a budget."""
    original_branch = session.inspect()["branch"]
    parent = session.checkpoint("lab-memory-parent")
    tag = parent.rsplit("/", 1)[-1][:12]
    before = session.inspect("memory")
    branches = []
    for index in range(3):
        name = f"lab-memory-{tag}-{index}"
        session.fork(name, cut=parent)
        rows = session.generate(tokens)
        branches.append({"branch": name, "rows": rows,
                         "cut": session.checkpoint("lab-private-end")})
    after = session.inspect("memory")
    session.checkout(original_branch)
    session.restore(parent)
    parent_after = session.checkpoint("lab-parent-unchanged")
    comparison = session.compare(parent, parent_after)
    if not _exact(comparison):
        raise RuntimeError("sibling continuation changed its parent")
    return {"parent_cut": parent, "branches": branches, "parent_comparison": comparison,
            "memory_before": before, "memory_after": after,
            "scope": "Sharing and private state; eviction depends on measured budget pressure."}


def circuit_panel(session, spec):
    """Measure declared residual contrasts and exactly replay one candidate."""
    measured = session.measure_circuit(spec)
    candidate = next(row for row in measured["report"]["rows"] if row["role"] == "candidate")
    replay = session.replay_circuit(candidate["evidence_ref"])
    if replay["exact"] is not True:
        raise RuntimeError("retained circuit intervention did not replay exactly")
    return {"measurement": measured, "candidate_replay": replay}


def diffusion_replay(session):
    """Called after prepare(); restore actual scheduler/RNG/block closure."""
    from mdb.bench import tensor_digest

    session.step("block")
    partial = session.checkpoint("lab-first-diffusion-block")
    session.generate()
    first_image = session.resolve_value("image")
    left = session.checkpoint("lab-image-end")
    session.replay(partial)
    second_image = session.resolve_value("image")
    right = session.checkpoint("lab-image-replayed-end")
    comparison = session.compare(left, right)
    first_digest, second_digest = tensor_digest(first_image), tensor_digest(second_image)
    if first_digest != second_digest or not _exact(comparison):
        raise RuntimeError("diffusion block-cut replay differs")
    return {"partial_cut": partial, "left_cut": left, "right_cut": right,
            "image_sha256": first_digest, "replayed_image_sha256": second_digest,
            "comparison": comparison, "exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", required=True, choices=("qwen2", "mamba"))
    parser.add_argument("--out", type=Path, required=True, help="new local output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import MambaConfig, MambaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    from mdb import BenchSession
    from mdb.bench_verify import verify

    torch.set_num_threads(1)
    torch.manual_seed(113)
    if args.tiny == "mamba":
        model = MambaForCausalLM(MambaConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=2, state_size=4, expand=2, conv_kernel=4)).eval()
    else:
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=128,
            attn_implementation="eager")).eval().to(torch.bfloat16)
    with BenchSession.from_resident(model, root=args.out / "session",
            session_id=f"tiny-lab-{args.tiny}", continuation_capacity=8) as session:
        session.prefill([1, 2, 3])
        report = {"scope": "Random CPU models; runtime mechanics only, no semantic experiment.",
                  "family": args.tiny, "identity": session.identity,
                  "history": rewind_and_replay(session),
                  "intervention": residual_intervention(session),
                  "siblings": three_siblings(session)}
        report["circuit"] = circuit_panel(session, {
            "name": "tiny-response", "sites": [0, 1],
            "contrasts": [{"name": "weaken", "mode": "scale", "value": 0.9}],
            "observable": {"kind": "token_margin", "token_id": 7, "against_token_id": 9},
            "query_token_ids": [4], "continuation_tokens": 1,
            "memory_mediation": args.tiny == "mamba", "seed": 0,
            "max_native_operations": 256, "max_capture_bytes": 64 << 20})
        report["status"] = session.inspect()
        report["performance"] = session.inspect("performance")
        session.export(args.out / "export")
    report["verification"] = verify(args.out / "export")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"family": args.tiny, "history_exact": report["history"]["exact"],
                      "circuit_replay_exact": report["circuit"]["candidate_replay"]["exact"],
                      "mechanics_status": report["verification"]["mechanics_status"],
                      "report": str(args.out / "report.json")}))


if __name__ == "__main__":
    main()
