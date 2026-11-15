#!/usr/bin/env python3
"""E6 worker: circuit-tracer attribution graphs + feature interventions on Gemma-2-2B.

Runs INSIDE the dedicated circuit-tracer venv on Beast (transformer-lens backend).
Reads its request from saturn_run_config.json; emits a gzip+b64 report on stdout under
the marker SATURN_E6_GZIP_B64=.  All heavy compute (model load, attribution, forward
interventions) runs here, under mrun.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import random
import time
import traceback
from pathlib import Path

MARKER = "SATURN_E6_GZIP_B64="

# ---- prompts: IDENTICAL to the canary measurement paper (run_sae_comparison.py) ----
IDENTITY_CONTEXTS = [("sitting", "a forest"), ("standing", "a snowy field"), ("resting", "a grassy meadow")]
IDENTITY_TEMPLATE = "A photorealistic {subject} {pose} in {scene}. The animal in this picture is a"
COLOR_CONTEXTS = [("car", "a table"), ("ball", "a shelf"), ("cup", "a desk")]
COLOR_TEMPLATE = "A {color} {object} on a {surface}. The color of the {object} is"


def render_prompt(behavior: str, context_idx: int, subject: str) -> str:
    if behavior == "identity":
        pose, scene = IDENTITY_CONTEXTS[context_idx]
        return IDENTITY_TEMPLATE.format(subject=subject, pose=pose, scene=scene)
    obj, surface = COLOR_CONTEXTS[context_idx]
    return COLOR_TEMPLATE.format(color=subject, object=obj, surface=surface)


def log(*a):
    print("[e6]", *a, flush=True)


# Saturn panel fillers (run_sae_comparison.py IDENTITY_FILLER / COLOR_FILLER): the filler prompt
# differs from the clean prompt only at the subject token, so positions align.
FILLERS = {"identity": "creature", "color": "plain"}


def run_interchange_item(model, tok, it, dev, ks, attr_batch_size, max_feature_nodes, offload):
    """Matched-operator E6: feature-level INTERCHANGE. Every transcoder feature that is active
    in the clean or the filler run at the chosen (layer, position) cells is set to its value on
    the filler prompt (Saturn's state-swap operator, expressed in the instrument's own feature
    basis). Error terms stay at their clean values, as circuit-tracer defines interventions."""
    import torch
    import torch.nn.functional as F
    from circuit_tracer import attribute
    from circuit_tracer.graph import compute_node_influence

    beh, cidx, subj = it["behavior"], int(it["context_idx"]), it["subject"]
    rec = {"behavior": beh, "context_idx": cidx, "subject": subj, "mode": "interchange"}
    prompt = render_prompt(beh, cidx, subj)
    fprompt = render_prompt(beh, cidx, FILLERS[beh])
    rec["prompt"], rec["filler_prompt"] = prompt, fprompt
    ct = tok(prompt)["input_ids"]; ft = tok(fprompt)["input_ids"]
    diff = [i for i in range(min(len(ct), len(ft))) if ct[i] != ft[i]]
    if len(ct) != len(ft) or len(diff) != 1:
        rec["error"] = f"misaligned filler (len {len(ct)} vs {len(ft)}, diff {diff})"
        return rec
    sp = diff[0]; seq = len(ct)
    rec["subj_pos"], rec["seq_len"] = sp, seq
    ans_id = tok.encode(" " + subj, add_special_tokens=False)[0]
    cand_words = it["candidates"]
    cids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in cand_words]

    clean_logits, clean_acts = model.get_activations(prompt)
    fill_logits, fill_acts = model.get_activations(fprompt)
    clean_acts = clean_acts.to(dev); fill_acts = fill_acts.to(dev)
    rec["acts_shape"] = list(clean_acts.shape)
    if int(clean_acts.shape[1]) != seq or int(fill_acts.shape[1]) != seq:
        rec["error"] = f"activation positions {clean_acts.shape[1]}/{fill_acts.shape[1]} != tokenizer seq {seq}"
        return rec
    n_layers = int(clean_acts.shape[0])
    lp_c = F.log_softmax(last_logits(clean_logits).float(), -1)
    lp_f = F.log_softmax(last_logits(fill_logits).float(), -1)
    rec["clean_lp_ans"] = float(lp_c[ans_id]); rec["filler_lp_ans"] = float(lp_f[ans_id])
    rec["clean_top1_cand"] = cand_words[int(lp_c[cids].argmax())]
    denom = rec["filler_lp_ans"] - rec["clean_lp_ans"]

    def swap(layers, positions, src):
        out = []
        for l in layers:
            for p in positions:
                c = clean_acts[l, p]; f = src[l, p]
                idx = torch.nonzero((c != 0) | (f != 0)).flatten()
                vals = f[idx].float().tolist()
                out += [(int(l), int(p), int(i), float(v)) for i, v in zip(idx.tolist(), vals)]
        return out

    def score(interv, fa):
        lg, _ = model.feature_intervention(prompt, interv, freeze_attention=fa)
        lp = F.log_softmax(last_logits(lg).float(), -1)
        d = float(lp[ans_id]) - rec["clean_lp_ans"]
        return {"n": len(interv), "dlp": d, "frac_to_filler": (d / denom) if denom else None,
                "flip_cand": cand_words[int(lp[cids].argmax())] != subj,
                "cand_top": cand_words[int(lp[cids].argmax())]}

    L = list(range(n_layers)); half = n_layers // 2
    arm_specs = {"noop_clean_values_all_layers": (L, [sp], clean_acts)}
    for l in L:
        arm_specs[f"subj_L{l}"] = ([l], [sp], fill_acts)
    arm_specs["subj_first_half"] = (L[:half], [sp], fill_acts)
    arm_specs["subj_second_half"] = (L[half:], [sp], fill_acts)
    arm_specs["subj_all_layers"] = (L, [sp], fill_acts)
    arm_specs["subj_and_later_all_layers"] = (L, list(range(sp, seq)), fill_acts)
    arm_specs["all_positions_all_layers"] = (L, list(range(seq)), fill_acts)
    built = {k: swap(*v) for k, v in arm_specs.items()}

    # graph-guided: the attribution graph's own top-k features (any position) set to filler values
    g = attribute(prompt, model, attribution_targets=torch.tensor([ans_id], device=dev),
                  max_feature_nodes=max_feature_nodes, offload=offload, verbose=False,
                  batch_size=attr_batch_size)
    g.to(dev)
    if dev == "cuda":
        try:
            model.to(dev)
        except Exception:  # noqa
            pass
    n_feat = int(len(g.selected_features)); n_logit = int(len(g.logit_targets))
    A = g.adjacency_matrix
    lw = torch.zeros(A.shape[0], device=A.device); lw[-n_logit:] = g.logit_probabilities
    infl = compute_node_influence(A, lw)[:n_feat].float().cpu().tolist()
    af = g.active_features[g.selected_features].cpu().tolist()
    order = sorted(range(n_feat), key=lambda i: infl[i], reverse=True)
    for k in ks:
        top = [af[i] for i in order[:k]]
        built[f"graph_top{k}_to_filler"] = [(int(l), int(p), int(f), float(fill_acts[l, p, f]))
                                            for (l, p, f) in top]

    rec["arms"] = {}
    for fa in (True, False):
        tag = "frozen" if fa else "unfrozen"
        rec["arms"][tag] = {k: score(v, fa) for k, v in built.items()}
    return rec


def last_logits(lg):
    """Return the final-position logits [vocab] from whatever shape the call returned."""
    while lg.ndim > 1:
        lg = lg[-1] if lg.shape[0] != 1 else lg[0]
    return lg


def main() -> int:
    t0 = time.time()
    cfg = json.loads(Path(os.environ.get("SATURN_RUN_CONFIG", "saturn_run_config.json")).read_text())
    payload = cfg.get("payload", cfg)
    smoke = bool(payload.get("smoke", False))
    items = payload["items"]  # [{behavior, context_idx, subject}]
    ks = payload.get("ks", [5, 20, 50])
    offload = payload.get("offload", "cpu")
    max_feature_nodes = payload.get("max_feature_nodes", 4096)
    n_rand = payload.get("n_rand_seeds", 3)
    model_name = payload.get("model_name", "google/gemma-2-2b")
    transcoder_set = payload.get("transcoder_set", "gemma")
    model_local_path = payload.get("model_local_path")  # local snapshot dir for offline tokenizer
    backend = payload.get("backend", "nnsight")  # TL 4.0 removed HookedTransformer -> use nnsight
    # attribute() allocates a [batch, seq, vocab] bf16 logit buffer; Gemma's 256k vocab makes the
    # default batch_size=512 a 4.46 GB allocation at seq 17, which OOMs a 16 GB card.
    attr_batch_size = int(payload.get("attr_batch_size", 64))
    freeze_attention = bool(payload.get("freeze_attention", True))  # circuit-tracer default

    import torch
    import torch.nn.functional as F
    import circuit_tracer
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.graph import compute_node_influence, compute_graph_scores

    # transformers 4.57.3's AutoTokenizer.from_pretrained(hub_name) calls is_base_mistral ->
    # model_info (network), which fails under HF_HUB_OFFLINE. Neutralize that patch so no
    # offline tokenizer load tries the network.
    try:
        from transformers import tokenization_utils_base as _tub
        if hasattr(_tub.PreTrainedTokenizerBase, "_patch_mistral_regex"):
            _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(
                lambda cls, tokenizer, *a, **k: tokenizer)
    except Exception as _e:  # noqa
        log("mistral-regex patch skip:", repr(_e))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("torch", torch.__version__, "cuda", torch.cuda.is_available(), "dev", dev,
        "ct", getattr(circuit_tracer, "__version__", "?"))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # tokenizer + layer count from the LOCAL snapshot path (offline, no model_info); used for
    # all our own tokenization regardless of backend.
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_local_path) if model_local_path \
        else AutoTokenizer.from_pretrained(model_name)
    n_layers = int(AutoConfig.from_pretrained(model_local_path or model_name).num_hidden_layers)

    log(f"loading ReplacementModel({model_name}, {transcoder_set}) bf16 backend={backend} ...")
    tload = time.time()
    kw = {"dtype": torch.bfloat16, "backend": backend, "device": dev}
    if backend == "transformerlens":
        kw["tokenizer"] = tok  # TL honors a preloaded tokenizer; nnsight loads its own
    model = ReplacementModel.from_pretrained(model_name, transcoder_set, **kw)
    try:
        model.eval()
    except Exception:  # noqa
        pass
    log(f"model loaded in {time.time()-tload:.1f}s; n_layers={n_layers}")

    # candidate token ids for flip detection (full subject/candidate list per behavior)
    def leading_space_id(word: str) -> int:
        ids = tok.encode(" " + word, add_special_tokens=False)
        return ids[0]

    report = {"schema": "saturn-e6-circuit-tracer-v1", "smoke": smoke, "model_name": model_name,
              "attr_batch_size": attr_batch_size, "freeze_attention": freeze_attention,
              "transcoder_set": transcoder_set, "n_layers": int(n_layers),
              "offload": offload, "max_feature_nodes": max_feature_nodes, "ks": ks,
              "torch": torch.__version__, "ct_version": getattr(circuit_tracer, "__version__", "?"),
              "items": []}

    if payload.get("e6_mode") == "interchange":
        report["e6_mode"] = "interchange"
        for it in items:
            try:
                rec = run_interchange_item(model, tok, it, dev, ks, attr_batch_size,
                                           max_feature_nodes, offload)
            except Exception as e:  # noqa
                rec = {"subject": it.get("subject"), "behavior": it.get("behavior"),
                       "error": repr(e), "traceback": traceback.format_exc()}
            report["items"].append(rec)
            if dev == "cuda":
                torch.cuda.empty_cache()
            log(f"interchange item {it['behavior']}/{it['context_idx']}/{it['subject']} done "
                f"err={rec.get('error')}")
        items = []
    for it in items:
        beh, cidx, subj = it["behavior"], int(it["context_idx"]), it["subject"]
        rec = {"behavior": beh, "context_idx": cidx, "subject": subj}
        try:
            prompt = render_prompt(beh, cidx, subj)
            rec["prompt"] = prompt
            # tokenize with the HF tokenizer (BOS added by default for gemma); backend-agnostic
            tokens1d = tok(prompt, return_tensors="pt")["input_ids"][0].to(dev)  # [seq]
            seq = int(tokens1d.shape[0])
            toklist = tokens1d.tolist()
            # nnsight tokenizes internally from a str (same tokenizer+BOS -> positions align with
            # our tok(prompt)); the TL backend takes the token tensor directly.
            model_input = prompt if backend == "nnsight" else tokens1d
            # candidates (behavior-level set) for flip
            cand_words = it["candidates"]
            cand_ids = {leading_space_id(w): w for w in cand_words}
            ans_id = leading_space_id(subj)
            # subject token position: first index whose id == subject's leading-space token
            subj_tok = leading_space_id(subj)
            subj_pos = next((i for i, t in enumerate(toklist) if t == subj_tok), None)
            rec["seq_len"] = int(seq)
            rec["subj_pos"] = subj_pos
            rec["ans_token"] = tok.decode([ans_id])
            if subj_pos is None:
                rec["error"] = "subject token not found"
                report["items"].append(rec)
                continue

            # --- clean logits via the replacement model's no-op ([] interventions).
            # With error terms preserved this reproduces the base model's logits, so it is the
            # clean baseline every arm is scored against (the model's own output machinery). ---
            noop_logits, _ = model.feature_intervention(model_input, [])
            noop_logits = last_logits(noop_logits).float()
            rec["noop_max_abs_diff"] = None  # nnsight no-op == base via preserved error terms
            clean_lp = F.log_softmax(noop_logits, dim=-1)
            rec["clean_lp_ans"] = float(clean_lp[ans_id].item())
            rec["clean_top1_id"] = int(noop_logits.argmax().item())
            rec["clean_top1_tok"] = tok.decode([rec["clean_top1_id"]])
            rec["clean_top1_is_ans"] = (rec["clean_top1_id"] == ans_id)
            # candidate-restricted clean top-1
            cids = list(cand_ids.keys())
            cand_clean = clean_lp[torch.tensor(cids, device=clean_lp.device)]
            rec["clean_top1_cand"] = cand_words[int(cand_clean.argmax().item())]

            # --- attribution graph targeting the answer logit only ---
            tatt = time.time()
            g = attribute(model_input, model, attribution_targets=torch.tensor([ans_id], device=dev),
                          max_feature_nodes=max_feature_nodes, offload=offload, verbose=False,
                          batch_size=attr_batch_size)
            rec["attr_s"] = round(time.time() - tatt, 2)
            g.to(dev)  # Graph.to() is in-place and returns None
            # authoritative subject position in the GRAPH's own token frame (nnsight may tokenize
            # the string slightly differently than our tok(); align node filtering + interventions).
            gtoks = g.input_tokens.tolist()
            subj_pos_g = next((i for i, t in enumerate(gtoks) if t == subj_tok), None)
            rec["subj_pos_graph"] = subj_pos_g
            rec["graph_seq_len"] = len(gtoks)
            if subj_pos_g is None:
                rec["error"] = "subject token not found in graph input_tokens"
                report["items"].append(rec)
                continue
            subj_pos = subj_pos_g  # use the graph frame downstream
            # graph scores
            try:
                rscore, cscore = compute_graph_scores(g)
                rec["replacement_score"] = float(rscore)
                rec["completeness_score"] = float(cscore)
                rec["error_share_end2end"] = float(1.0 - rscore)
            except Exception as e:  # noqa
                rec["score_error"] = repr(e)

            # node influence to the (single) answer logit node
            n_feat = int(len(g.selected_features))
            n_tok = int(g.n_pos)
            nL = int(getattr(g.cfg, "n_layers", n_layers))  # layers in the graph's unified cfg
            n_logit = int(len(g.logit_targets))
            A = g.adjacency_matrix
            lw = torch.zeros(A.shape[0], device=A.device)
            lw[-n_logit:] = g.logit_probabilities
            node_infl = compute_node_influence(A, lw)  # [n_nodes]
            feat_infl = node_infl[:n_feat].float()
            # map selected feature node -> (layer, pos, feature_idx)
            af = g.active_features[g.selected_features].to("cpu")  # [n_feat, 3]
            f_layer = af[:, 0].tolist(); f_pos = af[:, 1].tolist(); f_feat = af[:, 2].tolist()
            fi = feat_infl.to("cpu").tolist()
            tot_feat = float(sum(fi)) or 1e-12
            rec["n_feature_nodes"] = n_feat
            rec["n_error_nodes"] = int(n_tok * nL)
            # subject-position attribution fraction (feature-only)
            subj_feat = sum(v for v, p in zip(fi, f_pos) if p == subj_pos)
            rec["subj_pos_frac_feat"] = float(subj_feat / tot_feat)
            # layer distribution (feature influence by layer), overall + subject-only
            by_layer = {}
            by_layer_subj = {}
            for v, l, p in zip(fi, f_layer, f_pos):
                by_layer[l] = by_layer.get(l, 0.0) + v
                if p == subj_pos:
                    by_layer_subj[l] = by_layer_subj.get(l, 0.0) + v
            rec["layer_influence"] = {int(k): float(v) for k, v in sorted(by_layer.items())}
            rec["layer_influence_subjpos"] = {int(k): float(v) for k, v in sorted(by_layer_subj.items())}
            rec["max_layer_share_overall"] = float(max(by_layer.values()) / tot_feat) if by_layer else 0.0
            tot_subj = float(sum(by_layer_subj.values())) or 1e-12
            rec["max_layer_share_subjpos"] = float(max(by_layer_subj.values()) / tot_subj) if by_layer_subj else 0.0
            # commit-band share (L16-L22, the K/V write band) of subject-position feature influence
            rec["commitband_L16_22_share_subjpos"] = float(
                sum(v for l, v in by_layer_subj.items() if 16 <= l <= 22) / tot_subj) if by_layer_subj else 0.0
            # error-node influence share by layer (diagnostic)
            err_infl = node_infl[n_feat:n_feat + n_tok * nL].float().to("cpu")
            rec["error_influence_total"] = float(err_infl.sum().item())
            rec["feature_influence_total"] = tot_feat

            # --- rank subject-position feature nodes by influence ---
            subj_nodes = [(fi[i], f_layer[i], f_pos[i], f_feat[i]) for i in range(n_feat) if f_pos[i] == subj_pos]
            subj_nodes.sort(key=lambda x: x[0], reverse=True)
            rec["n_subj_feature_nodes"] = len(subj_nodes)
            # best single layer at subject pos
            Lstar = max(by_layer_subj.items(), key=lambda kv: kv[1])[0] if by_layer_subj else None
            rec["Lstar"] = int(Lstar) if Lstar is not None else None
            subj_nodes_Lstar = [n for n in subj_nodes if n[1] == Lstar]

            # candidate pools for random controls (ACTIVE features, not just selected)
            all_af = g.active_features.to("cpu")
            active_subj = [(int(all_af[i, 0]), subj_pos, int(all_af[i, 2]))
                           for i in range(all_af.shape[0]) if int(all_af[i, 1]) == subj_pos]
            active_subj_Lstar = [t for t in active_subj if t[0] == Lstar]

            def score_arm(interventions):
                if not interventions:
                    return None
                lg, _ = model.feature_intervention(model_input, interventions,
                                                   freeze_attention=freeze_attention)
                lg = last_logits(lg).float()
                lp = F.log_softmax(lg, dim=-1)
                dlp = float(lp[ans_id].item() - rec["clean_lp_ans"])
                vocab_top = int(lg.argmax().item())
                cand_lp = lp[torch.tensor(cids, device=lp.device)]
                cand_top_word = cand_words[int(cand_lp.argmax().item())]
                return {"n": len(interventions), "dlp": dlp,
                        "flip_vocab": (vocab_top != ans_id),
                        "vocab_top_tok": tok.decode([vocab_top]),
                        "flip_cand": (cand_top_word != subj),
                        "cand_top": cand_top_word}

            # attribution with offload="cpu" may leave params on CPU; ensure GPU for interventions
            if dev == "cuda":
                try:
                    model.to(dev)
                except Exception:  # noqa
                    pass
            arms = {}
            for k in ks:
                a = [(l, p, fe, 0.0) for (_, l, p, fe) in subj_nodes_Lstar[:k]]
                b = [(l, p, fe, 0.0) for (_, l, p, fe) in subj_nodes[:k]]
                arms[f"A_single_layer_k{k}"] = {"Lstar": int(Lstar) if Lstar is not None else None,
                                                "n_used": len(a), **(score_arm(a) or {})}
                arms[f"B_all_layers_k{k}"] = score_arm(b) or {"n": 0}
                # random controls (mean over seeds)
                for name, pool in (("randA", active_subj_Lstar), ("randB", active_subj)):
                    accs = []
                    for s in range(n_rand):
                        rng = random.Random(1000 * s + k)
                        samp = pool[:] if len(pool) <= k else rng.sample(pool, k)
                        r = score_arm([(l, p, fe, 0.0) for (l, p, fe) in samp])
                        if r:
                            accs.append(r)
                    if accs:
                        arms[f"{name}_k{k}"] = {
                            "n": accs[0]["n"],
                            "dlp": sum(x["dlp"] for x in accs) / len(accs),
                            "flip_cand_rate": sum(1 for x in accs if x["flip_cand"]) / len(accs),
                            "flip_vocab_rate": sum(1 for x in accs if x["flip_vocab"]) / len(accs),
                        }
            # D: the graph's own top-k features at ANY position (the sparse circuit the graph
            # presents), vs size-matched random active features at any position.
            all_nodes = sorted(((fi[i], f_layer[i], f_pos[i], f_feat[i]) for i in range(n_feat)),
                               key=lambda x: x[0], reverse=True)
            active_any = [(int(all_af[i, 0]), int(all_af[i, 1]), int(all_af[i, 2]))
                          for i in range(all_af.shape[0])]
            for k in ks:
                d = [(l, p, fe, 0.0) for (_, l, p, fe) in all_nodes[:k]]
                arms[f"D_graph_topk_any_k{k}"] = {
                    "positions": sorted({p for (_, p, _, _) in d}),
                    "layers": sorted({l for (l, _, _, _) in d}), **(score_arm(d) or {})}
                accs = []
                for s_ in range(n_rand):
                    rng = random.Random(7000 * s_ + k)
                    samp = rng.sample(active_any, min(k, len(active_any)))
                    r = score_arm([(l, p, fe, 0.0) for (l, p, fe) in samp])
                    if r:
                        accs.append(r)
                if accs:
                    arms[f"randD_k{k}"] = {
                        "n": accs[0]["n"], "dlp": sum(x["dlp"] for x in accs) / len(accs),
                        "flip_cand_rate": sum(1 for x in accs if x["flip_cand"]) / len(accs),
                        "flip_vocab_rate": sum(1 for x in accs if x["flip_vocab"]) / len(accs)}
            # C: all subject-pos selected features (whole-path reference)
            c = [(l, p, fe, 0.0) for (_, l, p, fe) in subj_nodes]
            arms["C_all_subjpos_feats"] = {"n_used": len(c), **(score_arm(c) or {})}
            rec["arms"] = arms
        except Exception as e:  # noqa
            rec["error"] = repr(e)
            rec["traceback"] = traceback.format_exc()
        report["items"].append(rec)
        if dev == "cuda":
            torch.cuda.empty_cache()
            rec["peak_vram_mb_after_item"] = float(torch.cuda.max_memory_allocated() / 1e6)
        log(f"item {beh}/{cidx}/{subj} done "
            f"(rscore={rec.get('replacement_score')}, maxlayer_subj={rec.get('max_layer_share_subjpos')})")

    if dev == "cuda":
        report["peak_vram_mb_torch"] = float(torch.cuda.max_memory_allocated() / 1e6)
    report["elapsed_s"] = round(time.time() - t0, 1)
    blob = base64.b64encode(gzip.compress(json.dumps(report).encode())).decode()
    print(MARKER + blob, flush=True)
    log("DONE", report["elapsed_s"], "s; peak_vram_torch_mb", report.get("peak_vram_mb_torch"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
