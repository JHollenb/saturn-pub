"""SAE verbs: ``sae`` (load / list feature bases) and ``features`` (active features at a residual
port with the error row, their direct effects on a consumer with exact closure, or a per-layer
profile of what the SAE drops). Feature edits ride the ordinary edit path: ``set``, ``whatif``
and ``compare`` take a feature expression as the value (``builtin.value_source``):

    set L14.h_out += feature[123]*2          # add 2 d_123
    whatif L20.h_out += feature[77,coef]*-1 consumer=M[a,b]   # remove a_77 d_77 (ablate f77)
    whatif L20.h_out += error*-1 consumer=M[a,b]               # splice the SAE reconstruction
    set L20.h_out = recon                    # replace the port by the reconstruction

Import is torch-free; the analysis layer is imported when an op runs.
"""

from __future__ import annotations

from typing import Any

from mdb.core.records import num

from .builtin import Target, resolve_target
from .errors import BadArgs, Refused
from .registry import Arg, Op, register

_EXACT, _MEASURED, _LOCAL = "derived-exact", "measured", "derived-local"


def _sets(ctx) -> dict[str, Any]:
    """The loaded SAE sets: held by the debugger (they belong to the model), else the context."""
    owner = ctx.debugger if ctx.debugger is not None else ctx
    found = getattr(owner, "_sae_sets", None)
    if found is None:
        found = {}
        owner._sae_sets = found
    return found


def _sae_error():
    from mdb.analysis.sae import SaeError

    return SaeError


def _pick(ctx, name: str | None, layer: int, port: str):
    """``(set, basis)`` covering ``L<layer>.<port>``."""
    sets = _sets(ctx)
    if not sets:
        raise Refused("no SAE loaded: sae load NAME PATH [layers=...]")
    chosen = [sets[name]] if name else list(sets.values())
    if name and name not in sets:
        raise Refused(f"no SAE named {name} (have: {', '.join(sets)})")
    hits = []
    for found in chosen:
        try:
            basis = found.for_site(layer, port)
        except _sae_error() as error:
            raise Refused(str(error)) from None
        if basis is not None:
            hits.append((found, basis))
    if not hits:
        raise Refused(f"no loaded SAE covers L{layer}.{port} (sites: "
                      + "; ".join(f"{s.name}: L<{','.join(map(str, s.layers))}>.{s.port}"
                                  for s in chosen) + ")")
    if len(hits) > 1:
        raise Refused(f"several SAEs cover L{layer}.{port}: name one with sae=NAME")
    return hits[0]


def _text(ctx, token: int) -> str | None:
    tokenizer = ctx.tokenizer
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(token)])
    except Exception:  # noqa: BLE001 - a label only
        return None


def _promotes(ctx, run, basis, feature: int, lens: int) -> dict[str, Any]:
    from mdb.analysis.sae import logit_lens

    ids = logit_lens(run, basis.direction(feature), lens)
    return {"authority": _LOCAL, "value": None, "meaning": "top tokens of W_U (gamma . d_f), "
            "float32 ranking: what the direction promotes, not a label",
            "ids": ids, "texts": [_text(ctx, i) for i in ids]}


# --------------------------------------------------------------------------- feature values


def is_feature_expression(value: Any) -> bool:
    from mdb.analysis.sae import is_expression

    return is_expression(value)


def feature_value(ctx, target: Target, value: str):
    """The float64 tensor (port shape) of a feature expression at the edited site, and its
    source text. ``feature[ID]`` needs nothing from the run; ``coef``, ``error`` and ``recon``
    read the site's current value (it must be computed)."""
    import torch

    from mdb.analysis.sae import edit_vector, parse_terms

    if target.layer_port is None:
        raise Refused(f"feature values apply to residual ports, not {target.text}")
    layer, port = target.layer_port
    try:
        terms = parse_terms(value)
    except _sae_error() as error:
        raise BadArgs(str(error)) from None
    found, basis = _pick(ctx, None, layer, port)
    values: dict[Any, Any] = {}
    for kind, _, coef, _, origin in terms:
        if not (kind in ("error", "recon") or coef) or origin in values:
            continue
        if origin is None:
            source_run, label = ctx.run, ""
        else:
            runs = getattr(ctx.debugger, "runs", {}) or {}
            if origin not in runs:
                raise Refused(f"no run named {origin} (info runs)")
            source_run, label = runs[origin], f"{origin}:"
        try:
            values[origin] = source_run.read(target.run_target)
        except Exception as error:  # noqa: BLE001 - not computed yet
            raise Refused(f"{value} needs the value at {label}{target.run_target}, which is not "
                          f"readable there ({error}); run past it (frame NAME: / until), or use "
                          "whatif from a later stop") from None
    try:
        vector = edit_vector(basis, terms, values)
    except _sae_error() as error:
        raise Refused(str(error)) from None
    shape = tuple(ctx.run.port_shape(port))
    if int(torch.tensor(shape).prod()) != vector.numel():
        raise Refused(f"{target.text} has shape {list(shape)}, the SAE direction "
                      f"{vector.numel()} values")
    return vector.reshape(shape), f"{value} (SAE {found.name} L{basis.layer})"


# --------------------------------------------------------------------------- sae


def _layers(text: str | None) -> list[int] | None:
    if text in (None, "", "all"):
        return None
    out: list[int] = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":", 1)
            out += list(range(int(a), int(b)))
        else:
            out.append(int(part))
    return out


def _sae(ctx, action, name, path, layers, k, port, cache, verify, layer):
    from mdb.analysis.sae import open_set

    sets = _sets(ctx)
    if action == "list":
        return {"saes": [s.describe() for s in sets.values()], "count": len(sets)}
    if action == "drop":
        if name not in sets:
            raise Refused(f"no SAE named {name}")
        sets.pop(name)
        return {"dropped": name, "count": len(sets)}
    if action != "load":
        raise BadArgs("write sae load NAME PATH [layers=A:B|i,j|all] [k=K] [port=h_out|h] "
                      "[cache=N] [verify=true] | sae list | sae drop NAME")
    if not name or not path:
        raise BadArgs("write sae load NAME PATH")
    try:
        found = open_set(name, path, layers=_layers(layers), k=k, port=port, cache=cache,
                         verify=bool(verify), layer=layer)
    except _sae_error() as error:
        raise Refused(str(error)) from None
    except ValueError as error:
        raise BadArgs(str(error)) from None
    d = int(ctx.run.adapter.model.config.hidden_size) if ctx.run is not None else None
    if d is not None:
        basis = found.get(found.layers[0])
        if basis.d_model != d:
            raise Refused(f"SAE {name} has d_model {basis.d_model}, the model {d}")
    sets[name] = found
    return {"loaded": found.describe(), "count": len(sets)}


# --------------------------------------------------------------------------- features


def _feature_rows(ctx, run, basis, split, dec, top, lens, by_effect):
    acts = dict(zip(dec.ids, dec.acts.tolist()))
    norms = dec.D.norm(dim=0).tolist()
    norm_of = dict(zip(dec.ids, norms))
    effects = {i: e for i, _, e in split["features"]} if split else {}
    order = sorted(dec.ids, key=(lambda i: -abs(effects[i])) if by_effect else
                   (lambda i: -acts[i]))
    rows = []
    for rank, feature in enumerate(order[:top]):
        row = {"feature": int(feature), "activation": num(acts[feature], _EXACT),
               "decoder_norm": num(norm_of[feature], _EXACT)}
        if split:
            row["effect"] = num(effects[feature], _EXACT, proof="P19")
        if lens and rank < 8:
            row["promotes"] = _promotes(ctx, run, basis, feature, lens)
        rows.append(row)
    rest = order[top:]
    more = {"count": len(rest), "activation_sum": num(sum(acts[i] for i in rest), _EXACT)}
    if split:
        more["effect_sum"] = num(sum(effects[i] for i in rest), _EXACT, proof="P19")
    return rows, more


def _consumer(ctx, text):
    from mdb.analysis.reads import readout

    from .predict import parse_consumer

    target = parse_consumer(ctx, text)
    try:
        reader = readout(ctx.run, target)
    except Exception as error:  # noqa: BLE001 - Refusal
        raise Refused(str(error)) from None
    if not reader.scalar:
        raise Refused("features consumer= takes a margin M[a,b] or a logit logits[i]")
    if reader.native is None:
        raise Refused(f"{reader.text} is not computed at {ctx.run.where}")
    return reader


def _split_json(split, reader):
    direct = split["direct"]
    parts = [abs(e) for _, _, e in split["features"]] + [abs(split["bias"]), abs(split["error"])]
    total_abs = sum(parts) or 1.0
    later = sum(v for _, _, v in split["downstream"])
    closure = split["native"] - (split["bias"] + split["error"]
                                 + sum(e for _, _, e in split["features"])
                                 + later + split["head_bias"] + split["remainder"])
    return {
        "consumer": reader.text, "native": num(split["native"], _MEASURED, proof="P19"),
        "direct": num(direct, _EXACT, proof="P19"),
        "direct_meaning": "rho c.x: the site's residual on the skip path to the readout",
        "features_sum": num(sum(e for _, _, e in split["features"]), _EXACT, proof="P19"),
        "bias": num(split["bias"], _EXACT, proof="P19"),
        "error": num(split["error"], _EXACT, proof="P19"),
        "error_share_of_direct": (num(split["error"] / direct, _EXACT) if direct else
                                  {"state": "unresolved", "reason": "direct effect is 0"}),
        "error_share_of_abs": num(abs(split["error"]) / total_abs, _EXACT),
        "later_layers": num(later, _EXACT, proof="P19"),
        "head_bias": num(split["head_bias"], _EXACT, proof="P19"),
        "remainder": num(split["remainder"], _MEASURED, proof="P19"),
        "closure": num(closure, _EXACT, proof="P19"),
        "identity_gap": num(split["identity_gap"], _EXACT),
        "closure_meaning": "native - (features + bias + error + later layers + head bias + "
                           "remainder), float64 summation only",
    }


def _profile(ctx, target_text, name, reader, shift=0):
    from mdb.analysis.sae import margin_split

    run = ctx.run
    port = target_text.rsplit(".", 1)[1]
    sets = _sets(ctx)
    chosen = [sets[name]] if name else list(sets.values())
    if not chosen:
        raise Refused("no SAE loaded: sae load NAME PATH")
    if len(chosen) > 1:
        raise Refused("several SAEs are loaded: name one with sae=NAME")
    found = chosen[0]
    rows = []
    for own in found.layers:
        layer = own if port == found.port else (own + 1 if port == "h" else own - 1)
        layer += shift
        if not 0 <= layer < run.adapter.layers:
            continue
        basis = found.get(own)
        split = margin_split(run, reader, basis, layer, port)
        stats = split["decomposition"].stats()
        entry = {"layer": layer, "site": f"L{layer}.{port}", "sae_layer": own,
                 **_split_json(split, reader),
                 "error_fraction_of_norm": stats["error_fraction_of_norm"],
                 "active": stats["active"]}
        for key in ("consumer", "native", "direct_meaning", "closure_meaning"):
            entry.pop(key)
        rows.append(entry)
    return {"mode": "profile", "sae": found.name, "consumer": reader.text, "shift": shift,
            "shift_meaning": "the layer-L SAE applied at L+shift (0 = its own site); a "
                             "misaligned site still closes, only the error fraction shows it",
            "native": num(float(reader.native), _MEASURED, proof="P19"),
            "layers": rows, "count": len(rows), "where": run.where}


def _match(ctx, run, target, basis, text, top, lens, dec):
    from mdb.analysis.sae import match_unit, unit_vector

    try:
        vector, meaning = unit_vector(run, target.row, text)
        found = match_unit(basis, vector, top)
    except _sae_error() as error:
        raise Refused(str(error)) from None
    except ValueError as error:
        raise Refused(str(error)) from None
    acts = dict(zip(dec.ids, dec.acts.tolist())) if dec is not None else {}
    rows = []
    for rank, (feature, cos, coef) in enumerate(zip(found["ids"], found["cos"], found["coef"])):
        row = {"feature": feature, "cos": num(cos, _EXACT), "coef": num(coef, _EXACT),
               "active_here": feature in acts}
        if feature in acts:
            row["activation"] = num(acts[feature], _EXACT)
        if lens and rank < 8:
            row["promotes"] = _promotes(ctx, run, basis, feature, lens)
        rows.append(row)
    return {"unit": text, "meaning": meaning, "norm": num(found["norm"], _EXACT),
            "features": rows,
            "outside_fraction": num(found["outside_fraction"], _EXACT),
            "outside_meaning": "||v - D c|| / ||v||, c the least-squares fit on the listed "
                               "decoder directions: the part of the unit's write no listed "
                               "feature spans",
            "ranking": "float32 cosine over all decoder directions (derived-local), the listed "
                       "cosines recomputed in float64"}


def _features(ctx, address, top, sae, consumer, lens, circuit, match=None, shift=0):
    from mdb.analysis.sae import margin_split

    run = ctx.run
    if run is None:
        raise Refused("no run: start one first")
    reader = _consumer(ctx, consumer) if consumer else None
    text = (address or "").strip()
    if "*" in text:
        if reader is None:
            raise BadArgs("a layer pattern (L*.h_out) needs consumer=M[a,b]")
        if not text.lstrip("t0123456789.").startswith("L*."):
            raise BadArgs("write features L*.h_out|L*.h consumer=M[a,b]")
        return _profile(ctx, text, sae, reader, int(shift or 0))
    found_target = resolve_target(ctx, text)
    if found_target.layer_port is None or found_target.selector:
        raise Refused(f"features reads a whole residual port (L<i>.h_out or L<i>.h), not "
                      f"{found_target.text}")
    layer, port = found_target.layer_port
    sae_set, basis = _pick(ctx, sae, layer, port)
    try:
        x = run.read(found_target.run_target)
    except Exception as error:  # noqa: BLE001 - not computed
        raise Refused(f"{found_target.run_target} has not been computed at {run.where}: "
                      f"{error}") \
            from None
    info = run.read_info(found_target.run_target)
    if reader is not None:
        if reader.row != found_target.row:
            raise Refused(f"{reader.text} is on row t{reader.row}, the port on "
                          f"t{found_target.row}")
        split = margin_split(run, reader, basis, layer, port)
        dec = split["decomposition"]
    else:
        split, dec = None, basis.decompose(x)
    stats = dec.stats()
    rows, more = _feature_rows(ctx, run, basis, split, dec, top, lens, reader is not None)
    error_row = {"norm": stats["error_norm"],
                 "fraction_of_norm": stats["error_fraction_of_norm"],
                 "fraction_of_energy": stats["error_fraction_of_energy"],
                 "meaning": "native minus reconstruction (b_dec + sum a_f d_f)"}
    out: dict[str, Any] = {
        "mode": "port", "target": found_target.run_target, "sae": basis.describe(),
        "value": {"norm": stats["x_norm"], "authority_note": info.get("authority")},
        "features": rows, "more": more, "error": error_row,
        "bias": {"norm": stats["bias_norm"]}, "reconstruction": {"norm": stats["recon_norm"]},
        "active": stats["active"], "identity_closure": stats["identity_closure"],
        "where": run.where}
    if match:
        out["match"] = _match(ctx, run, found_target, basis, match, top, lens, dec)
    if info.get("upstream_edits"):
        out["under_edits"] = [f"t{e['row']}.L{e['layer']}.{e['port']}"
                              for e in info["upstream_edits"]]
    if split is not None:
        accounting = _split_json(split, reader)
        out["accounting"] = accounting
        error_row["effect"] = accounting["error"]
        out["bias"]["effect"] = accounting["bias"]
        if circuit:
            from mdb.analysis.sae import circuit as build

            from .inspect import _session_id, _store
            from .predict import record_view

            try:
                built = build(run, reader, split, basis, top=min(top, 8),
                              session_id=_session_id(ctx))
            except Exception as error:  # noqa: BLE001 - Refusal
                raise Refused(str(error)) from None
            out["circuit"] = {"id": built.id, "record": record_view(built.record)["id"]}
            stored = _store(ctx, [built.record])
            if stored:
                out["circuit"]["stored"] = stored
    return out


register(Op("sae", "Load, list or drop SAE feature bases (per-layer residual SAEs)",
            (Arg("action", "choice", default="list", choices=("load", "list", "drop")),
             Arg("name", "str"), Arg("path", "str"),
             Arg("layers", "str", positional=False, help="A:B, i,j or all (default all)"),
             Arg("k", "int", positional=False, minimum=1, help="top-k when config.json lacks it"),
             Arg("port", "choice", positional=False, choices=("h_out", "h"),
                 help="site when config.json has no hook_point"),
             Arg("cache", "int", default=4, minimum=1, maximum=64, positional=False,
                 help="layers kept loaded (least recently used dropped)"),
             Arg("verify", "bool", default=False, positional=False,
                 help="hash each file on load (sha256)"),
             Arg("layer", "int", positional=False, minimum=0, help="layer of a single file")),
            "mutate", False, _sae,
            usage="sae load NAME PATH [layers=A:B] [k=K] [cache=N] [verify=true] | sae list | "
                  "sae drop NAME"))
register(Op("features", "SAE features at a residual port with the error row; with a consumer, "
            "each feature's, the bias's and the error's exact direct effect and the closure "
            "(L*.h_out: per-layer profile of what the SAE drops)",
            (Arg("address", "str", required=True),
             Arg("top", "int", default=8, minimum=1, maximum=64, positional=False),
             Arg("sae", "str", positional=False),
             Arg("consumer", "str", positional=False, help="M[a,b] or logits[i]"),
             Arg("lens", "int", default=0, minimum=0, maximum=8, positional=False,
                 help="top tokens each listed feature's direction promotes (logit lens)"),
             Arg("circuit", "bool", default=False, positional=False,
                 help="store a circuit record with the SAE expansion (needs consumer)"),
             Arg("match", "str", positional=False,
                 help="name an mdb unit: L<l>.f[i=N], L<l>.A[hH] or L<l>.A[hH,src=S]; lists "
                      "the decoder directions closest to its write"),
             Arg("shift", "int", default=0, minimum=-4, maximum=4, positional=False,
                 help="profile only: apply each layer's SAE this many layers off its site "
                      "(site-alignment check)")),
            "navigate", True, _features,
            usage="features ADDR [top=K] [sae=NAME] [consumer=M[a,b]] [lens=N] [circuit=true] "
                  "[match=UNIT] "
                  "| features L*.h_out consumer=M[a,b]"))


def _feature(ctx, feature, layer, sae, top):
    from mdb.analysis.sae import trace_feature

    run = ctx.run
    if run is None:
        raise Refused("no run: start one first")
    row = int(run.where.split(".", 1)[0][1:])
    stop = run.where.split(".", 1)[1]
    if stop not in ("logits", "sample"):
        raise Refused(f"t{row}.logits has not been computed at {run.where}: feature traces the "
                      f"whole row, run `until t{row}.logits` first")
    sets = _sets(ctx)
    found = sets.get(sae) if sae else (next(iter(sets.values())) if len(sets) == 1 else None)
    if found is None:
        raise Refused("no SAE chosen: load one (sae load) or name it with sae=NAME (sae list)")
    try:
        basis = found.get(int(layer))
        out = trace_feature(run, basis, int(feature), row, top=top)
    except _sae_error() as error:
        raise Refused(str(error)) from None
    probe = [{"layer": p["layer"], **{k: num(p[k], _EXACT) for k in ("h", "r", "h_out")}}
             for p in out["probe"]]
    writers = [{"layer": w["layer"], "attn": num(w["attn"], _EXACT),
                "mlp": num(w["mlp"], _EXACT)} for w in out["writers"]]
    first, last = out["first_above"], out["last_above"]
    return {
        "feature": out["feature"], "site": f"t{row}.{out['site']}", "sae": found.name,
        "active_at_site": out["active"], "cutoff": num(out["cutoff"], _EXACT),
        "cutoff_meaning": "smallest active float32 pre-activation at the site (top-k "
                          "threshold); the probe above it at the site means active",
        "enc_bias": num(out["enc_bias"], "supplied"),
        "probe": probe, "writers": writers,
        "first_above": f"t{row}.L{first[0]}.{first[1]}" if first else None,
        "last_above": f"t{row}.L{last[0]}.{last[1]}" if last else None,
        "heads": {"layer": out["attn_layer"],
                  "items": [{"index": h, "share": num(v, _EXACT, proof="P07")}
                            for h, v in out["heads"]]},
        "neurons": {"layer": out["mlp_layer"],
                    "items": [{"index": i, "share": num(v, _EXACT, proof="P15")}
                              for i, v in out["neurons"]]},
        "meaning": "probe = W_enc[f].x + b_enc[f] at every residual port; writers = W_enc[f]."
                   "o_l and W_enc[f].m_l (exact linear shares of the probe's change); heads "
                   "and neurons of the strongest writers",
        "where": run.where}


register(Op("feature", "Life of one SAE feature along the cursor's row: its encoder probe at "
            "every residual port (h, r, h_out), where it first crosses the site's top-k "
            "threshold, and which attention/MLP (heads, neurons) write it",
            (Arg("feature", "int", required=True, minimum=0),
             Arg("layer", "int", required=True, minimum=0, positional=False,
                 help="the SAE layer the feature belongs to"),
             Arg("sae", "str", positional=False),
             Arg("top", "int", default=4, minimum=1, maximum=16, positional=False)),
            "read", True, _feature,
            usage="feature ID layer=L [sae=NAME] [top=K]"))
