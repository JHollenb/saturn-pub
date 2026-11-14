"""circuit-tracer attribution graphs as mdb edges, and mdb's exact direct effects to check them
(SC8).

A circuit-tracer 0.5.0 ``Graph`` is a dense adjacency over nodes in the order ``[active
features, error nodes (layer-major, then position), token embeddings, logits]`` (rows: targets,
columns: sources); an entry is the direct effect of the source on the target in circuit-tracer's
local replacement model: attention patterns and RMSNorm scales frozen at the native forward,
each MLP replaced by its transcoder's features plus a fixed error, a target feature measured at
its pre-activation, a logit node as the demeaned unembedding row ``W_U[tok] - mean(W_U)`` read
through the frozen final norm.

``graph_edges`` turns a graph into mdb edges: each node named by ``core.foreign_names``
(``ct_node``, ``ct_error_node``, ``ct_token_node``, ``ct_logit_node``: the mdb address the
node reads or writes), each edge a ``circuit_map.Edge`` kind (``read`` when the source and
target rows differ: the path crosses attention; ``compose`` within a row; ``consume`` into a
logit) with its weight (``supplied``: circuit-tracer's number). ``direct_effects`` recomputes the
same matrix in float64 from an mdb run (``derived-exact`` on the run's captured attention
weights ``p``, residuals ``h``/``r`` for the frozen norm scales, and ``n``/``m`` for the error
nodes): the frozen system is linear, so a source's write propagated through the frozen
attention value paths and the residual stream is the edge. ``compare`` reports the agreement.

Running circuit-tracer itself needs its declared pins (transformers <= 4.57.3, huggingface-hub
< 1.0), nnsight with ``nnsight.intervention.tracing.tracer.Barrier`` (0.6.x / 0.7.x; moved in
0.8) and, for its TransformerLens backend, TransformerLens < 4 (``HookedTransformer``): it does
not run in the mdb library environment (transformers 5.13.1, nnsight 0.8.2, TL 4.4). The graph
is a plain tensor file, so mdb reads one made elsewhere without importing circuit-tracer.
"""

from __future__ import annotations

from typing import Any

from mdb.core.foreign_names import ct_error_node, ct_logit_node, ct_node, ct_token_node

KEYS = ("adjacency_matrix", "active_features", "input_tokens")


def read_graph(source: Any) -> dict[str, Any]:
    """A graph as a plain dict: a circuit-tracer ``Graph`` (duck-typed), a dict with its fields,
    or a path to either (``Graph.to_pt`` needs circuit-tracer importable to unpickle; a dict of
    tensors does not). ``n_layers`` comes from the dict or ``Graph.cfg.n_layers``."""
    import torch

    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        try:
            source = torch.load(source, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - a pickled Graph dict holds circuit-tracer objects
            source = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        source = {"adjacency_matrix": source.adjacency_matrix,
                  "active_features": source.active_features,
                  "input_tokens": source.input_tokens, "cfg": source.cfg,
                  "activation_values": source.activation_values,
                  "logit_token_ids": source.logit_token_ids}
    out = dict(source)
    missing = [k for k in KEYS if k not in out]
    if missing:
        raise ValueError(f"not a circuit-tracer graph: missing {missing}")
    if "n_layers" not in out:
        out["n_layers"] = int(out["cfg"].n_layers)
    if "logit_token_ids" not in out:
        targets = out["logit_targets"]
        out["logit_token_ids"] = torch.tensor([int(getattr(t, "vocab_idx", t)) for t in targets])
    return out


def nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """The graph's nodes in adjacency order, each an mdb foreign name with its ``kind``."""
    layers = int(graph["n_layers"])
    tokens = [int(t) for t in graph["input_tokens"].reshape(-1)]
    out = [{"kind": "feature", **ct_node(int(l), int(p), int(f))}
           for l, p, f in graph["active_features"].tolist()]
    out += [{"kind": "error", **ct_error_node(l, p)}
            for l in range(layers) for p in range(len(tokens))]
    out += [{"kind": "token", **ct_token_node(p, t)} for p, t in enumerate(tokens)]
    last = len(tokens) - 1
    out += [{"kind": "logit", **ct_logit_node(last, int(t), layers)}
            for t in graph["logit_token_ids"].reshape(-1)]
    if len(out) != graph["adjacency_matrix"].shape[0]:
        raise ValueError(f"{len(out)} nodes for a {tuple(graph['adjacency_matrix'].shape)} "
                         "adjacency (pruned or max_feature_nodes graphs are not supported)")
    return out


def edge_kind(src: dict[str, Any], dst: dict[str, Any]) -> str:
    if dst["kind"] == "logit":
        return "consume"
    return "read" if src["position"] != dst["position"] else "compose"


def graph_edges(source: Any, *, min_abs: float = 0.0) -> list[dict[str, Any]]:
    """Every nonzero edge (``|weight| > min_abs``) as ``{src, dst, weight, edge}``: the
    endpoints as foreign names, the weight ``supplied`` by circuit-tracer, ``edge`` the
    ``circuit_map.Edge`` over the node ids."""
    from mdb.analysis.circuit_map import Edge
    from mdb.analysis.finite import SUPPLIED, num

    graph = read_graph(source)
    named = nodes(graph)
    matrix = graph["adjacency_matrix"].double()
    out = []
    for dst_i, src_i in (matrix.abs() > min_abs).nonzero().tolist():
        src, dst = named[src_i], named[dst_i]
        out.append({"src": src, "dst": dst,
                    "weight": num(float(matrix[dst_i, src_i]), SUPPLIED),
                    "edge": Edge(src["id"], dst["id"], edge_kind(src, dst))})
    return out


# --------------------------------------------------------------------------- exact check


def _frozen(run: Any, rows: int) -> dict[str, Any]:
    """The frozen operands from the run (cursor at or past the last row's logits): per layer the
    attention weights ``p`` (rows, H, rows) and the norm scales of ``h`` and ``r``; the final
    norm scale; ``n`` and ``m`` for the error nodes. float64."""
    import torch

    from mdb.analysis.finite import f64

    model = run.adapter.model
    eps = float(model.config.rms_norm_eps)
    layers = run.adapter.layers

    def rho(x: Any) -> Any:
        return 1.0 / torch.sqrt(x.pow(2).mean(-1) + eps)

    def port(name: str) -> Any:
        return torch.stack([f64(run.read(f"t{r}.{name}")).reshape(-1) for r in range(rows)])

    out: dict[str, Any] = {"p": [], "rho_in": [], "rho_post": [], "n": [], "m": []}
    for layer in range(layers):
        p = torch.zeros(rows, run.adapter.num_heads, rows, dtype=torch.float64)
        for r in range(rows):
            p[r, :, : r + 1] = f64(run.read(f"t{r}.L{layer}.p")).reshape(run.adapter.num_heads, -1)
        out["p"].append(p)
        out["rho_in"].append(rho(port(f"L{layer}.h")))
        out["rho_post"].append(rho(port(f"L{layer}.r")))
        out["n"].append(port(f"L{layer}.n"))
        out["m"].append(port(f"L{layer}.m"))
    out["rho_final"] = rho(port(f"L{layers - 1}.h_out"))
    return out


def direct_effects(run: Any, graph: Any, transcoders: dict[str, list[Any]], *,
                   zero_rows: int = 1) -> Any:
    """The graph's adjacency recomputed by mdb in float64 (``derived-exact`` on the run's
    captured operands): every source's write (feature: activation x decoder row; error: native
    ``m`` minus the transcoder's reconstruction; token: its embedding row) carried through the
    frozen system (attention patterns and norm scales fixed, MLPs blocked) to every target
    (feature: encoder row . the change of ``n``; logit: demeaned unembedding row . the change
    of the final norm output). ``transcoders``: ``W_enc``, ``W_dec``, ``b_enc``, ``b_dec`` per
    layer (circuit-tracer's ``(d_transcoder, d_model)`` layout, ReLU). The first ``zero_rows``
    rows' error nodes are zero (circuit-tracer's ``zero_positions``, row 0; its graph holds no
    feature there). With ``zero_rows=0`` and ``b_dec`` = 0 every MLP write is a node, so the
    edges into a logit sum to its demeaned native value. Qwen2/Qwen3/Llama layout."""
    import torch

    from mdb.analysis.finite import f64

    graph = read_graph(graph)
    named = nodes(graph)
    tokens = [int(t) for t in graph["input_tokens"].reshape(-1)]
    rows = len(tokens)
    hf = run.adapter.model
    body = hf.model
    layers = run.adapter.layers
    heads, kv, dh = run.adapter.num_heads, run.adapter.num_kv_heads, run.adapter.head_dim
    group = heads // kv
    frozen = _frozen(run, rows)
    enc = [f64(w) for w in transcoders["W_enc"]]
    dec = [f64(w) for w in transcoders["W_dec"]]
    b_enc = [f64(b) for b in transcoders["b_enc"]]
    b_dec = [f64(b) for b in transcoders["b_dec"]]
    lm = f64(hf.lm_head.weight)
    unembed = lm - lm.mean(0, keepdim=True)
    weights = []
    for layer in range(layers):
        block = body.layers[layer]
        weights.append((f64(block.input_layernorm.weight), f64(block.self_attn.v_proj.weight),
                        f64(block.self_attn.o_proj.weight),
                        f64(block.post_attention_layernorm.weight)))
    gamma_final = f64(body.norm.weight)

    def carry(start: int, position: int, vector: Any) -> tuple[list[Any], Any]:
        """The change of ``n`` at every layer after ``start`` and of the final norm output."""
        dx = torch.zeros(rows, vector.shape[0], dtype=torch.float64)
        dx[position] = vector
        dn: list[Any] = [None] * layers
        for layer in range(start + 1, layers):
            g_in, w_v, w_o, g_post = weights[layer]
            a = g_in * frozen["rho_in"][layer][:, None] * dx
            v = (a @ w_v.T).reshape(rows, kv, dh)
            v = v.repeat_interleave(group, dim=1)                       # (rows, H, dh)
            attended = torch.einsum("ths,shd->thd", frozen["p"][layer], v)
            dr = dx + attended.reshape(rows, heads * dh) @ w_o.T
            dn[layer] = g_post * frozen["rho_post"][layer][:, None] * dr
            dx = dr
        return dn, gamma_final * frozen["rho_final"][-1] * dx[-1]

    size = len(named)
    out = torch.zeros(size, size, dtype=torch.float64)
    targets = [(i, n) for i, n in enumerate(named) if n["kind"] in ("feature", "logit")]
    for j, src in enumerate(named):
        if src["kind"] == "feature":
            layer, position = src["layer"], src["position"]
            act = torch.relu(enc[layer][src["feature"]] @ frozen["n"][layer][position]
                             + b_enc[layer][src["feature"]])
            start, vector = layer, act * dec[layer][src["feature"]]
        elif src["kind"] == "error":
            layer, position = src["layer"], src["position"]
            if position < zero_rows:
                continue
            pre = torch.relu(frozen["n"][layer][position] @ enc[layer].T + b_enc[layer])
            start = layer
            vector = frozen["m"][layer][position] - (pre @ dec[layer] + b_dec[layer])
        elif src["kind"] == "token":
            start, position = -1, src["position"]
            vector = f64(body.embed_tokens.weight[src["token"]])
        else:
            continue
        dn, dfinal = carry(start, position, vector)
        for i, dst in targets:
            if dst["kind"] == "feature":
                if dst["layer"] <= start or dn[dst["layer"]] is None:
                    continue
                out[i, j] = enc[dst["layer"]][dst["feature"]] @ dn[dst["layer"]][dst["position"]]
            else:
                out[i, j] = unembed[dst["token"]] @ dfinal
    return out


def compare(supplied: Any, exact: Any, *, floor: float = 0.0) -> dict[str, Any]:
    """Agreement of circuit-tracer's adjacency with mdb's exact one over the entries either
    makes nonzero (``|.| > floor``): count, largest absolute and relative gap, Pearson r, sign
    agreement, and the ratio of their Frobenius norms (all ``measured``)."""
    import torch

    from mdb.analysis.finite import MEASURED, num

    a, b = supplied.double().reshape(-1), exact.double().reshape(-1)
    keep = (a.abs() > floor) | (b.abs() > floor)
    a, b = a[keep], b[keep]
    gap = (a - b).abs()
    scale = b.abs().max().clamp(min=1e-300)
    r = torch.corrcoef(torch.stack([a, b]))[0, 1] if a.numel() > 1 else torch.tensor(1.0)
    return {"edges": int(keep.sum()), "max_abs_gap": num(float(gap.max()), MEASURED),
            "max_gap_over_max_edge": num(float(gap.max() / scale), MEASURED),
            "pearson_r": num(float(r), MEASURED),
            "sign_agree": int(((a > 0) == (b > 0)).sum()),
            "norm_ratio": num(float(a.norm() / b.norm().clamp(min=1e-300)), MEASURED)}


__all__ = ["compare", "direct_effects", "edge_kind", "graph_edges", "nodes", "read_graph"]
