"""Native edge test for Anthropic / Decode Research `circuit-tracer` attribution graphs.

`circuit-tracer` builds an *attribution graph* on a transcoder **replacement** model: it
decomposes each MLP into sparse transcoder features and reports which feature nodes (and
transcoder *error* nodes) influence a target logit. A graph edge is a candidate, not a
verdict. This module takes such a graph, turns each selected feature into a native
intervention on the **real** model weights at the residual-stream point the transcoder
writes to, runs the native continuation, and lets the unchanged native consumer decide
whether the edge *survives* or *collapses* -- using saturn-pub's :mod:`saturn_pub.trial`
seam (``Reading.from_external`` + the frozen :class:`~saturn_pub.trial.DecisionRule`).

Operators (both use the real weights and a native greedy continuation):

* ``carrier`` -- the saturn-pub transaction path. For a feature whose transcoder writes at
  one layer (a per-layer transcoder) and whose node sits at the **last** prompt position,
  step a :class:`~saturn_pub.Session` to that write boundary, subtract the feature's
  decoder contribution with an :class:`~saturn_pub.Act`, and arbitrate it with
  :func:`saturn_pub.trial.arbitrate`. This yields a sealed :class:`~saturn_pub.Receipt` and
  a cut that replays in a fresh process.
* ``residual_hook`` -- a position-general native re-test. For a feature at any position,
  add its decoder contribution to the real model's residual stream at each write layer with
  a forward hook, run the native forward, and grade the target-logit effect under the same
  frozen rule. This covers the subject-position features an attribution graph cares about.

Transcoder **error** nodes carry influence the transcoders do not reconstruct as a feature
direction; they are reported as *uncovered*, never silently dropped.

`circuit_tracer` itself is an **optional** dependency and is imported lazily -- importing
this module needs only torch (for the native path) and the stdlib. The offline tests and
the shipped notebook drive it with a synthetic graph object and re-derive the sealed
verdict table with :func:`verify_edge_bundle` (stdlib only, no torch, no model).
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import Act, Receipt
from ..trial import DecisionRule, Reading, grade
from ..values import canonical, describe, digest

# A decoder-write provider maps (feature_layer, feature_idx) -> [(write_layer, direction), ...].
# A per-layer transcoder returns one pair at its own layer; a cross-layer transcoder returns
# one pair per layer it writes to. ``direction`` is a 1-D residual-space tensor (d_model).
DecoderWrites = Callable[[int, int], Sequence[tuple[int, Any]]]

# Neutral native-necessity labels. The native drop measures only whether removing *this*
# feature's contribution is individually load-bearing on the real model -- it is NOT a verdict
# on the attribution graph. Agreement with the graph is a separate axis (``graph_vs_native``,
# which needs the graph's own predicted drop); see :func:`classify_graph_vs_native`.
_VERDICT_FROM_TRIAL = {
    "agree": "native_necessary",
    "invert": "native_effect_absent",
    "inconclusive": "inconclusive",
}


def default_edge_rule() -> DecisionRule:
    """The frozen rule the native edge test labels every edge under.

    ``consumer_logprob_drop`` is ``logprob(target | clean) - logprob(target | intervention)``
    in nats: higher means the intervention hurt the answer more. The label is about the
    *native* model only: ``native_necessary`` (drop >= 0.5 nats, removing this contribution is
    individually load-bearing), ``native_effect_absent`` (drop <= 0.1 nats, no individual
    effect), or ``inconclusive``. A single feature showing ``native_effect_absent`` is NOT by
    itself evidence against the graph -- under a redundant circuit many features each carry
    little alone. Whether that disagrees with the graph is the separate ``graph_vs_native``
    axis, which compares this drop to the replacement model's own predicted drop.
    """
    return DecisionRule(
        name="circuit-tracer-native-edge",
        version="1",
        metric="consumer_logprob_drop",
        present_threshold=0.5,
        absent_threshold=0.1,
        require_exact_gate=True,
        description=(
            "Native-necessity of one transcoder feature on the real model: native_necessary iff "
            "removing its decoder contribution drops the target log-prob by >= 0.5 nats; "
            "native_effect_absent iff the drop is <= 0.1 nats; inconclusive between. This labels "
            "the native model's response to the intervention, not the attribution graph."
        ),
    )


def classify_graph_vs_native(
    graph_delta: float | None, native_delta: float | None, rule: DecisionRule
) -> str:
    """Compare the graph's predicted target-logprob drop to the native one, same intervention.

    This is the scientific axis the single native label cannot answer: does the replacement
    model (where the graph lives) *predict* a large effect where the real model shows none (a
    genuine ``invert``, i.e. the graph is wrong), or do both show a small effect (``agree_small``
    -- redundancy, not a graph failure), or both large (``agree_large``)?
    """
    if graph_delta is None or native_delta is None:
        return "no_graph_prediction"
    present, absent = rule.present_threshold, rule.absent_threshold
    g_large, g_small = graph_delta >= present, graph_delta <= absent
    n_large, n_small = native_delta >= present, native_delta <= absent
    if g_small and n_small:
        return "agree_small"
    if g_large and n_large:
        return "agree_large"
    if g_large and n_small:
        return "invert"  # graph predicts necessity the native model does not show
    return "mixed"


# --------------------------------------------------------------------------------------
# Graph reading + edge selection (circuit-tracer's own influence algorithm, reimplemented).
# --------------------------------------------------------------------------------------


def _node_influence(adjacency: Any, logit_weights: Any, *, max_iter: int = 2000) -> Any:
    """Total node influence on the logits: ``w @ (A + A^2 + ...)`` over the normalised graph.

    This reproduces circuit-tracer's ``compute_node_influence`` (``normalize_matrix`` then the
    truncated Neumann series) so the edge ranking matches the tool, without importing it.
    """

    a = adjacency.abs()
    a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-10)
    current = logit_weights @ a
    influence = current.clone()
    for _ in range(max_iter):
        if not bool(current.any()):
            break
        current = current @ a
        influence = influence + current
    return influence


@dataclass(frozen=True)
class _GraphView:
    """Normalised view of the fields the edge test reads from an attribution graph."""

    active_features: Any  # [n_active, 3] int: (layer, pos, feature_idx)
    selected_features: Any  # [n_sel] int: indices into active_features
    activation_values: Any  # [n_active] float: clean activation, aligned with active_features
    adjacency_matrix: Any  # [n_nodes, n_nodes]
    logit_probabilities: Any  # [n_logit]
    n_pos: int
    n_layers: int
    input_tokens: list[int]


def _as_view(graph: Any) -> _GraphView:
    import torch

    def _t(x: Any) -> Any:
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    cfg = getattr(graph, "cfg", None)
    n_layers = int(getattr(cfg, "n_layers", 0) or getattr(graph, "n_layers", 0))
    if n_layers <= 0:
        raise ValueError("attribution graph must expose cfg.n_layers (or n_layers)")
    tokens = getattr(graph, "input_tokens")
    tokens = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    active = _t(graph.active_features).long()
    values = _t(graph.activation_values).float()
    # circuit-tracer stores one activation per *active* feature (the sparse activation
    # matrix's values), not one per selected node; index it through selected_features.
    if int(values.shape[0]) != int(active.shape[0]):
        raise ValueError(
            "attribution graph activation_values must align with active_features "
            f"({int(values.shape[0])} values for {int(active.shape[0])} active features)"
        )
    return _GraphView(
        active_features=active,
        selected_features=_t(graph.selected_features).long(),
        activation_values=values,
        adjacency_matrix=_t(graph.adjacency_matrix).float(),
        logit_probabilities=_t(graph.logit_probabilities).float(),
        n_pos=int(getattr(graph, "n_pos", len(tokens))),
        n_layers=n_layers,
        input_tokens=[int(t) for t in tokens],
    )


@dataclass(frozen=True)
class _SelectedEdge:
    rank: int
    layer: int
    position: int
    feature: int
    activation: float
    graph_weight: float
    # The replacement model's own predicted target-logprob drop for this feature's zero
    # ablation, measured with circuit-tracer's feature_intervention (filled by the caller that
    # still holds the replacement model). ``None`` when not supplied.
    graph_predicted_delta: float | None = None


@dataclass(frozen=True)
class GraphInfluence:
    """Influence summary over the graph's feature and transcoder-error nodes."""

    edges: tuple[_SelectedEdge, ...]
    n_feature_nodes: int
    n_error_nodes: int
    feature_influence_total: float
    error_influence_total: float

    @property
    def error_node_influence_share(self) -> float:
        total = self.feature_influence_total + self.error_influence_total
        return float(self.error_influence_total / total) if total else 0.0


def select_edges(graph: Any, *, top_k: int, influence: Any | None = None) -> GraphInfluence:
    """Rank selected feature nodes by influence on the target logit; summarise error nodes.

    ``influence`` may be supplied (e.g. circuit-tracer's own ``compute_node_influence``
    output over the same node ordering); otherwise it is recomputed here identically.
    """
    import torch

    view = _as_view(graph)
    n_feat = int(view.selected_features.shape[0])
    n_err = view.n_layers * view.n_pos
    n_logit = int(view.logit_probabilities.shape[0])
    a = view.adjacency_matrix
    if influence is None:
        logit_weights = torch.zeros(a.shape[0], dtype=a.dtype)
        logit_weights[-n_logit:] = view.logit_probabilities.to(a.dtype)
        influence = _node_influence(a, logit_weights)
    influence = influence.float()
    feat_infl = influence[:n_feat]
    err_infl = influence[n_feat : n_feat + n_err]
    af = view.active_features[view.selected_features]  # [n_feat, 3]
    order = torch.argsort(feat_infl, descending=True).tolist()
    edges = []
    for rank, i in enumerate(order[: int(top_k)]):
        layer, pos, feat = (int(x) for x in af[i].tolist())
        active_index = int(view.selected_features[i])
        edges.append(
            _SelectedEdge(
                rank=rank,
                layer=layer,
                position=pos,
                feature=feat,
                activation=float(view.activation_values[active_index]),
                graph_weight=float(feat_infl[i]),
            )
        )
    return GraphInfluence(
        edges=tuple(edges),
        n_feature_nodes=n_feat,
        n_error_nodes=n_err,
        feature_influence_total=float(feat_infl.clamp(min=0).sum()),
        error_influence_total=float(err_infl.clamp(min=0).sum()),
    )


# --------------------------------------------------------------------------------------
# Verdict row + table.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeRow:
    """One attribution edge tested natively (or reported uncovered)."""

    node_kind: str  # "feature" | "error"
    layer: int | None
    position: int | None
    feature: int | None
    graph_weight: float
    activation: float | None
    write_layers: tuple[int, ...]
    feature_type: str
    operator: str  # "carrier+arbitrate" | "residual_hook" | "uncovered"
    covered: bool
    verdict: str  # "survives" | "collapses" | "inconclusive" | "uncovered"
    native_logit_delta: float | None
    clean_logprob: float | None
    ablated_logprob: float | None
    top_token_clean: int | None
    top_token_ablated: int | None
    top_token_flip: bool | None
    reason: str
    reading_fingerprint: str | None
    rule_fingerprint: str | None
    receipt: str | None
    measurements: Mapping[str, Any] = field(default_factory=dict)
    # Graph side: the replacement model's predicted drop for the same intervention, and how it
    # compares to the native drop (see :func:`classify_graph_vs_native`). ``None`` when the
    # caller did not supply a graph prediction.
    graph_predicted_delta: float | None = None
    graph_vs_native: str | None = None

    def to_dict(self) -> dict[str, Any]:
        body = {
            "node_kind": self.node_kind,
            "layer": self.layer,
            "position": self.position,
            "feature": self.feature,
            "graph_weight": self.graph_weight,
            "activation": self.activation,
            "write_layers": list(self.write_layers),
            "feature_type": self.feature_type,
            "operator": self.operator,
            "covered": self.covered,
            "verdict": self.verdict,
            "native_logit_delta": self.native_logit_delta,
            "graph_predicted_delta": self.graph_predicted_delta,
            "graph_vs_native": self.graph_vs_native,
            "clean_logprob": self.clean_logprob,
            "ablated_logprob": self.ablated_logprob,
            "top_token_clean": self.top_token_clean,
            "top_token_ablated": self.top_token_ablated,
            "top_token_flip": self.top_token_flip,
            "reason": self.reason,
            "reading_fingerprint": self.reading_fingerprint,
            "rule_fingerprint": self.rule_fingerprint,
            "receipt": self.receipt,
            "measurements": dict(self.measurements),
        }
        body["fingerprint"] = digest(body)
        return body


@dataclass(frozen=True)
class EdgeVerdictTable:
    """The native verdict for every selected edge, plus honest uncovered accounting."""

    model_identity: str
    execution: Mapping[str, Any]
    prompt: str
    tokens: tuple[int, ...]
    target_token: int
    target_text: str
    subject_position: int | None
    top_k: int
    decision_rule: Mapping[str, Any]
    rows: tuple[EdgeRow, ...]
    graph_summary: Mapping[str, Any]
    environment: Mapping[str, Any]
    custody: Mapping[str, Any]
    # Group interventions (joint ablation / steering of a supernode or the top-k feature set);
    # each dict carries native_delta, graph_predicted_delta, verdict, graph_vs_native, members.
    group_interventions: tuple[Mapping[str, Any], ...] = ()

    def summary(self) -> dict[str, Any]:
        covered = [r for r in self.rows if r.covered]
        necessary = [r for r in covered if r.verdict == "native_necessary"]
        absent = [r for r in covered if r.verdict == "native_effect_absent"]
        inconclusive = [r for r in covered if r.verdict == "inconclusive"]
        n_feature = sum(1 for r in self.rows if r.node_kind == "feature")
        with_graph = [r for r in covered if r.graph_vs_native not in (None, "no_graph_prediction")]
        agree = [r for r in with_graph if r.graph_vs_native in ("agree_small", "agree_large")]
        invert = [r for r in with_graph if r.graph_vs_native == "invert"]
        mixed = [r for r in with_graph if r.graph_vs_native == "mixed"]
        return {
            "n_edges": len(self.rows),
            "n_feature_edges": n_feature,
            "n_error_nodes_reported": sum(1 for r in self.rows if r.node_kind == "error"),
            "n_covered": len(covered),
            "n_native_necessary": len(necessary),
            "n_native_effect_absent": len(absent),
            "n_inconclusive": len(inconclusive),
            "n_uncovered": sum(1 for r in self.rows if not r.covered),
            "covered_fraction": (len(covered) / n_feature) if n_feature else 0.0,
            "native_necessary_fraction": (len(necessary) / len(covered)) if covered else 0.0,
            "n_with_graph_prediction": len(with_graph),
            "n_agree_with_graph": len(agree),
            "n_invert_vs_graph": len(invert),
            "n_mixed_vs_graph": len(mixed),
            "graph_agreement_fraction": (len(agree) / len(with_graph)) if with_graph else None,
            "error_node_influence_share": self.graph_summary.get("error_node_influence_share"),
        }

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "saturn-pub-circuit-tracer-edge-verdicts-v1",
            "model_identity": self.model_identity,
            "execution": dict(self.execution),
            "prompt": self.prompt,
            "tokens": list(self.tokens),
            "target_token": self.target_token,
            "target_text": self.target_text,
            "subject_position": self.subject_position,
            "top_k": self.top_k,
            "decision_rule": dict(self.decision_rule),
            "graph_summary": dict(self.graph_summary),
            "summary": self.summary(),
            "rows": [r.to_dict() for r in self.rows],
            "group_interventions": [dict(g) for g in self.group_interventions],
            "environment": dict(self.environment),
            "custody": dict(self.custody),
            "arbiter": "unchanged native consumer; an attribution edge is a candidate, not a verdict",
        }
        body["fingerprint"] = digest(body)
        return body

    def seal(self, directory: str | Path) -> dict[str, Any]:
        """Write a hash-pinned bundle: ``edge_verdicts.json`` + ``manifest.json``.

        Returns the bundle fingerprint and manifest. :func:`verify_edge_bundle` re-derives
        every covered verdict from the sealed scalars offline (stdlib only).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        body = canonical(self.to_dict())
        (directory / "edge_verdicts.json").write_bytes(body)
        manifest = {"edge_verdicts.json": hashlib.sha256(body).hexdigest()}
        (directory / "manifest.json").write_bytes(canonical(manifest))
        return {
            "bundle": str(directory),
            "fingerprint": self.to_dict()["fingerprint"],
            "manifest": manifest,
        }


# --------------------------------------------------------------------------------------
# The native edge test.
# --------------------------------------------------------------------------------------


def _target_logprob(logits: Any, target: int) -> tuple[float, int]:
    import torch

    lp = torch.log_softmax(logits.reshape(-1).float(), dim=-1)
    return float(lp[target]), int(logits.reshape(-1).argmax())


def _residual_hook_effect(
    adapter: Any,
    tokens: Sequence[int],
    writes: Sequence[tuple[int, Any]],
    activation: float,
    target: int,
    clean_logits: Any,
) -> dict[str, Any]:
    """Native re-test: add ``-activation * direction`` at each write layer's resid_post, forward."""
    import torch

    handles = []

    def _hook_for(position: int, direction: Any):
        vec = (-float(activation)) * direction.to(adapter.dtype).reshape(-1)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            hidden[:, position, :] = hidden[:, position, :] + vec.to(hidden.device)
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook

    position = len(tokens) - 1
    # a feature node lives at one position; the write layers share it
    try:
        for write_layer, direction in writes:
            handle = adapter._layer_modules[write_layer].register_forward_hook(
                _hook_for(position, direction)
            )
            handles.append(handle)
        with torch.inference_mode():
            ablated = adapter.model(torch.tensor([list(tokens)], device=adapter.device)).logits[
                :, -1
            ]
    finally:
        for handle in handles:
            handle.remove()
    clean_lp, clean_top = _target_logprob(clean_logits, target)
    ablated_lp, ablated_top = _target_logprob(ablated, target)
    return {
        "clean_logprob": clean_lp,
        "ablated_logprob": ablated_lp,
        "consumer_logprob_drop": clean_lp - ablated_lp,
        "top_token_clean": clean_top,
        "top_token_ablated": ablated_top,
        "top_token_flip": clean_top != ablated_top,
    }


def _carrier_effect(
    adapter: Any,
    edge: _SelectedEdge,
    direction: Any,
    target: int,
    rule: DecisionRule,
    reading: Reading,
    tokens: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    """Arbitrate a last-position per-layer feature through the Session carrier."""

    from ..trial import arbitrate

    write_boundary_steps = 1 + (edge.layer + 1)  # embed + (layer+1) decoder layers
    session = adapter.session(list(tokens))
    session.continue_(write_boundary_steps)  # now at layer:(L+1); hidden == resid_post[L], last pos
    parent = session.capture(retain=False)
    vec = direction.to(adapter.dtype).reshape(1, 1, -1)
    act = Act.add(
        "hidden", vec, dose=-float(edge.activation), name="circuit-tracer-feature-ablation"
    )

    def native_driver(branch: Any) -> None:
        adapter.generate(branch, 1)

    def candidate_driver(branch: Any) -> None:
        branch.apply(act)
        adapter.generate(branch, 1)

    def effect(branches: Mapping[str, Any]) -> dict[str, Any]:
        nat_logits = branches["native"].read("logits")[0]
        cand_logits = branches["candidate"].read("logits")[0]
        clean_lp, clean_top = _target_logprob(nat_logits, target)
        ablated_lp, ablated_top = _target_logprob(cand_logits, target)
        return {
            "consumer_logprob_drop": clean_lp - ablated_lp,
            "clean_logprob": clean_lp,
            "ablated_logprob": ablated_lp,
            "top_token_clean": clean_top,
            "top_token_ablated": ablated_top,
            "top_token_flip": clean_top != ablated_top,
        }

    row = arbitrate(
        adapter,
        reading,
        rule,
        effect=effect,
        candidate=candidate_driver,
        native=native_driver,
        parent=parent,
    )
    return row, dict(row.measurements)


def native_group_intervention(
    adapter: Any,
    tokens: Sequence[int],
    members: Sequence[Mapping[str, Any]],
    target_token: int,
    *,
    name: str,
    multiplier: float,
    graph_predicted_delta: float | None = None,
    decision_rule: DecisionRule | None = None,
    clean_logits: Any | None = None,
) -> dict[str, Any]:
    """Jointly intervene on a whole feature set (a supernode / the top-k) on the real model.

    This matches the attribution-graphs paper's group interventions: instead of zeroing one
    feature, we steer the whole set at once. ``members`` is a list of
    ``{layer, position, feature, activation, writes:[(write_layer, direction), ...]}``. The
    ``multiplier`` m sets each feature's new value to ``m * activation`` (``m = 0`` ablates the
    set; ``m = -2`` steers it to minus twice its natural activation, as the paper does); the
    change added to the residual at each member's write layer and position is therefore
    ``(m - 1) * activation * direction``. The group is graded for native necessity under the
    frozen rule and compared to the replacement model's own predicted drop.
    """
    import torch

    rule = decision_rule or default_edge_rule()
    tokens = [int(t) for t in tokens]
    if clean_logits is None:
        with torch.inference_mode():
            clean_logits = adapter.model(torch.tensor([tokens], device=adapter.device)).logits[
                :, -1
            ]

    by_layer: dict[int, list[tuple[int, Any]]] = {}
    member_ids = []
    for m in members:
        act = float(m["activation"])
        pos = int(m["position"])
        member_ids.append(
            {
                "layer": int(m["layer"]),
                "position": pos,
                "feature": int(m["feature"]),
                "activation": act,
            }
        )
        for write_layer, direction in m["writes"]:
            vec = (float(multiplier) - 1.0) * act * direction.to(adapter.dtype).reshape(-1)
            by_layer.setdefault(int(write_layer), []).append((pos, vec))

    def _hook_for(entries: list[tuple[int, Any]]):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            for pos, vec in entries:
                hidden[:, pos, :] = hidden[:, pos, :] + vec.to(hidden.device)
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return hook

    handles = []
    try:
        for write_layer, entries in by_layer.items():
            handles.append(
                adapter._layer_modules[write_layer].register_forward_hook(_hook_for(entries))
            )
        with torch.inference_mode():
            steered = adapter.model(torch.tensor([tokens], device=adapter.device)).logits[:, -1]
    finally:
        for h in handles:
            h.remove()

    clean_lp, clean_top = _target_logprob(clean_logits, target_token)
    steered_lp, steered_top = _target_logprob(steered, target_token)
    drop = clean_lp - steered_lp
    reading = Reading.from_external(
        "circuit-tracer",
        claim=f"group '{name}' ({len(member_ids)} features, multiplier {multiplier}) "
        "is jointly load-bearing for the target logit",
        asserts_effect=True,
        detail={"name": name, "multiplier": multiplier, "n_members": len(member_ids)},
    )
    verdict, _classification, reason = grade(reading, rule, drop)
    return {
        "name": name,
        "multiplier": float(multiplier),
        "operator": "group_residual_hook",
        "n_members": len(member_ids),
        "members": member_ids,
        "verdict": _VERDICT_FROM_TRIAL[verdict],
        "native_logit_delta": drop,
        "graph_predicted_delta": graph_predicted_delta,
        "graph_vs_native": classify_graph_vs_native(graph_predicted_delta, drop, rule),
        "clean_logprob": clean_lp,
        "steered_logprob": steered_lp,
        "top_token_clean": clean_top,
        "top_token_steered": steered_top,
        "top_token_flip": clean_top != steered_top,
        "reason": reason,
        "reading_fingerprint": reading.fingerprint,
        "rule_fingerprint": rule.fingerprint,
    }


def native_edge_test(
    graph: Any,
    adapter: Any,
    *,
    prompt_tokens: Sequence[int],
    target_token: int,
    decoder_writes: DecoderWrites,
    top_k: int = 20,
    decision_rule: DecisionRule | None = None,
    operators: Sequence[str] = ("carrier", "residual_hook"),
    max_carrier_edges: int | None = None,
    influence: Any | None = None,
    selection: GraphInfluence | None = None,
    prompt_text: str = "",
    target_text: str = "",
    subject_position: int | None = None,
    graph_scores: Mapping[str, Any] | None = None,
    group_interventions: Sequence[Mapping[str, Any]] = (),
    store: Any | None = None,
) -> EdgeVerdictTable:
    """Turn each selected attribution edge into a native verdict on the real model.

    ``graph`` is a circuit-tracer ``Graph`` (or an object exposing the same fields; see
    :func:`build_synthetic_graph`). ``adapter`` is a saturn-pub decoder adapter wrapping the
    **real** model the graph was computed against. ``decoder_writes(layer, feature)`` returns
    the transcoder's residual-space write directions for a feature, one ``(write_layer,
    direction)`` pair per layer it writes to (one pair for a per-layer transcoder).

    Pass ``graph`` to select edges here, or a precomputed ``selection`` (a
    :class:`GraphInfluence`, e.g. ranked with circuit-tracer's own influence then kept while the
    heavy replacement model is freed) to run only the native test. The two are mutually
    sufficient; ``graph`` may be ``None`` when ``selection`` is given.

    ``max_carrier_edges`` caps how many (top-ranked, last-position, single-write-layer) edges use
    the heavier ``carrier``+``arbitrate`` path (which forks several native continuations for its
    sealed receipt + exact-replay gate); edges past the cap fall through to the single-forward
    ``residual_hook``, so every edge still gets a native verdict under the same frozen rule.

    Every selected feature becomes a :func:`Reading.from_external` asserting the edge is
    load-bearing, graded under the frozen ``decision_rule`` into a **native-necessity** label
    (about the real model only): ``native_necessary`` (removing this one feature is individually
    load-bearing), ``native_effect_absent`` (no individual effect -- under a redundant circuit
    this is expected and is *not* by itself evidence against the graph), or ``inconclusive``.
    When ``selection`` edges carry ``graph_predicted_delta`` (the replacement model's own
    predicted drop for the same intervention), each row also gets a ``graph_vs_native`` tag
    (:func:`classify_graph_vs_native`): ``invert`` means the graph predicted a large effect the
    native model does not show (a real disagreement), ``agree_small``/``agree_large`` mean both
    agree. ``group_interventions`` (built with :func:`native_group_intervention`) seal joint
    ablation/steering of whole supernodes or the top-k set. Transcoder error nodes are reported
    ``uncovered`` with their influence share. The result is a :class:`EdgeVerdictTable` whose
    :meth:`~EdgeVerdictTable.seal` writes a hash-pinned, offline-re-derivable bundle.
    """
    import torch

    rule = decision_rule or default_edge_rule()
    tokens = [int(t) for t in prompt_tokens]
    last_pos = len(tokens) - 1
    if selection is None:
        if graph is None:
            raise ValueError("native_edge_test needs either a graph or a precomputed selection")
        selection = select_edges(graph, top_k=top_k, influence=influence)

    with torch.inference_mode():
        clean_logits = adapter.model(torch.tensor([tokens], device=adapter.device)).logits[:, -1]

    use_carrier = "carrier" in operators
    use_hook = "residual_hook" in operators
    rows: list[EdgeRow] = []
    carrier_used = 0

    for edge in selection.edges:
        writes = [(int(wl), d) for wl, d in decoder_writes(edge.layer, edge.feature)]
        write_layers = tuple(wl for wl, _ in writes)
        in_range = all(0 <= wl < adapter.layers for wl in write_layers)
        feature_type = (
            "per-layer-transcoder"
            if len(writes) == 1 and write_layers[0] == edge.layer
            else "cross-layer-transcoder"
        )
        reading = Reading.from_external(
            "circuit-tracer",
            claim=(
                f"attribution edge: feature {edge.feature} (layer {edge.layer}, "
                f"position {edge.position}) is load-bearing for the target logit"
            ),
            asserts_effect=True,
            detail={
                "layer": edge.layer,
                "position": edge.position,
                "feature": edge.feature,
                "graph_weight": edge.graph_weight,
                "activation": edge.activation,
                "write_layers": list(write_layers),
                "feature_type": feature_type,
            },
        )

        if not in_range:
            rows.append(
                _uncovered_row(
                    edge,
                    write_layers,
                    feature_type,
                    reading,
                    rule,
                    reason="transcoder write layer is outside the model's layer range",
                )
            )
            continue

        carrier_ok = (
            use_carrier
            and len(writes) == 1
            and writes[0][0] == edge.layer
            and edge.position == last_pos
            and (max_carrier_edges is None or carrier_used < max_carrier_edges)
        )
        if carrier_ok:
            carrier_used += 1
            direction = writes[0][1]
            trial_row, meas = _carrier_effect(
                adapter, edge, direction, target_token, rule, reading, tokens
            )
            # cross-check with the position-matched residual hook (same intervention)
            if use_hook:
                hook = _residual_hook_effect(
                    adapter, tokens, writes, edge.activation, target_token, clean_logits
                )
                meas = {**meas, "residual_hook_crosscheck_drop": hook["consumer_logprob_drop"]}
            rows.append(
                EdgeRow(
                    node_kind="feature",
                    layer=edge.layer,
                    position=edge.position,
                    feature=edge.feature,
                    graph_weight=edge.graph_weight,
                    activation=edge.activation,
                    write_layers=write_layers,
                    feature_type=feature_type,
                    operator="carrier+arbitrate",
                    covered=True,
                    verdict=_VERDICT_FROM_TRIAL[trial_row.verdict],
                    native_logit_delta=trial_row.consumer_effect,
                    graph_predicted_delta=edge.graph_predicted_delta,
                    graph_vs_native=classify_graph_vs_native(
                        edge.graph_predicted_delta, trial_row.consumer_effect, rule
                    ),
                    clean_logprob=meas.get("clean_logprob"),
                    ablated_logprob=meas.get("ablated_logprob"),
                    top_token_clean=meas.get("top_token_clean"),
                    top_token_ablated=meas.get("top_token_ablated"),
                    top_token_flip=meas.get("top_token_flip"),
                    reason=trial_row.reason,
                    reading_fingerprint=reading.fingerprint,
                    rule_fingerprint=rule.fingerprint,
                    receipt=trial_row.receipt,
                    measurements=meas,
                )
            )
            continue

        if use_hook:
            hook = _residual_hook_effect(
                adapter, tokens, writes, edge.activation, target_token, clean_logits
            )
            effect = hook["consumer_logprob_drop"]
            verdict, _classification, reason = grade(reading, rule, effect)
            receipt = Receipt.make(
                operation="interop.circuit_tracer.native_edge_test.residual_hook",
                model_identity=adapter.model_identity,
                execution=dict(adapter.execution),
                reading=reading.to_dict(),
                decision_rule=rule.to_dict(),
                verdict=verdict,
                consumer_effect=effect,
                measurements=describe(hook),
                layer=edge.layer,
                position=edge.position,
                feature=edge.feature,
                write_layers=list(write_layers),
            )
            rows.append(
                EdgeRow(
                    node_kind="feature",
                    layer=edge.layer,
                    position=edge.position,
                    feature=edge.feature,
                    graph_weight=edge.graph_weight,
                    activation=edge.activation,
                    write_layers=write_layers,
                    feature_type=feature_type,
                    operator="residual_hook",
                    covered=True,
                    verdict=_VERDICT_FROM_TRIAL[verdict],
                    native_logit_delta=effect,
                    graph_predicted_delta=edge.graph_predicted_delta,
                    graph_vs_native=classify_graph_vs_native(
                        edge.graph_predicted_delta, effect, rule
                    ),
                    clean_logprob=hook["clean_logprob"],
                    ablated_logprob=hook["ablated_logprob"],
                    top_token_clean=hook["top_token_clean"],
                    top_token_ablated=hook["top_token_ablated"],
                    top_token_flip=hook["top_token_flip"],
                    reason=reason,
                    reading_fingerprint=reading.fingerprint,
                    rule_fingerprint=rule.fingerprint,
                    receipt=receipt.fingerprint,
                    measurements=hook,
                )
            )
            continue

        rows.append(
            _uncovered_row(
                edge,
                write_layers,
                feature_type,
                reading,
                rule,
                reason="no enabled operator covers this edge (not last position; hook disabled)",
            )
        )

    # transcoder error nodes: carried influence with no feature direction to intervene on.
    rows.append(
        EdgeRow(
            node_kind="error",
            layer=None,
            position=None,
            feature=None,
            graph_weight=selection.error_influence_total,
            activation=None,
            write_layers=(),
            feature_type="transcoder-error",
            operator="uncovered",
            covered=False,
            verdict="uncovered",
            native_logit_delta=None,
            clean_logprob=None,
            ablated_logprob=None,
            top_token_clean=None,
            top_token_ablated=None,
            top_token_flip=None,
            reason=(
                f"{selection.n_error_nodes} transcoder error nodes carry "
                f"{selection.error_node_influence_share:.3f} of node influence; the error term is "
                "the residual the transcoders do not reconstruct, so it has no feature direction "
                "to write as a native Act"
            ),
            reading_fingerprint=None,
            rule_fingerprint=None,
            receipt=None,
            measurements={
                "n_error_nodes": selection.n_error_nodes,
                "error_influence_total": selection.error_influence_total,
                "error_node_influence_share": selection.error_node_influence_share,
            },
        )
    )

    graph_summary = {
        "n_feature_nodes": selection.n_feature_nodes,
        "n_error_nodes": selection.n_error_nodes,
        "feature_influence_total": selection.feature_influence_total,
        "error_influence_total": selection.error_influence_total,
        "error_node_influence_share": selection.error_node_influence_share,
    }
    if graph_scores:
        graph_summary["scores"] = dict(graph_scores)

    table = EdgeVerdictTable(
        model_identity=adapter.model_identity,
        execution=dict(adapter.execution),
        prompt=prompt_text,
        tokens=tuple(tokens),
        target_token=int(target_token),
        target_text=target_text,
        subject_position=subject_position,
        top_k=int(top_k),
        decision_rule=rule.to_dict(),
        rows=tuple(rows),
        graph_summary=graph_summary,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{k: v for k, v in dict(adapter.execution.get("environment_versions", {})).items()},
        },
        custody=dict(store) if isinstance(store, Mapping) else {},
        group_interventions=tuple(dict(g) for g in group_interventions),
    )
    return table


def _uncovered_row(
    edge: _SelectedEdge,
    write_layers: tuple[int, ...],
    feature_type: str,
    reading: Reading,
    rule: DecisionRule,
    *,
    reason: str,
) -> EdgeRow:
    return EdgeRow(
        node_kind="feature",
        layer=edge.layer,
        position=edge.position,
        feature=edge.feature,
        graph_weight=edge.graph_weight,
        activation=edge.activation,
        write_layers=write_layers,
        feature_type=feature_type,
        operator="uncovered",
        covered=False,
        verdict="uncovered",
        native_logit_delta=None,
        clean_logprob=None,
        ablated_logprob=None,
        top_token_clean=None,
        top_token_ablated=None,
        top_token_flip=None,
        reason=reason,
        reading_fingerprint=reading.fingerprint,
        rule_fingerprint=rule.fingerprint,
        receipt=None,
        measurements={},
    )


# --------------------------------------------------------------------------------------
# Offline re-derivation (stdlib only: no torch, no circuit_tracer, no model).
# --------------------------------------------------------------------------------------


def verify_edge_bundle(bundle: str | Path) -> dict[str, Any]:
    """Re-derive every covered verdict in a sealed bundle from its scalars, offline.

    Checks ``edge_verdicts.json`` against its ``manifest.json`` SHA-256, rebuilds the frozen
    :class:`DecisionRule`, and re-grades each covered edge from its sealed
    ``native_logit_delta`` with :func:`saturn_pub.trial.grade`, confirming it reproduces the
    sealed verdict. Uncovered rows (error nodes, out-of-range writes) must stay uncovered.
    Stdlib only; needs no GPU, model, or ``circuit_tracer``.
    """
    directory = Path(bundle)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    body = (directory / "edge_verdicts.json").read_bytes()
    hash_ok = hashlib.sha256(body).hexdigest() == manifest.get("edge_verdicts.json")
    table = json.loads(body)
    stored_fp = table.get("fingerprint")
    recomputed = dict(table)
    recomputed.pop("fingerprint", None)
    fingerprint_ok = digest(recomputed) == stored_fp

    _rule_fields = {
        "name",
        "version",
        "metric",
        "present_threshold",
        "absent_threshold",
        "require_exact_gate",
        "require_sham_flat",
        "sham_metric",
        "sham_tolerance",
        "description",
    }
    rule_body = {k: v for k, v in table["decision_rule"].items() if k in _rule_fields}
    rule = DecisionRule(**rule_body)
    rule_fp_ok = rule.fingerprint == table["decision_rule"].get("fingerprint")

    verdicts: list[dict[str, Any]] = []
    verdict_ok = True
    for row in table["rows"]:
        stored = {k: v for k, v in row.items() if k != "fingerprint"}
        if digest(stored) != row.get("fingerprint"):
            verdict_ok = False
            verdicts.append(
                {"row": _row_id(row), "ok": False, "reason": "row fingerprint mismatch"}
            )
            continue
        if not row["covered"]:
            ok = row["verdict"] == "uncovered"
            verdict_ok = verdict_ok and ok
            verdicts.append({"row": _row_id(row), "ok": ok, "derived": "uncovered"})
            continue
        reading = Reading(
            instrument="external:circuit-tracer",
            claim=row.get("reason") or "attribution edge",
            asserts_effect=True,
            source="circuit-tracer",
        )
        derived, _classification, _reason = grade(reading, rule, float(row["native_logit_delta"]))
        derived_verdict = _VERDICT_FROM_TRIAL[derived]
        derived_gvn = classify_graph_vs_native(
            row.get("graph_predicted_delta"), float(row["native_logit_delta"]), rule
        )
        ok = derived_verdict == row["verdict"] and derived_gvn == row.get("graph_vs_native")
        verdict_ok = verdict_ok and ok
        verdicts.append(
            {
                "row": _row_id(row),
                "ok": ok,
                "derived": derived_verdict,
                "stored": row["verdict"],
                "derived_graph_vs_native": derived_gvn,
            }
        )

    group_ok = True
    group_checks: list[dict[str, Any]] = []
    for g in table.get("group_interventions", []):
        g_reading = Reading(
            instrument="external:circuit-tracer",
            claim=g.get("reason") or "group intervention",
            asserts_effect=True,
            source="circuit-tracer",
        )
        g_derived, _c, _r = grade(g_reading, rule, float(g["native_logit_delta"]))
        g_verdict = _VERDICT_FROM_TRIAL[g_derived]
        g_gvn = classify_graph_vs_native(
            g.get("graph_predicted_delta"), float(g["native_logit_delta"]), rule
        )
        ok = g_verdict == g["verdict"] and g_gvn == g.get("graph_vs_native")
        group_ok = group_ok and ok
        group_checks.append(
            {
                "group": g.get("name"),
                "ok": ok,
                "derived": g_verdict,
                "stored": g["verdict"],
                "derived_graph_vs_native": g_gvn,
            }
        )

    return {
        "ok": bool(hash_ok and fingerprint_ok and rule_fp_ok and verdict_ok and group_ok),
        "hash_ok": bool(hash_ok),
        "fingerprint_ok": bool(fingerprint_ok),
        "rule_fingerprint_ok": bool(rule_fp_ok),
        "verdict_ok": bool(verdict_ok),
        "group_ok": bool(group_ok),
        "bundle": str(directory),
        "summary": table.get("summary"),
        "verdicts": verdicts,
        "group_checks": group_checks,
    }


def _row_id(row: Mapping[str, Any]) -> str:
    if row["node_kind"] == "error":
        return "error-nodes"
    return f"L{row['layer']}/p{row['position']}/f{row['feature']}"


# --------------------------------------------------------------------------------------
# Synthetic graph for offline tests / the notebook's mechanics demo.
# --------------------------------------------------------------------------------------


@dataclass
class _SyntheticConfig:
    n_layers: int


@dataclass
class SyntheticAttributionGraph:
    """A minimal stand-in for a circuit-tracer ``Graph`` for offline tests and examples.

    It carries exactly the fields :func:`native_edge_test` reads, in circuit-tracer's node
    order (features, then ``n_layers * n_pos`` error nodes, then ``n_pos`` embed nodes, then
    logit nodes). No ``circuit_tracer`` import is required.
    """

    active_features: Any
    selected_features: Any
    activation_values: Any
    adjacency_matrix: Any
    logit_probabilities: Any
    input_tokens: Any
    n_pos: int
    cfg: _SyntheticConfig


def build_synthetic_graph(
    *,
    n_layers: int,
    input_tokens: Sequence[int],
    features: Sequence[tuple[int, int, int, float]],
    logit_weights: Sequence[float] = (1.0,),
    seed: int = 0,
    unselected: Sequence[tuple[int, int, int, float]] = (),
) -> SyntheticAttributionGraph:
    """Build a synthetic graph whose feature nodes feed the logit with decreasing influence.

    ``features`` is a list of ``(layer, position, feature_idx, activation)``. Earlier entries
    are given larger edges to the (single) logit node so the influence ranking is
    deterministic. The adjacency matrix is wired features -> logit directly.

    ``unselected`` adds active features that attribution did not select. As in circuit-tracer,
    ``active_features`` and ``activation_values`` then cover every active feature (unselected
    first), and ``selected_features`` indexes the selected ones within them.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    n_pos = len(input_tokens)
    n_feat = len(features)
    n_err = n_layers * n_pos
    n_embed = n_pos
    n_logit = len(logit_weights)
    n_nodes = n_feat + n_err + n_embed + n_logit
    adjacency = torch.zeros(n_nodes, n_nodes)
    # logit node (target node row) attends to feature + error source columns
    logit_row = n_nodes - n_logit
    for i in range(n_feat):
        adjacency[logit_row, i] = float(n_feat - i)  # decreasing feature influence
    # give error nodes a modest, nonzero influence so the error share is reported
    err_start = n_feat
    for j in range(n_err):
        adjacency[logit_row, err_start + j] = 0.3 + 0.01 * (torch.rand((), generator=g).item())
    every = list(unselected) + list(features)
    active = torch.tensor(
        [[layer, pos, feat] for (layer, pos, feat, _a) in every], dtype=torch.long
    )
    activation_values = torch.tensor([a for (_l, _p, _f, a) in every], dtype=torch.float32)
    n_skip = len(unselected)
    return SyntheticAttributionGraph(
        active_features=active,
        selected_features=torch.arange(n_skip, n_skip + n_feat, dtype=torch.long),
        activation_values=activation_values,
        adjacency_matrix=adjacency,
        logit_probabilities=torch.tensor(list(logit_weights), dtype=torch.float32),
        input_tokens=torch.tensor(list(input_tokens), dtype=torch.long),
        n_pos=n_pos,
        cfg=_SyntheticConfig(n_layers=n_layers),
    )


__all__ = [
    "native_edge_test",
    "native_group_intervention",
    "classify_graph_vs_native",
    "verify_edge_bundle",
    "select_edges",
    "default_edge_rule",
    "EdgeRow",
    "EdgeVerdictTable",
    "GraphInfluence",
    "SyntheticAttributionGraph",
    "build_synthetic_graph",
    "DecoderWrites",
]
