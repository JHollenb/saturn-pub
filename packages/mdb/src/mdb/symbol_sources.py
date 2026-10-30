"""Finite-vocabulary symbol proposals from explicit observation sources.

These readers never receive an expected answer or choose a native address.
The vocabulary and its semantic labels are supplied; scores and model-emitted
history are observations. Native consumers qualify their meaning separately.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from saturn_pub.values import digest

VOCABULARY_SCHEMA = "mdb-symbol-vocabulary-v1"
CANDIDATE_SCHEMA = "mdb-symbol-source-candidate-v1"
_EVALUATOR_FIELDS = frozenset({"expected", "expected_answer", "ground_truth", "gold",
                             "teacher", "oracle", "reference_answer", "target_label"})


class SymbolSourceError(ValueError):
    pass


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "fingerprint": digest(body)}


def _verify(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    body = dict(value)
    fingerprint = body.pop("fingerprint", None)
    if body.get("schema") != schema or fingerprint != digest(body):
        raise SymbolSourceError("source schema or fingerprint mismatch")
    return body


def _source_only(value: Any) -> None:
    if isinstance(value, Mapping):
        if any(str(key).casefold() in _EVALUATOR_FIELDS for key in value):
            raise SymbolSourceError("evaluator fields cannot enter a source reader")
        for item in value.values():
            _source_only(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _source_only(item)


def make_vocabulary(labels: Sequence[str], *, namespace: str,
                    evidence_refs: Sequence[str]) -> dict[str, Any]:
    labels = list(labels)
    if (not isinstance(namespace, str) or not namespace.strip() or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise SymbolSourceError("vocabulary needs distinct nonempty labels and a namespace")
    if not evidence_refs or any(not isinstance(ref, str) or not ref for ref in evidence_refs):
        raise SymbolSourceError("vocabulary needs provenance")
    return _seal({"schema": VOCABULARY_SCHEMA, "namespace": namespace,
                  "labels": labels, "evidence_refs": list(evidence_refs),
                  "label_origin": "supplied-vocabulary; no inferred semantic ontology"})


def _candidate(vocabulary, *, kind, model_identity, execution, source_cut,
               context, evidence_refs, producer_id, source, proposals, selected,
               reason=None):
    vocab = _verify(vocabulary, VOCABULARY_SCHEMA)
    if any(not isinstance(v, str) or not v for v in
           (model_identity, execution, source_cut, context, producer_id)):
        raise SymbolSourceError("candidate needs exact model, execution, cut and context")
    if not evidence_refs or any(not isinstance(ref, str) or not ref for ref in evidence_refs):
        raise SymbolSourceError("candidate needs source evidence")
    labels = vocab["labels"]
    _source_only(source)
    if selected is not None and selected not in labels:
        raise SymbolSourceError("selected symbol is outside the frozen vocabulary")
    return _seal({
        "schema": CANDIDATE_SCHEMA, "source_kind": kind, "producer_id": producer_id,
        "model_identity": model_identity, "execution": execution,
        "source_cut": source_cut, "context": context,
        "vocabulary": dict(vocabulary), "vocabulary_fingerprint": vocabulary["fingerprint"],
        "source": source, "source_digest": digest(source),
        "proposals": proposals, "selected_label": selected,
        "selected_symbol": None if selected is None else vocab["namespace"] + "." + selected,
        "abstain_reason": reason, "evidence_refs": list(evidence_refs),
        "authority": "source-bound-candidate-observation", "semantic_qualified": False,
        "causal_dependency_tested": False,
        "source_trust": "producer-supplied source; native witness verification is separate",
        "statuses": {"mechanics": "valid", "instrument": "source-observed",
                     "trend": "not-assessed", "terminal": "not-assessed"},
    })


def nominate_prompt(tokens: Sequence[str], vocabulary: Mapping[str, Any], *,
                    model_identity: str, execution: str, source_cut: str, context: str,
                    evidence_refs: Sequence[str], producer_id: str = "literal-token-reader",
                    source_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Match literal vocabulary tokens; parsing and tokenization stay explicit."""
    vocab = _verify(vocabulary, VOCABULARY_SCHEMA)
    tokens = list(tokens)
    if any(not isinstance(token, str) for token in tokens):
        raise SymbolSourceError("prompt tokens must be strings from the declared parser")
    matches = [{"label": token, "position": index} for index, token in enumerate(tokens)
               if token in vocab["labels"]]
    unique = list(dict.fromkeys(row["label"] for row in matches))
    return _candidate(vocabulary, kind="prompt", model_identity=model_identity,
        execution=execution, source_cut=source_cut, context=context,
        evidence_refs=evidence_refs, producer_id=producer_id,
        source={"tokens": tokens, "metadata": dict(source_metadata or {})},
        proposals=matches, selected=unique[0] if len(unique) == 1 else None,
        reason=None if len(unique) == 1 else "no-literal-symbol" if not unique else "multiple-literal-symbols")


def nominate_scores(scores: Sequence[float], vocabulary: Mapping[str, Any], *,
                    model_identity: str, execution: str, source_cut: str, context: str,
                    evidence_refs: Sequence[str], producer_id: str, top_k: int = 3,
                    source_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read a frozen decoder's finite logits in its original vocabulary order."""
    vocab = _verify(vocabulary, VOCABULARY_SCHEMA)
    scores = list(scores)
    if (len(scores) != len(vocab["labels"])
            or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
        raise SymbolSourceError("finite scores must match the frozen vocabulary width")
    if type(top_k) is not int or not 1 <= top_k <= len(scores):
        raise SymbolSourceError("top_k is outside the frozen vocabulary")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    top = scores[order[0]]
    tied = sum(score == top for score in scores) > 1
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    proposals = [{"label": vocab["labels"][i], "index": i,
                  "logit": float(scores[i]), "probability": weights[i] / total}
                 for i in order[:top_k]]
    return _candidate(vocabulary, kind="residual", model_identity=model_identity,
        execution=execution, source_cut=source_cut, context=context,
        evidence_refs=evidence_refs, producer_id=producer_id,
        source={"logits": scores, "top_margin": float(top - scores[order[1]]) if len(scores) > 1 else None,
                "metadata": dict(source_metadata or {}),
                "readout": "supplied-frozen-model-decoder; no new probe fit"},
        proposals=proposals, selected=None if tied else vocab["labels"][order[0]],
        reason="tied-native-scores" if tied else None)


def nominate_history(events: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, Any], *,
                     model_identity: str, execution: str, source_cut: str, context: str,
                     evidence_refs: Sequence[str], producer_id: str = "model-decision-history") -> dict[str, Any]:
    """Retain model-emitted decisions in order, without inserting teacher steps."""
    vocab = _verify(vocabulary, VOCABULARY_SCHEMA)
    history = []
    for index, event in enumerate(events):
        if (set(event) - {"index", "symbol", "source_cut", "receipt", "authority", "commit_status"}
                or event.get("authority") != "model-predicted"
                or event.get("index") != index
                or not isinstance(event.get("source_cut"), str) or not event.get("source_cut")
                or not isinstance(event.get("receipt"), str) or not event.get("receipt")
                or event.get("commit_status") not in {"committed", "rejected", "proposed"}):
            raise SymbolSourceError("history requires ordered model-predicted events with native references")
        history.append(dict(event))
    committed = [event for event in history if event["commit_status"] == "committed"]
    unknown = any(event.get("symbol") not in vocab["labels"] for event in committed)
    selected = committed[-1]["symbol"] if committed and not unknown else None
    return _candidate(vocabulary, kind="generated-history", model_identity=model_identity,
        execution=execution, source_cut=source_cut, context=context,
        evidence_refs=evidence_refs, producer_id=producer_id,
        source={"events": history, "textual_cot": False,
                "history_origin": "model-predicted decisions; native consumption checked separately"},
        proposals=[{"label": event["symbol"], "position": index,
                    "commit_status": event["commit_status"]}
                   for index, event in enumerate(history)], selected=selected,
        reason="unknown-history-symbol" if unknown else "no-committed-generated-history" if not committed else None)


def verify_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    body = _verify(value, CANDIDATE_SCHEMA)
    _verify(body["vocabulary"], VOCABULARY_SCHEMA)
    if (body["vocabulary_fingerprint"] != body["vocabulary"]["fingerprint"]
            or body["source_digest"] != digest(body["source"])
            or body["semantic_qualified"] is not False
            or body["causal_dependency_tested"] is not False):
        raise SymbolSourceError("source identity or candidate authority changed")
    arguments = {key: body[key] for key in ("model_identity", "execution", "source_cut", "context",
                                           "evidence_refs", "producer_id")}
    kind = body["source_kind"]
    source = body["source"]
    if kind == "prompt":
        expected = nominate_prompt(source["tokens"], body["vocabulary"],
                                   source_metadata=source["metadata"], **arguments)
    elif kind == "residual":
        expected = nominate_scores(source["logits"], body["vocabulary"],
                                   top_k=len(body["proposals"]),
                                   source_metadata=source["metadata"], **arguments)
    elif kind == "generated-history":
        expected = nominate_history(source["events"], body["vocabulary"], **arguments)
    else:
        raise SymbolSourceError("unknown symbol source kind")
    if dict(value) != expected:
        raise SymbolSourceError("candidate differs from its source recomputation")
    return {"passed": True, "fingerprint": value["fingerprint"],
            "authority": "source-bound-candidate-observation"}
