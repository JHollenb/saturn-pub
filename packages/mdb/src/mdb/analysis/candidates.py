"""First-class candidate sets for content readouts (symbol-table additions review 2026-11-14,
2.3 and 2.4).

A ``CandidateSet`` is a supplied word list turned into token ids by a declared rule, with its
provenance sealed (the sha256 of the canonical word list, its source and the rule): the
vocabulary is supplied, never inferred, and a reader never receives an expected answer or
chooses a native address (the rule ported from ``mdb_lanes.symbol_sources``). Per word it
records the token ids of ``" " + word``, ``n_pieces``, the first piece and whether that first
piece is shared with another word of the same set (``collides_within``) or of another declared
set (``collides_across``). Token facts are structural; the flags are ``derived-exact`` (a
deterministic function of the tokenizer files).

``content_flags`` adds to a content readout of one set:

* ``first_subword``: the best token is the first piece of a multi-piece word of the set;
* ``collision``: the best token stands for more than one word of the set (``words`` lists them);
* ``kind``: the other declared sets that also hold the best token (B1: the Czech country and
  language tokens are identical, so a "names the country" count includes language wins).

Readouts beside ``content`` (all float64, linear in the write, derived-exact P19):

* ``lexical_null(source_token)``: the same readout applied to ``W_U[source_token]`` itself
  (through the same ``gamma``): does the token's own unembedding already name the truth? A
  ``names_truth`` beside a lexical null that also names it is a test with no power
  (AUDIT-SCALE: 46-48/48 on the capital test).
* ``co_readout``: read set ``S`` with a whole subspace projected out of the write: by default
  the span of every ``gamma * W_U[t]`` of another set (``project_out``), or its top ``k``
  principal directions. One direction is not enough: projecting out a single country
  direction leaves the true country in the write (AUDIT-SCALE.md:133-157), so the default is
  the whole span. Reports ``raw``, ``projected``, ``lexical_null`` (the readout of the removed
  part alone: the projection of the write onto the subspace) and ``counts`` = projected names
  the truth and the removed part alone does not. The projection is ``supplied`` (the caller
  chose the subspace) and says so in ``premises``.
* ``switched``: the truth of set ``S`` made a function of the decoded token of another set
  (``switch``): rank the capital of the country the write names, not the item's true capital
  (the 2.7 switched readout; ``pairs`` maps the other set's word to this set's word).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .finite import EXACT, SUPPLIED, f64, num
from .forecast import Refusal

SCHEMA = "mdb-candidate-set-v1"
RULES = ("first", "single")


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class CandidateSet:
    """A supplied word list as token ids (module docstring). ``pieces[i]`` are the ids of
    ``" " + words[i]``; ``tokens[i]`` its first piece (the id a readout ranks)."""

    name: str
    words: tuple[str, ...]
    pieces: tuple[tuple[int, ...], ...]
    rule: str = "first"
    source: str = "supplied"
    pairs: dict[str, str] = field(default_factory=dict, compare=False)
    across: dict[int, tuple[str, ...]] = field(default_factory=dict, compare=False)

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(p[0] for p in self.pieces)

    @property
    def sha256(self) -> str:
        text = json.dumps({"schema": SCHEMA, "name": self.name, "words": list(self.words),
                           "rule": self.rule}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def token(self, word: str) -> int:
        if word not in self.words:
            raise Refusal(f"{word!r} is not a word of candidate set {self.name}")
        return self.pieces[self.words.index(word)][0]

    def words_of(self, token: int) -> list[str]:
        return [w for w, p in zip(self.words, self.pieces) if p[0] == int(token)]

    def collides_within(self, word: str) -> bool:
        return len(self.words_of(self.token(word))) > 1

    def multi_piece(self, token: int) -> bool:
        return any(p[0] == int(token) and len(p) > 1 for p in self.pieces)

    def candidates(self, truth: Any = None) -> Any:
        """The ``symbols.Candidates`` of this set; ``truth`` a word of the set or a token id."""
        from .symbols import Candidates

        if isinstance(truth, str):
            truth = self.token(truth)
        return Candidates(self.name, self.tokens, None if truth is None else int(truth),
                          info=self)

    def describe(self) -> dict[str, Any]:
        """Record-safe provenance: name, size, rule, source, sha256 and per-set flag counts."""
        within = sum(1 for w in self.words if self.collides_within(w))
        multi = sum(1 for p in self.pieces if len(p) > 1)
        return {"name": self.name, "size": len(self.words), "rule": self.rule,
                "source": self.source, "sha256": self.sha256,
                "words_collide_within": within, "multi_piece_words": multi,
                "first_tokens_shared_across": sum(1 for t in set(self.tokens) if t in self.across),
                "authority": "structural (tokenizer); flags derived-exact"}


def build(name: str, words: list[str], tokenizer: Any, *, rule: str = "first",
          source: str = "supplied", pairs: dict[str, str] | None = None) -> CandidateSet:
    """Tokenise ``" " + word`` for each word (``rule``: ``first`` keeps multi-piece words and
    ranks their first piece; ``single`` refuses them)."""
    if rule not in RULES:
        raise Refusal(f"rule is one of {RULES}, not {rule!r}")
    words = [str(w) for w in words]
    if len(words) < 2 or len(set(words)) != len(words) or any(not w.strip() for w in words):
        raise Refusal(f"candidate set {name} needs at least two distinct non-empty words")
    if tokenizer is None:
        raise Refusal("a candidate set from words needs a tokenizer (give token ids instead)")
    pieces = []
    for word in words:
        ids = tuple(int(i) for i in tokenizer.encode(" " + word, add_special_tokens=False))
        if not ids:
            raise Refusal(f"{word!r} encodes to no token")
        if rule == "single" and len(ids) != 1:
            raise Refusal(f"rule=single: {word!r} is {len(ids)} tokens")
        pieces.append(ids)
    return CandidateSet(name, tuple(words), tuple(pieces), rule, source, dict(pairs or {}))


def from_tokens(name: str, tokens: list[int], words: list[str] | None = None, *,
                source: str = "supplied") -> CandidateSet:
    """A set given as token ids (one piece each); ``words`` default to ``#<id>``."""
    words = list(words) if words is not None else [f"#{int(t)}" for t in tokens]
    if len(words) != len(tokens):
        raise Refusal("words and tokens differ in length")
    return CandidateSet(name, tuple(words), tuple((int(t),) for t in tokens), "first", source)


def cross(sets: list[CandidateSet]) -> list[CandidateSet]:
    """The sets with ``across`` filled: per first token, the other sets that hold it."""
    holders: dict[int, list[str]] = {}
    for s in sets:
        for t in set(s.tokens):
            holders.setdefault(t, []).append(s.name)
    out = []
    for s in sets:
        across = {t: tuple(n for n in holders[t] if n != s.name) for t in set(s.tokens)
                  if len(holders[t]) > 1}
        out.append(CandidateSet(s.name, s.words, s.pieces, s.rule, s.source, dict(s.pairs),
                                across))
    return out


def pair(target: CandidateSet, on: CandidateSet) -> CandidateSet:
    """``target`` with ``pairs`` from parallel word lists: ``on.words[i] -> target.words[i]``."""
    if len(target.words) != len(on.words):
        raise Refusal(f"sets {on.name} and {target.name} are not parallel ({len(on.words)} vs "
                      f"{len(target.words)} words)")
    return CandidateSet(target.name, target.words, target.pieces, target.rule, target.source,
                        dict(zip(on.words, target.words)), dict(target.across))


def load(path: str, name: str, tokenizer: Any, *, rule: str = "first") -> CandidateSet:
    """A set from a file: a JSON list of words, ``{"words": [...], "pairs": {...}}``, or text
    with one word per line."""
    text = Path(path).expanduser().read_text()
    pairs = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        pairs = data.get("pairs")
        data = data.get("words", [])
    return build(name, list(data), tokenizer, rule=rule, source=str(path), pairs=pairs)


# --------------------------------------------------------------------------- readouts


def _rank(scores: Any, tokens: tuple[int, ...], truth: int | None) -> dict[str, Any]:
    order = scores.argsort(descending=True)
    best, second = int(order[0]), int(order[1])
    out: dict[str, Any] = {"best": tokens[best],
                           "margin": num(float(scores[best] - scores[second]), EXACT, "P19")}
    if truth is not None and truth in tokens:
        where = tokens.index(truth)
        out["truth_rank"] = int((scores > scores[where]).sum()) + 1
        out["names_truth"] = tokens[best] == truth
    return out


def _rows(run: Any, tokens: tuple[int, ...], gamma: Any) -> Any:
    """``gamma * W_U[tokens]`` (n, d) float64 (rows sliced, then cast)."""
    torch = _torch()
    weight = run.adapter.model.lm_head.weight
    index = torch.tensor(list(tokens), dtype=torch.long, device=weight.device)
    return f64(weight.index_select(0, index)).cpu() * gamma


def content_flags(cand: Any, best: int, sets: list[Any]) -> dict[str, Any]:
    """``first_subword``, ``collision`` and ``kind`` of a best token (module docstring)."""
    info = getattr(cand, "info", None)
    words = info.words_of(best) if info is not None else []
    kind = sorted({c.name for c in sets if c.name != cand.name and best in c.tokens})
    out: dict[str, Any] = {"collision": len(words) > 1, "kind": kind,
                           "authority": "derived-exact (tokenizer)"}
    if info is not None:
        out["first_subword"] = info.multi_piece(best)
        out["words"] = words[:8]
    else:
        out["first_subword"] = None
        out["note"] = "token-id set: no word pieces known (declare it with words)"
    truth = getattr(cand, "truth", None)
    if truth is not None:
        out["truth_collides"] = info is not None and len(info.words_of(truth)) > 1
        out["truth_kind"] = sorted({c.name for c in sets if c.name != cand.name
                                    and truth in c.tokens})
    return out


def lexical_null(run: Any, cand: Any, source_token: int, basis: tuple[float, Any]) -> dict:
    """The readout of ``W_U[source_token]`` through the same ``rho * gamma`` (module doc)."""
    rho, gamma = basis
    weight = run.adapter.model.lm_head.weight
    direction = f64(weight[int(source_token)]).cpu().reshape(-1)
    scores = rho * (_rows(run, cand.tokens, gamma) @ direction)
    return {**_rank(scores, cand.tokens, cand.truth), "source_token": int(source_token),
            "meaning": "the readout of the source token's own unembedding row: does lexical "
                       "similarity alone name the truth?"}


@dataclass(frozen=True)
class CoReadout:
    """Read set ``target`` with the span of set ``project_out`` (or ``tokens``) removed;
    ``k``: keep only the top-k principal directions of that span (None: the whole span)."""

    target: str
    project_out: str | None = None
    tokens: tuple[int, ...] | None = None
    k: int | None = None

    def text(self) -> str:
        what = self.project_out or f"{len(self.tokens or ())} tokens"
        return f"{self.target} with span({what}){'' if self.k is None else f' top-{self.k}'} " \
               "projected out"


def subspace(run: Any, tokens: tuple[int, ...], gamma: Any, k: int | None = None) -> Any:
    """An orthonormal basis ``Q`` (d, r) of ``span(gamma * W_U[tokens])`` (rank-revealing SVD,
    float64), or its top-``k`` right singular directions."""
    torch = _torch()
    rows = _rows(run, tuple(dict.fromkeys(tokens)), gamma)
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    keep = int((s > s[0] * rows.shape[1] * 2.0 ** -52).sum()) if s.numel() else 0
    if k is not None:
        keep = min(keep, int(k))
    return vh[:keep].T.contiguous()


def co_readout(run: Any, write: Any, cand: Any, q: Any, basis: tuple[float, Any],
               spec: CoReadout) -> dict[str, Any]:
    """Raw, projected and removed-part readouts of ``cand`` (module docstring)."""
    rho, gamma = basis
    w = f64(write).reshape(-1).cpu()
    inside = q @ (q.T @ w)
    outside = w - inside
    rows = _rows(run, cand.tokens, gamma)
    raw = _rank(rho * (rows @ w), cand.tokens, cand.truth)
    projected = _rank(rho * (rows @ outside), cand.tokens, cand.truth)
    removed = _rank(rho * (rows @ inside), cand.tokens, cand.truth)
    out: dict[str, Any] = {
        "raw": raw, "projected": projected, "lexical_null": removed,
        "rank": int(q.shape[1]),
        "removed_norm_share": num(float(inside.norm() / w.norm()) if float(w.norm()) else 0.0,
                                  EXACT),
        "projection": {"text": spec.text(), "authority": SUPPLIED},
        "premises": ["the subspace is supplied by the caller (project_out / tokens / k); it is "
                     "removed from the write and so from every candidate's score",
                     "lexical_null reads the removed part alone"]}
    if "names_truth" in projected:
        out["counts"] = bool(projected["names_truth"]) and not bool(removed.get("names_truth"))
    return out


def switched(cand: Any, on: dict[str, Any], on_set: Any, scores_fn: Any) -> dict[str, Any]:
    """The readout of ``cand`` with its truth switched on the decoded token of ``on_set``
    (``on``: that set's content entry). ``scores_fn(tokens)``: the write's scores."""
    info = getattr(cand, "info", None)
    on_info = getattr(on_set, "info", None)
    if info is None or on_info is None or not info.pairs:
        return {"state": "unresolved", "reason": f"switch needs word pairs {on_set.name} -> "
                                                 f"{cand.name} (candidates ... paired=)"}
    decoded_words = on_info.words_of(on["best"])
    targets = [info.pairs[w] for w in decoded_words if w in info.pairs]
    if not targets:
        return {"state": "unresolved", "reason": f"the decoded {on_set.name} token has no pair "
                                                 f"in {cand.name}"}
    truth = info.token(targets[0])
    scores = scores_fn(cand.tokens)
    out = _rank(scores, cand.tokens, truth)
    out.update({"on": on_set.name, "decoded": on["best"], "decoded_words": decoded_words[:4],
                "truth": truth, "truth_word": targets[0],
                "ambiguous": len(set(targets)) > 1,
                "meaning": f"rank of the {cand.name} word paired with the {on_set.name} word the "
                           "write names (not the item's truth)"})
    return out


__all__ = ["CandidateSet", "CoReadout", "RULES", "SCHEMA", "build", "co_readout",
           "content_flags", "cross", "from_tokens", "lexical_null", "load", "pair", "subspace",
           "switched"]
