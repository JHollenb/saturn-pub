"""The embedding API: attach to an already-loaded Hugging Face causal LM and step it.

    dbg = mdb.attach(model, tokenizer)        # eager attention, eval mode, one sequence
    run = dbg.run("Weather: freezing")        # a port-level Run, cursor at t<last>.L0.h
    run.until("L1.p"); dbg.repl(run)          # or: ctx = dbg.ops(run); dispatch(ctx, ...)

This module lives in ``mdb.ops`` (the lowest layer that may know the operation registry);
``mdb.attach`` / ``mdb.open_tiny`` / ``mdb.Debugger`` resolve here lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mdb.core.errors import Refused
from mdb.engine.ports import PROGRAMS, program_for
from mdb.engine.run import Run

from .context import Context


class Debugger:
    """One attached model: its saturn_pub operation-granularity adapter and tokenizer."""

    def __init__(self, model: Any, tokenizer: Any, adapter: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.program = program_for(adapter)

    @property
    def family(self) -> str:
        return self.program.family

    def tokens(self, prompt: str | Sequence[int] | Any) -> list[int]:
        """Token ids of one sequence; text needs the tokenizer, batches are refused."""
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise Refused("text prompts need a tokenizer; attach(model, tokenizer) or pass "
                              "token ids")
            ids = self.tokenizer(prompt)["input_ids"]
        elif hasattr(prompt, "tolist"):
            if getattr(prompt, "ndim", 1) == 2:
                if prompt.shape[0] != 1:
                    raise Refused(f"mdb steps one sequence (batch 1), got batch {prompt.shape[0]}")
                prompt = prompt[0]
            elif getattr(prompt, "ndim", 1) != 1:
                raise Refused("token ids must be one sequence")
            ids = prompt.tolist()
        else:
            ids = list(prompt)
        if ids and isinstance(ids[0], (list, tuple)):
            if len(ids) != 1:
                raise Refused(f"mdb steps one sequence (batch 1), got batch {len(ids)}")
            ids = list(ids[0])
        if not ids:
            raise Refused("the prompt has no tokens")
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in ids):
            raise Refused("token ids must be nonnegative integers")
        if max(ids) >= self.adapter.vocab_size:
            raise Refused(f"token id {max(ids)} is outside the vocabulary "
                          f"({self.adapter.vocab_size})")
        if len(ids) > self.adapter.max_positions:
            raise Refused(f"{len(ids)} tokens exceed max_position_embeddings "
                          f"{self.adapter.max_positions}")
        return [int(t) for t in ids]

    def run(
        self,
        prompt: str | Sequence[int],
        *,
        retain_rows: int | None = None,
        lookahead: str | None = None,
    ) -> Run:
        """Prefill all but the last token; the cursor starts where the ``start`` op leaves it,
        at ``t<last>.L0.<first port>`` (this is the one implementation of ``start``).

        ``lookahead`` (``"none"``, the engine default, or ``"layer"``) is passed to ``Run``.
        """
        options: dict[str, Any] = {"retain_rows": retain_rows}
        if lookahead is not None:
            options["lookahead"] = lookahead
        run = Run(self.adapter, self.tokens(prompt), **options)
        run.until(f"L0.{run.program.names[0]}")
        return run

    def ops(self, run: Run | None = None):
        """A context for ``mdb.ops.dispatch`` bound to this model (and ``run``, if given)."""
        return Context(self, run)

    def repl(self, run: Run | None = None, **options: Any) -> int:
        """Interactive gdb-style REPL on this model; returns when the user quits."""
        from mdb.repl.shell import Shell  # sanctioned upward import: Debugger.repl convenience

        return Shell(self.ops(run), **options).loop()


def attach(model: Any, tokenizer: Any = None, *, project: Any = None) -> Debugger:
    """Attach to a loaded Hugging Face causal LM (Qwen2/Qwen3, eager attention, eval mode).

    ``project=`` (persistence of records, cursors and recipes) is not built yet and is
    refused rather than accepted silently.
    """
    if project is not None:
        raise Refused(
            "project persistence is not built yet (project index v4 and store wiring); "
            "attach without project= for now"
        )
    config = getattr(model, "config", None)
    if config is None or not hasattr(model, "parameters"):
        raise Refused("attach needs a loaded Hugging Face causal LM (a model with .config)")
    family = getattr(config, "model_type", None)
    if family not in PROGRAMS:
        raise Refused(
            f"no port program for model_type {family!r}; supported: {', '.join(sorted(PROGRAMS))}"
        )
    implementation = getattr(config, "_attn_implementation", None)
    if implementation != "eager":
        raise Refused(
            f"attention implementation is {implementation!r}; port stepping needs eager "
            "attention: load with from_pretrained(..., attn_implementation='eager')"
        )
    training = [name or "<model>" for name, module in model.named_modules() if module.training]
    if training:
        raise Refused(
            f"{len(training)} sub-module(s) are in training mode (dropout), e.g. "
            f"{training[0]!r}; call model.eval() first (attach never changes your model)"
        )
    from saturn_pub.adapters.decoder import DecoderAdapter

    try:
        adapter = DecoderAdapter(model, granularity="operation")
    except ValueError as error:
        raise Refused(f"cannot attach: {error}") from error
    return Debugger(model, tokenizer, adapter)


def open_tiny(family: str = "qwen2", dtype: str = "bf16", *, seed: int = 113) -> Debugger:
    """A random tiny eager CPU model (no tokenizer) for demos and tests."""
    return attach(_tiny(family, dtype, seed))


def _tiny(family: str, dtype: str, seed: int) -> Any:
    from mdb.engine.models import tiny_model

    return tiny_model(family, dtype, seed=seed)
