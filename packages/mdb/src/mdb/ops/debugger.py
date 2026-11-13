"""The embedding API: attach to an already-loaded Hugging Face causal LM and step it.

    dbg = mdb.attach(model, tokenizer)        # eager attention, eval mode, one sequence
    run = dbg.run("Weather: freezing")        # a port-level Run, cursor at t<last>.L0.h
    run.until("L1.p"); dbg.repl(run)          # or: ctx = dbg.ops(run); dispatch(ctx, ...)

With ``project=`` the debugger is durable: ``attach(model, tok, project="runs/t")`` creates or
reopens an ``mdb.core.project.Project`` and a session bound to the model identity and
execution; ``dbg.checkpoint(run, "pre")`` stores the replay recipe, the edit tensors and the
boundary cut; ``dbg.restart("pre")`` (any later process) rebuilds that cursor with the same cut
fingerprint; ``dbg.export(run, "t3.L1.u")`` returns an occurrence pointer ``mdb.resolve``
turns back into the tensor.

This module lives in ``mdb.ops`` (the lowest layer that may know the operation registry);
``mdb.attach`` / ``mdb.open_tiny`` / ``mdb.Debugger`` resolve here lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from saturn_pub.values import digest

from mdb.core import index
from mdb.core import recipe as recipes
from mdb.core.address import AddressError, Origin, parse
from mdb.core.errors import Refused
from mdb.core.pointers import RecordError, Unresolved, occurrence_pointer
from mdb.engine.ports import PROGRAMS, program_for, tensor_digest
from mdb.engine.run import ReplayMismatch, Run

from .context import Context


class Debugger:
    """One attached model: its saturn_pub operation-granularity adapter and tokenizer, and with
    a project, the durable session (``project``, ``session_id``). ``live`` is the run this
    debugger made or restarted last (the resolver's ``live`` tier reads its port cache)."""

    def __init__(
        self, model: Any, tokenizer: Any, adapter: Any, *, project: Any = None,
        session_id: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.program = program_for(adapter)
        self.project = project
        self.session_id = session_id
        self.live: Run | None = None

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
        self.live = run
        return run

    # --- durable checkpoints (project=) ---------------------------------------------------

    def _project(self, what: str) -> Any:
        if self.project is None:
            raise Refused(f"{what} needs a project: attach(model, tokenizer, project=PATH) "
                          "(run.checkpoint() keeps an in-memory checkpoint)")
        if self.project.read_only:
            raise Refused(f"{what} needs a writable project")
        return self.project

    def _numeric_scope(self) -> dict[str, Any]:
        return dict(self.adapter.execution.get("numeric_scope") or {})

    def _save_recipe(self, run: Run, name: str | None) -> dict[str, Any]:
        """Store the recipe of ``run``'s cursor (edit tensors as blobs) and its boundary cut."""
        project = self._project("a durable checkpoint")
        if run.adapter is not self.adapter:
            raise Refused("the run belongs to another attached model")
        checkpoint = run.checkpoint(name)
        cut = run.session.capture(retain=False)
        if cut.fingerprint != checkpoint["cut"]:
            raise Refused("the run's session is not at its cursor's boundary cut")
        try:
            recipe = recipes.from_checkpoint(checkpoint, tokens=run.tokens,
                                             layers=self.adapter.layers,
                                             tensors=run.edit_tensors(), store=project.store)
        except RecordError as error:
            raise Refused(f"the checkpoint does not fit one replay recipe: {error}") from error
        recipes.save(project, self.session_id, recipe)
        project.save_cut(self.session_id, cut)
        return recipe

    def checkpoint(self, run: Run, name: str) -> dict[str, Any]:
        """Make ``run``'s cursor durable under ``name``: the replay recipe, its edit tensors and
        masks (store blobs), the boundary cut, a ``checkpoint`` record and the session cursor."""
        if not isinstance(name, str) or not name.strip():
            raise Refused("a durable checkpoint needs a name")
        recipe = self._save_recipe(run, name)
        record = recipes.record_checkpoint(self.project, self.session_id, recipe)
        self.project.select(self.session_id, recipe["cut"])
        return {"name": name, "where": recipe["target"]["where"], "cut": recipe["cut"],
                "recipe": recipe["fingerprint"], "record": record["id"],
                "blobs": len(recipes.tensor_pointers(recipe))}

    def checkpoints(self) -> list[dict[str, Any]]:
        """This session's durable checkpoints, oldest first: name, where, cut, recipe."""
        project = self._project("listing durable checkpoints")
        return [{"name": r["body"]["name"], "where": r["body"]["where"], "cut": r["body"]["cut"],
                 "recipe": r["body"]["recipe"]}
                for r in recipes.checkpoints(project, self.session_id)]

    def restart(self, name: str) -> Run:
        """A new run at the durable checkpoint ``name`` (this session's, else the one other
        session that has it), replayed from the token ids: same cut fingerprint, same edits."""
        project = self._project("restart")
        found = recipes.find_checkpoint(project, name, session_id=self.session_id)
        if found is None:
            raise Refused(f"no durable checkpoint {name!r} in {project.root}")
        run = self.replay(recipes.load(project, found["body"]["recipe"]))
        self.live = run
        return run

    def replay(self, recipe: dict[str, Any]) -> Run:
        """A new run restored to a replay recipe of this project (``RecipeError`` refusals:
        model identity, numeric scope, missing or corrupt blob, fingerprint mismatch)."""
        return self._restored(recipes.to_checkpoint(recipe), recipe, self._project("replay").store)

    def _restored(self, checkpoint: dict[str, Any], recipe: dict[str, Any], store: Any,
                  root: Any = None) -> Run:
        recipes.check_binding(recipe, model_identity=self.adapter.model_identity,
                              numeric_scope=self._numeric_scope())
        tensors = recipes.load_tensors(recipe, store)
        tokens = recipes.load_tokens(recipe, store)
        transitions = checkpoint["transitions"] if root is not None else 0
        run = Run(self.adapter, tokens, lookahead=recipe["lookahead"] if root is None else "none",
                  root=root, root_transitions=transitions)
        run.hold(tensors)
        try:
            run.restore(checkpoint)
        except ReplayMismatch as error:
            raise recipes.fingerprint_mismatch(recipe, error.expected, error.reached) from error
        return run

    # --- occurrence pointers --------------------------------------------------------------

    def export(self, run: Run, target: str, *, keep: bool = False) -> dict[str, Any]:
        """An occurrence pointer to the value at ``target`` (a computed address of ``run``).

        The pointer names the address (origin: this session), the stored recipe of ``run``'s
        cursor, the transition and the boundary cut the value comes from, and the value's
        shape, dtype, payload sha256 and authority (``Run.read_info``). With ``keep`` the value
        is also stored as a blob and indexed, so ``mdb.resolve`` answers without a model.
        """
        project = self._project("export")
        try:
            where = run.address(run.parse(target))
            origin = run.origin_of(target)
            address = parse(where, ports=self.program.names)
        except (ValueError, AddressError) as error:
            raise Refused(f"cannot export {target!r}: {error}") from error
        info = run.read_info(target)
        if info["state"] != "resolved":
            raise Refused(f"the value at {where} is {info['state']}: {info['reason']}")
        value = run.read(target)
        recipe = self._save_recipe(run, None)
        address = replace(address, origin=Origin(branch=self.session_id))
        pointer = occurrence_pointer(
            address, program="stepped", shape=list(value.shape),
            dtype=str(value.dtype).removeprefix("torch."), sha256=tensor_digest(value),
            recipe=recipe["fingerprint"], event=origin["transitions"], cut=origin["cut"],
            authority=info["authority"])
        if keep:
            blob = project.store.put_tensor(value)["ref"]
            try:
                index.index_occurrence(project, self.session_id, pointer["address"],
                                       layers=self.adapter.layers, recipe_id=recipe["fingerprint"],
                                       event_id=origin["transitions"], cut_id=origin["cut"],
                                       blob_id=blob)
            except RecordError as error:
                raise Refused(f"cannot index {where}: {error}") from error
        return pointer

    def live_occurrence(self, checkpoint: dict[str, Any]) -> tuple[Any, str | None]:
        """The resolver's ``live`` tier: the value at ``checkpoint["where"]`` from the live run's
        port cache (nothing is computed), when that run reached it from the same cut with the
        same layer edits. ``Unresolved`` otherwise."""
        run, where = self.live, checkpoint["where"]
        if run is None:
            raise Unresolved("no live run")
        if digest(run.tokens) != checkpoint["prompt_sha256"]:
            raise Unresolved("the live run has another prompt")
        try:
            origin = run.origin_of(where)
        except ValueError as error:
            raise Unresolved(f"the live run has not computed {where}: {error}") from error
        same = (origin["cut"], origin["transitions"], origin["port"]) == (
            checkpoint["cut"], checkpoint["transitions"], checkpoint["port"])
        if not same or (origin["port"] is not None and origin["edits"] != checkpoint["edits"]):
            raise Unresolved("the live run reached this address from another state")
        row, label = where.split(".", 1)
        value = (run.row_captures(int(row[1:])) or {}).get(label)
        if value is None:
            raise Unresolved(f"{where} is not in the live run's port cache")
        return value, run.read_info(where)["authority"]

    def compute_occurrence(
        self, checkpoint: dict[str, Any], recipe: dict[str, Any], store: Any, *, root: Any = None,
    ) -> tuple[Any, str | None]:
        """The resolver's ``cut`` (``root`` a stored boundary cut: one read-only probe) and
        ``replay`` (``root`` None: replay from the token ids) tiers: the value at the cursor of
        ``checkpoint`` (``mdb.core.recipe.occurrence_checkpoint``) and its authority."""
        try:
            run = self._restored(checkpoint, recipe, store, root)
        except Refused:
            raise
        except ValueError as error:   # another execution, an address with no stop, ...
            raise Unresolved(str(error)) from error
        return run.read(), run.read_info()["authority"]

    def ops(self, run: Run | None = None):
        """A context for ``mdb.ops.dispatch`` bound to this model (and ``run``, if given)."""
        return Context(self, run)

    def repl(self, run: Run | None = None, **options: Any) -> int:
        """Interactive gdb-style REPL on this model; returns when the user quits."""
        from mdb.repl.shell import Shell  # sanctioned upward import: Debugger.repl convenience

        return Shell(self.ops(run), **options).loop()


def attach(
    model: Any, tokenizer: Any = None, *, project: Any = None, session: str | None = None
) -> Debugger:
    """Attach to a loaded Hugging Face causal LM (Qwen2/Qwen3, eager attention, eval mode).

    ``project`` (a path, or an ``mdb.core.project.Project``) makes the debugger durable: the
    project is created or reopened and the session ``session`` is opened, bound to the model
    identity and execution. The default session id is the family plus a digest of both, so a
    later process attaching the same model under the same numeric scope reopens it.
    """
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
    if project is None:
        if session is not None:
            raise Refused("session= names a session of a project; pass project= too")
        return Debugger(model, tokenizer, adapter)
    return _durable(model, tokenizer, adapter, project, session)


def _durable(model: Any, tokenizer: Any, adapter: Any, project: Any, session: str | None):
    """The debugger of ``attach(..., project=)``: open the project and the session."""
    from mdb.core.project import Project, ProjectError

    if session is None:
        binding = {"model_identity": adapter.model_identity, "execution": adapter.execution}
        session = f"{program_for(adapter).family}-{digest(binding)[:12]}"
    try:
        Origin(branch=session)   # a session id is an address origin: a branch name
    except AddressError as error:
        raise Refused(f"session id {session!r} is not a branch name: {error}") from error
    try:
        opened = project if isinstance(project, Project) else Project(project)
        opened.open_session(session, model_identity=adapter.model_identity,
                            execution=adapter.execution)
    except (ProjectError, OSError) as error:
        raise Refused(f"cannot open project {project}: {error}") from error
    return Debugger(model, tokenizer, adapter, project=opened, session_id=session)


def open_tiny(family: str = "qwen2", dtype: str = "bf16", *, seed: int = 113) -> Debugger:
    """A random tiny eager CPU model (no tokenizer) for demos and tests."""
    return attach(_tiny(family, dtype, seed))


def _tiny(family: str, dtype: str, seed: int) -> Any:
    from mdb.engine.models import tiny_model

    return tiny_model(family, dtype, seed=seed)
