"""The live session of ``mdb serve --tiny|--model``: one attached debugger driven over HTTP.

Every action is ``mdb.ops.dispatch`` on the debugger's one ``Context`` (the registry the REPL
uses); the console runs a raw command line through the REPL's own ``Shell``. Nothing here
implements a verb.

One model, one thread. The model and the run are not thread-safe, so every job (op, console
line, live view, reduction) runs on ONE worker thread, in arrival order. At most ``QUEUE_LIMIT``
jobs wait behind the running one; a request that would make it ``QUEUE_LIMIT + 1`` is refused
with ``Busy`` (HTTP 409 ``busy``) and nothing runs. ``seq`` numbers ops and console lines in
execution order.

Interrupt. The REPL's Ctrl-C raises ``KeyboardInterrupt`` inside the running command; the engine
rolls a half-done cursor move back (``mdb.engine.run``) and the shell reports ``interrupted at
<where>``. The engine has no polled stop flag, so ``interrupt()`` delivers the same
``KeyboardInterrupt`` to the worker thread (``PyThreadState_SetAsyncExc``), once per job and
only while a job runs; a delivery that has not fired when the job ends is withdrawn.

Events. Each completed op or console line is published to every subscriber (the session's
WebSocket): ``{op, seq, source, where, stopped, state, stale}``; no tensor travels.
"""
from __future__ import annotations

import asyncio
import ctypes
import io
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

QUEUE_LIMIT = 32
SUBSCRIBER_LIMIT = 256
_UNSET = object()
# The views a client re-fetches after an op of each effect.
STALE = {"read": [], "navigate": ["frame", "timeline"], "mutate": ["frame", "timeline", "tables"],
         "native": ["frame", "timeline", "tables"]}
# Local REPL commands the web console refuses: they read or write files on the server, or end it.
CONSOLE_REFUSED = ("source", "record", "quit")


class Busy(Exception):
    """The worker's queue is full: the request was not queued."""


@dataclass
class _Job:
    fn: Any           # () -> raw value; the only part a KeyboardInterrupt can end
    interrupted: Any  # () -> raw value reported when a KeyboardInterrupt ended ``fn``
    finish: Any       # raw value -> the request's value (runs after the job settled)
    future: Future
    interruptible: bool  # an op or console line; never a view or a reduction
    cancelled: bool = False  # an interrupt arrived while it waited: it never runs


class Interrupted(Exception):
    """A read (a live view, a reduction) that an interrupt ended."""


def error_json(error) -> dict[str, Any]:
    """An ``OpError`` as ``python -m mdb.repl --json`` prints it."""
    return {"error": error.to_json(), "state": error.state, "reason": error.message}


def interrupted_json(where: str) -> dict[str, Any]:
    """An interrupted command as ``python -m mdb.repl --json`` prints it."""
    return {"interrupted": where, "state": "unresolved", "reason": f"interrupted at {where}"}


def stopped_by(result: dict[str, Any]) -> str | None:
    """What ended a command: the run-to-hit ``reason`` of its result (breakpoint, watch, budget,
    end-of-context, reached, error, ...), ``interrupted``, ``refused`` for an error, None for a
    command that is not a run-to-hit."""
    if "interrupted" in result:
        return "interrupted"
    if "error" in result:
        return "refused"
    reason = result.get("reason")
    return reason if isinstance(reason, str) else None


class LiveSession:
    """One attached debugger (``mdb.attach(..., project=)``) and the worker that owns it."""

    def __init__(self, debugger: Any, *, queue_limit: int = QUEUE_LIMIT):
        from mdb.repl.shell import Shell

        self.debugger = debugger
        self.sid: str = debugger.session_id
        self.ctx = debugger.ops()
        self.shell = Shell(self.ctx, out=io.StringIO())
        self.queue_limit = queue_limit
        self._jobs: queue.Queue = queue.Queue(maxsize=queue_limit + 1)
        self._lock = threading.Lock()
        self._current: _Job | None = None
        self._waiting: list[_Job] = []  # queued, not started (in order)
        self._sent = False
        self._pending = 0
        self.seq = 0  # ops and console lines completed
        self.running: str | None = None  # name of the op or console line in flight
        self._subscribers: set[tuple[Any, asyncio.Queue]] = set()
        self._thread = threading.Thread(target=self._work, name="mdb-live", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ worker

    def submit(self, fn, interrupted, finish, *, interruptible=True) -> Future:
        """Queue ``fn()`` for the worker; ``Busy`` when ``queue_limit`` jobs already wait."""
        job = _Job(fn, interrupted, finish, Future(), interruptible)
        with self._lock:
            waiting = self._pending - (1 if self._current is not None else 0)
            if waiting >= self.queue_limit:
                raise Busy(f"{waiting} requests are queued (limit {self.queue_limit})")
            self._pending += 1
            self._waiting.append(job)
        self._jobs.put_nowait(job)
        return job.future

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            value = self._attempt(job)
            if not isinstance(value, BaseException):
                try:
                    value = job.finish(value)
                except Exception as error:  # noqa: BLE001 - handed to the waiting request
                    value = error
            with self._lock:
                self._pending -= 1
            if isinstance(value, BaseException):
                job.future.set_exception(value)
            else:
                job.future.set_result(value)

    def _attempt(self, job: _Job):
        """Run one job's ``fn``. A ``KeyboardInterrupt`` that fires inside ``fn`` gives
        ``job.interrupted()``; one that fires after ``fn`` returned keeps its value. A job an
        interrupt cancelled while it waited never runs and also gives ``job.interrupted()``. When
        this returns no interrupt is pending, so ``finish`` cannot be cut short."""
        value = _UNSET
        try:
            with self._lock:
                self._waiting.remove(job)
                if not job.cancelled:
                    self._current, self._sent = job, False
            if not job.cancelled:
                value = job.fn()
            self._settle()
        except KeyboardInterrupt:
            self._settle()
        except Exception as error:  # noqa: BLE001 - handed to the waiting request
            self._settle()
            return error
        if value is _UNSET:
            try:
                return job.interrupted()
            except Exception as error:  # noqa: BLE001
                return error
        return value

    def _settle(self) -> None:
        """No job is current; withdraw an interrupt that was sent but has not fired. The sent
        interrupt may fire inside this very call; it is then spent, and the call repeats."""
        while True:
            try:
                with self._lock:
                    self._current = None
                    if self._sent:
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(self._thread.ident), None)
                        self._sent = False
                return
            except KeyboardInterrupt:
                continue

    def interrupt(self) -> dict[str, Any]:
        """Deliver Ctrl-C to the running op or console line (once) and cancel every queued job
        (each gets the interrupted result: an op or console line ``interrupted at <where>``, a
        view or reduction ``Interrupted``). Says whether one was running and how many queued
        jobs were cancelled. A running view or reduction is never interrupted."""
        with self._lock:
            cancelled = 0
            for job in self._waiting:
                if not job.cancelled:
                    job.cancelled = True
                    cancelled += 1
            if self._current is None or not self._current.interruptible:
                return {"interrupted": False, "running": None, "cancelled": cancelled,
                        "reason": "nothing is running" + (
                            f"; cancelled {cancelled} queued" if cancelled else "")}
            if not self._sent:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self._thread.ident), ctypes.py_object(KeyboardInterrupt))
                self._sent = True
            return {"interrupted": True, "running": self.running, "cancelled": cancelled,
                    "reason": None}

    def close(self) -> None:
        """Stop the worker after the jobs already queued."""
        self._jobs.put(None)
        self._thread.join(timeout=30)

    # ------------------------------------------------------------------ commands

    @property
    def where(self) -> str | None:
        run = self.ctx.run
        try:
            return run.where if run is not None else None
        except Exception:  # noqa: BLE001 - reporting the cursor must never fail
            return None

    def op(self, name: str, args: dict[str, Any]) -> Future:
        """Dispatch one registered op; the future's value is ``{seq, result, where}`` where
        ``result`` is exactly what ``python -m mdb.repl --json`` prints for it."""
        from mdb.ops import OpError, dispatch, registry

        effect = registry()[name].effect

        def run():
            self.running = name
            try:
                return dispatch(self.ctx, name, args)
            except OpError as error:
                return error_json(error)

        def interrupted():
            return interrupted_json(self.where or "no run")

        def finish(result):
            body = self._done(name, effect, "op", stopped_by(result),
                              result.get("state", "resolved"))
            body["result"] = result
            return body

        return self.submit(run, interrupted, finish)

    def console(self, line: str) -> Future:
        """One command line through the REPL shell; the value is ``{seq, text, where}`` with the
        text the REPL prints. ``source``, ``record`` and ``quit`` are refused here."""
        from mdb.ops import OpError, registry

        text = line.strip()
        verb = text.partition(" ")[0]
        try:
            name = self.shell.resolve(verb) if verb else None
        except OpError:
            name = None  # the shell prints the same error itself
        label = name or verb or "console"
        ops = registry()
        effect = ops[name].effect if name in ops else "read"
        before = []

        def run():
            self.running = label
            if name in CONSOLE_REFUSED:
                return (f"error: `{name}` is not available from the web console (it reads or "
                        "writes files on the server, or ends the session)")
            before.append(self.ctx.table.last_stop)
            self.shell.out = io.StringIO()
            self.shell.execute(text)
            return self.shell.out.getvalue().rstrip("\n")

        def interrupted():
            return f"interrupted at {self.where or 'no run'}"

        def finish(out):
            stopped = "interrupted" if out.startswith("interrupted at ") else None
            stop = self.ctx.table.last_stop if before else None
            if stopped is None and stop is not None and stop is not before[0] \
                    and stop["where"] == self.where:
                kinds = {self.ctx.table.items[i].kind for i in stop["ids"]
                         if i in self.ctx.table.items}
                stopped = "watch" if kinds == {"watch"} else "breakpoint"
            body = self._done(label, effect, "console", stopped,
                              "refused" if out.startswith("error") else "resolved")
            body["text"] = out
            return body

        return self.submit(run, interrupted, finish)

    def call(self, fn) -> Future:
        """Run a read (a live view, a reduction) on the worker: no event, no ``seq``; an
        interrupt ends it with ``Interrupted``."""

        def interrupted():
            raise Interrupted("interrupted")

        return self.submit(fn, interrupted, lambda value: value, interruptible=False)

    def _done(self, name, effect, source, stopped, state) -> dict[str, Any]:
        self.seq += 1
        self.running = None
        where = self.where
        self.publish({"op": name, "seq": self.seq, "source": source, "where": where,
                      "stale": STALE.get(effect, STALE["native"]), "stopped": stopped,
                      "state": state})
        return {"seq": self.seq, "where": where}

    # ------------------------------------------------------------------ events

    def subscribe(self) -> asyncio.Queue:
        """A queue that receives every event published from now on (call on the event loop)."""
        events: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_LIMIT)
        with self._lock:
            self._subscribers.add((asyncio.get_running_loop(), events))
        return events

    def unsubscribe(self, events: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = {s for s in self._subscribers if s[1] is not events}

    def publish(self, event: dict[str, Any]) -> None:
        """Hand ``event`` to every subscriber's loop. A subscriber that fell
        ``SUBSCRIBER_LIMIT`` events behind loses the oldest (clients re-fetch on any event)."""
        with self._lock:
            targets = list(self._subscribers)
        for loop, events in targets:
            try:
                loop.call_soon_threadsafe(_offer, events, dict(event))
            except RuntimeError:  # the loop closed: the socket is gone
                self.unsubscribe(events)


def _offer(events: asyncio.Queue, event: dict[str, Any]) -> None:
    if events.full():
        events.get_nowait()
    events.put_nowait(event)


def open_debugger(project, *, tiny: str | None = None, model: str | None = None,
                  dtype: str = "bf16", device: str = "cpu", seed: int = 113):
    """The debugger of ``mdb serve --tiny FAMILY | --model PATH``, attached with ``project`` so
    its checkpoints are durable. ``--model`` reads a local checkpoint only (never downloads);
    its tokenizer is loaded when the directory has one."""
    from pathlib import Path

    from mdb.ops.debugger import attach

    if (tiny is None) == (model is None):
        raise ValueError("give one of tiny= or model=")
    if tiny is not None:
        from mdb.engine.models import tiny_model

        return attach(tiny_model(tiny, dtype, seed=seed), None, project=project)
    root = Path(model).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise ValueError(f"--model needs a Hugging Face checkpoint directory ({root} has no "
                         "config.json)")
    from mdb.engine.models import load_local

    loaded = load_local(root, dtype=dtype, device=device)
    tokenizer = None
    if any((root / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    return attach(loaded, tokenizer, project=project)
