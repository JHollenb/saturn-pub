"""A client of a live ``mdb serve`` session: gdb's ``target remote`` for models.

The model and the debugger live in the server process (``mdb serve --project DIR --model PATH``
on the machine with the weights); this module drives them over the server's HTTP API and needs
only the standard library, so the machine that types the commands needs no model framework.

    remote = Remote("http://127.0.0.1:8765")
    remote.console("until t3.L1.p")          # the text the REPL prints
    remote.op("print", address="p", top=3)    # the JSON `python -m mdb.repl --json` prints
    RemoteShell(remote).loop()                 # the same REPL prompt, served remotely

    mdb engine --connect http://127.0.0.1:8765 [--script FILE]

Every command runs through the server's op registry, so a remote command and a local one are
the same operation. Three things differ from the local REPL and are handled here:
``quit`` ends the client and leaves the server session as it is; ``source FILE`` reads the
file on the client and sends its lines; Ctrl-C asks the server to interrupt the running op and
then keeps waiting for that op's own answer. ``record`` is refused by the server. An empty line
repeats nothing (the local REPL repeats the last motion).

The server has no authentication: it accepts loopback by default, so reach a remote machine
through a tunnel (``ssh -L 8765:127.0.0.1:8765 host``) rather than by binding it to a network.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable, TextIO

Transport = Callable[[str, str, "dict[str, Any] | None"], "tuple[int, Any]"]
QUIT = ("quit", "q", "exit")
MAX_SOURCE_DEPTH = 16
API = "/api/v1"


class RemoteError(Exception):
    """The server refused a request before any op ran (``busy``, ``unknown-op``, ``not_live``,
    ``host_not_allowed``, ...) or could not be reached (``unreachable``, ``bad-response``)."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(f"{kind}: {message}")
        self.kind, self.message, self.status = kind, message, status


def http_transport(base: str, timeout: float) -> Transport:
    """``send(method, path, body) -> (status, json)`` over urllib; no proxy, no redirect."""

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def send(method: str, path: str, body: dict[str, Any] | None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with opener.open(request, timeout=timeout) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise RemoteError("unreachable", f"cannot reach {base}: {reason}") from error
        try:
            return status, json.loads(raw)
        except ValueError as error:
            raise RemoteError("bad-response", f"{base}{path} answered {status} with a body "
                              "that is not JSON (is this an mdb serve address?)",
                              status) from error

    return send


def _base(url: str) -> str:
    parts = urllib.parse.urlsplit(url if "://" in url else f"http://{url}")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise RemoteError("bad-url", f"{url!r} is not an http(s) address of an mdb serve")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise RemoteError("bad-url", f"{url!r}: give the server address only, as "
                          f"{parts.scheme}://{parts.netloc}")
    return f"{parts.scheme}://{parts.netloc}"


class Remote:
    """One live session of an ``mdb serve``. ``where`` and ``seq`` follow the last answer."""

    def __init__(self, url: str, *, timeout: float = 3600.0,
                 transport: Transport | None = None):
        self.url = _base(url)
        self._send = transport or http_transport(self.url, timeout)
        self._requests = 0
        hello = self._data("GET", "/hello")
        if not isinstance(hello, dict) or "version" not in hello:
            raise RemoteError("bad-response", f"{self.url} is not an mdb serve")
        self.version: str = str(hello["version"])
        self.sid = hello.get("live")
        if not self.sid:
            raise RemoteError("not_live", f"{self.url} holds no live session (it was started "
                              "without --tiny or --model); there is nothing to step")
        self.where: str | None = None
        self.seq = 0

    # ------------------------------------------------------------------ wire

    def _data(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        status, answer = self._send(method, API + path, body)
        if isinstance(answer, dict) and "data" in answer:
            return answer["data"]
        if isinstance(answer, dict) and "refused" in answer:
            raise RemoteError(str(answer["refused"]), str(answer.get("detail") or ""), status)
        raise RemoteError("bad-response", f"{method} {path} answered {status} without data",
                          status)

    def _session(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = self._data(method, f"/s/{self.sid}{path}", body)
        if isinstance(data, dict):
            if "where" in data:
                self.where = data["where"]
            if isinstance(data.get("seq"), int):
                self.seq = data["seq"]
        return data

    # ------------------------------------------------------------------ commands

    def op(self, name: str, args: dict[str, Any] | None = None, **more: Any) -> dict[str, Any]:
        """One registered op. The value is the op's own JSON, exactly what ``python -m mdb.repl
        --json`` prints for the command: a result, ``{error, state, reason}`` when the op
        refused, or ``{interrupted, state, reason}``. ``RemoteError`` is raised only when the
        server refused the request itself."""
        self._requests += 1
        body = {"op": name, "args": {**(args or {}), **more}, "request_id": f"c{self._requests}"}
        return self._session("POST", "/op", body)["result"]

    def console(self, line: str) -> str:
        """One command line through the server's REPL parser; the text the REPL prints."""
        return self._session("POST", "/console", {"line": line})["text"]

    def interrupt(self) -> dict[str, Any]:
        """Ctrl-C for the op the server is running; says whether one was running."""
        return self._session("POST", "/interrupt")

    def frame(self) -> dict[str, Any]:
        """The live frame: cursor, the current layer's ports, edits, breakpoints, checkpoints."""
        return self._session("GET", "/live")


class RemoteShell:
    """The REPL prompt over a ``Remote``: same lines in, same text out."""

    def __init__(self, remote: Remote, *, out: TextIO | None = None, prompt: str = "(mdb) ",
                 poll: float = 0.1):
        self.remote = remote
        self.out = out if out is not None else sys.stdout
        self.prompt = prompt
        self.poll = poll
        self.errors = 0
        self.depth = 0
        self._failed = False
        self._interrupted = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mdb-remote")

    def write(self, text: str) -> None:
        self.out.write(text + "\n")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _console(self, line: str) -> str:
        """Send one line; Ctrl-C while it runs interrupts the server's op, and the wait goes on
        until that op answers (``interrupted at WHERE``). A second Ctrl-C asks again."""
        future = self._pool.submit(self.remote.console, line)
        while True:
            try:
                return future.result(timeout=self.poll)
            except FutureTimeout:
                continue
            except KeyboardInterrupt:
                self._interrupted = True
                try:
                    self.remote.interrupt()
                except RemoteError as error:
                    self.write(f"error ({error.kind}): interrupt was not delivered: "
                               f"{error.message}")

    def execute(self, line: str) -> bool:
        """Run one command line; False means quit. Server refusals are printed, not raised."""
        text = line.strip()
        if self.depth == 0:
            self._interrupted = False
        if not text or text.startswith("#"):
            return True
        verb, _, rest = text.partition(" ")
        if verb in QUIT:
            return False
        self._failed = False
        if verb == "source":
            self._source_line(rest.strip())
            return True
        try:
            out = self._console(text)
        except RemoteError as error:
            out = f"error ({error.kind}): {error.message}"
        if out.startswith("error") or out.startswith("interrupted at "):
            self.errors += 1
            self._failed = True
        if out:
            self.write(out)
        return True

    def _source_line(self, rest: str) -> None:
        words = rest.split()
        keep_going = bool(words) and words[0] == "-k"
        names = words[1:] if keep_going else words
        if len(names) != 1:
            self.errors += 1
            self._failed = True
            self.write("error (bad-args): usage: source [-k] FILE")
            return
        try:
            self.source(names[0], keep_going=keep_going)
        except OSError as error:
            self.errors += 1
            self._failed = True
            self.write(f"error (bad-args): cannot read {names[0]}: {error.strerror or error}")

    def source(self, path: str, *, echo: bool = True, keep_going: bool = False) -> int:
        """Send a local command file line by line. It stops at the first error unless
        ``keep_going`` (``source -k``). Returns the number of errors."""
        if self.depth >= MAX_SOURCE_DEPTH:
            self.errors += 1
            self._failed = True
            self.write(f"error (bad-args): source nests more than {MAX_SOURCE_DEPTH} deep: {path}")
            return 1
        before = self.errors
        lines = Path(path).expanduser().read_text().splitlines()
        self.depth += 1
        try:
            for number, raw in enumerate(lines, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if echo:
                    self.write(f"{self.prompt}{line}")
                self._failed = False
                if not self.execute(line) or self._interrupted:
                    break
                if self._failed and not keep_going:
                    if self.depth == 1:
                        self.write(f"error: stopped sourcing {path} at line {number} "
                                   "(use `source -k FILE` to keep going)")
                    break
        finally:
            self.depth -= 1
            if keep_going:
                self._failed = False
        return self.errors - before

    def loop(self, stream: TextIO | None = None) -> int:
        stream = stream if stream is not None else sys.stdin
        interactive = stream is sys.stdin and stream.isatty()
        if interactive:
            try:
                import readline  # noqa: F401 - line editing and history for input()
            except ImportError:
                pass
        try:
            while True:
                try:
                    if interactive:
                        line = input(self.prompt)
                    else:
                        line = stream.readline()
                        if not line:
                            break
                        line = line.rstrip("\n")
                        if line.strip() and not line.strip().startswith("#"):
                            self.write(f"{self.prompt}{line.strip()}")
                except EOFError:
                    break
                except KeyboardInterrupt:
                    self.write("")
                    continue
                if not self.execute(line):
                    break
        finally:
            self.close()
        return 0 if self.errors == 0 else 1


def main(url: str, *, start: str | None = None, script: str | None = None,
         out: TextIO | None = None) -> int:
    """``mdb engine --connect URL``: attach to the server's live session and run the prompt."""
    try:
        remote = Remote(url)
    except RemoteError as error:
        print(f"mdb engine: {error.message} ({error.kind})", file=sys.stderr)
        return 2
    shell = RemoteShell(remote, out=out)
    try:
        if start is not None:
            shell.write(f"{shell.prompt}{start}")
            shell.execute(start)
        if script:
            try:
                shell.source(script)
            except OSError as error:
                print(f"mdb engine: cannot read {script}: {error.strerror or error}",
                      file=sys.stderr)
                return 2
            return 0 if shell.errors == 0 else 1
        return shell.loop()
    finally:
        shell.close()
