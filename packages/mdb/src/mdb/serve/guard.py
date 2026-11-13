"""The outermost ASGI layer of ``mdb serve``: who may talk to the server, how much they may
send, and the security headers on every response (including 4xx and the catch-all 500).

Pure ASGI (no framework import), wrapped *around* the FastAPI app so that responses produced by
Starlette's own error middleware (the 500 path) pass through it too.

* Host guard (``http`` and ``websocket``): the ``Host`` header must be one of the loopback names
  with the bound port (``127.0.0.1:P``, ``localhost:P``, ``[::1]:P``; the bare forms when ``P``
  is 80) or an explicit ``--allow-host``; an ``Origin`` header, when present, must be
  ``http://`` plus one of those hosts. This is what stops DNS rebinding and cross-site requests
  from a browser: a foreign page can reach loopback, but it cannot make the browser send a
  loopback ``Host``/``Origin``. Refusals are 403 (``host_not_allowed``, ``origin_not_allowed``);
  a missing ``Host`` is 400; a WebSocket is closed before it is accepted.
* Body limit: request bodies above ``body_limit`` bytes are refused 413 ``over_limit``. Streamed
  bytes are counted (``Content-Length`` is not trusted) and nothing past the limit is buffered.
* Headers: CSP, ``X-Content-Type-Options``, ``Referrer-Policy``, ``Cross-Origin-Resource-Policy``
  and ``Cache-Control`` on every HTTP response. An exception that escapes the app becomes a 500
  ``internal_error`` that names no exception text, path or traceback (those go to the log).
"""
from __future__ import annotations

import logging

from .caps import canonical, refused

LOG = logging.getLogger("mdb.serve")
BODY_LIMIT = 256 * 1024
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
       "frame-ancestors 'none'")
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")


def _split(entry):
    """``(name, port or None)`` of ``HOST``, ``HOST:PORT``, ``[v6]`` or ``[v6]:PORT``."""
    entry = entry.strip().lower()
    if entry.startswith("["):
        name, _, rest = entry.partition("]")
        return name + "]", (rest[1:] if rest.startswith(":") else None) or None
    name, _, port = entry.partition(":")
    return name, port or None


def allowed_hosts(port, extra=()):
    """The accepted ``Host`` header values (lower case) for a bound ``port``.

    ``extra`` entries are ``HOST`` (taken as ``HOST:port``) or ``HOST:PORT`` (exact). The bare
    form of a name is accepted only when the port is 80. ``port`` None (embedding and tests;
    the CLI always passes the real port) leaves the loopback names to ``HostMatcher.ok``."""
    hosts = set()
    for entry in (*LOOPBACK_NAMES, *extra):
        name, given = _split(entry)
        if not name:
            continue
        if given is not None:
            hosts.add(f"{name}:{given}")
        elif port is not None:
            hosts.add(f"{name}:{port}")
            if port == 80:
                hosts.add(name)
    return hosts


class HostMatcher:
    def __init__(self, port, extra=()):
        self.port, self.exact = port, allowed_hosts(port, extra)

    def ok(self, host: str) -> bool:
        host = host.strip().lower()
        if host in self.exact:
            return True
        if self.port is None:
            name, given = _split(host)
            return name in LOOPBACK_NAMES and given is not None and given.isdigit()
        return False


_DUPLICATE = object()


def _header(scope, name: bytes):
    """The header's value, None when absent, ``_DUPLICATE`` when it occurs more than once."""
    found = [value for key, value in scope.get("headers") or () if key == name]
    if len(found) > 1:
        return _DUPLICATE
    return found[0].decode("latin-1") if found else None


class _BodyTooLarge(Exception):
    pass


class SecurityLayer:
    def __init__(self, app, *, port=None, allow_hosts=(), body_limit=BODY_LIMIT, csp=CSP):
        self.app, self.hosts, self.body_limit, self.csp = app, HostMatcher(port, allow_hosts), \
            body_limit, csp
        self.state = getattr(app, "state", None)

    # -- helpers
    def _security_headers(self, path):
        return [(b"content-security-policy", self.csp.encode()),
                (b"x-content-type-options", b"nosniff"),
                (b"referrer-policy", b"no-referrer"),
                (b"cross-origin-resource-policy", b"same-origin"),
                (b"cache-control", b"no-store" if path.startswith("/api/") else b"no-cache")]

    async def _respond(self, scope, send, status, reason, detail, **extra):
        body = canonical(refused(reason, detail, **extra))
        headers = [(b"content-type", b"application/json"),
                   (b"content-length", str(len(body)).encode()),
                   *self._security_headers(scope.get("path", ""))]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    def _check(self, scope):
        """``(status, reason, detail)`` when the request must be refused, else None."""
        host = _header(scope, b"host")
        if host is _DUPLICATE or _header(scope, b"origin") is _DUPLICATE:
            return 403, "duplicate_header", "a Host or Origin header occurs more than once"
        if host is None or not host.strip():
            return 400, "missing_host", "the request has no Host header"
        if not self.hosts.ok(host):
            return 403, "host_not_allowed", "this server answers only to its own loopback " \
                "address (or an --allow-host); a foreign Host is refused"
        origin = _header(scope, b"origin")
        if origin is not None:
            origin = origin.strip().lower()
            if not origin.startswith("http://") or not self.hosts.ok(origin[len("http://"):]):
                return 403, "origin_not_allowed", "the Origin is not this server"
        return None

    # -- ASGI
    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        verdict = self._check(scope)
        if verdict is not None:
            if kind == "websocket":
                await receive()  # websocket.connect
                await send({"type": "websocket.close", "code": 1008})
            else:
                await self._respond(scope, send, *verdict)
            return
        if kind == "websocket":
            await self.app(scope, receive, send)
            return
        await self._http(scope, receive, send)

    async def _http(self, scope, receive, send):
        state = {"started": False, "over": False}
        limit = self.body_limit

        async def refuse_over(received):
            state["over"] = True
            if not state["started"]:
                state["started"] = True
                await self._respond(scope, send, 413, "over_limit",
                                    f"request body is over the {limit} byte limit"
                                    + ("" if received is None else f" (saw {received} bytes)"),
                                    cap=limit)

        declared = _header(scope, b"content-length")
        if declared is not None and declared.strip().isdigit() and int(declared) > limit:
            await refuse_over(None)
            return
        seen = 0

        async def guarded_receive():
            nonlocal seen
            if state["over"]:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    await refuse_over(seen)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if state["over"]:
                return
            if message["type"] == "http.response.start":
                state["started"] = True
                skip = {b"content-security-policy", b"x-content-type-options", b"referrer-policy",
                        b"cross-origin-resource-policy", b"cache-control"}
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in skip]
                message = {**message, "headers": headers
                           + self._security_headers(scope.get("path", ""))}
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except Exception:  # noqa: BLE001 - nothing may escape to the client as a traceback
            if state["over"]:
                return
            LOG.exception("unhandled error serving %s %s", scope.get("method"),
                          scope.get("path"))
            if not state["started"]:
                state["started"] = True
                await self._respond(scope, send, 500, "internal_error",
                                    "the server failed on this request; see its log")
