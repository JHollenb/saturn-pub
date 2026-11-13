"""``mdb.repl.remote``: the REPL over a live ``mdb serve`` prints what the local REPL prints.

The gate: one command script, run (a) by a local ``Shell`` on a tiny model and (b) by a
``RemoteShell`` against a server holding the same tiny model, gives the same text. Then the
typed op route, refusals, client-side ``source``, Ctrl-C, and one pass over a real socket.
"""
from __future__ import annotations

import io
import socket
import threading
import time

import pytest

from mdb.repl.remote import Remote, RemoteError, RemoteShell, http_transport
from mdb.repl.remote import main as remote_main

SCRIPT = [
    "start 1 5 9 3",
    "until t3.L1.p",
    "print p top=3",
    "backtrace",
    "break L*.o",
    "continue",
    "info breakpoints",
    "delete",
    "set L1.o = 0.5 add",
    "info edits",
    "abort",
    "whatif L1.m = 0.5 add --consumer M[4,9]",
    "print nosuchport",
    "finish",
    "explain L1.o",
    "reverse-step",
    "where",
]


# --------------------------------------------------------------------------- no server needed

def test_importing_the_client_loads_no_framework():
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys; import mdb.repl.remote; "
            "bad = [m for m in ('torch', 'numpy', 'fastapi', 'transformers') if m in sys.modules]; "
            "sys.exit(repr(bad) if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("url", ["ftp://127.0.0.1:1", "http://127.0.0.1:1/api/v1", "http://",
                                 "http://127.0.0.1:1/?x=1"])
def test_bad_urls_are_refused_before_any_request(url):
    def never(*_):
        raise AssertionError("no request may be sent")

    with pytest.raises(RemoteError) as caught:
        Remote(url, transport=never)
    assert caught.value.kind == "bad-url"


def test_unreachable_server_is_a_remote_error(capsys):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(RemoteError) as caught:
        Remote(f"http://127.0.0.1:{port}", timeout=5)
    assert caught.value.kind == "unreachable"
    assert remote_main(f"http://127.0.0.1:{port}") == 2
    assert "Traceback" not in capsys.readouterr().err


def _fake(answers):
    """A transport over a dict ``(method, path) -> callable(body) -> (status, json)``."""
    def send(method, path, body):
        return answers[(method, path)](body)
    return send


def _hello(live="s1"):
    return lambda body: (200, {"data": {"version": "0.1.0", "live": live}})


def test_a_recorded_only_server_is_refused():
    with pytest.raises(RemoteError) as caught:
        Remote("http://127.0.0.1:9", transport=_fake({("GET", "/api/v1/hello"): _hello(None)}))
    assert caught.value.kind == "not_live"


def test_not_an_mdb_server():
    send = _fake({("GET", "/api/v1/hello"): lambda body: (200, {"data": {"hi": 1}})})
    with pytest.raises(RemoteError) as caught:
        Remote("http://127.0.0.1:9", transport=send)
    assert caught.value.kind == "bad-response"


def test_ctrl_c_interrupts_the_server_and_waits_for_its_answer():
    """Ctrl-C during a running line posts /interrupt, then the line's own answer is printed."""
    release = threading.Event()
    calls = []

    def console(body):
        calls.append(("console", body["line"]))
        assert release.wait(20)
        return 200, {"data": {"line": body["line"], "seq": 1, "where": "t3.L0.q",
                              "text": "interrupted at t3.L0.q"}}

    def interrupt(body):
        calls.append(("interrupt", None))
        release.set()
        return 200, {"data": {"interrupted": True, "running": "continue", "reason": None}}

    remote = Remote("http://127.0.0.1:9", transport=_fake({
        ("GET", "/api/v1/hello"): _hello(),
        ("POST", "/api/v1/s/s1/console"): console,
        ("POST", "/api/v1/s/s1/interrupt"): interrupt}))
    out = io.StringIO()
    shell = RemoteShell(remote, out=out, poll=0.05)
    polls = {"n": 0}
    real = shell._pool.submit

    def submit(fn, *args):
        future = real(fn, *args)
        wait = future.result

        def result(timeout=None):
            polls["n"] += 1
            if polls["n"] == 3:  # the user presses Ctrl-C while the line is running
                raise KeyboardInterrupt
            return wait(timeout=timeout)

        future.result = result
        return future

    shell._pool.submit = submit
    try:
        assert shell.execute("continue") is True
    finally:
        shell.close()
    assert calls == [("console", "continue"), ("interrupt", None)]
    assert out.getvalue() == "interrupted at t3.L0.q\n"
    assert shell.errors == 1 and remote.where == "t3.L0.q"


def test_quit_is_local_and_source_reads_the_client_file(tmp_path):
    sent = []

    def console(body):
        sent.append(body["line"])
        text = "error (bad-args): no" if body["line"] == "bad" else f"ok {body['line']}"
        return 200, {"data": {"line": body["line"], "seq": len(sent), "where": None,
                              "text": text}}

    remote = Remote("http://127.0.0.1:9", transport=_fake({
        ("GET", "/api/v1/hello"): _hello(), ("POST", "/api/v1/s/s1/console"): console}))
    script = tmp_path / "cmds.mdb"
    script.write_text("# comment\nwhere\nbad\nstep\n")
    out = io.StringIO()
    shell = RemoteShell(remote, out=out)
    try:
        assert shell.execute(f"source {script}") is True
        assert sent == ["where", "bad"]  # stops at the first error, as the local REPL
        assert "stopped sourcing" in out.getvalue()
        assert shell.execute(f"source -k {script}") is True
        assert sent == ["where", "bad", "where", "bad", "step"]
        assert shell.execute("source") is True and "usage: source" in out.getvalue()
        assert shell.execute(f"source {tmp_path / 'absent'}") is True
        assert "cannot read" in out.getvalue()
        for word in ("quit", "q", "exit"):
            assert shell.execute(word) is False
        assert "quit" not in sent and "q" not in sent
    finally:
        shell.close()


def test_a_refused_request_is_printed_not_raised():
    remote = Remote("http://127.0.0.1:9", transport=_fake({
        ("GET", "/api/v1/hello"): _hello(),
        ("POST", "/api/v1/s/s1/console"):
            lambda body: (409, {"refused": "busy", "state": "refused", "detail": "queue full"})}))
    out = io.StringIO()
    shell = RemoteShell(remote, out=out)
    try:
        assert shell.execute("step") is True
    finally:
        shell.close()
    assert out.getvalue() == "error (busy): queue full\n" and shell.errors == 1


# --------------------------------------------------------------------------- a real session

@pytest.fixture()
def served(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from mdb.serve.app import create_app
    from mdb.serve.live import LiveSession, open_debugger

    live = LiveSession(open_debugger(tmp_path / "served", tiny="qwen2"))
    app = create_app(tmp_path / "served", live=live)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        def send(method, path, body):
            answer = client.request(method, path, json=body) if body is not None \
                else client.request(method, path)
            return answer.status_code, answer.json()

        yield {"send": send, "app_factory": create_app, "live": live, "tmp": tmp_path}
    live.close()


def _local_text(tmp_path, lines):
    from mdb.repl.shell import Shell
    from mdb.serve.live import open_debugger

    out = io.StringIO()
    shell = Shell(open_debugger(tmp_path / "local", tiny="qwen2").ops(), out=out)
    for line in lines:
        shell.write(f"{shell.prompt}{line}")
        shell.execute(line)
    return out.getvalue(), shell.errors


def test_remote_repl_prints_what_the_local_repl_prints(served, tmp_path):
    remote = Remote("http://127.0.0.1:8765", transport=served["send"])
    out = io.StringIO()
    shell = RemoteShell(remote, out=out)
    try:
        for line in SCRIPT:
            shell.write(f"{shell.prompt}{line}")
            assert shell.execute(line) is True
    finally:
        shell.close()
    text, errors = _local_text(tmp_path, SCRIPT)
    assert out.getvalue() == text
    assert shell.errors == errors >= 1  # `print nosuchport` at least
    assert "error" in text and "[measured]" in text and "[derived-exact" in text
    assert remote.where is not None and remote.seq == len(SCRIPT)


def test_typed_ops_and_request_refusals(served):
    remote = Remote("http://127.0.0.1:8765", transport=served["send"])
    assert remote.op("start", prompt=[1, 5, 9, 3])["where"] == remote.where
    stopped = remote.op("until", {"target": "t3.L1.p"})
    assert remote.where == "t3.L1.p" and stopped.get("state", "resolved") == "resolved"
    printed = remote.op("print", address="p", top=3)
    assert "pointer" in str(printed) and "authority" in str(printed)
    refused = remote.op("print", address="nosuchport")  # the op's own refusal: a value
    assert refused["state"] == "refused" and "error" in refused
    with pytest.raises(RemoteError) as caught:  # the request's refusal: an exception
        remote.op("nosuchop")
    assert caught.value.kind == "unknown-op" and caught.value.status == 404
    assert remote.interrupt()["interrupted"] is False
    frame = remote.frame()
    assert isinstance(frame, dict) and frame


def test_over_a_real_socket(served, tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = served["app_factory"](tmp_path / "served", port=port, live=served["live"])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        url = f"http://127.0.0.1:{port}"
        script = tmp_path / "walk.mdb"
        script.write_text("\n".join(SCRIPT[:4]) + "\n")
        out = io.StringIO()
        assert remote_main(url, script=str(script), out=out) == 0
        assert out.getvalue() == _local_text(tmp_path, SCRIPT[:4])[0]
        # The Host guard still applies to this client: a name the server does not accept.
        with pytest.raises(RemoteError) as caught:
            Remote(f"http://localhost.:{port}", transport=_wrong_host(url, port))
        assert caught.value.kind == "host_not_allowed"
    finally:
        server.should_exit = True
        thread.join(timeout=30)
    assert not thread.is_alive()


def _wrong_host(url, port):
    import json
    import urllib.error
    import urllib.request

    def send(method, path, body):
        request = urllib.request.Request(url + path, method=method,
                                         headers={"Host": f"evil.example:{port}"})
        try:
            with urllib.request.urlopen(request, timeout=10) as answer:
                return answer.status, json.loads(answer.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    return send


def test_http_transport_reports_non_json(tmp_path):
    """A port that answers HTTP but is not an mdb serve."""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html>hi</html>")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        send = http_transport(f"http://127.0.0.1:{server.server_port}", 10)
        with pytest.raises(RemoteError) as caught:
            send("GET", "/api/v1/hello", None)
        assert caught.value.kind == "bad-response"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    assert not thread.is_alive()
