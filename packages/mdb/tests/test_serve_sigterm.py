"""One SIGTERM ends ``mdb serve`` (M39): also with an events websocket open, which uvicorn's
graceful shutdown would otherwise wait on for ever."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
websockets_sync = pytest.importorskip("websockets.sync.client")

SRC = Path(__file__).resolve().parents[1] / "src"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(tmp_path):
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(SRC), str(SRC.parents[2] / "src"), os.environ.get("PYTHONPATH", "")]),
        "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "mdb", "serve", "--project", str(tmp_path), "--tiny", "qwen2",
         "--port", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 90
        while True:
            if proc.poll() is not None:
                raise AssertionError(f"server died: {proc.stderr.read().decode()[-800:]}")
            try:
                sessions = json.load(urllib.request.urlopen(f"{url}/api/v1/sessions", timeout=2))
                break
            except Exception:
                assert time.time() < deadline, "server did not come up"
                time.sleep(0.2)
    except BaseException:
        proc.kill()
        raise
    return proc, port, json.dumps(sessions)


def _sid(text):
    import re

    return re.search(r'"(?:id|sid)": ?"([^"]+)"', text).group(1)


@pytest.mark.parametrize("socket_open", [False, True], ids=["no-socket", "events-socket-open"])
def test_one_sigterm_ends_the_server(tmp_path, socket_open):
    proc, port, sessions = _serve(tmp_path)
    ws = None
    try:
        if socket_open:
            ws = websockets_sync.connect(
                f"ws://127.0.0.1:{port}/api/v1/s/{_sid(sessions)}/events",
                additional_headers={"Origin": f"http://127.0.0.1:{port}"})
            assert json.loads(ws.recv(timeout=10))["data"]["type"] == "hello"
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pytest.fail("the server was still alive 3 s after one SIGTERM")
        assert proc.returncode in (0, -signal.SIGTERM)
    finally:
        if ws is not None:
            ws.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stderr.close()
