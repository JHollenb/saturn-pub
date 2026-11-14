"""``python -m mdb.serve --project PATH [--tiny FAMILY | --model DIR] [--port N]
[--host 127.0.0.1] [--allow-remote --allow-host HOST[:PORT] ...]``.

Without ``--tiny``/``--model`` the server is a read-only viewer of the recorded project. With
one of them it also holds a live debugger attached with ``--project`` (``mdb serve --project
runs/t --tiny qwen2``): the web app steps it through the op registry, and checkpoints made
there are stored in the project. Live mode executes model code and accepts edits.

Binds loopback by default. There is no authentication, which is acceptable only on loopback:
a non-loopback host is refused unless ``--allow-remote`` is given, and then a warning is printed.
Every request must carry a ``Host`` (and ``Origin``, when sent) naming this server: the loopback
names with the bound port, or an explicit ``--allow-host``. ``--allow-remote`` requires at least
one ``--allow-host``, because without it no remote client's ``Host`` would be accepted.
"""
from __future__ import annotations

import argparse
import ipaddress
import sys
from pathlib import Path

# ``mdb.engine.models.TINY_FAMILIES``, spelled out so the read-only viewer imports without torch
# (a test checks the two agree).
TINY_FAMILIES = ("qwen2", "qwen3", "llama", "mistral")


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb serve", description=__doc__.splitlines()[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-remote", action="store_true",
                        help="bind a non-loopback host (no authentication!)")
    parser.add_argument("--allow-host", action="append", default=[], metavar="HOST[:PORT]",
                        help="also accept this Host header (repeatable); HOST alone means "
                        "HOST:<port>. Required with --allow-remote")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tiny", choices=TINY_FAMILIES,
                        help="live mode on a random tiny model (tests, demos)")
    source.add_argument("--model", metavar="DIR",
                        help="live mode on a local Hugging Face checkpoint (never downloads)")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=113, help="tiny model seed")
    parser.add_argument("--heartbeat", type=float, default=10.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    live_mode = bool(args.tiny or args.model)
    if not 1 <= args.port <= 65535:
        print("mdb serve: --port must be 1..65535 (the Host guard needs the bound port)",
              file=sys.stderr)
        return 2
    if args.allow_remote and not args.allow_host:
        print("mdb serve: --allow-remote needs --allow-host HOST[:PORT] naming how clients will "
              "reach this server (a request whose Host is not listed is refused)",
              file=sys.stderr)
        return 2
    if not is_loopback(args.host):
        if not args.allow_remote:
            print(f"mdb serve: refusing non-loopback host {args.host!r}: the server has no "
                  "authentication. Pass --allow-remote to bind it anyway.", file=sys.stderr)
            return 2
        print(f"mdb serve: WARNING: serving {args.project} on {args.host} without "
              "authentication; anyone who can reach this port can read the project.",
              file=sys.stderr)
        if live_mode:
            print("mdb serve: WARNING: LIVE MODE on a non-loopback host: anyone who can reach "
                  "this port can run the model, stage edits and write checkpoints into "
                  f"{args.project}.", file=sys.stderr)
    try:
        import uvicorn
    except ImportError:
        print("mdb serve: needs the serve extra (pip install 'mdb-runtime[serve]')",
              file=sys.stderr)
        return 2
    from .app import create_app

    live = None
    if live_mode:
        from ..core.errors import OpError
        from .live import LiveSession, open_debugger

        try:
            debugger = open_debugger(args.project, tiny=args.tiny, model=args.model,
                                     dtype=args.dtype, device=args.device, seed=args.seed)
        except (OpError, ValueError, OSError) as error:
            print(f"mdb serve: cannot attach the live model: "
                  f"{getattr(error, 'message', error)}", file=sys.stderr)
            return 1
        live = LiveSession(debugger)
    try:
        app = create_app(args.project, heartbeat=args.heartbeat, port=args.port,
                         allow_hosts=args.allow_host, live=live)
    except (ValueError, OSError) as error:
        print(f"mdb serve: {error}", file=sys.stderr)
        return 1
    from .app import STATIC

    if not (STATIC / "index.html").is_file():
        print(f"mdb serve: web build absent at {STATIC}; run `npm run build` in "
              "packages/mdb/web (the API at /api/v1 is served without it)", file=sys.stderr)
    mode = f"live session {live.sid}" if live else "read-only"
    print(f"mdb serve: http://{args.host}:{args.port}/  (project {args.project}; {mode})",
          file=sys.stderr)
    try:  # one SIGTERM/SIGINT ends it: app.py's lifespan closes the event sockets first
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                    timeout_graceful_shutdown=5)
    finally:
        if live is not None:
            live.interrupt()
            live.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
