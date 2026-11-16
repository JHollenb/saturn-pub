"""``python -m mdb.serve --project PATH [--tiny FAMILY | --model PATH|NAME] [--port N]
[--host 127.0.0.1] [--allow-remote --allow-host HOST[:PORT] ...]``.

``--model`` takes the same forms as ``mdb engine MODEL`` (a directory, a cached name, an alias
from the config, ``tiny:FAMILY``). Defaults (project, host, port, allowed hosts, dtype, device)
may come from a config file or ``MDB_*`` variables (``mdb.core.settings``); a flag wins.
``--allow-remote`` is never read from a config file, ``serve.allow_hosts`` never from a project
file, and only ``--allow-host`` typed on the command line satisfies ``--allow-remote``.

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

from ..core.settings import TINY_FAMILIES  # noqa: F401 - the one list (core.settings)

# setting name -> the flag that sets it (``mdb config explain`` names the flag)
FLAGS = {"project.dir": "--project", "project.session": "--session", "serve.host": "--host",
         "serve.port": "--port", "serve.allow_hosts": "--allow-host", "model.dtype": "--dtype",
         "model.device": "--device"}


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv=None):
    from ..core import settings
    from ..core.errors import BadArgs
    from ..core.locate import parse_target

    parser = argparse.ArgumentParser(
        prog="mdb serve",
        description="Web viewer of a recorded project. With --tiny or --model it also holds a "
                    "live debugger that the web app steps (live mode executes model code and "
                    "accepts edits). Binds loopback only unless --allow-remote is given; "
                    "there is no authentication.",
        epilog="Defaults come from mdb.toml / ~/.config/mdb/config.toml and MDB_* variables "
               "(mdb config show; docs/CONFIG.md).")
    parser.add_argument("--project", type=Path,
                        help="the project to serve (required unless [project] dir is set)")
    parser.add_argument("--session", metavar="NAME",
                        help="live session id in the project (default: family + model digest)")
    parser.add_argument("--host", help=f"bind host (default {settings.DEFAULT_HOST})")
    parser.add_argument("--port", type=int, help=f"port (default {settings.DEFAULT_PORT})")
    parser.add_argument("--allow-remote", action="store_true",
                        help="bind a non-loopback host (no authentication!)")
    parser.add_argument("--allow-host", action="append", default=None, metavar="HOST[:PORT]",
                        help="also accept this Host header (repeatable); HOST alone means "
                        "HOST:<port>. Required with --allow-remote")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tiny", choices=TINY_FAMILIES,
                        help="live mode on a random tiny model (tests, demos)")
    source.add_argument("--model", metavar="PATH|NAME",
                        help="live mode on a checkpoint directory, cached model name, model "
                             "alias or tiny:FAMILY (never downloads)")
    parser.add_argument("--dtype", choices=settings.DTYPES,
                        help=f"weights dtype (default {settings.DEFAULT_DTYPE})")
    parser.add_argument("--device", help=f"torch device (default {settings.DEFAULT_DEVICE})")
    parser.add_argument("--seed", type=int, default=settings.DEFAULT_SEED,
                        help="tiny model seed")
    parser.add_argument("--heartbeat", type=float, default=10.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cli = {"project.dir": args.project, "project.session": args.session,
           "serve.host": args.host, "serve.port": args.port,
           "serve.allow_hosts": tuple(args.allow_host) if args.allow_host else None,
           "model.dtype": args.dtype, "model.device": args.device}
    try:
        if args.port is not None and not 1 <= args.port <= 65535:
            raise settings.ConfigError("--port must be 1..65535 (the Host guard needs the "
                                       "bound port)")
        config = settings.activate(settings.resolve(cli, flags=FLAGS, project=args.project))
        if args.model:
            kind, value = parse_target(args.model)
            from ..repl.__main__ import _say_alias

            _say_alias("mdb serve", config, args.model)
            if kind == "connect":
                raise BadArgs("--model names a model to load here, not a server address")
            if kind == "tiny":
                args.model, args.tiny = None, value
            else:
                args.model = value   # open_debugger resolves it (locate_model)
    except (settings.ConfigError, BadArgs) as error:
        print(f"mdb serve: {getattr(error, 'message', error)}", file=sys.stderr)
        return 2
    if config.get("project.dir") is None:
        parser.error("name the project: --project DIR (or [project] dir in a config file)")
    args.project = Path(config.get("project.dir"))
    args.host, args.port = config.get("serve.host"), config.get("serve.port")
    typed_hosts = list(args.allow_host or [])    # --allow-remote needs these typed
    args.allow_host = list(config.get("serve.allow_hosts"))
    args.dtype, args.device = config.get("model.dtype"), config.get("model.device")
    session = config.get("project.session")
    args.session = None if session == "auto" else session
    live_mode = bool(args.tiny or args.model)
    if args.allow_remote and not typed_hosts:
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
                                     dtype=args.dtype, device=args.device, seed=args.seed,
                                     session=args.session)
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
