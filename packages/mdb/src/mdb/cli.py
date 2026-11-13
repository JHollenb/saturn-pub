"""``mdb``: the engine's command line; lane commands come from the ``mdb-lanes`` plug-in.

Engine verbs (implemented in this distribution):

    mdb engine ...                port-level REPL (``python -m mdb.repl``)
    mdb serve --project DIR [--port N] [--allow-host H ...] [--allow-remote]
                                  read-only web viewer of a recorded project (``mdb.serve``)
    mdb lanes [VERB ...]          list, or run, a verb of the lanes plug-in
    mdb version

Every other verb (``open``, ``run``, ``attach``, ``project``, ``capabilities``, ``bench``, ``vm``,
...) is provided by the plug-in through the ``mdb.lanes`` entry-point group. ``mdb lanes VERB``
runs it; a bare ``mdb VERB`` runs it too, after a one-line notice on stderr. Without the plug-in
such a verb exits with status 2 and says so.

Importing this module loads no model framework, no numpy, no web framework and no lane module.
"""
import sys

ENGINE_VERBS = ("engine", "serve", "lanes", "version")
USAGE = ("usage: mdb {engine,serve,lanes,version} ...  |  mdb lanes VERB ...\n"
         "  engine   port-level model REPL (mdb engine --tiny qwen2 --tokens '1 5 9')\n"
         "  serve    read-only viewer: mdb serve --project DIR [--port N]\n"
         "  lanes    commands of the mdb-lanes plug-in (mdb lanes lists them)\n"
         "  version  print the engine version")


def _no_plugin(verb):
    what = "has no commands" if verb == "lanes" else f"cannot run {verb!r}, which is not an engine verb"
    print(f"mdb lanes {what}: the mdb-lanes plug-in is not installed (no 'mdb.lanes' entry points "
          "found). Engine verbs: " + ", ".join(ENGINE_VERBS) + ".", file=sys.stderr)
    return 2


def _lanes(argv, *, notice):
    from .core import plugins

    verb = argv[0]
    run = plugins.verb(verb)
    if run is None:
        if not plugins.installed():
            return _no_plugin(verb)
        print(f"mdb: unknown verb {verb!r}; engine verbs: {', '.join(ENGINE_VERBS)}; "
              f"lanes verbs: {', '.join(plugins.verbs())}", file=sys.stderr)
        return 2
    if notice:
        print(f"mdb: {verb!r} is a lanes command (mdb-lanes plug-in); running it as "
              f"'mdb lanes {verb}'", file=sys.stderr)
    return run(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    verb = argv[0]
    if verb == "engine":
        from .repl.__main__ import main as engine_main
        return engine_main(argv[1:])
    if verb == "serve":
        from .serve.__main__ import main as serve_main  # fastapi loads only when it runs
        return serve_main(argv[1:])
    if verb == "version":
        from . import __version__
        print(__version__)
        return 0
    if verb == "lanes":
        if len(argv) == 1:
            from .core import plugins
            if not plugins.installed():
                return _no_plugin("lanes")
            print("\n".join(plugins.verbs()))
            return 0
        return _lanes(argv[1:], notice=False)
    return _lanes(argv, notice=True)
