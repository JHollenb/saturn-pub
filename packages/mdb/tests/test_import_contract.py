"""Import contract of the engine set (pure standard library: ``ast`` over the source files).

Engine set: ``mdb/__init__.py``, ``mdb/cli.py``, ``mdb/__main__.py`` and every module under
``mdb/{core,engine,analysis,ops,repl,serve}``. Rules, with no allow-list:

1. No engine module imports a flat lane module (``mdb.<module>`` outside the engine set) or
   ``mdb_lanes``: ``import``/``from`` statements (absolute or relative, at any depth, including
   function-local ones), ``importlib.import_module``/``__import__`` with a literal name, and the
   lazy export table of ``mdb/__init__.py``.
2. Layers import only downward (or within their own layer):
   core < engine < analysis < ops < repl < serve < top (``__init__``, ``cli``, ``__main__``).
   The one exception is marked at its own line: an engine-to-engine import whose line carries
   ``# sanctioned upward import`` (today only ``Debugger.repl`` in ``mdb/ops/debugger.py``,
   a function-local convenience). The marker never excuses rule 1 or rule 3.
3. ``from mdb import NAME`` inside the engine names only engine subpackages/modules or the
   engine's own exports (a lane export would reach lane code through the plug-in).
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "mdb"
LAYERS = ("core", "engine", "analysis", "ops", "repl", "serve")
TOP = ("__init__", "cli", "__main__")
RANK = {**{name: i for i, name in enumerate(LAYERS)}, "top": len(LAYERS)}
PACKAGE_NAMES = {"__version__", "attach", "open_tiny", "Debugger"}


def _engine_modules():
    out = {}
    for name in TOP:
        path = SRC / f"{name}.py"
        if path.is_file():
            out["mdb" if name == "__init__" else f"mdb.{name}"] = path
    for layer in LAYERS:
        for path in sorted((SRC / layer).rglob("*.py")):
            parts = path.relative_to(SRC).with_suffix("").parts
            parts = parts[:-1] if parts[-1] == "__init__" else parts
            out[".".join(("mdb", *parts))] = path
    return out


def _layer(module):
    parts = module.split(".")
    if len(parts) == 1 or parts[1] in TOP:
        return "top"
    return parts[1] if parts[1] in LAYERS else None


def _resolve(module, path, level, name):
    """Absolute module of an import statement inside ``module`` (``path`` is its file)."""
    if not level:
        return name
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    base = package[:len(package) - level + 1]
    return ".".join([*base, name] if name else base)


def _imports(module, path):
    """``(line, imported module, names)`` of every import in ``module``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, alias.name, ()) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(module, path, node.level, node.module)
            out.append((node.lineno, target, tuple(a.name for a in node.names)))
        elif (isinstance(node, ast.Call) and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and ((isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                   or (isinstance(node.func, ast.Name)
                       and node.func.id in ("import_module", "__import__")))):
            name = node.args[0].value
            level = len(name) - len(name.lstrip("."))
            out.append((node.lineno, _resolve(module, path, level, name.lstrip(".")) if level
                        else name, ()))
    if module == "mdb":
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "_EXPORTS" for t in node.targets):
                table = ast.literal_eval(node.value)
                out += [(node.lineno, "mdb." + target[0], ()) for target in table.values()]
    return out


def _targets(imported, names):
    """Modules an import reaches: ``from mdb.x import y`` may name submodule ``mdb.x.y``."""
    yield imported
    for name in names:
        if name != "*":
            yield f"{imported}.{name}"


MARKER = "sanctioned upward import"


def _violations():
    engine = _engine_modules()
    problems = []
    for module, path in engine.items():
        mine = _layer(module)
        lines = path.read_text().splitlines()
        for line, imported, names in _imports(module, path):
            marked = MARKER in lines[line - 1]
            where = f"{path.relative_to(SRC.parent)}:{line}"
            if imported.split(".")[0] == "mdb_lanes":
                problems.append(f"{where} imports {imported} (the lanes plug-in)")
                continue
            if imported.split(".")[0] != "mdb":
                continue
            if imported == "mdb":
                bad = [n for n in names if n not in PACKAGE_NAMES and n not in LAYERS
                       and n not in TOP]
                if bad:
                    problems.append(f"{where} takes {bad} from the mdb package (lane exports)")
                for n in names:
                    if n in LAYERS and RANK[n] > RANK[mine]:
                        problems.append(f"{where} ({mine}) imports upward: mdb.{n}")
                continue
            reached = _layer(imported)
            if reached is None or imported.split(".")[1] not in (*LAYERS, *TOP):
                problems.append(f"{where} imports flat lane module {imported}")
                continue
            if RANK[reached] > RANK[mine] and not marked:
                problems.append(f"{where} ({mine}) imports upward: {imported} ({reached})")
    return problems


def test_the_only_marked_upward_import_is_the_repl_convenience():
    marked = [(m, i) for m, path in _engine_modules().items()
              for i, line in enumerate(path.read_text().splitlines(), 1) if MARKER in line]
    assert [m for m, _ in marked] == ["mdb.ops.debugger"]


def test_engine_modules_are_found():
    engine = _engine_modules()
    assert {"mdb", "mdb.cli", "mdb.core.project_read", "mdb.serve.app",
            "mdb.analysis.explain"} <= set(engine)
    assert all(_layer(m) is not None for m in engine)


def test_the_engine_never_imports_a_lane_module_and_layers_only_import_downward():
    problems = _violations()
    assert not problems, "\n".join(problems)


def test_the_checker_catches_each_kind_of_violation(tmp_path, monkeypatch):
    """The rules are live: a planted lane import, an upward import and a lane export each fail."""
    root = tmp_path / "mdb"
    for layer in LAYERS:
        (root / layer).mkdir(parents=True)
        (root / layer / "__init__.py").write_text("")
    (root / "__init__.py").write_text("_EXPORTS = {'attach': ('ops.debugger', 'attach')}\n")
    (root / "core" / "a.py").write_text("from ..project import Project\n")
    (root / "core" / "b.py").write_text("def f():\n    from mdb.serve import app\n")
    (root / "ops" / "c.py").write_text("from mdb import BenchSession\n")
    (root / "repl" / "d.py").write_text("import importlib\nimportlib.import_module('mdb_lanes.x')\n")
    (root / "serve" / "e.py").write_text("from mdb import Project  # sanctioned upward import\n")
    monkeypatch.setattr(__import__(__name__), "SRC", root)
    found = "\n".join(_violations())
    assert "flat lane module mdb.project" in found
    assert "imports upward: mdb.serve" in found
    assert "['BenchSession']" in found
    assert "mdb_lanes.x" in found
    assert "['Project']" in found  # the marker excuses only engine-to-engine upward imports
