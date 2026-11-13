#!/usr/bin/env python3
"""Plan, dry-run and verify the split of ``mdb`` into an engine and ``mdb_lanes``.

Standard library only (``ast``, ``tokenize``, ``pathlib``, ``json``, ``hashlib``,
``compileall``, ``argparse``). Pure static analysis: nothing under analysis is
imported or executed.

Subcommands::

    analyze  --package-root packages/mdb [--json OUT] [--assume-cut FILE]
    plan     --package-root packages/mdb [--json OUT] [--assume-cut FILE] [--strict-engine] [--repin]
    apply    --package-root packages/mdb --out DIR [--repin [--repin-json]]   (copy; refuses DIR inside the repo)
    apply    --package-root packages/mdb --in-place --expect-branch B [--repin] [--allow-dirty PATH ...]
             [--dry-run | --resume | --rollback]
    repin    DIR [--repin-json]          (re-pin sha256 literals in an applied copy)
    verify   DIR [--side SIDE_DIR] [--json OUT]
    report   DIR [--to REPORT.md]

Re-pinning. A 64-hex literal in a ``.py`` file that equals the pre-split sha256 of a
file whose bytes the split changes is replaced by that file's post-split sha256,
iterated to a fixed point (re-pinning changes the pinning file, which may itself be
pinned). Pins inside a cycle (a file pinning itself, directly or through a loop of
files that all change) have no fixed point; they are reported, never rewritten.
JSON/text pins are only listed unless ``--repin-json``.

Rule-based edits beyond import statements (each also listed in manual-fixes.md when
the rule cannot decide): enumerable dynamic import prefixes (``import_module("mdb."
+ name)`` with ``name`` drawn from a literal tuple or from literal call sites), the
seal/source-hash keys in the same loop, literal module tuples consumed by
``import_module``, descriptive ``"api"``/``"public_module"`` strings, and file paths
anchored at ``Path(__file__)`` whose package directory lands in the other
distribution. Versioned identity strings (``mdb.x.v1``) are never edited.

Layout model. A tree may be unsplit (everything under ``packages/mdb``) or
already split (``packages/mdb`` plus a sibling ``packages/mdb-lanes`` holding
``src/mdb_lanes``, ``tests`` and ``qualification``). Every module has a logical
key (dotted path inside its package, ``""`` for the top ``__init__``) and a
current *home* (``mdb`` or ``mdb_lanes``). The split decides a destination home;
lanes is sticky (nothing already in lanes is planned back into the engine), so
re-running on an already split tree plans zero moves and zero rewrites, and a
re-run after new files appear plans only those. ``apply --out`` copies into a fresh
directory; ``apply --in-place`` performs the same plan in a git checkout with
``git mv`` for moves and plain writes for rewritten/generated files, journaled under
``<git dir>/split-lanes/`` so an interrupted run can be resumed or rolled back.

Recognised string-reference contexts (rewritten when they resolve to a module
whose home changes; everything else is reported, never rewritten):
  importlib    first argument of import_module / find_spec / __import__ /
               runpy.run_module / pytest.importorskip
  dash-m       a string element following a "-m" element in a list/tuple
               literal, or "-m mdb.x" inside any string (shell command text)
  patch        first argument of *.patch(...) and monkeypatch/*.setattr/
               delattr(...) when that argument is a dotted string
  sys-modules  sys.modules["mdb.x"], sys.modules.get/pop/setdefault("mdb.x"),
               monkeypatch.setitem/delitem(sys.modules, "mdb.x", ...)
  payload      dict literal value or keyword argument named module /
               worker_module / module_name / entrypoint / entry_point /
               main_module / worker_entry
  tables       module tables used as import_module("." + T[...]) (cli TOOLS is
               a registry: lane verbs are reported for re-registration)
"""
from __future__ import annotations

import argparse
import ast
import compileall
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import tokenize
from collections import defaultdict, deque
from pathlib import Path

ENGINE = "mdb"
LANES = "mdb_lanes"
LANES_DIST = "mdb-lanes"
HOMES = (ENGINE, LANES)
ENGINE_SUBPACKAGES = ("core", "engine", "analysis", "ops", "repl", "serve")
ENGINE_TOP_MODULES = ("", "__main__", "cli")
REGISTRY_TABLES = {("cli", "TOOLS")}
AUX_AREAS = ("examples", "tools", "experiments")
PACKAGE_DIRS = ("src", "tests", "qualification", "examples", "experiments", "tools", "docs", "results")
CLOSURE_KINDS = {"import", "from-import", "importlib", "dunder-import", "run-module", "dash-m",
                 "payload-module", "lazy-table", "lazy-export-attr", "parent-package", "path-file"}
CLASSIFY_KINDS = CLOSURE_KINDS - {"parent-package", "path-file"} | {"patch-target", "sys-modules", "importorskip",
                                                       "path-src", "registry"}
MODULE_KEYS = {"module", "worker_module", "module_name", "entrypoint", "entry_point", "main_module",
               "worker_entry"}
IMPORT_CALLS = {"import_module": "importlib", "find_spec": "importlib", "__import__": "dunder-import",
                "run_module": "run-module", "importorskip": "importorskip"}
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".toml", ".yaml", ".yml", ".cfg", ".csv", ".tsv",
                 ".html", ".ini"}
TEXT_SCAN_SKIP_DIRS = {"assets", "__pycache__", ".git", "node_modules", ".venv", "vendor"}
TEXT_SCAN_MAX_BYTES = 8 << 20
DATA_COPY_MAX_BYTES = 2 << 20
HEX64 = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
HEX64_S = re.compile(r"[0-9a-f]{64}")
DOTTED_RE = re.compile(r"(?<![\w.])(mdb(?:_lanes)?(?:\.[A-Za-z_]\w*)+)")
DOTTED_RE_B = re.compile(rb"(?<![\w.])(mdb(?:_lanes)?(?:\.[A-Za-z_]\w*)+)")
PKGPATH_RE_B = re.compile(rb"(?<![\w/])(mdb(?:_lanes)?/[\w/]+\.py)")
DASH_M_RE = re.compile(r"-m\s+(mdb(?:_lanes)?(?:\.[A-Za-z_]\w*)+)")
HASHY = re.compile(r"sha|digest|hash|md5|blake", re.I)
IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
PICKLE_CALLS = {"pickle.dump", "pickle.dumps", "pickle.load", "pickle.loads", "torch.save", "torch.load",
                "cloudpickle.dump", "cloudpickle.dumps", "dill.dump", "dill.dumps", "joblib.dump",
                "joblib.load", "shelve.open", "copyreg.pickle", "marshal.dump", "marshal.dumps"}
PICKLE_DEFS = {"__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__getnewargs__"}
MANIFEST = "split-manifest.json"
SIDE_DIR = "_split"
BOM = b"\xef\xbb\xbf"
DESCRIPTIVE_KEYS = {"api", "public_module"}
IDENTITY_RE = re.compile(r"(?:mdb|mdb_lanes)(?:\.\w+)+\.v\d+")
SEAL_CALLS = {"package_files", "seal_package"}
INPLACE_DIR = "split-lanes"  # under the git dir: never shows in `git status`
LANES_VERB_GROUP = "mdb.lanes"


# ---------------------------------------------------------------- utilities

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def other_home(home: str) -> str:
    return LANES if home == ENGINE else ENGINE


def full_name(home: str, key: str) -> str:
    return home + ("." + key if key else "")


def mid(home: str, key: str) -> str:
    return home + ":" + key


def split_mid(value: str):
    home, key = value.split(":", 1)
    return home, key


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def call_name(node):
    if isinstance(node, ast.Call):
        return dotted(node.func) or ""
    return ""


def func_tail(node):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
    return ""


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def find_repo_root(start: Path):
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


class Lines:
    """Byte offsets of AST (line, col) positions; col offsets are UTF-8 bytes."""

    def __init__(self, data: bytes):
        self.data = data
        self.starts = [0]
        for line in data.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))
        self.bom = 3 if data.startswith(BOM) else 0

    def off(self, line: int, col: int) -> int:
        return self.starts[line - 1] + col + (self.bom if line == 1 else 0)

    def span(self, node):
        return self.off(node.lineno, node.col_offset), self.off(node.end_lineno, node.end_col_offset)

    def text(self, start, end):
        return self.data[start:end].decode("utf-8", "replace")

    def line_text(self, line: int) -> str:
        if line - 1 >= len(self.starts) - 1:
            return ""
        return self.data[self.starts[line - 1]:self.starts[line]].decode("utf-8", "replace").rstrip("\r\n")

    def count(self) -> int:
        return len(self.starts) - 1


# ---------------------------------------------------------------- layout

class FileInfo:
    __slots__ = ("path", "root", "area", "rel", "home", "key", "is_package", "data", "sha", "lines")

    def __init__(self, path, root, area, rel, home=None, key=None, is_package=False):
        self.path, self.root, self.area, self.rel = path, root, area, rel
        self.home, self.key, self.is_package = home, key, is_package
        self.data = path.read_bytes()
        self.sha = sha256(self.data)
        self.lines = len(self.data.splitlines())

    @property
    def mid(self):
        return mid(self.home, self.key) if self.area == "src" else None

    @property
    def label(self):
        """Path relative to the packages/ directory (stable across homes)."""
        if self.area == "src":
            return f"{self.root}/src/{self.home}/{self.rel}"
        return f"{self.root}/{self.area}/{self.rel}"


class Layout:
    """Discover every relevant file of an unsplit or already split tree."""

    def __init__(self, package_root: Path):
        self.engine_root = package_root.resolve()
        self.lanes_root = self.engine_root.parent / LANES_DIST
        self.packages_dir = self.engine_root.parent
        self.files: list[FileInfo] = []
        self.modules: dict[str, FileInfo] = {}
        self.package_data: list[tuple[str, Path, str]] = []  # (home, path, rel)
        self.conflicts: list[str] = []
        for root_name, root, home in ((self.engine_root.name, self.engine_root, ENGINE),
                                      (LANES_DIST, self.lanes_root, LANES)):
            if not root.exists():
                continue
            src = root / "src" / home
            if src.exists():
                for path in sorted(src.rglob("*")):
                    if "__pycache__" in path.parts or not path.is_file():
                        continue
                    rel = path.relative_to(src).as_posix()
                    if path.suffix != ".py":
                        if path.suffix not in {".pyc", ".pyo"}:
                            self.package_data.append((home, path, rel))
                        continue
                    parts = rel[:-3].split("/")
                    is_package = parts[-1] == "__init__"
                    key = ".".join(parts[:-1] if is_package else parts)
                    info = FileInfo(path, root_name, "src", rel, home, key, is_package)
                    if info.mid in self.modules:
                        self.conflicts.append(info.mid)
                    self.modules[info.mid] = info
                    self.files.append(info)
            areas = ("tests", "qualification") + (AUX_AREAS if home == ENGINE else ())
            for area in areas:
                base = root / area
                if not base.exists():
                    continue
                for path in sorted(base.rglob("*.py")):
                    if "__pycache__" in path.parts or not path.is_file():
                        continue
                    self.files.append(FileInfo(path, root_name, area, path.relative_to(base).as_posix(),
                                               home=home))
        self.by_path = {f.path: f for f in self.files}
        self.by_resolved = {f.path.resolve(): f for f in self.files}
        keys = defaultdict(set)
        for info in self.modules.values():
            keys[info.key].add(info.home)
        self.split_keys = {k for k, v in keys.items() if len(v) > 1 and k}
        self.conflicts += sorted(k for k in self.split_keys
                                 if not any(self.modules[mid(h, k)].is_package for h in keys[k]))

    def area_root(self, home: str, area: str) -> Path:
        root = self.engine_root if home == ENGINE else self.lanes_root
        return root / "src" / home if area == "src" else root / area

    def lookup(self, home: str, key: str, strict: bool = False):
        """Return (FileInfo, stale) for module ``home.key``; non-strict falls back to the other home."""
        info = self.modules.get(mid(home, key))
        if info is not None or strict or key == "":
            return info, False
        info = self.modules.get(mid(other_home(home), key))
        return info, info is not None

    def resolve_dotted(self, name: str, strict: bool = False, require_full: bool = False):
        parts = name.split(".")
        if parts[0] not in HOMES:
            return None
        for k in range(len(parts), 0, -1):
            info, stale = self.lookup(parts[0], ".".join(parts[1:k]), strict)
            if info is not None:
                if require_full and k != len(parts):
                    return None
                return info, parts[k:], stale
        return None


# ---------------------------------------------------------------- scanning

class Ref:
    __slots__ = ("kind", "line", "col", "start", "end", "text", "target", "rest", "lazy", "optional",
                 "type_checking", "confident", "written", "level", "names", "stmt", "note", "file")

    def __init__(self, kind, line, col, *, start=None, end=None, text="", target=None, rest="",
                 lazy=False, optional=False, type_checking=False, confident=True, written=None,
                 level=0, names=None, stmt=None, note=""):
        self.kind, self.line, self.col = kind, line, col
        self.start, self.end, self.text = start, end, text
        self.target, self.rest = target, rest
        self.lazy, self.optional, self.type_checking = lazy, optional, type_checking
        self.confident, self.written, self.level = confident, written, level
        self.names, self.stmt, self.note = names, stmt, note
        self.file = None

    def to_json(self):
        out = {k: getattr(self, k) for k in ("kind", "line", "col", "text", "target", "lazy", "optional")}
        if self.type_checking:
            out["type_checking"] = True
        if not self.confident:
            out["confident"] = False
        if self.note:
            out["note"] = self.note
        if self.rest:
            out["rest"] = self.rest
        return out


class Scan:
    """Everything one pass over a file's AST yields."""

    def __init__(self, info: FileInfo):
        self.info = info
        self.refs: list[Ref] = []
        self.dynamic: list[dict] = []
        self.unconfident: list[dict] = []
        self.mentions: list[dict] = []
        self.path_refs: list[dict] = []
        self.siblings: list[dict] = []
        self.registrations: list[dict] = []
        self.source_bound: list[dict] = []
        self.pickle: list[dict] = []
        self.hex_consts: list[dict] = []
        self.prefix_imports: list[dict] = []
        self.identity_strings: list[dict] = []
        self.package_seals: list[dict] = []
        self.attr_chains: list[tuple] = []  # (dotted, start, end, line, col)
        self.top_names: set[str] = set()
        self.open_namespace = False
        self.syntax_error = None
        self.lines: Lines | None = None


def _scope(node):
    lazy = optional = type_checking = False
    child, parent = node, getattr(node, "_sl_parent", None)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            lazy = True
        elif isinstance(parent, ast.Try) or type(parent).__name__ == "TryStar":
            if child in parent.body and any(_catches_import(h) for h in parent.handlers):
                optional = True
        elif isinstance(parent, ast.If) and child in parent.body:
            test = dotted(parent.test) or ""
            if test.endswith("TYPE_CHECKING"):
                type_checking = True
        child, parent = parent, getattr(parent, "_sl_parent", None)
    return lazy, optional, type_checking


def _catches_import(handler):
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any((dotted(n) or "").split(".")[-1] in IMPORT_ERRORS for n in names)


def _str_const(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _tokens_span(lines: Lines, node, stop_name: str):
    """Byte span of the module part of ``from <module> import`` inside statement ``node``."""
    start, end = lines.span(node)
    seg = lines.data[start:end]
    seg_lines = seg.splitlines(keepends=True)
    seg_starts = [0]
    for line in seg_lines:
        seg_starts.append(seg_starts[-1] + len(line))
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(seg).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None

    def boff(row, col):
        line = seg_lines[row - 1].decode("utf-8")
        return seg_starts[row - 1] + len(line[:col].encode("utf-8"))

    seen_from = False
    parts = []
    for tok in tokens:
        if not seen_from:
            if tok.type == tokenize.NAME and tok.string == "from":
                seen_from = True
            continue
        if tok.type == tokenize.NAME and tok.string == stop_name:
            break
        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            continue
        parts.append(tok)
    if not parts:
        return None
    return start + boff(*parts[0].start), start + boff(*parts[-1].end), "".join(t.string for t in parts)


class Scanner:
    def __init__(self, layout: Layout, exports: dict | None = None, strict: bool = False):
        self.layout = layout
        self.exports = exports or {}
        self.strict = strict

    # -- resolution helpers
    def resolve_abs(self, name, require_full=False):
        return self.layout.resolve_dotted(name, strict=self.strict, require_full=require_full)

    def package_key(self, info: FileInfo):
        if info.area != "src":
            return None
        if info.is_package:
            return info.key
        return info.key.rpartition(".")[0] if "." in info.key else ""

    def resolve_relative(self, info, level, module):
        base = self.package_key(info)
        if base is None:
            return None
        parts = base.split(".") if base else []
        if level - 1 > len(parts):
            return None
        parts = parts[:len(parts) - (level - 1)] if level > 1 else parts
        if module:
            parts = parts + module.split(".")
        return ".".join(parts)

    def module_target(self, home, key):
        info, stale = self.layout.lookup(home, key, self.strict)
        return (info.mid if info else None), stale

    # -- main
    def scan(self, info: FileInfo) -> Scan:
        result = Scan(info)
        lines = Lines(info.data)
        result.lines = lines
        try:
            tree = ast.parse(info.data, filename=str(info.path))
        except SyntaxError as exc:
            result.syntax_error = f"{exc.msg} (line {exc.lineno})"
            return result
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._sl_parent = node
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and _str_const(body[0].value) is not None:
                    docstrings.add(id(body[0].value))
        self._top_names(tree, result)
        self._tree = tree
        self._assigns = _module_assigns(tree)
        tables = self._tables(tree, info, lines, result)
        handled = set()
        sibling_dir = info.path.parent
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._import(node, info, lines, result, sibling_dir)
            elif isinstance(node, ast.ImportFrom):
                self._import_from(node, info, lines, result, sibling_dir)
            elif isinstance(node, ast.Call):
                self._call(node, info, lines, result, handled, tables)
            elif isinstance(node, (ast.List, ast.Tuple)):
                self._dash_m_list(node, info, lines, result, handled)
                if len(node.elts) >= 2 and _str_const(node.elts[0]) == ENGINE and \
                        _str_const(node.elts[1]) == "mdb-runtime":
                    s, e = lines.span(node)
                    result.package_seals.append({"line": node.lineno, "call": "distribution-table",
                                                 "package": ENGINE, "start": s, "end": e, "form": "table-row"})
            elif isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if _str_const(k) in MODULE_KEYS and _str_const(v) is not None:
                        self._string_ref(v, "payload-module", info, lines, result, handled)
                    elif _str_const(k) in DESCRIPTIVE_KEYS and _str_const(v) is not None and id(v) not in handled:
                        self._descriptive(v, info, lines, result, handled)
                if node.keys and all(_str_const(k) is not None for k in node.keys):
                    for k, v in zip(node.keys, node.values):
                        value = v.value if isinstance(v, ast.Constant) else None
                        if isinstance(value, str) and HEX64_S.fullmatch(value):
                            result.hex_consts.append({"line": v.lineno, "value": value, "context": _str_const(k)})
            elif isinstance(node, ast.Subscript):
                if dotted(node.value) == "sys.modules" and _str_const(node.slice) is not None:
                    self._string_ref(node.slice, "sys-modules", info, lines, result, handled)
            elif isinstance(node, ast.Attribute):
                name = dotted(node)
                if name and name.split(".")[0] in HOMES and "." in name:
                    s, e = lines.span(node)
                    result.attr_chains.append((name, s, e, node.lineno, node.col_offset))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PICKLE_DEFS:
                result.pickle.append({"line": node.lineno, "what": "def " + node.name})
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and \
                        HEX64_S.fullmatch(value.value):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    result.hex_consts.append({"line": value.lineno, "value": value.value,
                                              "context": ",".join(dotted(t) or "?" for t in targets)})
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and IDENTITY_RE.fullmatch(node.value):
                s, e = lines.span(node)
                result.identity_strings.append({"line": node.lineno, "text": node.value, "start": s, "end": e})
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in handled:
                if isinstance(getattr(node, "_sl_parent", None), ast.JoinedStr):
                    self._mention(node, info, lines, result, docstring=False, fstring=True)
                    continue
                self._inline_dash_m(node, info, lines, result)
                self._mention(node, info, lines, result, docstring=id(node) in docstrings)
                self._path_ref(node, info, lines, result)
                value = node.value
                if HEX64_S.fullmatch(value) and not any(h["line"] == node.lineno and
                                                              h["value"] == value for h in result.hex_consts):
                    result.hex_consts.append({"line": node.lineno, "value": value, "context": ""})
        self._source_bound(tree, info, result)
        aliases = {}
        for ref in result.refs:
            if ref.kind == "from-import":
                for n in ref.names:
                    if n["module"] and n["target"]:
                        aliases[n["asname"] or n["name"]] = n["target"]
            elif ref.kind == "import" and ref.target and ref.note.startswith("alias:"):
                aliases[ref.note[6:]] = ref.target
        for reg in result.registrations:
            reg["owner_module"] = info.mid
            qual = reg.get("qualname") or ""
            if qual.startswith("attr:"):
                base, _, cls = qual[5:].rpartition(".")
                if base in aliases:
                    reg["owner_module"] = aliases[base]
                    reg["qualname"] = cls + " (instance of a class in " + aliases[base] + \
                        "; digest binds type(impl).__module__/__qualname__)"
        for ref in result.refs:
            ref.file = info
        self._tree = self._assigns = None
        return result

    def _top_names(self, tree, result):
        def visit(body):
            for stmt in body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    result.top_names.add(stmt.name)
                    if stmt.name == "__getattr__":
                        result.open_namespace = True
                elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    for target in targets:
                        for n in ast.walk(target):
                            if isinstance(n, ast.Name):
                                result.top_names.add(n.id)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    for alias in stmt.names:
                        if alias.name == "*":
                            result.open_namespace = True
                        else:
                            result.top_names.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)) or \
                        type(stmt).__name__ == "TryStar":
                    for field_name in ("body", "orelse", "finalbody"):
                        visit(getattr(stmt, field_name, []) or [])
                    for handler in getattr(stmt, "handlers", []) or []:
                        visit(handler.body)
        visit(tree.body)

    # -- module tables: import_module("." + T[x], ...) or module, attr = T[x]; import_module("." + module)
    def _tables(self, tree, info, lines, result):
        tables = {}
        names_used = set()
        for node in ast.walk(tree):
            if func_tail(node) == "import_module" and node.args:
                for n in ast.walk(node.args[0]):
                    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                        names_used.add(n.value.id)
                    elif isinstance(n, ast.Name):
                        func = n
                        parent = getattr(node, "_sl_parent", None)
                        while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.Module)):
                            parent = getattr(parent, "_sl_parent", None)
                        for stmt in ast.walk(parent) if parent is not None else ():
                            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Subscript) and \
                                    isinstance(stmt.value.value, ast.Name):
                                targets = [t for tgt in stmt.targets for t in ast.walk(tgt) if isinstance(t, ast.Name)]
                                if any(t.id == func.id for t in targets):
                                    names_used.add(stmt.value.value.id)
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id in names_used:
                        for k, v in zip(stmt.value.keys, stmt.value.values):
                            tables.setdefault(target.id, []).append((k, v))
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and \
                    isinstance(stmt.targets[0], ast.Subscript) and isinstance(stmt.targets[0].value, ast.Name) \
                    and stmt.targets[0].value.id in names_used:
                tables.setdefault(stmt.targets[0].value.id, []).append((stmt.targets[0].slice, stmt.value))
        resolved = {}
        for name, entries in tables.items():
            registry = info.area == "src" and (info.key, name) in REGISTRY_TABLES
            for k, v in entries:
                node = v.elts[0] if isinstance(v, ast.Tuple) and v.elts else v
                value = _str_const(node)
                if value is None:
                    continue
                base = self.package_key(info)
                key = ".".join(p for p in (base, value) if p)
                target, stale = self.module_target(info.home or ENGINE, key) if info.area == "src" else (None, False)
                lazy, optional, tc = _scope(node)
                s, e = lines.span(node)
                ref = Ref("registry" if registry else "lazy-table", node.lineno, node.col_offset, start=s, end=e,
                          text=lines.text(s, e), target=target, written="table", lazy=lazy, optional=optional,
                          note=f"{name}[{_str_const(k)!s}]")
                result.refs.append(ref)
                if name == "_EXPORTS" or (info.area == "src" and info.key == "" and not registry):
                    attr = _str_const(k)
                    if attr and target:
                        resolved[attr] = target
        if info.area == "src" and info.key == "" and info.home == ENGINE:
            self.exports.update(resolved)
        return set(tables)

    def _alias_span(self, lines, alias):
        if not hasattr(alias, "lineno"):
            return None
        s, e = lines.span(alias)
        text = lines.data[s:s + len(alias.name.encode())].decode("utf-8", "replace")
        if text != alias.name:
            return None
        return s, s + len(alias.name.encode())

    def _import(self, node, info, lines, result, sibling_dir):
        lazy, optional, tc = _scope(node)
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in HOMES:
                res = self.resolve_abs(alias.name, require_full=True)
                span = self._alias_span(lines, alias)
                ref = Ref("import", node.lineno, node.col_offset, start=span[0] if span else None,
                          end=span[1] if span else None, text=alias.name,
                          target=res[0].mid if res else None, lazy=lazy, optional=optional, type_checking=tc,
                          confident=span is not None, written=top,
                          note=("alias:" + alias.asname) if alias.asname else "unaliased")
                if res and res[2]:
                    ref.note += ";stale"
                result.refs.append(ref)
            elif info.area != "src":
                self._sibling(node, top, info, result, sibling_dir, lazy)

    def _sibling(self, node, top, info, result, sibling_dir, lazy):
        for candidate in (sibling_dir / (top + ".py"), sibling_dir / top / "__init__.py"):
            if candidate.exists():
                result.siblings.append({"line": node.lineno, "name": top, "lazy": lazy,
                                        "target": candidate.relative_to(sibling_dir).as_posix()})
                return

    def _import_from(self, node, info, lines, result, sibling_dir):
        lazy, optional, tc = _scope(node)
        level, module = node.level, node.module or ""
        if level == 0:
            top = module.split(".")[0]
            if top not in HOMES:
                if info.area != "src" and top:
                    self._sibling(node, top, info, result, sibling_dir, lazy)
                return
            home = top
            base_key = ".".join(module.split(".")[1:])
            written = top
        else:
            base_key = self.resolve_relative(info, level, module)
            if base_key is None:
                result.dynamic.append({"line": node.lineno, "what": "relative import outside a package",
                                       "text": lines.line_text(node.lineno).strip()})
                return
            home = info.home
            written = "relative"
        base_info, stale = self.layout.lookup(home, base_key, self.strict or level > 0)
        span = _tokens_span(lines, node, "import")
        expected = "." * level + module
        confident = span is not None and span[2] == expected
        names = []
        for alias in node.names:
            sub_key = ".".join(p for p in (base_key, alias.name) if p)
            sub_info, sub_stale = self.layout.lookup(home, sub_key, self.strict or level > 0) \
                if alias.name != "*" else (None, False)
            a_span = lines.span(alias) if hasattr(alias, "lineno") else None
            if sub_info is not None:
                names.append({"name": alias.name, "asname": alias.asname, "module": True,
                              "target": sub_info.mid, "span": a_span, "stale": sub_stale})
            else:
                target = base_info.mid if base_info else None
                kind = "attr"
                if base_info and base_info.key == "" and base_info.home == ENGINE and alias.name in self.exports:
                    kind = "export"
                names.append({"name": alias.name, "asname": alias.asname, "module": False,
                              "target": target, "span": a_span, "kind": kind,
                              "export_target": self.exports.get(alias.name) if kind == "export" else None,
                              "stale": stale})
        stmt = lines.span(node)
        ref = Ref("from-import", node.lineno, node.col_offset,
                  start=span[0] if span else None, end=span[1] if span else None,
                  text=expected, target=base_info.mid if base_info else None, lazy=lazy, optional=optional,
                  type_checking=tc, confident=confident, written=written, level=level, names=names, stmt=stmt)
        result.refs.append(ref)

    def _call(self, node, info, lines, result, handled, tables):
        tail = func_tail(node)
        name = call_name(node)
        if name in PICKLE_CALLS or (tail in {"dump", "dumps", "load", "loads"} and
                                    name.split(".")[0] in {"pickle", "cloudpickle", "dill", "joblib"}):
            result.pickle.append({"line": node.lineno, "what": name})
        if tail in IMPORT_CALLS and node.args:
            arg = node.args[0]
            value = _str_const(arg)
            kind = IMPORT_CALLS[tail]
            if tail == "find_spec" and not name.startswith(("importlib", "util")):
                kind = "importlib"
            if value is not None:
                if value.startswith(".") and info.area == "src":
                    self._relative_string(arg, value, info, lines, result, handled, kind)
                else:
                    self._string_ref(arg, kind, info, lines, result, handled, optional=tail == "importorskip")
            else:
                uses_table = any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and
                                 n.value.id in tables for n in ast.walk(arg)) or \
                    any(isinstance(n, ast.Name) and n.id == "module" for n in ast.walk(arg)) and tables
                rule = None
                if not uses_table and isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add) and \
                        _str_const(arg.left) in (ENGINE + ".", LANES + ".") and isinstance(arg.right, ast.Name):
                    rule = self._prefix_import(node, arg, kind, info, lines, result)
                elif not uses_table and isinstance(arg, ast.Name):
                    values, _, elts = _binding_values(arg, self._tree)
                    if elts:
                        for elt in elts:
                            if id(elt) not in handled:
                                self._string_ref(elt, kind, info, lines, result, handled)
                        rule = "literal-tuple-elements"
                if not uses_table:
                    prefix = _str_const(arg.left) if isinstance(arg, ast.BinOp) else None
                    s, e = lines.span(node)
                    if rule or kind in {"importlib", "run-module"} or prefix is not None or "mdb" in lines.text(s, e):
                        result.dynamic.append({"line": node.lineno, "what": f"{name}(<non-literal>)",
                                               "prefix": prefix, "text": lines.text(s, e)[:160], "rule": rule})
        elif tail == "patch" and node.args and _str_const(node.args[0]) is not None:
            self._string_ref(node.args[0], "patch-target", info, lines, result, handled)
        elif tail in {"setattr", "delattr"} and node.args and _str_const(node.args[0]) is not None and \
                "." in _str_const(node.args[0]) and not name.startswith(("builtins", "object")) and name != tail:
            self._string_ref(node.args[0], "patch-target", info, lines, result, handled)
        elif tail in {"setitem", "delitem"} and len(node.args) >= 2 and dotted(node.args[0]) == "sys.modules" \
                and _str_const(node.args[1]) is not None:
            self._string_ref(node.args[1], "sys-modules", info, lines, result, handled)
        elif tail in {"get", "pop", "setdefault"} and isinstance(node.func, ast.Attribute) and \
                dotted(node.func.value) == "sys.modules" and node.args and _str_const(node.args[0]) is not None:
            self._string_ref(node.args[0], "sys-modules", info, lines, result, handled)
        for keyword in node.keywords:
            if keyword.arg in MODULE_KEYS and _str_const(keyword.value) is not None:
                self._string_ref(keyword.value, "payload-module", info, lines, result, handled)
        if tail in SEAL_CALLS and node.args:
            arg = node.args[0]
            s, e = lines.span(node)
            if _str_const(arg) in HOMES:
                result.package_seals.append({"line": node.lineno, "call": tail, "package": _str_const(arg),
                                             "start": s, "end": e, "form": "literal"})
            elif isinstance(arg, ast.Name):
                _, _, elts = _binding_values(arg, self._tree)
                if elts and any(_str_const(x) in HOMES for x in elts):
                    ts, te = lines.span(elts[0]._sl_parent)
                    result.package_seals.append({"line": node.lineno, "call": tail, "package": None,
                                                 "start": ts, "end": te, "form": "literal-tuple",
                                                 "names": [_str_const(x) for x in elts]})
        if tail in {"register_operation", "_register_operation", "registered", "register"}:
            receiver = dotted(node.func.value) if isinstance(node.func, ast.Attribute) else None
            if tail == "_register_operation" or (receiver or "").split(".")[-1] == "Act":
                self._registration(node, tail, info, lines, result)

    def _prefix_import(self, node, arg, kind, info, lines, result):
        """import_module("mdb." + name): enumerate ``name`` and record the prefix span (and, inside the
        same loop, the seal-key spans ``seal["packages"]["mdb"]`` and ``"mdb/" + name``)."""
        home = _str_const(arg.left)[:-1]
        var = arg.right.id
        values, scope_node, _ = _binding_values(arg.right, self._tree)
        spans = []
        s, e = lines.span(arg.left)
        raw = lines.data[s:e]
        if raw.count(home.encode() + b".") == 1:
            off = raw.index(home.encode() + b".")
            spans.append([s + off, s + off + len(home), "import-prefix", arg.left.lineno, arg.left.col_offset])
        if isinstance(scope_node, (ast.For, ast.AsyncFor, ast.comprehension)):
            body = scope_node.body if isinstance(scope_node, (ast.For, ast.AsyncFor)) else [scope_node._sl_parent]
            for stmt in body:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Subscript) and _str_const(n.slice) == home and \
                            isinstance(n.value, ast.Subscript) and _str_const(n.value.slice) == "packages":
                        a, b = lines.span(n.slice)
                        r = lines.data[a:b]
                        if r.count(home.encode()) == 1:
                            o = r.index(home.encode())
                            spans.append([a + o, a + o + len(home), "seal-package-key", n.lineno, n.col_offset])
                    elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) and \
                            _str_const(n.left) == home + "/" and isinstance(n.right, ast.Name) and n.right.id == var:
                        a, b = lines.span(n.left)
                        r = lines.data[a:b]
                        if r.count(home.encode() + b"/") == 1:
                            o = r.index(home.encode() + b"/")
                            spans.append([a + o, a + o + len(home), "source-hash-key", n.lineno, n.col_offset])
        bs, be = lines.span(arg)
        result.prefix_imports.append({"line": node.lineno, "col": node.col_offset, "home": home, "var": var,
                                      "values": values, "spans": spans, "expr": [bs, be],
                                      "text": lines.text(*lines.span(node))[:160]})
        lazy, optional, tc = _scope(node)
        for value in values or ():
            name = home + "." + value
            res = self.resolve_abs(name)
            result.refs.append(Ref(kind, node.lineno, node.col_offset, text=name,
                                   target=res[0].mid if res else None, rest=".".join(res[1]) if res else "",
                                   lazy=lazy, optional=optional, type_checking=tc, written=home,
                                   note="enumerated"))
        return "prefix-enumerated" if values is not None else "prefix-unenumerable"

    def _descriptive(self, node, info, lines, result, handled):
        """Module paths named as documentation values ("api": "mdb.x.Cls", "public_module": "mdb.x")."""
        value = node.value
        handled.add(id(node))
        if IDENTITY_RE.fullmatch(value):
            return
        s, e = lines.span(node)
        raw = lines.data[s:e]
        lazy, optional, tc = _scope(node)
        for match in DOTTED_RE.finditer(value):
            name = match.group(1)
            res = self.resolve_abs(name)
            if not res:
                continue
            encoded = name.encode("utf-8", "surrogatepass")
            # Unique occurrence, and not a prefix of a longer dotted name in the same literal.
            hits = [m.start() for m in re.finditer(re.escape(encoded) + rb"(?![\w.])", raw)]
            confident = len(hits) == 1
            start = s + hits[0] if confident else None
            result.refs.append(Ref("api-string", node.lineno, node.col_offset, start=start,
                                   end=start + len(encoded) if confident else None, text=name,
                                   target=res[0].mid, rest=".".join(res[1]), lazy=lazy, optional=optional,
                                   type_checking=tc, confident=confident, written=name.split(".")[0]))

    def _registration(self, node, tail, info, lines, result):
        implementation = None
        if tail in {"register_operation", "_register_operation", "register"} and len(node.args) >= 3:
            implementation = node.args[2]
        for keyword in node.keywords:
            if keyword.arg == "implementation":
                implementation = keyword.value
        qualname = None
        if isinstance(implementation, ast.Name):
            qualname = _resolve_local_def(node, implementation.id)
        op_id = node.args[0] if node.args else None
        s, e = lines.span(node)
        result.registrations.append({
            "line": node.lineno, "call": tail, "text": lines.text(s, e)[:200],
            "operation_id": ast.unparse(op_id) if op_id is not None else None,
            "implementation": ast.unparse(implementation) if implementation is not None else None,
            "qualname": qualname})

    def _relative_string(self, node, value, info, lines, result, handled, kind):
        level = len(value) - len(value.lstrip("."))
        key = self.resolve_relative(info, level, value[level:])
        handled.add(id(node))
        lazy, optional, tc = _scope(node)
        target = None
        if key is not None:
            target_info, _ = self.layout.lookup(info.home, key, True)
            target = target_info.mid if target_info else None
        s, e = lines.span(node)
        raw = lines.data[s:e]
        confident = raw.count(value.encode("utf-8", "surrogatepass")) == 1
        if confident:
            off = raw.index(value.encode("utf-8", "surrogatepass"))
            s, e = s + off, s + off + len(value.encode("utf-8", "surrogatepass"))
        ref = Ref(kind, node.lineno, node.col_offset, start=s if confident else None,
                  end=e if confident else None, text=value, target=target, lazy=lazy, optional=optional,
                  type_checking=tc, confident=confident, written="relative", level=level)
        result.refs.append(ref)

    def _string_ref(self, node, kind, info, lines, result, handled, optional=False):
        value = _str_const(node)
        handled.add(id(node))
        if value is None:
            return
        name = value.split(":")[0] if kind == "payload-module" else value
        if name.split(".")[0] not in HOMES:
            return
        res = self.resolve_abs(name)
        lazy, opt, tc = _scope(node)
        s, e = lines.span(node)
        raw = lines.data[s:e]
        confident = raw.count(name.encode("utf-8", "surrogatepass")) == 1 and not isinstance(getattr(node, "_sl_parent", None),
                                                                      ast.JoinedStr)
        if confident:
            off = raw.index(name.encode("utf-8", "surrogatepass"))
            s, e = s + off, s + off + len(name.encode("utf-8", "surrogatepass"))
        ref = Ref(kind, node.lineno, node.col_offset, start=s if confident else None,
                  end=e if confident else None, text=name, target=res[0].mid if res else None,
                  rest=".".join(res[1]) if res else "", lazy=lazy, optional=optional or opt,
                  type_checking=tc, confident=confident, written=name.split(".")[0])
        if res and res[2]:
            ref.note = "stale"
        result.refs.append(ref)

    def _dash_m_list(self, node, info, lines, result, handled):
        elts = node.elts
        for i, elt in enumerate(elts[:-1]):
            if _str_const(elt) == "-m" and _str_const(elts[i + 1]) is not None:
                self._string_ref(elts[i + 1], "dash-m", info, lines, result, handled)

    def _inline_dash_m(self, node, info, lines, result):
        value = node.value
        if "-m" not in value:
            return
        for match in DASH_M_RE.finditer(value):
            name = match.group(1)
            res = self.resolve_abs(name)
            s, e = lines.span(node)
            raw = lines.data[s:e]
            confident = raw.count(name.encode("utf-8", "surrogatepass")) == 1
            if confident:
                off = raw.index(name.encode("utf-8", "surrogatepass"))
                s, e = s + off, s + off + len(name.encode("utf-8", "surrogatepass"))
            lazy, opt, tc = _scope(node)
            result.refs.append(Ref("dash-m", node.lineno, node.col_offset, start=s if confident else None,
                                   end=e if confident else None, text=name,
                                   target=res[0].mid if res else None, rest=".".join(res[1]) if res else "",
                                   lazy=lazy, optional=opt, confident=confident, written=name.split(".")[0],
                                   note="inline"))

    def _mention(self, node, info, lines, result, docstring, fstring=False):
        value = node.value
        if "mdb" not in value:
            return
        for match in DOTTED_RE.finditer(value):
            name = match.group(1)
            if "-m " + name in value or "-m\t" + name in value:
                continue
            res = self.resolve_abs(name)
            if not res:
                continue
            result.mentions.append({"line": node.lineno, "text": name, "target": res[0].mid,
                                    "exact": value.strip() == name, "docstring": docstring, "fstring": fstring,
                                    "context": lines.line_text(node.lineno).strip()[:160]})
        for match in PKGPATH_RE_B.finditer(value.encode("utf-8", "surrogatepass")):
            path = match.group(1).decode()
            home, _, rel = path.partition("/")
            key = rel[:-3].replace("/", ".")
            if key.endswith(".__init__"):
                key = key[:-9]
            info2, _ = self.layout.lookup(home, key, self.strict)
            if info2:
                result.mentions.append({"line": node.lineno, "text": path, "target": info2.mid, "exact":
                                        value.strip() == path, "docstring": docstring, "fstring": fstring,
                                        "pkgpath": True, "context": lines.line_text(node.lineno).strip()[:160]})

    def _path_ref(self, node, info, lines, result):
        value = node.value
        if not value or len(value) > 300 or "\n" in value:
            return
        first = value.split("/")[0]
        run_path = False
        parent = getattr(node, "_sl_parent", None)
        for _ in range(6):
            if parent is None:
                break
            if func_tail(parent) == "run_path":
                run_path = True
                break
            parent = getattr(parent, "_sl_parent", None)
        anchor = self._path_anchor(node, lines)
        if ((first in PACKAGE_DIRS and first != "src" and "/" in value) or value in PACKAGE_DIRS) and \
                _path_context(node):
            result.path_refs.append({"line": node.lineno, "text": value, "kind": "package-dir", "dir": first,
                                     "run_path": run_path, **anchor})
            if value.endswith(".py") and first in {"qualification", "tests"}:
                result.path_refs[-1]["file"] = value
            return
        if value.startswith("src/") or "/src/mdb" in value:
            result.path_refs.append({"line": node.lineno, "text": value, "kind": "src-path", "dir": "src",
                                     "run_path": run_path, **anchor})
            return
        if value.endswith(".py") and "/" not in value and " " not in value:
            candidate = info.path.parent / value
            if candidate.exists():
                result.path_refs.append({"line": node.lineno, "text": value, "kind": "sibling-file",
                                         "file": value, "run_path": run_path, **anchor})
            return
        if value.endswith(".py") and "/" in value and " " not in value:
            candidate = info.path.parent / value
            if candidate.exists():
                result.path_refs.append({"line": node.lineno, "text": value, "kind": "relative-file",
                                         "file": value, "run_path": run_path, **anchor})

    def _path_anchor(self, node, lines):
        """For ``BASE / "text"``: how many parents above ``__file__`` BASE is (None when BASE is not
        anchored at the file), BASE's span, and whether the path feeds sys.path or PYTHONPATH."""
        parent = getattr(node, "_sl_parent", None)
        if not (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div) and parent.right is node):
            return {"anchor_up": None}
        base = parent.left
        up = _anchor(base, self._assigns or {})
        out = {"anchor_up": up, "base": list(lines.span(base)), "expr": list(lines.span(parent)),
               "base_atom": isinstance(base, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)), "ctx": None}
        wrapper = getattr(parent, "_sl_parent", None)
        if isinstance(wrapper, ast.Call) and func_tail(wrapper) == "str" and len(wrapper.args) == 1 and \
                wrapper.args[0] is parent:
            holder = getattr(wrapper, "_sl_parent", None)
            if isinstance(holder, ast.Call) and call_name(holder) == "sys.path.insert" and len(holder.args) == 2 \
                    and holder.args[1] is wrapper and isinstance(getattr(holder, "_sl_parent", None), ast.Expr):
                out.update(ctx="sys.path", ctx_span=list(lines.span(holder)),
                           ctx_index=lines.text(*lines.span(holder.args[0])))
            elif isinstance(holder, ast.Dict) and any(v is wrapper and _str_const(k) == "PYTHONPATH"
                                                       for k, v in zip(holder.keys, holder.values)):
                out.update(ctx="pythonpath", ctx_span=list(lines.span(wrapper)))
            elif isinstance(holder, ast.Assign) and holder.value is wrapper and any(
                    isinstance(t, ast.Subscript) and _str_const(t.slice) == "PYTHONPATH" for t in holder.targets):
                out.update(ctx="pythonpath", ctx_span=list(lines.span(wrapper)))
        return out

    def _source_bound(self, tree, info, result):
        seen = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            tail = func_tail(node)
            what = None
            if tail in {"getsource", "getsourcelines", "getsourcefile", "getfile"}:
                what = "inspect." + tail + "(" + (ast.unparse(node.args[0])[:80] if node.args else "") + ")"
            elif tail in {"read_bytes", "read_text"} and isinstance(node.func, ast.Attribute) and \
                    _mentions_file(node.func.value):
                what = ast.unparse(node)[:120]
            elif HASHY.search(tail or "") and any(_mentions_file(a) for a in node.args):
                what = ast.unparse(node)[:120]
            elif HASHY.search(tail or "") and any(_mentions_names(a) for a in node.args):
                what = "name-hash: " + ast.unparse(node)[:120]
            if what and node.lineno not in seen:
                seen.add(node.lineno)
                result.source_bound.append({"line": node.lineno, "what": what})


def _module_assigns(tree):
    """Module-level names bound exactly once by a plain assignment -> value node."""
    seen, out = defaultdict(int), {}
    for stmt in tree.body:
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets = [stmt.target]
        for target in targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    seen[n.id] += 1
                    if isinstance(target, ast.Name):
                        out[n.id] = stmt.value
    return {k: v for k, v in out.items() if seen[k] == 1}


def _enclosing_function(node):
    parent = getattr(node, "_sl_parent", None)
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parent = getattr(parent, "_sl_parent", None)
    return parent


def _local_assign(name_node):
    """How ``name_node.id`` is bound in its enclosing function (nested defs excluded).

    Returns ("module", None) outside functions, ("param", None), ("one", value), ("many", None)
    or ("none", None) when the function does not bind it.
    """
    func = _enclosing_function(name_node)
    if func is None:
        return "module", None
    if any(isinstance(a, ast.arg) and a.arg == name_node.id for a in ast.walk(func.args)):
        return "param", None
    found = []
    stack = list(func.body)
    while stack:
        stmt = stack.pop()
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name_node.id
                                                for t in stmt.targets):
            found.append(stmt.value)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and \
                stmt.target.id == name_node.id and stmt.value is not None:
            found.append(stmt.value)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.NamedExpr)) and any(
                isinstance(n, ast.Name) and n.id == name_node.id and isinstance(n.ctx, ast.Store)
                for n in ast.walk(getattr(stmt, "target", None) or ast.Tuple(elts=[], ctx=ast.Load()))):
            found.extend([None, None])
        stack.extend(ast.iter_child_nodes(stmt))
    if not found:
        return "none", None
    return ("one", found[0]) if len(found) == 1 else ("many", None)


def _anchor(expr, assigns, depth=0):
    """Number of parent steps above ``__file__`` that ``expr`` denotes, or None if not file-anchored.

    0 = the file itself, 1 = its directory, 2 = the directory above, ...
    """
    if depth > 12 or expr is None:
        return None
    if isinstance(expr, ast.Call):
        tail = func_tail(expr)
        if tail in {"resolve", "absolute"} and isinstance(expr.func, ast.Attribute) and not expr.args:
            return _anchor(expr.func.value, assigns, depth + 1)
        if tail in {"Path", "PurePath", "abspath", "realpath"} and len(expr.args) == 1:
            return _anchor(expr.args[0], assigns, depth + 1)
        if tail == "dirname" and len(expr.args) == 1:
            inner = _anchor(expr.args[0], assigns, depth + 1)
            return None if inner is None else inner + 1
        return None
    if isinstance(expr, ast.Name):
        if expr.id == "__file__":
            return 0
        status, local = _local_assign(expr)
        if status == "one":
            return _anchor(local, assigns, depth + 1)
        if status in {"module", "none"} and expr.id in assigns:
            return _anchor(assigns[expr.id], assigns, depth + 1)
        return None
    if isinstance(expr, ast.Attribute) and expr.attr == "parent":
        inner = _anchor(expr.value, assigns, depth + 1)
        return None if inner is None else inner + 1
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Attribute) and expr.value.attr == "parents" \
            and isinstance(expr.slice, ast.Constant) and isinstance(expr.slice.value, int):
        inner = _anchor(expr.value.value, assigns, depth + 1)
        return None if inner is None else inner + expr.slice.value + 1
    return None


def _binding_values(name_node, tree):
    """Statically enumerate the string values ``name_node`` can take.

    Returns (values or None, binding node, literal element nodes or None). Handled bindings: a for
    loop / comprehension over a literal tuple or list (inline or a module-level name), and a
    function parameter whose every call site in the module passes a string literal.
    """
    if tree is None:
        return None, None, None
    name = name_node.id
    seqs = {}
    for k, v in _module_assigns(tree).items():
        if isinstance(v, (ast.Tuple, ast.List)) and v.elts and all(_str_const(x) is not None for x in v.elts):
            seqs[k] = v.elts

    def literal_elts(it):
        if isinstance(it, (ast.Tuple, ast.List)) and it.elts and all(_str_const(x) is not None for x in it.elts):
            return it.elts
        if isinstance(it, ast.Name) and it.id in seqs:
            return seqs[it.id]
        return None

    child, parent = name_node, getattr(name_node, "_sl_parent", None)
    while parent is not None:
        if isinstance(parent, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in parent.generators:
                if isinstance(gen.target, ast.Name) and gen.target.id == name:
                    elts = literal_elts(gen.iter)
                    return ([_str_const(x) for x in elts] if elts else None), gen, elts
        if isinstance(parent, (ast.For, ast.AsyncFor)) and child in parent.body and \
                isinstance(parent.target, ast.Name) and parent.target.id == name:
            elts = literal_elts(parent.iter)
            return ([_str_const(x) for x in elts] if elts else None), parent, elts
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in parent.args.posonlyargs + parent.args.args]
            if name in params:
                index = params.index(name)
                values = []
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id == parent.name]
                for call in calls:
                    arg = call.args[index] if index < len(call.args) else next(
                        (k.value for k in call.keywords if k.arg == name), None)
                    if _str_const(arg) is None:
                        return None, parent, None
                    values.append(_str_const(arg))
                return (sorted(set(values)) if values else None), parent, None
            if any(a.arg == name for a in parent.args.kwonlyargs):
                return None, parent, None
        child, parent = parent, getattr(parent, "_sl_parent", None)
    return None, None, None


def _path_context(node):
    parent = getattr(node, "_sl_parent", None)
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div):
        return True
    if isinstance(parent, ast.Call) and func_tail(parent) in {"Path", "PurePath", "PosixPath", "joinpath",
                                                               "with_name", "join", "run_path", "open"}:
        return True
    return False


def _mentions_file(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == "__file__":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "__file__":
            return True
    return False


def _mentions_names(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in {"__module__", "__qualname__"}:
            return True
        if isinstance(n, ast.Name) and n.id == "__name__":
            return True
    return False


def _own_bindings(scope):
    """Statements binding names in ``scope`` itself (not inside nested defs, classes or lambdas)."""
    stack = list(scope.body)
    while stack:
        stmt = stack.pop()
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        for field_name in ("body", "orelse", "finalbody", "handlers"):
            stack.extend(getattr(stmt, field_name, []) or [])


def _resolve_local_def(node, name, depth=0):
    """Qualname of the def/class bound to ``name`` where ``node`` sits (innermost scope first).

    A name bound to ``SomeClass(...)`` resolves to that class: saturn_pub digests a callable
    instance by ``type(impl).__module__`` and ``__qualname__``.
    """
    parent = getattr(node, "_sl_parent", None)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            for stmt in _own_bindings(parent):
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and stmt.name == name:
                    return _qualname(stmt)
                if isinstance(stmt, ast.Assign) and depth < 3 and \
                        any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets) and \
                        isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Name):
                        inner = _resolve_local_def(stmt, func.id, depth + 1)
                        if inner:
                            return inner.split(" ")[0] + " (instance; digest binds type(impl).__module__/__qualname__)"
                    elif isinstance(func, ast.Attribute) and dotted(func):
                        return "attr:" + dotted(func)
        parent = getattr(parent, "_sl_parent", None)
    return None


def _qualname(node):
    parts = [node.name]
    parent = getattr(node, "_sl_parent", None)
    while parent is not None and not isinstance(parent, ast.Module):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append("<locals>")
            parts.append(parent.name)
        elif isinstance(parent, ast.ClassDef):
            parts.append(parent.name)
        parent = getattr(parent, "_sl_parent", None)
    return ".".join(reversed(parts))


def load_cuts(path: Path | None):
    """Lines ``importer[:line] imported`` (``#`` comments allowed); importer is relative to the
    package root (``src/mdb/cli.py``) or to packages/ (``mdb/src/mdb/cli.py``)."""
    if not path:
        return []
    out = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where, _, module = line.partition(" ")
        module = module.strip()
        file, _, number = where.rpartition(":") if where.rpartition(":")[2].isdigit() else (where, "", "")
        if not module:
            raise SystemExit(f"--assume-cut: cannot parse {raw!r} (expected 'importer:line module')")
        out.append({"spec": line, "file": file, "line": int(number) if number else None, "module": module})
    return out


def load_keep(path: Path | None):
    if path is None or not path.exists():
        return []
    keep = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            keep.append(line)
    return keep


# ---------------------------------------------------------------- analysis

class Analysis:
    def __init__(self, package_root: Path, keep_file: Path | None = None, strict: bool = False,
                 assume_cut: Path | None = None):
        started = time.time()
        self.layout = layout = Layout(package_root)
        self.keep_file = keep_file
        self.keep = load_keep(keep_file)
        self.assume_cut_file = assume_cut
        self.cut_specs = load_cuts(assume_cut)
        self.cut_matched, self.cut_unmatched = [], []
        self.scanner = Scanner(layout, strict=strict)
        init = layout.modules.get(mid(ENGINE, ""))
        self.scans: dict[Path, Scan] = {}
        if init is not None:
            self.scans[init.path] = self.scanner.scan(init)
        for info in layout.files:
            if info.path not in self.scans:
                self.scans[info.path] = self.scanner.scan(info)
        self.exports = dict(self.scanner.exports)
        self._closure()
        self._classify_tests()
        self._orphans()
        self.elapsed = round(time.time() - started, 2)

    # -- closure
    def module_edges(self):
        """Edges (src mid -> dst mid) with kind/line/lazy for every src module."""
        edges = defaultdict(list)
        for info in self.layout.modules.values():
            scan = self.scans[info.path]
            for ref in scan.refs:
                for target, kind, extra in self._ref_targets(ref):
                    if target and target != info.mid:
                        edges[info.mid].append({"to": target, "kind": kind, "line": ref.line,
                                                "lazy": ref.lazy, "optional": ref.optional,
                                                "type_checking": ref.type_checking, "text": extra})
            for pref in scan.path_refs:
                if pref["kind"] in {"sibling-file", "relative-file"}:
                    target = self.layout.by_resolved.get((info.path.parent / pref["file"]).resolve())
                    if target is not None and target.area == "src" and target.mid != info.mid:
                        edges[info.mid].append({"to": target.mid, "kind": "path-file", "line": pref["line"],
                                                "lazy": False, "optional": False, "type_checking": False,
                                                "text": "path " + pref["text"]})
            parts = info.key.split(".") if info.key else []
            for i in range(len(parts) - 1, -1, -1):
                parent_key = ".".join(parts[:i])
                parent, _ = self.layout.lookup(info.home, parent_key, True)
                if parent is not None and parent.mid != info.mid:
                    edges[info.mid].append({"to": parent.mid, "kind": "parent-package", "line": 0,
                                            "lazy": False, "optional": False, "type_checking": False,
                                            "text": "package " + full_name(info.home, parent_key)})
                    break
        return edges

    def _ref_targets(self, ref):
        """Yield (target mid, kind, text) for every module a ref binds."""
        if ref.kind == "from-import":
            for name in ref.names:
                if name["module"]:
                    yield name["target"], "from-import", ref.text + " import " + name["name"]
                elif name.get("kind") == "export":
                    yield name["export_target"], "lazy-export-attr", ref.text + " import " + name["name"]
                else:
                    yield name["target"], "from-import", ref.text + " import " + name["name"]
        else:
            yield ref.target, ref.kind, ref.text

    def _apply_cuts(self):
        """Drop edges listed in --assume-cut (``importer[:line] imported``); unmatched specs are errors."""
        if not self.cut_specs:
            return
        by_file = {}
        for info in self.layout.modules.values():
            rel = info.label.split("/", 1)[1]  # strip the distribution directory
            by_file[info.label] = by_file[rel] = info.mid
        for spec in self.cut_specs:
            src = by_file.get(spec["file"])
            if src is None:
                self.cut_unmatched.append({**spec, "why": "importer is not a module of this tree"})
                continue
            res = self.layout.resolve_dotted(spec["module"], strict=True)
            target = res[0].mid if res else None
            kept, dropped = [], []
            for edge in self.edges.get(src, []):
                hit = (edge["to"] == target or edge["text"].split(" import ")[0] == spec["module"]) and \
                    (spec["line"] is None or edge["line"] == spec["line"])
                (dropped if hit else kept).append(edge)
            if not dropped:
                self.cut_unmatched.append({**spec, "why": "no edge at that line to that module"})
                continue
            self.edges[src] = kept
            self.cut_matched.append({**spec, "from": src, "edges": len(dropped),
                                     "to": sorted({e["to"] for e in dropped})})

    def _closure(self):
        layout = self.layout
        self.edges = self.module_edges()
        self._apply_cuts()
        roots = {}
        for info in layout.modules.values():
            if info.home != ENGINE:
                continue
            if info.key in ENGINE_TOP_MODULES or info.key.split(".")[0] in ENGINE_SUBPACKAGES:
                roots[info.mid] = "root"
        self.keep_unresolved = []
        for name in self.keep:
            res = layout.resolve_dotted(name if name.startswith(HOMES) else ENGINE + "." + name, strict=True,
                                        require_full=True)
            if res and res[0].home == ENGINE:
                roots[res[0].mid] = "keep"
            else:
                self.keep_unresolved.append(name)
        self.roots = roots
        self.engine_errors = []
        self.closure, self.pred = self._bfs(set(roots))
        self.side = {}
        for info in layout.modules.values():
            self.side[info.mid] = "engine" if info.mid in self.closure else "lanes"
        self.dest_home = {m: (ENGINE if s == "engine" else LANES) for m, s in self.side.items()}
        for src in self.closure:
            for edge in self.edges.get(src, ()):
                if edge["kind"] in CLOSURE_KINDS and edge["to"].startswith(LANES + ":"):
                    self.engine_errors.append({"from": src, **edge})

    def _bfs(self, roots, skip_edge=None, skip_node=None, blocked=frozenset()):
        seen = set(roots)
        pred = {r: None for r in roots}
        queue = deque(sorted(roots))
        while queue:
            node = queue.popleft()
            for edge in self.edges.get(node, ()):
                to = edge["to"]
                if edge["kind"] not in CLOSURE_KINDS or to in seen or to.startswith(LANES + ":"):
                    continue
                if skip_edge is not None and (node, to) == skip_edge:
                    continue
                if to == skip_node or to in blocked:
                    continue
                seen.add(to)
                pred[to] = (node, edge)
                queue.append(to)
        return seen, pred

    def strict_violations(self):
        """Engine-closure modules outside the six subpackages, top modules and keep list, each with
        the edge chain from a root that pulls it in."""
        out = []
        for m in sorted(self.closure):
            if m in self.roots:
                continue
            steps = self.chain(m)
            out.append({"module": m, "lines": self.layout.modules[m].lines,
                        "via": [f"{self.layout.modules[s['from']].label}:{s['line']} -> {s['to']} ({s['kind']})"
                                for s in steps]})
        return out

    def chain(self, target):
        steps = []
        node = target
        while self.pred.get(node) is not None:
            parent, edge = self.pred[node]
            steps.append({"from": parent, "to": node, "line": edge["line"], "kind": edge["kind"],
                          "lazy": edge["lazy"], "text": edge["text"]})
            node = parent
        return list(reversed(steps))

    def lines_of(self, mids):
        return sum(self.layout.modules[m].lines for m in mids)

    def edges_to_cut(self):
        closure_only = {m for m in self.closure if m not in self.roots}
        pairs = defaultdict(list)
        for src in sorted(self.closure):
            for edge in self.edges.get(src, ()):
                if edge["kind"] in CLOSURE_KINDS and edge["to"] in closure_only:
                    pairs[(src, edge["to"])].append(edge)
        ranked = []
        for (src, to), group in pairs.items():
            remaining, _ = self._bfs(set(self.roots), skip_edge=(src, to))
            dragged = sorted(self.closure - remaining)
            ranked.append({
                "from": src, "to": to,
                "file": self.layout.modules[src].label,
                "lines_at": sorted({e["line"] for e in group}),
                "kinds": sorted({e["kind"] for e in group}),
                "top_level": any(not e["lazy"] and not e["type_checking"] for e in group),
                "all_lazy": all(e["lazy"] for e in group),
                "type_checking_only": all(e["type_checking"] for e in group),
                "texts": sorted({e["text"] for e in group})[:4],
                "drag_modules": len(dragged), "drag_lines": self.lines_of(dragged),
                "dragged": dragged})
        ranked.sort(key=lambda r: (-r["drag_lines"], -r["drag_modules"], r["from"], r["to"]))
        return ranked

    def boundary(self):
        """Edges from root modules into flat (non-root) modules, with what each reaches on its own.

        ``reach`` is everything the target pulls in by itself; ``exclusive`` is what leaves the
        engine if that one edge is cut while every other edge stays.
        """
        out = []
        pairs = defaultdict(list)
        for src in sorted(self.roots):
            for edge in self.edges.get(src, ()):
                if edge["kind"] in CLOSURE_KINDS and edge["to"] in self.closure and edge["to"] not in self.roots:
                    pairs[(src, edge["to"])].append(edge)
        for (src, to), group in pairs.items():
            reach, _ = self._bfs({to}, blocked=set(self.roots))
            reach = {m for m in reach if m not in self.roots}
            remaining, _ = self._bfs(set(self.roots), skip_edge=(src, to))
            exclusive = self.closure - remaining
            out.append({"from": src, "to": to, "file": self.layout.modules[src].label,
                        "lines_at": sorted({e["line"] for e in group}),
                        "kinds": sorted({e["kind"] for e in group}),
                        "top_level": any(not e["lazy"] and not e["type_checking"] for e in group),
                        "all_lazy": all(e["lazy"] for e in group),
                        "reach_modules": len(reach), "reach_lines": self.lines_of(reach),
                        "exclusive_modules": len(exclusive), "exclusive_lines": self.lines_of(exclusive)})
        out.sort(key=lambda r: (-r["reach_lines"], r["from"], r["to"]))
        return out

    def components(self):
        """Strongly connected components among flat engine modules (roots excluded), largest first."""
        nodes = sorted(m for m in self.closure if m not in self.roots)
        node_set = set(nodes)
        succ = {n: sorted({e["to"] for e in self.edges.get(n, ()) if e["kind"] in CLOSURE_KINDS and
                           e["to"] in node_set}) for n in nodes}
        index, low, on_stack, stack, out = {}, {}, set(), [], []
        counter = [0]
        for start in nodes:
            if start in index:
                continue
            work = [(start, 0)]
            while work:
                node, i = work.pop()
                if i == 0:
                    index[node] = low[node] = counter[0]
                    counter[0] += 1
                    stack.append(node)
                    on_stack.add(node)
                recurse = False
                for j in range(i, len(succ[node])):
                    nxt = succ[node][j]
                    if nxt not in index:
                        work.append((node, j + 1))
                        work.append((nxt, 0))
                        recurse = True
                        break
                    if nxt in on_stack:
                        low[node] = min(low[node], index[nxt])
                if recurse:
                    continue
                if low[node] == index[node]:
                    comp = []
                    while True:
                        item = stack.pop()
                        on_stack.discard(item)
                        comp.append(item)
                        if item == node:
                            break
                    out.append(sorted(comp))
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
        out = [c for c in out if len(c) > 1]
        out.sort(key=lambda c: -self.lines_of(c))
        return [{"modules": len(c), "lines": self.lines_of(c), "members": c} for c in out]

    def cut_all_boundary(self):
        """Closure if every edge from a root into a flat module were removed (the full cut set)."""
        saved = self.edges
        self.edges = {src: [e for e in es if not (src in self.roots and e["to"] not in self.roots)]
                      for src, es in saved.items()}
        try:
            remaining, _ = self._bfs(set(self.roots))
        finally:
            self.edges = saved
        return sorted(remaining - set(self.roots))

    def per_root_reach(self):
        out = []
        for root in sorted(self.roots):
            reach, _ = self._bfs({root}, blocked=set(self.roots) - {root})
            flat = {m for m in reach if m not in self.roots}
            out.append({"root": root, "flat_modules": len(flat), "flat_lines": self.lines_of(flat)})
        out.sort(key=lambda r: -r["flat_lines"])
        return out

    def module_cuts(self):
        out = []
        for m in sorted(self.closure):
            if m in self.roots:
                continue
            remaining, _ = self._bfs(set(self.roots), skip_node=m)
            dragged = sorted(self.closure - remaining)
            importers = sorted({src for src in self.closure for e in self.edges.get(src, ())
                                if e["to"] == m and e["kind"] in CLOSURE_KINDS})
            out.append({"module": m, "drag_modules": len(dragged), "drag_lines": self.lines_of(dragged),
                        "engine_importers": importers, "dragged": dragged})
        out.sort(key=lambda r: (-r["drag_lines"], r["module"]))
        return out

    # -- tests
    def _classify_tests(self):
        layout = self.layout
        tests = [f for f in layout.files if f.area == "tests"]
        by_label = {f.label: f for f in tests}
        own = {}
        helper_links = defaultdict(set)
        for info in tests:
            scan = self.scans[info.path]
            deps = {"engine": set(), "lanes_hard": set(), "lanes_optional": set(), "function_local_only": True,
                    "reasons": []}
            for ref in scan.refs:
                if ref.kind not in CLASSIFY_KINDS and ref.kind != "from-import":
                    continue
                for target, kind, text in self._ref_targets(ref):
                    if not target:
                        continue
                    home, key = split_mid(target)
                    if key == "" and kind in {"import", "from-import"}:
                        continue
                    side = self.side.get(target, "lanes")
                    if side == "engine":
                        deps["engine"].add(target)
                    elif ref.optional or kind == "importorskip":
                        deps["lanes_optional"].add(target)
                    else:
                        deps["lanes_hard"].add(target)
                        deps["reasons"].append(f"{kind}:{ref.line}:{text}")
                        if not ref.lazy:
                            deps["function_local_only"] = False
            for pref in scan.path_refs:
                if pref.get("dir") == "qualification":
                    deps["lanes_hard"].add("qualification")
                    deps["reasons"].append(f"path:{pref['line']}:{pref['text']}")
                    deps["function_local_only"] = False
                elif pref["kind"] == "src-path":
                    for m in self._src_path_targets(pref["text"]):
                        (deps["engine"] if self.side.get(m) == "engine" else deps["lanes_hard"]).add(m)
            for sib in scan.siblings:
                target = info.path.parent / sib["target"]
                for f in tests:
                    if f.path == target or (sib["target"].endswith("__init__.py") and f.path == target):
                        helper_links[info.label].add(f.label)
            for pref in scan.path_refs:
                if pref["kind"] in {"sibling-file", "relative-file"}:
                    target = (info.path.parent / pref["file"]).resolve()
                    for f in tests:
                        if f.path.resolve() == target:
                            helper_links[info.label].add(f.label)
            own[info.label] = deps
        # Transitive helper dependencies.
        effective = {}
        for label in own:
            seen, stack = {label}, [label]
            while stack:
                for nxt in helper_links.get(stack.pop(), ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            merged = {"engine": set(), "lanes_hard": set(), "lanes_optional": set(), "function_local_only": True,
                      "reasons": [], "helpers": sorted(seen - {label})}
            for item in seen:
                d = own[item]
                merged["engine"] |= d["engine"]
                merged["lanes_hard"] |= d["lanes_hard"]
                merged["lanes_optional"] |= d["lanes_optional"]
                merged["function_local_only"] &= d["function_local_only"] or not d["lanes_hard"]
                merged["reasons"] += [(item + ": " + r) if item != label else r for r in d["reasons"]]
            effective[label] = merged
        self.tests = {}
        for label, deps in effective.items():
            info = by_label[label]
            name = Path(info.rel).name
            if info.root == LANES_DIST:
                side, why = "lanes", "already in mdb-lanes (lanes is sticky)"
            elif not deps["lanes_hard"] and not deps["lanes_optional"]:
                side, why = "engine", "no lane references"
            elif not deps["lanes_hard"]:
                side, why = "engine", "lane references are all optional"
            elif not deps["engine"]:
                side, why = "lanes", "references resolve only to lanes"
            else:
                side, why = "straddling", "hard references to both sides"
            self.tests[label] = {"side": side, "why": why, "file": info, "name": name,
                                 "engine_refs": sorted(deps["engine"]), "lane_refs": sorted(deps["lanes_hard"]),
                                 "lane_optional": sorted(deps["lanes_optional"]),
                                 "lane_refs_function_local_only": deps["function_local_only"]
                                 if deps["lanes_hard"] else None,
                                 "helpers": deps["helpers"], "reasons": deps["reasons"][:12]}
        # Placement: straddling defaults to lanes (lanes may import mdb.*); helpers follow importers.
        placement = {}
        for label, rec in self.tests.items():
            placement[label] = {"lanes"} if rec["side"] in {"lanes", "straddling"} else {"engine"}
        importers = defaultdict(set)
        for label, rec in self.tests.items():
            for helper in rec["helpers"]:
                importers[helper].add(label)
        for _ in range(4):
            for label, rec in self.tests.items():
                if rec["name"] == "conftest.py":
                    placement[label] = {"engine", "lanes"}
                    rec["shared"] = "conftest"
                    continue
                needed = set().union(*(placement[i] for i in importers[label])) if importers[label] else set()
                if not needed:
                    continue
                if rec["name"].startswith("test_") or rec["file"].root == LANES_DIST:
                    placement[label] = placement[label] | needed
                else:
                    placement[label] = needed
                if len(placement[label]) > 1:
                    rec["shared"] = "needed by both sides"
        for label, rec in self.tests.items():
            rec["placement"] = sorted(placement[label])

    def _src_path_targets(self, text):
        out = []
        match = re.search(r"src/(mdb(?:_lanes)?)/([\w/]+)\.py", text)
        if match:
            key = match.group(2).replace("/", ".")
            if key.endswith("__init__"):
                key = key[:-len("__init__")].rstrip(".")
            info, _ = self.layout.lookup(match.group(1), key)
            if info:
                out.append(info.mid)
        return out

    def _orphans(self):
        referenced = set()
        for scan in self.scans.values():
            for ref in scan.refs:
                for target, _, _ in self._ref_targets(ref):
                    if target and (scan.info.mid != target):
                        referenced.add(target)
            for mention in scan.mentions:
                referenced.add(mention["target"])
            for pref in scan.path_refs:
                if pref["kind"] == "src-path":
                    referenced.update(self._src_path_targets(pref["text"]))
        for info in self.layout.modules.values():
            for scan_ref in self.scans[info.path].path_refs:
                if scan_ref["kind"] in {"sibling-file", "relative-file"}:
                    target = (info.path.parent / scan_ref["file"]).resolve()
                    for other in self.layout.modules.values():
                        if other.path.resolve() == target:
                            referenced.add(other.mid)
        self.orphans = sorted(m for m, info in self.layout.modules.items()
                              if m not in referenced and info.key not in ENGINE_TOP_MODULES and
                              not info.is_package and m not in self.roots)

    # -- destinations
    def file_dest(self, info: FileInfo):
        """List of (dest root dir name, dest area, dest rel, dest home) for a source file."""
        if info.area == "src":
            home = self.dest_home[info.mid]
            return [(self.root_for(home), "src", info.rel, home)]
        if info.area == "qualification":
            return [(LANES_DIST, "qualification", info.rel, LANES)]
        if info.area == "tests":
            rec = self.tests[info.label]
            return [(self.root_for(ENGINE if p == "engine" else LANES), "tests", info.rel,
                     ENGINE if p == "engine" else LANES) for p in rec["placement"]]
        return [(self.layout.engine_root.name, info.area, info.rel, None)]

    def root_for(self, home):
        return self.layout.engine_root.name if home == ENGINE else LANES_DIST

    def dest_path(self, root, area, rel, home=None):
        if area == "src":
            return f"{root}/src/{home}/{rel}"
        return f"{root}/{area}/{rel}"

    # -- summary
    def to_json(self):
        layout = self.layout
        modules = {}
        for m, info in sorted(layout.modules.items()):
            modules[m] = {"side": self.side[m], "path": info.label, "lines": info.lines,
                          "reason": self.roots.get(m) or ("closure" if m in self.closure else "not in closure"),
                          "importers": sorted({src for src, es in self.edges.items() for e in es
                                               if e["to"] == m and e["kind"] != "parent-package"})}
            if m in self.closure and m not in self.roots:
                steps = self.chain(m)
                modules[m]["chain"] = steps
                modules[m]["importing_line"] = steps[-1] if steps else None
        engine = [m for m in modules if modules[m]["side"] == "engine"]
        lanes = [m for m in modules if modules[m]["side"] == "lanes"]
        tests = {label: {k: v for k, v in rec.items() if k != "file"} for label, rec in sorted(self.tests.items())}
        straddling = sorted(label for label, rec in self.tests.items() if rec["side"] == "straddling")
        dynamic, unconfident = [], []
        for path, scan in self.scans.items():
            for d in scan.dynamic:
                dynamic.append({"file": scan.info.label, **d})
            for ref in scan.refs:
                if not ref.confident:
                    unconfident.append({"file": scan.info.label, **ref.to_json()})
        unresolved = []
        for path, scan in self.scans.items():
            for ref in scan.refs:
                for target, kind, text in self._ref_targets(ref):
                    if target is None and kind not in {"registry", "lazy-table"} and \
                            (text.split(".")[0] in HOMES or ref.written == "relative"):
                        unresolved.append({"file": scan.info.label, "line": ref.line, "kind": kind, "text": text})
                    elif target is None and kind in {"registry", "lazy-table"}:
                        unresolved.append({"file": scan.info.label, "line": ref.line, "kind": kind, "text": text,
                                           "note": ref.note})
        syntax = [{"file": s.info.label, "error": s.syntax_error} for s in self.scans.values() if s.syntax_error]
        registry = []
        for path, scan in self.scans.items():
            for ref in scan.refs:
                if ref.kind == "registry":
                    registry.append({"file": scan.info.label, "line": ref.line, "entry": ref.note,
                                     "module": ref.target, "side": self.side.get(ref.target)})
        return {
            "schema": "mdb-split-lanes-analysis-v1",
            "package_root": str(layout.engine_root),
            "lanes_root_exists": layout.lanes_root.exists(),
            "elapsed_s": self.elapsed,
            "engine_subpackages_present": {name: (mid(ENGINE, name) in layout.modules or
                                                  (layout.area_root(ENGINE, "src") / name).exists())
                                           for name in ENGINE_SUBPACKAGES},
            "keep": self.keep, "keep_unresolved": self.keep_unresolved,
            "assume_cut_file": str(self.assume_cut_file) if self.assume_cut_file else None,
            "assumed_cuts_matched": self.cut_matched, "assumed_cuts_unmatched": self.cut_unmatched,
            "strict_engine_violations": self.strict_violations(),
            "layout_conflicts": layout.conflicts,
            "counts": {
                "modules": len(modules), "engine_modules": len(engine), "lanes_modules": len(lanes),
                "engine_lines": self.lines_of(engine), "lanes_lines": self.lines_of(lanes),
                "roots": len(self.roots), "closure_only": len(self.closure) - len(self.roots),
                "tests_engine": sum(r["side"] == "engine" for r in self.tests.values()),
                "tests_lanes": sum(r["side"] == "lanes" for r in self.tests.values()),
                "tests_straddling": len(straddling),
                "qualification_files": sum(f.area == "qualification" for f in layout.files),
                "qualification_lines": sum(f.lines for f in layout.files if f.area == "qualification"),
            },
            "roots": self.roots,
            "engine_closure": sorted(m for m in self.closure if m not in self.roots),
            "modules": modules,
            "edges_to_cut": self.edges_to_cut(),
            "module_cuts": self.module_cuts(),
            "boundary_edges": self.boundary(),
            "per_root_reach": self.per_root_reach(),
            "cycles": self.components(),
            "closure_after_cutting_all_boundary_edges": self.cut_all_boundary(),
            "root_lines": self.lines_of(self.roots),
            "engine_imports_mdb_lanes": self.engine_errors,
            "tests": tests,
            "straddling_tests": straddling,
            "registry_entries": registry,
            "lazy_exports": self.exports,
            "dynamic_imports": dynamic,
            "unconfident_spans": unconfident,
            "unresolved_references": unresolved,
            "syntax_errors": syntax,
            "orphan_modules": self.orphans,
            "sibling_imports": [{"file": s.info.label, **d} for s in self.scans.values() for d in s.siblings],
            "path_references": [{"file": s.info.label, **d} for s in self.scans.values() for d in s.path_refs],
        }


# ---------------------------------------------------------------- planning

class Plan:
    def __init__(self, analysis: Analysis):
        self.a = analysis
        layout = analysis.layout
        self.moves, self.copies, self.creates = [], [], []
        self.files = []
        self.errors, self.registry, self.unconfident, self.warnings = [], [], [], []
        self.aux_lane_refs, self.optional_engine_test_refs, self.splits = [], [], []
        self.data_files = []
        self.manual, self.rule_fixes, self.path_checks = [], [], []
        self.identity_strings, self.identity_refused = [], []
        labels = {f.label for f in layout.files}
        for info in layout.files:
            dests = analysis.file_dest(info)
            for index, (root, area, rel, home) in enumerate(dests):
                dest = analysis.dest_path(root, area, rel, home)
                if dest != info.label and dest in labels:
                    continue  # an in-place copy already exists and is planned on its own
                if info.area in AUX_AREAS:
                    # Scripts beside the engine (examples/tools/experiments) stay in place and may name
                    # either distribution: their lane references are rewritten like lane code. Vendored
                    # frozen sources are never edited (their lane references stay listed in aux_lane_refs).
                    self._aux(info)
                    if "vendor" in info.rel.split("/"):
                        self.files.append(self._file_record(info, dest, None, []))
                        continue
                    edits = self._edits(info, LANES, dest)
                    self.files.append(self._file_record(info, dest, None, edits))
                    continue
                if dest != info.label:
                    (self.moves if index == 0 and not self._exists_at(dest) else self.copies).append(
                        {"from": info.label, "to": dest, "area": info.area})
                edits = self._edits(info, home, dest)
                self.files.append(self._file_record(info, dest, home, edits))
        self.path_breaks = [c for c in self.path_checks if not c["ok"]]
        self._package_data()
        self._generated()
        self.contents = None
        self.repin = None

    # -- rule-based edits beyond import statements
    def _manual(self, info, dest, line, start, why, rule, sub=None, block=None):
        """A fix the tool will not make. ``sub`` = (old, new) applied to the post-split line;
        ``block`` = a literal replacement text when the fix is not a one-line substitution."""
        self.manual.append({"dest": dest, "source": info.label, "source_line": line, "start": start,
                            "why": why, "rule": rule, "sub": list(sub) if sub else None, "block": block})

    def _rule(self, info, dest, rule, line, old, new):
        self.rule_fixes.append({"file": dest, "source": info.label, "line": line, "rule": rule, "old": old,
                                "new": new})

    def _rule_edits(self, info, scan, file_home, dest, paths_only=False, record_only=False):
        a, lines = self.a, scan.lines
        out = []
        engine_side = dest.split("/", 1)[0] != LANES_DIST and info.area not in AUX_AREAS
        if not paths_only:
            for rec in scan.prefix_imports:
                old_expr = lines.text(*rec["expr"])
                if rec["values"] is None:
                    self._manual(info, dest, rec["line"], rec["expr"][0], "dynamic import prefix: the values of "
                                 f"`{rec['var']}` are not statically enumerable; check every module it can name",
                                 "prefix-import")
                    continue
                targets = {}
                for value in rec["values"]:
                    res = a.layout.resolve_dotted(rec["home"] + "." + value, strict=True)
                    if res and not res[1]:
                        targets[value] = a.dest_home.get(res[0].mid, res[0].home)
                    else:
                        targets[value] = None
                homes = set(targets.values())
                if None in homes or len(homes) != 1:
                    mapping = ", ".join(f'"{v}": "{(h or rec["home"]) + "." + v}"' for v, h in sorted(targets.items()))
                    self._manual(info, dest, rec["line"], rec["expr"][0],
                                 "dynamic import prefix names modules on both sides (or unresolved): "
                                 + ", ".join(f"{v}->{h}" for v, h in sorted(targets.items())), "prefix-import",
                                 sub=(old_expr, "{" + mapping + "}[" + rec["var"] + "]"))
                    continue
                new_home = homes.pop()
                if new_home == rec["home"]:
                    continue
                if engine_side and new_home == LANES:
                    self.errors.append({"file": dest, "line": rec["line"], "kind": "prefix-import",
                                        "text": old_expr, "module": rec["values"],
                                        "error": "engine-side file imports lane modules by prefix"})
                    continue
                for s, e, kind, line, col in rec["spans"]:
                    out.append(_edit(lines, s, e, rec["home"], new_home, line, col, "rule-" + kind,
                                     f"enumerated values {rec['values']} all move to {new_home}"))
                    self._rule(info, dest, "prefix-import/" + kind, line, rec["home"], new_home)
            for seal in scan.package_seals:
                self._seal_manual(info, scan, seal, dest, engine_side)
        for pref in scan.path_refs:
            check, edit = self._path_eval(info, scan, pref, dest)
            if check is None:
                continue
            self.path_checks.append(check)
            if edit is not None and not record_only:
                out.append(edit)
        return out

    def _seal_manual(self, info, scan, seal, dest, engine_side):
        lines = scan.lines
        lanes_exist = any(h == LANES for h in self.a.dest_home.values())
        if not lanes_exist:
            return
        text = lines.text(seal["start"], seal["end"])
        if seal["form"] == "table-row":
            self._manual(info, dest, seal["line"], seal["start"],
                         "worker payloads ship only installed `mdb`; lane workers also need `mdb_lanes` sealed and "
                         "shipped (add a row when mdb-lanes is installed)", "payload-distributions",
                         block="add after this row:\n"
                               '                *((("mdb_lanes", "mdb-lanes", version("mdb-lanes")),)\n'
                               '                  if importlib.util.find_spec("mdb_lanes") else ()),')
            return
        if engine_side:
            return
        if seal["form"] == "literal-tuple":
            names = seal["names"]
            if LANES in names:
                return
            new = text.replace(f'"{ENGINE}"', f'"{ENGINE}", "{LANES}"', 1)
            self._manual(info, dest, seal["line"], seal["start"],
                         "lane-side code seals `mdb` packages; lane modules now live in `mdb_lanes`", "package-seal",
                         sub=(text, new) if new != text else None)
            return
        line = lines.line_text(seal["line"])
        m = re.search(r'\(f"mdb/\{(\w+)\}", (\w+)\) for \1, \2 in package_files\("mdb"\)\.items\(\)', line)
        if m:
            old = m.group(0)
            new = (f'(f"{{_package}}/{{{m.group(1)}}}", {m.group(2)}) for _package in ("{ENGINE}", "{LANES}") '
                   f'for {m.group(1)}, {m.group(2)} in package_files(_package).items()')
            self._manual(info, dest, seal["line"], seal["start"],
                         "lane-side payload ships only `mdb` sources; lane modules now live in `mdb_lanes`",
                         "package-seal", sub=(old, new))
            return
        if seal["call"] == "seal_package" and seal["package"] == ENGINE:
            self._manual(info, dest, seal["line"], seal["start"],
                         "lane-side code seals only `mdb`; lane modules now live in `mdb_lanes` (the sealed value's "
                         "shape changes; owner decides)", "package-seal",
                         sub=(text, f'{{_package: seal_package(_package) for _package in ("{ENGINE}", "{LANES}")}}'))
            return
        self._manual(info, dest, seal["line"], seal["start"],
                     f"lane-side code calls {text}; lane modules now live in `mdb_lanes`. No mechanical "
                     "replacement: decide whether this site must also cover `package_files(\"mdb_lanes\")`",
                     "package-seal")

    def _target_root(self, d, new_base):
        """Distribution directory that holds package directory ``d`` after the split."""
        engine = self.a.layout.engine_root.name
        if d == "qualification":
            return LANES_DIST
        if d == "tests":
            return new_base
        if d == "src":
            return None  # both have src/; resolved by context
        return engine

    def _path_eval(self, info, scan, pref, dest):
        """(check record, edit or None) for one file-path reference at ``dest``."""
        import posixpath
        a, layout, lines = self.a, self.a.layout, scan.lines
        engine = layout.engine_root.name
        kind = pref["kind"]
        base = {"referrer": dest, "source": info.label, "line": pref["line"], "text": pref["text"], "kind": kind,
                "run_path": pref.get("run_path", False)}
        if kind in {"sibling-file", "relative-file"}:
            expected = posixpath.normpath(posixpath.join(posixpath.dirname(dest), pref["file"]))
            target = layout.by_resolved.get((info.path.parent / pref["file"]).resolve())
            if target is not None:
                targets = {a.dest_path(*d) for d in a.file_dest(target)}
            else:
                current = (info.path.parent / pref["file"]).resolve()
                try:
                    rel = current.relative_to(layout.packages_dir.resolve()).as_posix()
                except ValueError:
                    rel = None
                targets = {data_dest(rel, engine)} if rel else set()
            ok = expected in targets
            check = {**base, "expected": expected, "target_dests": sorted(targets), "ok": ok}
            if ok:
                return check, None
            if target is not None and target.area == "src" and pref.get("anchor_up") == 1 and pref.get("expr"):
                t_dest = a.dest_home.get(target.mid, target.home)
                module = full_name(t_dest, target.key)
                path_name = "Path" if "Path" in scan.top_names else '__import__("pathlib").Path'
                old = lines.text(*pref["expr"])
                new = f'{path_name}(__import__("importlib.util").util.find_spec("{module}").origin)'
                check["fix"] = "rule"
                self._rule(info, dest, "sibling-module-file", pref["line"], old, new)
                return check, _edit(lines, pref["expr"][0], pref["expr"][1], old, new, pref["line"], 0,
                                    "rule-path", "sibling module file now lives in " + t_dest)
            check["fix"] = "manual"
            self._manual(info, dest, pref["line"], lines.starts[pref["line"] - 1],
                         f"file path `{pref['text']}` no longer lands on its target ({', '.join(sorted(targets))})",
                         "path")
            return check, None
        if kind not in {"package-dir", "src-path"}:
            return None, None
        up = pref.get("anchor_up")
        if up is None:
            self.path_checks.append({**base, "ok": True, "anchored": False})
            return None, None
        old_parts, new_parts = info.label.split("/"), dest.split("/")
        if up >= len(old_parts) or up >= len(new_parts):
            return None, None
        old_base = "/".join(old_parts[:len(old_parts) - up])
        new_base = "/".join(new_parts[:len(new_parts) - up])
        if old_base not in {engine, LANES_DIST}:
            return {**base, "ok": True, "anchored": True, "base": old_base}, None
        d = pref["text"].split("/")[0]
        if old_base == LANES_DIST and old_base == new_base:
            return {**base, "ok": True, "anchored": True, "base": old_base}, None
        target = self._target_root(d, new_base)
        check = {**base, "anchored": True, "base_before": old_base, "base_after": new_base}
        m = re.fullmatch(r"src/(mdb|mdb_lanes)/([\w/]+)\.py", pref["text"])
        if d == "src" and m and pref.get("base_atom") and pref.get("expr"):
            key = m.group(2).replace("/", ".")
            key = key[:-len(".__init__")] if key.endswith(".__init__") else ("" if key == "__init__" else key)
            target_info, _ = layout.lookup(m.group(1), key)
            if target_info is not None:
                t_home = a.dest_home.get(target_info.mid, target_info.home)
                t_root = engine if t_home == ENGINE else LANES_DIST
                new_text = f"src/{t_home}/{m.group(2)}.py"
                if t_root == new_base and new_text == pref["text"]:
                    return {**check, "ok": True}, None
                btext = lines.text(*pref["base"])
                base_new = btext if t_root == new_base else f'{btext}.parent / "{t_root}"'
                old = lines.text(*pref["expr"])
                new = f'{base_new} / "{new_text}"'
                check.update(ok=False, why=f"module source now at {t_root}/{new_text}", fix="rule")
                self._rule(info, dest, "path-src-module", pref["line"], old, new)
                return check, _edit(lines, pref["expr"][0], pref["expr"][1], old, new, pref["line"], 0,
                                    "rule-path", "module source file moved")
        if d == "src":
            if new_base == engine or pref["text"].startswith("src/" + LANES):
                return {**check, "ok": True}, None
            check.update(ok=False, why=f"{LANES_DIST}/src holds {LANES}, not {ENGINE}")
            btext = lines.text(*pref["base"])
            sub_old = lines.text(*pref["expr"])
            engine_src = f'{btext}.parent / "{engine}" / "src"'
            if pref.get("ctx") == "sys.path" and pref.get("base_atom"):
                old = lines.text(*pref["ctx_span"])
                idx = pref["ctx_index"]
                new = f"sys.path[{idx}:{idx}] = [str({sub_old}), str({engine_src})]"
                check["fix"] = "rule"
                self._rule(info, dest, "path-src-sys.path", pref["line"], old, new)
                return check, _edit(lines, pref["ctx_span"][0], pref["ctx_span"][1], old, new, pref["line"], 0,
                                    "rule-path", "lane script needs both src roots")
            if pref.get("ctx") == "pythonpath" and pref.get("base_atom"):
                old = lines.text(*pref["ctx_span"])
                os_name = "os" if "os" in scan.top_names else '__import__("os")'
                new = f"{os_name}.pathsep.join([str({sub_old}), str({engine_src})])"
                check["fix"] = "rule"
                self._rule(info, dest, "path-src-pythonpath", pref["line"], old, new)
                return check, _edit(lines, pref["ctx_span"][0], pref["ctx_span"][1], old, new, pref["line"], 0,
                                    "rule-path", "lane subprocess needs both src roots")
            check["fix"] = "manual"
            plain = pref["text"] == "src"
            self._manual(info, dest, pref["line"], pref["expr"][0],
                         f"`{sub_old}` now names {LANES_DIST}/src ({LANES}); if the code means the engine sources "
                         "use the replacement", "path-src",
                         sub=(sub_old, engine_src) if pref.get("base_atom") and plain else None)
            return check, None
        if target == new_base:
            return {**check, "ok": True}, None
        why = (f"qualification/ moves to {LANES_DIST}" if d == "qualification"
               else f"{LANES_DIST} has no {d}/ directory")
        check.update(ok=False, why=why, target_root=target)
        if pref.get("base_atom") and "/" not in new_base and "/" not in target:
            btext = lines.text(*pref["base"])
            new = f'{btext}.parent / "{target}"'
            check["fix"] = "rule"
            self._rule(info, dest, "path-package-root", pref["line"], btext, new)
            return check, _edit(lines, pref["base"][0], pref["base"][1], btext, new, pref["line"], 0, "rule-path",
                                f"{d}/ lives under {target}")
        check["fix"] = "manual"
        self._manual(info, dest, pref["line"], pref["expr"][0] if pref.get("expr") else lines.starts[pref["line"] - 1],
                     why + "; base expression is not a simple name/attribute", "path")
        return check, None

    def _guard_identity(self, info, dest, edits):
        """Refuse any edit touching a versioned identity string (``mdb.x.v1``)."""
        scan = self.a.scans[info.path]
        for ident in scan.identity_strings:
            self.identity_strings.append({"file": dest, "source": info.label, "line": ident["line"],
                                          "text": ident["text"]})
        kept = []
        for edit in edits:
            hit = next((i for i in scan.identity_strings if edit["start"] < i["end"] and i["start"] < edit["end"]), None)
            if hit:
                self.identity_refused.append({"file": dest, "line": edit["line"], "text": hit["text"],
                                              "edit": edit["kind"]})
                self.errors.append({"file": dest, "line": edit["line"], "kind": "identity-string",
                                    "text": hit["text"], "error": "edit would change a versioned identity string"})
                continue
            kept.append(edit)
        return kept

    def _exists_at(self, dest):
        return (self.a.layout.packages_dir / dest).exists()

    def _file_record(self, info, dest, home, edits):
        data = info.data
        new = apply_edits(data, edits) if edits else data
        return {"source": info.label, "dest": dest, "home": home, "sha_before": info.sha, "sha_after": sha256(new),
                "changed": new != data, "edits": edits}

    def _aux(self, info):
        scan = self.a.scans[info.path]
        for ref in scan.refs:
            for target, kind, text in self.a._ref_targets(ref):
                if target and self.a.dest_home.get(target) == LANES and split_mid(target)[0] == ENGINE:
                    self.aux_lane_refs.append({"file": info.label, "line": ref.line, "kind": kind, "text": text,
                                               "target": target})

    def _generated(self):
        a, layout = self.a, self.a.layout
        lane_keys = {split_mid(m)[1] for m, h in a.dest_home.items() if h == LANES}
        needed = {""}
        for key in lane_keys:
            parts = key.split(".") if key else []
            for i in range(len(parts)):
                needed.add(".".join(parts[:i]))
        for key in sorted(needed):
            exists_src = any(a.dest_home.get(m) == LANES and split_mid(m)[1] == key and
                             layout.modules[m].is_package for m in layout.modules)
            if not exists_src:
                rel = (key.replace(".", "/") + "/" if key else "") + "__init__.py"
                dest = f"{LANES_DIST}/src/{LANES}/{rel}"
                if not (layout.packages_dir / dest).exists():
                    doc = ('"""MDB lane modules split from the mdb engine."""\n' if not key else
                           f'"""Lane modules of mdb.{key}."""\n')
                    self.creates.append({"to": dest, "content": doc, "why": "package marker"})
        pyproject = f"{LANES_DIST}/pyproject.toml"
        if not (layout.packages_dir / pyproject).exists():
            self.creates.append({"to": pyproject, "content": self.lanes_pyproject(), "why": "distribution metadata"})

    def lanes_verbs(self):
        """(verb, "mdb_lanes.module:main") for every cli TOOLS entry whose module moves to lanes."""
        out = []
        for item in self.registry:
            m = re.match(r"TOOLS\[['\"]?([^'\"\]]+)['\"]?\]", item["entry"] or "")
            if not m:
                continue
            home, key = split_mid(item["module"])
            out.append((m.group(1), f"{full_name(self.a.dest_home.get(item['module'], home), key)}:main"))
        return sorted(set(out))

    def lanes_pyproject(self):
        """pyproject.toml for mdb-lanes: depends on mdb-runtime, registers lane verbs in the
        ``mdb.lanes`` entry-point group, mirrors the engine's extras, carries lane package data."""
        engine_py = self.a.layout.engine_root / "pyproject.toml"
        extras = []
        try:
            import tomllib
            data = tomllib.loads(engine_py.read_text()) if engine_py.exists() else {}
            extras = sorted(data.get("project", {}).get("optional-dependencies", {}))
            requires_python = data.get("project", {}).get("requires-python", ">=3.10")
            engine_dist = data.get("project", {}).get("name", "mdb-runtime")
        except Exception:  # noqa: BLE001 - tomllib missing or malformed: fall back to defaults
            requires_python, engine_dist = ">=3.10", "mdb-runtime"
        w = ["[project]", f'name = "{LANES_DIST}"', 'version = "0.1.0"',
             'description = "Experiment-specific lane modules split from the mdb engine."',
             f'requires-python = "{requires_python}"', f'dependencies = ["{engine_dist}"]', ""]
        if extras:
            w.append("[project.optional-dependencies]")
            for extra in extras:
                w.append(f'{extra} = ["{engine_dist}[{extra}]"]')
            w.append("")
        verbs = self.lanes_verbs()
        if verbs:
            w.append(f'[project.entry-points."{LANES_VERB_GROUP}"]')
            for verb, target in verbs:
                w.append(f'{json.dumps(verb)} = "{target}"')
            w.append("")
        w += ["[build-system]", 'requires = ["hatchling>=1.25"]', 'build-backend = "hatchling.build"', "",
              "[tool.hatch.build.targets.wheel]", f'packages = ["src/{LANES}"]']
        data_files = sorted({d["to"].split("/src/", 1)[1] for d in self.data_files
                             if d["to"].startswith(LANES_DIST + "/src/")})
        if data_files:
            # Hatch ships every tracked, non-ignored file under a `packages` directory, so the lane
            # package data below is carried without force-include (which would add a duplicate entry).
            w.append("# Package data carried with src/" + LANES + ": " + ", ".join(data_files))
        w += ["", "[tool.pytest.ini_options]", 'testpaths = ["tests"]', 'pythonpath = ["src"]', ""]
        return "\n".join(w)

    def _package_data(self):
        a = self.a
        for home, path, rel in a.layout.package_data:
            referrers = set()
            name = Path(rel).name
            if home == ENGINE and rel.split("/")[0] in ENGINE_SUBPACKAGES:
                # Data inside an engine subpackage (e.g. serve/static) belongs to that subpackage.
                current = str(path.relative_to(a.layout.packages_dir))
                self.data_files.append({"from": str(path), "to": current, "referrers": [ENGINE], "in_place": True})
                continue
            for info in a.layout.modules.values():
                if name.encode() in info.data:
                    referrers.add(a.dest_home[info.mid])
            dest_homes = referrers or {home}
            current = str(path.relative_to(a.layout.packages_dir))
            for dest_home in sorted(dest_homes):
                root = a.root_for(dest_home)
                to = f"{root}/src/{dest_home}/{rel}"
                self.data_files.append({"from": str(path), "to": to, "referrers": sorted(referrers),
                                        "in_place": to == current})

    # -- edits
    def _edits(self, info, file_home, dest):
        a = self.a
        scan = a.scans[info.path]
        lines = scan.lines
        edits = []
        is_src = info.area == "src"
        engine_side = file_home == ENGINE
        rewritten_unaliased = set()
        for ref in scan.refs:
            if ref.kind == "from-import":
                edits += self._from_import(info, scan, ref, file_home, dest)
                continue
            target = ref.target
            if not target or ref.note == "enumerated":
                continue
            t_home, t_key = split_mid(target)
            t_dest = a.dest_home.get(target, t_home)
            if ref.kind in {"registry", "lazy-table"}:
                if t_dest != file_home:
                    item = {"file": dest, "line": ref.line, "entry": ref.note, "module": target}
                    (self.registry if ref.kind == "registry" else self.errors).append(
                        {**item, "error": "lazy table entry crosses the split"} if ref.kind == "lazy-table" else item)
                continue
            if ref.written == "relative":
                if t_dest == file_home and t_home == (info.home or file_home):
                    continue
                if engine_side and t_dest == LANES:
                    self._engine_ref(info, dest, ref, target)
                    continue
                new = full_name(t_dest, t_key)
                edits += self._span_edit(info, scan, ref, ref.text, new, "relative string -> absolute")
                continue
            if t_dest == ref.written:
                continue
            if ref.kind == "api-string" and engine_side and t_dest == LANES:
                new_text = full_name(LANES, t_key) + ref.text[len(full_name(ref.written, t_key)):]
                self._manual(info, dest, ref.line, ref.start if ref.start is not None else lines.starts[ref.line - 1],
                             f"engine-side description names lane API `{ref.text}`; the engine must not mention "
                             f"{LANES} by default, so either accept this string or move the entry to mdb-lanes",
                             "api-string", sub=(ref.text, new_text))
                continue
            if engine_side and t_dest == LANES:
                if info.area == "tests" and ref.optional:
                    self.optional_engine_test_refs.append({"file": dest, "line": ref.line, "text": ref.text})
                else:
                    self._engine_ref(info, dest, ref, target)
                    continue
            old_prefix = full_name(t_home if ref.written == t_home else ref.written, t_key)
            new_prefix = full_name(t_dest, t_key)
            if not ref.text.startswith(old_prefix):
                self.unconfident.append({"file": dest, "line": ref.line, "text": ref.text,
                                         "why": "written prefix does not match resolved module"})
                continue
            new_text = new_prefix + ref.text[len(old_prefix):]
            edits += self._span_edit(info, scan, ref, ref.text, new_text, ref.kind)
            if ref.kind == "import" and ref.note.startswith("unaliased"):
                rewritten_unaliased.add((old_prefix, new_prefix))
        for old_prefix, new_prefix in rewritten_unaliased:
            for name, s, e, line, col in scan.attr_chains:
                if name == old_prefix:
                    if lines.data[s:e].decode("utf-8", "replace") != name:
                        self.unconfident.append({"file": dest, "line": line, "text": name,
                                                 "why": "attribute chain not contiguous"})
                        continue
                    edits.append(_edit(lines, s, e, name, new_prefix, line, col, "attr-usage",
                                       "usage of unaliased import"))
        if rewritten_unaliased:
            still_bound = any(r.kind == "import" and r.written == ENGINE and r.target and
                              a.dest_home.get(r.target) == ENGINE and
                              (r.note.startswith("unaliased") or r.text == ENGINE) for r in scan.refs)
            rewritten = tuple(old for old, _ in rewritten_unaliased)
            uses_root = any(n.split(".")[0] == ENGINE and not n.startswith(rewritten) for n, *_ in scan.attr_chains)
            if uses_root and not still_bound:
                self.warnings.append({"file": dest, "why": f"'{ENGINE}' name may become unbound after rewriting "
                                      f"{sorted(rewritten)}"})
        edits += self._rule_edits(info, scan, file_home, dest)
        edits = self._guard_identity(info, dest, edits)
        edits.sort(key=lambda e: e["start"])
        for prev, cur in zip(edits, edits[1:]):
            if cur["start"] < prev["end"]:
                self.unconfident.append({"file": dest, "line": cur["line"], "text": cur["old"],
                                         "why": "overlapping edits; both dropped"})
                edits = [e for e in edits if e is not prev and e is not cur]
        return edits

    def _engine_ref(self, info, dest, ref, target):
        self.errors.append({"file": dest, "line": ref.line, "kind": ref.kind, "text": ref.text, "module": target,
                            "lazy": ref.lazy, "error": "engine-side file references a lane module"})

    def _span_edit(self, info, scan, ref, old, new, reason):
        if ref.start is None or not ref.confident:
            self.unconfident.append({"file": info.label, "line": ref.line, "text": ref.text, "kind": ref.kind,
                                     "why": "span not located exactly; not rewritten"})
            return []
        lines = scan.lines
        current = lines.data[ref.start:ref.end].decode("utf-8", "replace")
        if current != old:
            self.unconfident.append({"file": info.label, "line": ref.line, "text": ref.text, "kind": ref.kind,
                                     "why": f"span text {current!r} != {old!r}"})
            return []
        return [_edit(lines, ref.start, ref.end, old, new, ref.line, ref.col, ref.kind, reason)]

    def _from_import(self, info, scan, ref, file_home, dest):
        a = self.a
        lines = scan.lines
        groups = defaultdict(list)
        cur_home = info.home or file_home
        engine_side = file_home == ENGINE
        errors = []
        for name in ref.names:
            target = name["target"]
            if target is None:
                groups[("keep", ref.text)].append(name)
                continue
            t_home, t_key = split_mid(target)
            t_dest = a.dest_home.get(target, t_home)
            base_key = t_key.rpartition(".")[0] if name["module"] else t_key
            if name["module"] and "." not in t_key:
                base_key = ""
            if ref.written == "relative" and t_dest == file_home and t_home == cur_home:
                groups[("keep", ref.text)].append(name)
                continue
            if ref.written != "relative" and t_dest == ref.written:
                groups[("keep", ref.text)].append(name)
                continue
            if engine_side and t_dest == LANES:
                if info.area == "tests" and ref.optional:
                    self.optional_engine_test_refs.append({"file": dest, "line": ref.line,
                                                           "text": f"from {ref.text} import {name['name']}"})
                else:
                    errors.append((name, target))
                    groups[("keep", ref.text)].append(name)
                    continue
            groups[("abs", full_name(t_dest, base_key))].append(name)
        for name, target in errors:
            self.errors.append({"file": dest, "line": ref.line, "kind": "from-import",
                                "text": f"from {ref.text} import {name['name']}", "module": target,
                                "lazy": ref.lazy, "error": "engine-side file references a lane module"})
        if list(groups) == [("keep", ref.text)] or not groups:
            return []
        if len(groups) == 1:
            (_, new_base), = groups
            if ref.start is None or not ref.confident:
                self.unconfident.append({"file": info.label, "line": ref.line, "text": ref.text,
                                         "kind": "from-import", "why": "module span not located; not rewritten"})
                return []
            return [_edit(lines, ref.start, ref.end, lines.text(ref.start, ref.end), new_base, ref.line, ref.col,
                          "from-import", "relative -> absolute" if ref.written == "relative" else "home change")]
        # Mixed statement: split into one statement per destination.
        parts = []
        for (mode, base), names in groups.items():
            texts = []
            for name in names:
                if name["span"] is None:
                    self.unconfident.append({"file": info.label, "line": ref.line, "text": ref.text,
                                             "why": "alias span missing; mixed import not split"})
                    return []
                texts.append(lines.text(*name["span"]))
            parts.append(f"from {base} import {', '.join(texts)}")
        s, e = ref.stmt
        original = lines.text(s, e)
        dropped_comment = "#" in original
        self.splits.append({"file": dest, "line": ref.line, "old": original, "new": "; ".join(parts),
                            "comment_dropped": dropped_comment})
        return [_edit(lines, s, e, original, "; ".join(parts), ref.line, ref.col, "from-import-split",
                      "mixed destinations")]

    # -- contents and manual fixes
    def build_contents(self):
        by_label = {f.label: f for f in self.a.layout.files}
        self.contents = {}
        for record in self.files:
            info = by_label[record["source"]]
            self.contents[record["dest"]] = apply_edits(info.data, record["edits"]) if record["edits"] else info.data
        return self.contents

    def manual_final(self):
        """Manual fixes located in the post-split file (line and current text after every rule edit
        and re-pin), with the replacement line when the fix is a one-line substitution."""
        by_label = {f.label: f for f in self.a.layout.files}
        records = {f["dest"]: f for f in self.files}
        out = []
        for m in self.manual:
            record = records.get(m["dest"])
            data = (self.contents or {}).get(m["dest"]) if record else None
            off = m["start"]
            if data is None:
                data = by_label[m["source"]].data
            else:
                for edit in record["edits"]:
                    if edit["end"] <= m["start"]:
                        off += len(edit["new"].encode("utf-8", "surrogatepass")) - \
                            len(edit["old"].encode("utf-8", "surrogatepass"))
                    elif edit["start"] <= m["start"] < edit["end"]:
                        off += edit["start"] - m["start"]
            line_no = data.count(b"\n", 0, off) + 1
            start = data.rfind(b"\n", 0, off) + 1
            end = data.find(b"\n", off)
            current = data[start:end if end >= 0 else len(data)].decode("utf-8", "replace").rstrip("\r")
            replacement, note = None, None
            if m["sub"]:
                old, new = m["sub"]
                if current.count(old) == 1:
                    replacement = current.replace(old, new)
                else:
                    note = "the text to replace spans lines; replace it textually: " + json.dumps(old) + " -> " + \
                        json.dumps(new)
            out.append({"file": m["dest"], "line": line_no, "source": m["source"], "source_line": m["source_line"],
                        "rule": m["rule"], "why": m["why"], "current": current, "replacement": replacement,
                        "block": m["block"], "note": note})
        out += self._cli_manual()
        return out

    def _cli_manual(self):
        verbs = self.lanes_verbs()
        if not verbs:
            return []
        cli = next((f for f in self.files if f["source"].endswith(f"/src/{ENGINE}/cli.py")), None)
        if cli is None:
            return []
        entries = ", ".join(f'"{v}"' for v, _ in verbs)
        block = (f"1. Delete the TOOLS entries for the lane verbs ({entries}); importing them as `mdb.<module>` "
                 "fails after the split.\n"
                 "2. After the `if argv and argv[0] in TOOLS:` dispatch, add:\n"
                 "    if argv:\n"
                 "        import importlib.metadata\n"
                 f'        lanes = {{ep.name: ep for ep in importlib.metadata.entry_points(group="{LANES_VERB_GROUP}")}}\n'
                 "        if argv[0] in lanes:\n"
                 "            return lanes[argv[0]].load()(argv[1:])\n"
                 "3. In the argparse `choices` list, append the lane verbs:\n"
                 f'    *sorted(ep.name for ep in importlib.metadata.entry_points(group="{LANES_VERB_GROUP}"))')
        return [{"file": cli["dest"], "line": None, "source": cli["source"], "source_line": None,
                 "rule": "cli-entry-points", "why": f"{len(verbs)} lane verbs are registered by mdb-lanes in the "
                 f"`{LANES_VERB_GROUP}` entry-point group; the engine cli must read that group (cli.py is not "
                 "edited by this tool)", "current": None, "replacement": None, "block": block, "note": None}]

    def to_json(self):
        changed = [f for f in self.files if f["changed"]]
        return {
            "schema": "mdb-split-lanes-plan-v1",
            "package_root": str(self.a.layout.engine_root),
            "counts": {"moves": len(self.moves), "copies": len(self.copies), "creates": len(self.creates),
                       "files": len(self.files), "files_changed": len(changed),
                       "files_byte_identical": len(self.files) - len(changed),
                       "edits": sum(len(f["edits"]) for f in self.files), "errors": len(self.errors),
                       "registry_reregister": len(self.registry), "unconfident": len(self.unconfident),
                       "mixed_import_splits": len(self.splits), "path_breaks": len(self.path_breaks),
                       "path_breaks_fixed_by_rule": sum(b.get("fix") == "rule" for b in self.path_breaks),
                       "path_breaks_manual": sum(b.get("fix") != "rule" for b in self.path_breaks),
                       "path_refs_not_file_anchored": sum(c.get("anchored") is False for c in self.path_checks),
                       "rule_fixes": len(self.rule_fixes),
                       "manual_fixes": len(self.manual) + len(self._cli_manual()),
                       "identity_strings_checked": len(self.identity_strings),
                       "identity_edits_refused": len(self.identity_refused),
                       "data_files_to_place": sum(not d["in_place"] for d in self.data_files)},
            "rule_fixes": self.rule_fixes, "manual_fixes": self.manual_final(),
            "identity_strings": self.identity_strings, "identity_edits_refused": self.identity_refused,
            "lanes_verbs": [{"verb": v, "target": t} for v, t in self.lanes_verbs()],
            "repin": self.repin.to_json() if self.repin else None,
            "moves": self.moves, "copies": self.copies, "creates": self.creates,
            "files": self.files, "errors": self.errors, "registry_reregister": self.registry,
            "unconfident": self.unconfident, "warnings": self.warnings, "aux_lane_refs": self.aux_lane_refs,
            "optional_engine_test_refs": self.optional_engine_test_refs, "mixed_import_splits": self.splits,
            "data_files": self.data_files, "path_checks": self.path_checks, "path_breaks": self.path_breaks,
        }


def _edit(lines, start, end, old, new, line, col, kind, reason):
    return {"start": start, "end": end, "line": line, "col": col, "old": old, "new": new, "kind": kind,
            "reason": reason}


def apply_edits(data: bytes, edits):
    out = bytearray(data)
    for edit in sorted(edits, key=lambda e: e["start"], reverse=True):
        old = edit["old"].encode()
        if bytes(out[edit["start"]:edit["end"]]) != old:
            raise ValueError(f"span mismatch at {edit['line']}:{edit['col']}: {edit['old']!r}")
        out[edit["start"]:edit["end"]] = edit["new"].encode()
    return bytes(out)


def data_dest(label, engine_name):
    """Where a non-module file (path relative to packages/) lands: qualification/ moves to lanes."""
    root, _, rest = label.partition("/")
    if root == engine_name and rest.startswith("qualification/"):
        return f"{LANES_DIST}/{rest}"
    return label


def _sccs(nodes, succ):
    """Tarjan strongly connected components (iterative)."""
    index, low, on_stack, stack, out = {}, {}, set(), [], []
    counter = [0]
    for start in sorted(nodes):
        if start in index:
            continue
        work = [(start, iter(sorted(succ.get(start, ()))))]
        index[start] = low[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for nxt in children:
                if nxt not in nodes:
                    continue
                if nxt not in index:
                    index[nxt] = low[nxt] = counter[0]
                    counter[0] += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(sorted(succ.get(nxt, ())))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    item = stack.pop()
                    on_stack.discard(item)
                    comp.append(item)
                    if item == node:
                        break
                out.append(sorted(comp))
    return out


def repin_fixed_point(contents, before):
    """Core of :class:`Repin` on plain dicts (``name -> bytes``, ``name -> pre-split sha256``).

    Returns (new contents, rounds, resolvable occurrences, cycles, cycle pins, ambiguous). With real
    data a pin cycle cannot be built (each pin equals a current sha256: a preimage), so cycle
    detection is a guard; it matters when ``before`` is not the bytes' own digest.
    """
    by_before = defaultdict(list)
    for dest, value in before.items():
        by_before[value].append(dest)
    start_sha = {d: sha256(b) for d, b in contents.items()}
    occ, ambiguous = [], []
    for p, data in contents.items():
        for m in HEX64.finditer(data):
            value = m.group(0).decode()
            dests = by_before.get(value)
            if not dests:
                continue
            if len(dests) > 1:
                same_side = [d for d in dests if d.split("/", 1)[0] == p.split("/", 1)[0]]
                if len(same_side) == 1:
                    dests = same_side
                elif len({start_sha[d] for d in dests}) == 1:
                    dests = dests[:1]
                else:
                    ambiguous.append({"file": p, "line": data.count(b"\n", 0, m.start()) + 1,
                                           "value": value, "candidates": sorted(dests)})
                    continue
            occ.append((p, m.start(), value, dests[0]))
    will = {d for d in contents if start_sha[d] != before[d]}
    grew = True
    while grew:
        grew = False
        for p, _, _, d in occ:
            if d in will and p not in will:
                will.add(p)
                grew = True
    succ = defaultdict(set)
    for p, _, _, d in occ:
        if d in will:
            succ[p].add(d)
    comp_of, cyclic = {}, set()
    for i, comp in enumerate(_sccs(will, succ)):
        for node in comp:
            comp_of[node] = i
        if len(comp) > 1 or comp[0] in succ.get(comp[0], ()):
            cyclic.add(i)
    resolvable, cycle_pins = [], []
    for o in occ:
        p, s, value, d = o
        if d not in will:
            continue
        if comp_of.get(p) == comp_of.get(d) and comp_of.get(d) in cyclic:
            cycle_pins.append({"file": p, "line": contents[p].count(b"\n", 0, s) + 1, "value": value,
                                    "pinned": d})
        else:
            resolvable.append(o)
    cycles = []
    for i in sorted(cyclic):
        members = sorted(n for n, c in comp_of.items() if c == i)
        cycles.append({"members": members,
                            "pins": [c for c in cycle_pins if c["file"] in members]})
    cur = dict(contents)
    rounds = 0
    while True:
        shas = {d: sha256(cur[d]) for d in will}
        updates = defaultdict(list)
        for p, s, _, d in resolvable:
            want = shas[d].encode()
            if cur[p][s:s + 64] != want:
                updates[p].append((s, want))
        if not updates:
            break
        rounds += 1
        if rounds > len(will) + 2:
            raise RuntimeError("re-pinning did not reach a fixed point (internal error: cycle not detected)")
        for p, ups in updates.items():
            buf = bytearray(cur[p])
            for s, want in ups:
                buf[s:s + 64] = want
            cur[p] = bytes(buf)
    return cur, rounds, resolvable, cycles, cycle_pins, ambiguous


class Repin:
    """Re-pin sha256 literals to post-split file bytes, iterated to a fixed point.

    Nodes are planned destination files (plus JSON/text pinning files with ``repin_json``). An
    occurrence is a 64-hex literal in a node equal to the pre-split sha256 of a destination whose
    bytes will change. Occurrences inside a pin cycle (a strongly connected component of changing
    files, including a file pinning itself) are unresolvable and left as they are.
    """

    def __init__(self, plan, hazards, apply=True, repin_json=False):
        a, layout = plan.a, plan.a.layout
        engine = layout.engine_root.name
        self.repin_json, self.applied = repin_json, apply
        if plan.contents is None:
            plan.build_contents()
        contents = dict(plan.contents)
        before = {r["dest"]: r["sha_before"] for r in plan.files}
        source_of = {r["dest"]: r["source"] for r in plan.files}
        text_pins = [p for p in hazards.pins if not p["pinned_by"].endswith(".py")]
        self.text_nodes = {}
        if repin_json:
            for label in sorted({p["pinned_by"] for p in text_pins}):
                path = layout.packages_dir / label
                if not path.is_file():
                    continue
                dest = data_dest(label, engine)
                data = path.read_bytes()
                contents[dest] = data
                before[dest] = sha256(data)
                source_of[dest] = label
                self.text_nodes[dest] = label
        cur, self.rounds, resolvable, self.cycles, self.cycle_pins, self.ambiguous = repin_fixed_point(contents, before)
        self.pins = []
        for p, s, value, d in resolvable:
            new = cur[p][s:s + 64].decode()
            if new != value:
                self.pins.append({"file": p, "line": cur[p].count(b"\n", 0, s) + 1, "pinned": d,
                                  "pinned_source": source_of.get(d), "old": value, "new": new,
                                  "kind": "python" if p.endswith(".py") else "text"})
        self.final_sha = {d: sha256(b) for d, b in cur.items()}
        self.map = [{"old_sha256": before[d], "new_sha256": self.final_sha[d], "path_before": source_of[d],
                     "path_after": d} for d in sorted(cur) if self.final_sha[d] != before[d]]
        self.contents = cur
        if apply:
            for record in plan.files:
                data = cur[record["dest"]]
                record["repins"] = sum(1 for x in self.pins if x["file"] == record["dest"])
                record["sha_after"] = self.final_sha[record["dest"]]
                record["changed"] = record["sha_after"] != record["sha_before"]
                plan.contents[record["dest"]] = data
            plan.repin = self
        # JSON/text pins left stale (not re-pinned): the pinned file's bytes differ at every destination.
        dests_of = defaultdict(list)
        for record in plan.files:
            dests_of[record["source"]].append(record["dest"])
        final = self.final_sha if apply else {d: sha256(b) for d, b in plan.contents.items()}
        self.final_for_json = final
        self.json_stale = []
        for pin in text_pins:
            where = data_dest(pin["pinned_by"], engine)
            if where in self.text_nodes:
                continue  # re-pinned (or reported as a cycle) above
            dests = dests_of.get(pin["pinned_file"], [])
            if dests and all(final.get(d, pin["sha256"]) != pin["sha256"] for d in dests):
                self.json_stale.append({"file": where, "source": pin["pinned_by"], "line": pin["line"],
                                        "pinned_file": pin["pinned_file"], "old": pin["sha256"],
                                        "new": sorted({final[d] for d in dests})})
        self.text_contents = {d: cur[d] for d in self.text_nodes if cur[d] != contents[d]} if apply else {}
        self.predicted_tests = predict_pin_tests(plan, sorted({p["source"] for p in self.json_stale}))
        outside = [p for p in hazards.pins if p["pinned_by"].endswith(".py") and p["bytes_change"] and
                   p["pinned_by"] not in {r["source"] for r in plan.files}]
        self.python_pins_outside_plan = outside

    def to_json(self):
        py = [p for p in self.pins if p["kind"] == "python"]
        return {
            "schema": "mdb-split-lanes-repin-v1", "applied": self.applied, "repin_json": self.repin_json,
            "rounds": self.rounds,
            "counts": {"python_pins_repinned": len(py),
                       "python_files_repinned": len({p["file"] for p in py}),
                       "text_pins_repinned": len(self.pins) - len(py),
                       "cycles": len(self.cycles), "cycle_pins": len(self.cycle_pins),
                       "ambiguous": len(self.ambiguous), "json_pins_stale": len(self.json_stale),
                       "json_files_stale": len({p["file"] for p in self.json_stale}),
                       "files_with_new_sha": len(self.map),
                       "predicted_failing_tests": sum(1 for x in self.predicted_tests
                                                      if x["match"] == "path" and x["hashes"]),
                       "tests_reading_stale_json_by_path": sum(1 for x in self.predicted_tests if x["match"] == "path"),
                       "tests_naming_stale_json_file_name_only": sum(1 for x in self.predicted_tests
                                                                    if x["match"] == "file-name"),
                       "python_pins_outside_plan": len(self.python_pins_outside_plan)},
            "map": self.map, "pins": self.pins, "cycles": self.cycles, "ambiguous": self.ambiguous,
            "json_pins_stale": self.json_stale, "predicted_failing_tests": self.predicted_tests,
            "python_pins_outside_plan": self.python_pins_outside_plan,
        }


def predict_pin_tests(plan, json_sources):
    """Test functions that read one of ``json_sources`` (by its path tail or file name), directly
    or through a module-level name / helper function of the same test file. ``hashes`` marks tests
    whose code (or helper) also computes a digest. Predicted to fail: ``match == "path"`` and
    ``hashes``; a ``file-name``-only match (e.g. a generic ``custody.json``) is a weak candidate."""
    if not json_sources:
        return []
    keys = []
    for label in json_sources:
        rest = label.split("/", 1)[1]
        area, _, tail = rest.partition("/")
        keys.append((label, tail.encode(), Path(label).name.encode()))
    by_label = {f.label: f for f in plan.a.layout.files}
    dests = defaultdict(list)
    for record in plan.files:
        dests[record["source"]].append(record["dest"])
    out = []
    for label, info in by_label.items():
        if info.area != "tests":
            continue
        data = info.data
        hits = [(j, tail if tail in data else name, "path" if tail in data else "file-name")
                for j, tail, name in keys if tail in data or name in data]
        if not hits:
            continue
        try:
            tree = ast.parse(data)
        except SyntaxError:
            continue
        text = data.decode("utf-8", "replace")

        def segment(node):
            return ast.get_source_segment(text, node) or ""
        tests, helpers, names = [], {}, {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                (tests.append((node.name, node)) if node.name.startswith("test") else
                 helpers.__setitem__(node.name, segment(node)))
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                        tests.append((node.name + "::" + item.name, item))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                seg = segment(node)
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        names[n.id] = seg
        for json_label, needle, match in hits:
            needle_s = needle.decode()
            carriers = {n for n, seg in names.items() if needle_s in seg}
            carriers |= {h for h, seg in helpers.items() if needle_s in seg or
                         any(re.search(rf"\b{re.escape(c)}\b", seg) for c in carriers)}
            for test_name, node in tests:
                seg = segment(node)
                used = needle_s in seg or any(re.search(rf"\b{re.escape(c)}\b", seg) for c in carriers)
                if not used:
                    continue
                helper_text = " ".join(helpers[h] for h in carriers if h in helpers)
                hashes = bool(HASHY.search(seg) or HASHY.search(helper_text))
                for dest in dests.get(label, [label]):
                    out.append({"test": f"{dest}::{test_name}", "json": json_label, "match": match,
                                "hashes": hashes})
    out.sort(key=lambda r: (r["test"], r["json"]))
    return out


# ---------------------------------------------------------------- hazards

class Hazards:
    """Identity hazards (a)-(e) for the planned split. Detection only; nothing is fixed."""

    def __init__(self, analysis: Analysis, plan: Plan):
        a, layout = analysis, analysis.layout
        self.a, self.plan = a, plan
        self.fate = {}
        for f in plan.files:
            self.fate.setdefault(f["source"], []).append(f)
        self.registered, self.source_bound, self.schema_strings, self.pickle = [], [], [], []
        self.pins, self.provenance, self.text_mentions, self.text_pkgpaths = [], [], [], []
        for scan in a.scans.values():
            info = scan.info
            module_change = None
            if info.area == "src":
                dest = a.dest_home[info.mid]
                if dest != info.home:
                    module_change = [full_name(info.home, info.key), full_name(dest, info.key)]
            changed = any(f["changed"] for f in self.fate.get(info.label, []))
            for reg in scan.registrations:
                owner_change = module_change
                owner = reg.get("owner_module")
                if owner and owner != info.mid and owner in layout.modules:
                    o_home, o_key = split_mid(owner)
                    o_dest = a.dest_home[owner]
                    owner_change = [full_name(o_home, o_key), full_name(o_dest, o_key)] if o_dest != o_home else None
                self.registered.append({"file": info.label, **reg, "module_change": owner_change,
                                        "registering_module_change": module_change, "bytes_change": changed})
            for item in scan.source_bound:
                self.source_bound.append({"file": info.label, **item, "module_change": module_change,
                                          "bytes_change": changed})
            for p in scan.pickle:
                self.pickle.append({"file": info.label, **p, "module_change": module_change})
            for m in scan.mentions:
                t_home, t_key = split_mid(m["target"])
                if a.dest_home.get(m["target"], t_home) != t_home:
                    self.schema_strings.append({"file": info.label, **m})
        self._pins()
        self._provenance()
        self._text_files()

    def _pins(self):
        a, layout = self.a, self.a.layout
        by_sha = defaultdict(list)
        for info in layout.files:
            by_sha[info.sha].append(info)
        for scan in a.scans.values():
            for h in scan.hex_consts:
                for pinned in by_sha.get(h["value"], ()):
                    self.pins.append(self._pin_record(scan.info.label, h["line"], h.get("context", ""), pinned))
        self._by_sha = by_sha

    def _pin_record(self, where, line, context, pinned):
        fates = self.fate.get(pinned.label, [])
        return {"pinned_by": where, "line": line, "context": context, "pinned_file": pinned.label,
                "sha256": pinned.sha, "moves_to": sorted({f["dest"] for f in fates if f["dest"] != pinned.label}),
                "bytes_change": any(f["changed"] for f in fates)}

    def _provenance(self):
        layout = self.a.layout
        path = layout.engine_root / "PROVENANCE.json"
        if not path.exists():
            return
        data = json.loads(path.read_text())
        by_rel = {}
        for info in layout.files:
            if info.root == layout.engine_root.name:
                rel = (f"src/{info.home}/{info.rel}" if info.area == "src" else f"{info.area}/{info.rel}")
                by_rel[rel] = info

        def walk(value, where):
            if isinstance(value, dict):
                for key in ("extracted_path", "path"):
                    rel = value.get(key)
                    if isinstance(rel, str) and rel in by_rel:
                        info = by_rel[rel]
                        fates = self.fate.get(info.label, [])
                        shas = {k: v for k, v in value.items() if k.endswith("sha256") and isinstance(v, str)}
                        recorded = value.get("final_sha256") or value.get("sha256") or value.get("extracted_sha256")
                        self.provenance.append({
                            "entry": where, "field": key, "path": rel,
                            "recorded_sha256_matches_now": recorded == info.sha if recorded else None,
                            "moves_to": sorted({f["dest"] for f in fates if f["dest"] != info.label}),
                            "bytes_change": any(f["changed"] for f in fates), "sha_fields": sorted(shas)})
                for key, item in value.items():
                    walk(item, f"{where}.{key}" if where else key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    walk(item, f"{where}[{i}]")

        walk(data, "")

    def _text_files(self):
        a, layout = self.a, self.a.layout
        moved_names = {}
        for m, info in layout.modules.items():
            if a.dest_home[m] != info.home:
                moved_names[full_name(info.home, info.key)] = m
        roots = [layout.engine_root] + ([layout.lanes_root] if layout.lanes_root.exists() else [])
        self.text_files_scanned = 0
        self.text_files_skipped_large = []
        by_sha = self._by_sha
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in TEXT_SCAN_SKIP_DIRS]
                for filename in filenames:
                    path = Path(dirpath) / filename
                    if path.suffix not in TEXT_SUFFIXES:
                        continue
                    size = path.stat().st_size
                    if size > TEXT_SCAN_MAX_BYTES:
                        self.text_files_skipped_large.append({"file": str(path.relative_to(layout.packages_dir)),
                                                              "bytes": size})
                        continue
                    data = path.read_bytes()
                    self.text_files_scanned += 1
                    label = str(path.relative_to(layout.packages_dir))
                    counts = defaultdict(int)
                    for match in DOTTED_RE_B.finditer(data):
                        name = match.group(1).decode()
                        res = layout.resolve_dotted(name)
                        if res and full_name(res[0].home, res[0].key) in moved_names:
                            counts[res[0].mid] += 1
                    if counts:
                        self.text_mentions.append({"file": label, "modules": dict(counts)})
                    pk = defaultdict(int)
                    for match in PKGPATH_RE_B.finditer(data):
                        home, _, rel = match.group(1).decode().partition("/")
                        key = rel[:-3].replace("/", ".")
                        if key.endswith("__init__"):
                            key = key[:-len("__init__")].rstrip(".")
                        info, _ = layout.lookup(home, key)
                        if info and a.dest_home[info.mid] != info.home:
                            pk[info.mid] += 1
                    if pk:
                        self.text_pkgpaths.append({"file": label, "modules": dict(pk)})
                    if path.name == "PROVENANCE.json":
                        continue
                    seen = set()
                    for match in HEX64.finditer(data):
                        value = match.group(0).decode()
                        if value in seen:
                            continue
                        seen.add(value)
                        for pinned in by_sha.get(value, ()):
                            line = data.count(b"\n", 0, match.start()) + 1
                            self.pins.append(self._pin_record(label, line, "text file", pinned))

    def to_json(self):
        pins_breaking = [p for p in self.pins if p["bytes_change"]]
        return {
            "schema": "mdb-split-lanes-hazards-v1",
            "a_registered_operations": self.registered,
            "b_source_bound": self.source_bound,
            "b_pinned_file_hashes": self.pins,
            "b_pins_broken_by_rewrite": pins_breaking,
            "c_module_name_strings_in_code": self.schema_strings,
            "c_module_names_in_text_files": self.text_mentions,
            "c_package_paths_in_text_files": self.text_pkgpaths,
            "d_provenance": self.provenance,
            "e_serialization_sites": self.pickle,
            "text_files_scanned": self.text_files_scanned,
            "text_files_skipped_large": self.text_files_skipped_large,
        }


# ---------------------------------------------------------------- apply

def refuse_inside_repo(out: Path, package_root: Path):
    repo = find_repo_root(package_root) or package_root.resolve().parents[1]
    if is_inside(out, repo) or is_inside(out, package_root):
        raise SystemExit(f"refusing to write inside the source tree: {out} is inside {repo}")


def prepare(package_root: Path, keep: Path | None, *, assume_cut: Path | None = None, repin: bool = False,
            repin_json: bool = False, strict_engine: bool = False):
    """Analysis, plan (with rule edits), hazards and re-pin result, shared by every subcommand."""
    analysis = Analysis(package_root, keep, assume_cut=assume_cut)
    if analysis.cut_unmatched:
        raise SystemExit("--assume-cut entries that match no edge:\n" + "\n".join(
            f"  {c['spec']}: {c['why']}" for c in analysis.cut_unmatched))
    if strict_engine:
        violations = analysis.strict_violations()
        if violations:
            lines = [f"--strict-engine: the engine closure holds {len(violations)} module(s) outside "
                     f"{', '.join(ENGINE_SUBPACKAGES)}, the top modules and the keep list. Cut these edges "
                     "(or add the module to the keep list):"]
            for v in violations:
                lines.append(f"  {v['module']} ({v['lines']} lines) via " + " ; ".join(v["via"]))
            raise SystemExit("\n".join(lines))
    plan = Plan(analysis)
    plan.build_contents()
    hazards = Hazards(analysis, plan)
    result = Repin(plan, hazards, apply=repin, repin_json=repin and repin_json)
    return analysis, plan, hazards, result


def manual_fixes_markdown(plan):
    items = plan.manual_final()
    w = ["# mdb lane split: manual fixes", "",
         f"{len(items)} item(s) the split tool does not edit. Line numbers and `current` text are in the "
         "post-split file (after rule edits and re-pinning). Rule-based fixes already applied: "
         f"{len(plan.rule_fixes)}.", ""]
    for i, m in enumerate(items, 1):
        where = f"`{m['file']}`" + (f":{m['line']}" if m["line"] else "")
        src = f" (source `{m['source']}`" + (f":{m['source_line']}" if m["source_line"] else "") + ")"
        w.append(f"## {i}. {where}{src} [{m['rule']}]")
        w.append("")
        w.append(m["why"])
        w.append("")
        if m["current"] is not None:
            w += ["current:", "```python", m["current"], "```"]
        if m["replacement"] is not None:
            w += ["replacement:", "```python", m["replacement"], "```"]
        elif m["block"]:
            w += ["change:", "```text", m["block"], "```"]
        elif m["note"]:
            w.append(m["note"])
        else:
            w.append("replacement: none mechanical (owner decision).")
        w.append("")
    return "\n".join(w) + "\n", len(items)


def write_side(side: Path, analysis, plan, hazards, repin, manifest, keep=None):
    side.mkdir(parents=True, exist_ok=True)
    (side / "analysis.json").write_text(json.dumps(analysis.to_json(), indent=1, sort_keys=True, default=str))
    (side / "plan.json").write_text(json.dumps(plan.to_json(), indent=1, sort_keys=True, default=str))
    (side / "hazards.json").write_text(json.dumps(hazards.to_json(), indent=1, sort_keys=True, default=str))
    (side / "repin-map.json").write_text(json.dumps(repin.to_json(), indent=1, sort_keys=True, default=str))
    text, _ = manual_fixes_markdown(plan)
    (side / "manual-fixes.md").write_text(text)
    if keep and Path(keep).exists():
        shutil.copyfile(keep, side / "keep.txt")
    if analysis.assume_cut_file:
        shutil.copyfile(analysis.assume_cut_file, side / "assume-cut.txt")
    (side / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))


def _manifest(analysis, plan, mode):
    return {"schema": "mdb-split-lanes-manifest-v1", "mode": mode,
            "source_package_root": str(analysis.layout.engine_root), "files": [], "created": [], "data": [],
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def _test_data_files(analysis, plan):
    """Small non-Python files in tests/ or qualification/ that copied tests name: (source, [dests])."""
    root_e = analysis.layout.engine_root
    copied_text = b"\n".join(plan.contents[r["dest"]] for r in plan.files if r["dest"].split("/")[1] == "tests")
    out = []
    for area in ("tests", "qualification"):
        for home_root in (root_e, analysis.layout.lanes_root):
            base = home_root / area
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix == ".py" or "__pycache__" in path.parts:
                    continue
                if path.stat().st_size > DATA_COPY_MAX_BYTES or path.name.encode() not in copied_text:
                    continue
                rel = path.relative_to(base).as_posix()
                sides = [LANES_DIST] if area == "qualification" else [root_e.name, LANES_DIST]
                out.append((path, area, rel, sides))
    return out


def do_apply(package_root: Path, out: Path, keep: Path | None, clean: bool = False, *, repin: bool = False,
             repin_json: bool = False, assume_cut: Path | None = None, strict_engine: bool = False):
    out = out.resolve()
    refuse_inside_repo(out, package_root)
    if out.exists() and any(out.iterdir()) and not (clean and (out / SIDE_DIR / MANIFEST).exists()):
        raise SystemExit(f"output directory is not empty: {out} (pass --clean to replace a previous apply)")
    analysis, plan, hazards, result = prepare(package_root, keep, assume_cut=assume_cut, repin=repin,
                                              repin_json=repin_json, strict_engine=strict_engine)
    if out.exists() and any(out.iterdir()):
        shutil.rmtree(out)  # only a previous apply (manifest present) reaches here
    packages = out / "packages"
    manifest = _manifest(analysis, plan, "copy")
    for record in plan.files:
        new = plan.contents[record["dest"]]
        if sha256(new) != record["sha_after"]:
            raise SystemExit("internal error: plan hash mismatch for " + record["source"])
        target = packages / record["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(new)
        manifest["files"].append({"source": record["source"], "dest": record["dest"], "sha_before": record["sha_before"],
                                  "sha_after": record["sha_after"], "changed": record["changed"],
                                  "edits": len(record["edits"]), "repins": record.get("repins", 0)})
    for create in plan.creates:
        target = packages / create["to"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(create["content"])
        manifest["created"].append(create["to"])
    for data in plan.data_files:
        target = packages / data["to"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data["from"], target)
        manifest["data"].append(data["to"])
    for path, area, rel, sides in _test_data_files(analysis, plan):
        for side_name in sides:
            target = packages / side_name / area / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                manifest["data"].append(f"{side_name}/{area}/{rel}")
    for dest, data in result.text_contents.items():
        target = packages / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest["data"].append(dest)
    root_e = analysis.layout.engine_root
    for name in ("pyproject.toml", "PROVENANCE.json", "AGENTS.md"):
        source = root_e / name
        if source.exists():
            target = packages / root_e.name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest["data"].append(f"{root_e.name}/{name}")
    write_side(out / SIDE_DIR, analysis, plan, hazards, result, manifest, keep)
    return analysis, plan, hazards, manifest


def do_repin_copy(out: Path, repin_json: bool = False):
    """Re-pin an applied copy in place (same plan as apply, re-derived from the recorded source)."""
    out = out.resolve()
    side = out / SIDE_DIR
    manifest = json.loads((side / MANIFEST).read_text())
    packages = out / "packages"
    keep = side / "keep.txt" if (side / "keep.txt").exists() else None
    cut = side / "assume-cut.txt" if (side / "assume-cut.txt").exists() else None
    analysis, plan, hazards, result = prepare(Path(manifest["source_package_root"]), keep, assume_cut=cut,
                                              repin=True, repin_json=repin_json)
    recorded = {r["dest"]: r for r in manifest["files"]}
    drift = [r["dest"] for r in plan.files if sha256((packages / r["dest"]).read_bytes()) not in
             {recorded.get(r["dest"], {}).get("sha_after"), r["sha_after"]}]
    if drift:
        raise SystemExit(f"{len(drift)} file(s) in the copy differ from the recorded apply; first: {drift[:3]}")
    for record in plan.files:
        path = packages / record["dest"]
        if path.read_bytes() != plan.contents[record["dest"]]:
            path.write_bytes(plan.contents[record["dest"]])
        rec = recorded.get(record["dest"])
        if rec is not None:
            rec.update(sha_after=record["sha_after"], changed=record["changed"], repins=record.get("repins", 0))
    for dest, data in result.text_contents.items():
        (packages / dest).parent.mkdir(parents=True, exist_ok=True)
        (packages / dest).write_bytes(data)
    write_side(side, analysis, plan, hazards, result, manifest, keep)
    return result


# ---------------------------------------------------------------- in-place (git)

def _git(repo: Path, *args, check=True, input_text=None):
    import subprocess
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, input=input_text)
    if check and proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args[:4])}... failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def _porcelain(repo: Path):
    raw = _git(repo, "status", "--porcelain=v1", "-z", "-uall")
    entries, items = [], raw.split("\0")
    i = 0
    while i < len(items):
        item = items[i]
        if not item:
            i += 1
            continue
        code, path = item[:2], item[3:]
        entries.append((code, path))
        if code[0] in "RC":
            i += 1  # the original path follows a rename/copy entry
        i += 1
    return entries


def build_inplace_ops(analysis, plan, result, repo: Path, tracked: set):
    layout = analysis.layout
    rel_packages = layout.packages_dir.resolve().relative_to(repo.resolve()).as_posix()

    def P(label):
        return f"{rel_packages}/{label}" if rel_packages != "." else label

    engine = layout.engine_root.name
    by_label = {f.label: f for f in layout.files}
    ops = []
    moves = []
    for m in plan.moves:
        moves.append((P(m["from"]), P(m["to"])))
    qual_prefix = P(engine + "/qualification/")
    planned_sources = {P(m["from"]) for m in plan.moves}
    for path in sorted(tracked):
        if path.startswith(qual_prefix) and path not in planned_sources and not path.endswith(".py"):
            moves.append((path, P(data_dest(path[len(rel_packages) + 1:] if rel_packages != "." else path, engine))))
    writes, adds = [], []
    for d in plan.data_files:
        if d["in_place"]:
            continue
        src = Path(d["from"]).resolve().relative_to(repo.resolve()).as_posix()
        source_home = ENGINE if f"/src/{ENGINE}/" in "/" + src else LANES
        if source_home in d["referrers"] or src not in tracked:
            writes.append({"op": "write", "path": P(d["to"]), "data": Path(d["from"]).read_bytes(), "new": True})
            adds.append(P(d["to"]))
        else:
            moves.append((src, P(d["to"])))
    groups = defaultdict(list)
    for src, dst in moves:
        kind = "git-mv" if src in tracked else "mv"
        if kind == "git-mv" and Path(src).name == Path(dst).name:
            groups[(kind, str(Path(dst).parent))].append(src)
        else:
            ops.append({"op": kind, "from": src, "to": dst})
    for (kind, dest_dir), sources in sorted(groups.items()):
        for i in range(0, len(sources), 200):
            ops.append({"op": "git-mv-many", "sources": sources[i:i + 200], "dest_dir": dest_dir})
    copies = {c["to"] for c in plan.copies}
    for record in plan.files:
        info = by_label[record["source"]]
        content = plan.contents[record["dest"]]
        dest = P(record["dest"])
        if record["dest"] in copies:
            writes.append({"op": "write", "path": dest, "data": content, "new": True})
            adds.append(dest)
        elif content != info.data:
            writes.append({"op": "write", "path": dest, "data": content, "new": False, "prev": info.data})
    for create in plan.creates:
        writes.append({"op": "write", "path": P(create["to"]), "data": create["content"].encode(), "new": True})
        adds.append(P(create["to"]))
    for path, area, rel, sides in _test_data_files(analysis, plan):
        for side_name in sides:
            target = P(f"{side_name}/{area}/{rel}")
            current = path.resolve().relative_to(repo.resolve()).as_posix()
            if area == "qualification" or target == current or (repo / target).exists():
                continue
            writes.append({"op": "write", "path": target, "data": path.read_bytes(), "new": True})
            adds.append(target)
    for dest, data in result.text_contents.items():
        label = result.text_nodes[dest]
        writes.append({"op": "write", "path": P(dest), "data": data, "new": False,
                       "prev": (layout.packages_dir / label).read_bytes()})
    seen = set()
    for w in writes:
        if w["path"] in seen:
            raise SystemExit("internal error: two writes to " + w["path"])
        seen.add(w["path"])
    ops += writes
    if adds:
        for i in range(0, len(adds), 200):
            ops.append({"op": "git-add", "paths": sorted(adds)[i:i + 200]})
    return ops


def _op_text(op):
    if op["op"] == "git-mv":
        return f"git mv -- {op['from']} {op['to']}"
    if op["op"] == "mv":
        return f"mv {op['from']} {op['to']}   # untracked"
    if op["op"] == "git-mv-many":
        return f"mkdir -p {op['dest_dir']} && git mv -- {' '.join(op['sources'])} {op['dest_dir']}/"
    if op["op"] == "write":
        return f"write {op['path']} sha256={op['sha']}" + (" (new)" if op["new"] else " (rewrite)")
    if op["op"] == "git-add":
        return "git add -- " + " ".join(op["paths"])
    return json.dumps(op)


def _op_done(repo: Path, op):
    if op["op"] in {"git-mv", "mv"}:
        return not (repo / op["from"]).exists() and (repo / op["to"]).exists()
    if op["op"] == "git-mv-many":
        return all(not (repo / s).exists() and (repo / op["dest_dir"] / Path(s).name).exists()
                   for s in op["sources"])
    if op["op"] == "write":
        path = repo / op["path"]
        return path.is_file() and sha256(path.read_bytes()) == op["sha"]
    return False


def _run_op(repo: Path, side: Path, op):
    if op["op"] == "git-mv":
        (repo / op["to"]).parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "mv", "--", op["from"], op["to"])
    elif op["op"] == "mv":
        (repo / op["to"]).parent.mkdir(parents=True, exist_ok=True)
        os.rename(repo / op["from"], repo / op["to"])
    elif op["op"] == "git-mv-many":
        (repo / op["dest_dir"]).mkdir(parents=True, exist_ok=True)
        _git(repo, "mv", "--", *op["sources"], op["dest_dir"] + "/")
    elif op["op"] == "write":
        path = repo / op["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((side / "blobs" / op["sha"]).read_bytes())
    elif op["op"] == "git-add":
        _git(repo, "add", "--", *op["paths"])


def _undo_op(repo: Path, side: Path, op):
    if op["op"] == "git-add":
        _git(repo, "rm", "--cached", "-q", "--ignore-unmatch", "--", *op["paths"])
    elif op["op"] == "write":
        path = repo / op["path"]
        if op["new"]:
            if path.exists():
                path.unlink()
        else:
            path.write_bytes((side / "blobs" / op["prev_sha"]).read_bytes())
    elif op["op"] == "git-mv":
        (repo / op["from"]).parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "mv", "--", op["to"], op["from"])
    elif op["op"] == "mv":
        (repo / op["from"]).parent.mkdir(parents=True, exist_ok=True)
        os.rename(repo / op["to"], repo / op["from"])
    elif op["op"] == "git-mv-many":
        by_dir = defaultdict(list)
        for s in op["sources"]:
            by_dir[str(Path(s).parent)].append(f"{op['dest_dir']}/{Path(s).name}")
        for directory, moved in by_dir.items():
            (repo / directory).mkdir(parents=True, exist_ok=True)
            _git(repo, "mv", "--", *moved, directory + "/")


def _prune_empty(root: Path):
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def do_apply_in_place(package_root: Path, keep: Path | None, *, expect_branch: str | None, allow_dirty=(),
                      dry_run=False, resume=False, rollback=False, repin=False, repin_json=False,
                      assume_cut: Path | None = None, strict_engine: bool = False, echo=print):
    package_root = package_root.resolve()
    repo = Path(_git(package_root, "rev-parse", "--show-toplevel").strip())
    side = Path(_git(package_root, "rev-parse", "--absolute-git-dir").strip()) / INPLACE_DIR
    journal = side / "journal.jsonl"
    ops_file = side / "ops.json"
    if rollback:
        return _rollback(repo, side, echo)
    problems = []
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if not expect_branch:
        problems.append("--expect-branch is required for --in-place")
    elif branch != expect_branch:
        problems.append(f"current branch is {branch!r}, expected {expect_branch!r}")
    allowed = [a.rstrip("/") for a in (str(Path(x).resolve().relative_to(repo)) if Path(x).is_absolute() else x
                                       for x in allow_dirty)]
    dirty = [(c, p) for c, p in _porcelain(repo)
             if not any(p == a or p.startswith(a + "/") for a in allowed)]
    if resume:
        if not ops_file.exists():
            problems.append(f"--resume: no journal at {side}")
    else:
        if dirty:
            problems.append(f"working tree not clean outside --allow-dirty ({len(dirty)} entries; first: "
                            + ", ".join(f"{c.strip()} {p}" for c, p in dirty[:5]) + ")")
        if ops_file.exists() and not (side / "complete").exists():
            problems.append(f"an earlier in-place run is incomplete ({side}); pass --resume or --rollback")
    if problems and not dry_run:
        raise SystemExit("refusing --in-place:\n  " + "\n  ".join(problems))
    if resume:
        state = json.loads(ops_file.read_text())
        ops = state["ops"]
    else:
        analysis, plan, hazards, result = prepare(package_root, keep, assume_cut=assume_cut, repin=repin,
                                                  repin_json=repin_json, strict_engine=strict_engine)
        if plan.errors:
            msg = (f"plan has {len(plan.errors)} error(s) (engine-side references to lane modules or refused "
                   "identity edits); fix them first:\n  " + "\n  ".join(
                       f"{e['file']}:{e['line']} {e.get('text')} ({e['error']})" for e in plan.errors[:40]))
            if not dry_run:
                raise SystemExit(msg)
            problems.append(msg)
        tracked = set(_git(repo, "ls-files", "-z").split("\0")) - {""}
        ops = build_inplace_ops(analysis, plan, result, repo, tracked)
        for op in ops:
            if op["op"] == "write":
                op["sha"] = sha256(op["data"])
        if dry_run:
            for op in ops:
                echo(_op_text(op))
            counts = defaultdict(int)
            for op in ops:
                counts[op["op"]] += len(op.get("sources", op.get("paths", [1])))
            echo("# " + json.dumps({"branch": branch, "would_refuse": problems, "ops": dict(counts),
                                    "side": str(side)}, sort_keys=True))
            return {"dry_run": True, "ops": ops, "problems": problems, "side": side}
        if side.exists():
            shutil.rmtree(side)
        (side / "blobs").mkdir(parents=True)
        for op in ops:
            if op["op"] == "write":
                (side / "blobs" / op["sha"]).write_bytes(op.pop("data"))
                if not op["new"]:
                    prev = op.pop("prev")
                    op["prev_sha"] = sha256(prev)
                    (side / "blobs" / op["prev_sha"]).write_bytes(prev)
        lanes_existed = analysis.layout.lanes_root.exists()
        manifest = _manifest(analysis, plan, "in-place")
        manifest["repo"] = str(repo)
        manifest["branch"] = branch
        manifest["head"] = _git(repo, "rev-parse", "HEAD").strip()
        for record in plan.files:
            manifest["files"].append({"source": record["source"], "dest": record["dest"],
                                      "sha_before": record["sha_before"], "sha_after": record["sha_after"],
                                      "changed": record["changed"], "edits": len(record["edits"]),
                                      "repins": record.get("repins", 0)})
        manifest["created"] = [c["to"] for c in plan.creates]
        write_side(side, analysis, plan, hazards, result, manifest, keep)
        ops_file.write_text(json.dumps({"ops": ops, "lanes_root_existed": lanes_existed,
                                        "lanes_root": str(analysis.layout.lanes_root),
                                        "head": manifest["head"], "branch": branch}, indent=1))
    done = set()
    if journal.exists():
        done = {json.loads(line)["i"] for line in journal.read_text().splitlines() if line.strip()}
    with journal.open("a") as handle:
        for i, op in enumerate(ops):
            if i in done:
                continue
            if not _op_done(repo, op):
                _run_op(repo, side, op)
            handle.write(json.dumps({"i": i, "op": op["op"]}) + "\n")
            handle.flush()
    # git does not track directories: drop the ones the moves emptied (rollback recreates them).
    for op in ops:
        for src in op.get("sources", [op.get("from")] if op["op"] in {"git-mv", "mv"} else []):
            _prune_empty_parents(repo / src, repo)
    (side / "complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    echo(f"in-place split complete: {len(ops)} operations; journal and side files in {side}")
    return {"dry_run": False, "ops": ops, "side": side}


def _rollback(repo: Path, side: Path, echo=print):
    ops_file = side / "ops.json"
    if not ops_file.exists():
        raise SystemExit(f"--rollback: nothing to roll back ({side} has no ops.json)")
    state = json.loads(ops_file.read_text())
    ops = state["ops"]
    journal = side / "journal.jsonl"
    done = []
    if journal.exists():
        done = [json.loads(line)["i"] for line in journal.read_text().splitlines() if line.strip()]
    for i in reversed(done):
        _undo_op(repo, side, ops[i])
    lanes_root = Path(state["lanes_root"])
    if not state["lanes_root_existed"]:
        _prune_empty(lanes_root)
    for op in ops:
        if op["op"] in {"git-mv", "mv"}:
            _prune_empty_parents(repo / op["to"], repo)
    shutil.rmtree(side)
    echo(f"rolled back {len(done)} operations")
    return {"rolled_back": len(done)}


def _prune_empty_parents(path: Path, stop: Path):
    parent = path.parent
    while parent != stop and is_inside(parent, stop):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


# ---------------------------------------------------------------- verify

def do_verify(out: Path, side: Path | None = None):
    out = out.resolve()
    side = (side or out / SIDE_DIR).resolve()
    manifest = json.loads((side / MANIFEST).read_text())
    hazards = json.loads((side / "hazards.json").read_text())
    packages = out / "packages"
    engine_name = Path(manifest["source_package_root"]).name
    engine_root = packages / engine_name
    layout = Layout(engine_root)
    scanner = Scanner(layout, strict=True)
    init = layout.modules.get(mid(ENGINE, ""))
    if init is not None:
        scanner.scan(init)
    pycache = side / "pycache"
    previous_prefix = sys.pycache_prefix
    sys.pycache_prefix = str(pycache)
    compile_errors = []
    try:
        for info in layout.files:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ok = compileall.compile_file(str(info.path), quiet=1, legacy=False)
            if not ok:
                compile_errors.append({"file": info.label, "error": buffer.getvalue().strip()[-400:]})
    finally:
        sys.pycache_prefix = previous_prefix
    unresolved, attr_missing, registry_pending, sibling_missing = [], [], [], []
    scans = {}
    for info in layout.files:
        scans[info.path] = scan = scanner.scan(info)
    top_names = {info.mid: scans[info.path] for info in layout.modules.values()}
    for info in layout.files:
        scan = scans[info.path]
        if scan.syntax_error:
            continue
        for ref in scan.refs:
            if ref.kind == "registry":
                if ref.target is None:
                    registry_pending.append({"file": info.label, "line": ref.line, "entry": ref.note,
                                             "text": ref.text})
                continue
            if ref.kind == "from-import":
                if ref.target is None:
                    unresolved.append({"file": info.label, "line": ref.line, "kind": "from-import",
                                       "text": ref.text, "lazy": ref.lazy, "optional": ref.optional})
                    continue
                for name in ref.names:
                    if name["module"] or name["name"] == "*":
                        continue
                    target_scan = top_names.get(name["target"])
                    if target_scan is not None and not target_scan.open_namespace and \
                            name["name"] not in target_scan.top_names:
                        attr_missing.append({"file": info.label, "line": ref.line, "module": name["target"],
                                             "name": name["name"]})
                continue
            if ref.kind in {"lazy-table"} and ref.target is None:
                unresolved.append({"file": info.label, "line": ref.line, "kind": ref.kind, "text": ref.text})
                continue
            written_ok = ref.written in HOMES or ref.written == "relative"
            if ref.target is None and written_ok and ref.kind not in {"registry"}:
                unresolved.append({"file": info.label, "line": ref.line, "kind": ref.kind, "text": ref.text,
                                   "lazy": ref.lazy, "optional": ref.optional})
        for sib in scan.siblings:
            if not (info.path.parent / sib["target"]).exists():
                sibling_missing.append({"file": info.label, **sib})
        for pref in scan.path_refs:
            if pref["kind"] in {"sibling-file", "relative-file"} and not (info.path.parent / pref["file"]).exists():
                sibling_missing.append({"file": info.label, "line": pref["line"], "name": pref["text"]})
    plan = json.loads((side / "plan.json").read_text())
    path_missing = []
    for check in plan.get("path_checks", []):
        if "expected" not in check or check.get("fix") == "rule":
            continue
        area = check["expected"].split("/")[1] if "/" in check["expected"] else ""
        if area in {"src", "tests", "qualification", "examples", "tools"} and \
                not (packages / check["expected"]).exists():
            path_missing.append(check)
    shutil.rmtree(pycache, ignore_errors=True)
    engine_src = engine_root / "src" / ENGINE
    lanes_in_engine = []
    for path in sorted(engine_src.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            data = path.read_bytes()
            if LANES.encode() in data:
                for i, line in enumerate(data.splitlines(), 1):
                    if LANES.encode() in line:
                        lanes_in_engine.append({"file": str(path.relative_to(packages)), "line": i,
                                                "text": line.decode("utf-8", "replace").strip()[:160]})
    lanes_in_engine_tests = []
    for path in sorted((engine_root / "tests").rglob("*.py")) if (engine_root / "tests").exists() else []:
        data = path.read_bytes()
        if LANES.encode() in data:
            lanes_in_engine_tests.append(str(path.relative_to(packages)))
    changed, identical, drift = 0, 0, []
    for record in manifest["files"]:
        data = (packages / record["dest"]).read_bytes()
        if sha256(data) != record["sha_after"]:
            drift.append(record["dest"])
        if record["sha_after"] != record["sha_before"]:
            changed += 1
        else:
            identical += 1
    pins = []
    dest_of = defaultdict(list)
    for record in manifest["files"]:
        dest_of[record["source"]].append(record["dest"])
    for pin in hazards["b_pinned_file_hashes"]:
        for dest in dest_of.get(pin["pinned_file"], []):
            now = sha256((packages / dest).read_bytes())
            if now != pin["sha256"]:
                pins.append({**pin, "dest": dest, "sha_in_copy": now})
    # Pins measured fresh in the result: a 64-hex literal in a .py file equal to the pre-split sha256
    # of a file the split changed is stale; one equal to a current file's sha256 is live.
    old_shas = {r["sha_before"]: r["dest"] for r in manifest["files"] if r["sha_after"] != r["sha_before"]}
    current = {}
    for info in layout.files:
        current.setdefault(info.sha, info.label)
    stale_py, live_py = [], 0
    for info in layout.files:
        for m in HEX64.finditer(info.data):
            value = m.group(0).decode()
            if value in current:
                live_py += 1
            elif value in old_shas:
                stale_py.append({"file": info.label, "line": info.data.count(b"\n", 0, m.start()) + 1,
                                 "value": value, "pinned": old_shas[value]})
    identity_missing = []
    for ident in plan.get("identity_strings", []):
        path = packages / ident["file"]
        if not path.exists() or ident["text"].encode() not in path.read_bytes():
            identity_missing.append(ident)
    identity_changes = []
    for item in hazards["a_registered_operations"]:
        if item["module_change"]:
            identity_changes.append({"hazard": "a", "file": item["file"], "line": item["line"],
                                     "module_change": item["module_change"], "qualname": item["qualname"]})
    for item in hazards["b_source_bound"]:
        if item["module_change"] or item["bytes_change"]:
            identity_changes.append({"hazard": "b", "file": item["file"], "line": item["line"],
                                     "what": item["what"], "module_change": item["module_change"],
                                     "bytes_change": item["bytes_change"]})
    result = {
        "schema": "mdb-split-lanes-verify-v1",
        "out": str(out),
        "files_checked": len(layout.files),
        "modules": {"engine": sum(1 for m in layout.modules.values() if m.home == ENGINE),
                    "lanes": sum(1 for m in layout.modules.values() if m.home == LANES)},
        "compile_errors": compile_errors,
        "syntax_errors": [{"file": s.info.label, "error": s.syntax_error} for s in scans.values() if s.syntax_error],
        "unresolved": unresolved,
        "attribute_not_found": attr_missing,
        "registry_pending_reregistration": registry_pending,
        "sibling_missing": sibling_missing,
        "path_references_missing_in_copy": path_missing,
        "mdb_lanes_in_engine_package": lanes_in_engine,
        "engine_tests_mentioning_mdb_lanes": lanes_in_engine_tests,
        "files_changed": changed, "files_byte_identical": identical, "copy_drift": drift,
        "pins_broken_in_copy": pins,
        "stale_python_pins": stale_py, "live_python_pins": live_py,
        "identity_strings_checked": len(plan.get("identity_strings", [])),
        "identity_strings_missing": identity_missing,
        "identity_changes": identity_changes,
        "layout_conflicts": layout.conflicts,
    }
    (side / "verify.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    return result


# ---------------------------------------------------------------- report

def do_report(out: Path, report_path: Path | None = None, side: Path | None = None):
    side = side or out / SIDE_DIR
    an = json.loads((side / "analysis.json").read_text())
    pl = json.loads((side / "plan.json").read_text())
    hz = json.loads((side / "hazards.json").read_text())
    vf = json.loads((side / "verify.json").read_text()) if (side / "verify.json").exists() else None
    c = an["counts"]
    w = []
    w.append("# mdb lane split: dry run\n")
    w.append(f"Source: `{an['package_root']}`; applied copy: `{out}`. Analysis {an['elapsed_s']} s.\n")
    w.append("Engine subpackages present: " + ", ".join(f"{k}={'yes' if v else 'no'}"
                                                       for k, v in an["engine_subpackages_present"].items()) + "\n")
    w.append("## 1. Engine vs lanes (measured)\n")
    w.append(f"| | modules | lines |\n|---|---|---|\n| engine | {c['engine_modules']} | {c['engine_lines']} |\n"
             f"| lanes | {c['lanes_modules']} | {c['lanes_lines']} |\n")
    w.append(f"Roots: {c['roots']}; pulled in by closure only: {c['closure_only']}. Keep list: {an['keep'] or 'empty'}"
             + (f"; unresolved keep entries: {an['keep_unresolved']}" if an["keep_unresolved"] else "") + "\n")
    w.append("Engine closure (non-root): " + ", ".join(f"`{m.split(':')[1]}`" for m in an["engine_closure"]) + "\n")
    w.append(f"Tests: engine {c['tests_engine']}, lanes {c['tests_lanes']}, straddling {c['tests_straddling']}. "
             f"Qualification files moved: {c['qualification_files']} ({c['qualification_lines']} lines).\n")
    if an["straddling_tests"]:
        w.append("Straddling tests (placed in lanes in the copy):\n")
        for label in an["straddling_tests"]:
            t = an["tests"][label]
            w.append(f"- `{label}` lane refs: {', '.join(x.split(':')[-1] for x in t['lane_refs'][:6])}"
                     f"{' (all function-local)' if t['lane_refs_function_local_only'] else ''}")
        w.append("")
    w.append("## 2. Edges to cut, ranked by drag\n")
    w.append("| # | importing file:line | imports | drag modules | drag lines | top-level? |\n|---|---|---|---|---|---|")
    ranked = [e for e in an["edges_to_cut"] if e["drag_modules"]]
    for i, e in enumerate(ranked, 1):
        w.append(f"| {i} | `{e['file']}`:{','.join(map(str, e['lines_at'][:4]))} | `{e['to'].split(':')[1]}` "
                 f"| {e['drag_modules']} | {e['drag_lines']} | {'top' if e['top_level'] else ('lazy' if e['all_lazy'] else 'guarded/TC')} |")
    w.append(f"\n{len(an['edges_to_cut']) - len(ranked)} further engine edges drag nothing on their own "
             "(another path keeps the target).\n")
    w.append("### Boundary: edges from root modules into flat modules\n")
    w.append("`reach` = flat modules the target pulls in by itself; `exclusive` = what leaves if only this edge is cut.\n")
    w.append("| root file:line | imports | kind | reach mod/lines | exclusive mod/lines |\n|---|---|---|---|---|")
    for e in an.get("boundary_edges", []):
        w.append(f"| `{e['file']}`:{','.join(map(str, e['lines_at'][:4]))} | `{e['to'].split(':')[1]}` | "
                 f"{'top' if e['top_level'] else ('lazy' if e['all_lazy'] else 'guarded')} {'/'.join(e['kinds'])} | "
                 f"{e['reach_modules']}/{e['reach_lines']} | {e['exclusive_modules']}/{e['exclusive_lines']} |")
    cyc = an.get("cycles", [])
    if cyc:
        w.append(f"\nImport cycles among flat engine modules: {len(cyc)}; largest has {cyc[0]['modules']} modules / "
                 f"{cyc[0]['lines']} lines: " + ", ".join(f"`{m.split(':')[1]}`" for m in cyc[0]["members"]))
    w.append(f"\nIf every boundary edge above is cut, the engine is the roots ({an.get('root_lines')} lines) plus "
             f"{len(an.get('closure_after_cutting_all_boundary_edges', []))} flat modules.")
    w.append("\nPer-root reach (flat modules each root pulls in on its own):\n")
    for r in an.get("per_root_reach", []):
        if r["flat_modules"]:
            w.append(f"- `{r['root'].split(':')[1] or '__init__'}`: {r['flat_modules']} modules / {r['flat_lines']} lines")
    w.append("")
    w.append("Per-module cut (remove the module from the engine entirely), top 25:\n")
    for m in an["module_cuts"][:25]:
        w.append(f"- `{m['module'].split(':')[1]}`: {m['drag_modules']} modules / {m['drag_lines']} lines; "
                 f"engine importers: {', '.join(x.split(':')[1] or '__init__' for x in m['engine_importers'][:6])}")
    w.append("\n## 3. Verification of the copy\n")
    if vf:
        w.append(f"Files checked {vf['files_checked']}; compile errors {len(vf['compile_errors'])}; "
                 f"unresolved {len(vf['unresolved'])}; attribute-not-found {len(vf['attribute_not_found'])}; "
                 f"`mdb_lanes` in engine package {len(vf['mdb_lanes_in_engine_package'])}; "
                 f"files rewritten {vf['files_changed']}, byte-identical {vf['files_byte_identical']}; "
                 f"registry entries pending re-registration {len(vf['registry_pending_reregistration'])}.\n")
        for key in ("compile_errors", "unresolved", "attribute_not_found", "sibling_missing",
                    "mdb_lanes_in_engine_package"):
            if vf[key]:
                w.append(f"{key}:")
                for item in vf[key][:80]:
                    w.append(f"- {json.dumps(item, sort_keys=True)[:300]}")
                w.append("")
    breaks = pl.get("path_breaks", [])
    w.append(f"File-path references (anchored at `__file__`) that would not land on their target after the move: "
             f"{len(breaks)}; fixed by rule {sum(b.get('fix') == 'rule' for b in breaks)}, manual "
             f"{sum(b.get('fix') != 'rule' for b in breaks)}; not file-anchored (not checked): "
             f"{pl['counts'].get('path_refs_not_file_anchored', 0)}")
    for b in breaks[:80]:
        w.append(f"- [{b.get('fix', 'manual')}] `{b['referrer']}`:{b['line']} `{b['text']}`: "
                 f"{b.get('why') or ('expected ' + b['expected'] + ', target now ' + ', '.join(b['target_dests']))}")
    if vf:
        w.append(f"\nPath references missing in the copy (measured): {len(vf.get('path_references_missing_in_copy', []))}\n")
    w.append(f"Plan errors (engine-side references to lane modules, not rewritten): {len(pl['errors'])}")
    for item in pl["errors"][:60]:
        w.append(f"- `{item['file']}`:{item['line']} {item.get('kind', '')} `{item.get('text', item.get('entry'))}`")
    w.append(f"\nRegistry (cli TOOLS) entries to re-register via mdb-lanes entry points: "
             + ", ".join(f"`{r['entry']}`" for r in pl["registry_reregister"]) + "\n")
    w.append("## 4. Identity hazards\n")
    w.append("### (a) registered operations\n")
    for r in hz["a_registered_operations"]:
        w.append(f"- `{r['file']}`:{r['line']} {r['call']} id={r['operation_id']} impl={r['implementation']} "
                 f"qualname={r['qualname']} module change={r['module_change']} bytes change={r['bytes_change']}")
    w.append("\n### (b) source-bound identities\n")
    for r in hz["b_source_bound"]:
        if r["module_change"] or r["bytes_change"]:
            w.append(f"- `{r['file']}`:{r['line']} {r['what']} (module change={bool(r['module_change'])}, "
                     f"bytes change={r['bytes_change']})")
    w.append(f"\nPinned file hashes found (64-hex literal equal to a current file sha256): {len(hz['b_pinned_file_hashes'])}; "
             f"pins whose file bytes the plan changes: {len(hz['b_pins_broken_by_rewrite'])}")
    code_pins = [p for p in hz["b_pins_broken_by_rewrite"] if p["pinned_by"].endswith(".py")]
    text_pins = [p for p in hz["b_pins_broken_by_rewrite"] if not p["pinned_by"].endswith(".py")]
    w.append(f"\nPinned in Python code (runtime/test checks that would now fail): {len(code_pins)} pins on "
             f"{len({p['pinned_file'] for p in code_pins})} files, from {len({p['pinned_by'] for p in code_pins})} files.")
    for p in code_pins[:80]:
        w.append(f"- `{p['pinned_file']}` pinned by `{p['pinned_by']}`:{p['line']} ({p['context']})")
    w.append(f"\nPinned in text/JSON records (historical seals): {len(text_pins)} pins on "
             f"{len({p['pinned_file'] for p in text_pins})} files, from {len({p['pinned_by'] for p in text_pins})} files.")
    if vf:
        w.append(f"\nMeasured in the copy: {len(vf['pins_broken_in_copy'])} pins no longer match.\n")
    w.append("### (c) module names in strings\n")
    w.append(f"In code: {len(hz['c_module_name_strings_in_code'])} string mentions of moved modules; "
             f"text files with dotted names of moved modules: {len(hz['c_module_names_in_text_files'])}; "
             f"text files with `mdb/<moved>.py` package paths: {len(hz['c_package_paths_in_text_files'])}.\n")
    w.append("### (d) PROVENANCE.json\n")
    prov = hz["d_provenance"]
    w.append(f"{len(prov)} entries name files; moving: {sum(bool(p['moves_to']) for p in prov)}; "
             f"bytes changing: {sum(p['bytes_change'] for p in prov)}; recorded sha already stale: "
             f"{sum(p['recorded_sha256_matches_now'] is False for p in prov)}\n")
    w.append("### (e) serialization sites in moved modules\n")
    for r in hz["e_serialization_sites"]:
        if r["module_change"]:
            w.append(f"- `{r['file']}`:{r['line']} {r['what']}")
    w.append("\n## 5. Not confident / dynamic\n")
    w.append(f"Unconfident spans (not rewritten): {len(pl['unconfident'])}")
    for u in pl["unconfident"][:60]:
        w.append(f"- `{u['file']}`:{u['line']} `{u['text']}`: {u['why']}")
    w.append(f"\nDynamic imports not resolved: {len(an['dynamic_imports'])}")
    for d in an["dynamic_imports"][:60]:
        w.append(f"- `{d['file']}`:{d['line']} {d['what']} prefix={d.get('prefix')}")
    w.append(f"\nMixed from-imports split: {len(pl['mixed_import_splits'])}; warnings: {len(pl['warnings'])}; "
             f"aux (examples/tools/experiments) references to lane modules, not rewritten: {len(pl['aux_lane_refs'])}")
    rp = pl.get("repin") or json.loads((side / "repin-map.json").read_text()) if (side / "repin-map.json").exists() else None
    if rp:
        c = rp["counts"]
        w.append("\n## 6. Re-pinning\n")
        w.append(f"Applied: {rp['applied']}; rounds to fixed point: {rp['rounds']}; Python pins re-pinned "
                 f"{c['python_pins_repinned']} in {c['python_files_repinned']} files; cycles {c['cycles']} "
                 f"({c['cycle_pins']} pins); ambiguous {c['ambiguous']}; JSON/text pins left stale "
                 f"{c['json_pins_stale']} in {c['json_files_stale']} files; predicted failing tests "
                 f"{c['predicted_failing_tests']}; Python pins outside the plan {c['python_pins_outside_plan']}.")
    w.append(f"\n## 7. Rule fixes and manual fixes\n\nRule fixes applied: {pl['counts'].get('rule_fixes', 0)}; "
             f"manual fixes listed in manual-fixes.md: {len(pl.get('manual_fixes', []))}; identity strings checked "
             f"{pl['counts'].get('identity_strings_checked', 0)}, edits refused {pl['counts'].get('identity_edits_refused', 0)}.")
    text = "\n".join(w) + "\n"
    target = report_path or (side / "report.md")
    target.write_text(text)
    return target


# ---------------------------------------------------------------- CLI

def default_package_root():
    return Path(__file__).resolve().parents[1]


def default_keep(package_root: Path):
    candidate = package_root / "tools" / "split_lanes_keep.txt"
    return candidate if candidate.exists() else None


def _dump(value, path):
    text = json.dumps(value, indent=1, sort_keys=True, default=str)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(prog="split_lanes", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "plan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--package-root", type=Path, default=default_package_root())
        p.add_argument("--keep", type=Path, default=None)
        p.add_argument("--assume-cut", type=Path, default=None,
                       help="file of 'importer:line module' edges treated as absent")
        if name != "analyze":
            p.add_argument("--strict-engine", action="store_true",
                           help="fail if the engine closure holds modules outside the engine subpackages/keep list")
            p.add_argument("--repin", action="store_true", help="re-pin sha256 literals in .py files")
            p.add_argument("--repin-json", action="store_true", help="also re-pin JSON/text files (opt-in)")
        if name == "apply":
            p.add_argument("--out", type=Path, default=None)
            p.add_argument("--clean", action="store_true")
            p.add_argument("--in-place", action="store_true", help="apply in this git checkout (git mv + writes)")
            p.add_argument("--expect-branch", default=None)
            p.add_argument("--allow-dirty", nargs="*", default=[], metavar="PATH")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--rollback", action="store_true")
        else:
            p.add_argument("--json", type=Path, default=None)
    p = sub.add_parser("repin")
    p.add_argument("out", type=Path)
    p.add_argument("--repin-json", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("out", type=Path)
    p.add_argument("--side", type=Path, default=None, help="side directory (in-place: <git dir>/split-lanes)")
    p.add_argument("--json", type=Path, default=None)
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    p.add_argument("--side", type=Path, default=None)
    p.add_argument("--to", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command in {"analyze", "plan", "apply"}:
        keep = args.keep or default_keep(args.package_root)
    if args.command == "analyze":
        analysis = Analysis(args.package_root, keep, assume_cut=args.assume_cut)
        value = analysis.to_json()
        _dump(value, args.json)
        print(json.dumps({**value["counts"], "assumed_cuts_matched": len(value["assumed_cuts_matched"]),
                          "assumed_cuts_unmatched": len(value["assumed_cuts_unmatched"])}, sort_keys=True))
    elif args.command == "plan":
        analysis, plan, hazards, result = prepare(args.package_root, keep, assume_cut=args.assume_cut,
                                                  repin=args.repin, repin_json=args.repin_json,
                                                  strict_engine=args.strict_engine)
        value = plan.to_json()
        value["hazards"] = hazards.to_json()
        value["repin_preview"] = result.to_json()
        _dump(value, args.json)
        print(json.dumps({**value["counts"], **{"repin_" + k: v for k, v in result.to_json()["counts"].items()},
                          "repin_rounds": result.rounds}, sort_keys=True))
    elif args.command == "apply":
        if args.in_place:
            if args.out:
                raise SystemExit("--in-place and --out are exclusive")
            result = do_apply_in_place(args.package_root, keep, expect_branch=args.expect_branch,
                                       allow_dirty=args.allow_dirty, dry_run=args.dry_run, resume=args.resume,
                                       rollback=args.rollback, repin=args.repin, repin_json=args.repin_json,
                                       assume_cut=args.assume_cut, strict_engine=args.strict_engine)
            return 0
        if not args.out:
            raise SystemExit("apply needs --out DIR (copy) or --in-place")
        _, plan, _, manifest = do_apply(args.package_root, args.out, keep, clean=args.clean, repin=args.repin,
                                        repin_json=args.repin_json, assume_cut=args.assume_cut,
                                        strict_engine=args.strict_engine)
        print(json.dumps(plan.to_json()["counts"], sort_keys=True))
    elif args.command == "repin":
        result = do_repin_copy(args.out, args.repin_json)
        print(json.dumps({**result.to_json()["counts"], "rounds": result.rounds}, sort_keys=True))
    elif args.command == "verify":
        result = do_verify(args.out, args.side)
        if args.json:
            _dump(result, args.json)
        summary = {k: (len(v) if isinstance(v, list) else v) for k, v in result.items()
                   if k not in {"schema", "out"}}
        print(json.dumps(summary, sort_keys=True))
    elif args.command == "report":
        print(do_report(args.out, args.to, args.side))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
