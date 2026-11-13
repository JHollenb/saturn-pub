"""Tests for tools/split_lanes.py on a synthetic mini package (pure Python, no torch)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "split_lanes.py"
# A plain constant, not a path expression: split_lanes classifies tests that build
# "<package>/qualification" paths as lane tests, and this synthetic path is not one.
QDIR = "qualification"


def _load_tool():
    spec = importlib.util.spec_from_file_location("split_lanes_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SL = _load_tool()

LANE_B_CRLF = (b'"""Lane b with CRLF endings."""\r\n'
               b"from .util import helper  # keep this comment\r\n"
               b"\r\n"
               b"\r\n"
               b"def f():\r\n"
               b"    return helper()\r\n")
LANE_C_NO_NEWLINE = b'from . import lane_b\nVALUE = lane_b.f()'

FILES = {
    "src/mdb/__init__.py": '''"""Lazy API."""
from importlib import import_module

_EXPORTS = {"Thing": ("store", "Thing")}


def __getattr__(name):
    module, attribute = _EXPORTS[name]
    return getattr(import_module("." + module, __name__), attribute)
''',
    "src/mdb/__main__.py": "from .cli import main\n\nraise SystemExit(main())\n",
    "src/mdb/cli.py": '''import importlib
from pathlib import Path

WORKER = Path(__file__).parent / "workers/run_w.py"

TOOLS = {"store": "store"}
TOOLS["lane"] = "lane_tool"


def main(argv=None):
    argv = list(argv or [])
    return importlib.import_module("." + TOOLS[argv[0]], __package__).main(argv[1:])
''',
    "src/mdb/core/__init__.py": "",
    "src/mdb/core/address.py": '''from ..util import helper


def lazy():
    from mdb import heavy
    return heavy, helper
''',
    "src/mdb/core/records.py": '''import importlib
from unittest import mock


def load():
    return importlib.import_module("mdb.dyn_target")


def patched():
    return mock.patch("mdb.lane_b.f")
''',
    "src/mdb/util.py": "from . import errors\n\n\ndef helper():\n    return errors.CODE\n",
    "src/mdb/errors.py": "CODE = 1\n",
    "src/mdb/heavy.py": "import mdb.heavy_dep\n\nX = mdb.heavy_dep.Y\n",
    "src/mdb/heavy_dep.py": "Y = 2\n" + "# padding\n" * 40,
    "src/mdb/store.py": "class Thing:\n    pass\n",
    "src/mdb/dyn_target.py": "Z = 3\n",
    "src/mdb/lane_tool.py": '''"""Uses mdb.lane_b in a docstring mention that must stay."""
import subprocess
import sys

import mdb.lane_b
import mdb.lane_c as lc
from mdb import lane_c as c2
from mdb import util, lane_b
from .util import helper
from . import lane_c

NAME = "mdb.lane_b"
NOT_A_MODULE = "mdb.lane_bx"
OTHER = "xmdb.lane_b"
PAYLOAD = {"module": "mdb.lane_b", "note": "see mdb.lane_b for details"}


def run():
    subprocess.run([sys.executable, "-m", "mdb.lane_b"], check=False)
    return mdb.lane_b.f(), lc, c2, util, lane_b, helper, lane_c


def main(argv):
    return 0
''',
    "src/mdb/workers/__init__.py": "",
    "src/mdb/workers/run_w.py": "print('worker')\n",
    "src/mdb/lane_pkg/__init__.py": "",
    "src/mdb/lane_pkg/inner.py": "from ..util import helper\nfrom .. import lane_b\n",
    "src/mdb/lane_ops.py": '''import hashlib
from pathlib import Path

from saturn_pub import Act


def build():
    def operation(values):
        return values
    Act.register_operation("mdb.test.op", "1", operation, pure=True)
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class _Closure:
    def __call__(self, reads):
        return reads


def closure_act():
    operation = _Closure()
    Act.register_operation("mdb.test.closure", "1", operation, pure=True)
''',
    "src/mdb/lane_ops2.py": "from saturn_pub import Act\n\nfrom . import lane_ops as _prev\n\n\n"
                            "def act():\n    operation = _prev._Closure()\n"
                            "    Act.register_operation('mdb.test.reuse', '1', operation, pure=True)\n",
    "tests/conftest.py": '"""guard"""\n',
    "tests/test_engine.py": '''from unittest import mock

from mdb.core import address


def test_x():
    with mock.patch("mdb.util.helper"):
        assert address
''',
    "tests/test_lane.py": '''from pathlib import Path
from unittest import mock

from mdb.lane_b import f


DATA = Path(__file__).parents[1] / "experiments/some.json"


def test_f(monkeypatch):
    monkeypatch.setattr("mdb.lane_b.f", lambda: 0)
    with mock.patch("mdb.lane_b.f"):
        assert f
''',
    "tests/test_straddle.py": "from mdb.util import helper\nfrom mdb.lane_b import f\n",
    "tests/test_optional.py": '''from mdb.util import helper

try:
    import mdb.lane_c as lane_c
except ImportError:
    lane_c = None
''',
    "tests/test_qual.py": '''from pathlib import Path

from mdb.util import helper

Q = Path(__file__).parents[1] / "qualification"
''',
    "qualification/q.py": '''import runpy
from pathlib import Path

from attention_support import g
from mdb.lane_b import f

H = runpy.run_path(str(Path(__file__).with_name("attention_support.py")))
''',
    "qualification/attention_support.py": "import mdb.util\n\n\ndef g():\n    return mdb.util.helper()\n",
    "pyproject.toml": '[project]\nname = "mdb-runtime"\n[project.scripts]\nmdb = "mdb.cli:main"\n',
}


def make_repo(tmp_path, extra=None, fake_git=True):
    repo = tmp_path / "repo"
    if fake_git:
        (repo / ".git").mkdir(parents=True)
    root = repo / "packages" / "mdb"
    for rel, text in {**FILES, **(extra or {})}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "src/mdb/lane_b.py").write_bytes(LANE_B_CRLF)
    (root / "src/mdb/lane_c.py").write_bytes(LANE_C_NO_NEWLINE)
    lane_b_sha = hashlib.sha256(LANE_B_CRLF).hexdigest()
    lane_c_sha = hashlib.sha256(LANE_C_NO_NEWLINE).hexdigest()
    (root / QDIR / "pins.json").write_text(json.dumps({"mdb/lane_b.py": lane_b_sha,
                                                               "mdb/lane_c.py": lane_c_sha}))
    if extra is not None:
        pins_py = f'LANE_B = "{lane_b_sha}"\nLANE_C = "{lane_c_sha}"\n'
        (root / QDIR / "pins_py.py").write_text(pins_py)
        (root / QDIR / "pin2.py").write_text(f'PINS_PY = "{hashlib.sha256(pins_py.encode()).hexdigest()}"\n')
    return repo, root


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture()
def applied(repo, tmp_path):
    _, root = repo
    out = tmp_path / "out"
    analysis, plan, hazards, manifest = SL.do_apply(root, out, None)
    return root, out, analysis, plan, hazards


def side(analysis, key):
    return analysis.side[SL.mid("mdb", key)]


def test_closure_and_sides(repo):
    _, root = repo
    analysis = SL.Analysis(root)
    engine = {k for k in ("", "__main__", "cli", "core", "workers", "workers.run_w", "core.address", "core.records", "util", "errors",
                          "heavy", "heavy_dep", "store", "dyn_target")}
    lanes = {"lane_tool", "lane_b", "lane_c", "lane_pkg", "lane_pkg.inner", "lane_ops", "lane_ops2"}
    assert {k for k in engine if side(analysis, k) == "engine"} == engine
    assert {k for k in lanes if side(analysis, k) == "lanes"} == lanes
    # relative depth 2 (core/address -> util), depth 1 (util -> errors), lazy (heavy), importlib literal
    # (dyn_target) and the lazy export table (store) all pull into the closure.
    chain = analysis.chain(SL.mid("mdb", "errors"))
    assert [step["to"] for step in chain][-1] == "mdb:errors"
    data = analysis.to_json()
    registry = {(r["entry"], r["side"]) for r in data["registry_entries"]}
    assert ("TOOLS[lane]", "lanes") in registry and ("TOOLS[store]", "engine") in registry


def test_edges_to_cut_ranked_by_drag(repo):
    _, root = repo
    ranked = SL.Analysis(root).edges_to_cut()
    top = ranked[0]
    assert (top["from"], top["to"]) == ("mdb:core.address", "mdb:heavy")
    assert set(top["dragged"]) == {"mdb:heavy", "mdb:heavy_dep"}
    assert top["all_lazy"] and not top["top_level"]
    assert top["drag_lines"] == 3 + 41


def test_import_forms_are_detected(repo):
    _, root = repo
    analysis = SL.Analysis(root)
    by_file = {scan.info.label: scan for scan in analysis.scans.values()}
    kinds = {ref.kind for ref in by_file["mdb/src/mdb/lane_tool.py"].refs}
    assert {"import", "from-import", "dash-m", "payload-module"} <= kinds
    assert {ref.kind for ref in by_file["mdb/src/mdb/core/records.py"].refs} >= {"importlib", "patch-target"}
    assert {ref.kind for ref in by_file["mdb/tests/test_lane.py"].refs} >= {"patch-target"}
    table = [ref for ref in by_file["mdb/src/mdb/cli.py"].refs if ref.kind == "registry"]
    assert {ref.target for ref in table} == {"mdb:store", "mdb:lane_tool"}
    lazy = [ref for ref in by_file["mdb/src/mdb/core/address.py"].refs if ref.lazy]
    assert lazy and lazy[0].names[0]["target"] == "mdb:heavy"
    q = by_file["mdb/qualification/q.py"]
    assert [s["name"] for s in q.siblings] == ["attention_support"]
    assert any(p["kind"] == "sibling-file" and p["run_path"] for p in q.path_refs)


def test_test_classification(repo):
    _, root = repo
    tests = SL.Analysis(root).tests
    sides = {Path(label).name: rec["side"] for label, rec in tests.items()}
    assert sides["test_engine.py"] == "engine"
    assert sides["test_lane.py"] == "lanes"
    assert sides["test_straddle.py"] == "straddling"
    assert sides["test_optional.py"] == "engine"
    assert sides["test_qual.py"] == "straddling"  # engine import plus a qualification path
    conftest = tests["mdb/tests/conftest.py"]
    assert conftest["placement"] == ["engine", "lanes"]


def test_apply_rewrites_are_span_exact(applied):
    root, out, analysis, plan, _ = applied
    lanes = out / "packages/mdb-lanes/src/mdb_lanes"
    tool = (lanes / "lane_tool.py").read_text()
    assert "import mdb_lanes.lane_b\n" in tool
    assert "import mdb_lanes.lane_c as lc\n" in tool
    assert "from mdb_lanes import lane_c as c2\n" in tool
    assert "from mdb import util; from mdb_lanes import lane_b\n" in tool
    assert "from mdb.util import helper\n" in tool
    assert "from . import lane_c\n" in tool  # lane -> lane relative stays relative
    assert "return mdb_lanes.lane_b.f()" in tool
    assert '"-m", "mdb_lanes.lane_b"' in tool
    assert '{"module": "mdb_lanes.lane_b", "note": "see mdb.lane_b for details"}' in tool
    assert 'NAME = "mdb.lane_b"' in tool  # exact dotted string outside a recognised context: reported only
    assert 'NOT_A_MODULE = "mdb.lane_bx"' in tool and 'OTHER = "xmdb.lane_b"' in tool
    assert tool.startswith('"""Uses mdb.lane_b in a docstring mention that must stay."""')
    # CRLF and comments survive; only the module span changes.
    assert (lanes / "lane_b.py").read_bytes() == LANE_B_CRLF.replace(b"from .util import", b"from mdb.util import")
    # No final newline, nothing to change: byte-identical.
    assert (lanes / "lane_c.py").read_bytes() == LANE_C_NO_NEWLINE
    inner = (lanes / "lane_pkg/inner.py").read_text()
    assert inner == "from mdb.util import helper\nfrom .. import lane_b\n"
    lane_test = (out / "packages/mdb-lanes/tests/test_lane.py").read_text()
    assert 'monkeypatch.setattr("mdb_lanes.lane_b.f"' in lane_test
    assert 'mock.patch("mdb_lanes.lane_b.f")' in lane_test
    assert "from mdb_lanes.lane_b import f" in lane_test
    q = (out / "packages/mdb-lanes/qualification/q.py").read_text()
    assert "from mdb_lanes.lane_b import f" in q and "from attention_support import g" in q
    records = {r["source"]: r for r in plan.files}
    for label, record in records.items():
        if not record["changed"]:
            assert record["sha_before"] == record["sha_after"]
    assert not records["mdb/src/mdb/lane_c.py"]["changed"]
    assert not records["mdb/src/mdb/util.py"]["changed"]
    # Mention reported as an unrecognised-context string, never rewritten.
    mentions = [m for m in plan.a.scans[root / "src/mdb/lane_tool.py"].mentions if m["exact"]]
    assert any(m["text"] == "mdb.lane_b" for m in mentions)


def test_engine_reference_to_lane_is_an_error_not_a_rewrite(applied):
    root, out, analysis, plan, _ = applied
    errors = [e for e in plan.errors if e["file"].endswith("core/records.py")]
    assert errors and errors[0]["text"] == "mdb.lane_b.f"
    assert 'mock.patch("mdb.lane_b.f")' in (out / "packages/mdb/src/mdb/core/records.py").read_text()
    assert {r["entry"] for r in plan.registry} == {"TOOLS[lane]"}


def test_optional_lane_import_in_engine_test(applied):
    root, out, analysis, plan, _ = applied
    text = (out / "packages/mdb/tests/test_optional.py").read_text()
    assert "import mdb_lanes.lane_c as lane_c" in text
    assert plan.optional_engine_test_refs


def test_verify_copy(applied):
    root, out, *_ = applied
    result = SL.do_verify(out)
    assert result["compile_errors"] == []
    assert result["unresolved"] == []
    assert result["sibling_missing"] == []
    # The only mdb_lanes mention allowed in the engine is none: the lane verb stays as a pending registry entry.
    assert result["mdb_lanes_in_engine_package"] == []
    assert [r["entry"] for r in result["registry_pending_reregistration"]] == ["TOOLS[lane]"]
    assert result["copy_drift"] == []
    assert result["files_changed"] >= 5 and result["files_byte_identical"] >= 5


def test_identity_hazards(applied):
    root, out, analysis, plan, hazards = applied
    data = hazards.to_json()
    regs = [r for r in data["a_registered_operations"] if r["file"].endswith("lane_ops.py")]
    assert regs and regs[0]["qualname"] == "build.<locals>.operation"
    assert regs[0]["module_change"] == ["mdb.lane_ops", "mdb_lanes.lane_ops"]
    assert any(r["file"].endswith("lane_ops.py") for r in data["b_source_bound"])
    broken = {p["pinned_file"] for p in data["b_pins_broken_by_rewrite"]}
    assert broken == {"mdb/src/mdb/lane_b.py"}  # lane_c moves but keeps its bytes
    assert {p["pinned_file"] for p in data["b_pinned_file_hashes"]} >= {"mdb/src/mdb/lane_c.py"}
    assert any(m["modules"] for m in data["c_package_paths_in_text_files"])
    verify = SL.do_verify(out)
    assert {p["pinned_file"] for p in verify["pins_broken_in_copy"]} == {"mdb/src/mdb/lane_b.py"}


def test_refuses_to_write_inside_source_tree(repo):
    repo_dir, root = repo
    before = snapshot(repo_dir)
    for target in (root / "split-out", repo_dir / "elsewhere"):
        with pytest.raises(SystemExit):
            SL.do_apply(root, target, None)
    assert snapshot(repo_dir) == before


def test_source_tree_untouched_by_apply(repo, tmp_path):
    repo_dir, root = repo
    before = snapshot(repo_dir)
    SL.do_apply(root, tmp_path / "out2", None)
    assert snapshot(repo_dir) == before


def test_idempotent_on_split_copy(applied, tmp_path):
    root, out, *_ = applied
    split_root = out / "packages" / "mdb"
    analysis = SL.Analysis(split_root)
    plan = SL.Plan(analysis)
    counts = plan.to_json()["counts"]
    assert counts["moves"] == 0 and counts["copies"] == 0 and counts["creates"] == 0
    assert counts["edits"] == 0 and counts["files_changed"] == 0
    assert counts["data_files_to_place"] == 0
    # A new lane module appearing after the split is the only thing planned.
    (split_root / "src/mdb/lane_new.py").write_text("from mdb_lanes import lane_b\nfrom .util import helper\n")
    plan = SL.Plan(SL.Analysis(split_root))
    assert [m["to"] for m in plan.moves] == ["mdb-lanes/src/mdb_lanes/lane_new.py"]
    edited = [f for f in plan.files if f["changed"]]
    assert [f["source"] for f in edited] == ["mdb/src/mdb/lane_new.py"]


def test_keep_file_adds_roots(repo, tmp_path):
    _, root = repo
    keep = tmp_path / "keep.txt"
    keep.write_text("# extra roots\nlane_c  # keep it\n")
    analysis = SL.Analysis(root, keep)
    assert side(analysis, "lane_c") == "engine" and side(analysis, "lane_b") == "engine"
    assert analysis.roots["mdb:lane_c"] == "keep"


def test_path_file_edge_and_path_breaks(applied):
    root, out, analysis, plan, hazards = applied
    edges = [e for e in analysis.edges["mdb:cli"] if e["kind"] == "path-file"]
    assert [e["to"] for e in edges] == ["mdb:workers.run_w"]
    breaks = [b for b in plan.path_breaks if b["referrer"].endswith("test_lane.py")]
    assert breaks and "experiments" in breaks[0]["why"]
    q = [c for c in plan.path_checks if c["referrer"].endswith("qualification/q.py")]
    assert q and all(c["ok"] for c in q)
    assert SL.do_verify(out)["path_references_missing_in_copy"] == []


def test_instance_registration_qualname(applied):
    *_, hazards = applied
    regs = {r["operation_id"]: r for r in hazards.to_json()["a_registered_operations"]}
    assert regs["'mdb.test.closure'"]["qualname"].startswith("_Closure (instance")


def test_instance_of_class_from_other_module(applied):
    *_, hazards = applied
    regs = {r["operation_id"]: r for r in hazards.to_json()["a_registered_operations"]}
    reuse = regs["'mdb.test.reuse'"]
    assert reuse["owner_module"] == "mdb:lane_ops"
    assert reuse["module_change"] == ["mdb.lane_ops", "mdb_lanes.lane_ops"]


# ---------------------------------------------------------------- rule edits, re-pinning, in-place

EXTRA = {
    "src/mdb/lane_dyn.py": '''import importlib

MODS = ("lane_b", "lane_c")
SHARED = ("mdb.lane_b", "mdb.util")


def _load(name):
    return importlib.import_module("mdb." + name)


A = _load("lane_b")
B = _load("lane_c")


def proof(seal, declared):
    rows = {}
    for name in MODS:
        module = importlib.import_module("mdb." + name)
        rows[name] = (module, seal["packages"]["mdb"].get(name + ".py"), declared.get("mdb/" + name + ".py"))
    return rows


def hashes():
    return {name: importlib.import_module(name) for name in SHARED}


INFO = {"api": "mdb.lane_b.f", "public_module": "mdb.lane_c and mdb.lane_b", "source": {"demo": "mdb.lane_b.v1"}}
''',
    "src/mdb/lane_ship.py": '''from pathlib import Path

RANKING = Path(__file__).parent / "util.py"
VIEW = "viewer.html"
''',
    "src/mdb/assets/viewer.html": "<html></html>\n",
    "src/mdb/lane_pay.py": '''def package_files(name):
    return {}


SOURCES = [*((f"mdb/{name}", path) for name, path in package_files("mdb").items())]
''',
    "src/mdb/core/caps.py": 'CAPS = {"x": {"api": "mdb.lane_b.f"}}\n',
    "src/mdb/core/rows.py": 'def version(d):\n    return d\n\n\nROWS = (("mdb", "mdb-runtime", version("mdb-runtime")),)\n',
    "qualification/scr.py": '''import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
OUT = HERE.parent / "results/out.json"


def scratch(tmp_path):
    return tmp_path / "results"
''',
    "tests/test_env.py": '''from pathlib import Path

from mdb.lane_b import f

ENV = {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
''',
    "tests/test_pins.py": '''import hashlib
import json
from pathlib import Path

from mdb.lane_b import f

PINS = Path(__file__).parents[1] / "qualification" / "pins.json"


def test_pins():
    data = json.loads(PINS.read_text())
    assert hashlib.sha256(b"") and data


def test_other():
    assert f
''',
    "tests/test_src_path.py": '''from pathlib import Path

from mdb.lane_b import f

HERE = Path(__file__).parents[1]
LANE_SRC = HERE / "src/mdb/lane_c.py"
ENGINE_SRC = HERE / "src/mdb/util.py"
''',
    "experiments/exp.py": "import mdb.util\nfrom mdb.lane_b import f\n",
    "tools/gen.py": '''from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / "qualification"
''',
}


@pytest.fixture()
def repo2(tmp_path):
    return make_repo(tmp_path, EXTRA)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_dynamic_import_rules(repo2, tmp_path):
    _, root = repo2
    out = tmp_path / "out"
    analysis, plan, hazards, manifest = SL.do_apply(root, out, None)
    dyn = (out / "packages/mdb-lanes/src/mdb_lanes/lane_dyn.py").read_text()
    assert 'return importlib.import_module("mdb_lanes." + name)' in dyn          # call-site enumeration
    assert 'module = importlib.import_module("mdb_lanes." + name)' in dyn        # loop over literal tuple
    assert 'seal["packages"]["mdb_lanes"].get(name + ".py")' in dyn              # same-loop seal key
    assert 'declared.get("mdb_lanes/" + name + ".py")' in dyn                    # same-loop source-hash key
    assert 'SHARED = ("mdb_lanes.lane_b", "mdb.util")' in dyn                    # element-wise tuple
    assert '"api": "mdb_lanes.lane_b.f"' in dyn
    assert '"public_module": "mdb_lanes.lane_c and mdb_lanes.lane_b"' in dyn
    assert '{"demo": "mdb.lane_b.v1"}' in dyn                                    # identity string untouched
    assert any(i["text"] == "mdb.lane_b.v1" for i in plan.identity_strings)
    assert plan.identity_refused == []
    engine_caps = (out / "packages/mdb/src/mdb/core/caps.py").read_text()
    assert '"api": "mdb.lane_b.f"' in engine_caps                                # engine side: manual, not edited
    verify = SL.do_verify(out)
    assert verify["unresolved"] == [] and verify["compile_errors"] == []
    assert verify["mdb_lanes_in_engine_package"] == []
    assert verify["identity_strings_missing"] == []
    manual = plan.manual_final()
    rules = {m["rule"] for m in manual}
    assert {"api-string", "payload-distributions", "package-seal", "cli-entry-points"} <= rules
    caps = next(m for m in manual if m["rule"] == "api-string")
    assert caps["replacement"] == 'CAPS = {"x": {"api": "mdb_lanes.lane_b.f"}}'
    seal = next(m for m in manual if m["rule"] == "package-seal")
    assert 'for _package in ("mdb", "mdb_lanes")' in seal["replacement"]
    assert (out / "_split/manual-fixes.md").read_text().count("\n## ") == len(manual)


def test_mixed_prefix_import_is_manual(repo2, tmp_path):
    _, root = repo2
    (root / "src/mdb/lane_mixed.py").write_text(
        'import importlib\n\n\ndef load_all():\n'
        '    return [importlib.import_module("mdb." + name) for name in ("lane_b", "util")]\n')
    analysis, plan, hazards, result = SL.prepare(root, None)
    item = next(m for m in plan.manual_final() if m["file"].endswith("lane_mixed.py"))
    assert item["rule"] == "prefix-import"
    assert '{"lane_b": "mdb_lanes.lane_b", "util": "mdb.util"}[name]' in item["replacement"]
    record = next(f for f in plan.files if f["source"].endswith("lane_mixed.py"))
    assert not record["changed"]


def test_path_rules(repo2, tmp_path):
    _, root = repo2
    out = tmp_path / "out"
    analysis, plan, hazards, manifest = SL.do_apply(root, out, None)
    scr = (out / "packages/mdb-lanes/qualification/scr.py").read_text()
    assert 'sys.path[0:0] = [str(HERE.parent / "src"), str(HERE.parent.parent / "mdb" / "src")]' in scr
    assert 'OUT = HERE.parent.parent / "mdb" / "results/out.json"' in scr
    assert 'return tmp_path / "results"' in scr                                   # not file-anchored: untouched
    env = (out / "packages/mdb-lanes/tests/test_env.py").read_text()
    assert ('"PYTHONPATH": __import__("os").pathsep.join([str(Path(__file__).parents[1] / "src"), '
            'str(Path(__file__).parents[1].parent / "mdb" / "src")])') in env
    lane = (out / "packages/mdb-lanes/tests/test_lane.py").read_text()
    assert 'DATA = Path(__file__).parents[1].parent / "mdb" / "experiments/some.json"' in lane
    ship = (out / "packages/mdb-lanes/src/mdb_lanes/lane_ship.py").read_text()
    assert 'RANKING = Path(__import__("importlib.util").util.find_spec("mdb.util").origin)' in ship
    src_path = (out / "packages/mdb-lanes/tests/test_src_path.py").read_text()
    assert 'LANE_SRC = HERE / "src/mdb_lanes/lane_c.py"' in src_path
    assert 'ENGINE_SRC = HERE.parent / "mdb" / "src/mdb/util.py"' in src_path
    gen = (out / "packages/mdb/tools/gen.py").read_text()
    assert 'destination = root.parent / "mdb-lanes" / "qualification"' in gen
    counts = plan.to_json()["counts"]
    assert counts["path_breaks_manual"] == 0 and counts["path_breaks_fixed_by_rule"] >= 6
    assert counts["path_refs_not_file_anchored"] >= 1
    assert SL.do_verify(out)["compile_errors"] == []


def test_repin_fixed_point_and_json_report(repo2, tmp_path):
    _, root = repo2
    lane_b_old = _sha(LANE_B_CRLF)
    out = tmp_path / "out"
    analysis, plan, hazards, manifest = SL.do_apply(root, out, None, repin=True)
    rp = plan.repin
    new_lane_b = _sha((out / "packages/mdb-lanes/src/mdb_lanes/lane_b.py").read_bytes())
    pins_py = (out / "packages/mdb-lanes/qualification/pins_py.py").read_bytes()
    assert f'LANE_B = "{new_lane_b}"'.encode() in pins_py
    assert f'LANE_C = "{_sha(LANE_C_NO_NEWLINE)}"'.encode() in pins_py            # unchanged file: pin stays
    pin2 = (out / "packages/mdb-lanes/qualification/pin2.py").read_text()
    assert pin2 == f'PINS_PY = "{_sha(pins_py)}"\n'                                # second-order pin
    assert rp.rounds == 2 and rp.cycles == []
    stale = [p for p in rp.json_stale if p["file"].endswith("qualification/pins.json")]
    assert [p["old"] for p in stale] == [lane_b_old]
    tests = {t["test"]: t for t in rp.predicted_tests}
    assert tests["mdb-lanes/tests/test_pins.py::test_pins"]["hashes"] is True
    assert "mdb-lanes/tests/test_pins.py::test_other" not in tests
    side = json.loads((out / "_split/repin-map.json").read_text())
    assert {"old_sha256", "new_sha256", "path_before", "path_after"} <= set(side["map"][0])
    assert any(m["path_after"].endswith("pins_py.py") for m in side["map"])
    verify = SL.do_verify(out)
    assert verify["stale_python_pins"] == []
    assert verify["live_python_pins"] >= 3
    # JSON is never edited by default.
    assert not (out / "packages/mdb-lanes/qualification/pins.json").exists() or lane_b_old in \
        (out / "packages/mdb-lanes/qualification/pins.json").read_text()


def test_repin_json_opt_in_and_repin_subcommand(repo2, tmp_path):
    _, root = repo2
    out = tmp_path / "out"
    SL.do_apply(root, out, None)
    assert SL.do_verify(out)["stale_python_pins"]                                  # before re-pinning
    result = SL.do_repin_copy(out, repin_json=True)
    assert result.json_stale == []
    pins = json.loads((out / "packages/mdb-lanes/qualification/pins.json").read_text())
    assert pins["mdb/lane_b.py"] == _sha((out / "packages/mdb-lanes/src/mdb_lanes/lane_b.py").read_bytes())
    assert SL.do_verify(out)["stale_python_pins"] == []


def test_repin_cycle_guard():
    a, b, c = b'X = "%s"\n', b'Y = "%s"\n', b'Z = "%s"\n'
    fake_a, fake_b = "a" * 64, "b" * 64
    contents = {"a.py": a % fake_b.encode(), "b.py": b % fake_a.encode(), "c.py": c % fake_a.encode()}
    before = {"a.py": fake_a, "b.py": fake_b, "c.py": _sha(contents["c.py"])}
    cur, rounds, resolvable, cycles, cycle_pins, ambiguous = SL.repin_fixed_point(contents, before)
    assert [cy["members"] for cy in cycles] == [["a.py", "b.py"]]
    assert len(cycle_pins) == 2
    assert cur["a.py"] == contents["a.py"] and cur["b.py"] == contents["b.py"]
    assert cur["c.py"] == c % _sha(contents["a.py"]).encode() and rounds == 1


def test_assume_cut_and_strict_engine(repo, tmp_path):
    _, root = repo
    with pytest.raises(SystemExit) as info:
        SL.prepare(root, None, strict_engine=True)
    assert "mdb:heavy" in str(info.value) and "core/address.py:5" in str(info.value)
    cuts = tmp_path / "cuts.txt"
    cuts.write_text("# lazy heavy import\nsrc/mdb/core/address.py:5 mdb.heavy\n")
    analysis = SL.Analysis(root, assume_cut=cuts)
    assert side(analysis, "heavy") == "lanes" and side(analysis, "heavy_dep") == "lanes"
    assert analysis.cut_matched and not analysis.cut_unmatched
    plan = SL.Plan(analysis)
    assert any(e["file"].endswith("core/address.py") for e in plan.errors)
    bad = tmp_path / "bad.txt"
    bad.write_text("src/mdb/core/address.py:99 mdb.heavy\n")
    with pytest.raises(SystemExit):
        SL.prepare(root, None, assume_cut=bad)


def test_lanes_pyproject(repo2, tmp_path):
    import tomllib
    _, root = repo2
    analysis, plan, hazards, result = SL.prepare(root, None)
    content = next(c["content"] for c in plan.creates if c["to"] == "mdb-lanes/pyproject.toml")
    data = tomllib.loads(content)
    assert data["project"]["dependencies"] == ["mdb-runtime"]
    assert data["project"]["entry-points"]["mdb.lanes"] == {"lane": "mdb_lanes.lane_tool:main"}
    assert "mdb_lanes/assets/viewer.html" in content
    assert any(d["to"] == "mdb-lanes/src/mdb_lanes/assets/viewer.html" for d in plan.data_files)


def _git(repo, *args):
    import subprocess
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def make_git_repo(tmp_path):
    repo, root = make_repo(tmp_path, EXTRA, fake_git=False)
    records = root / "src/mdb/core/records.py"  # in-place refuses plans with engine -> lane errors
    records.write_text(records.read_text().replace('mock.patch("mdb.lane_b.f")', 'mock.patch("mdb.util.helper")'))
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "split@example.invalid")
    _git(repo, "config", "user.name", "split test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, root


def tree(repo):
    return {p.relative_to(repo).as_posix(): p.read_bytes() for p in sorted(repo.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(repo).parts}


def test_in_place_apply_verify_idempotent_rollback(tmp_path):
    repo, root = make_git_repo(tmp_path)
    before = tree(repo)
    with pytest.raises(SystemExit):
        SL.do_apply_in_place(root, None, expect_branch="other", repin=True, echo=lambda *_: None)
    printed = []
    dry = SL.do_apply_in_place(root, None, expect_branch="main", repin=True, dry_run=True, echo=printed.append)
    assert dry["dry_run"] and any(line.startswith("git mv -- ") or line.startswith("mkdir -p ") for line in printed)
    assert tree(repo) == before
    SL.do_apply_in_place(root, None, expect_branch="main", repin=True, echo=lambda *_: None)
    side = Path(_git(repo, "rev-parse", "--absolute-git-dir").strip()) / "split-lanes"
    assert (repo / "packages/mdb-lanes/src/mdb_lanes/lane_b.py").exists()
    assert not (root / "src/mdb/lane_b.py").exists()
    assert (repo / "packages/mdb-lanes/qualification/pins.json").exists()        # non-.py qualification moves
    assert not (root / "qualification").exists()                                   # emptied directories pruned
    assert (repo / "packages/mdb-lanes/src/mdb_lanes/assets/viewer.html").exists()
    status = _git(repo, "status", "--porcelain")
    assert "??" not in status                                                      # every new file is tracked
    verify = SL.do_verify(repo, side)
    assert verify["compile_errors"] == [] and verify["unresolved"] == []
    assert verify["stale_python_pins"] == [] and verify["identity_strings_missing"] == []
    replan = SL.Plan(SL.Analysis(root)).to_json()["counts"]
    assert replan["moves"] == 0 and replan["edits"] == 0 and replan["creates"] == 0
    with pytest.raises(SystemExit):                                                # dirty tree now
        SL.do_apply_in_place(root, None, expect_branch="main", echo=lambda *_: None)
    SL.do_apply_in_place(root, None, expect_branch="main", rollback=True, echo=lambda *_: None)
    assert tree(repo) == before
    assert _git(repo, "status", "--porcelain") == ""


def test_in_place_resume_after_interruption(tmp_path, monkeypatch):
    repo, root = make_git_repo(tmp_path)
    real = SL._run_op
    calls = {"n": 0}

    def flaky(repo_, side_, op):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("simulated crash")
        return real(repo_, side_, op)

    monkeypatch.setattr(SL, "_run_op", flaky)
    with pytest.raises(RuntimeError):
        SL.do_apply_in_place(root, None, expect_branch="main", repin=True, echo=lambda *_: None)
    with pytest.raises(SystemExit):                                                 # incomplete run blocks a fresh one
        SL.do_apply_in_place(root, None, expect_branch="main", allow_dirty=["packages"], echo=lambda *_: None)
    monkeypatch.setattr(SL, "_run_op", real)
    SL.do_apply_in_place(root, None, expect_branch="main", resume=True, echo=lambda *_: None)
    side = Path(_git(repo, "rev-parse", "--absolute-git-dir").strip()) / "split-lanes"
    verify = SL.do_verify(repo, side)
    assert verify["compile_errors"] == [] and verify["unresolved"] == [] and verify["copy_drift"] == []


def test_in_place_allow_dirty(tmp_path):
    repo, root = make_git_repo(tmp_path)
    (repo / "notes.txt").write_text("scratch\n")
    with pytest.raises(SystemExit):
        SL.do_apply_in_place(root, None, expect_branch="main", echo=lambda *_: None)
    result = SL.do_apply_in_place(root, None, expect_branch="main", allow_dirty=["notes.txt"], dry_run=True,
                                  echo=lambda *_: None)
    assert result["problems"] == []


def test_aux_scripts_rewritten_vendor_untouched(repo2, tmp_path):
    _, root = repo2
    (root / "experiments/vendor").mkdir(parents=True)
    (root / "experiments/vendor/frozen.py").write_text("from mdb.lane_b import f\n")
    analysis, plan, hazards, result = SL.prepare(root, None)
    exp = plan.contents["mdb/experiments/exp.py"]
    assert exp == b"import mdb.util\nfrom mdb_lanes.lane_b import f\n"
    assert plan.contents["mdb/experiments/vendor/frozen.py"] == b"from mdb.lane_b import f\n"
    assert any(r["file"].endswith("vendor/frozen.py") for r in plan.aux_lane_refs)
