import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdb import symbol_reads
from mdb.core import pointers as P
from mdb.core.address import parse
from mdb.core.pointers import RecordError, Unresolved

# Produced by the pre-move mdb.symbol_reads (worktree base bf9d31d) on the inputs of
# _outputs(); the move must keep every byte. The "outside the root" error is left out because
# its text holds a machine-specific resolved path.
GOLDEN = json.loads(r"""
{
 "bounded": {
  "a": 0.5,
  "b": 3,
  "c": true,
  "d": [
   1,
   [
    2,
    3
   ],
   {
    "e": "\u00fc"
   }
  ],
  "f": null
 },
 "errors": {
  "array": "record.a holds an array or tensor; reference it by pointer",
  "bytes": "record.a holds raw bytes; reference the artifact by pointer",
  "items": "record has 65 items (> 64); store it by pointer",
  "nested": "record[1] holds more than 64 items across nested sequences; store it by pointer",
  "nonfinite": "record.a is not finite",
  "sha": "checkpoint sha256 must be a sha256 hex digest",
  "size": "record is 68144 bytes (> 65536); store it by pointer",
  "unresolved": "an unverified model name says what would resolve it ('resolve_by')"
 },
 "file_bare": {
  "digest": {
   "sha256": "f903a924500a33050df97cb71aa2dec71cb35d001c0216c265ff588099239243",
   "size": 12
  },
  "kind": "file",
  "locator": {
   "path": "plan.json",
   "root": "artifacts"
  }
 },
 "file_full": {
  "array": "x",
  "digest": {
   "sha256": "43d4724f2ec6e63299d5bcc11fbf50fc1b7183299e36ec485749de55245f0e89",
   "size": 40
  },
  "json_pointer": "/a",
  "kind": "file",
  "locator": {
   "path": "data/a.npy",
   "root": "artifacts"
  },
  "role": "mass",
  "selector": [
   1,
   2
  ]
 },
 "file_given_sha": {
  "digest": {
   "sha256": "abababababababababababababababababababababababababababababababab",
   "size": 12
  },
  "kind": "file",
  "locator": {
   "path": "plan.json",
   "root": "artifacts"
  }
 },
 "identity": {
  "checkpoint_sha256": "efefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefef",
  "identity_status": "verified",
  "name": "tiny"
 },
 "param_resolve_by": {
  "kind": "parameter",
  "model": {
   "identity_status": "unresolved",
   "name": "m",
   "resolve_by": "hf revision 123"
  },
  "parameter": "w"
 },
 "param_unresolved": {
  "kind": "parameter",
  "model": {
   "identity_status": "unresolved",
   "name": "flux2-klein",
   "resolve_by": "checkpoint sha256 of the file holding this parameter"
  },
  "parameter": "blocks.3.attn.qkv.weight"
 },
 "param_verified": {
  "kind": "parameter",
  "model": {
   "checkpoint_sha256": "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
   "identity_status": "verified",
   "name": "qwen2.5-1.5b"
  },
  "parameter": "model.layers.20.self_attn.v_proj.weight",
  "selector": [
   3
  ]
 },
 "run_bare": {
  "kind": "run",
  "run_id": "job-2"
 },
 "run_plan": {
  "kind": "run",
  "plan": {
   "digest": {
    "sha256": "f903a924500a33050df97cb71aa2dec71cb35d001c0216c265ff588099239243",
    "size": 12
   },
   "kind": "file",
   "locator": {
    "path": "plan.json",
    "root": "artifacts"
   }
  },
  "plan_sha256": "f903a924500a33050df97cb71aa2dec71cb35d001c0216c265ff588099239243",
  "run_id": "job-1"
 },
 "seal": "e7b82245e3826af80ee42eac2c003b6d853196b203e0642e5c55b5fe2ab48118",
 "sealed": {
  "fingerprint": "36f1a5a060ee20d292cd972679b57548294b54c9c8eb485e01b7b4b00244517c",
  "x": 1,
  "y": "z"
 }
}
""")


def _outputs(module, tmp_path):
    root_dir = tmp_path / "art"
    (root_dir / "data").mkdir(parents=True)
    (root_dir / "data" / "a.npy").write_bytes(b"abc\x00" * 10)
    (root_dir / "plan.json").write_bytes(b'{"plan": 1}\n')
    m = module
    root = m.declare_root("artifacts", root_dir)
    out = {
        "file_full": m.file_pointer(root_dir / "data" / "a.npy", root=root, array="x",
                                    selector=(1, 2), json_pointer="/a", role="mass"),
        "file_bare": m.file_pointer(root_dir / "plan.json", root=root),
        "file_given_sha": m.file_pointer(root_dir / "plan.json", root=root,
                                         sha256="sha256:" + "ab" * 32),
        "param_verified": m.parameter_pointer("qwen2.5-1.5b",
                                              "model.layers.20.self_attn.v_proj.weight",
                                              selector=(3,), checkpoint_sha256="cd" * 32),
        "param_unresolved": m.parameter_pointer("flux2-klein", "blocks.3.attn.qkv.weight"),
        "param_resolve_by": m.parameter_pointer("m", "w", resolve_by="hf revision 123"),
        "run_bare": m.run_pointer("job-2"),
        "identity": m.model_identity("tiny", checkpoint_sha256="ef" * 32),
    }
    plan = out["file_bare"]
    out["run_plan"] = m.run_pointer("job-1", plan_sha256=plan["digest"]["sha256"], plan=plan)
    record = {"a": np.float32(0.5), "b": np.int64(3), "c": np.bool_(True),
              "d": [1, [2, 3], {"e": "ü"}], "f": None}
    out["bounded"] = m.check_bounded(record)
    out["seal"] = m.seal_digest(out["bounded"])
    out["sealed"] = m._sealed({"x": 1, "y": "z"}, "t")
    errors = {}
    error_class = getattr(m, "SymbolReadError", None) or m.RecordError
    for name, call in {
        "array": lambda: m.check_bounded({"a": np.zeros(2)}),
        "bytes": lambda: m.check_bounded({"a": b"x"}),
        "nonfinite": lambda: m.check_bounded({"a": float("nan")}),
        "items": lambda: m.check_bounded(list(range(65))),
        "nested": lambda: m.check_bounded([[0] * 40, [0] * 40]),
        "size": lambda: m.check_bounded({"k%d" % i: "x" * 4000 for i in range(17)}),
        "unresolved": lambda: m.model_identity("m"),
        "sha": lambda: m.parameter_pointer("m", "w", checkpoint_sha256="xyz"),
    }.items():
        with pytest.raises(error_class) as caught:
            call()
        errors[name] = str(caught.value)
    out["errors"] = errors
    return out


@pytest.mark.parametrize("module", [symbol_reads, P], ids=["symbol_reads", "core.pointers"])
def test_moved_pointer_functions_are_byte_identical(module, tmp_path):
    out = _outputs(module, tmp_path)
    assert json.dumps(out, sort_keys=True, ensure_ascii=False) == json.dumps(
        GOLDEN, sort_keys=True, ensure_ascii=False)
    for key in ("file_full", "param_verified", "run_plan", "bounded"):
        assert P._canonical(out[key]) == P._canonical(GOLDEN[key])


def test_symbol_reads_reexports_the_same_objects():
    for name in ("MAX_INLINE_ITEMS", "MAX_META_BYTES", "MAX_RECORD_BYTES", "MAX_RECORD_ITEMS",
                 "MAX_TEXT_BYTES", "check_bounded", "seal_digest", "declare_root", "file_pointer",
                 "resolve_pointer", "verify_pointer", "model_identity", "parameter_pointer",
                 "run_pointer", "file_sha256", "_bounded", "_canonical", "_sealed", "_sha",
                 "_text", "_number", "_identity_record"):
        assert getattr(symbol_reads, name) is getattr(P, name), name
    assert symbol_reads.SymbolReadError is RecordError
    assert issubclass(RecordError, ValueError)


def test_core_imports_load_neither_torch_nor_numpy():
    code = ("import sys\n"
            "import mdb.core.address, mdb.core.pointers, mdb.core.records, mdb.core.index\n"
            "bad = [m for m in ('torch', 'numpy') if m in sys.modules]\n"
            "assert not bad, bad\n"
            "from mdb.core.pointers import check_bounded\n"
            "check_bounded({'a': [1, 2.5, None, 'x']})\n"
            "assert 'numpy' not in sys.modules\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent)


def test_bounded_without_numpy_loaded_matches_with_numpy():
    value = {"a": 1, "b": [0.5, True, None], "c": {"d": "e"}}
    assert P.check_bounded(value) == value


# --------------------------------------------------------------------------- occurrence pointers

def _occurrence(**changes):
    options = {"program": "stepped", "shape": [1, 12, 128], "dtype": "bfloat16",
               "sha256": "ab" * 32, "recipe": "a1" * 32, "event": 3, "cut": "cd" * 32}
    options.update(changes)
    address = options.pop("address", "t18.L20.v[kv0]")
    return P.occurrence_pointer(address, **options)


def test_occurrence_pointer_shape_and_round_trip():
    pointer = _occurrence(authority="measured")
    assert pointer == {
        "kind": "occurrence", "address": parse("t18.L20.v[kv0]").to_json(), "program": "stepped",
        "recipe": "a1" * 32, "event": 3, "cut": "cd" * 32,
        "value": {"shape": [1, 12, 128], "dtype": "bfloat16", "sha256": "ab" * 32},
        "authority": "measured"}
    assert P.validate_occurrence_pointer(pointer) == pointer
    assert P.validate_pointer(json.loads(json.dumps(pointer))) == pointer
    bare = _occurrence(sha256=None, recipe=None, event=None, cut=None)
    assert bare["value"]["sha256"] is None and bare["cut"] is None
    assert "authority" not in bare
    assert P.validate_occurrence_pointer(bare) == bare
    from_object = P.occurrence_pointer(parse("t18.L20.v[kv0]"), program="stepped", shape=[1],
                                       dtype="float32")
    assert from_object["address"] == parse("t18.L20.v[kv0]").to_json()


@pytest.mark.parametrize("changes, message", [
    ({"address": "cache.L20.v[row=18]"}, "not a state cell"),
    ({"address": "t18.L20.v~prefill"}, "differs from its address program"),
    ({"address": "t18.L20.zz"}, "unknown port"),
    ({"program": "has space"}, "short reference"),
    ({"program": ""}, "short reference"),
    ({"event": -1}, "event index"),
    ({"event": True}, "event index"),
    ({"cut": "not-a-sha"}, "sha256"),
    ({"sha256": "12"}, "sha256"),
    ({"shape": [1, -2]}, "shape"),
    ({"shape": "1,2"}, "shape"),
    ({"dtype": ""}, "dtype"),
    ({"authority": "guessed"}, "authority"),
    ({"recipe": "x" * 129}, "short reference"),
    ({"shape": list(range(65))}, "store it by pointer"),
])
def test_occurrence_pointer_refusals(changes, message):
    with pytest.raises((RecordError, ValueError), match=message):
        _occurrence(**changes)


def test_occurrence_pointer_validator_refuses_non_canonical_input():
    pointer = _occurrence()
    for broken in (
        {**pointer, "extra": 1},
        {k: v for k, v in pointer.items() if k != "cut"},
        {**pointer, "address": {**pointer["address"], "port": "bad name"}},
        {**pointer, "address": {**pointer["address"], "clock": {"token": -1}}},
        {**pointer, "address": "t18.L20.v[kv0]"},
        {**pointer, "value": {"shape": [1], "dtype": "f"}},
        {**pointer, "kind": "file"},
        {**pointer, "cut": "sha256:" + "cd" * 32},
    ):
        with pytest.raises(RecordError):
            P.validate_occurrence_pointer(broken)


def test_stored_pointers_accept_ports_outside_the_default_list():
    """Records are read without the engine's port program; port lists apply when given."""
    qwen3 = _occurrence(address={**parse("t1.L0.q").to_json(), "port": "q_norm"})
    assert P.validate_pointer(qwen3)["address"]["port"] == "q_norm"
    assert P.validate_occurrence_pointer(qwen3, ports=("q", "q_norm")) == qwen3
    with pytest.raises(RecordError, match="unknown port 'q_norm'"):
        P.validate_occurrence_pointer(qwen3, ports=("q",))
    with pytest.raises(RecordError, match="unknown port 'q_norm'"):
        _occurrence(address="t1.L0.q_norm")
    assert _occurrence(address="t1.L0.q_norm", ports=("q_norm",))["address"]["port"] == "q_norm"
    from mdb.core.address import Occurrence

    layerless = Occurrence(1, "q_norm")
    assert _occurrence(address=layerless)["address"]["clock"] == {"token": 1}
    with pytest.raises(RecordError, match="needs a layer"):
        _occurrence(address=layerless, ports=("q_norm",))


# --------------------------------------------------------------------------- resolvers

def test_unregistered_kind_raises_unresolved():
    assert "occurrence" not in P.resolvers()
    with pytest.raises(Unresolved, match="no resolver registered for pointer kind 'occurrence'"):
        P.resolve(_occurrence())
    with pytest.raises(Unresolved, match="'run'"):
        P.resolve(P.run_pointer("job-1"))
    with pytest.raises(Unresolved):
        P.resolve({"no": "kind"})
    assert set(P.resolvers()) == {"file", "parameter"}


def test_register_resolver_dispatches_and_refuses_duplicates():
    calls = []

    def fake(pointer, **context):
        calls.append(context)
        return "value"

    try:
        P.register_resolver("occurrence", fake)
        assert P.resolve(_occurrence(), cache="c") == "value"
        assert calls == [{"cache": "c"}]
        with pytest.raises(ValueError, match="already registered"):
            P.register_resolver("occurrence", fake)
        P.register_resolver("occurrence", lambda p, **c: "replaced", replace=True)
        assert P.resolve(_occurrence()) == "replaced"
    finally:
        P.unregister_resolver("occurrence")
    with pytest.raises(Unresolved):
        P.resolve(_occurrence())


def test_file_resolver(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"payload")
    root = P.declare_root("data", tmp_path)
    pointer = P.file_pointer(tmp_path / "x.bin", root=root)
    assert P.resolve(pointer, roots={"data": tmp_path}) == tmp_path / "x.bin"
    with pytest.raises(Unresolved, match="needs roots") as caught:
        P.resolve(pointer)
    assert caught.value.refused is False
    with pytest.raises(Unresolved, match="not declared"):
        P.resolve(pointer, roots={"other": tmp_path})
    (tmp_path / "x.bin").write_bytes(b"PAYLOAD")
    with pytest.raises(Unresolved, match="does not match") as caught:
        P.resolve(pointer, roots={"data": tmp_path})
    assert caught.value.refused is True
    (tmp_path / "x.bin").unlink()
    with pytest.raises(Unresolved, match="missing"):
        P.resolve(pointer, roots={"data": tmp_path})


def test_parameter_resolver():
    weights = {"w": [[1, 2], [3, 4]]}
    verified = P.parameter_pointer("m", "w", selector=(1,), checkpoint_sha256="ab" * 32)
    assert P.resolve(verified, parameters=weights, checkpoint_sha256="ab" * 32) == [3, 4]
    with pytest.raises(Unresolved, match="needs the context's checkpoint"):
        P.resolve(verified, parameters=weights)
    with pytest.raises(Unresolved, match="checkpoint differs") as caught:
        P.resolve(verified, parameters=weights, checkpoint_sha256="cd" * 32)
    assert caught.value.refused
    unverified = P.parameter_pointer("m", "w")
    with pytest.raises(Unresolved, match="identity is unresolved"):
        P.resolve(unverified, parameters=weights)
    assert P.resolve(unverified, parameters=weights, allow_unverified=True) == weights["w"]
    with pytest.raises(Unresolved, match="not in this model"):
        P.resolve(P.parameter_pointer("m", "v"), parameters=weights, allow_unverified=True)
    with pytest.raises(Unresolved, match="parameters= or model="):
        P.resolve(unverified, allow_unverified=True)
    with pytest.raises(Unresolved, match="selector is outside"):
        P.resolve(P.parameter_pointer("m", "w", selector=(9,)), parameters=weights,
                  allow_unverified=True)

    class Model:
        def named_parameters(self):
            return iter(weights.items())

    assert P.resolve(unverified, model=Model(), allow_unverified=True) == weights["w"]


def test_validate_pointer_kinds(tmp_path):
    (tmp_path / "x").write_bytes(b"1")
    file_pointer = P.file_pointer(tmp_path / "x", root=P.declare_root("r", tmp_path))
    for pointer in (file_pointer, P.parameter_pointer("m", "w"), P.run_pointer("r1"),
                    {"kind": "cas", "digest": "ab" * 32}):
        assert P.validate_pointer(pointer) == pointer
    for broken in ({"kind": "file", "locator": {}}, {"kind": "parameter", "parameter": ""},
                   {"kind": "run", "run_id": " x"}, {"digest": "x"},
                   {"kind": "cas", "data": np.zeros(2)}):
        with pytest.raises(RecordError):
            P.validate_pointer(broken)


def test_a_resolver_returning_none_raises_unresolved():
    P.register_resolver("occurrence", lambda pointer, **context: None)
    try:
        with pytest.raises(Unresolved, match="returned no value"):
            P.resolve(_occurrence())
    finally:
        P.unregister_resolver("occurrence")
    # falsy values that are values are returned
    P.register_resolver("occurrence", lambda pointer, **context: 0)
    try:
        assert P.resolve(_occurrence()) == 0
    finally:
        P.unregister_resolver("occurrence")


def test_malformed_stored_pointers_never_raise_keyerror_or_typeerror(tmp_path):
    good_file = {"kind": "file", "locator": {"root": "data", "path": "x"},
                 "digest": {"sha256": "ab" * 32, "size": 1}}
    good_parameter = P.parameter_pointer("m", "w")
    roots = {"data": tmp_path}
    weights = {"w": [[1, 2], [3, 4]]}
    broken_files = [
        {"kind": "file"}, {**good_file, "locator": None}, {**good_file, "locator": {"root": "data"}},
        {**good_file, "locator": {"root": 3, "path": "x"}},
        {**good_file, "locator": {"root": "data", "path": ["x"]}},
        {**good_file, "digest": {"sha256": "ab" * 32}}, {**good_file, "digest": 5},
        {**good_file, "digest": {"sha256": "ab" * 32, "size": "1"}},
        {**good_file, "digest": {"sha256": "ab" * 32, "size": True}},
        {**good_file, "digest": {"sha256": "nothex", "size": 1}},
        {k: v for k, v in good_file.items() if k != "locator"},
    ]
    broken_parameters = [
        {"kind": "parameter"}, {**good_parameter, "model": None}, {**good_parameter, "model": 3},
        {**good_parameter, "model": {"name": "m"}}, {**good_parameter, "parameter": 7},
        {**good_parameter, "parameter": ""}, {**good_parameter, "selector": 3},
        {**good_parameter, "selector": "ab"},
        {k: v for k, v in good_parameter.items() if k != "model"},
        {**good_parameter, "model": {"name": "m", "identity_status": "verified"}},
    ]
    for pointer in broken_files:
        with pytest.raises((Unresolved, RecordError)):
            P.resolve(pointer, roots=roots)
    for pointer in broken_parameters:
        with pytest.raises((Unresolved, RecordError)):
            P.resolve(pointer, parameters=weights, allow_unverified=True)
    with pytest.raises(Unresolved):
        P.resolve(P.parameter_pointer("m", "w"), parameters=[1, 2], allow_unverified=True)
    with pytest.raises(Unresolved):
        P.resolve(P.parameter_pointer("m", "w"), model=object(), allow_unverified=True)
    assert P.resolve(P.parameter_pointer("m", "w", selector=[1, 0]), parameters=weights,
                     allow_unverified=True) == 3
    with pytest.raises(Unresolved, match="outside the parameter"):
        P.resolve(P.parameter_pointer("m", "w", selector=[5]), parameters=weights,
                  allow_unverified=True)


# --------------------------------------------------------------------------- C1: path escape

def _escape_world(tmp_path):
    import hashlib
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir(), outside.mkdir()
    (root / "ok.bin").write_bytes(b"inside")
    secret = outside / "secret.txt"
    secret.write_bytes(b"TOP SECRET")
    digest = {"sha256": hashlib.sha256(b"TOP SECRET").hexdigest(), "size": 10}
    return root, outside, secret, digest


def _file(path, digest):
    return {"kind": "file", "locator": {"root": "art", "path": path}, "digest": digest}


def test_file_pointer_cannot_leave_its_root(tmp_path):
    root, outside, secret, digest = _escape_world(tmp_path)
    roots = {"art": str(root)}
    for path in ("../outside/secret.txt", str(secret), "a/../../outside/secret.txt",
                 "..\\outside\\secret.txt", "\\outside\\secret.txt", "", "  ", "x\0y", "C:/x"):
        pointer = _file(path, digest)
        with pytest.raises(RecordError):
            P.validate_pointer(pointer)
        with pytest.raises(Unresolved) as caught:
            P.resolve(pointer, roots=roots)
        assert caught.value.refused is True


def test_symlink_out_of_the_root_is_refused_and_one_in_the_root_allowed(tmp_path):
    import os
    root, outside, secret, digest = _escape_world(tmp_path)
    roots = {"art": str(root)}
    os.symlink(secret, root / "escape.txt")
    os.symlink(outside / "nope", root / "escape-missing.txt")
    os.symlink(root / "ok.bin", root / "inside-link.bin")
    with pytest.raises(Unresolved, match="outside root") as caught:
        P.resolve(_file("escape.txt", digest), roots=roots)
    assert caught.value.refused is True
    ok = P.file_pointer(root / "ok.bin", root=P.declare_root("art", root))
    # the path returned is the resolved real path that was checked (a later symlink swap of the
    # unresolved name cannot redirect the caller)
    assert P.resolve(_file("inside-link.bin", ok["digest"]), roots=roots) == (root / "ok.bin").resolve()
    assert P.resolve(_file("ok.bin", ok["digest"]), roots=roots) == (root / "ok.bin").resolve()


def test_old_pointer_api_is_confined_like_resolve(tmp_path):
    """Audit round 3: resolve_pointer / verify_pointer hashed ``../outside``, absolute paths and
    symlinks out of the root, and a missing outside file raised FileNotFoundError."""
    import os
    root, outside, secret, digest = _escape_world(tmp_path)
    roots = {"art": str(root)}
    os.symlink(secret, root / "to-existing")
    os.symlink(outside / "nope", root / "to-missing")
    os.symlink(outside, root / "dir-link")
    os.symlink(root / "loop2", root / "loop1")
    os.symlink(root / "loop1", root / "loop2")
    paths = ("../outside/secret.txt", "../outside/nope", str(secret), str(outside / "nope"),
             "to-existing", "to-missing", "dir-link/secret.txt", "dir-link/nope",
             "..\\outside\\secret.txt", "", "x\0y", "loop1")
    errors = set()
    for path in paths:
        for fn in (P.resolve_pointer, P.verify_pointer):
            for d in (digest, {"sha256": "0" * 64, "size": 1}):
                with pytest.raises(RecordError) as caught:
                    fn(_file(path, d), roots)
                errors.add((type(caught.value), str(caught.value)))
    assert errors == {(RecordError, P.ESCAPE)}      # one outcome: existing or missing target
    ok = P.file_pointer(root / "ok.bin", root=P.declare_root("art", root))
    assert P.resolve_pointer(ok, roots) == root / "ok.bin"
    assert P.verify_pointer(ok, roots) == root / "ok.bin"
    with pytest.raises(RecordError, match="undeclared root"):
        P.resolve_pointer(ok, {})
    with pytest.raises(RecordError, match="digest"):
        P.verify_pointer({**ok, "digest": {**ok["digest"], "size": 99}}, roots)


def test_outside_root_outcome_does_not_depend_on_the_target(tmp_path, monkeypatch):
    """Existing, missing, right-digest and wrong-digest targets: one exception, no hashing."""
    import os
    root, outside, secret, digest = _escape_world(tmp_path)
    roots = {"art": str(root)}
    os.symlink(secret, root / "to-existing")
    os.symlink(outside / "nope", root / "to-missing")
    hashed = []
    monkeypatch.setattr(P, "file_sha256", lambda path: hashed.append(path) or "x")
    wrong = {"sha256": "0" * 64, "size": 1}
    outcomes, links = set(), set()
    for path in ("to-existing", "to-missing"):
        for d in (digest, wrong):
            with pytest.raises(Unresolved) as caught:
                P.resolve(_file(path, d), roots=roots)
            links.add((caught.value.refused, caught.value.reason, type(caught.value)))
    for path in (str(secret), str(outside / "nope"), "../outside/secret.txt", "../outside/nope"):
        for d in (digest, wrong):
            with pytest.raises(Unresolved) as caught:
                P.resolve(_file(path, d), roots=roots)
            outcomes.add((caught.value.refused, caught.value.reason, type(caught.value)))
    assert hashed == []
    assert len(links) == 1 and len(outcomes) == 1     # one outcome, whatever the target is


def test_nested_relative_path_resolves(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.bin").write_bytes(b"deep")
    pointer = P.file_pointer(tmp_path / "a" / "b" / "x.bin", root=P.declare_root("d", tmp_path))
    assert pointer["locator"]["path"] == "a/b/x.bin"
    assert P.validate_pointer(pointer) == pointer
    assert P.resolve(pointer, roots={"d": tmp_path}) == tmp_path / "a" / "b" / "x.bin"


# --------------------------------------------------------------------------- C6: never raw

def test_resolve_never_raises_a_raw_exception():
    for pointer in ({"kind": ["file"]}, {"kind": {"a": 1}}, {"kind": 5}, {"kind": None}):
        with pytest.raises(Unresolved) as caught:
            P.resolve(pointer)
        assert type(caught.value) is Unresolved
    with pytest.raises(Unresolved) as caught:
        P.resolve({"kind": ["file"]})
    assert caught.value.refused is True

    class Boom(KeyError):
        pass

    for name, error in (("boom-key", KeyError("missing")), ("boom-type", TypeError("x")),
                        ("boom-record", RecordError("y")), ("boom-os", OSError("z"))):
        P.register_resolver(name, lambda p, _e=error, **c: (_ for _ in ()).throw(_e))
        try:
            with pytest.raises(Unresolved) as caught:
                P.resolve({"kind": name})
        finally:
            P.unregister_resolver(name)
        assert caught.value.refused is True
        assert caught.value.reason == f"resolver {name} failed: {type(error).__name__}"
        assert caught.value.__cause__ is error
    P.register_resolver("own-unresolved", lambda p, **c: (_ for _ in ()).throw(
        Unresolved("mine", refused=False)))
    try:
        with pytest.raises(Unresolved, match="^mine$") as caught:
            P.resolve({"kind": "own-unresolved"})
        assert caught.value.refused is False
    finally:
        P.unregister_resolver("own-unresolved")


def test_malformed_pointer_fuzz_has_no_raw_exceptions(tmp_path):
    import random
    rng = random.Random(5)
    junk = [None, 1, "x", [], {}, True, 1.5, {"a": 1}, ["art"], {"root": "art"}, "/etc/hosts", -1, 10**30]
    (tmp_path / "ok.bin").write_bytes(b"inside")
    total = raw = 0
    for kind in ("file", "parameter", "run", "occurrence", "cas", None, 5, ["file"]):
        for _ in range(150):
            pointer = {"kind": kind}
            for field in ("locator", "digest", "parameter", "model", "selector", "run_id",
                          "address", "array"):
                if rng.random() < 0.5:
                    pointer[field] = rng.choice(junk)
            if rng.random() < 0.3:
                pointer["locator"] = {"root": rng.choice(["art", 1, None]),
                                      "path": rng.choice(["ok.bin", 1, None, "../x"])}
            if rng.random() < 0.3:
                pointer["digest"] = {"sha256": rng.choice(["ab" * 32, "z", None]),
                                     "size": rng.choice([6, "6", True, None])}
            ctx = rng.choice([{}, {"roots": {"art": str(tmp_path)}}, {"roots": {"art": 5}},
                              {"parameters": {"w": [1]}}, {"parameters": [1]},
                              {"model": object()}, {"allow_unverified": True, "parameters": {}}])
            total += 1
            try:
                P.resolve(pointer, **ctx)
            except Unresolved:
                pass
            except Exception:  # noqa: BLE001
                raw += 1
    assert total >= 1200 and raw == 0
