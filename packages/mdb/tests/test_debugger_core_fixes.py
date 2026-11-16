"""Regression tests for the 2026-11-14 debugger-core code audit (F1-F12): each reproduces the
audit's scenario on a tiny CPU model and asserts the fix (refusal or honest record)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")

from mdb.analysis import circuit_map, sae  # noqa: E402
from mdb.analysis.branches import edit  # noqa: E402
from mdb.analysis.finite import f64  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.pools import Pool  # noqa: E402
from mdb.core import custody  # noqa: E402
from mdb.core.records import seal_record  # noqa: E402
from mdb.engine import core_file  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

P = [1, 5, 9, 3, 7]
D = [2, 8, 4, 6, 3]
ITEMS = [{"id": f"i{k}", "ids": [1 + k, 5 + k, 9, 3 + k, 7], "roles": {"carrier": 2},
          "country": "AB"[k % 2]} for k in range(4)]
CELL = "cache.L1.v[kv0,row=@carrier]"


@pytest.fixture(scope="module")
def dbg():
    return open_tiny("qwen3", "fp32")


@pytest.fixture(scope="module")
def base(dbg):
    run = dbg.run(P, name="base")
    run.until("t4.logits")
    return run


@pytest.fixture(scope="module")
def other(dbg):
    run = dbg.run(D, name="other")
    run.until("t4.logits")
    return run


# ------------------------------------------------------------------ F1 pools

def _saved(dbg, tmp_path, items=ITEMS):
    pool = dbg.pool(items, CELL, labels=("country",), name="cv")
    return pool, pool.save(tmp_path / "pool.json")


def test_f1_reordered_items_refused(dbg, tmp_path):
    _, path = _saved(dbg, tmp_path)
    data = json.loads(path.read_text())
    data["items"] = data["items"][::-1]
    path.write_text(json.dumps(data))
    with pytest.raises(Refusal, match="items differ"):
        Pool.load(path)


def test_f1_edited_label_and_tokens_refused(dbg, tmp_path):
    _, path = _saved(dbg, tmp_path)
    data = json.loads(path.read_text())
    data["items"][0]["labels"]["country"] = "Z"
    path.write_text(json.dumps(data))
    with pytest.raises(Refusal, match="labels differ"):
        Pool.load(path)
    _, path = _saved(dbg, tmp_path)
    data = json.loads(path.read_text())
    data["items"][1]["origin"]["tokens_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(Refusal, match="tokens digest"):
        Pool.load(path)


def test_f1_untampered_roundtrip_and_loo_excludes_item(dbg, tmp_path):
    pool, path = _saved(dbg, tmp_path)
    back = Pool.load(path)
    assert back.ids == pool.ids and torch.equal(back.values, pool.values)
    loo = back.mean_for("i0")
    assert "i0" not in loo.pointer["items"]
    assert torch.equal(loo.value, f64(pool.values[1:]).mean(0))


def test_f1_duplicate_prompt_excluded_by_default(dbg):
    dup = [dict(ITEMS[0], id="a"), dict(ITEMS[0], id="b"), dict(ITEMS[1], id="c")]
    pool = dbg.pool(dup, CELL, name="dup")
    assert pool.duplicates() == 1
    loo = pool.mean_for("a")
    assert loo.pointer["items"] == ["c"] and loo.pointer["duplicates_dropped"] == 1
    by_id = pool.mean_for("a", exclude="id")
    assert by_id.pointer["items"] == ["b", "c"]
    assert pool.donor("a", rule="any").pointer["donor"] == "c"
    assert pool.seal()["body"]["duplicates"] == 1
    with pytest.raises(Refusal):
        pool.mean_for("a", exclude="nonsense")


def test_f12_pool_label_precedence_and_kernel_path(dbg):
    items = [dict(it, labels={"country": "N" + it["country"]}) for it in ITEMS]
    pool = dbg.pool(items, CELL, labels=("country",), name="prec")
    assert pool.items[0]["labels"]["country"] == "NA"          # nested labels win everywhere
    assert pool._label(items[0], "country") == "NA"
    body = pool.seal()["body"]
    assert body["custody"]["kernel_path"] == "mdb-prefill-batched"
    assert pool.items[0]["origin"]["prefill"] == 4
    ports = dbg.pool(ITEMS[:2], "t@last.L1.h", name="ports")
    assert ports.seal()["body"]["custody"]["kernel_path"] == "mdb-stepped"


# ------------------------------------------------------------------ F2 / F8 / F9 core files

def _dump(dbg, tmp_path, name="core.pt"):
    run = dbg.run(P, name="src", prefill=2)
    run.until("t4.logits")
    path = tmp_path / name
    return run, path, run.dump(str(path), row=2)


def test_f2_tampered_prompt_refused(dbg, tmp_path):
    _, path, _ = _dump(dbg, tmp_path)
    assert core_file.load(path).run(dbg.adapter).core_check["ok"]
    data = torch.load(path, weights_only=True)
    data["prompt"] = [1, 5, 9, 4, 8]
    torch.save(data, path)
    with pytest.raises(ValueError, match="prompt differs"):
        core_file.load(path)


def test_f2_tampered_prompt_refused_without_capture(dbg, tmp_path):
    run = dbg.run(P, name="nocap", prefill=2)
    run.until("t3.L0.h")
    path = tmp_path / "nocap.pt"
    run.dump(str(path), row=2, capture=())
    data = torch.load(path, weights_only=True)
    data["prompt"] = [1, 5, 9, 4, 8]
    torch.save(data, path)
    with pytest.raises(ValueError, match="prompt differs"):
        core_file.load(path)


def test_f2_tampered_tokens_refused(dbg, tmp_path):
    _, path, _ = _dump(dbg, tmp_path, "tok.pt")
    data = torch.load(path, weights_only=True)
    data["tokens"] = [int(t) + 1 for t in data["tokens"]]
    torch.save(data, path)
    with pytest.raises(ValueError, match="tokens differ"):
        core_file.load(path)


def test_f8_post_split_edit_not_captured(dbg, tmp_path):
    run = dbg.run(P, name="edited", prefill=2)
    run.until("t3.L0.h")
    run.set("t3.L0.m", 5.0, "add")
    run.until("t4.logits")
    with pytest.raises(ValueError, match="after t2.start"):
        run.dump(str(tmp_path / "x.pt"), row=2, capture=("t4.logits",))
    record = run.dump(str(tmp_path / "y.pt"), row=2)          # default capture is dropped
    assert record["body"]["post_split_edits"] == 1 and record["body"]["captured"] == []
    assert core_file.load(tmp_path / "y.pt").run(dbg.adapter).core_check["state"] == "unresolved"


def test_f9_other_weights_is_a_replica(dbg, tmp_path):
    _, path, _ = _dump(dbg, tmp_path, "id.pt")
    cf = core_file.load(path)
    assert cf.run(dbg.adapter, check=False).core["replica"] is False
    cf.record["model_identity"] = "same shape, other weights"       # e.g. Base vs Instruct
    run = cf.run(dbg.adapter, check=False)
    assert run.core["same_identity"] is False and run.core["replica"] is True


# ------------------------------------------------------------------ F3 / F4 / F5 / F12 holds, branches

def test_f3_branches_on_held_run_reference_has_the_holds(dbg, base, other):
    run = dbg.run(P, name="bh")
    run.hold("t@last.L*.p", source=other)
    br = dbg.branches(run, Margin(4, 3, 7), at="t4.L0.h")
    br.add("M", [edit("t4.L0.m", other.ref("t4.L0.m"))])
    br.add_null("self", "self", like="M")
    rows = {r["name"]: r for r in br.run()}
    assert torch.equal(br.reference.read("t4.logits"), br.base_logits)
    assert rows["self"]["dM"]["value"] == 0.0
    assert [h["name"] for h in run.holds] == ["hold1"]          # arm holds released, base kept


def test_f4_edit_at_held_port_refused_or_recorded(dbg, base):
    run = dbg.run(P, name="ov")
    run.hold("t@last.L1.mlp_norm_scale", source=base)
    run.until("t4.L1.h")
    with pytest.raises(ValueError, match="held by hold1"):
        run.set("t4.L1.n", 0.0)
    assert all(e["origin"]["kind"] == "hold" for e in run.edits)       # nothing of ours staged
    run.set("t4.L1.n", 0.0, override_hold=True)
    run.until("t4.logits")
    entry = [e for e in run.committed_edits() if e["layer"] == 1 and e["port"] == "n"
             and e.get("origin", {}).get("kind") != "hold"][0]
    assert entry["origin"]["overridden_by"] == ["hold1"]
    assert torch.equal(run.read("t4.logits"), base.read("t4.logits"))   # it had no effect


def test_f4_hold_over_staged_edit_refused(dbg, base):
    run = dbg.run(P, name="ov2")
    run.until("t4.L1.h")
    run.set("t4.L1.n", 0.0)
    with pytest.raises(ValueError, match="would override the staged edit"):
        run.hold("t@last.L1.mlp_norm_scale", source=base)
    assert run.holds == []


def test_f5_arm_rows_exclude_path_edits(dbg, other):
    run = dbg.run(P, name="pe", prefill=3)
    run.until("t3.L0.h")
    run.set("t3.L0.m", 0.5, "add")
    br = dbg.branches(run, Margin(4, 3, 7), at="t4.L0.h")
    br.add("M", [edit("t4.L0.m", other.ref("t4.L0.m"))])
    row = br.run()[0]
    assert [(e["row"], e["layer"], e["port"]) for e in row["edits"]] == [(4, 0, "m")]
    at = br.seal()["body"]["at"]
    assert [(e["row"], e["port"]) for e in at["path_edits"]] == [(3, "m")]


def test_f12_branch_stage_later_port_and_out_of_order(dbg, other):
    run = dbg.run(P, name="b3")
    br = dbg.branches(run, Margin(4, 3, 7), at="t4.L1.r")
    br.add("M", [edit("t4.L1.m", other.ref("t4.L1.m"))])
    assert br.run()[0]["edits"][0]["port"] == "m"             # later in the branch layer: fine
    with pytest.raises(Refusal, match="before the branch point"):
        br.add("early", [edit("t4.L1.h", other.ref("t4.L1.h"))])
        br.run(["early"])
    run2 = dbg.run(P, name="b4")
    br2 = dbg.branches(run2, Margin(4, 3, 7), at="t4.L0.h")
    br2.add("two", [edit("t4.L1.m", other.ref("t4.L1.m")), edit("t4.L0.m", other.ref("t4.L0.m"))])
    row = br2.run()[0]
    assert [e["layer"] for e in row["edits"]] == [0, 1] and run2.where == "t4.L0.h"


def test_f12_hold_names_monotonic_after_release(dbg, base):
    run = dbg.run(P, name="n")
    run.hold("t@last.L1.p", source=base)
    run.hold("t@last.L0.p", source=base)
    run.release("hold1")
    run.hold("t@last.L1.p", source=base)
    assert [h["name"] for h in run.holds] == ["hold2", "hold3"]


# ------------------------------------------------------------------ F6 circuit map

def test_f6_group_writer_sees_data_edits(dbg):
    ad = dbg.adapter
    w = ad.model.model.layers[1].self_attn.o_proj.weight
    before = circuit_map.group_writer(ad, 1).clone()
    w.data.mul_(2.0)
    try:
        assert torch.equal(circuit_map.group_writer(ad, 1), before * 2)
    finally:
        w.data.div_(2.0)
    assert torch.equal(circuit_map.group_writer(ad, 1), before)


# ------------------------------------------------------------------ F7 custody

_ID = {"family": "qwen2", "parameters_sha256": "0" * 64}


def _seal():
    return seal_record(session_id="s", kind="supplied-forecast", model_identity=_ID,
                       execution={"device": "cpu"}, authority="supplied",
                       body={"schema": "x", "target": {"row": 1}, "edit": {},
                             "supplied": {"value": 1.0, "authority": "supplied"},
                             "library": {"name": "t"}, "address_map": {},
                             "sealed_at": {"where": "t1.logits"}})


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("MDB_SCRIPT", "MDB_PREREG", "MDB_RECORDS", "MDB_KERNEL_PATH"):
        monkeypatch.delenv(k, raising=False)
    custody.clear()
    custody.reread()
    yield monkeypatch
    custody.clear()


def test_f7_missing_script_raises_every_time(clean_env):
    clean_env.setenv("MDB_SCRIPT", "/nonexistent/driver.py")
    for _ in range(3):
        with pytest.raises(FileNotFoundError):
            _seal()


def test_f7_kernel_path_alone_declares(clean_env):
    clean_env.setenv("MDB_KERNEL_PATH", "batched")
    held = _seal()["body"]["custody"]
    assert held["kernel_path"] == "batched"


# ------------------------------------------------------------------ F10 / F11 transcoders

def _tc_state(d=8, d_sae=64, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {"W_enc": (torch.randn(d_sae, d, generator=g)).to(torch.bfloat16),
            "b_enc": (torch.randn(d_sae, generator=g)).to(torch.bfloat16),
            "W_dec": torch.randn(d_sae, d, generator=g).to(torch.bfloat16),
            "b_dec": torch.randn(d, generator=g).to(torch.bfloat16)}


def _open(tmp_path, state, config=None, fname="config.json"):
    safetensors.save_file(state, str(tmp_path / "layer_1.safetensors"))
    if config is not None:
        (tmp_path / fname).write_text(config)
    return sae.open_transcoders("t", str(tmp_path))


def test_f10_active_set_chosen_in_float32(tmp_path):
    state = _tc_state()
    tc = _open(tmp_path, state).get(1)
    g = torch.Generator().manual_seed(11)
    for _ in range(8):
        x = torch.randn(8, generator=g, dtype=torch.float32) * 30
        pre = torch.addmv(state["b_enc"].float(), state["W_enc"].float(), x)
        ids = (pre > 0).nonzero().reshape(-1)
        want = pre.index_select(0, ids)
        order = want.argsort(descending=True)
        got_ids, got_vals = tc.select(x)
        assert got_ids == ids[order].tolist() and torch.equal(got_vals, want[order])


def test_f11_skip_and_jumprelu_refused_and_yaml_read(tmp_path):
    state = _tc_state()
    state["W_skip"] = torch.zeros(8, 8, dtype=torch.bfloat16)
    with pytest.raises(sae.SaeError, match="W_skip"):
        _open(tmp_path, state).get(1)
    jump = tmp_path / "j"
    jump.mkdir()
    state = _tc_state()
    state["activation_function.threshold"] = torch.zeros(64, dtype=torch.bfloat16)
    with pytest.raises(sae.SaeError, match="threshold"):
        _open(jump, state).get(1)
    ok = tmp_path / "y"
    ok.mkdir()
    tset = _open(ok, _tc_state(), "model_name: Qwen/Qwen3-1.7B\nd_sae: 64\n", "config.yaml")
    assert tset.describe()["model"] == "Qwen/Qwen3-1.7B"
