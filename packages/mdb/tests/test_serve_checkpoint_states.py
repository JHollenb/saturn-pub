"""``mdb serve`` reads the cuts of ``checkpoint-state`` records (``Debugger.checkpoint`` stores
no native cut beside them): a session checkpointed that way has addressable cuts, its slot
tensors read back bit for bit from the parts, and a project that holds both old native cuts and
new checkpoint states shows both."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import mdb  # noqa: E402
from mdb.core.project_read import ProjectReader  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.serve.app import create_app  # noqa: E402
from mdb.serve.recorded import SessionModel  # noqa: E402

PORT = 8767
BASE = f"http://127.0.0.1:{PORT}"
P = [1, 5, 9, 3, 7, 2, 11]


def _np(t):
    t = t.detach().to("cpu")
    return (t.to(torch.float32) if t.dtype == torch.bfloat16 else t).numpy()


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_a_checkpoint_state_session_has_addressable_cuts(tmp_path, dtype):
    root = tmp_path / "p"
    dbg = mdb.attach(tiny_model("qwen2", dtype), project=root)
    run = dbg.run(P, prefill=2)
    run.until("t3.L0.h")
    run.set("t3.L0.m", 0.25, "add")
    run.until("t4.L1.p")
    info = dbg.checkpoint(run, "pre")
    reader = ProjectReader(root)
    assert reader.cut_ids(dbg.session_id) == []                  # no native cut stored
    model = SessionModel(reader, dbg.session_id)
    cut = run._trail[-1].fingerprint
    assert info["cut"] in model.cuts or cut in model.cuts
    assert model.selected == cut and model.main[-1] == cut
    assert model.tokens == P[:5]                     # the context before the cursor's row
    native = run._cut_of(len(run._trail) - 1)
    for layer in range(dbg.adapter.layers):
        array, meta = model.cut_slot(cut, "keys", layer)
        assert (array == _np(native.slot("keys")[layer])).all()
        assert "checkpoint" in " ".join(meta["via"])
    assert model.slot_info(cut, "keys", 0)["shape"] == list(native.slot("keys")[0].shape)
    c = TestClient(create_app(root, port=PORT), base_url=BASE)
    sessions = c.get("/api/v1/sessions").json()["data"]
    assert [s for s in sessions if s.get("sid", s.get("session_id")) == dbg.session_id
            or dbg.session_id in str(s)]
    frame = c.get(f"/api/v1/s/{dbg.session_id}/frame")
    assert frame.status_code == 200, frame.text
    timeline = c.get(f"/api/v1/s/{dbg.session_id}/timeline")
    assert timeline.status_code == 200, timeline.text


def test_old_native_cuts_and_new_checkpoint_states_both_display(tmp_path):
    root = tmp_path / "p"
    dbg = mdb.attach(tiny_model("qwen2", "fp32"), project=root)
    run = dbg.run(P, prefill=2)
    run.until("t3.L0.h")
    old = run.session.capture(retain=False)                      # as a project written before
    dbg.project.save_cut(dbg.session_id, old)
    dbg.project.select(dbg.session_id, old.fingerprint)
    run.until("t4.L1.p")
    dbg.checkpoint(run, "new")
    new = run._trail[-1].fingerprint
    model = SessionModel(ProjectReader(root), dbg.session_id)
    assert old.fingerprint in model.cut_ids and new in model.cut_ids
    assert model.selected == old.fingerprint                     # the session's own selection
    states = [c for c in model.cut_ids if c in model.states and not model._native(c)]
    assert new in states and old.fingerprint not in states
    # the same kind of descriptor either way, and both readable
    for c in (old.fingerprint, new):
        desc = model.cuts[c]
        assert isinstance(desc["execution_point"], dict) and isinstance(desc["surface"], dict)
        array, _ = model.cut_slot(c, "keys", 0)
        assert array.shape[2] >= 3
    # a native cut and a checkpoint state of one fingerprint: the descriptors agree
    dbg.project.save_cut(dbg.session_id, run._cut_of(len(run._trail) - 1))
    both = SessionModel(ProjectReader(root), dbg.session_id)
    assert both._native(new)
    from mdb.serve.states import StateCuts

    state = StateCuts(ProjectReader(root), dbg.session_id).cuts[new]["descriptor"]
    for key in ("parent", "boundary", "execution_point", "surface"):
        assert state[key] == both.cuts[new][key], key
    from mdb.serve.views.common import cell_address

    c = TestClient(create_app(root, port=PORT), base_url=BASE)
    assert c.get(f"/api/v1/s/{dbg.session_id}/timeline").status_code == 200
    for cut in (old.fingerprint, new):
        addr = cell_address(model, *model.cell_of_cut[cut])
        frame = c.get(f"/api/v1/s/{dbg.session_id}/frame", params={"addr": addr})
        assert frame.status_code == 200, frame.text
        assert frame.json()["data"]["focus"]["cut"] == cut
