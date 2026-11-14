"""A circuit whose body does not fit one record is paged (issue M38), and carries its ledger id.

On a 40-layer tiny Qwen3 (4 heads) ``top=8 depth=3`` is 371 nodes and 66 KiB of body: the record
bound (64 KiB) refused it before. Now the record keeps the root and the first levels, and each
group that does not fit is a pointer to a sha256-checked child page, held on the circuit with no
project and in the project store with one; the server draws it live (the op route) and recorded
(``/stored/{id}?node=``); the ledger id is in the record body and stays what it was.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mdb.analysis import circuits  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.core.bounded import json_bytes  # noqa: E402
from mdb.core.circuit_pages import PAGE_BYTES  # noqa: E402
from mdb.core.pointers import MAX_RECORD_BYTES  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops.debugger import attach  # noqa: E402
from mdb.serve.app import create_app  # noqa: E402
from mdb.serve.live import LiveSession  # noqa: E402

PORT = 8767
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = [1, 5, 9, 3, 7, 2, 11, 4, 6, 8, 10, 12]
BIG = {"target": "M[4,9]", "top": 8, "depth": 3}


@pytest.fixture(scope="module")
def model():
    return tiny_model("qwen3", "fp32", layers=40)


def _walk(node, pages=None, load=None):
    """Every node of a circuit through its pages: ``(node, in_page)``."""
    yield node
    for child in node.get("children", ()):
        yield from _walk(child, pages, load)
    if "paged" in node and load is not None:
        for child in load(node["paged"]["pointer"])["node"]["children"]:
            yield from _walk(child, pages, load)


@pytest.fixture(scope="module")
def bare(model):
    """The big circuit on a debugger with no project: pages are held on the circuit."""
    run = attach(model).run(PROMPT)
    run.until("t11.logits")
    return circuits.build(run, Margin(11, 4, 9), top=8, depth=3)


def test_a_circuit_over_the_record_bound_builds_and_its_root_page_fits(bare):
    body = verify_record(bare.record)["body"]
    assert body["count"] == 371
    assert len(json_bytes(body)) <= PAGE_BYTES + 4096 < MAX_RECORD_BYTES
    paged = [n for n in _walk(body["root"]) if "paged" in n]
    assert paged, "the 66 KiB body must have paged groups"
    for node in paged:
        assert "children" not in node           # paged, not truncated: nothing listed twice
        assert node["paged"]["state"] == "paged"
        assert node["paged"]["children"]["authority"] == "derived-exact"
        assert not node.get("unexamined")        # paged children WERE examined
    # every node is reachable: the root page plus the pages hold all 371, none counted twice
    every = list(_walk(body["root"], load=lambda p: circuits.load_page(bare, p)))
    assert len(every) == body["count"]
    assert len({n["id"] for n in every}) == body["count"]
    full = sum(len(json_bytes(p)) for p in bare.pages.values()) + len(json_bytes(body))
    assert full > MAX_RECORD_BYTES              # the old one-record body could not hold this
    assert all(len(json_bytes(p)) <= PAGE_BYTES for p in bare.pages.values())


def test_a_child_page_resolves_without_a_project_and_unknown_ids_are_refused(bare):
    body = bare.record["body"]
    target = next(n for n in _walk(body["root"]) if "paged" in n)
    found = circuits.page(bare, target["id"])
    assert found["schema"] == circuits.PAGE_SCHEMA and found["node"]["id"] == target["id"]
    assert found["node"]["children"] and found["node"]["name"] == target["name"]
    last = f"n{body['count']}"                    # a node deep in some page, found by id
    assert circuits.page(bare, last)["node"]["id"] == last
    assert circuits.page(bare, "n1")["node"]["id"] == "n1"
    for wrong in ("n0", f"n{body['count'] + 1}", "x3", "n"):
        with pytest.raises(Refusal, match="no node"):
            circuits.page(bare, wrong)


def test_a_changed_page_is_refused_without_a_project(model):
    run = attach(model).run(PROMPT)
    run.until("t11.logits")
    circuit = circuits.build(run, Margin(11, 4, 9), top=8, depth=3)
    target = next(n for n in _walk(circuit.record["body"]["root"]) if "paged" in n)
    sha = target["paged"]["pointer"]["sha256"]
    circuit.pages[sha]["node"]["name"] = "tampered"
    with pytest.raises(Refusal, match="differs from the pointer's sha256"):
        circuits.page(circuit, target["id"])
    del circuit.pages[sha]
    with pytest.raises(Refusal, match="not held"):
        circuits.page(circuit, target["id"])


def test_a_small_circuit_is_not_paged_and_keeps_one_body(model):
    run = attach(model).run(PROMPT)
    run.until("t11.logits")
    small = circuits.build(run, Margin(11, 4, 9), top=2, depth=3)
    assert small.pages == {}
    assert not [n for n in _walk(small.record["body"]["root"]) if "paged" in n]


def test_the_ledger_id_is_in_the_record_and_each_circuit_has_its_own(model):
    run = attach(model).run(PROMPT)
    run.until("t11.logits")
    first = circuits.build(run, Margin(11, 4, 9), top=1, depth=1)
    second = circuits.build(run, Margin(11, 4, 9), top=1, depth=1)
    ids = [c.record["body"]["ledger_id"] for c in (first, second)]
    assert [(i["value"], i["authority"]) for i in ids] == [(1, "derived-exact"),
                                                          (2, "derived-exact")]
    assert first.record["id"] != second.record["id"]   # same circuit, different ledger number


# --------------------------------------------------------------------------- server, with a project


def _op(client, sid, op, args=None):
    return client.post(f"/api/v1/s/{sid}/op", json={"op": op, "args": args or {},
                                                    "request_id": "r"})


def _result(response):
    assert response.status_code == 200, response.text
    return response.json()["data"]["result"]


@pytest.fixture(scope="module")
def project(model, tmp_path_factory):
    """A project recorded live: the big circuit (id 1), a restart, then a small one (id 2)."""
    root = tmp_path_factory.mktemp("paged") / "proj"
    session = LiveSession(attach(model, None, project=root))
    client = TestClient(create_app(root, port=PORT, live=session), base_url=BASE)
    sid = session.sid
    out = {"root": root, "sid": sid}
    try:
        _op(client, sid, "start", {"prompt": PROMPT})
        _op(client, sid, "until", {"target": "t11.logits"})
        out["built"] = _result(_op(client, sid, "circuit", BIG))
        out["root_show"] = _result(_op(client, sid, "circuit", {"target": "show", "id": 1}))
        body = out["built"]["record"]["body"]
        out["paged"] = [n for n in _walk(body["root"]) if "paged" in n]
        out["page"] = _result(_op(client, sid, "circuit", {
            "target": "show", "id": 1, "node": out["paged"][0]["id"]}))
        out["unknown"] = _op(client, sid, "circuit", {"target": "show", "id": 1, "node": "n999"})
        _op(client, sid, "start", {"prompt": PROMPT})     # carries circuit 1 to the new run
        _op(client, sid, "until", {"target": "t11.logits"})
        out["second"] = _result(_op(client, sid, "circuit",
                                    {"target": "M[4,9]", "top": 1, "depth": 1}))
        out["carried_show"] = _result(_op(client, sid, "circuit", {"target": "show", "id": 1}))
        out["carried_page"] = _result(_op(client, sid, "circuit", {
            "target": "show", "id": 1, "node": out["paged"][0]["id"]}))
        out["ledger"] = client.get(f"/api/v1/s/{sid}/live/ledger").json()["data"]["circuits"]
    finally:
        session.close()
    return out


def test_live_build_show_and_pages_are_drawn_through_the_op_route(project):
    built = project["built"]
    assert built["action"] == "build" and built["id"] == 1
    body = built["record"]["body"]
    assert body["count"] == 371 and len(json_bytes(body)) < MAX_RECORD_BYTES
    assert project["root_show"]["record"]["body"] == body
    assert project["paged"] and all(n["paged"]["pointer"]["state"] == "stored"
                                    for n in project["paged"])
    page = project["page"]
    assert page["action"] == "show" and page["node"] == project["paged"][0]["id"]
    assert page["page"]["node"]["id"] == page["node"] and page["page"]["node"]["children"]
    assert "body" not in page["record"]               # the page answer does not repeat the body
    refused = project["unknown"].json()
    assert "no node 'n999'" in json.dumps(refused)


def test_the_ledger_id_stays_for_a_circuit_carried_to_a_later_run(project):
    assert project["second"]["id"] == 2
    assert project["second"]["record"]["body"]["ledger_id"]["value"] == 2
    carried = project["carried_show"]
    assert carried["from_previous_run"] is True and carried["id"] == 1
    assert carried["record"]["body"]["ledger_id"]["value"] == 1
    assert [row["id"] for row in project["ledger"]] == [1, 2]
    # its pages still resolve from the project store after the run that built it is gone
    assert project["carried_page"]["page"]["node"]["id"] == project["paged"][0]["id"]


def test_the_rendered_walk_names_the_paged_groups_and_the_page_command(project):
    from mdb.repl.render import render

    text = render(project["root_show"])
    assert re.search(r"\[paged: .* children examined, .* nodes below, stored as page [0-9a-f]{12};"
                     r" circuit show 1 node=n\d+\]", text)
    page_lines = render(project["page"])
    assert page_lines.startswith(f"circuit 1 node {project['page']['node']}")


# --------------------------------------------------------------------------- server, recorded


@pytest.fixture(scope="module")
def recorded(project):
    client = TestClient(create_app(project["root"], port=PORT), base_url=BASE)
    sessions = client.get("/api/v1/sessions").json()["data"]
    sid = next(s["id"] for s in sessions if s["records"])
    rows = client.get(f"/api/v1/s/{sid}/records", params={"kind": "circuit"}).json()["data"]
    return client, sid, rows["rows"]


def _stored(recorded, rid, **params):
    client, sid, _ = recorded
    return client.get(f"/api/v1/s/{sid}/stored/{rid}", params=params)


def test_recorded_mode_draws_a_circuit_that_did_not_fit_one_body(project, recorded):
    big = project["built"]["record"]["id"]
    response = _stored(recorded, big)
    assert response.status_code == 200, response.text      # was 409 over_cap before paging
    stored = response.json()["data"]
    assert stored["ledger_id"] == 1 and stored["node"] is None
    assert any("paged" in n for n in _walk(stored["body"]["root"]))
    first = project["paged"][0]["id"]
    page = _stored(recorded, big, node=first)
    assert page.status_code == 200, page.text
    data = page.json()["data"]
    assert data["node"] == first and data["record_schema"] == circuits.PAGE_SCHEMA
    assert data["body"]["node"]["id"] == first and data["body"]["node"]["children"]
    # the page the server reads is the page the live op returned
    assert data["body"]["node"]["name"] == project["page"]["page"]["node"]["name"]
    inline = _stored(recorded, big, node="n2").json()["data"]["body"]
    assert inline["state"] == "inline" and inline["node"]["id"] == "n2"
    assert _stored(recorded, big, node="n99999").status_code == 404


def test_recorded_ledger_ids_are_stable_across_reopen(project, recorded):
    _, _, rows = recorded
    ledger = {}
    for row in rows:
        ledger[row["id"]] = _stored(recorded, row["id"]).json()["data"]["ledger_id"]
    assert sorted(ledger.values()) == [1, 2]
    reopened = TestClient(create_app(project["root"], port=PORT), base_url=BASE)
    sid = recorded[1]
    for rid, expected in ledger.items():
        again = reopened.get(f"/api/v1/s/{sid}/stored/{rid}").json()["data"]
        assert again["ledger_id"] == expected


def test_recorded_mode_refuses_a_changed_page_blob(project, recorded):
    big = project["built"]["record"]["id"]
    first = project["paged"][0]
    ref = first["paged"]["pointer"]["ref"]
    blob = Path(project["root"]) / "store" / "tensors" / f"{ref}.safetensors"
    original = blob.read_bytes()
    try:
        blob.write_bytes(original[:-3] + bytes((b ^ 1) for b in original[-3:]))
        response = _stored(recorded, big, node=first["id"])
        assert response.status_code == 409 and response.json()["refused"] == "page_refused"
    finally:
        blob.write_bytes(original)
    assert _stored(recorded, big, node=first["id"]).status_code == 200
