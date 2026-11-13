import json
import pytest
from mdb_lanes.edit_search import EditSearch


def search(tmp_path):
    return EditSearch.create(tmp_path / "search", binding=dict(parent="cut", model="weights", execution="BF16-CUDA-B1", consumer="native RGB"),
                             atoms=["a", "b", "c"], observable="both colors swapped; manual shape review")


def evidence(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(dict(mechanics="verified", image="native")))
    return [path]


def test_nonmonotonic_search_retains_smaller_cases_of_failed_superset(tmp_path):
    s = search(tmp_path); refs = evidence(tmp_path)
    full = s.freeze(["a", "b", "c"], {"expect": "edit"})
    s.observe(full["candidate"], evidence=refs, verdict="sufficient")
    pair = s.freeze(["a", "b"], {"expect": "edit"})
    s.observe(pair["candidate"], evidence=refs, verdict="insufficient")
    assert ["a"] in [v["atoms"] for v in s.propose()]
    one = s.freeze(["a"], {"expect": "unknown"})
    s.observe(one["candidate"], evidence=refs, verdict="sufficient")
    assert not s.status()["minimum_cardinality_established"]
    empty = s.freeze([], {"expect": "base"})
    s.observe(empty["candidate"], evidence=refs, verdict="insufficient")
    assert s.status()["minimum_cardinality_established"]
    assert s.summary()["observed"]==4 and s.summary()["smallest_observed"][0]["atoms"]==["a"]


def test_resume_freeze_identity_and_immutable_outcomes(tmp_path):
    s = search(tmp_path); refs = evidence(tmp_path)
    frozen = s.freeze(["b", "a"], {"direction": 1})
    assert frozen["atoms"] == ["a", "b"]
    with pytest.raises(FileExistsError): s.freeze(["a", "b"], {})
    resumed = EditSearch(s.root)
    resumed.observe(frozen["candidate"], evidence=refs, verdict="uncertain")
    with pytest.raises(FileExistsError): resumed.observe(frozen["candidate"], evidence=refs, verdict="sufficient")
    assert resumed.status()["records"][0]["verdict"] == "uncertain"
    refs[0].write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"): resumed.status()


def test_freeze_tampering_and_binding_refused(tmp_path):
    s = search(tmp_path); f = s.freeze(["a"], {})
    p = s.root / "candidates" / f["candidate"] / "freeze.json"
    v = json.loads(p.read_text()); v["prediction"] = {"cheat": True}; p.write_text(json.dumps(v))
    with pytest.raises(ValueError, match="freeze differs"): s.status()
    with pytest.raises(ValueError, match="complete"): EditSearch.create(tmp_path / "bad", binding={}, atoms=["a"], observable="x")
    with pytest.raises(ValueError, match="unknown"): s.key(["other"])


def test_cli_dispatch_stays_framework_free(tmp_path,capsys):
    import sys
    from mdb.cli import main
    s=search(tmp_path)
    assert main(["edit-search","propose",str(s.root),"--limit","1"])==0
    assert json.loads(capsys.readouterr().out)[0]["atoms"]==[]
    assert "torch" not in sys.modules


def test_invalid_numeric_records_do_not_poison_resumable_search(tmp_path):
    s=search(tmp_path)
    with pytest.raises(ValueError):s.freeze(["a"],{"bad":float("nan")})
    assert not s.status()["records"]
    f=s.freeze(["a"],{})
    with pytest.raises(ValueError):s.observe(f["candidate"],evidence=evidence(tmp_path),verdict="uncertain",measurements={"bad":float("inf")})
    assert s.status()["records"][0]["verdict"]=="pending"
