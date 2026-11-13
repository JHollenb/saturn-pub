import runpy
import subprocess
import sys
from pathlib import Path


WORKER = Path(__file__).resolve().parents[1] / "qualification/flux_binding_minimal.py"


def test_stream_and_primed_subset_panels_have_exact_bounded_work():
    module = runpy.run_path(str(WORKER))
    arms = module["specs"]()
    assert [a["id"] for a in arms] == ["sham", "I", "IC", "TC", "ITC"]
    assert module["expected_counts"](arms) == dict(loads=1,encodes=0,noise_draws=0,
        native_transitions=395,acts=10,receipts=405,images=5)
    for prime,slots in [("I",["image"]),("T",["text"]),("IT",["image","text"])]:
        arms = module["specs"]("subsets",prime)
        assert all(a["slots"] == slots for a in arms)
        assert [a["C"] for a in arms] == ["base","all","valid","padding","color"]
        assert module["expected_counts"](arms)["acts"] == 5*len(slots)+4


def test_resident_dispatch_reaches_worker_before_argument_parser():
    # Same entry-point regression as the previous failed binding launch. Running
    # without a request must fail at worker setup, rather than CLI positional args.
    import os
    env=dict(os.environ,SATURN_RESIDENT_WORKER="1")
    env.pop("SATURN_RUN_CONFIG",None);env.pop("SATURN_DEBUG_REQUEST",None)
    result=subprocess.run([sys.executable,"-I",str(WORKER)],env=env,capture_output=True,text=True)
    assert result.returncode != 0 and "SATURN_RUN_CONFIG" in result.stderr
    assert "arguments are required" not in result.stderr


def test_remote_lane_reuses_recorded_paths_and_refuses_explicit_conflict(monkeypatch):
    import os
    from mdb_lanes.execution_environment import intent
    module=runpy.run_path(str(WORKER))
    for name in ["MDB_WORKER_UV_CACHE_DIR","MDB_WORKER_TORCH_HOME","MDB_WORKER_MODEL_ROOTS","MDB_WORKER_PYTHON","MDB_WORKER_UV_LINK_MODE"]:
        monkeypatch.delenv(name,raising=False)
    remote=dict(intent(),uv_cache_dir="/mnt/hdd8tb/mdb/uv-cache",torch_home="/mnt/hdd8tb/mdb/torch",extra_model_roots="/mnt/big/llm-models")
    module["bind_environment"](remote)
    assert intent()==remote and "/Users/" not in intent()["uv_cache_dir"]
    monkeypatch.setenv("MDB_WORKER_UV_CACHE_DIR","/explicit/other")
    import pytest
    with pytest.raises(ValueError,match="explicit worker environment"):
        module["bind_environment"](remote)


def test_writer_campaign_counts_candidates_separately_from_native_transitions():
    module=runpy.run_path(str(WORKER.with_name("flux_binding_writers.py")))
    arms=module["specs"]()
    assert [a["id"]for a in arms]==["IC.sham","A.fixed","M.fixed","AM.predicted","A.recomputed.native"]
    assert module["expected_counts"](arms)==dict(loads=1,encodes=0,noise_draws=0,native_transitions=395,
        acts=14,receipts=409,images=5,candidate_blocks=4)


def test_search_schedule_restores_base_at_each_read_and_is_bounded():
    module=runpy.run_path(str(WORKER))
    arms=[dict(id="row4.step2",slots=["image"],C="scheduled",rows_by_step={"2":[4],"3":[]})]
    module["validate_search"](arms)
    assert module["expected_counts"](arms)["acts"]==3
    import pytest
    with pytest.raises(ValueError,match="rows"):
        module["validate_search"]([dict(arms[0],rows_by_step={"2":[7],"3":[]})])
    with pytest.raises(ValueError,match="measured lane bounds"):
        module["stage"](None,None,"search","I",None,8000)
    assert not module["has_historical_control"](arms)
    assert module["has_historical_control"]([dict(arms[0],id="control",rows_by_step={"2":[],"3":[]})])


def test_carrier_restoration_requires_matched_source_and_counts_each_act():
    import pytest
    m=runpy.run_path(str(WORKER))
    base=dict(id="control",slots=["image"],C="scheduled",rows_by_step={"2":[],"3":[]},restore_slots=[],restore_index=0)
    restored=dict(base,id="joint",rows_by_step={"2":[4,11],"3":[]},restore_slots=["image","text"])
    m["validate_search"]([base,restored])
    assert m["expected_counts"]([base,restored])==dict(loads=1,encodes=0,noise_draws=0,native_transitions=158,acts=8,receipts=166,images=2)
    with pytest.raises(ValueError,match="matched first control"):
        m["validate_search"]([restored])
    with pytest.raises(ValueError,match="boundary"):
        m["validate_search"]([base,dict(restored,restore_index=1)])
    with pytest.raises(ValueError,match="slots"):
        m["validate_search"]([base,dict(restored,restore_slots=["conditioning"])])
    m["validate_search"]([base,dict(restored,text_keep_rows=[4,11])])
    for ix in([4,4],[512],[1.0]):
        with pytest.raises(ValueError,match="retained text rows"):
            m["validate_search"]([base,dict(restored,text_keep_rows=ix)])
