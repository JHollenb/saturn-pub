"""Saved-only Qwen publication fixtures; no checkpoint/model execution."""
from __future__ import annotations

import copy
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).parents[1]
PATH = HERE/"qualification/publish_flux_functional_qwen_prefix.py"
P = runpy.run_path(str(PATH))
T = runpy.run_path(str(HERE/"tests/test_audit_flux_functional_qwen_prefix.py"))


@pytest.fixture(scope="module")
def compact(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen-publication")
    raw,summary,digest = T["saved_fixture"](root/"fixture",numerical_miss=True)
    auditor = HERE/"qualification/audit_flux_functional_qwen_prefix.py"
    config = dict(native_root=str(raw),native_manifest_sha256=digest,native_request_sha256=summary["request_sha256"],source_job=summary["job_id"],output_scratch=str(root/"outputs"))
    request = dict(config=config,source_hashes={"audit_flux_functional_qwen_prefix.py":P["AUDITOR_SHA256"]})
    request["request_sha256"] = __import__("hashlib").sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    request_path,config_path = root/"request.json",root/"config.json"
    request_path.write_text(json.dumps(request))
    config_path.write_text(json.dumps(config))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES="",SATURN_RESIDENT_WORKER="1",SATURN_DEBUG_REQUEST=str(request_path),SATURN_RUN_CONFIG=str(config_path),MRUN_JOB_ID="job-qwen-publication-fixture")
    result = subprocess.run([sys.executable,"-I","-B",str(auditor)],cwd=HERE/"qualification",env=env,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    ready = json.loads(next(line.split(" ",1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    qualification = root/"qualification.json"
    qualification.write_text(json.dumps(dict(status="passed",synthetic_fixture_only=True,source_sha256={"qualification/audit_flux_functional_qwen_prefix.py":P["AUDITOR_SHA256"]})))
    return Path(ready["output_root"]),ready["files_manifest_sha256"],qualification


def test_publication_preserves_misses_signs_zero_rows_sources_and_catalog(compact,tmp_path,monkeypatch):
    root,digest,qualification = compact
    def forbidden(*args,**kwargs):
        raise AssertionError("publisher loaded raw arrays")
    monkeypatch.setattr(np,"load",forbidden)
    output = tmp_path/"publication"
    receipt = P["publish"](root,output,expected_manifest=digest,qualification=qualification)
    report = P["read"](output/"publication.json")
    assert (output/"audit-data.json").read_bytes() == (root/"audit.json").read_bytes()
    assert report["accounts"]["first_value_discrepancy"]["prompt-01"] == "hidden.2"
    assert report["accounts"]["cases"]["prompt-01"]["fields"]["hidden.2"]["padded_query_rows"]["comparison"]["changed_values"] == 1
    assert report["accounts"]["signed_prompt_response"]["hidden.2"]["signed_mean"] < 0
    assert report["field_counts"]["overlap"] is True and report["field_counts"]["primary_full_fields"] == 12
    assert len(list((output/"compact-evidence/catalog/equations").glob("*.json"))) == 2
    assert report["catalog"]["dependent_prompts"] == 2
    assert receipt["new_model_calls"] == receipt["new_numerical_recurrence"] == 0
    assert receipt["raw_arrays_loaded"] is receipt["weights_loaded"] is False
    for name in ("hidden-errors.png","conditioning-errors.png","prompt-responses.png"):
        with Image.open(output/name) as image:
            assert image.width >= 1200 and image.height >= 900
    assert all(row["within_figure"] for rows in P["read"](output/"figure-layout.json").values() for row in rows)


def test_countertrends_undefined_and_all_zero_plot_values_remain_visible(compact,tmp_path,monkeypatch):
    report = copy.deepcopy(P["read"](compact[0]/"audit.json"))
    response = report["accounts"]["signed_prompt_response"]
    response["hidden.1"].update(cosine=-.75,relative_rms=3.,predicted_signed_mean=-2.,native_signed_mean=1.)
    response["hidden.0"].update(cosine=None,relative_rms=None)
    original = P["figures"].__globals__["finish"]
    seen = []
    def finish(fig,path):
        fig.canvas.draw()
        for ax in fig.axes:
            low,high = ax.get_ylim()
            for line in ax.lines:
                values = [float(value) for value in line.get_ydata() if value == value]
                assert all(low-1e-8 <= value <= high+1e-8 for value in values)
                seen.extend(values)
        return original(fig,path)
    monkeypatch.setitem(P["figures"].__globals__,"finish",finish)
    output = tmp_path/"figures"
    output.mkdir()
    P["figures"](report,output)
    assert -.75 in seen and -2. in seen and 3. in seen and 0. in seen


def test_compact_and_catalog_integrity_refusals(compact,tmp_path):
    root,digest,qualification = compact
    with (root/"audit.json").open("ab") as stream:
        stream.write(b" ")
    try:
        with pytest.raises(ValueError,match="compact.*differs"):
            P["publish"](root,tmp_path/"bad",expected_manifest=digest,qualification=qualification)
    finally:
        path = root/"audit.json"
        path.write_bytes(path.read_bytes()[:-1])
    assert not (tmp_path/"bad").exists()
    stale = tmp_path/"stale"
    shutil.copytree(root,stale)
    record = next((stale/"catalog/equations").glob("*.json"))
    value = P["read"](record)
    value["evidence"]["audit_sha256"] = "0"*64
    record.write_text(json.dumps(value))
    manifest = {path.relative_to(stale).as_posix():P["sha"](path) for path in stale.rglob("*") if path.is_file() and path.name!="files.json"}
    (stale/"files.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="catalog/audit"):
        P["publish"](stale,tmp_path/"bad-catalog",expected_manifest=P["sha"](stale/"files.json"),qualification=qualification)


def test_actual_publication_cli(compact,tmp_path):
    root,digest,qualification = compact
    output = tmp_path/"cli"
    result = subprocess.run([sys.executable,"-I","-B",str(PATH),"--pack",str(root),"--out",str(output),"--expected-manifest",digest,"--qualification",str(qualification)],
        cwd=tmp_path,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["catalog_records"] == 2 and receipt["raw_arrays_loaded"] is False
    assert receipt["outputs"]["publication.json"] == P["sha"](output/"publication.json")
