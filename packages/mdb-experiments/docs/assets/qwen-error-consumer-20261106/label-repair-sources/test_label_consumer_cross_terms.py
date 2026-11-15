"""Metadata-only source-order labels, including numerically ambiguous zeros."""
import copy
import runpy
from pathlib import Path

import pytest

A = runpy.run_path(str(Path(__file__).with_name("label_consumer_cross_terms.py")))


def fixture():
    def row(bounds):
        result = dict(bounds=bounds)
        for kind, order in A["TERM_ORDER"].items():
            # Equal summaries deliberately cannot identify the term names.
            terms = {name: dict(elements=256, rms=0., signed_mean=0.) for name in order}
            pairs = [dict(left=copy.deepcopy(terms[left]), right=copy.deepcopy(terms[right]), dot=0., signed_cross_term=0., cosine=None)
                for index, left in enumerate(order) for right in order[index+1:]]
            result[kind] = dict(terms=terms, pairwise_cross_terms=pairs, energy_identity_residual=0.)
        return result
    return {lane: dict(whole=row([0,512]), windows=[row([0,256]), row([256,512])]) for lane in ("functional", "native")}


def test_source_order_resolves_equal_zero_summaries_without_changing_metrics():
    original = fixture(); frozen = A["canonical"](original)
    result, affected = A["labeled_accounts"](original)
    assert A["canonical"](original) == frozen and len(affected) == 36
    assert result["native"]["whole"]["output"]["pairwise_cross_terms"][0]["left_term"] == "R"
    assert result["native"]["whole"]["output"]["pairwise_cross_terms"][0]["right_term"] == "M"
    assert result["functional"]["windows"][1]["residual"]["pairwise_cross_terms"][2]["left_term"] == "projection"
    assert result["functional"]["windows"][1]["residual"]["pairwise_cross_terms"][2]["right_term"] == "finite_addition_defect"


def test_wrong_summary_pair_count_signedzero_and_coverage_refused():
    accounts = fixture(); accounts["native"]["windows"][0]["output"]["pairwise_cross_terms"][1]["right"]["rms"] = 1.
    with pytest.raises(ValueError, match="pinned source order"):
        A["labeled_accounts"](accounts)
    accounts = fixture(); accounts["native"]["whole"]["output"]["pairwise_cross_terms"].pop()
    with pytest.raises(ValueError, match="pair census"):
        A["labeled_accounts"](accounts)
    accounts = fixture(); accounts["native"]["whole"]["output"]["pairwise_cross_terms"][0]["left"]["signed_mean"] = -0.
    with pytest.raises(ValueError, match="pinned source order"):
        A["labeled_accounts"](accounts)
    accounts = fixture(); accounts["native"]["windows"].pop()
    with pytest.raises(ValueError, match="coverage differs"):
        A["labeled_accounts"](accounts)


def test_repair_binds_metadata_source_and_refuses_overwrites(tmp_path):
    pack = tmp_path/"pack"; pack.mkdir()
    report = dict(schema="mdb-qwen-error-consumer-audit-v1", audit_sha256=A["READER_SHA256"], residual_accounts=fixture(),
        native_manifest_sha256="1"*64, job_id="job-synthetic", request_sha256="2"*64)
    A["save"](pack/"audit.json",report); audit_sha=A["sha"](pack/"audit.json")
    A["save"](pack/"files.json", {"audit.json":audit_sha}); manifest=A["sha"](pack/"files.json")
    source = Path(__file__).with_name("audit_qwen_error_consumer.py")
    receipt=A["repair"](pack,tmp_path/"labels",expected_manifest=manifest,expected_audit=audit_sha,reader_source=source)
    assert receipt["affected_records"]==36 and receipt["original_files_unchanged"] and receipt["arrays_read"]==0
    assert A["sha"](pack/"audit.json")==audit_sha
    with pytest.raises(FileExistsError):
        A["repair"](pack,tmp_path/"labels",expected_manifest=manifest,expected_audit=audit_sha,reader_source=source)
    with pytest.raises(ValueError, match="checksum differs"):
        A["repair"](pack,tmp_path/"wrong",expected_manifest=manifest,expected_audit="0"*64,reader_source=source)
