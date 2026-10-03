from pathlib import Path

import pytest

from mdb.assay_custody import remove_source_store


def test_source_deletion_refuses_non_assay_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "selected.json").write_text("{}")
    with pytest.raises(ValueError, match="canonical assay"):
        remove_source_store(source, tmp_path)
    assert source.exists()


@pytest.mark.parametrize("source", [Path("/mnt/big/mdb/qualification/job-012345abcdef"),
                                    Path("/mnt/big/saturn-open-bench-families/old")])
def test_source_deletion_refuses_root_and_other_store(source):
    with pytest.raises(ValueError, match="outside"):
        remove_source_store(source, Path("/mnt/big/mdb/qualification/job-012345abcdef"))


@pytest.mark.parametrize("base", ["/mnt/big/mdb/qualification", "/mnt/hdd8tb/mdb/qualification"])
def test_canonical_volume_keeps_path_boundary(base):
    root = Path(base) / "job-012345abcdef"
    with pytest.raises(ValueError, match="outside"):
        remove_source_store(root / ".." / "other-job" / "source", root)


def test_ext4_volume_rejects_non_job_identity():
    with pytest.raises(ValueError, match="canonical assay"):
        remove_source_store("/mnt/hdd8tb/mdb/qualification/old/session",
                            "/mnt/hdd8tb/mdb/qualification/old")
