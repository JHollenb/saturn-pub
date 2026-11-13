"""Explicit large-transaction admission preserves default and integrity bounds."""
import pytest

from mdb_lanes.forecast_transaction import (
    ForecastTransaction,
    ForecastTransactionError,
    verify_transaction,
)


def sealed(tmp_path):
    tx = ForecastTransaction(tmp_path / "tx", model_identity="m", input_identity="i",
        source_identity="s", frontier={"kind": "raw-input"})
    tx.predict("record", lambda context: {"predictions": {"constant": 3}})
    seal = tx.seal()
    tx.close()
    return tx.root, seal["seal_sha256"]


def test_large_sparse_geometry_needs_explicit_admission(tmp_path):
    root, seal = sealed(tmp_path)
    # The geometry scan counts every file. This sparse unsealed ancillary file
    # exercises the real >1GiB guard without allocating or reading GiBs.
    with (root / "ancillary.dat").open("wb") as stream:
        stream.truncate((1 << 30) + 1)
    with pytest.raises(ForecastTransactionError, match="geometry exceeds bound"):
        verify_transaction(root, expected_seal_sha256=seal)
    assert verify_transaction(root, expected_seal_sha256=seal, max_bytes=1536 << 20)["verified"]
    assert verify_transaction(root, expected_seal_sha256=seal, max_bytes=3 << 30)["verified"]
    assert verify_transaction(root, expected_seal_sha256=seal, max_bytes=8 << 30)["verified"]
    with pytest.raises(ForecastTransactionError, match="geometry exceeds bound"):
        verify_transaction(root, expected_seal_sha256=seal, max_bytes=1 << 30)
    with pytest.raises(ForecastTransactionError, match="global seal differs"):
        verify_transaction(root, expected_seal_sha256="0" * 64, max_bytes=3 << 30)
    record = root / "forecasts/record.json"
    record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(ForecastTransactionError, match="sealed artifact changed"):
        verify_transaction(root, expected_seal_sha256=seal, max_bytes=3 << 30)


@pytest.mark.parametrize("bound", [0, -1, True, 1.5, (8 << 30) + 1])
def test_invalid_explicit_bounds_are_rejected(tmp_path, bound):
    root, seal = sealed(tmp_path)
    with pytest.raises(ForecastTransactionError, match="invalid offline byte bound"):
        verify_transaction(root, expected_seal_sha256=seal, max_bytes=bound)
