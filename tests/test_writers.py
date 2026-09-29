import numpy as np
import pytest
import torch

from saturn_pub import StateCut
from saturn_pub.writers import fit_low_rank_writer, save_low_rank_writer


def cut(value, model):
    return StateCut(model, {}, "paired", None, {"text": value})


def test_fit_serving_artifact_and_provenance(tmp_path):
    torch.manual_seed(3)
    x = torch.randn(1, 12, 4)
    target = x + x * 0.25 + 0.7
    arrays, metadata = fit_low_rank_writer(
        cut(x, "recipient"), cut(target, "donor"), address="text", rank=4
    )
    assert metadata["relative_fit_mse"] < 1e-6
    result = save_low_rank_writer(tmp_path / "writer.npz", arrays, metadata)
    with np.load(tmp_path / "writer.npz", allow_pickle=False) as package:
        delta = ((x.numpy() - package["mean"]) @ package["basis"]) @ package["weights"] + package[
            "bias"
        ]
    np.testing.assert_allclose(x.numpy() + delta, target.numpy(), atol=0.002)
    assert result["recipient_model"] == "recipient"
    assert result["donor_model"] == "donor"
    assert result["package_bytes"] > 0


def test_invalid_pair_rejected():
    with pytest.raises(ValueError, match="matching geometry"):
        fit_low_rank_writer(
            cut(torch.zeros(1, 2, 4), "a"), cut(torch.zeros(1, 3, 4), "b"), address="text"
        )
