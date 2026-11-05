"""Consumer orientation and independent BF16 writer instrument; no model run."""
from pathlib import Path
import runpy

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_context_transfer.py"))


def test_frozen_rgb_reader_has_expected_orientation_without_uint8_wrap():
    image = np.full((512, 512, 3), 255, dtype=np.uint8)
    image[:, :256] = [255, 0, 0]
    image[:, 256:] = [0, 0, 255]
    expected = 2 * 255 / 256
    assert A["rgb_score"](image) == expected
    assert A["rgb_score"](image[:, ::-1]) == -expected
    assert A["rgb_score"](np.zeros_like(image)) == 0


@pytest.mark.parametrize("value", [np.zeros((256, 256, 3), np.uint8), np.zeros((512, 512, 3), float)])
def test_reader_rejects_resized_or_reinterpreted_consumer(value):
    with pytest.raises(ValueError, match="original512px"):
        A["rgb_score"](value)


def test_writer_reconstruction_matches_eager_torch_bf16_boundaries():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    tensors = [torch.tensor(rng.normal(size=(1, 3, 8)), dtype=torch.bfloat16) for _ in range(5)]
    image, a, m, ga, gm = tensors
    expected = ((image + ga * a) + gm * m).float().numpy()
    ports = dict(attention_used=a.float().numpy(), mlp_used=m.float().numpy(),
                 gate_attention=ga.float().numpy(), gate_mlp=gm.float().numpy())
    actual = A["writer_equation"](image.float().numpy(), ports)
    assert np.array_equal(actual, expected)
    # One final cast erases native operation boundaries and is not equivalent.
    fused = A["bf16"](image.float().numpy() + ports["gate_attention"] * ports["attention_used"]
                       + ports["gate_mlp"] * ports["mlp_used"])
    assert not np.array_equal(fused, actual)


def test_difference_keeps_a_small_residual_despite_same_coarse_semantics():
    base = np.zeros((3,), np.float32)
    altered = base.copy()
    altered[0] = 1e-7
    result = A["difference"](altered, base)
    assert not result["exact"] and result["changed_scalars"] == 1 and result["rms"] > 0
