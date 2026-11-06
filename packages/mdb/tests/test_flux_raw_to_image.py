"""Small installed-package mechanics for the raw-prompt composition worker."""
from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/flux_raw_to_image.py"
W = runpy.run_path(str(PATH))


def test_contract_distinguishes_complete_shared_image_forecast_from_kernel_claim():
    contract = W["contract"]()
    assert contract["decoder"]["whole_image_shared_primitive_forecast"] is True
    assert contract["decoder"]["independent_decoder_kernel_arithmetic"] is False
    assert contract["resources"] == {
        "pipeline_assemblies": 1,
        "component_hydrations": 3,
        "cuda_placements": 6,
        "cpu_offloads": 6,
        "max_gpu_components": 1,
        "persistent_weight_banks": 0,
    }


def test_native_source_contract_is_runtime_bound_without_model_calls():
    config = {"native_sources": {name: {"sha256": "runtime-bound"} for name in W["NATIVE_SOURCE_MODULES"]}}
    contract = W["_native_source_contract"](config)
    assert set(contract["sources"]) == set(W["NATIVE_SOURCE_MODULES"])
    assert all(row["binding"] == "runtime-observed" for row in contract["sources"].values())
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in contract["sources"].values())


def test_decoder_input_forces_cpu_frontend_boundary():
    seen = {}

    class Pipeline:
        @staticmethod
        def _unpack_latents_with_ids(latent, image_ids, height, width):
            seen["latent_device"] = latent.device.type
            seen["ids_device"] = image_ids.device.type
            assert (height, width) == (32, 32)
            return torch.ones((1, 32, 32, 32), dtype=torch.bfloat16)

        @staticmethod
        def _unpatchify_latents(value):
            seen["unpatch_device"] = value.device.type
            return torch.ones((1, 32, 64, 64), dtype=torch.bfloat16)

    vae = SimpleNamespace(
        config=SimpleNamespace(block_out_channels=(128, 256, 512, 512), batch_norm_eps=1e-5),
        bn=SimpleNamespace(
            running_mean=torch.zeros(32, dtype=torch.bfloat16),
            running_var=torch.ones(32, dtype=torch.bfloat16),
        ),
    )
    state = {
        "latent": torch.zeros((1, 1024, 128), dtype=torch.bfloat16),
        "img_ids": torch.zeros((1, 1024, 4), dtype=torch.int64),
    }
    result = W["_decoder_input"](vae, state, {}, {"Flux2KleinPipeline": Pipeline})
    assert result.device.type == "cpu" and result.dtype == torch.bfloat16
    assert tuple(result.shape) == (1, 32, 64, 64)
    assert seen == {"latent_device": "cpu", "ids_device": "cpu", "unpatch_device": "cpu"}


def test_learned_forward_guard_releases_only_after_seal():
    module = torch.nn.Linear(3, 3, bias=False).eval()
    released = {"native": False}
    counters = {}
    with W["_learned_forward_guard"]([module], released, counters):
        with pytest.raises(ValueError, match="before global forecast seal"):
            module(torch.ones(1, 3))
        assert counters["guard_refusals"] == 1
        released["native"] = True
        output = module(torch.ones(1, 3))
    assert tuple(output.shape) == (1, 3)
    assert counters["native_module_entries"] >= 1
