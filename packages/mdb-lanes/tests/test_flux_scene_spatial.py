"""Mechanics that make native-row spatial intervention claims interpretable."""
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

HELPER = runpy.run_path(str(Path(__file__).resolve().parents[1]/"qualification/flux_scene_spatial.py"))


def example():
    ids = np.asarray([[0,h,w,0] for h in range(8) for w in range(8)])
    inventory = {"components":[{"bounds":[16,48,48,80],"centroid":[32,64]},
                               {"bounds":[80,48,112,80],"centroid":[96,64]}]}
    return ids,inventory


def test_row_permutation_does_not_move_the_native_windows():
    ids,inventory = example()
    order = np.random.default_rng(471).permutation(len(ids))
    original = HELPER["supports"](inventory,inventory,ids,height=128,width=128)
    permuted = HELPER["supports"](inventory,inventory,ids[order],height=128,width=128)
    for key in original["masks"]:
        assert {tuple(ids[i]) for i in original["masks"][key]}=={tuple(ids[order][i]) for i in permuted["masks"][key]}


def test_ambiguous_native_grid_is_rejected():
    ids,inventory = example()
    ids[1] = ids[0]
    with pytest.raises(ValueError,match="grid"):
        HELPER["supports"](inventory,inventory,ids,height=128,width=128)


def test_native_bfloat16_patch_keeps_unwritten_bytes_and_parent():
    current = torch.arange(24,dtype=torch.bfloat16).reshape(1,8,3)
    before = current.clone()
    source = -current-1
    value = HELPER["patch_latent"](current,source,[1,6])
    assert value.dtype==torch.bfloat16
    assert torch.equal(current,before)
    assert torch.equal(value[:,[1,6]],source[:,[1,6]])
    assert torch.equal(value[:,[0,2,3,4,5,7]],current[:,[0,2,3,4,5,7]])


def test_disjoint_stacked_acts_have_exact_union_and_complement():
    ids,inventory = example()
    masks = HELPER["supports"](inventory,inventory,ids,height=128,width=128)["masks"]
    current = torch.arange(128,dtype=torch.bfloat16).reshape(1,64,2)
    source = -current-1
    patch = HELPER["patch_latent"]
    stacked = patch(patch(current,source,masks["left"]),source,masks["right"])
    assert torch.equal(stacked,patch(current,source,masks["both"]))
    assert torch.equal(stacked[:,masks["outside"]],current[:,masks["outside"]])
    assert len(masks["shifted"])==len(masks["both"])
