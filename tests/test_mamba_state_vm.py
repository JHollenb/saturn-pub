from __future__ import annotations

from hashlib import sha256

import pytest
import torch

from mdb.mamba_state_vm import MambaStateVM, MambaStateVMError
from mdb.model_os_mamba_state import (
    OP_FORCED_INSTALL,
    OP_READ,
    OP_RESIDUAL_WRITE,
)


ABI = sha256(b"saturn-mvm-tiny-mamba-fp32").hexdigest()


@pytest.fixture
def task(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(23)
    model = transformers.MambaForCausalLM(
        transformers.MambaConfig(
            vocab_size=64, hidden_size=16, state_size=8, num_hidden_layers=3,
            intermediate_size=32, conv_kernel=4, time_step_rank=4,
        )
    ).eval().to(torch.float32)
    vm = MambaStateVM(model, model_abi_sha256=ABI, model_id="tiny-mamba",
                      tokenizer_identity="tiny-tokenizer")
    # Three same-length token spans differing only before a single carrier cut.
    cut_pos = 3
    base_ids = [1, 7, 3, 11, 20, 21, 22, 23]
    generic_ids = [1, 7, 3, 12, 20, 21, 22, 23]
    target_ids = [1, 7, 3, 13, 20, 21, 22, 23]
    suffix = base_ids[cut_pos + 1:]
    readout = {"answer_base": 20, "answer_target": 21, "distractors": [22, 23, 30]}
    programs = {
        name: vm.capture(ids, cut_pos=cut_pos, pending_token_id=ids[cut_pos + 1]).relocated(device="cpu")
        for name, ids in (("base", base_ids), ("generic", generic_ids), ("target", target_ids))
    }
    vm.mount("sess", programs["base"], suffix=suffix, span_ids=base_ids, cut_pos=cut_pos,
             positions=[cut_pos], readout=readout)
    return vm, programs, dict(base_ids=base_ids, generic_ids=generic_ids, target_ids=target_ids,
                              suffix=suffix, cut_pos=cut_pos, readout=readout)


def test_exact_validity_gates_are_zero_and_replay_matches(task):
    vm, _programs, _meta = task
    gate = vm.gates("sess")
    assert gate["noop_max_abs"] == 0.0 and gate["noop_exact"] is True
    assert gate["uninstall_max_abs"] == 0.0 and gate["uninstall_exact"] is True
    # staged-from-cut vs native full forward is a tight recurrent replay
    assert gate["replay_max_abs"] < 1e-2


def test_fork_is_a_bit_exact_noop(task):
    vm, _programs, _meta = task
    child = vm.fork("sess", "sess-fork")
    assert child["op"] == "fork" and child["forked_from"] == "sess"
    assert child["fork_noop_max_abs"] == 0.0 and child["fork_exact"] is True
    assert child["program_fingerprint"] == vm._session("sess").base_program.manifest()["fingerprint"]


def test_swap_read_and_residual_and_forced_modes_commit_with_receipts(task):
    vm, programs, _meta = task
    read = vm.swap("sess", donor=programs["generic"], layers=[2], mode=OP_READ)
    assert read["status"] == "committed" and read["formed_state_read"] is True
    assert "necessity_delta" in read and read["receipt"]["decision"] == "commit"

    forced = vm.swap("sess", donor=programs["target"], layers=[1, 2], mode=OP_FORCED_INSTALL)
    assert forced["forced_formed_install"] is True and forced["status"] == "committed"

    resid = vm.swap("sess", donor=programs["target"], layers=[2], mode=OP_RESIDUAL_WRITE)
    assert resid["residual_write"] is True and resid["status"] == "committed"
    # the mounted session stays anchored on its base after each measurement
    assert vm._session("sess").kernel.current_statecut == vm._session("sess").base_cut


def test_diff_flags_changed_layers_and_identity(task):
    vm, programs, _meta = task
    changed = vm.diff(programs["base"], programs["generic"])
    assert changed["changed_layers"], "base and generic must differ at the carrier layer onward"
    same = vm.diff(programs["base"], programs["base"])
    assert same["changed_layers"] == [] and all(same["identical_layers"])


def test_sweep_returns_per_layer_profile_gates_and_batched_vs_kernel_audit(task):
    vm, programs, _meta = task
    out = vm.sweep("sess", donor=programs["generic"], mode=OP_READ)
    n = out["n_layers"]
    assert len(out["necessity_per_layer"]) == n == 3
    assert out["gates"]["noop_max_abs"] == 0.0 and out["gates"]["uninstall_max_abs"] == 0.0
    # the batched per-layer path equals one audited kernel swap at the dominant layer
    assert out["audit"]["max_abs_batched_vs_kernel"] <= 1e-3

    rw = vm.sweep("sess", donor=programs["base"], target=programs["target"], mode=OP_RESIDUAL_WRITE)
    assert len(rw["resid_write_per_layer"]) == n


def test_clock_canary_and_write_vs_reference(task):
    vm, programs, _meta = task
    clock = vm.clock("sess", reference=programs["generic"])
    assert clock["n_layers"] == 3
    assert len(clock["surv_norm_frac"]) == 3 and len(clock["write_norm"]) == 3
    # recomputed subject state matches the native cache recurrent state
    assert clock["canary_recurrent_max_abs"] <= 1e-3
    assert len(clock["write_vs_reference"]["delta_H_l2"]) == 3


def test_unmount_is_durable_and_proves_uninstall(task):
    vm, _programs, _meta = task
    out = vm.unmount("sess")
    assert out["uninstall_max_abs"] == 0.0 and out["uninstall_exact"] is True
    assert out["durable"] is True
    with pytest.raises(MambaStateVMError):
        vm._session("sess")
