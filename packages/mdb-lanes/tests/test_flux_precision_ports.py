"""The diagnostic observer preserves native execution and always detaches."""
import runpy
from pathlib import Path

import pytest


def test_installed_transaction_contract_before_native_work(tmp_path):
    check = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'qualification' / 'flux_precision_cut.py'))['check_transaction_contract']
    result = check(tmp_path/'transaction')
    assert result['verified'] and result['phase'] == 'closed'


@pytest.mark.parametrize('fail', [False, True])
def test_real_flux_ports_preserve_outputs_and_restore_hooks(fail):
    import torch
    from diffusers.models.transformers import transformer_flux2 as native
    capture = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'qualification' / 'flux_precision_cut.py'))['capture_ports']
    torch.manual_seed(19)
    block = native.Flux2TransformerBlock(8, 1, 8).eval().to(torch.bfloat16)
    image = torch.randn(1, 4, 8).to(torch.bfloat16)
    text = torch.randn(1, 3, 8).to(torch.bfloat16)
    mods = [torch.randn(1, 48).to(torch.bfloat16) for _ in range(2)]
    def step():
        return block(image, text, *mods)
    original = native.dispatch_attention_fn
    ports = {}
    counts = {'dispatch': 0}
    hooks = lambda: [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in block.modules()]
    before_hooks = hooks()
    with torch.no_grad():
        expected = step()
        def remember(key, value):
            ports[key] = value.detach().clone()
            if fail and key == 'attention.raw':
                raise RuntimeError('injected observer failure')
        if fail:
            with pytest.raises(RuntimeError, match='injected observer failure'):
                capture(block, native, 3, remember, counts, step)
        else:
            actual = capture(block, native, 3, remember, counts, step)
            for x, y in zip(expected, actual):
                assert torch.equal(x, y)
            for role, value in zip(('text', 'image'), actual):
                remember(role + '.exit', value)
            role_keys = {'norm1', 'modulated', 'q_projection', 'k_projection', 'v_projection',
                         'q_normalized', 'k_normalized', 'q_positioned', 'k_positioned',
                         'attention', 'after_attention', 'norm2', 'mlp_input', 'mlp_paired',
                         'mlp_silu', 'mlp_activation', 'mlp', 'exit'}
            assert set(ports) == {role + '.' + key for role in ('image', 'text') for key in role_keys} | {'attention.raw'}
            assert ports['attention.raw'].shape == (1, 7, 8)
    assert counts == {'dispatch': 1}
    assert native.dispatch_attention_fn is original
    assert hooks() == before_hooks
