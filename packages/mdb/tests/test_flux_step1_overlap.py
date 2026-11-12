import runpy
from pathlib import Path
import numpy as np
import pytest
import torch
from saturn_pub import Act,Session
from mdb.native_navigation import NativeCutNavigator

HERE=Path(__file__).resolve().parents[1];H=runpy.run_path(str(HERE/'qualification/flux_step1_overlap.py'))


def test_overlap_census():
    assert H['census']()==dict(loads=1,encodes=0,noise_draws=0,native_transitions=1292,acts=36,receipts=1328,images=18,candidate_blocks=8,cached_projections=4)


def test_explicit_step1_cache_all_RAM_exits_match_native_and_default_stays_step3():
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'));model,adapter,session=tiny['tiny_session']()
    n=NativeCutNavigator(session,max_steps=112);n.until(logical_step=1,phase='single',index=6);base=n.capture()
    changed=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=1)
    changed.apply(Act.replace('image',base.payload['image']+.125,name='Rsource'))
    changed.apply(Act.replace('text',base.payload['text']+.25,name='Tsource'));shape=changed.capture()
    capture=runpy.run_path(str(HERE/'qualification/flux_single_contraction.py'))['capture_candidate'];block=model.single_transformer_blocks[6]
    with pytest.raises(ValueError):capture(block,base)
    sources=[capture(block,c,logical_step=1) for c in (base,shape)]
    t,w=sources[0]['text_rows'],sources[0]['attention_width']
    for r in (0,1):
        for bits in ('00','10','01','11'):
            cat=H['selected_concat'](sources[0]['ports']['original_concat'],sources[1]['ports']['original_concat'],bits=bits,text_rows=t,width=w)
            with torch.inference_mode():p=block.attn.to_out(cat)[:,t:]
            reference=H['gated_exit']((base if r==0 else shape).payload['image'].float().numpy(),sources[0]['ports']['gate'].float().numpy(),p.float().numpy())
            own='00' if r==0 else '11';selection=''.join(c for i,c in enumerate('AM') if bits[i]!=own[i])
            other=sources[1-r]
            actual=sources[r] if not selection else capture(block,base if r==0 else shape,logical_step=1,source={'parent':(shape if r==0 else base).fingerprint,'concat':other['ports']['original_concat'],'text_rows':t,'attention_width':w},selection=selection)
            np.testing.assert_array_equal(reference,actual['image'].float().numpy())
