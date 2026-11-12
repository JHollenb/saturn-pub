import runpy
from pathlib import Path
import torch
from saturn_pub import Session
from mdb.native_navigation import NativeCutNavigator
HERE=Path(__file__).resolve().parents[1];H=runpy.run_path(str(HERE/'qualification/flux_interaction_writes.py'))


def test_clock_budget_and_composition_preserve_text_and_unselected_partition():
    points={'a':{'phase':'single','logical_step':2,'index':18},'b':{'phase':'single','logical_step':3,'index':18}}
    assert H['census'](points)==dict(loads=1,encodes=0,noise_draws=0,native_transitions=272,acts=32,receipts=304,images=16,candidate_blocks=8,cached_projections=10)
    c={b:torch.full((1,3,8),float(i),dtype=torch.bfloat16) for i,b in enumerate(('00','10','01','11'))};r=H['compose'](c,text_rows=1,width=2)
    for v in r.values():assert torch.equal(v[:,:1],c['11'][:,:1])
    assert torch.equal(r['Alinear'][:,1:,2:],c['11'][:,1:,2:])
    assert torch.equal(r['Mlinear'][:,1:,:2],c['11'][:,1:,:2])


def test_composed_native_cached_exits_match_full_block_substitutions():
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'));model,adapter,session=tiny['tiny_session']();nav=NativeCutNavigator(session,max_steps=112);nav.until(logical_step=2,phase='single',index=0);root=nav.session.capture(retain=False)
    capture=runpy.run_path(str(HERE/'qualification/flux_single_contraction.py'))['capture_candidate'];cuts={};ports={}
    for b in ('00','10','01','11'):
        state=root.payload;state['image']+=int(b[0])*.2;state['text']+=int(b[1])*.3;cut=Session(adapter,state,parent=root.fingerprint).capture(retain=False);cuts[b]=cut
        ports[b]=capture(model.single_transformer_blocks[0],cut,logical_step=2,single_index=0)
    t,w=ports['11']['text_rows'],ports['11']['attention_width'];mix=H['compose']({b:v['ports']['original_concat'] for b,v in ports.items()},text_rows=t,width=w)
    for case,cat in mix.items():
        with torch.inference_mode():p=model.single_transformer_blocks[0].attn.to_out(cat)
        cached=cuts['11'].payload['image']+ports['11']['ports']['gate']*p[:,t:]
        actual=capture(model.single_transformer_blocks[0],cuts['11'],logical_step=2,single_index=0,selection='AM',source={'parent':'derived-four-native-sources','concat':cat,'text_rows':t,'attention_width':w})
        assert torch.equal(cached,actual['image']) and torch.equal(actual['text'],ports['11']['text'])
