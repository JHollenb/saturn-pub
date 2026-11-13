"""Typed latent-source mediation before the established final-pass V reader."""
import json
import runpy
from pathlib import Path


def run(*,pipe,adapter,parents,source,output,config,request,event,**_):
    import diffusers.models.transformers.transformer_flux2 as native
    import numpy as np
    import torch
    from saturn_pub import Act, Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.native_navigation import NativeCutNavigator
    here=Path(__file__).resolve().parent
    future=runpy.run_path(str(here/'flux_reader_route_future.py'))
    reader=runpy.run_path(str(here/'flux_latent_reader.py'));h=reader['H']
    read,save=[h[k] for k in ('read','save')];rows=read(source/'fresh-parents.json');original_prefix=read(source/'prefixes.json')['operands']
    source_store=LocalStore(source/'store');store=LocalStore(output/'store')
    counts=dict(native_transitions=0,native_sdpa=0,native_blocks=0,prefix_stops=0,latent_acts=0)
    active=dict(mode='continuation',text_rows=512);newparents={};prefixes={};records={};bindings={}
    handles=[b.register_forward_hook(lambda m,a,v:counts.__setitem__('native_blocks',counts['native_blocks']+1)) for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]]
    def receipt(value,case):
        with (output/'upstream-receipts.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,receipt=value.to_dict()),sort_keys=True)+'\n')
    try:
        with torch.inference_mode(),reader['ReaderBoundary'](native,counts,active) as boundary:
            for r in config['upstream_receivers']:
                other=('c1-' if r.startswith('c0-') else 'c0-')+r.split('-')[1]
                initial=source_store.load(rows[r]['initial'],device='cuda')
                nav=NativeCutNavigator(Session.from_cut(adapter,initial),max_steps=84)
                active['mode']='continuation'
                for _ in range(84):receipt(nav.step(),r);counts['native_transitions']+=1
                project=nav.session.capture(retain=False);store.save(project)
                if (project.payload['step'],project.payload['phase'],project.payload['cursor'])!=(3,'project',0):raise ValueError('upstream project clock differs')
                for role in ('own','swap'):
                    key=r+'-'+role;nav=NativeCutNavigator(Session.from_cut(adapter,project),max_steps=2)
                    active['mode']='continuation'
                    if role=='swap':
                        receipt(nav.apply(Act.replace('latent',parents[other].payload['latent'].clone(),name=key+'.latent')),key);counts['latent_acts']+=1
                        if describe(nav.session.read('conditioning'))!=describe(initial.payload['conditioning']):raise ValueError('latent Act changed conditioner')
                    receipt(nav.step(),key);counts['native_transitions']+=1
                    parent=nav.session.capture(retain=False);store.save(parent);newparents[key]=parent
                    if role=='own' and describe(parent.payload)!=describe(parents[r].payload):raise ValueError('upstream own-parent replay differs')
                    expected_image=parents[other if role=='swap' else r].payload['image']
                    bindings[key]=dict(project_parent=project.fingerprint,recipient=r,latent_source=other if role=='swap' else r,
                        image_matches_source=torch.equal(parent.payload['image'],expected_image),text_matches_recipient=torch.equal(parent.payload['text'],parents[r].payload['text']),
                        latent=describe(parent.payload['latent']),conditioning=describe(parent.payload['conditioning']))
                    active['mode']='prefix';boundary.captured=None
                    try:nav.step()
                    except reader['PrefixStop']:pass
                    else:raise ValueError('upstream reader prefix did not stop')
                    if boundary.captured is None or describe(nav.session.capture(retain=False).payload)!=describe(parent.payload):raise ValueError('upstream prefix mutated state')
                    expected_v=h['load_array'](source,original_prefix[(other if role=='swap' else r)+'.base']['v'])[:,512:]
                    actual_v=boundary.captured['v'][:,512:];diff=actual_v.astype(np.float64)-expected_v
                    bindings[key]['image_values_from_latent_source']=dict(exact=np.array_equal(actual_v,expected_v),rms=float(np.sqrt(np.mean(diff*diff))),max_abs=float(np.abs(diff).max()),values=int(actual_v.size))
                    bindings[key]['text_ports_from_recipient']={}
                    for field in ('q','k','v'):
                        a=boundary.captured[field][:,:512];b=h['load_array'](source,original_prefix[r+'.base'][field])[:,:512]
                        bindings[key]['text_ports_from_recipient'][field]=dict(exact=np.array_equal(a,b),rms=float(np.sqrt(np.mean((a.astype(np.float64)-b)**2))),values=int(a.size))
                    receipt(nav.session.attempts[-1],key);prefixes[key]=boundary.captured
                    records[key]={**rows[r],'parent':parent.fingerprint,'point':dict(parent.execution_point),'project_parent':project.fingerprint,'latent_source':bindings[key]['latent_source']}
                    event('upstream-parent-captured',case=key,counts=counts.copy())
    finally:
        for handle in handles:handle.remove()
    operands={}
    for key,base in prefixes.items():
        opposite=key.rsplit('-',1)[0]+('-swap' if key.endswith('-own') else '-own')
        operands[key+'.base']={f:h['array'](output,v) for f,v in base.items()}
        # Only V differs in this mediator intervention; Q and K stay recipient-native.
        operands[key+'.delta']={f:h['array'](output,prefixes[opposite]['v'] if f=='v' else v) for f,v in base.items()}
    save(output/'prefixes.json',dict(operands=operands,counts=counts.copy()))
    save(output/'fresh-parents.json',records);save(output/'upstream-bindings.json',bindings)
    return future['run'](pipe=pipe,adapter=adapter,parents=newparents,source=output,output=output,config=config,request=request,event=event,initial_counts=counts)
