"""Separate inherited Z2 from the current N2 write, through two consumers."""
import json
import runpy
from pathlib import Path


def forecast_frontier():
    return dict(kind='architectural',step=2,phase='scheduler',operands='observed Z2,N2,sigmas; known original step3 modulation/parameters')


def forecast_record(value):
    return dict(predictions=value)


def commit(latent,noise,sigmas,step=2):
    """Preserve the original scheduler's multiply, addition and cast order."""
    import torch
    product=(sigmas[step+1]-sigmas[step])*noise
    return (latent.to(torch.float32)+product).to(noise.dtype),product


def run(*,pipe,adapter,parents,source,output,config,request,event,**_):
    import diffusers.models.transformers.transformer_flux2 as native
    import torch
    from saturn_pub import Act,Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import clone,describe

    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator
    here=Path(__file__).resolve().parent
    reader=runpy.run_path(str(here/'flux_latent_reader.py'));h=reader['H']
    digest=runpy.run_path(str(here/'flux_reader_route_future.py'))['payload_digest']
    read,save,array=[h[k] for k in ('read','save','array')]
    store=LocalStore(output/'store');origins=read(source/'fresh-parents.json');source_store=LocalStore(source/'store')
    counts=dict(model_loads=1,encodes=0,noise_draws=0,native_blocks=0,native_sdpa=0,native_transitions=0,
        mathematical_commits=0,forecast_linear=0,forecast_layernorm=0,native_scheduler_components=0,latent_acts=0,images=0)
    phase=dict(name='prefix',case=None);sources={};projects={};predictions={};case_specs={};cases={};value_sources={};base_attention={}
    modules={pipe.transformer.x_embedder,pipe.transformer.transformer_blocks[0].norm1,pipe.transformer.transformer_blocks[0].attn.to_v}
    def guard(module,args):
        if phase['name']=='forecast' and module not in modules:raise ValueError('undeclared module in commit/value forecast')
        if phase['name']=='component':raise ValueError('scheduler component entered learned module')
    handles=[torch.nn.modules.module.register_module_forward_pre_hook(guard)]
    for block in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]:
        handles.append(block.register_forward_hook(lambda m,a,v:counts.__setitem__('native_blocks',counts['native_blocks']+1)))
    original_dispatch=native.dispatch_attention_fn
    def dispatch(q,k,v,*args,**kwargs):
        counts['native_sdpa']+=1
        if phase['name']=='first':
            key=phase['case'];spec=case_specs[key];r=spec['receiver']
            if spec['kind']=='latent' and spec['z']==spec['n']==0:
                base_attention[r]={f:describe(t) for f,t in [('q',q),('k',k),('text_v',v[:,:512]) ]}
            if spec['kind']=='band':
                for f,t in [('q',q),('k',k),('text_v',v[:,:512])]:
                    if describe(t)!=base_attention[r][f]:raise ValueError('band arm changed receiving query/key/text values')
                outside=[describe(v[:,:1024]),describe(v[:,1280:])];layout=dict(shape=list(v.shape),stride=list(v.stride()))
                v=v.clone();v[:,1024:1280]=value_sources[spec['producer']][:,512:768]
                after=[describe(v[:,:1024]),describe(v[:,1280:])]
                if outside!=after:raise ValueError('band arm changed outside-band values')
                cases[key]['installed_band']=array(output,v[:,1024:1280])
                cases[key]['protected_ports']=base_attention[r]
                cases[key]['outside_band_values']=dict(before=outside,after=after,layout_before=layout,layout_after=dict(shape=list(v.shape),stride=list(v.stride())))
        return original_dispatch(q,k,v,*args,**kwargs)
    def values_hook(module,args,value):
        if phase['name']=='first' and case_specs[phase['case']]['kind']=='latent':
            key=phase['case'];value_sources[key]=value.unflatten(-1,(24,128)).detach().clone()
            cases[key]['actual_values']=array(output,value_sources[key])
    handles.append(pipe.transformer.transformer_blocks[0].attn.to_v.register_forward_hook(values_hook))
    native.dispatch_attention_fn=dispatch
    def receipt(value,key):
        with (output/'receipts.jsonl').open('a') as stream:stream.write(json.dumps(dict(case=key,receipt=value.to_dict()),sort_keys=True)+'\n')
    tx=None
    try:
        with torch.inference_mode():
            for r in parents:
                nav=NativeCutNavigator(Session.from_cut(adapter,source_store.load(origins[r]['initial'],device='cuda')),max_steps=84)
                for i in range(84):
                    receipt(nav.step(),r);counts['native_transitions']+=1
                    if i==82:
                        cut=nav.session.capture(retain=False);store.save(cut);sources[r]=cut
                        if (cut.payload['step'],cut.payload['phase'])!=(2,'scheduler'):raise ValueError('source commit clock differs')
                projects[r]=nav.session.capture(retain=False);store.save(projects[r])
                if describe(projects[r].payload['latent'])!=describe(parents[r].payload['latent']):raise ValueError('own prefix replay changed Z3')
                event('commit-source-captured',case=r,counts=counts.copy())
            save(output/'source-cuts.json',dict(scheduler={r:s.fingerprint for r,s in sources.items()},project={r:s.fingerprint for r,s in projects.items()}))
            save(output/'source-operands.json',{r:dict(latent=array(output,s.payload['latent']),noise=array(output,s.payload['noise_prediction']),
                sigmas=s.payload['sigmas'].float().cpu().tolist(),sigmas_descriptor=describe(s.payload['sigmas']),
                modulation=describe(parents[r].payload['context']['img_mod'])) for r,s in sources.items()})
            tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity={r:s.fingerprint for r,s in sources.items()},source_identity=request['request_sha256'],
                frontier=forecast_frontier(),
                execution_scope='Predict mixed commits and complete I3/V3 before any hybrid component target or future; images are native observations.',
                budgets=dict(max_native_calls=28,max_bytes=32<<20,max_wall_ms=600000))
            tx.frontier('hypotheses',config['semantic_hypotheses'])
            tx.frontier('source-cuts',read(output/'source-cuts.json'))
            block=pipe.transformer.transformer_blocks[0]
            for name,module in [('x_embedder.weight',pipe.transformer.x_embedder),('transformer_blocks.0.attn.to_v.weight',block.attn.to_v)]:
                if digest(module.weight)!=config['forecast_parameter_sha256'][name]:raise ValueError('forecast weight differs from original recipe')
            phase['name']='forecast'
            for r in parents:
                other=('c1-' if r.startswith('c0-') else 'c0-')+r.split('-')[1]
                for zi,ni in [(0,0),(0,1),(1,0),(1,1)]:
                    key=f'{r}.latent-z{zi}n{ni}';spec=dict(receiver=r,kind='latent',z=zi,n=ni,z_source=other if zi else r,n_source=other if ni else r);case_specs[key]=spec
                    def predict(_,spec=spec,key=key,r=r):
                        z,p=commit(sources[spec['z_source']].payload['latent'],sources[spec['n_source']].payload['noise_prediction'],sources[r].payload['sigmas'])
                        image=pipe.transformer.x_embedder(z);counts['forecast_linear']+=1
                        shift,scale,*_=torch.chunk(parents[r].payload['context']['img_mod'].reshape(1,1,-1),6,dim=-1)
                        normalized=block.norm1(image);counts['forecast_layernorm']+=1
                        values=block.attn.to_v((1+scale)*normalized+shift).unflatten(-1,(24,128));counts['forecast_linear']+=1;counts['mathematical_commits']+=1
                        predictions[key]=dict(spec=spec,latent=array(output,z),product=array(output,p),image=array(output,image),values=array(output,values))
                        return forecast_record(predictions[key])
                    tx.predict(key,predict)
                for zi,ni in [(0,1),(1,0),(1,1)]:
                    key=f'{r}.band-z{zi}n{ni}';producer=f'{r}.latent-z{zi}n{ni}'
                    case_specs[key]=dict(receiver=r,kind='band',z=zi,n=ni,producer=producer)
                    tx.predict(key,lambda _,producer=producer,r=r:forecast_record(dict(producer=producer,values=predictions[producer]['values'],band=[512,768],receiving_parent=projects[r].fingerprint)))
            if counts['native_transitions']!=336 or counts['native_scheduler_components'] or counts['images']:raise ValueError('hybrid target executed before seal')
            seal=tx.seal();verify_transaction(output/'transaction',expected_seal_sha256=seal['seal_sha256'])
            save(output/'predictions.json',predictions);save(output/'case-specs.json',case_specs)
            event('all-predictions-sealed',seal_sha256=seal['seal_sha256'],counts=counts.copy())
            # Every native producer runs before a band-only branch consumes its V.
            order=[k for k,s in case_specs.items() if s['kind']=='latent']+[k for k,s in case_specs.items() if s['kind']=='band']
            for key in order:
                spec=case_specs[key];r=spec['receiver'];cases[key]=dict(spec=spec)
                def execute(_):
                    nav=NativeCutNavigator(Session.from_cut(adapter,projects[r]),max_steps=28)
                    if spec['kind']=='latent':
                        phase['name']='component';p=clone(sources[r].payload)
                        p['latent']=clone(sources[spec['z_source']].payload['latent']);p['noise_prediction']=clone(sources[spec['n_source']].payload['noise_prediction'])
                        reference=adapter.advance(p);counts['native_scheduler_components']+=1
                        cases[key]['component']=dict(kind='explicit original scheduler component input substitution; not a Session Act',
                            source_cut=sources[r].fingerprint,z=describe(p['latent']),noise=describe(p['noise_prediction']),latent=array(output,reference['latent']))
                        receipt(nav.apply(Act.replace('latent',reference['latent'],name=key+'.commit-result')),key);counts['latent_acts']+=1
                    phase.update(name='continuation',case=key)
                    points=[]
                    for local in range(28):
                        phase['name']='first' if local==1 else 'continuation'
                        receipt(nav.step(),key);counts['native_transitions']+=1
                        if local==0:
                            cases[key]['project_image']=array(output,nav.session.read('image'))
                            cases[key]['project_text']=describe(nav.session.read('text'))
                        if local in (0,1,26,27):
                            cut=nav.session.capture(retain=False);points.append(dict(local=local,point=dict(cut.execution_point),fields=describe(cut.payload)))
                            if local==0:cases[key]['project_modulation']=describe(cut.payload['context']['img_mod'])
                        if local==1:cases[key]['joint0_image']=array(output,nav.session.read('image'))
                    cases[key]['trajectory']=points;cases[key]['latent']=array(output,nav.session.read('latent'))
                    phase['name']='decode';path=output/(key+'.png');adapter.decode(nav.session,height=512,width=512).save(path);counts['images']+=1
                    cases[key]['png']=dict(path=path.name,sha256=h['sha'](path))
                    return dict(case=key,outputs=cases[key])
                tx.execute(key,execute);save(output/'cases'/(key+'.json'),cases[key]);event('native-future-complete',case=key,counts=counts.copy())
            verify_transaction(output/'transaction',expected_seal_sha256=seal['seal_sha256'])
            save(output/'transaction-report.json',tx.report());save(output/'cases.json',cases)
            if counts!=config['planned_counts']:raise ValueError('commit-factorial physical census differs')
            return counts
    finally:
        native.dispatch_attention_fn=original_dispatch
        for handle in handles:handle.remove()
        if tx is not None:tx.close()
