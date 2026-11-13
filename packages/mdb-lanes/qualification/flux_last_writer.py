"""Cross single19 residual and raw fused write before actual readout and images."""
import json
import runpy
from pathlib import Path


def cases(receivers):
    return {f'{r}.r{ri}u{ui}':dict(receiver=r,R=ri,U=ui,
        residual_source=opposite(r) if ri else r,write_source=opposite(r) if ui else r)
        for r in receivers for ri,ui in ((0,0),(0,1),(1,0),(1,1))}


def opposite(r):return ('c1-' if r.startswith('c0-') else 'c0-')+r.split('-')[1]


def transplant_active(phase):return phase=='target-writer'


def run(*,pipe,adapter,parents,source,output,config,request,event,**_):
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    from saturn_pub import Act,Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'))
    previous=runpy.run_path(str(here/'flux_commit_factorial.py'));digest=runpy.run_path(str(here/'flux_reader_route_future.py'))['payload_digest']
    read,save,array=[h[k] for k in ('read','save','array')]
    store=LocalStore(output/'store');old=LocalStore(source/'store');origins=read(source/'fresh-parents.json')
    counts=dict(model_loads=1,encodes=0,noise_draws=0,native_blocks=0,native_sdpa=0,native_transitions=0,
        forecast_linear=0,forecast_layernorm=0,forecast_silu=0,mathematical_writes=0,mathematical_commits=0,typed_acts=0,write_transplants=0,source_write_captures=0,images=0)
    phase=dict(name='prefix',case=None);entries={};sources={};predictions={};rows={};specs=cases(parents);handles=[];tx=None
    block=pipe.transformer.single_transformer_blocks[19];first=pipe.transformer.transformer_blocks[0]
    allowed=set(pipe.transformer.norm_out.modules())|{pipe.transformer.proj_out,pipe.transformer.x_embedder,first.norm1,first.attn.to_v}
    def guard(module,args):
        if phase['name']=='forecast' and module not in allowed:raise ValueError('undeclared module in last-writer forecast')
    handles.append(torch.nn.modules.module.register_module_forward_pre_hook(guard))
    for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]:
        handles.append(b.register_forward_hook(lambda m,a,v:counts.__setitem__('native_blocks',counts['native_blocks']+1)))
    for module in allowed:
        kind='forecast_linear' if isinstance(module,torch.nn.Linear) else 'forecast_layernorm' if isinstance(module,torch.nn.LayerNorm) else 'forecast_silu' if isinstance(module,torch.nn.SiLU) else None
        if kind:
            def called(m,a,v,kind=kind):
                if phase['name']=='forecast':counts[kind]+=1
            handles.append(module.register_forward_hook(called))
    original_dispatch=native.dispatch_attention_fn
    def dispatch(*a,**kw):counts['native_sdpa']+=1;return original_dispatch(*a,**kw)
    native.dispatch_attention_fn=dispatch
    def write_hook(module,args,value):
        key=phase['case']
        if phase['name']=='source-writer':sources[key]['U_tensor']=value.detach().clone();sources[key]['U']=array(output,value);counts['source_write_captures']+=1
        if transplant_active(phase['name']):
            spec=specs[key];replacement=sources[spec['write_source']]['U_tensor'].clone()
            if (replacement.shape,replacement.stride(),replacement.dtype,replacement.device)!=(value.shape,value.stride(),value.dtype,value.device):raise ValueError('write transplant layout differs')
            rows[key]['computed_U']=array(output,value);rows[key]['installed_U']=array(output,replacement);counts['write_transplants']+=1
            return replacement
    def value_hook(module,args,value):
        if phase['name']=='first-reader':rows[phase['case']]['values']=array(output,value.unflatten(-1,(24,128)))
    handles.append(block.attn.to_out.register_forward_hook(write_hook));handles.append(first.attn.to_v.register_forward_hook(value_hook))
    def receipt(value,key):
        with (output/'receipts.jsonl').open('a') as f:f.write(json.dumps(dict(case=key,receipt=value.to_dict()),sort_keys=True)+'\n')
    try:
        with torch.inference_mode():
            for r in parents:
                phase['case']=r;sources[r]={}
                nav=NativeCutNavigator(Session.from_cut(adapter,old.load(origins[r]['initial'],device='cuda')),max_steps=84)
                for i in range(84):
                    phase['name']='source-writer' if i==81 else 'prefix'
                    receipt(nav.step(),r);counts['native_transitions']+=1
                    if i==80:
                        cut=nav.session.capture(retain=False);store.save(cut);entries[r]=cut
                        if (cut.payload['step'],cut.payload['phase'],cut.payload['cursor'])!=(2,'single',19):raise ValueError('last writer entry clock differs')
                        p=cut.payload;R=torch.cat([p['text'],p['image']],dim=1);gate=p['context']['single_mod'].reshape(1,1,-1).chunk(3,dim=-1)[2]
                        sources[r].update(entry=cut.fingerprint,R=array(output,R),gate=array(output,gate),Z2=array(output,p['latent']),
                            step2_temb=describe(p['context']['temb']),step3_modulation=describe(parents[r].payload['context']['img_mod']))
                    if i==81:sources[r]['Y']=array(output,torch.cat([nav.session.read('text'),nav.session.read('image')],dim=1))
                    if i==82:sources[r]['N2']=array(output,nav.session.capture(retain=False).payload['noise_prediction'])
                if describe(nav.session.read('latent'))!=describe(parents[r].payload['latent']):raise ValueError('own prefix differs from prior source')
                sources[r]['Z3']=array(output,nav.session.read('latent'));event('writer-source-captured',case=r,counts=counts.copy())
            public_sources={r:{k:v for k,v in s.items() if k!='U_tensor'} for r,s in sources.items()};save(output/'sources.json',public_sources)
            tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity={r:c.fingerprint for r,c in entries.items()},source_identity=request['request_sha256'],
                frontier=dict(kind='architectural',step=2,phase='single',cursor=19,operands='observed R19/U19, recipient gate/temb/Z2/schedule; known step3 modulation; original weights'),
                execution_scope='Sealed Y19,N2,Z3,I3,V3 before mixed target blocks; actual native outputs feed31-operation image continuations.',
                budgets=dict(max_native_calls=16,max_bytes=32<<20,max_wall_ms=600000))
            tx.frontier('hypotheses',config['semantic_hypotheses']);tx.frontier('sources',public_sources)
            for name,module in [('norm_out.linear.weight',pipe.transformer.norm_out.linear),('proj_out.weight',pipe.transformer.proj_out),
                ('x_embedder.weight',pipe.transformer.x_embedder),('transformer_blocks.0.attn.to_v.weight',first.attn.to_v)]:
                if digest(module.weight)!=config['forecast_parameter_sha256'][name]:raise ValueError('forecast parameter differs from recipe')
            phase['name']='forecast'
            for key,spec in specs.items():
                def predict(_,key=key,spec=spec):
                    r=spec['receiver'];p=entries[r].payload;rp=entries[spec['residual_source']].payload
                    R=torch.cat([rp['text'],rp['image']],dim=1);gate=p['context']['single_mod'].reshape(1,1,-1).chunk(3,dim=-1)[2]
                    Y=R+gate*sources[spec['write_source']]['U_tensor'];counts['mathematical_writes']+=1
                    noise=pipe.transformer.proj_out(pipe.transformer.norm_out(Y[:,512:],p['context']['temb']))
                    Z,product=previous['commit'](p['latent'],noise,p['sigmas']);counts['mathematical_commits']+=1
                    I=pipe.transformer.x_embedder(Z);shift,scale,*_=parents[r].payload['context']['img_mod'].reshape(1,1,-1).chunk(6,dim=-1)
                    V=first.attn.to_v((1+scale)*first.norm1(I)+shift).unflatten(-1,(24,128))
                    predictions[key]=dict(spec=spec,Y=array(output,Y),N2=array(output,noise),Z3=array(output,Z),image=array(output,I),values=array(output,V),product=array(output,product))
                    return previous['forecast_record'](predictions[key])
                tx.predict(key,predict)
            if counts['native_transitions']!=336 or counts['write_transplants'] or counts['images']:raise ValueError('target ran before seal')
            seal=tx.seal();verify_transaction(output/'transaction',expected_seal_sha256=seal['seal_sha256']);save(output/'predictions.json',predictions)
            event('all-predictions-sealed',seal_sha256=seal['seal_sha256'],counts=counts.copy())
            for key,spec in specs.items():
                r=spec['receiver'];rows[key]=dict(spec=spec)
                def execute(_):
                    nav=NativeCutNavigator(Session.from_cut(adapter,entries[r]),max_steps=31);rp=entries[spec['residual_source']].payload
                    phase.update(name='setup-target',case=key)
                    for f in ('image','text'):receipt(nav.apply(Act.replace(f,rp[f],name=key+'.'+f)),key);counts['typed_acts']+=1
                    current=nav.session.capture(retain=False).payload;own=entries[r].payload
                    protected=[f for f in own if f not in ('image','text')]
                    if any(describe(current[f])!=describe(own[f]) for f in protected):raise ValueError('R Act changed receiving closure')
                    rows[key]['closure']={f:describe(current[f]) for f in protected};rows[key]['R']=array(output,torch.cat([current['text'],current['image']],dim=1))
                    rows[key]['gate']=array(output,current['context']['single_mod'].reshape(1,1,-1).chunk(3,dim=-1)[2]);rows[key]['trajectory']=[]
                    for local in range(31):
                        phase['name']='target-writer' if local==0 else 'first-reader' if local==4 else 'continuation'
                        receipt(nav.step(),key);counts['native_transitions']+=1
                        if local in (0,1,2,3,4,30):
                            cut=nav.session.capture(retain=False);rows[key]['trajectory'].append(dict(local=local,point=dict(cut.execution_point),fields=describe(cut.payload)))
                            if local in (0,1,2):
                                store.save(cut);rows[key].setdefault('checkpoints',{})[('readout','scheduler','project')[local]]=cut.fingerprint
                            if local==0:rows[key]['Y']=array(output,torch.cat([cut.payload['text'],cut.payload['image']],dim=1))
                            if local==1:rows[key]['N2']=array(output,cut.payload['noise_prediction'])
                            if local==2:rows[key]['Z3']=array(output,cut.payload['latent'])
                            if local==3:rows[key]['image']=array(output,cut.payload['image']);rows[key]['step3_modulation']=describe(cut.payload['context']['img_mod'])
                    rows[key]['latent']=array(output,nav.session.read('latent'));phase['name']='decode'
                    path=output/(key+'.png');adapter.decode(nav.session,height=512,width=512).save(path);counts['images']+=1
                    rows[key]['png']=dict(path=path.name,sha256=h['sha'](path));return dict(case=key,outputs=rows[key])
                tx.execute(key,execute);save(output/'cases'/(key+'.json'),rows[key]);event('native-future-complete',case=key,counts=counts.copy())
            verify_transaction(output/'transaction',expected_seal_sha256=seal['seal_sha256']);save(output/'transaction-report.json',tx.report());save(output/'cases.json',rows)
            if counts!=config['planned_counts']:raise ValueError('last-writer physical census differs: '+str(counts))
            return counts
    finally:
        native.dispatch_attention_fn=original_dispatch
        for handle in handles:handle.remove()
        if tx is not None:tx.close()
