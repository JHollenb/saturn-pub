"""Prospective reader predictions and unchanged native suffix for selected routes."""
import hashlib
import runpy
from pathlib import Path


def specs(head,band):
    return [dict(id='base',kind='base'),dict(id='full',kind='full'),
        dict(id='dose375',kind='dose',dose=.375),dict(id='dose625',kind='dose',dose=.625),
        dict(id='head-add',kind='head-add',head=head),dict(id='head-remove',kind='head-remove',head=head),
        dict(id='band-add',kind='band-add',band=band),dict(id='band-remove',kind='band-remove',band=band)]


def ports(base,changed,spec,bf16,nt=512):
    kind=spec['kind'];full=kind in ('full','head-remove','band-remove')
    result={key:(changed if full else base)[key].copy() for key in ('k','v')}
    if kind=='dose':
        for key in result:result[key][:,nt:]=bf16(base[key][:,nt:]+spec['dose']*(changed[key][:,nt:]-base[key][:,nt:]))
    elif kind in ('band-add','band-remove'):
        lo,hi=nt+spec['band']*256,nt+(spec['band']+1)*256
        for key in result:result[key][:,lo:hi]=(base if full else changed)[key][:,lo:hi]
    for key in result:
        if not (result[key][:,:nt]==base[key][:,:nt]).all():raise ValueError('text KV was modified')
    return result


def payload_digest(value):
    import numpy as np
    if hasattr(value,'detach'):return hashlib.sha256(value.detach().view(__import__('torch').uint16).cpu().numpy().astype('<u2',copy=False).tobytes()).hexdigest()
    return hashlib.sha256((np.ascontiguousarray(value,np.float32).view(np.uint32)>>16).astype('<u2').tobytes()).hexdigest()


def run(*,pipe,adapter,parents,source,output,config,request,event):
    import diffusers.models.transformers.transformer_flux2 as native
    import torch
    from saturn_pub import Session
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    core=runpy.run_path(str(Path(__file__).with_name('flux_reader_arithmetic.py')))
    attention=runpy.run_path(str(Path(__file__).with_name('flux_reader_attention.py')))
    routes=runpy.run_path(str(Path(__file__).with_name('flux_reader_routes.py')))
    h=core['H'];read,save,load,sha=[h[k] for k in ('read','save','load_array','sha')]
    prefixes=read(source/'prefixes.json')['operands'];block=pipe.transformer.transformer_blocks[0]
    modules=dict(projection=block.attn.to_add_out,norm=block.norm2_context,ff=block.ff_context)
    for name,module in (('attn.to_add_out',modules['projection']),('ff_context.linear_in',modules['ff'].linear_in),('ff_context.linear_out',modules['ff'].linear_out)):
        record=config['text_parameters'][name+'.weight']
        if payload_digest(module.weight)!=record['descriptor']['sha256']:raise ValueError('original forecast parameter differs')
    counts=dict(sdpa=0,staged_matmuls=0,staged_softmax=0,linear=0,layernorm=0,swiglu=0,
        native_sdpa=0,native_blocks=0,native_transitions=0,images=0,model_loads=1,encodes=0,noise_draws=0)
    phase={'name':'forecast','case':None};allowed=set(modules['ff'].modules())|{modules['projection'],modules['norm']}
    def guard(module,args):
        if phase['name']=='forecast' and module not in allowed:raise ValueError('undeclared native module during primitive forecast')
    handles=[torch.nn.modules.module.register_module_forward_pre_hook(guard)]
    for b in [*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks]:
        handles.append(b.register_forward_hook(lambda m,a,v:counts.__setitem__('native_blocks',counts['native_blocks']+1)))
    observed={}
    def after(m,args):
        if phase['name']=='first':observed['after_attention']=h['array'](output,args[0])
    def mlp(m,args,value):
        if phase['name']=='first':observed['mlp']=h['array'](output,value)
    handles.extend([block.norm2_context.register_forward_pre_hook(after),block.ff_context.register_forward_hook(mlp)])
    prepared={};predictions={};profiles={};tensor=lambda x:torch.from_numpy(x.copy()).to('cuda',torch.bfloat16)
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=config['parents'],source_identity=request['request_sha256'],
        frontier=dict(kind='architectural',boundary='observed positioned text Q/full KV; original parent text/modulation',step=3,block=0),
        execution_scope='Externally counted CUDA native primitives forecast16reader heads/text writes before any native full block.12FlashSDPA/48linear/16norm/16SwiGLU. Then16full native suffixes. No independent Flash kernel implementation or full image forecast.',
        budgets=dict(max_native_calls=16,max_bytes=16<<20,max_wall_ms=600000))
    tx.frontier('selection',dict(head=config['selected_head'],band=config['selected_band'],source='Left route screen only; fixed before native futures',source_audit_sha256=config['route_audit_sha256']))
    with torch.inference_mode():
        for receiver,parent in parents.items():
            b,d=[{f:load(source,a) for f,a in prefixes[receiver+'.'+n].items()} for n in ('base','delta')]
            q=tensor(b['q']);head0,p0=attention['attention'](q,tensor(b['k']),tensor(b['v']),'flash-text',counts)
            head1,p1=attention['attention'](q,tensor(d['k']),tensor(d['v']),'flash-text',counts)
            text,mod=parent.payload['text'],parent.payload['context']['txt_mod']
            for spec in specs(config['selected_head'],config['selected_band']):
                key=receiver+'.'+spec['id'];kv=ports(b,d,spec,core['R']['bf16'])
                prepared[key]=dict(kv=kv,spec=spec,base=b,head_source=head1 if spec['kind']=='head-add' else head0)
                def predict(_,spec=spec,key=key,kv=kv):
                    if spec['kind'] in ('base','head-add'):head=head0
                    elif spec['kind'] in ('full','head-remove'):head=head1
                    else:
                        head,profile=attention['attention'](q,tensor(kv['k']),tensor(kv['v']),'flash-text',counts);profiles[key]=profile
                    if spec['kind']=='head-add':head=routes['mix_heads'](head0,head1,[spec['head']])
                    elif spec['kind']=='head-remove':head=routes['mix_heads'](head1,head0,[spec['head']])
                    values=core['native_tail'](head,text,mod,modules,counts=counts)
                    fields={f:h['array'](output,values[f]) for f in ('head','after_attention','mlp','text')}
                    predictions[key]=dict(fields=fields,ports={f:payload_digest(v) for f,v in kv.items()},spec=spec,
                        structural_prediction=dict(reference='full' if spec['kind'] in ('head-remove','band-remove') else 'base',
                            first_image_matches=('base' if spec['kind']=='head-add' else 'full') if spec['kind'].startswith('head-') else None,
                            earliest_possible_image_change=1 if spec['kind'].startswith('head-') else 0,
                            expected_first_image_change=1 if spec['kind'].startswith('head-') else (0 if spec['kind'] not in ('base',) else None)))
                    return dict(predictions=predictions[key])
                tx.predict(key,predict);event('prediction-persisted',case=key,counts=counts.copy())
            profiles[receiver]=dict(base=p0,full=p1)
        if counts['sdpa']!=12 or counts['linear']!=48 or counts['native_blocks'] or counts['native_sdpa']:raise ValueError('forecast cohort census/order differs')
        seal=tx.seal();verify_transaction(output/'transaction',expected_seal_sha256=seal['seal_sha256'])
        save(output/'predictions.json',predictions);save(output/'profiles.json',profiles);event('all-predictions-sealed',seal_sha256=seal['seal_sha256'],counts=counts.copy())
    original_dispatch=native.dispatch_attention_fn
    footprint={}
    def dispatch(q,k,v,*args,**kwargs):
        counts['native_sdpa']+=1
        if phase['name']!='first':return original_dispatch(q,k,v,*args,**kwargs)
        item=prepared[phase['case']];b=item['base'];nt=b['q'].shape[1]
        for name,value in (('q',q[:,:nt]),('k',k),('v',v)):
            if not torch.equal(value.float().cpu(),torch.from_numpy(b[name])):raise ValueError('native prefix differs before intervention: '+name)
        def install(value,name):
            desired=tensor(item['kv'][name]);out=torch.empty_strided(value.shape,value.stride(),device=value.device,dtype=value.dtype);out.copy_(desired)
            if payload_digest(out)!=predictions[phase['case']]['ports'][name]:raise ValueError('installed ports differ from sealed inputs')
            return out
        k,v=install(k,'k'),install(v,'v')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            result=original_dispatch(q,k,v,*args,**kwargs);torch.cuda.synchronize()
        selected=[dict(name=x.key,calls=x.count) for x in trace.key_averages() if 'scaled_dot' in x.key]
        spec=item['spec'];changed=[]
        if spec['kind'].startswith('head-'):
            base=result;result=result.clone();i=spec['head'];replacement=item['head_source'].reshape(1,nt,24,128)
            result[:,:nt,i,:]=replacement[:,:,i,:];changed=[i]
            others=[j for j in range(24) if j!=i]
            if not torch.equal(result[:,:nt,others],base[:,:nt,others]) or not torch.equal(result[:,nt:],base[:,nt:]):raise ValueError('head edit changed undeclared ports')
        observed['head']=h['array'](output,result[:,:nt].flatten(2,3))
        footprint.update(changed_text_heads=changed,image_heads_preserved=True,kv_sha256={f:payload_digest(x) for f,x in (('k',k),('v',v))},
            selected_aten_operations=selected,q_shape=list(q.shape),q_stride=list(q.stride()),k_stride=list(k.stride()),v_stride=list(v.stride()))
        return result
    cases={}
    def receipt(rec,key):
        with (output/'receipts.jsonl').open('a') as f:
            import json
            f.write(json.dumps(dict(case=key,receipt=rec.to_dict()),sort_keys=True)+'\n')
    native.dispatch_attention_fn=dispatch
    try:
        with torch.inference_mode():
            for key,item in prepared.items():
                receiver=key.split('.')[0];nav=NativeCutNavigator(Session.from_cut(adapter,parents[receiver]),max_steps=27)
                phase.update(name='first',case=key);observed.clear();footprint.clear();records=[]
                def execute(_):
                    rec=nav.step();receipt(rec,key);counts['native_transitions']+=1
                    observed['text']=h['array'](output,nav.session.read('text'))
                    return dict(fields=dict(observed),footprint=dict(footprint))
                tx.execute(key,execute)
                first=dict(observed);first_image=h['array'](output,nav.session.read('image'))
                for local in range(27):
                    if local:
                        phase['name']='continuation';receipt(nav.step(),key);counts['native_transitions']+=1
                    cut=nav.session.capture(retain=False);p=cut.payload
                    fields={f:describe(p[f]) for f in ('text','image','noise_prediction','latent') if f in p}
                    records.append(dict(local=local,point=dict(cut.execution_point),fields=fields))
                phase['name']='decode';path=output/(key+'.png');adapter.decode(nav.session,height=512,width=512).save(path);counts['images']+=1
                cases[key]=dict(first_fields=first,first_image=first_image,footprint=dict(footprint),records=records,
                    latent=h['array'](output,nav.session.read('latent')),image=dict(path=path.name,sha256=sha(path)),spec=item['spec'])
                save(output/'cases'/(key+'.json'),cases[key]);event('native-future-complete',case=key,counts=counts.copy())
    finally:
        native.dispatch_attention_fn=original_dispatch
        for handle in handles:handle.remove()
        tx.close()
    verify_transaction(output/'transaction',expected_seal_sha256=seal['seal_sha256'])
    save(output/'transaction-report.json',tx.report());save(output/'cases.json',cases)
    if counts!={**counts,'sdpa':12,'linear':48,'layernorm':16,'swiglu':16,'native_sdpa':400,'native_blocks':400,'native_transitions':432,'images':16}:
        raise ValueError('native future physical census differs')
    return counts
