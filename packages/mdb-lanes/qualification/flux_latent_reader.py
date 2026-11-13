"""Latent closure and prospective key/value reader decomposition in native FLUX.

The numerical forecast starts at captured, positioned native Q/K/V. It predicts
the text head and receiving text write, not a raw-prompt or whole-image future.
"""
import hashlib
import json
import runpy
from pathlib import Path

H=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
R=runpy.run_path(str(Path(__file__).with_name('flux_reentry_reference.py')))


def wide(value):
    if isinstance(value,dict):return {k:wide(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(wide(v) for v in value)
    if hasattr(value,'detach'):
        return (value.detach().float() if value.is_floating_point() else value.detach()).cpu().numpy().copy()
    return value


def predict_text(q,k,v,text,txt_mod,weights,*,eps,query_chunk=64,projection_chunk=256):
    import numpy as np
    q,k,v,text,txt_mod=map(R['_native'],(q,k,v,text,txt_mod))
    width=q.shape[-2]*q.shape[-1];head=np.empty((1,q.shape[1],width),np.float32)
    keys=k.astype(np.float64).transpose(0,2,3,1)
    values=v.astype(np.float64).transpose(0,2,1,3)
    for first in range(0,q.shape[1],query_chunk):
        last=min(first+query_chunk,q.shape[1])
        logits=q[:,first:last].astype(np.float64).transpose(0,2,1,3)@keys/np.sqrt(float(q.shape[-1]))
        logits-=logits.max(axis=-1,keepdims=True)
        p=np.exp(logits);p/=p.sum(axis=-1,keepdims=True)
        head[:,first:last]=R['bf16']((p@values).transpose(0,2,1,3).reshape(1,last-first,width))
    mods=np.split(txt_mod.reshape(1,1,6*width),6,axis=-1)
    attended=R['_linear'](head,weights,'attn.to_add_out',projection_chunk)
    after=R['bf16'](text+R['bf16'](mods[2]*attended))
    ff_input=R['_modulate'](after,mods[3],mods[4],eps)
    paired=R['_linear'](ff_input,weights,'ff_context.linear_in',projection_chunk)
    u,w=np.split(paired,2,axis=-1)
    activation=R['bf16'](R['_silu'](u)*w)
    mlp=R['_linear'](activation,weights,'ff_context.linear_out',projection_chunk)
    return dict(head=head,after_attention=after,mlp=mlp,text=R['bf16'](after+R['bf16'](mods[5]*mlp)))


class PrefixStop(Exception):
    pass


class ReaderBoundary:
    """One scoped dispatcher boundary; original native consumer executes once."""
    def __init__(self,native,counts,active):
        self.native,self.counts,self.active=native,counts,active
        self.original=native.dispatch_attention_fn
        self.captured=None;self.expected=None;self.key_source=None;self.value_source=None
        self.fields={};self.footprint=None

    def __enter__(self):
        self.native.dispatch_attention_fn=self.dispatch;return self

    def __exit__(self,*error):
        self.native.dispatch_attention_fn=self.original

    def dispatch(self,q,k,v,*args,**kwargs):
        import torch
        mode=self.active['mode'];nt=self.active['text_rows']
        if mode=='prefix':
            self.captured=dict(q=wide(q[:,:nt]),k=wide(k),v=wide(v))
            self.counts['prefix_stops']+=1
            raise PrefixStop('intentional stop before original attention consumer')
        if mode not in ('closure','reader','continuation'):
            raise ValueError('attention target outside released native window')
        if mode=='reader':
            if self.expected is None:raise ValueError('reader has no captured operand identity')
            for name,tensor in [('q',q[:,:nt]),('k',k),('v',v)]:
                if not torch.equal(tensor.float().cpu(),torch.from_numpy(self.expected[name])):
                    raise ValueError('native reader operands differ from captured prefix: '+name)
            def replace(tensor,source):
                if source is None:return tensor
                result=torch.empty_strided(tensor.shape,tensor.stride(),dtype=tensor.dtype,device=tensor.device)
                result.copy_(tensor);result[:,nt:]=torch.from_numpy(source[:,nt:]).to(tensor.device,tensor.dtype)
                if not torch.equal(result[:,:nt],tensor[:,:nt]):raise ValueError('port edit changed text operands')
                return result
            k,v=replace(k,self.key_source),replace(v,self.value_source)
            self.footprint=dict(q_text_fp32_sha256=hashlib.sha256(wide(q[:,:nt]).tobytes()).hexdigest(),
                key_after_fp32_sha256=hashlib.sha256(wide(k).tobytes()).hexdigest(),
                value_after_fp32_sha256=hashlib.sha256(wide(v).tobytes()).hexdigest(),
                key_shape=list(k.shape),value_shape=list(v.shape),image_range=[nt,k.shape[1]],
                key_replaced=self.key_source is not None,value_replaced=self.value_source is not None)
        self.counts['native_sdpa']+=1
        result=self.original(q,k,v,*args,**kwargs)
        if mode=='reader':self.fields['head']=wide(result[:,:nt].flatten(2,3))
        return result


def run_campaign(*,model,adapter,commits,projects,weights,output,config,request,decode,event=lambda *a,**k:None):
    import diffusers.models.transformers.transformer_flux2 as native
    import numpy as np
    import torch
    from saturn_pub import Act, Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, load_forecast_tree, verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator
    output.mkdir(parents=True,exist_ok=False);save,read=H['save'],H['read']
    store=LocalStore(output/'store')
    counts=dict(prefix_stops=0,mathematical_text_predictions=0,native_transitions=0,
        completed_blocks=0,block_entries=0,native_sdpa=0,latent_acts=0,images=0,encodes=0,noise_draws=0)
    active=dict(mode='setup',text_rows=next(iter(projects.values())).payload['text'].shape[1])
    handles=[]
    def entry(m,a):
        if active['mode'] not in ('prefix','closure','reader','continuation'):raise ValueError('native block outside allowed window')
        counts['block_entries']+=1
    def complete(m,a,v):counts['completed_blocks']+=1
    for block in [*model.transformer_blocks,*model.single_transformer_blocks]:
        handles.extend([block.register_forward_pre_hook(entry),block.register_forward_hook(complete)])
    block=model.transformer_blocks[0]
    def observe_after(m,a):
        if active['mode']=='reader':boundary.fields['after_attention']=wide(a[0])
    def observe_mlp(m,a,v):
        if active['mode']=='reader':boundary.fields['mlp']=wide(v)
    handles.extend([block.norm2_context.register_forward_pre_hook(observe_after),block.ff_context.register_forward_hook(observe_mlp)])
    guard=torch.nn.modules.module.register_module_forward_pre_hook(
        lambda m,a:(_ for _ in ()).throw(ValueError('native module during equation prediction')) if active['mode']=='forecast' else None)
    def receipt(value,arm,kind='native'):
        store.receipt(value)
        with (output/'receipts.jsonl').open('a') as f:f.write(json.dumps(dict(arm=arm,kind=kind,receipt=value.to_dict()),sort_keys=True)+'\n')
    def retain(arm,local,session):
        cut=session.capture(retain=False);p=cut.payload
        fields={'image':p['image'],'text':p['text']} if local<=25 else ({'noise':p['noise_prediction']} if local==26 else {'latent':p['latent']})
        row=dict(local=local,point=dict(cut.execution_point),cut=cut.fingerprint,
            outputs={k:dict(descriptor=describe(v),archive=H['array'](output,v) if local in (0,1,5,27) else None) for k,v in fields.items()})
        if local in (0,1,27):store.save(cut)
        return row
    prefixes={};transactions={};predictions={};seals={};closure={};readers={}
    try:
        for receiver in ('left','right'):
            b,d=commits[receiver+'.base'],commits[receiver+'.delta']
            differences=[k for k in b.payload if describe(b.payload[k])!=describe(d.payload[k])]
            if differences!=['latent']:raise ValueError('latent is not the sole differing live commit field')
            if describe(projects[receiver+'.base'].payload['text'])!=describe(projects[receiver+'.delta'].payload['text']):
                raise ValueError('projected text differs')
        save(output/'closure-predictions.json',dict(boundary='observed step3/project',rule='Full live state differs only in latent; latent transplant equals the opposite observed parent before execution.',
            cases={r+'.'+name:r+'.'+target for r in ('left','right') for name,target in
                [('base','base'),('delta','delta'),('restore','base'),('transplant','delta')]},counts=counts.copy()))
        with ReaderBoundary(native,counts,active) as boundary:
            for key,cut in projects.items():
                store.save(cut);nav=NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=1)
                active['mode']='prefix';boundary.captured=None
                try:nav.step()
                except PrefixStop:pass
                else:raise ValueError('attention prefix unexpectedly completed')
                if boundary.captured is None or describe(nav.session.capture(retain=False).payload)!=describe(cut.payload):
                    raise ValueError('prefix stop mutated source state')
                receipt(nav.session.attempts[-1],key,'expected-prefix-stop')
                prefixes[key]=boundary.captured
                event('prefix_captured',case=key,counts=counts.copy())
            prefix_records={key:{k:H['array'](output,v) for k,v in p.items()} for key,p in prefixes.items()}
            save(output/'prefixes.json',dict(operands=prefix_records,counts=counts.copy(),frontier='actual native positioned text queries and complete positioned keys/raw values; no native attention/MLP returns'))
            for receiver in ('left','right'):
                b,d=prefixes[receiver+'.base'],prefixes[receiver+'.delta'];nt=active['text_rows']
                if not all(np.array_equal(b[k][:,:nt] if k!='q' else b[k],d[k][:,:nt] if k!='q' else d[k]) for k in ('q','k','v')):
                    raise ValueError('unchanged text does not preserve captured text operands')
                for ki,vi in [(0,0),(1,0),(0,1),(1,1)]:
                    key=f'{receiver}.k{ki}v{vi}';k=b['k'].copy();v=b['v'].copy()
                    if ki:k[:,nt:]=d['k'][:,nt:]
                    if vi:v[:,nt:]=d['v'][:,nt:]
                    tx=ForecastTransaction(output/'transactions'/key,model_identity=adapter.model_identity,
                        input_identity=key+':'+projects[receiver+'.base'].fingerprint,source_identity=request['request_sha256'],
                        frontier=dict(kind='architectural',boundary='native positioned QKV before step3/joint0 attention',
                            prefix=prefix_records[receiver+'.base'],donor=prefix_records[receiver+'.delta']),
                        execution_scope='Full text attention and receiving text write from observed native QKV, original text residual/modulation/weights. Later image continuation is observed, not numerically forecast.',
                        budgets=dict(max_native_calls=1,max_bytes=64<<20,max_wall_ms=600000))
                    active['mode']='forecast'
                    def predict(_):
                        p=projects[receiver+'.base'].payload
                        result=predict_text(b['q'],k,v,wide(p['text']),wide(p['context']['txt_mod']),weights,
                            eps=config['layernorm_eps'],query_chunk=config['query_chunk'],projection_chunk=config['projection_chunk'])
                        counts['mathematical_text_predictions']+=1
                        return dict(predictions={name:H['array'](output,value) for name,value in result.items()})
                    tx.predict('text',predict);seals[key]=tx.seal();verify_transaction(output/'transactions'/key,expected_seal_sha256=seals[key]['seal_sha256'])
                    predictions[key]=load_forecast_tree(output/'transactions'/key,read(output/'transactions'/key/'forecasts/text.json')['predictions'])
                    transactions[key]=tx;event('text_prediction_sealed',case=key,counts=counts.copy())
            if counts['native_sdpa'] or counts['completed_blocks'] or counts['mathematical_text_predictions']!=8:
                raise ValueError('forecast-before-attention chronology violated')
            save(output/'cohort.json',dict(predictions=predictions,seals=seals,counts=counts.copy(),all_predictions_before_native_attention=True))
            for receiver in ('left','right'):
                for name,parent_name,zname,target in [('base','base','base','base'),('delta','delta','delta','delta'),
                        ('restore','delta','base','base'),('transplant','base','delta','delta')]:
                    key=receiver+'.'+name;parent=commits[receiver+'.'+parent_name];z=commits[receiver+'.'+zname].payload['latent']
                    nav=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=28);store.save(parent)
                    active['mode']='closure';rec=nav.apply(Act.replace('latent',z.clone(),name=key+'.latent'));receipt(rec,key,'latent-act');counts['latent_acts']+=1
                    entry_cut=nav.session.capture(retain=False)
                    if describe(entry_cut.payload)!=describe(commits[receiver+'.'+target].payload):raise ValueError('declared latent transplant changed other live state')
                    store.save(entry_cut);records=[]
                    for local in range(28):
                        receipt(nav.step(),key);counts['native_transitions']+=1;records.append(retain(key,local,nav.session))
                    image=decode(nav.session,key);counts['images']+=1
                    closure[key]=dict(parent=parent.fingerprint,target=receiver+'.'+target,input_cut=entry_cut.fingerprint,
                        records=records,final_payload=describe(nav.session.capture(retain=False).payload),image=image)
                    save(output/'closure-rows'/(key+'.json'),closure[key])
                    event('closure_complete',case=key,counts=counts.copy())
            for receiver in ('left','right'):
                b,d=prefixes[receiver+'.base'],prefixes[receiver+'.delta']
                for ki,vi in [(0,0),(1,0),(0,1),(1,1)]:
                    key=f'{receiver}.k{ki}v{vi}';active['mode']='reader'
                    boundary.expected=b;boundary.key_source=d['k'] if ki else None;boundary.value_source=d['v'] if vi else None
                    boundary.fields={};boundary.footprint=None
                    nav=NativeCutNavigator(Session.from_cut(adapter,projects[receiver+'.base']),max_steps=27)
                    first={}
                    def execute(_):
                        receipt(nav.step(),key);counts['native_transitions']+=1
                        boundary.fields['text']=wide(nav.session.read('text'))
                        first.update({k:H['array'](output,v) for k,v in boundary.fields.items()})
                        return dict(fields=first,port_intervention=boundary.footprint)
                    transactions[key].execute('joint0',execute)
                    records=[retain(key,1,nav.session)];active['mode']='continuation'
                    for local in range(2,28):
                        receipt(nav.step(),key);counts['native_transitions']+=1;records.append(retain(key,local,nav.session))
                    image=decode(nav.session,key);counts['images']+=1
                    readers[key]=dict(ki=ki,vi=vi,fields=first,port_intervention=boundary.footprint,records=records,image=image)
                    save(output/'reader-rows'/(key+'.json'),readers[key])
                    event('reader_complete',case=key,counts=counts.copy())
    finally:
        guard.remove()
        for handle in handles:handle.remove()
    if counts!={**counts,'prefix_stops':4,'mathematical_text_predictions':8,'native_transitions':440,'completed_blocks':400,'block_entries':404,'native_sdpa':400,'latent_acts':8,'images':16}:
        raise ValueError('physical execution census differs: '+str(counts))
    save(output/'closure.json',closure);save(output/'readers.json',readers)
    return dict(counts=counts,closure_cases=list(closure),reader_cases=list(readers),cohort_sha256=H['sha'](output/'cohort.json'))
