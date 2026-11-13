"""Step1 pulse persistence, carrier handoff and first-joint A/M interventions."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'));save,sha,identity=[u[k] for k in ('save','sha','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staged identities differ')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    output=Path('step1-writers-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
    root=Path(config['input_root']).resolve(strict=True)
    for name,m in config['input_files'].items():
        src,dst=root/name,Path('input-data')/name
        if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=m['bytes'] or sha(src)!=m['sha256']:raise ValueError('Historical CAS differs')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(src.read_bytes())
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.tensor_archive import write_tensor_archive
    helper=runpy.run_path(str(payload/'flux_step1_writers.py'));expected=helper['census']()
    restore=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['restore_slots']
    candidate=runpy.run_path(str(payload/'flux_joint_writes.py'))['joint_candidate']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[]
    def event(kind,**kw):
        row={'kind':kind,'seconds':time.monotonic()-started,'counts':dict(counts),**kw};timeline.append(row);print('MDB_STEP1_WRITERS '+json.dumps(row,sort_keys=True),flush=True)
    def on_receipt(r):
        v=r.to_dict();receipts.append(v);counts['receipts']+=1;counts['native_transitions' if v['operation']=='continue' else 'acts']+=1
        if any(counts[k]>expected[k] for k in ('native_transitions','acts')) or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite step1 budget exceeded')
    def array(name,v):return write_tensor_archive(output/'arrays'/(name+'.npy'),v,root=output)
    def cpu(v):return v.detach().float().cpu().numpy()
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading')
    pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    original=LocalStore(Path('input-data/store'));root_cut=original.load(config['root_cut'],device='cuda')
    programs={k:original.load(v,device='cuda') for k,v in config['program_cuts'].items()};E={k:v.payload['conditioning'] for k,v in programs.items()}
    if root_cut.model_identity!=adapter.model_identity or dict(root_cut.execution)!=dict(adapter.execution):raise ValueError('Source numerical lane differs')
    if root_cut.execution_point['logical_step']!=1 or root_cut.execution_point['phase']!='project':raise ValueError('Step1 project parent required')
    store=LocalStore(output/'store');store.save(root_cut)
    save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory),'program_cuts':config['program_cuts']})
    binding={'root_cut':root_cut.fingerprint,'program_cuts':config['program_cuts'],'cuts':{k:list(v) for k,v in helper['CUTS'].items()},'native_steps_before_seal':0,
        'scope':'Observed same-L1 programs; futurepulse/handoffs unknown. FullsourceA/M observed later; numericalexitforecasts separatelysealed beforemixedvalidations/futures.'}
    save(output/'binding.json',binding)
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=root_cut.fingerprint,source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','binding':binding},
        execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('binding',binding);tx.predict('pulse_and_writers',lambda _:{'predictions':{'hypotheses':config['hypotheses'],'handoffs':config['handoffs'],'write_rows':config['write_rows']}})
        seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            prefixes={};cuts={};pre2={};pre_joint={};captured={};ports={};post_joint={}
            for role in ('base','color','shape','joint'):
                start=len(receipts);n=NativeCutNavigator(Session.from_cut(adapter,root_cut),max_steps=28,on_receipt=on_receipt)
                acts=restore(n,{'conditioning':E[role]},label='prepare.'+role,store=store);cuts[role]={}
                while n.point['logical_step']==1:
                    if role in ('base','shape') and n.point['phase']=='joint' and n.point['index']==0:
                        parent=n.capture();store.save(parent);pre_joint[role]=parent
                        captured[role]=candidate(pipe.transformer.transformer_blocks[0],parent);counts['candidate_blocks']+=1
                    n.step()
                    if role in ('base','shape'):
                        for name,(phase,index) in helper['CUTS'].items():
                            if n.point['logical_step']==1 and n.point['phase']==phase and n.point['index']==index:
                                cut=n.capture();store.save(cut);cuts[role][name]=cut
                                if name=='after_joint0':
                                    post_joint[role]=cut
                                    if not torch.equal(cut.payload['image'],captured[role]['image']) or not torch.equal(cut.payload['text'],captured[role]['text']):raise ValueError('Captured candidate source differs from native')
                pre2[role]=n.capture();store.save(pre2[role]);prefixes[role]={'path':receipts[start:],'actions':acts}
                event('source_step_complete',role=role)
            for name in helper['CUTS']:
                b,s=cuts['base'][name].payload,cuts['shape'][name].payload
                if any(describe(b[k])!=describe(s[k]) for k in b if k not in ('conditioning','image','text')):raise ValueError('Handoff sources differ outside declared carriers')
            for role in ('base','shape'):
                ports[role]={'residual':pre_joint[role].payload['image'],**captured[role]['ports']}
            port_records={role:{k:array(role+'.'+k,v) for k,v in values.items()} for role,values in ports.items()}
            numeric_binding={'parents':{k:v.fingerprint for k,v in pre_joint.items()},'post_joint':{k:v.fingerprint for k,v in post_joint.items()},'ports':port_records,
                'scope':'Observed native R/A/M/gates; sameR/gates, actualtextprograms differ. Sequentialfrozen-write substitution, not independentA/Mformation forecast.'}
            save(output/'numeric-binding.json',numeric_binding)
            predicted=helper['predict_exits']({k:cpu(v) for k,v in ports['base'].items()},{k:cpu(v) for k,v in ports['shape'].items()})
            forecast_records={k:array('predictedI.'+k,torch.from_numpy(v).to(torch.bfloat16)) for k,v in predicted.items()}
            numeric_seal={'binding_sha256':sha(output/'numeric-binding.json'),'predicted_exits':forecast_records,'counts_at_seal':dict(counts),
                'scope':'Allfourcompleteexitfields sealed beforemixedcandidatevalidations or anyRGBfuture; primitiveequation fixed in mainrequest.'}
            save(output/'numeric-seal.json',numeric_seal);event('numeric_sealed',sha256=sha(output/'numeric-seal.json'))
            validations={}
            for bits in ('10','01','11'):
                actual=candidate(pipe.transformer.transformer_blocks[0],pre_joint['base'],attention=ports['shape' if bits[0]=='1' else 'base']['attention'],mlp=ports['shape' if bits[1]=='1' else 'base']['mlp']);counts['candidate_blocks']+=1
                values=torch.from_numpy(predicted[bits]).to('cuda',torch.bfloat16)
                validations[bits]={'native_image':array('validatedI.'+bits,actual['image']),'predicted_exact':bool(torch.equal(values,actual['image'])),
                    'text_unchanged':bool(torch.equal(actual['text'],post_joint['base'].payload['text'])),'scope':'Actualfullblock withfrozenobservedA/M returns; directcall separatelycounted, never nativeNavigation receipt'}
            save(output/'write-validations.json',validations)
            save(output/'cuts.json',{'step1':{role:{k:v.fingerprint for k,v in values.items()} for role,values in cuts.items()},'pre2':{k:v.fingerprint for k,v in pre2.items()}})
            def finish(name,n,*,parent,actions,initial_path=(),reset=True,meta=None,expected_png=None):
                start=len(receipts);trace=[]
                def apply(current,value):return restore(current,{'conditioning':value},label=name+'.reset2',store=store)
                def boundary(current):
                    cut=current.capture();store.save(cut);k=current.point['logical_step']
                    trace.append({'step':k,'cut':cut.fingerprint,'latent':array(name+'.trace'+str(k),cut.payload['latent'])})
                actions.extend(helper['run_reset_suffix'](n,reset_conditioning=E['base'] if reset else None,apply=apply,on_boundary=boundary))
                final=n.capture();store.save(final);png=output/(name+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                row={'id':name,'parent':parent,'actions':actions,'path':[*initial_path,*receipts[start:]],'trace':trace,'final_cut':final.fingerprint,'latent':trace[-1]['latent'],'png':png.name,'png_sha256':sha(png),**(meta or {})}
                if expected_png is not None:row['expected_png']=expected_png;row['png_control_exact']=sha(png)==expected_png
                save(output/'rows'/(name+'.json'),row);rows.append(row);event('image_complete',branch=name)
            for role in ('base','color','shape','joint'):
                n=NativeCutNavigator(Session.from_cut(adapter,pre2[role]),max_steps=56,on_receipt=on_receipt)
                finish('pulse.'+role,n,parent=root_cut.fingerprint,actions=list(prefixes[role]['actions']),initial_path=prefixes[role]['path'],meta={'kind':'pulse','role':role},expected_png=config['historical_pngs']['base'] if role=='base' else None)
            for role in ('shape','joint'):
                n=NativeCutNavigator(Session.from_cut(adapter,pre2[role]),max_steps=56,on_receipt=on_receipt)
                finish('continuous.'+role,n,parent=pre2[role].fingerprint,actions=[],reset=False,meta={'kind':'continuous','role':role},expected_png=config['historical_pngs'][role])
            pulse_shape=next(r['png_sha256'] for r in rows if r['id']=='pulse.shape')
            for site in helper['CUTS']:
                for mode in ('I','T','IT'):
                    parent=cuts['base'][site];source=cuts['shape'][site];name=site+'.'+mode
                    n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=84,on_receipt=on_receipt);start=len(receipts)
                    replacements={k:source.payload[k] for k,bit in (('image','I'),('text','T')) if bit in mode}
                    acts=restore(n,replacements,label=name,store=store);prefix=receipts[start:]
                    finish(name,n,parent=parent.fingerprint,actions=acts,initial_path=prefix,meta={'kind':'handoff','site':site,'mode':mode,'source':source.fingerprint},expected_png=pulse_shape if mode=='IT' else None)
            for bits in ('10','01'):
                for text_role in ('base','shape'):
                    parent=post_joint[text_role];name='law.A'+bits[0]+'M'+bits[1]+'.T'+text_role
                    n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=82,on_receipt=on_receipt);start=len(receipts)
                    acts=restore(n,{'image':torch.from_numpy(predicted[bits]).to('cuda',torch.bfloat16)},label=name,store=store);prefix=receipts[start:]
                    finish(name,n,parent=parent.fingerprint,actions=acts,initial_path=prefix,meta={'kind':'write-law','bits':bits,'text_role':text_role,'prediction':forecast_records[bits]})
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('pulse_handoffs_and_writes',execute)
    if counts!=expected:raise ValueError('Completed physical census differs: '+str(counts))
    save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    summary={'schema':'mdb-step1-writers-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'numeric_seal_sha256':sha(output/'numeric-seal.json'),
        'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']}
    save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
