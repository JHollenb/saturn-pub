"""Prospective operand-linearization and predicted-exit native consumer tests."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'));save,sha,identity=[u[k] for k in ('save','sha','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staging differs')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch);output=Path('interaction-writes-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
    root=Path(config['input_root']).resolve(strict=True)
    for name,m in config['input_files'].items():
        src,dst=root/name,Path('input-data')/name
        if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=m['bytes'] or sha(src)!=m['sha256']:raise ValueError('Source CAS differs')
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
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.tensor_archive import write_tensor_archive,load_tensor_archive
    H=runpy.run_path(str(payload/'flux_interaction_writes.py'));capture=runpy.run_path(str(payload/'flux_single_contraction.py'))['capture_candidate'];linear=runpy.run_path(str(payload/'flux_conditioning_program.py'))['predict_projection'];exit_math=runpy.run_path(str(payload/'flux_step1_overlap.py'))['gated_exit'];restore=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['restore_slots'];expected=H['census']({k:v['point'] for k,v in config['selected'].items()})
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[];bindings={};all_numeric={};all_validations={};forecasts={}
    def event(kind,**kw):
        r={'kind':kind,'counts':dict(counts),'seconds':time.monotonic()-started,**kw};timeline.append(r);print('MDB_INTERACTION_WRITES '+json.dumps(r,sort_keys=True),flush=True)
    def on_receipt(r):
        v=r.to_dict();receipts.append(v);counts['receipts']+=1;counts['native_transitions' if v['operation']=='continue' else 'acts']+=1
        if counts['native_transitions']>expected['native_transitions'] or counts['acts']>32 or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite writer budget exceeded')
    def array(name,v):return write_tensor_archive(output/'arrays'/(name+'.npy'),v,root=output)
    def cpu(v):return v.detach().float().cpu().numpy()
    def load(v):return torch.from_numpy(load_tensor_archive(Path('input-data'),v)).to('cuda',torch.bfloat16)
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading');pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');store=LocalStore(output/'store');original=LocalStore(Path('input-data/store'));save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory)})
    prepared={}
    for seed,s in config['selected'].items():
        before={b:original.load(fp,device='cuda') for b,fp in s['before'].items()};after={b:original.load(fp,device='cuda') for b,fp in s['after'].items()}
        for c in (*before.values(),*after.values()):
            if c.model_identity!=adapter.model_identity or dict(c.execution)!=dict(adapter.execution):raise ValueError('Numerical lane differs')
            store.save(c)
        cats={b:load(p['concat']) for b,p in s['ports'].items()};gate=load(s['ports']['11']['gate']);t,w=s['ports']['11']['text_rows'],s['ports']['11']['attention_width'];mix=H['compose'](cats,text_rows=t,width=w);block=pipe.transformer.single_transformer_blocks[s['point']['index']]
        text_linear=(after['10'].payload['text'].float()+after['01'].payload['text'].float()-after['00'].payload['text'].float()).to(torch.bfloat16)
        bindings[seed]={'point':s['point'],'before':s['before'],'after':s['after'],'text_rows':t,'attention_width':w,'gate':array(seed+'.gate',gate),'weight':array(seed+'.weight',block.attn.to_out.weight),'bias':None if block.attn.to_out.bias is None else array(seed+'.bias',block.attn.to_out.bias),'text_linear':array(seed+'.text_linear',text_linear),'mixed_concats':{k:array(seed+'.'+k+'.concat',v) for k,v in mix.items()}}
        prepared[seed]={'before':before,'after':after,'cats':cats,'mix':mix,'gate':gate,'t':t,'block':block,'text_linear':text_linear}
    save(output/'binding.json',bindings)
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=identity(bindings),source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','observed':'Fourfullnative R/A/M/T states,actualweights; mixedprojections/futures unexecuted'},execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('binding',bindings)
        def predict(_):
            for seed,p in prepared.items():
                forecasts[seed]={};records={};b=bindings[seed]
                for case,cat in p['mix'].items():
                    P=linear(cpu(cat[:,p['t']:]),cpu(p['block'].attn.to_out.weight),None if p['block'].attn.to_out.bias is None else cpu(p['block'].attn.to_out.bias));I=exit_math(cpu(p['before']['11'].payload['image']),cpu(p['gate']),P)
                    forecasts[seed][case]={'P':torch.from_numpy(P).to('cuda',torch.bfloat16),'I':torch.from_numpy(I).to('cuda',torch.bfloat16)}
                    records[case]={'P':array(seed+'.'+case+'.predictedP',torch.from_numpy(P).to(torch.bfloat16)),'I':array(seed+'.'+case+'.predictedI',torch.from_numpy(I).to(torch.bfloat16))}
                all_numeric[seed]=records
            save(output/'forecasts.json',all_numeric);return {'predictions':{'fields':all_numeric,'hypotheses':config['hypotheses'],'binding_sha256':sha(output/'binding.json'),'counts_at_seal':dict(counts)}}
        tx.predict('frozen_mixed_writes',predict);seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            for seed,p in prepared.items():
                s=config['selected'][seed];native={};validations={};t=p['t'];gate=p['gate'];R=p['before']['11'].payload['image'];T=p['after']['11'].payload['text']
                for case,cat in {'sham':p['cats']['11'],**p['mix']}.items():
                    with torch.inference_mode():P=p['block'].attn.to_out(cat);I=R+gate*P[:,t:]
                    counts['cached_projections']+=1;native[case]=I;validation={'P':array(seed+'.'+case+'.nativeP',P[:,t:]),'I':array(seed+'.'+case+'.nativeI',I)}
                    if case=='sham':actual=p['after']['11'].payload['image'];unchanged=True
                    else:
                        c=capture(p['block'],p['before']['11'],logical_step=s['point']['logical_step'],single_index=s['point']['index'],selection='AM',source={'parent':identity({'derived_from':s['before'],'recipe':case}),'concat':cat,'text_rows':t,'attention_width':bindings[seed]['attention_width']});counts['candidate_blocks']+=1;actual=c['image'];unchanged=bool(torch.equal(c['text'],T))
                    validation.update(fullblock_image=array(seed+'.'+case+'.fullblockI',actual),native_cached_exact=bool(torch.equal(I,actual)),text_unchanged=unchanged);validations[case]=validation
                all_validations[seed]=validations
                for case in H['CASES']:
                    image=forecasts[seed]['AMlinear']['I'] if case=='predictedAM' else native['AMlinear'] if case=='AMTlinear' else native['sham'] if case=='Tlinear' else native[case]
                    text=p['text_linear'] if case in ('Tlinear','AMTlinear') else T;parent=p['after']['11'];n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=H['remaining'](s['point']),on_receipt=on_receipt);start=len(receipts)
                    actions=restore(n,{'image':image,'text':text},label=seed+'.'+case,store=store);trace=[]
                    while n.point['logical_step']<4:
                        if n.point['phase']=='project':
                            c=n.session.capture(retain=False);store.save(c);trace.append({'point':n.point,'cut':c.fingerprint})
                        n.step()
                    final=n.session.capture(retain=False);store.save(final);trace.append({'point':n.point,'cut':final.fingerprint});png=output/(seed+'.'+case+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                    row={'id':seed+'.'+case,'seed':seed,'case':case,'parent':parent.fingerprint,'actions':actions,'path':receipts[start:],'trace':trace,'final_cut':final.fingerprint,'latent':array(seed+'.'+case+'.latent',final.payload['latent']),'png':png.name,'png_sha256':sha(png)}
                    if case=='sham':row.update(expected_png=config['historical_pngs'][seed],png_control_exact=sha(png)==config['historical_pngs'][seed])
                    rows.append(row);save(output/'rows'/(row['id']+'.json'),row);event('image_complete',branch=row['id'])
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('mixed_native_validations_and_futures',execute)
    if counts!=expected:raise ValueError('Completed writer census differs: '+str(counts))
    save(output/'validations.json',all_validations);save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    summary={'schema':'mdb-interaction-writes-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']};save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()});runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
