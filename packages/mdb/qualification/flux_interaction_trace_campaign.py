"""Replay saved Z2 corners, trace operators and capture peak native A/M operands."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'));save,sha,identity=[u[k] for k in ('save','sha','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staging differs')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch);output=Path('interaction-trace-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
    for seed,source in config['sources'].items():
        root=Path(source['input_root']).resolve(strict=True)
        for name,m in source['input_files'].items():
            src,dst=root/name,Path('input-data')/seed/name
            if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=m['bytes'] or sha(src)!=m['sha256']:raise ValueError('Source CAS differs')
            dst.parent.mkdir(parents=True,exist_ok=True)
            with dst.open('xb') as f:f.write(src.read_bytes())
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from mdb.native_navigation import NativeCutNavigator
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.tensor_archive import write_tensor_archive
    H=runpy.run_path(str(payload/'flux_interaction_trace.py'));capture=runpy.run_path(str(payload/'flux_single_contraction.py'))['capture_candidate'];expected=H['census']()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[];traces={};selected={}
    def event(kind,**kw):
        r={'kind':kind,'counts':dict(counts),'seconds':time.monotonic()-started,**kw};timeline.append(r);print('MDB_INTERACTION_TRACE '+json.dumps(r,sort_keys=True),flush=True)
    def on_receipt(r):
        v=r.to_dict();receipts.append(v);counts['receipts']+=1;counts['native_transitions' if v['operation']=='continue' else 'acts']+=1
        if counts['native_transitions']>448 or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite trace budget exceeded')
    def array(name,v):return write_tensor_archive(output/'arrays'/(name+'.npy'),v,root=output)
    def cuts(nav):return {b:n.session.capture(retain=False) for b,n in nav.items()}
    def sparse(point):return point['phase'] in ('project','readout','scheduler') or (point['phase']=='single' and point['index']==0)
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading');pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');store=LocalStore(output/'store');save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory)})
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=identity(config['sources']),source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','capture_scope':'saved-step2-project-corners'},execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('inputs',config['sources']);tx.predict('trace_policy',lambda _:{'predictions':config['hypotheses']});seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            for seed,source in config['sources'].items():
                original=LocalStore(Path('input-data')/seed/'store');roots={b:original.load(source['cuts'][b],device='cuda') for b in H['BITS']}
                if any(c.model_identity!=adapter.model_identity or dict(c.execution)!=dict(adapter.execution) for c in roots.values()):raise ValueError('Numerical lane differs')
                for c in roots.values():store.save(c)
                nav={b:NativeCutNavigator(Session.from_cut(adapter,c),max_steps=56,on_receipt=on_receipt) for b,c in roots.items()};paths={b:[] for b in nav};trace=[];best=None;current=cuts(nav)
                def observe(current,point,previous=None):
                    values={b:c.payload for b,c in current.items()};measures={};vectors={}
                    for slot in H['live_slots'](point):measures[slot],vectors[slot]=H['measure']({b:v[slot] for b,v in values.items()},None if previous is None else previous.get(slot))
                    retained=sparse(point)
                    if retained:
                        for c in current.values():store.save(c)
                    row={'point':point,'cuts':{b:c.fingerprint for b,c in current.items()},'full_cuts_retained':retained,'measurements':measures};return row,vectors
                row,vectors=observe(current,nav['00'].point);trace.append(row)
                for ordinal in range(56):
                    before=current;point=nav['00'].point;prior=vectors
                    for b,n in nav.items():
                        if n.point!=point:raise ValueError('Four-corner clock drift')
                        paths[b].append(n.step().to_dict())
                    current=cuts(nav);after=nav['00'].point;row,vectors=observe(current,after,prior);row['operator']=point;row['ordinal']=ordinal;trace.append(row)
                    if H['candidate'](point):
                        score=row['measurements']['image']['write_interaction_l2']
                        if best is None or score>best['score']:best={'score':score,'point':point,'ordinal':ordinal,'before':before,'after':current}
                if best is None:raise ValueError('No applicable candidate')
                bindings={'point':best['point'],'ordinal':best['ordinal'],'score':best['score'],'before':{},'after':{},'ports':{}}
                for b in H['BITS']:
                    pre,post=best['before'][b],best['after'][b];store.save(pre);store.save(post);bindings['before'][b]=pre.fingerprint;bindings['after'][b]=post.fingerprint
                    block=pipe.transformer.single_transformer_blocks[best['point']['index']];v=capture(block,pre,single_index=best['point']['index'],logical_step=best['point']['logical_step']);counts['candidate_blocks']+=1
                    if not torch.equal(v['image'],post.payload['image']) or not torch.equal(v['text'],post.payload['text']):raise ValueError('Captured native ports differ from replay')
                    bindings['ports'][b]={'concat':array(seed+'.'+b+'.concat',v['ports']['original_concat']),'projection':array(seed+'.'+b+'.projection',v['ports']['projection']),'gate':array(seed+'.'+b+'.gate',v['ports']['gate']),
                        'text_rows':v['text_rows'],'attention_width':v['attention_width'],'image_exact':True,'text_exact':True,'equation_exact':v['equation_exact']}
                    del v
                selected[seed]=bindings;save(output/('selected-'+seed+'.json'),bindings)
                for b,n in nav.items():
                    final=current[b];png=output/(seed+'.'+b+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                    rows.append({'id':seed+'.'+b,'seed':seed,'bits':b,'parent':roots[b].fingerprint,'path':paths[b],'final_cut':final.fingerprint,'png':png.name,'png_sha256':sha(png),'expected_png':source['pngs'][b],'png_control_exact':sha(png)==source['pngs'][b]})
                traces[seed]=trace;save(output/('trace-'+seed+'.json'),trace);event('seed_complete',seed=seed,selected=bindings['point'],score=bindings['score']);del nav,current,best,roots,prior,vectors
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('native_suffixes_and_selected_ports',execute)
    if counts!=expected:raise ValueError('Completed trace census differs: '+str(counts))
    save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline);save(output/'selected.json',selected)
    summary={'schema':'mdb-interaction-trace-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']};save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()});runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
