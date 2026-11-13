"""One native lease: identical image pairs, crossed text/latent, complete tail cuts."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'));save,sha,identity=[u[k] for k in ('save','sha','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staging differs')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch);output=Path('suffix-context-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
    root=Path(config['input_root']).resolve(strict=True)
    for name,m in config['input_files'].items():
        src,dst=root/name,Path('input-data')/name
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
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb_lanes.tensor_archive import write_tensor_archive
    H=runpy.run_path(str(payload/'flux_suffix_context.py'));capture=runpy.run_path(str(payload/'flux_single_contraction.py'))['capture_candidate'];restore=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['restore_slots'];euler=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['predict_euler'];expected=H['census']()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[];ports={};predictions={};candidate={}
    def event(kind,**kw):
        r={'kind':kind,'counts':dict(counts),'seconds':time.monotonic()-started,**kw};timeline.append(r);print('MDB_SUFFIX_CONTEXT '+json.dumps(r,sort_keys=True),flush=True)
    def on_receipt(r):
        v=r.to_dict();receipts.append(v);counts['receipts']+=1;counts['native_transitions' if v['operation']=='continue' else 'acts']+=1
        if counts['native_transitions']>48 or counts['acts']>32 or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite suffix budget exceeded')
    def array(name,v):return write_tensor_archive(output/'arrays'/(name+'.npy'),v,root=output)
    def cpu(v):return v.detach().float().cpu().numpy()
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading');pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');store=LocalStore(output/'store');original=LocalStore(Path('input-data/store'));save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory)})
    cuts={fp:original.load(fp,device='cuda') for fp in set(config['parents'].values())|{fp for pair in config['pairs'].values() for fp in pair.values()}}
    for c in cuts.values():
        if c.model_identity!=adapter.model_identity or dict(c.execution)!=dict(adapter.execution):raise ValueError('Numerical lane differs')
        store.save(c)
    parents={seed:cuts[fp] for seed,fp in config['parents'].items()};H['shared_closure'](parents)
    binding={'parents':config['parents'],'pairs':config['pairs'],'scope':config['scope']};save(output/'binding.json',binding)
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=identity(binding),source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','observed':'Two known native image pairs and native T/Z parents; all hybrid suffixes unexecuted'},execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('binding',binding);tx.predict('frozen_suffix_hypotheses',lambda _: {'predictions':{'hypotheses':config['hypotheses'],'binding_sha256':sha(output/'binding.json'),'counts_at_seal':dict(counts)}});seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            for row in H['cases']():
                name=row['id'];parent=parents[row['latent']];image=cuts[config['pairs'][row['pair']][str(row['bit'])]].payload['image'];text=parents[row['text']].payload['text'];n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=3,on_receipt=on_receipt);start=len(receipts)
                actions=restore(n,{'image':image,'text':text},label=name,store=store);pre=n.session.capture(retain=False);store.save(pre);key=f"{row['pair']}.I{row['bit']}.T{row['text']}"
                if key not in ports:
                    c=capture(pipe.transformer.single_transformer_blocks[19],pre,logical_step=3,single_index=19);counts['candidate_blocks']+=1;t,w=c['text_rows'],c['attention_width'];candidate[key]={'image':c['image'],'text':c['text']}
                    ports[key]={'source_cut':pre.fingerprint,'text_rows':t,'attention_width':w,'A':array(key+'.A',c['ports']['original_concat'][:,t:,:w]),'M':array(key+'.M',c['ports']['original_concat'][:,t:,w:]),'P':array(key+'.P',c['ports']['projection'][:,t:]),'gate':array(key+'.gate',c['ports']['gate']),'equation_exact':c['equation_exact']}
                n.step();post=n.session.capture(retain=False);store.save(post);parity={'image':bool(torch.equal(candidate[key]['image'],post.payload['image'])),'text':bool(torch.equal(candidate[key]['text'],post.payload['text']))}
                n.step();scheduler=n.session.capture(retain=False);store.save(scheduler);dt=float((scheduler.payload['sigmas'][4]-scheduler.payload['sigmas'][3]).item())
                e=ForecastTransaction(output/'euler-transactions'/name,model_identity=adapter.model_identity,input_identity=scheduler.fingerprint,source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','observed':'Native last-block/readout outputs; final scheduler commit and RGB not yet executed'},execution_scope='CompleteEulercommit fromobservedZ/velocity/schedule',budgets={'max_native_calls':1,'max_wall_ms':60000})
                with e:
                    e.frontier('scheduler',{'cut':scheduler.fingerprint,'sigma_delta':dt})
                    def predict(_):
                        value=euler(cpu(scheduler.payload['latent']),cpu(scheduler.payload['noise_prediction']),dt);descriptor=array(name+'.predictedZ4',torch.from_numpy(value).to(torch.bfloat16));predictions[name]={'latent':descriptor,'scheduler_cut':scheduler.fingerprint,'sigma_delta':dt,'counts_at_seal':dict(counts)};return {'predictions':predictions[name]}
                    e.predict('next_latent',predict);euler_seal=e.seal();counts['euler_forecasts']+=1;e.execute('native_scheduler',lambda _:n.step().to_dict())
                final=n.session.capture(retain=False);store.save(final);png=output/(name+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                result={**row,'parent':parent.fingerprint,'actions':actions,'path':receipts[start:],'pre_cut':pre.fingerprint,'post19_cut':post.fingerprint,'scheduler_cut':scheduler.fingerprint,'final_cut':final.fingerprint,'port_key':key,'candidate_native_parity':parity,'euler_seal':euler_seal,'png':png.name,'png_sha256':sha(png)}
                if row['pair']==row['text']==row['latent']:result.update(expected_png=config['historical_pngs'][row['pair']][str(row['bit'])],png_control_exact=sha(png)==config['historical_pngs'][row['pair']][str(row['bit'])])
                rows.append(result);save(output/'rows'/(name+'.json'),result);event('image_complete',branch=name)
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('crossed_suffix_futures',execute)
    if counts!=expected:raise ValueError('Completed suffix census differs: '+str(counts))
    save(output/'ports.json',ports);save(output/'forecasts.json',predictions);save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    summary={'schema':'mdb-suffix-context-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']};save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()});runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
