"""Fixed candidate E, earlier onset, and repeated predictedT through native futures."""
import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'))
    save,sha,identity=[u[k] for k in ('save','sha','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staging differs')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    output=Path('edit-onset-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
    root=Path(config['input_root']).resolve(strict=True)
    for name,meta in config['input_files'].items():
        src,dst=root/name,Path('input-data')/name
        if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=meta['bytes'] or sha(src)!=meta['sha256']:raise ValueError('Input custody differs')
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
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.tensor_archive import load_tensor_archive,write_tensor_archive
    helper=runpy.run_path(str(payload/'flux_edit_onset.py'));expected=helper['census'](config['onsets'])
    restore=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['restore_slots']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[]
    def event(kind,**kw):
        row={'kind':kind,'seconds':time.monotonic()-started,'counts':dict(counts),**kw};timeline.append(row)
        print('MDB_EDIT_ONSET '+json.dumps(row,sort_keys=True),flush=True)
    def on_receipt(r):
        row=r.to_dict();receipts.append(row);counts['receipts']+=1;counts['native_transitions' if row['operation']=='continue' else 'acts']+=1
        if any(counts[k]>expected[k] for k in ('native_transitions','acts')) or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite execution budget exceeded')
    def array(name,v):return write_tensor_archive(output/'arrays'/(name+'.npy'),v,root=output)
    def cpu(v):return v.detach().float().cpu().numpy()
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading')
    pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    original=LocalStore(Path('input-data/store'));initial=original.load(config['initial_cut'],device='cuda');historical=original.load(config['historical_step3'],device='cuda')
    if initial.model_identity!=adapter.model_identity or dict(initial.execution)!=dict(adapter.execution):raise ValueError('Source numeric profile differs')
    if tensor_descriptor(pipe.transformer.context_embedder.weight)!=config['projector_weight_descriptor']:raise ValueError('Projector weight identity differs')
    values={k:torch.from_numpy(load_tensor_archive(Path('input-data'),r)).to('cuda',torch.bfloat16) for k,r in config['conditioning'].items()}
    saved_prediction=torch.from_numpy(load_tensor_archive(Path('input-data'),config['predicted_additive_T'])).to('cuda',torch.bfloat16)
    store=LocalStore(output/'store');store.save(initial);store.save(historical)
    save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory),'versions':{n:importlib.metadata.version(n) for n in ('torch','diffusers','transformers','mdb-runtime','saturn-pub')}})
    binding={'initial_cut':initial.fingerprint,'historical_step3':historical.fingerprint,'conditioning':config['conditioning'],'projector_weight_descriptor':config['projector_weight_descriptor'],
        'predicted_additive_T':array('predicted.additive.T',saved_prediction),'onsets':config['onsets'],'native_steps_before_seal':counts['native_transitions'],
        'scope':'Oldmeasuredencodings and unchangedpreviouslysealedcandidateE/initialT. Newonsetfuturesunknown; no newpromptencodes or noise. PredictedT repeated ateachremainingprojectread.'}
    save(output/'binding.json',binding)
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=initial.fingerprint,source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','binding':binding},
        execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('binding',binding)
        tx.predict('timing_and_repeated_reads',lambda _: {'predictions':{'hypotheses':config['hypotheses'],'onsets':config['onsets'],'unchanged_predicted_additive_T':binding['predicted_additive_T']}})
        seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            frontiers={};base_path_start=len(receipts)
            nav=NativeCutNavigator(Session.from_cut(adapter,initial),max_steps=112,on_receipt=on_receipt)
            def baseline_boundary(n):
                cut=n.capture();store.save(cut);frontiers[str(n.point['logical_step'])]=cut
            helper['run_suffix'](nav,on_boundary=baseline_boundary)
            if describe(frontiers['3'].payload)!=describe(historical.payload):raise ValueError('Historicalstep3payload reconstruction differs')
            final=nav.capture();png=output/'base.png';adapter.decode(nav.session,height=512,width=512).save(png);counts['images']+=1
            if sha(png)!=config['historical_pngs']['base']:raise ValueError('Historical basePNG differs')
            rows.append({'id':'base','onset':0,'role':'base','predicted':False,'actions':[],'path':receipts[base_path_start:],'trace':[{'step':k,'cut':v.fingerprint,'latent':array('base.trace'+k,v.payload['latent'])} for k,v in frontiers.items()],
                'project_reads':[],'final_cut':final.fingerprint,'latent':array('base.latent',final.payload['latent']),'png':png.name,'png_sha256':sha(png)})
            save(output/'frontiers.json',{k:v.fingerprint for k,v in frontiers.items()});event('baseline_complete')
            for onset in config['onsets']:
                for role in (*values,'stacked_additive'):
                    predicted=role=='stacked_additive';source_role='additive' if predicted else role;name='onset'+str(onset)+'.'+role
                    start=len(receipts);n=NativeCutNavigator(Session.from_cut(adapter,frontiers[str(onset)]),max_steps=28*(4-onset),on_receipt=on_receipt)
                    actions=restore(n,{'conditioning':values[source_role]},label=name,store=store);trace=[];reads=[]
                    def boundary(current):
                        cut=current.capture();store.save(cut);k=current.point['logical_step']
                        trace.append({'step':k,'cut':cut.fingerprint,'latent':array(name+'.trace'+str(k),cut.payload['latent'])})
                    def project(current,k,receipt):
                        if source_role=='additive':
                            native=current.session.read('text')
                            reads.append({'step':k,'project_receipt':receipt.to_dict(),'native_text':array(name+'.nativeT'+str(k),native),
                                'native_descriptor':tensor_descriptor(native),'prediction_difference_l2':float(torch.linalg.vector_norm((saved_prediction.float()-native.float()).flatten()).item())})
                        if predicted:
                            actions.extend(restore(current,{'text':saved_prediction},label=name+'.step'+str(k),store=store))
                    helper['run_suffix'](n,on_project=project,on_boundary=boundary)
                    final=n.capture();png=output/(name+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                    rows.append({'id':name,'onset':onset,'role':source_role,'predicted':predicted,'actions':actions,'path':receipts[start:],'trace':trace,'project_reads':reads,
                        'final_cut':final.fingerprint,'latent':trace[-1]['latent'],'png':png.name,'png_sha256':sha(png)})
                    event('image_complete',branch=name)
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('earlier_onset_native_futures',execute)
    if counts!=expected:raise ValueError('Completed physical census differs: '+str(counts))
    save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    summary={'schema':'mdb-edit-onset-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,
        'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']}
    save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
