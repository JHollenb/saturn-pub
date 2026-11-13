"""Fresh-context composition with joint encoding quarantined until prediction seal."""
import gc
import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload = Path.cwd()
    u = runpy.run_path(str(payload / 'flux_predicted_future_campaign.py'))
    save, sha, identity = [u[k] for k in ('save', 'sha', 'identity')]
    config, request = [json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG', 'SATURN_DEBUG_REQUEST')]
    if config != request['config'] or any(sha(p) != h for p, h in request['source_hashes'].items()):
        raise ValueError('Staging differs')
    scratch = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID']
    scratch.mkdir(parents=True, exist_ok=False)
    os.chdir(scratch)
    output = Path('frozen-composition-result'); output.mkdir()
    save(output / 'plan.json', config); save(output / 'request.json', request)
    root = Path(config['input_root']).resolve(strict=True)
    for name, meta in config['input_files'].items():
        src, dst = root / name, Path('input-data') / name
        if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size != meta['bytes'] or sha(src) != meta['sha256']:
            raise ValueError('Historical input custody differs')
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('xb') as f: f.write(src.read_bytes())
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import load_tensor_archive, write_tensor_archive
    helper = runpy.run_path(str(payload / 'flux_frozen_composition.py'))
    projection = runpy.run_path(str(payload / 'flux_conditioning_program.py'))['predict_projection']
    restore = runpy.run_path(str(payload / 'flux_scene_dependencies.py'))['restore_slots']
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    counts = dict(loads=0, encodes=0, noise_draws=0, native_transitions=0, acts=0, receipts=0, images=0)
    receipts, rows, timeline = [], [], []
    def event(kind, **kw):
        row = {'kind': kind, 'seconds': time.monotonic()-started, 'counts': dict(counts), **kw}
        timeline.append(row)
        print('MDB_FROZEN_COMPOSITION ' + json.dumps(row, sort_keys=True), flush=True)
    def on_receipt(r):
        row = r.to_dict(); receipts.append(row); counts['receipts'] += 1
        counts['native_transitions' if row['operation'] == 'continue' else 'acts'] += 1
        if counts['native_transitions'] > 389 or counts['acts'] > 10 or time.monotonic()-started > config['max_wall_seconds']:
            raise RuntimeError('Finite campaign budget exceeded')
    def cpu(v): return v.detach().float().cpu().numpy()
    def array(name, v): return write_tensor_archive(output / 'arrays' / (name+'.npy'), v, root=output)
    model = Path(config['model_path'])
    inventory = {p.relative_to(model).as_posix(): sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory) != config['checkpoint_sha256']: raise ValueError('Checkpoint differs')
    event('loading')
    pipe = Flux2KleinPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16, local_files_only=True)
    counts['loads'] += 1; pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae): component.eval().requires_grad_(False)
    pipe.text_encoder.to('cuda')
    released = {'joint': False, 'project': False}
    original_encode = pipe.encode_prompt
    def encode(role):
        if role == 'joint' and not released['joint']: raise ValueError('New joint encoding quarantined until seal')
        with torch.inference_mode():
            e, ids = original_encode(config['prompts'][role], device=torch.device('cuda'), num_images_per_prompt=1, max_sequence_length=512)
        counts['encodes'] += 1; event('encoded', role=role)
        return e.cpu(), ids.cpu()
    encodings = {role: encode(role) for role in ('base','color','shape')}
    if not all(torch.equal(ids, encodings['base'][1]) for e, ids in encodings.values()): raise ValueError('Source position IDs differ')
    old = {role: load_tensor_archive(Path('input-data'), record) for role, record in config['old_conditioning'].items()}
    values = helper['compose']({k: cpu(v[0]) for k,v in encodings.items()}, old['joint'], old['additive'])
    projector = pipe.transformer.context_embedder
    binding = {'observed_sources': {k: array('E.'+k, v[0]) for k,v in encodings.items()}, 'source_ids': array('source.ids', encodings['base'][1]),
        'old_conditioning': config['old_conditioning'], 'composed': {k: array('E.'+k, torch.from_numpy(v).to(torch.bfloat16)) for k,v in values.items()},
        'weight': array('projector.weight', projector.weight), 'bias': None if projector.bias is None else array('projector.bias', projector.bias),
        'new_joint_encoded': False, 'native_transitions_before_seal': 0, 'clock': {'step':3,'phase':'project'},
        'scope': 'Newjoint E and allnewdenoiser results unavailable. Threeobserved sourceencodings, fixedhistoricalinteraction, actualprojectorweights.'}
    save(output/'binding.json',binding)
    def guard(module,args):
        if not released['project']: raise ValueError('Native projector entered before seal')
    handle = projector.register_forward_pre_hook(guard)
    tx = ForecastTransaction(output/'transaction', model_identity=identity(inventory), input_identity=identity(binding), source_identity=identity(request['source_hashes']),
        frontier=helper['forecast_frontier'](binding), execution_scope='Frozen newcontext Ecomposition and initialT forecast, joint encoding quarantined untilseal; laterbaseL3 observednativehistory.',
        budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    predicted, forecasts, native_fields, projected = {}, {}, {}, {}
    with tx:
        tx.frontier('binding',binding)
        def forecast(_):
            for role,value in values.items():
                p = projection(value,cpu(projector.weight),None if projector.bias is None else cpu(projector.bias))
                predicted[role] = torch.from_numpy(p).to(torch.bfloat16)
                forecasts[role] = array('predictedT.'+role,predicted[role])
            save(output/'forecasts.json',forecasts)
            return {'predictions':{'initial_text':forecasts,'hypotheses':config['hypotheses'],'rules':config['rules']}}
        tx.predict('frozen_rules',forecast); seal=tx.seal(); event('sealed',seal=seal)
        released['joint']=True; released['project']=True; handle.remove()
        def execute(_):
            encodings['joint']=encode('joint')
            save(output/'new-joint.json',{'encoding':array('E.joint',encodings['joint'][0]),'ids':array('joint.ids',encodings['joint'][1]),'observed_after_seal':True})
            if not torch.equal(encodings['joint'][1],encodings['base'][1]): raise ValueError('Joint position IDs differ')
            pipe.text_encoder=None; pipe.tokenizer=None
            gc.collect(); torch.cuda.empty_cache()
            pipe.transformer.to('cuda'); pipe.vae.to('cuda'); pipe.vae.bn.to('cpu')
            adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
            store=LocalStore(output/'store')
            save(output/'source.json',{'model_identity':adapter.model_identity,'checkpoint_sha256':identity(inventory),'execution':dict(adapter.execution),'versions':{n:importlib.metadata.version(n) for n in ('torch','diffusers','transformers','mdb-runtime','saturn-pub')}})
            with torch.inference_mode():
                latent,img_ids=pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,torch.bfloat16,torch.device('cuda'),torch.Generator(device='cuda').manual_seed(config['seed']))
                counts['noise_draws']+=1
                schedule=None if getattr(pipe.scheduler.config,'use_flow_sigmas',False) else np.linspace(1.,.25,4)
                timesteps,_=retrieve_timesteps(pipe.scheduler,4,torch.device('cuda'),sigmas=schedule,mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
            session=adapter.session(latent=latent,conditioning=encodings['base'][0].to('cuda'),img_ids=img_ids,txt_ids=encodings['base'][1].to('cuda'),timesteps=timesteps,sigmas=pipe.scheduler.sigmas.clone())
            nav=NativeCutNavigator(session,max_steps=112,on_receipt=on_receipt)
            initial=nav.capture();store.save(initial);prefix=[]
            while nav.point['logical_step']<3: prefix.append(nav.step().to_dict())
            frontier=nav.capture();store.save(frontier)
            if len(prefix)!=84 or nav.point['phase']!='project': raise ValueError('Base prefix clock differs')
            save(output/'frontier.json',{'initial':initial.fingerprint,'common_project_frontier':frontier.fingerprint,'path':prefix})
            def finish(name,nav,actions,initial_path=()):
                path=list(initial_path)
                while nav.point['logical_step']<4: path.append(nav.step().to_dict())
                final=nav.capture();store.save(final)
                png=output/(name+'.png'); adapter.decode(nav.session,height=512,width=512).save(png);counts['images']+=1
                rows.append({'id':name,'actions':actions,'path':path,'final_cut':final.fingerprint,'latent':array(name+'.latent',final.payload['latent']),'png':png.name,'png_sha256':sha(png)})
                event('image_complete',branch=name)
            finish('plain.base',nav,[])
            all_values={**{k:cpu(v[0]) for k,v in encodings.items()},**values}
            for role,value in all_values.items():
                n=NativeCutNavigator(Session.from_cut(adapter,frontier),max_steps=28,on_receipt=on_receipt)
                actions=restore(n,{'conditioning':torch.from_numpy(value).to('cuda',torch.bfloat16)},label='E.'+role,store=store)
                first=n.step().to_dict();projected[role]=n.capture();store.save(projected[role])
                native_fields[role]=array('nativeT.'+role,projected[role].payload['text'])
                finish('native.'+role,n,actions,[first])
            save(output/'native-fields.json',native_fields)
            for role in values:
                n=NativeCutNavigator(Session.from_cut(adapter,projected[role]),max_steps=27,on_receipt=on_receipt)
                actions=restore(n,{'text':predicted[role].to('cuda')},label='predictedT.'+role,store=store)
                finish('predicted.'+role,n,actions)
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('quarantined_joint_and_native_futures',execute)
    save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    if counts != dict(loads=1,encodes=4,noise_draws=1,native_transitions=389,acts=10,receipts=399,images=11): raise ValueError('Census differs: '+str(counts))
    summary={'schema':'mdb-frozen-composition-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']}
    save(output/'summary.json',summary)
    save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__': worker()
