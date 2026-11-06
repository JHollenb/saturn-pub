"""Matched delivered-BF16 conditioner doses on saved native MDB parents."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ('sham', 'A.half-context', 'A.context', 'A.mid', 'A.full', 'B.half', 'B.full', 'B.attribute-norm')


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def expected_counts():
    return dict(loads=1, encodes=0, noise_draws=0, native_transitions=3584,
                native_blocks=3200, acts=128, receipts=3712, images=64)


def interpolate_bits(base, source, alpha):
    import numpy as np
    if (base.dtype != np.uint16 or source.dtype != np.uint16 or base.shape != source.shape
            or base.ndim != 3 or not math.isfinite(alpha) or alpha < 0):
        raise ValueError('finite positive interpolation and matched BF16 row geometry required')
    if alpha == 0:
        return base.copy()
    if alpha == 1:
        return source.copy()
    h = runpy.run_path(str(HERE / 'flux_binding_generalization_audit.py'))
    b, s = [h['decode_bf16'](x).astype(np.float64) for x in (base, source)]
    return h['round_bf16']((b + alpha * (s-b)).astype(np.float32))


def delta_norm(base, value):
    import numpy as np
    decode = lambda x: (x.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    delta = decode(value) - decode(base)
    if not np.isfinite(delta).all():
        raise ValueError('nonfinite delivered conditioner')
    return float(np.linalg.norm(delta.reshape(-1)))


def match_norm(base, source, target):
    """Fit one amplitude from supplied input bits only; never inspect a future."""
    if not math.isfinite(target) or target < 0:
        raise ValueError('finite nonnegative norm required')
    if target == 0:
        return 0., base.copy(), 0.
    original = delta_norm(base, source)
    if original == 0:
        raise ValueError('zero source direction cannot supply requested nonzero norm')
    low, high = 0., max(1., 2*target/original)
    candidates = []
    for _ in range(48):
        alpha = (low + high)/2
        value = interpolate_bits(base, source, alpha)
        landed = delta_norm(base, value)
        candidates.append((abs(landed-target), alpha, value, landed))
        if landed < target:
            low = alpha
        else:
            high = alpha
    _, alpha, value, landed = min(candidates, key=lambda item: (item[0], item[1]))
    return alpha, value, landed


def dose_panel(base_a, source_a, base_b, source_b):
    b_half = interpolate_bits(base_b, source_b, .5)
    target_half, target_b, target_a = delta_norm(base_b, b_half), delta_norm(base_b, source_b), delta_norm(base_a, source_a)
    ah, vh, nh = match_norm(base_a, source_a, target_half)
    ac, vc, nc = match_norm(base_a, source_a, target_b)
    bh, vb, nb = match_norm(base_b, source_b, target_a)
    mid = (ac+1)/2
    specs = [dict(arm='sham', support='A', alpha=0., value=base_a.copy()),
             dict(arm='A.half-context', support='A', alpha=ah, value=vh, matched_to='B.half', target_norm=target_half),
             dict(arm='A.context', support='A', alpha=ac, value=vc, matched_to='B.full', target_norm=target_b),
             dict(arm='A.mid', support='A', alpha=mid, value=interpolate_bits(base_a, source_a, mid)),
             dict(arm='A.full', support='A', alpha=1., value=source_a.copy()),
             dict(arm='B.half', support='B', alpha=.5, value=b_half),
             dict(arm='B.full', support='B', alpha=1., value=source_b.copy()),
             dict(arm='B.attribute-norm', support='B', alpha=bh, value=vb, matched_to='A.full', target_norm=target_a)]
    for spec in specs:
        base = base_a if spec['support'] == 'A' else base_b
        spec['landed_delta_l2'] = delta_norm(base, spec['value'])
        spec['extrapolation'] = spec['alpha'] > 1
        if 'target_norm' in spec:
            spec['match_rel_error'] = ((spec['landed_delta_l2']-spec['target_norm'])/spec['target_norm']
                                       if spec['target_norm'] else None)
    return specs


def worker():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    h = helpers(); read, save, sha, identity = [h[k] for k in ('read', 'save', 'sha', 'identity')]
    payload = Path.cwd()
    req, c = [read(os.environ[key]) for key in ('SATURN_DEBUG_REQUEST', 'SATURN_RUN_CONFIG')]
    if req['config'] != c or any(sha(payload/n) != v for n,v in req['source_hashes'].items()):
        raise ValueError('request/source binding differs')
    source = Path(c['main_root']); source_inventory = read(source/'files.json')
    if sha(source/'files.json') != c['main_manifest_sha256'] or sha(Path(c['context_root'])/'files.json') != c['context_manifest_sha256']:
        raise ValueError('source inventory changed')
    verified = set()
    def member(name):
        p = source/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(source):
            raise ValueError('unsafe source member')
        if name not in verified:
            if sha(p) != source_inventory[name]:
                raise ValueError('source custody changed: '+name)
            verified.add(name)
        return p
    parents, addresses = read(member('parents.json')), read(member('token-addresses.json'))
    if read(member('request.json'))['request_sha256'] != c['main_request_sha256'] or read(member('plan.json'))['contexts'] != c['contexts']:
        raise ValueError('source contexts/request changed')
    resolver = runpy.run_path(str(HERE/'flux_binding_context_controls.py'))['select_rows']
    selectors = {x['id']:dict(A=addresses['selectors'][x['id']]['changed'],
                             B=resolver(x, addresses['tokens'][x['id']], addresses['selectors'][x['id']])['rows']) for x in c['contexts']}
    if selectors != c['selectors']:
        raise ValueError('frozen selector differs')
    out = Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    save(out/'plan.json',c); save(out/'request.json',req); save(out/'source-parents.json',parents); save(out/'source-token-addresses.json',addresses)
    counts = {k:0 for k in expected_counts()}; active = dict(case='setup',arm='setup',forecast=False); boot=time.monotonic()
    def event(kind, **fields):
        row=dict(kind=kind,seconds=time.monotonic()-boot,counts=counts.copy(),**fields)
        with (out/'timeline.jsonl').open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_MATCHED_DOSE '+json.dumps(row,sort_keys=True),flush=True)
        if row['seconds']>c['max_wall_seconds']:raise RuntimeError('finite matched-dose wall budget')
    event('setup')
    import importlib.metadata
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError('CUDA admission required')
    for n,v in c['original_source_sha256'].items():
        if sha(Path(importlib.metadata.distribution('diffusers').locate_file(n)))!=v:raise ValueError('native source changed')
    runpy.run_path(str(HERE/'flux_source_receiver.py'))['RecipeBank'](read('parameter-artifacts.json'),c['model_path'])
    pipe=Flux2KleinPipeline.from_pretrained(c['model_path'],torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    original,store=LocalStore(source/'store'),LocalStore(out/'store')
    def parent(fp):
        metadata=read(member('store/cuts/'+fp+'.json'))
        for page in metadata['slots'].values():member('store/pages/'+page+'.json')
        for blob in metadata['physical_blobs']:member('store/blobs/'+blob+'.safetensors')
        cut=original.load(fp,device='cuda')
        if cut.model_identity!=adapter.model_identity or cut.execution_point['logical_step']!=2 or cut.execution_point['phase']!='project':
            raise ValueError('source model/clock changed')
        store.save(cut);return cut
    def bits(value):return value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()
    def array(value):
        a=bits(value) if hasattr(value,'detach') else np.asarray(value)
        if a.dtype!=np.uint16:raise ValueError('BF16 raw bits required')
        digest=identity(dict(shape=list(a.shape),bytes=hashlib.sha256(a.tobytes()).hexdigest()))
        path=out/'arrays'/(digest+'.npy')
        if not path.exists():path.parent.mkdir(exist_ok=True);np.save(path,a,allow_pickle=False)
        return dict(path=path.relative_to(out).as_posix(),sha256=sha(path),shape=list(a.shape),dtype='bfloat16',storage_dtype='uint16')
    def from_bits(a):return torch.from_numpy(a.copy()).view(torch.bfloat16).to('cuda')
    def block_entry(module,args):
        if active['forecast']:raise RuntimeError('target block entered forecast')
        counts['native_blocks']+=1
    hooks=[b.register_forward_pre_hook(block_entry) for b in (*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks)]
    receipts=gzip.open(out/'receipts.jsonl.gz','wt',compresslevel=1)
    def receipt(value):
        body=value.to_dict();key='native_transitions' if body['operation']=='continue' else 'acts'
        counts[key]+=body.get('steps',1);counts['receipts']+=1
        if counts[key]>expected_counts()[key]:raise RuntimeError('finite operation budget')
        receipts.write(json.dumps(dict(case=active['case'],arm=active['arm'],receipt=body),sort_keys=True)+'\n')
    def capture(nav):cut=nav.session.capture(retain=False);store.save(cut);return cut
    actions=[]
    def apply(nav,value,label):
        before=capture(nav);r=nav.apply(Act.replace('conditioning',value,name=label));after=capture(nav)
        if describe(after.payload)!=describe({**before.payload,'conditioning':value}):raise ValueError('undeclared C write')
        actions.append(dict(case=active['case'],arm=active['arm'],slot='conditioning',label=label,before=before.fingerprint,after=after.fingerprint,receipt=r.to_dict()))
    predictions,recipes={},{};counts_at_seal=counts.copy()
    tx=ForecastTransaction(out/'transaction',model_identity=adapter.model_identity,input_identity=identity(c),source_identity=identity(req['source_hashes']),
        frontier=dict(kind='architectural',observed='Eight saved primed project2 parents and original contextual conditioning, previous native images known; no new native prefix, encoder or noise.'),
        execution_scope='CUDA BF16 original Linear projections; source-only norm matching before native suffix release. No attention or pixel forecasts.',
        budgets=dict(max_native_calls=1,max_wall_ms=840000))
    def forecast(_):
        active['forecast']=True
        try:
            for context in c['contexts']:
                cid=context['id'];base=parent(parents[cid]['primed']);donor=parent(parents[cid]['source'])
                bc,dc=base.payload['conditioning'],donor.payload['conditioning'];a,b=selectors[cid]['A'],selectors[cid]['B']
                panel=dose_panel(bits(bc[:,a]),bits(dc[:,a]),bits(bc[:,b]),bits(dc[:,b]));recipes[cid]={};predictions[cid]={}
                for spec in panel:
                    name=spec['arm'];ix=selectors[cid][spec['support']];value=spec.pop('value');mixed=bc.clone();mixed[:,ix]=from_bits(value)
                    recipes[cid][name]={**spec,'rows':ix,'value_rows':array(value)}
                    predictions[cid][name]=dict(image=array(pipe.transformer.x_embedder(base.payload['latent'])),text=array(pipe.transformer.context_embedder(mixed)))
                del base,donor,bc,dc,mixed,panel
        finally:active['forecast']=False
        if counts!=counts_at_seal:raise ValueError('native entry before seal')
        return dict(predictions=dict(counts_at_seal=counts_at_seal,projections=predictions,conditioning_recipes=recipes,semantic=c['hypotheses']))
    rows,fields=[],[]
    with torch.inference_mode():
        tx.predict('matched_dose_predictions',forecast);save(out/'source-doses.json',recipes);seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            for context in c['contexts']:
                cid=context['id'];active['case']=cid
                for name in ARMS:
                    active['arm']=name;cut=parent(parents[cid]['primed']);base_c=cut.payload['conditioning'].clone()
                    nav=NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=56,on_receipt=receipt);del cut
                    recipe=recipes[cid][name];value=np.load(out/recipe['value_rows']['path'],allow_pickle=False)
                    marks={}
                    event('branch_started',case=cid,arm=name)
                    for step in range(56):
                        point=nav.point
                        if point['phase']=='project':
                            current=base_c.clone()
                            if point['logical_step']==2:current[:,recipe['rows']]=from_bits(value)
                            apply(nav,current,name+'.C'+str(point['logical_step']));marks['project'+str(point['logical_step'])+'_input']=capture(nav).fingerprint
                        if step==1:
                            entry=capture(nav);marks['joint0_entry']=entry.fingerprint
                            for f in ('image','text'):fields.append(dict(case=cid,arm=name,field=f,prediction=predictions[cid][name][f],actual=array(entry.payload[f])))
                            del entry
                        nav.step()
                    final=capture(nav);png=cid+'.'+name+'.png';adapter.decode(nav.session,height=512,width=512).save(out/png);counts['images']+=1
                    rows.append(dict(case=cid,arm=name,png=png,png_sha256=sha(out/png),source_parent=parents[cid]['primed'],final_cut=final.fingerprint,final_latent=array(final.payload['latent']),marks=marks))
                    event('image_complete',case=cid,arm=name);del nav,base_c,final,current
                save(out/('completed-'+cid+'.json'),[r for r in rows if r['case']==cid])
            return dict(counts=counts.copy(),images=len(rows))
        with tx:tx.execute('native_dose_curves',execute)
    receipts.close()
    for hook in hooks:hook.remove()
    if counts!=expected_counts():raise ValueError('execution census differs: '+str(counts))
    for name,value in (('rows.json',rows),('fields.json',fields),('actions.json',actions)):save(out/name,value)
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json','retrieve_flux_encoder.py'):shutil.copyfile(payload/name,out/name)
    save(out/'native-contract.json',dict(device=torch.cuda.get_device_name(),dtype='bfloat16',batch_size=1,torch=torch.__version__,diffusers=importlib.metadata.version('diffusers'),bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    save(out/'external-source.json',dict(main_root=str(source),main_manifest_sha256=c['main_manifest_sha256'],referenced_files={n:source_inventory[n] for n in sorted(verified)}))
    save(out/'summary.json',dict(schema='mdb-binding-matched-dose-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),counts=counts,counts_at_seal=counts_at_seal,seal=seal,transaction=verify_transaction(tx.root),seconds=time.monotonic()-boot,mechanics_status='pending-independent-audit',terminal_status='unassessed'))
    event('complete');save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,main_run,context_run):
    from mdb.job_support import scheduler_api,one_lease_scheduling_contract
    from mdb.payload import stage_worker
    h=helpers();read,save,sha=[h[k] for k in ('read','save','sha')];old=read(main_run/'request.json');context=read(context_run/'request.json')
    jobs=[read(r/'submission.json')['job_id'] for r in (main_run,context_run)]
    roots=[]
    for prior,jid in zip((old,context),jobs):
        _,raw,_=scheduler_api().request('GET','/api/jobs/'+jid,timeout_s=20)
        if json.loads(raw)['state']!='succeeded':raise ValueError('completed source runs required')
        roots.append(str(Path(prior['config']['output_scratch'])/jid/'binding-minimal-result'))
    code='import pathlib,json,hashlib\nroots='+repr(roots)+'\nout=[]\nfor root in roots:\n p=pathlib.Path(root);out.append(dict(sha=hashlib.sha256((p/"files.json").read_bytes()).hexdigest(),plan=json.loads((p/"plan.json").read_text()),addresses=json.loads((p/"token-addresses.json").read_text()) if (p/"token-addresses.json").exists() else None))\nprint(json.dumps(out))\n'
    meta=json.loads(subprocess.run(['ssh','beast','python3','-I','-'],input=code,text=True,capture_output=True,check=True).stdout)
    resolver=runpy.run_path(str(HERE/'flux_binding_context_controls.py'))['select_rows'];addresses=meta[0]['addresses'];contexts=old['config']['contexts']
    selectors={x['id']:dict(A=addresses['selectors'][x['id']]['changed'],B=resolver(x,addresses['tokens'][x['id']],addresses['selectors'][x['id']])['rows']) for x in contexts}
    c=dict(schema='mdb-binding-matched-dose-plan-v1',main_root=roots[0],context_root=roots[1],main_manifest_sha256=meta[0]['sha'],context_manifest_sha256=meta[1]['sha'],main_request_sha256=old['request_sha256'],contexts=contexts,selectors=selectors,arms=list(ARMS),model_path=old['config']['model_path'],original_source_sha256=old['config']['original_source_sha256'],output_scratch='/mnt/big/mdb/research',directory_custody=True,max_wall_seconds=840,
        interpolation='Endpoint0/1 original BF16 bits. Other rows: float64(base+alpha*(source-base)), cast float32, explicit ties-to-even BF16. Norms measured on actual delivered BF16 delta; amplitude solved solely from supplied C bits.',
        hypotheses=dict(stage='Frozen exploratory predictions before any new native execution; previous natural/full-dose controls explicitly known.',
            primary_cases=['anchor','seed','wording','objects'],primary_rank='For each primary case, signed RGB source-axis increment of A.context exceeds B.full under matched delivered norm; per-case misses retained.',
            high_context='B.attribute-norm is predicted to retain base assignment or show collateral, rather than a clean source assignment; all source swaps refute this expectation.',
            low_attribute='A.half-context/A.context are predicted to retain base assignment; A.mid may be mixed. Numerical and semantic forecasts have separate scopes.',
            replay='sham/A.full/B.full exactly replay prior I/IC.early/context-controls final payloads and PNGs.',
            alternatives=['address-compatible direction','amplitude/threshold explanation','nonlinear direction-by-receiver interaction','extrapolation collateral'],
            caveat='B.attribute-norm usually extrapolates beyond the natural donor; equal L2 norm is not equal semantic information. No held-out image forecast.'))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    extras=[(HERE/n,n) for n in ('flux_binding_generalization.py','flux_binding_context_controls.py','flux_binding_generalization_audit.py','flux_source_receiver.py','flux_family_custody.py','retrieve_flux_encoder.py')]
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):shutil.copyfile(main_run/'payload'/n,run/n);extras.append((run/n,n))
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[*roots,c['model_path'],c['output_scratch'],'binding-minimal-result']));extras.append((pre,pre.name))
    req=dict(schema='mdb-binding-matched-dose-request-v1',worker_script=Path(__file__).name,config=c,model=old['model'],execution_environment=old['execution_environment'],needs=dict(cuda=True,host='beast'),task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},
        resource_exception=dict(ram_mb=10240,vram_mb=10240,cpu_threads=4,disk_gb=4,est_wall_s=420,calibration_job=jobs[1],measured_peak_rss_mb=2104.2,measured_peak_vram_mb=9238.0,reason='Same original resident model/saved-parent56op suffix;8images measured61.52s/2104.2MBRSS/9238MBVRAM. Current64images/3584ops/3200blocks/128Acts, no encoder/noise/prefix; sequential branches, original BF16 arrays/CAS, no duplicated weight bank.10GiBRAM/VRAM,4GBdisk/420sestimate/840sworker/900souter;0retries.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=192,logical_branches=64,logical_checkpoint_replays=64,max_local_work_items=256,checkpoint_policy='One finite original CUDA B1 lease;8source-only norm panels,128projectedfields sealed before64suffixes/3584ops/3200blocks/128CActs/64VAEs;0encodes/noise/prefix/retries.'))
    req['request_sha256']=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',req)
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment']);stage_worker(run/'payload',worker=Path(__file__),config=c,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'],counts=expected_counts())


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    h=helpers();read,save,sha=[h[k] for k in ('read','save','sha')];req=read(run/'request.json');runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in req['source_hashes'].items()):raise ValueError('changed/already submitted payload')
    command=worker_command(req['worker_script']);command.insert(command.index('python')+1,'-B')
    resources={k:req['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job=guarded_mrun_launch(command,request=req,model=req['model'],task='forward',experiment='mdb-binding-matched-dose',payload=run/'payload',gpu=True,needs=req['needs'],prefer_host='beast',timeout_s=900,preflight=True,retry_on_kill=False,**resources,
        note='Matched delivered-BF16 dose/address assay:8contexts×8arms=64images/3584navops/3200blocks/128CActs;128projectionforecasts globally sealed first. Three known replays/context. No encoder/noise/prefix; source-only amplitude solve, native suffix/VAE;Bhigh extrapolation declared. One CUDA B1 lease,420sestimate/840sworker/900souter,0retries.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--main-run',type=Path);p.add_argument('--context-run',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.main_run.resolve(),a.context_run.resolve()),sort_keys=True))
