"""Trace early I/T formation with typed handoffs, clamps and rescue."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'));save,sha,identity=[u[k] for k in ('save','sha','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staging differs')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    output=Path('shape-formation-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
    root=Path(config['input_root']).resolve(strict=True)
    for name,m in config['input_files'].items():
        src,dst=root/name,Path('input-data')/name
        if src.is_symlink() or not src.resolve().is_relative_to(root) or src.stat().st_size!=m['bytes'] or sha(src)!=m['sha256']:raise ValueError('Historical CAS differs')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as f:f.write(src.read_bytes())
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import write_tensor_archive
    H=runpy.run_path(str(payload/'flux_shape_formation.py'));expected=H['census'](fresh=config.get('fresh_seed') is not None)
    reset=runpy.run_path(str(payload/'flux_step1_writers.py'))['run_reset_suffix'];restore=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['restore_slots']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[]
    def event(kind,**kw):
        row={'kind':kind,'seconds':time.monotonic()-started,'counts':dict(counts),**kw};timeline.append(row);print('MDB_SHAPE_FORMATION '+json.dumps(row,sort_keys=True),flush=True)
    def on_receipt(r):
        row=r.to_dict();receipts.append(row);counts['receipts']+=1;counts['native_transitions' if row['operation']=='continue' else 'acts']+=1
        if any(counts[k]>expected[k] for k in ('native_transitions','acts')) or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite formation budget exceeded')
    def array(name,value):return write_tensor_archive(output/'arrays'/(name+'.npy'),value,root=output)
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading');pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');store=LocalStore(output/'store');original=LocalStore(Path('input-data/store'))
    roots={k:original.load(v,device='cuda') for k,v in config['input_cuts'].items()}
    for cut in roots.values():
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution):raise ValueError('Numerical lane differs')
        store.save(cut)
    Eshape=roots['shape'].payload['conditioning'];save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory)})
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=identity(config['input_cuts']),source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','input_cuts':config['input_cuts']},execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('inputs',config['input_cuts']);tx.predict('formation_roles',lambda _:{'predictions':config['hypotheses']});seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            nonlocal roots
            if config.get('fresh_seed') is not None:
                old_roots=roots;p=roots['base'].payload;start=len(receipts)
                with torch.inference_mode():
                    latent,ids=pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,torch.bfloat16,torch.device('cuda'),torch.Generator(device='cuda').manual_seed(config['fresh_seed']))
                counts['noise_draws']+=1
                if not torch.equal(ids,p['img_ids']):raise ValueError('Fresh source coordinates differ')
                session=adapter.session(latent=latent,conditioning=p['conditioning'],img_ids=ids,txt_ids=p['txt_ids'],timesteps=p['timesteps'],sigmas=p['sigmas'])
                n=NativeCutNavigator(session,max_steps=28,on_receipt=on_receipt);initial=n.capture();store.save(initial)
                for _ in range(28):n.step()
                common=n.capture();store.save(common);base_path=receipts[start:];roots={};preparation={}
                for role in ('base','shape'):
                    n=NativeCutNavigator(Session.from_cut(adapter,common),max_steps=6,on_receipt=on_receipt);start=len(receipts)
                    actions=restore(n,{'conditioning':old_roots[role].payload['conditioning']},label='fresh.'+role,store=store)
                    for _ in range(6):n.step()
                    roots[role]=n.capture();store.save(roots[role]);preparation[role]={'actions':actions,'path':receipts[start:],'cut':roots[role].fingerprint}
                save(output/'fresh-context.json',{'seed':config['fresh_seed'],'initial':initial.fingerprint,'common_step1':common.fingerprint,'base_path':base_path,'preparation':preparation,'scope':'Freshnoise/receivingstate; same observedsourceencodings/prompt/checkpoint; allgenerationaftersemanticseal'})
            sources={};paths={}
            for role,cut in roots.items():
                n=NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=7,on_receipt=on_receipt);sources[role]={};start=len(receipts)
                for i in range(8):
                    c=n.capture();store.save(c);sources[role][i]=c
                    if n.point['phase']!='single' or n.point['index']!=i or n.point['logical_step']!=1:raise ValueError('Source clock differs')
                    if i<7:n.step()
                paths[role]=receipts[start:]
            for i in range(8):
                lo,hi=sources['base'][i],sources['shape'][i]
                if any(describe(v)!=describe(hi.payload[k]) for k,v in lo.payload.items() if k not in ('conditioning','image','text')):raise ValueError('Source closure differs')
            binding={'cuts':{r:{str(i):c.fingerprint for i,c in cuts.items()} for r,cuts in sources.items()},'prefix_paths':paths,'source_visibility':'Nativebase/shape I/T atboundaries0..7 observed afterseal; interventions usecompletecorresponding fields, noindependentformationforecast'};save(output/'binding.json',binding)
            for recipe in H['recipes']():
                name=recipe['id'];parent=sources[recipe['role']][recipe['entry']];n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=78-recipe['entry'],on_receipt=on_receipt);actions=[];early=[];start=len(receipts)
                for i in range(recipe['entry'],8):
                    pre=n.capture();store.save(pre);selected=H['edits'](recipe,i,sources)
                    if selected:actions.extend(restore(n,selected,label=name+'.at'+str(i),store=store))
                    post=n.capture();store.save(post);early.append({'index':i,'before':pre.fingerprint,'after':post.fingerprint,'writes':list(selected)})
                    if i<7:n.step()
                trace=[]
                def apply(current,value):return restore(current,{'conditioning':value},label=name+'.shapeE2',store=store)
                def boundary(current):
                    c=current.capture();store.save(c);k=current.point['logical_step'];trace.append({'step':k,'cut':c.fingerprint,'latent':array(name+'.latent'+str(k),c.payload['latent'])})
                actions.extend(reset(n,reset_conditioning=Eshape,apply=apply,on_boundary=boundary))
                final=n.capture();store.save(final);png=output/(name+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                row={'id':name,'recipe':recipe,'parent':parent.fingerprint,'actions':actions,'early':early,'path':receipts[start:],'trace':trace,'final_cut':final.fingerprint,'latent':trace[-1]['latent'],'png':png.name,'png_sha256':sha(png)}
                expected_png=config['historical_pngs'].get('weak' if name=='reference.base' else 'strong' if name=='reference.shape' or recipe['mode']=='IT' else '')
                if expected_png:row.update(expected_png=expected_png,png_control_exact=sha(png)==expected_png)
                rows.append(row);save(output/'rows'/(name+'.json'),row);event('image_complete',branch=name)
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('native_formation_and_futures',execute)
    if counts!=expected:raise ValueError('Completed census differs: '+str(counts))
    save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    summary={'schema':'mdb-shape-formation-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']}
    save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()});runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
