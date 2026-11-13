"""Single6 RAMT factorial under common later shape conditioning, plus predicted corners."""
import itertools
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
    output=Path('step1-overlap-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
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
    from mdb.forecast_transaction import ForecastTransaction,verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import write_tensor_archive
    H=runpy.run_path(str(payload/'flux_step1_overlap.py'));expected=H['census']()
    reset_helper=runpy.run_path(str(payload/'flux_step1_writers.py'))['run_reset_suffix']
    linear=runpy.run_path(str(payload/'flux_conditioning_program.py'))['predict_projection']
    capture=runpy.run_path(str(payload/'flux_single_contraction.py'))['capture_candidate']
    restore=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['restore_slots']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[]
    def event(kind,**kw):
        row={'kind':kind,'seconds':time.monotonic()-started,'counts':dict(counts),**kw};timeline.append(row);print('MDB_STEP1_OVERLAP '+json.dumps(row,sort_keys=True),flush=True)
    def on_receipt(r):
        v=r.to_dict();receipts.append(v);counts['receipts']+=1;counts['native_transitions' if v['operation']=='continue' else 'acts']+=1
        if any(counts[k]>expected[k] for k in ('native_transitions','acts')) or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite overlap budget exceeded')
    def array(name,v):return write_tensor_archive(output/'arrays'/(name+'.npy'),v,root=output)
    def cpu(v):return v.detach().float().cpu().numpy()
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading');pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');store=LocalStore(output/'store');original=LocalStore(Path('input-data/store'))
    roots={k:original.load(v,device='cuda') for k,v in config['input_cuts'].items()}
    for cut in roots.values():
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution):raise ValueError('Numerical lane differs')
        store.save(cut)
    Eshape=roots['shape'].payload['conditioning'];block=pipe.transformer.single_transformer_blocks[6]
    save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory)})
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=identity(config['input_cuts']),source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','input_cuts':config['input_cuts']},
        execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('inputs',config['input_cuts']);tx.predict('common_future_RAMT',lambda _:{'predictions':config['hypotheses']});seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            parents={};posts={};captures={};paths={}
            for role,root_cut in roots.items():
                start=len(receipts);n=NativeCutNavigator(Session.from_cut(adapter,root_cut),max_steps=7,on_receipt=on_receipt)
                for _ in range(6):n.step()
                parents[role]=n.capture();store.save(parents[role]);captures[role]=capture(block,parents[role],logical_step=1);counts['candidate_blocks']+=1
                n.step();posts[role]=n.capture();store.save(posts[role]);paths[role]=receipts[start:]
                if not torch.equal(posts[role].payload['image'],captures[role]['image']) or not torch.equal(posts[role].payload['text'],captures[role]['text']):raise ValueError('Candidate/native source differs')
            if any(describe(parents['base'].payload[k])!=describe(parents['shape'].payload[k]) for k in parents['base'].payload if k not in ('conditioning','image','text')):raise ValueError('Unmatched source closure')
            t,w=captures['base']['text_rows'],captures['base']['attention_width'];b,s=captures['base']['ports'],captures['shape']['ports']
            if not torch.equal(b['gate'],s['gate']):raise ValueError('Common gate required')
            binding={'parents':{k:v.fingerprint for k,v in parents.items()},'posts':{k:v.fingerprint for k,v in posts.items()},'prefix_paths':paths,'text_rows':t,'attention_width':w,
                'fields':{role:{'R':array(role+'.R',parents[role].payload['image']),'A':array(role+'.A',captures[role]['ports']['original_concat'][:,t:,:w]),'M':array(role+'.M',captures[role]['ports']['original_concat'][:,t:,w:])} for role in roots},
                'gate':array('gate',b['gate']),'weight':array('output.weight',block.attn.to_out.weight),'bias':None if block.attn.to_out.bias is None else array('output.bias',block.attn.to_out.bias),
                'scope':'Observed R/A/M andactualPweights. Source00/11nativeP alreadyobserved bysourcecapture, but not equationinputs; mixed10/01P not yetexecuted. No upstreamA/M formation forecast.'}
            save(output/'binding.json',binding)
            projected_pred={};image_pred={};p_records={};i_records={}
            weight,bias=cpu(block.attn.to_out.weight),None if block.attn.to_out.bias is None else cpu(block.attn.to_out.bias)
            for bits in ('00','10','01','11'):
                a=cpu((s if bits[0]=='1' else b)['original_concat'][:,t:,:w]);m=cpu((s if bits[1]=='1' else b)['original_concat'][:,t:,w:])
                p=linear(np.concatenate((a,m),axis=2),weight,bias);projected_pred[bits]=p;p_records[bits]=array('predictedP.'+bits,torch.from_numpy(p).to(torch.bfloat16))
                for r,role in enumerate(('base','shape')):
                    key=str(r)+bits;value=H['gated_exit'](cpu(parents[role].payload['image']),cpu(b['gate']),p);image_pred[key]=value;i_records[key]=array('predictedI.'+key,torch.from_numpy(value).to(torch.bfloat16))
            numeric={'binding_sha256':sha(output/'binding.json'),'P':p_records,'I':i_records,'counts_at_seal':dict(counts),'scope':'FP64linear/BF16 predictions before4newcachedP evaluations,6mixedfullblocks andallRGB. Native00/11sourceprojections alreadyobserved, not supplied asnumericinputs.'}
            save(output/'numeric-seal.json',numeric);event('numeric_sealed',sha256=sha(output/'numeric-seal.json'))
            projected={};cached={};p_native={}
            for bits in ('00','10','01','11'):
                cat=H['selected_concat'](b['original_concat'],s['original_concat'],bits=bits,text_rows=t,width=w)
                with torch.inference_mode():p=block.attn.to_out(cat)[:,t:];projected[bits]=p;counts['cached_projections']+=1
                p_native[bits]=array('nativeP.'+bits,p)
                for r,role in enumerate(('base','shape')):cached[str(r)+bits]=parents[role].payload['image']+b['gate']*p
            save(output/'native-projections.json',p_native);validations={}
            for r,role in enumerate(('base','shape')):
                own='00' if r==0 else '11';other='shape' if r==0 else 'base'
                for bits in ('00','10','01','11'):
                    key=str(r)+bits;selection=''.join(c for i,c in enumerate('AM') if bits[i]!=own[i])
                    if not selection:actual=captures[role]
                    else:
                        actual=capture(block,parents[role],logical_step=1,source={'parent':parents[other].fingerprint,'concat':captures[other]['ports']['original_concat'],'text_rows':t,'attention_width':w},selection=selection);counts['candidate_blocks']+=1
                    validations[key]={'image':array('validatedI.'+key,actual['image']),'cached_exact':bool(torch.equal(cached[key],actual['image'])),'text_unchanged':bool(torch.equal(actual['text'],captures[role]['text']))}
            save(output/'validations.json',validations)
            def finish(name,n,parent,actions,initial_path,meta,expected_png=None):
                start=len(receipts);trace=[]
                def apply(current,value):return restore(current,{'conditioning':value},label=name+'.shapeE2',store=store)
                def boundary(current):
                    cut=current.capture();store.save(cut);k=current.point['logical_step'];trace.append({'step':k,'cut':cut.fingerprint,'latent':array(name+'.trace'+str(k),cut.payload['latent'])})
                actions.extend(reset_helper(n,reset_conditioning=Eshape,apply=apply,on_boundary=boundary))
                final=n.capture();store.save(final);png=output/(name+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                row={'id':name,'parent':parent,'actions':actions,'path':[*initial_path,*receipts[start:]],'trace':trace,'final_cut':final.fingerprint,'latent':trace[-1]['latent'],'png':png.name,'png_sha256':sha(png),**meta}
                if expected_png:row.update(expected_png=expected_png,png_control_exact=sha(png)==expected_png)
                rows.append(row);save(output/'rows'/(name+'.json'),row);event('image_complete',branch=name)
            for r,a,m,text_role in itertools.product((0,1),(0,1),(0,1),('base','shape')):
                key=str(r)+str(a)+str(m);name='cube.R'+str(r)+'A'+str(a)+'M'+str(m)+'.T'+text_role;parent=posts[text_role]
                n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=71,on_receipt=on_receipt);start=len(receipts)
                actions=restore(n,{'image':cached[key]},label=name,store=store);prefix=receipts[start:]
                control=config['historical_pngs']['weak'] if key=='000' and text_role=='base' else config['historical_pngs']['strong'] if key=='111' and text_role=='shape' else None
                finish(name,n,parent.fingerprint,actions,prefix,{'RAM':key,'text_role':text_role,'kind':'native-cached'},control)
            for key,role in (('000','base'),('111','shape')):
                name='predicted.'+key+'.T'+role;parent=posts[role];n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=71,on_receipt=on_receipt);start=len(receipts)
                actions=restore(n,{'image':torch.from_numpy(image_pred[key]).to('cuda',torch.bfloat16)},label=name,store=store);prefix=receipts[start:]
                finish(name,n,parent.fingerprint,actions,prefix,{'RAM':key,'text_role':role,'kind':'predicted-exit','prediction':i_records[key]})
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('common_future_cube',execute)
    if counts!=expected:raise ValueError('Completed census differs: '+str(counts))
    save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    summary={'schema':'mdb-step1-overlap-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'numeric_seal_sha256':sha(output/'numeric-seal.json'),
        'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']}
    save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
