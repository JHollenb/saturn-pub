"""One finite CUDA lease: source-assisted fixed attention correction × receiver."""
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();u=runpy.run_path(str(payload/'flux_predicted_future_campaign.py'));save,sha,identity=[u[k] for k in ('save','sha','identity')]
    config,request=[json.loads(Path(os.environ[k]).read_text()) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(p)!=h for p,h in request['source_hashes'].items()):raise ValueError('Staging differs')
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch);output=Path('receiver-cross-result');output.mkdir();save(output/'plan.json',config);save(output/'request.json',request)
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
    from mdb_lanes.tensor_archive import write_tensor_archive,load_tensor_archive
    H=runpy.run_path(str(payload/'flux_receiver_cross.py'));old=runpy.run_path(str(payload/'flux_interaction_writes.py'));capture=runpy.run_path(str(payload/'flux_single_contraction.py'))['capture_candidate'];linear=runpy.run_path(str(payload/'flux_conditioning_program.py'))['predict_projection'];exit_math=runpy.run_path(str(payload/'flux_step1_overlap.py'))['gated_exit'];restore=runpy.run_path(str(payload/'flux_scene_dependencies.py'))['restore_slots'];expected=H['census'](config['selected'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.monotonic();counts={k:0 for k in expected};receipts=[];rows=[];timeline=[];bindings={};numeric={};validations={};forecasts={}
    def event(kind,**kw):
        r={'kind':kind,'counts':dict(counts),'seconds':time.monotonic()-started,**kw};timeline.append(r);print('MDB_RECEIVER_CROSS '+json.dumps(r,sort_keys=True),flush=True)
    def on_receipt(r):
        v=r.to_dict();receipts.append(v);counts['receipts']+=1;counts['native_transitions' if v['operation']=='continue' else 'acts']+=1
        if counts['native_transitions']>60 or counts['acts']>40 or time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('Finite cross budget exceeded')
    def array(name,v):return write_tensor_archive(output/'arrays'/(name+'.npy'),v,root=output)
    def cpu(v):return v.detach().float().cpu().numpy()
    def load(v):return torch.from_numpy(load_tensor_archive(Path('input-data'),v)).to('cuda',torch.bfloat16)
    model=Path(config['model_path']);inventory={p.relative_to(model).as_posix():sha(p) for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.json','.txt','.model','.tiktoken'}}
    if identity(inventory)!=config['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    event('loading');pipe=Flux2KleinPipeline.from_pretrained(str(model),torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None);counts['loads']+=1
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');adapter=Flux2KleinAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda');store=LocalStore(output/'store');original=LocalStore(Path('input-data/store'));save(output/'source.json',{'model_identity':adapter.model_identity,'execution':dict(adapter.execution),'checkpoint_sha256':identity(inventory)})
    prepared={};deltas={}
    for seed,s in config['selected'].items():
        before,after=[original.load(s[k]['11'],device='cuda') for k in ('before','after')]
        for c in (before,after):
            if c.model_identity!=adapter.model_identity or dict(c.execution)!=dict(adapter.execution):raise ValueError('Numerical lane differs')
            store.save(c)
        cats={b:load(p['concat']) for b,p in s['ports'].items()};gate=load(s['ports']['11']['gate']);t,w=s['ports']['11']['text_rows'],s['ports']['11']['attention_width'];mix=old['compose'](cats,text_rows=t,width=w)
        deltas[seed]=mix['Alinear'][:,t:,:w].float()-cats['11'][:,t:,:w].float()
        block=pipe.transformer.single_transformer_blocks[s['point']['index']]
        bindings[seed]={'point':s['point'],'before':before.fingerprint,'after':after.fingerprint,'text_rows':t,'attention_width':w,'gate':array(seed+'.gate',gate),'weight':array(seed+'.weight',block.attn.to_out.weight),'bias':None if block.attn.to_out.bias is None else array(seed+'.bias',block.attn.to_out.bias),'delta':array(seed+'.delta',deltas[seed]),'native_concat':array(seed+'.native_concat',cats['11'])}
        prepared[seed]={'before':before,'after':after,'cat':cats['11'],'gate':gate,'t':t,'w':w,'block':block}
        del cats,mix
    save(output/'binding.json',bindings)
    tx=ForecastTransaction(output/'transaction',model_identity=adapter.model_identity,input_identity=identity(bindings),source_identity=identity(request['source_hashes']),frontier={'kind':'architectural','observed':'Two native receiver R/A/M/T/gates and fourcorner A observations; cross projections and futures unexecuted'},execution_scope=config['scope'],budgets={'max_native_calls':1,'max_wall_ms':config['max_wall_seconds']*1000})
    with tx:
        tx.frontier('binding',bindings)
        def predict(_):
            for row in H['cases'](config['selected']):
                if row['donor'] is None or row['predicted']:continue
                name=row['id'];p=prepared[row['receiver']];cat=H['compose'](p['cat'],deltas[row['donor']],text_rows=p['t'],width=p['w'],dose=row['dose'])
                P=linear(cpu(cat[:,p['t']:]),cpu(p['block'].attn.to_out.weight),None if p['block'].attn.to_out.bias is None else cpu(p['block'].attn.to_out.bias));I=exit_math(cpu(p['before'].payload['image']),cpu(p['gate']),P)
                forecasts[name]={'I':torch.from_numpy(I).to('cuda',torch.bfloat16)}
                numeric[name]={'P':array(name+'.predictedP',torch.from_numpy(P).to(torch.bfloat16)),'I':array(name+'.predictedI',forecasts[name]['I']),'concat':array(name+'.concat',cat)}
            save(output/'forecasts.json',numeric);return {'predictions':{'fields':numeric,'hypotheses':config['hypotheses'],'binding_sha256':sha(output/'binding.json'),'counts_at_seal':dict(counts)}}
        tx.predict('fixed_correction_cross',predict);seal=tx.seal();event('sealed',seal=seal)
        def execute(_):
            native={}
            for row in H['cases'](config['selected']):
                p=prepared[row['receiver']];name=row['id'];parent=p['after'];T=parent.payload['text']
                if row['predicted']:
                    target=f"{row['receiver']}.from{row['donor']}.+1";image=forecasts[target]['I'];validation_key=target
                else:
                    cat=p['cat'] if row['donor'] is None else H['compose'](p['cat'],deltas[row['donor']],text_rows=p['t'],width=p['w'],dose=row['dose'])
                    with torch.inference_mode():P=p['block'].attn.to_out(cat);image=p['before'].payload['image']+p['gate']*P[:,p['t']:]
                    counts['cached_projections']+=1;native[name]=image;validation_key=name
                    if row['donor'] is None:actual=parent.payload['image'];unchanged=True
                    else:
                        c=capture(p['block'],p['before'],logical_step=3,single_index=18,selection='A',source={'parent':identity({'derived_from':bindings,'recipe':row}),'concat':cat,'text_rows':p['t'],'attention_width':p['w']});counts['candidate_blocks']+=1;actual=c['image'];unchanged=bool(torch.equal(c['text'],T))
                    validations[name]={'P':array(name+'.nativeP',P[:,p['t']:]),'I':array(name+'.nativeI',image),'fullblock_image':array(name+'.fullblockI',actual),'native_cached_exact':bool(torch.equal(image,actual)),'text_unchanged':unchanged}
                n=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=3,on_receipt=on_receipt);start=len(receipts);actions=restore(n,{'image':image,'text':T},label=name,store=store)
                while n.point['logical_step']<4:n.step()
                final=n.session.capture(retain=False);store.save(final);png=output/(name+'.png');adapter.decode(n.session,height=512,width=512).save(png);counts['images']+=1
                result={**row,'parent':parent.fingerprint,'actions':actions,'path':receipts[start:],'final_cut':final.fingerprint,'latent':array(name+'.latent',final.payload['latent']),'png':png.name,'png_sha256':sha(png),'validation_key':validation_key}
                if row['donor'] is None:result.update(expected_png=config['historical_pngs'][row['receiver']]['sham'],png_control_exact=sha(png)==config['historical_pngs'][row['receiver']]['sham'])
                if row['receiver']==row['donor'] and abs(row['dose'])==1:
                    result['historical_png_exact']=sha(png)==config['historical_pngs'][row['receiver']]['Alinear' if row['dose']==1 else 'Aopposite']
                rows.append(result);save(output/'rows'/(name+'.json'),result);event('image_complete',branch=name)
            return {'counts':dict(counts),'rows':len(rows)}
        tx.execute('cross_native_validations_and_futures',execute)
    if counts!=expected:raise ValueError('Completed cross census differs: '+str(counts))
    save(output/'validations.json',validations);save(output/'rows.json',rows);save(output/'receipts.json',receipts);save(output/'timeline.json',timeline)
    summary={'schema':'mdb-receiver-cross-v1','job_id':os.environ['MRUN_JOB_ID'],'request_sha256':request['request_sha256'],'counts':counts,'seal':seal,'transaction':verify_transaction(tx.root),'worker_seconds':time.monotonic()-started,'output_root':str(Path.cwd()),'mechanics_status':'pending-independent-audit','terminal_status':'unassessed','scope':config['scope']};save(output/'summary.json',summary);save(output/'files.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()});runpy.run_path(str(payload/'flux_family_custody.py'))['emit'](output,summary,compression_level=1)


if __name__=='__main__':worker()
