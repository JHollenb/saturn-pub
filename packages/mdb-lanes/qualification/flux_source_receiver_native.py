"""One resident native replay of a pre-sealed source/receiver forecast cohort."""
from __future__ import annotations

import faulthandler
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def worker():
    here=Path.cwd();h=runpy.run_path(str(here/'flux_source_receiver.py'))
    read,save,sha=(h[k] for k in ('read','save','sha'))
    config,request=[read(os.environ[k]) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config!=request['config'] or any(sha(here/n)!=v for n,v in request['source_hashes'].items()):
        raise ValueError('native worker source/request mismatch')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from contextlib import nullcontext

    import diffusers.models.transformers.transformer_flux2 as native
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act, Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.flux_derived_terminal import derive_terminal
    from mdb_lanes.flux_final_latent_closure import FluxFinalLatentClosureAdapter
    from mdb_lanes.forecast_transaction import verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator
    started=time.monotonic();faulthandler.dump_traceback_later(90,repeat=True)
    out=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'
    out.mkdir(parents=True,exist_ok=False)
    for name in ('plan.json','parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(here/name,out/name)
    save(out/'request.json',request)
    source=Path(config['source_root']);forecast=Path(config['forecast_root'])
    if sha(source/'files.json')!=config['source_files_sha256'] or sha(forecast/'files.json')!=config['forecast_files_sha256']:
        raise ValueError('source or forecast inventory mismatch')
    for name,record in config['input_files'].items():
        if sha(source/name)!=record['sha256']:
            raise ValueError('source changed: '+name)
    manifest=read(forecast/'files.json')
    for name,digest in manifest.items():
        if sha(forecast/name)!=digest:
            raise ValueError('sealed forecast artifact changed: '+name)
    cohort=read(forecast/'cohort.json')
    expected={a['id'] for a in h['arms']() if a['kind']!='sham'}
    if (sha(forecast/'cohort.json')!=config['forecast_cohort_sha256']
            or cohort['counts']['native_model_forwards']!=0
            or cohort['counts']['mathematical_transitions']!=300
            or cohort['all_forecasts_before_native'] is not True
            or cohort['sham_reuses']!='right.base'
            or set(cohort['programs'])!=expected or set(cohort['seals'])!=expected
            or any(len(p['records'])!=50 for p in cohort['programs'].values())):
        raise ValueError('cohort chronology mismatch')
    for key,seal in cohort['seals'].items():
        verify_transaction(forecast/'transactions'/key,expected_seal_sha256=seal['seal_sha256'])
    # Bind the consumer's current original checkpoint to the same complete-file
    # identities used by the forecast, without loading exported parameter arrays.
    identity=h['RecipeBank'](read(here/'parameter-artifacts.json'),config['model_path'],max_cache_bytes=1)
    save(out/'checkpoint-verified.json',dict(parameter_manifest_sha256=sha(here/'parameter-artifacts.json'),
        parameter_records=len(identity.records),raw_parameter_bytes=identity.bytes,
        retained_parameter_bytes=identity.retained,source_files=sorted(identity.signatures)))
    original=LocalStore(source/'typed-store')
    for key,fp in config['parents'].items():
        cut=original.load(fp)
        if cut.fingerprint!=fp or dict(cut.execution_point)!=config['point']:
            raise ValueError('native source parent clock/identity differs: '+key)
    del cut
    save(out/'forecast-verified.json',dict(forecast_job=config['forecast_job'],cohort_sha256=config['forecast_cohort_sha256'],
        files_sha256=config['forecast_files_sha256'],files_verified=len(manifest),new_native_calls=0))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    counts=dict(loads=0,native_transitions=0,acts=0,receipts=0,blocks=0,images=0,vae_calls=0,encodes=0,noise_draws=0)
    active=dict(phase='setup',arm=None)
    def event(kind,**kw):
        if time.monotonic()-started>config['max_wall_seconds']:raise RuntimeError('finite native wall budget')
        row=dict(kind=kind,seconds=time.monotonic()-started,counts=counts.copy(),**kw)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_SOURCE_RECEIVER_NATIVE '+json.dumps(row,sort_keys=True),flush=True)
    event('loading')
    pipe=Flux2KleinPipeline.from_pretrained(config['model_path'],torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu');counts['loads']=1
    adapter=FluxFinalLatentClosureAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    store=LocalStore(out/'store')
    parents={k:original.load(fp,device='cuda') for k,fp in config['parents'].items()}
    origins={k:original.load(fp,device='cuda') for k,fp in config['origin_parents'].items()}
    for cut in (*parents.values(),*origins.values()):
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution):
            raise ValueError('native ABI/model differs from saved source')
        store.save(cut)
    def receipt(value,arm):
        store.receipt(value);counts['receipts']+=1
        body=value.to_dict();counts['native_transitions' if body['operation']=='continue' else 'acts']+=body.get('steps',1)
        with (out/'receipts.jsonl').open('a') as f:f.write(json.dumps(dict(arm=arm,receipt=body),sort_keys=True)+'\n')
    def tensor(value,reference):
        v=torch.from_numpy(np.array(value,copy=True)).to(device=reference.device,dtype=reference.dtype)
        result=torch.empty_strided(reference.shape,reference.stride(),device=reference.device,dtype=reference.dtype)
        result.copy_(v)
        if not torch.equal(result.float().cpu(),torch.from_numpy(np.array(value,copy=True))):
            raise ValueError('payload lowering changed BF16 bits')
        return result
    def render(session,key):
        counts['vae_calls']+=1
        image=adapter.decode(session,height=512,width=512);path=out/(key+'.png');image.save(path);counts['images']+=1
        return dict(path=path.name,sha256=sha(path))
    targets={m for m in (*pipe.transformer.transformer_blocks,*pipe.transformer.single_transformer_blocks)}
    def guard(module,args):
        if active['phase']!='native':raise ValueError('native transformer called before prediction images')
        if module in targets:counts['blocks']+=1
    handles=[m.register_forward_pre_hook(guard) for m in pipe.transformer.modules()]
    f=h['helper']('flux_prompt_factorial.py');rows={};predicted_images={}
    joint_observer=h['helper']('flux_native_joint_ports.py')['NativeJointPorts']
    try:
        # Genuine VAE predictions are rendered from the sealed terminal arrays
        # before any new native denoiser transition. The shell is explicitly
        # derived from an observed project0 origin, never an unavailable future.
        for arm in h['arms']():
            if arm['kind']=='sham':continue
            program=cohort['programs'][arm['id']]
            value=h['load_array'](forecast,program['terminal']['archive'])
            origin=origins[arm['receiver']]
            derived=derive_terminal(adapter,origin,predicted_latent=tensor(value,origin.payload['latent']),name=arm['id']+'.forecast')
            save(out/'derivations'/(arm['id']+'.json'),derived.derivation)
            session=Session.from_cut(adapter,derived.scaffold);receipt(session.apply(derived.action),arm['id']+'.predicted')
            predicted_images[arm['id']]=render(session,arm['id']+'.predicted')
            if counts['blocks'] or counts['native_transitions']:raise ValueError('prediction rendering after native target')
            event('prediction_image_saved',arm=arm['id'])
        save(out/'predicted-images.json',dict(images=predicted_images,counts=counts.copy(),cohort_sha256=config['forecast_cohort_sha256']))
        active['phase']='native'
        for arm in h['arms']():
            active['arm']=arm['id'];parent=parents[arm['receiver']]
            nav=NativeCutNavigator(Session.from_cut(adapter,parent),max_steps=50)
            store.save(parent)
            key='right.base' if arm['kind']=='sham' else arm['id']
            program=cohort['programs'][key]
            if arm['kind']!='base':
                new_text=tensor(h['load_array'](forecast,program['input_text']),parent.payload['text'])
                action=Act.replace('text',new_text,name=arm['id']+'.text')
                rec=nav.apply(action);receipt(rec,arm['id'])
                entry=nav.session.capture(retain=False)
                if describe(entry.payload)!=describe(dict(parent.payload,text=new_text)):
                    raise ValueError('text Act changed other state')
                store.save(entry)
            records=[]
            for position,pred in enumerate(program['records']):
                logical,local=pred['logical'],pred['local']
                observer=None
                if logical==2 and local in (6,7):
                    observer=h['SinglePorts'](pipe.transformer.single_transformer_blocks[local-6],native,config['sample_indices'])
                elif logical==3 and local==1:
                    c=dict.fromkeys(('native_blocks','native_attention','native_image_ff','native_text_ff','native_linears','native_sdpa'),0)
                    observer=joint_observer(pipe.transformer.transformer_blocks[0],native,c,[0,31,511,1023],[0,31,255,511])
                    observer.begin()
                with observer if observer is not None else nullcontext():
                    rec=nav.step();receipt(rec,arm['id'])
                cut=nav.session.capture(retain=False);p=cut.payload
                ports={}
                if isinstance(observer,h['SinglePorts']):ports=observer.fields
                elif observer is not None:
                    ports,_=observer.finish(p)
                fields={'image':p['image'],'text':p['text']} if local<=25 else ({'noise':p['noise_prediction']} if local==26 else {'latent':p['latent']})
                values={};errors={}
                for name,value in fields.items():
                    wide=value.detach().float().cpu().numpy()
                    values[name]=f['value_record'](wide,archive=h['array'](out,wide) if h['retained'](logical,local) else None,witness_count=64)
                    errors[name]=dict(witness=f['comparison'](values[name]['witness']['values'],pred['outputs'][name]['witness']['values']))
                    if pred['outputs'][name]['archive']:
                        errors[name]['full']=f['comparison'](wide,h['load_array'](forecast,pred['outputs'][name]['archive']))
                record=dict(logical=logical,local=local,point=dict(cut.execution_point),cut=cut.fingerprint,outputs=values,errors=errors,
                    ports={k:h['array'](out,v) for k,v in ports.items()})
                records.append(record)
                if h['retained'](logical,local):store.save(cut)
                (out/'native').mkdir(exist_ok=True)
                with (out/'native'/(arm['id']+'.jsonl')).open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
                event('native_transition',arm=arm['id'],logical=logical,local=local)
            final=nav.session.capture(retain=False);store.save(final)
            row=dict(arm=arm,parent=parent.fingerprint,final=final.fingerprint,final_payload=describe(final.payload),records=records,image=render(nav.session,arm['id']+'.native'))
            save(out/'rows'/(arm['id']+'.json'),row);rows[arm['id']]=row
    finally:
        for handle in handles:handle.remove()
    if (counts['native_transitions']!=350 or counts['blocks']!=315 or counts['images']!=13
            or counts['acts']!=11 or counts['receipts']!=361):
        raise ValueError('physical census mismatch: '+str(counts))
    save(out/'summary.json',dict(schema='mdb-source-receiver-native-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],
        mechanics_status='complete',terminal_status='unassessed',counts=counts,seconds=time.monotonic()-started,
        output_root=str(out.resolve()),forecast_job=config['forecast_job'],cohort_sha256=config['forecast_cohort_sha256']))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],
        output_root=str(out.resolve()),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':worker()
