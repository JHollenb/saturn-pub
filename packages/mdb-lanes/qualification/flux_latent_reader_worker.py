"""One resident unchanged native consumer for the latent/reader campaign."""
import faulthandler
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def worker():
    here=Path.cwd();c=runpy.run_path(str(here/'flux_latent_reader.py'));h=c['H'];read,save,sha=(h[k] for k in ('read','save','sha'))
    cfg,req=[read(os.environ[k]) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if cfg!=req['config'] or any(sha(here/n)!=v for n,v in req['source_hashes'].items()):raise ValueError('worker source/request changed')
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub.store import LocalStore

    from mdb_lanes.flux_final_latent_closure import FluxFinalLatentClosureAdapter
    started=time.monotonic();faulthandler.dump_traceback_later(90,repeat=True)
    source=Path(cfg['source_root']);out=Path(cfg['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'
    if sha(source/'files.json')!=cfg['source_files_sha256']:raise ValueError('source inventory changed')
    for name,record in cfg['input_files'].items():
        if sha(source/name)!=record['sha256'] or (source/name).stat().st_size!=record['bytes']:raise ValueError('source changed: '+name)
    source_store=LocalStore(source/'store')
    for group,phase,index in [('commits','project',None),('projects','joint',0)]:
        for fp in cfg[group].values():
            cut=source_store.load(fp);point=dict(cut.execution_point)
            if cut.fingerprint!=fp or (point['logical_step'],point['phase'],point['index'])!=(3,phase,index):
                raise ValueError('source frontier clock differs')
    bank=h['RecipeBank'](read(here/'parameter-artifacts.json'),cfg['model_path'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    print('MDB_LATENT_READER loading verified original model',flush=True)
    pipe=Flux2KleinPipeline.from_pretrained(cfg['model_path'],torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter=FluxFinalLatentClosureAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    commits={k:source_store.load(fp,device='cuda') for k,fp in cfg['commits'].items()}
    projects={k:source_store.load(fp,device='cuda') for k,fp in cfg['projects'].items()}
    for cut in [*commits.values(),*projects.values()]:
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution):raise ValueError('model/backend identity differs')
    weights={k:bank.load(v) for k,v in cfg['text_parameters'].items()}
    def event(kind,**fields):
        if time.monotonic()-started>cfg['max_wall_seconds']:raise RuntimeError('finite worker budget exceeded')
        row=dict(kind=kind,seconds=time.monotonic()-started,**fields)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_LATENT_READER '+json.dumps(row,sort_keys=True),flush=True)
    def decode(session,key):
        path=out/(key+'.png');adapter.decode(session,height=512,width=512).save(path)
        return dict(path=path.name,sha256=sha(path))
    result=c['run_campaign'](model=pipe.transformer,adapter=adapter,commits=commits,projects=projects,weights=weights,
        output=out,config=cfg,request=req,decode=decode,event=event)
    result['counts']['model_loads']=1
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(here/name,out/name)
    known={}
    for key,records in cfg['calibration'].items():
        known[key]={}
        for name,record in records.items():
            target=out/'known'/record['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/record['path'],target)
            known[key][name]=dict(record,path='known/'+record['path'])
    save(out/'known-calibrations.json',dict(records=known,source_job=cfg['source_job'],scope='Already known native endpoints; excluded from numerical forecast operands.'))
    save(out/'request.json',req);save(out/'plan.json',cfg)
    save(out/'summary.json',dict(schema='mdb-latent-reader-summary-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        mechanics_status='complete',terminal_status='unassessed',seconds=time.monotonic()-started,output_root=str(out.resolve()),
        scope=cfg['scope'],parameter_reads=bank.reads,parameter_cache_peak_bytes=bank.peak_retained,**result))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out.resolve()),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':worker()
