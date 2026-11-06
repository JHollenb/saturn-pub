"""One model acquisition, sealed reader forecasts, then sixteen native futures."""
import faulthandler
import importlib.metadata
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def worker():
    boot=time.monotonic();faulthandler.dump_traceback_later(60,repeat=True)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    program=runpy.run_path('flux_reader_route_future.py');core=runpy.run_path('flux_reader_arithmetic.py');h=core['H']
    read,save,sha=[h[k] for k in ('read','save','sha')]
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=read(os.environ['SATURN_RUN_CONFIG'])
    if req['config']!=c or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('source/config differs')
    source=Path(c['input_root'])
    if sha(source/'files.json')!=c['source_files_sha256']:raise ValueError('source inventory differs')
    for name,r in c['inputs'].items():
        if sha(source/name)!=r['sha256'] or (source/name).stat().st_size!=r['bytes']:raise ValueError('input differs')
    for name,expected in c['original_source_sha256'].items():
        if sha(Path(importlib.metadata.distribution('diffusers').locate_file(name)))!=expected:raise ValueError('original native source differs')
    # Verify the original immutable checkpoint and named tensor headers without
    # hydrating a second model or retaining exported parameter files.
    bank=h['RecipeBank'](read('parameter-artifacts.json'),c['model_path'])
    import torch
    torch.set_num_threads(4);core['configure_eager_determinism']()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    print('MDB_ROUTE_FUTURE torch-ready',time.monotonic()-boot,flush=True)
    from diffusers import Flux2KleinPipeline
    from saturn_pub.store import LocalStore

    from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter
    print('MDB_ROUTE_FUTURE loading-original-pipeline',time.monotonic()-boot,flush=True)
    pipe=Flux2KleinPipeline.from_pretrained(c['model_path'],torch_dtype=torch.bfloat16,local_files_only=True,text_encoder=None,tokenizer=None)
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer,pipe.vae):component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter=FluxFinalLatentClosureAdapter.from_pipeline(pipe,granularity='operation',residency='resident',device='cuda')
    parents={r:LocalStore(source/'store').load(fp,device='cuda') for r,fp in c['parents'].items()}
    for cut in parents.values():
        point=dict(cut.execution_point)
        if cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution) or (point['logical_step'],point['phase'],point['index'])!=(3,'joint',0):
            raise ValueError('native model/clock identity differs')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'native-contract.json',dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        dtype='bfloat16',physical_batch=1,image_size=[512,512],bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        math_sdpa_allow_reduced_precision=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),tf32=False,deterministic=True,
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        prediction='Native Flash/linear/LayerNorm/SwiGLU primitives; any earlier prefix blocks are declared in expected_prefix_counts and counted separately before all target suffixes seal.',
        consumer='Original full native blocks and VAE; selected first-attention port interventions only.'))
    started=time.monotonic()
    def event(kind,**fields):
        row=dict(kind=kind,seconds=time.monotonic()-started,**fields)
        with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_ROUTE_FUTURE '+json.dumps(row,sort_keys=True),flush=True)
        if row['seconds']>c['max_wall_seconds']:raise RuntimeError('finite native future budget exceeded')
    if c.get('upstream_receivers'):program=runpy.run_path('flux_reader_upstream.py')
    counts=program['run'](pipe=pipe,adapter=adapter,parents=parents,source=source,output=out,config=c,request=req,event=event)
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):shutil.copyfile(name,out/name)
    save(out/'summary.json',dict(schema='mdb-reader-route-future-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),counts=counts,seconds=time.monotonic()-started,bootstrap_seconds=started-boot,
        parameter_source_bytes_verified=bank.bytes,parameter_payload_reads=bank.payload_reads,mechanics_status='complete',terminal_status='unassessed',scope=c['scope']))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':worker()
