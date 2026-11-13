"""One finite CUDA lease: eight known QKV corners, seven numerical lanes."""
import faulthandler
import importlib.metadata
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def worker():
    boot = time.monotonic()
    faulthandler.dump_traceback_later(60, repeat=True)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    core = runpy.run_path('flux_reader_arithmetic.py')
    attention = runpy.run_path('flux_reader_attention.py')
    h = core['H']; read, save, sha = h['read'], h['save'], h['sha']
    req = read(os.environ['SATURN_DEBUG_REQUEST']); c = read(os.environ['SATURN_RUN_CONFIG'])
    if c != req['config'] or any(sha(Path(n)) != v for n, v in req['source_hashes'].items()):
        raise ValueError('payload/config differs')
    import torch
    from saturn_pub.store import LocalStore
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    print('MDB_READER_ATTENTION torch-ready', time.monotonic()-boot, flush=True)
    torch.set_num_threads(4); core['configure_eager_determinism']()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(c['input_root'])
    for n, r in c['inputs'].items():
        if sha(source/n) != r['sha256'] or (source/n).stat().st_size != r['bytes']:
            raise ValueError('source input differs: '+n)
    for name, expected in c['original_source_sha256'].items():
        path = Path(importlib.metadata.distribution('diffusers').locate_file(name))
        if sha(path) != expected:
            raise ValueError('original dispatch/module source differs')
    out = Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    save(out/'plan.json', c); save(out/'request.json', req)
    save(out/'native-contract.json', dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, tf32=False, deterministic=True, inference_mode=True,
        default_bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        bf16_accumulation=getattr(torch.backends.cuda.matmul, 'allow_bf16_accumulation', None),
        fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        math_sdpa_allow_reduced_precision=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        sdpa_enabled={n:getattr(torch.backends.cuda,n)() for n in ('flash_sdp_enabled','math_sdp_enabled','mem_efficient_sdp_enabled','cudnn_sdp_enabled')},
        original_source_sha256=c['original_source_sha256'], original_kernel_selection='not retained by source; compare output archives',
        query_scope='512 actual text queries; extra1024 queries synthetic in full-shape controls, not recovered image queries',
        mask=None, dropout=0., is_causal=False, scale=None, enable_gqa=False))
    bank = h['RecipeBank'](read('parameter-artifacts.json'), c['model_path'])
    weights = {k:bank.load(v) for k,v in c['text_parameters'].items()}
    modules = core['make_modules'](weights, device='cuda', eps=c['eps'])
    del weights
    store = LocalStore(source/'store')
    parents = {r:store.load(fp).payload for r,fp in c['parents'].items()}
    prefixes = read(source/'prefixes.json')['operands']
    targets = read(source/'readers.json'); forecasts = read(source/'cohort.json')['predictions']
    counts = dict(sdpa=0, staged_matmuls=0, staged_softmax=0, linear=0, layernorm=0, swiglu=0,
                  full_models=0, blocks=0, attention_modules=0, images=0)
    allowed = set(modules['ff'].modules()) | {modules['projection'], modules['norm']}
    def guard(module, args):
        if module not in allowed:
            raise ValueError('undeclared module execution')
    handle = torch.nn.modules.module.register_module_forward_pre_hook(guard)
    cases = {}; started = time.monotonic()
    tensor = lambda x:torch.from_numpy(x.copy()).to('cuda', torch.bfloat16)
    try:
        with torch.inference_mode():
            for receiver in ('left', 'right'):
                b,d = [{f:h['load_array'](source,a) for f,a in prefixes[receiver+'.'+n].items()} for n in ('base','delta')]
                parent = parents[receiver]
                text = parent['text'].to('cuda', torch.bfloat16)
                mod = parent['context']['txt_mod'].to('cuda', torch.bfloat16)
                for ki,vi in ((0,0),(1,0),(0,1),(1,1)):
                    key = f'{receiver}.k{ki}v{vi}'
                    k,v = b['k'].copy(), b['v'].copy()
                    if ki:k[:,c['text_rows']:] = d['k'][:,c['text_rows']:]
                    if vi:v[:,c['text_rows']:] = d['v'][:,c['text_rows']:]
                    q,k,v = map(tensor,(b['q'],k,v))
                    row = dict(heads={}, tails={}, profiles={}, errors={}, source_observed=targets[key]['fields'], source_forecast=forecasts[key])
                    for lane in attention['LANES']:
                        # Explicitly forced optional FLASH may be unavailable. Preserve
                        # that per-row instrument outcome; never suppress other arms.
                        try:
                            head, profile = attention['attention'](q,k,v,lane,counts)
                        except RuntimeError as exc:
                            if lane != 'flash-text':raise
                            row['errors'][lane] = dict(instrument_status='not-applicable', error=str(exc)); continue
                        if not bool(torch.isfinite(head).all()):raise ValueError('nonfinite attention output')
                        row['heads'][lane] = h['array'](out,head)
                        row['profiles'][lane] = profile
                        if lane in ('auto-full-zero','math-text'):
                            tail = core['native_tail'](head,text,mod,modules,counts=counts)
                            row['tails'][lane] = {f:h['array'](out,tail[f]) for f in ('after_attention','mlp','text')}
                            del tail
                        del head
                    cases[key] = row; save(out/'rows'/(key+'.json'),row)
                    event = dict(case=key,seconds=time.monotonic()-started,counts=counts.copy())
                    with (out/'timeline.jsonl').open('a') as f:f.write(json.dumps(event,sort_keys=True)+'\n')
                    print('MDB_READER_ATTENTION '+json.dumps(event,sort_keys=True),flush=True)
                    if event['seconds']>c['max_wall_seconds']:raise RuntimeError('finite attention budget exceeded')
    finally:
        handle.remove()
    if counts['linear']!=48 or counts['layernorm']!=16 or counts['swiglu']!=16 or counts['staged_matmuls']!=32 or counts['staged_softmax']!=16:
        raise ValueError('primitive execution census differs')
    save(out/'cases.json',cases)
    for name in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(name,out/name)
    save(out/'summary.json',dict(schema='mdb-reader-attention-summary-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),counts=counts,seconds=time.monotonic()-started,bootstrap_seconds=started-boot,
        parameter_cache_peak_bytes=bank.peak_retained,mechanics_status='complete',terminal_status='unassessed',scope=c['scope']))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),
        files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':worker()
