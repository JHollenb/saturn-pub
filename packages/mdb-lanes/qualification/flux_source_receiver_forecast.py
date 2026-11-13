"""CPU-only sealed forecasts; native model consumer is a separate finite lease."""
import faulthandler
import gc
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def worker():
    here = Path.cwd()
    h = runpy.run_path(str(here/'flux_source_receiver.py'))
    read, save, sha = (h[k] for k in ('read','save','sha'))
    config, request = [read(os.environ[k]) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if config != request['config'] or any(sha(here/n) != v for n,v in request['source_hashes'].items()):
        raise ValueError('source/request changed')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(config['cpu_threads'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only forecast lane requires hidden CUDA')
    import torch
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, load_forecast_tree, verify_transaction
    from mdb_lanes.tensor_archive import load_tensor_archive
    started = time.monotonic()
    attempts=[]
    def deny_native(module,args):
        attempts.append(type(module).__name__)
        raise ValueError('native module execution forbidden in mathematical forecast lease')
    guard=torch.nn.modules.module.register_module_forward_pre_hook(deny_native)
    faulthandler.dump_traceback_later(90, repeat=True)
    out = Path(config['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    save(out/'request.json', request)
    save(out/'plan.json', config)
    for n in ('parameter-artifacts.json','retrieve_weights.py','source-binding.json'):
        shutil.copyfile(here/n, out/n)
    source = Path(config['source_root'])
    if sha(source/'files.json') != config['source_files_sha256']:
        raise ValueError('original source inventory mismatch')
    for name, record in config['input_files'].items():
        if sha(source/name) != record['sha256'] or (source/name).stat().st_size != record['bytes']:
            raise ValueError('source input changed: '+name)
    binding = read(out/'source-binding.json')
    original, store = LocalStore(source/'typed-store'), LocalStore(out/'store')
    cuts = {key:original.load(fp) for key,fp in config['parents'].items()}
    for key,cut in cuts.items():
        if cut.fingerprint != config['parents'][key] or dict(cut.execution_point) != config['point']:
            raise ValueError('source clock/identity differs')
        store.save(cut)
    def wide(value):
        if isinstance(value, dict):
            return {k:wide(v) for k,v in value.items()}
        if isinstance(value, tuple):
            return tuple(wide(v) for v in value)
        if hasattr(value,'detach'):
            return (value.detach().float() if value.is_floating_point() else value.detach()).numpy().copy()
        return value
    payloads = {key:wide(cut.payload) for key,cut in cuts.items()}
    texts = {key:p['text'] for key,p in payloads.items()}
    bank = h['RecipeBank'](read(out/'parameter-artifacts.json'), config['model_path'],config['max_parameter_cache_bytes'])
    counts = dict(mathematical_transitions=0,native_model_forwards=0,encodes=0,noise_draws=0,images=0)
    def event(kind, **fields):
        if time.monotonic()-started > config['max_wall_seconds']:
            raise RuntimeError('finite forecast wall budget exhausted')
        row = dict(kind=kind, seconds=time.monotonic()-started, counts=counts.copy(), **fields)
        with (out/'timeline.jsonl').open('a') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n')
        print('MDB_SOURCE_RECEIVER '+json.dumps(row,sort_keys=True),flush=True)
    f = h['helper']('flux_prompt_factorial.py')
    programs, seals = {}, {}
    for arm in h['arms']():
        if arm['kind'] == 'sham':
            continue
        p = dict(payloads[arm['receiver']],text=h['text_input'](texts,arm))
        root = out/'transactions'/arm['id']
        tx = ForecastTransaction(root, model_identity=cuts[arm['receiver']].model_identity,
            input_identity=arm['id']+':'+cuts[arm['receiver']].fingerprint, source_identity=request['request_sha256'],
            frontier=dict(kind='architectural',boundary='observed step2/single0',parent=cuts[arm['receiver']].fingerprint),
            execution_scope=config['scope'], budgets=dict(max_native_calls=1,max_bytes=64<<20,max_wall_ms=config['max_wall_seconds']*1000))
        input_text = h['array'](out,p['text'])
        def predict(_):
            records = []
            for logical,local,fields,ports in h['suffix_events'](p,binding,lambda r:load_tensor_archive(source,r),bank,config):
                outputs = {k:f['value_record'](v,archive=h['array'](out,v) if h['retained'](logical,local) else None,
                    witness_count=64) for k,v in fields.items()}
                sampled = {k:h['array'](out,v) for k,v in ports.items()}
                records.append(dict(logical=logical,local=local,outputs=outputs,ports=sampled))
                counts['mathematical_transitions'] += 1
                event('predicted_transition',arm=arm['id'],logical=logical,local=local)
                gc.collect()
            if len(records) != 50:
                raise ValueError('incomplete fifty-transition suffix')
            return dict(predictions=dict(arm=arm,input_text=input_text,records=records,
                terminal=records[-1]['outputs']['latent'],input_payload=describe(cuts[arm['receiver']].payload)))
        tx.predict('program',predict)
        programs[arm['id']] = load_forecast_tree(root,read(root/'forecasts/program.json')['predictions'])
        seals[arm['id']] = tx.seal()
        verify_transaction(root,expected_seal_sha256=seals[arm['id']]['seal_sha256'])
        event('arm_sealed',arm=arm['id'])
    if counts['mathematical_transitions'] != 300:
        raise ValueError('forecast census mismatch')
    save(out/'cohort.json',dict(schema='mdb-source-receiver-cohort-v1',programs=programs,seals=seals,
        sham_reuses='right.base',counts=counts.copy(),all_forecasts_before_native=True))
    summary = dict(schema='mdb-source-receiver-forecast-v1',job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=request['request_sha256'],mechanics_status='complete',terminal_status='unassessed',
        counts=counts,seconds=time.monotonic()-started,cohort_sha256=sha(out/'cohort.json'),
        parameter_bits_bytes=bank.bytes,parameter_reads=bank.reads,parameter_payload_reads=bank.payload_reads,
        parameter_cache_hits=bank.cache_hits,parameter_cache_peak_bytes=bank.peak_retained,
        parameter_cache_limit_bytes=bank.limit,native_module_attempts=attempts,output_root=str(out.resolve()))
    save(out/'summary.json',summary)
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],
        output_root=str(out.resolve()),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()
    guard.remove()


if __name__ == '__main__':
    worker()
