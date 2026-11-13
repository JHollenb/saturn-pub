"""Saved-parent contextual-row controls: eight images, no encodes or noise draws.

Stage: python flux_binding_context_controls.py RUN --main-run COMPLETED_MAIN_RUN
Submit: python flux_binding_context_controls.py RUN --submit
The finite native worker runs only inside the separately admitted mrun lease.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = 'IC.context-controls'


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def select_rows(context, token_record, previous_selector):
    """First two unchanged, valid prompt-content rows after all changed IDs.

    Recorded absolute attribute spans recover the raw prompt's character origin;
    no new tokenizer, encoder or image-based choice is involved.
    """
    base, source = token_record['base'], token_record['source']
    prompt = context['prompts']['base']
    origins = {span['start'] - prompt.index(span['attribute']) for span in base['attribute_spans']}
    if len(origins) != 1:
        raise ValueError('recorded attribute spans disagree on prompt origin')
    start = origins.pop()
    end = start + len(prompt)
    source_prompt = context['prompts']['source']
    source_origins = {span['start'] - source_prompt.index(span['attribute']) for span in source['attribute_spans']}
    if len(source_origins) != 1:
        raise ValueError('source attribute spans disagree on prompt origin')
    source_start = source_origins.pop()
    source_end = source_start + len(source_prompt)
    changed = [i for i, (a, b) in enumerate(zip(base['ids'], source['ids'], strict=True)) if a != b]
    if changed != previous_selector['changed'] or not changed:
        raise ValueError('paired source changed-ID inventory differs')
    last = max(changed)
    candidates = [i for i, (a, b) in enumerate(zip(base['ids'], source['ids'], strict=True))
                  if i > last and a == b and base['mask'][i] and source['mask'][i]
                  and start <= base['offsets'][i][0] < base['offsets'][i][1] <= end
                  and source_start <= source['offsets'][i][0] < source['offsets'][i][1] <= source_end
                  and not (base['pieces'][i].startswith('<|') and base['pieces'][i].endswith('|>'))]
    if len(candidates) < 2:
        raise ValueError('fewer than two valid unchanged prompt-content rows after changed support')
    chosen = candidates[:2]
    return dict(rows=chosen, candidates=candidates, last_changed_id_row=last,
                raw_prompt_character_interval=[start, end], policy='first-two; fixed from source metadata before new execution',
                token_ids=[base['ids'][i] for i in chosen], pieces=[base['pieces'][i] for i in chosen],
                offsets=[base['offsets'][i] for i in chosen],
                limitation='Unchanged IDs are contextual states, not pure attribute vectors or guaranteed semantic nulls.')


def expected_counts():
    return dict(loads=1, encodes=0, noise_draws=0, native_transitions=448,
                native_blocks=400, acts=16, receipts=464, images=8)


def worker():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    h = helpers()
    read, save, sha, identity = [h[k] for k in ('read', 'save', 'sha', 'identity')]
    payload = Path.cwd()
    request, config = [read(os.environ[key]) for key in ('SATURN_DEBUG_REQUEST', 'SATURN_RUN_CONFIG')]
    if request['config'] != config or any(sha(payload / name) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('worker source/request binding differs')
    source = Path(config['source_root'])
    if sha(source / 'files.json') != config['source_manifest_sha256']:
        raise ValueError('completed source inventory changed')
    inventory = read(source / 'files.json')
    verified = set()
    def source_file(name):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe source member')
        path = source / relative
        if name not in verified:
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or sha(path) != inventory[name]:
                raise ValueError('source member custody differs: ' + name)
            verified.add(name)
        return path
    plan = read(source_file('plan.json'))
    parents = read(source_file('parents.json'))
    addresses = read(source_file('token-addresses.json'))
    if config['contexts'] != plan['contexts'] or config['source_request_sha256'] != read(source_file('request.json'))['request_sha256']:
        raise ValueError('source context/request identity differs')
    selections = {c['id']: select_rows(c, addresses['tokens'][c['id']], addresses['selectors'][c['id']]) for c in config['contexts']}
    if selections != config['selections']:
        raise ValueError('frozen contextual row selection changed')
    out = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', config)
    save(out / 'request.json', request)
    save(out / 'source-plan.json', plan)
    save(out / 'source-token-addresses.json', addresses)
    save(out / 'source-parents.json', parents)
    counts = {key: 0 for key in expected_counts()}
    active = dict(case='setup', forecast=False)
    started = time.monotonic()
    def event(kind, **items):
        value = dict(kind=kind, seconds=time.monotonic()-started, counts=counts.copy(), **items)
        with (out / 'timeline.jsonl').open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')
        print('MDB_BINDING_CONTEXT_CONTROL ' + json.dumps(value, sort_keys=True), flush=True)
        if value['seconds'] > config['max_wall_seconds']:
            raise RuntimeError('finite context-control wall budget')
    event('addresses_frozen', selections=selections)
    import importlib.metadata
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError('CUDA admission required')
    for name, digest in config['original_source_sha256'].items():
        if sha(Path(importlib.metadata.distribution('diffusers').locate_file(name))) != digest:
            raise ValueError('original source differs: ' + name)
    runpy.run_path(str(payload / 'flux_source_receiver.py'))['RecipeBank'](read(payload / 'parameter-artifacts.json'), config['model_path'])
    pipe = Flux2KleinPipeline.from_pretrained(config['model_path'], torch_dtype=torch.bfloat16,
                                             local_files_only=True, text_encoder=None, tokenizer=None)
    counts['loads'] += 1
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity='operation', residency='resident', device='cuda')
    original, store = LocalStore(source / 'store'), LocalStore(out / 'store')
    def load_source(fingerprint):
        cut_meta = read(source_file('store/cuts/' + fingerprint + '.json'))
        for page in cut_meta['slots'].values():
            source_file('store/pages/' + page + '.json')
        for blob in cut_meta['physical_blobs']:
            source_file('store/blobs/' + blob + '.safetensors')
        cut = original.load(fingerprint, device='cuda')
        if cut.fingerprint != fingerprint or cut.model_identity != adapter.model_identity:
            raise ValueError('source cut/model identity differs')
        point = cut.execution_point
        if point['logical_step'] != 2 or point['phase'] != 'project':
            raise ValueError('source project clock differs')
        store.save(cut)
        return cut
    def array(value):
        value = value.detach().contiguous().cpu()
        if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError('finite BF16 array required')
        bits = value.view(torch.uint16).numpy()
        digest = identity(dict(shape=list(bits.shape), dtype='uint16', bytes_sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
        path = out / 'arrays' / (digest + '.npy')
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            np.save(path, bits, allow_pickle=False)
        return dict(path=path.relative_to(out).as_posix(), sha256=sha(path), shape=list(bits.shape), dtype='bfloat16', storage_dtype='uint16')
    def block_entry(module, args):
        if active['forecast']:
            raise RuntimeError('native target block entered during projection forecast')
        counts['native_blocks'] += 1
    hooks = [block.register_forward_pre_hook(block_entry) for block in
             (*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks)]
    receipts = gzip.open(out / 'receipts.jsonl.gz', 'wt', compresslevel=1)
    def receipt(value):
        body = value.to_dict()
        key = 'native_transitions' if body['operation'] == 'continue' else 'acts'
        counts[key] += body.get('steps', 1)
        counts['receipts'] += 1
        if counts[key] > expected_counts()[key]:
            raise RuntimeError('finite operation budget exceeded')
        receipts.write(json.dumps(dict(case=active['case'], arm=ARM, receipt=body), sort_keys=True) + '\n')
    def capture(nav):
        cut = nav.session.capture(retain=False)
        store.save(cut)
        return cut
    actions = []
    def apply(nav, value, label):
        before = capture(nav)
        act = Act.replace('conditioning', value, name=label)
        r = nav.apply(act)
        after = capture(nav)
        if describe(after.payload) != describe({**before.payload, 'conditioning': value}):
            raise ValueError('conditioning intervention changes another slot')
        actions.append(dict(case=active['case'], slot='conditioning', label=label,
                            before=before.fingerprint, after=after.fingerprint, receipt=r.to_dict()))
    predictions, diagnostics = {}, {}
    before_forecast = counts.copy()
    tx = ForecastTransaction(out / 'transaction', model_identity=adapter.model_identity,
        input_identity=identity(config), source_identity=identity(request['source_hashes']),
        frontier=dict(kind='architectural', observed='Eight completed main-run primed project2 parents and contextual base/source conditioner rows; no new encoder/noise/prefix.'),
        execution_scope='Original CUDA BF16 Linear projections; native56-step suffixes and VAE observed separately.',
        budgets=dict(max_native_calls=1, max_wall_ms=540000))
    def forecast(_):
        active['forecast'] = True
        try:
            for context in config['contexts']:
                cid = context['id']
                base = load_source(parents[cid]['primed'])
                donor = load_source(parents[cid]['source'])
                bc, dc = base.payload['conditioning'], donor.payload['conditioning']
                indices = selections[cid]['rows']
                mixed = bc.clone()
                mixed[:, indices] = dc[:, indices]
                b = bc[:, indices].detach().contiguous().cpu().view(torch.uint16).numpy()
                d = dc[:, indices].detach().contiguous().cpu().view(torch.uint16).numpy()
                changed = [int(np.count_nonzero(b[:, j] != d[:, j])) for j in range(len(indices))]
                diagnostics[cid] = dict(source_parent=base.fingerprint, source_conditioner_cut=donor.fingerprint,
                    rows=indices, base_row_descriptors=[describe(bc[:, i:i+1]) for i in indices],
                    source_row_descriptors=[describe(dc[:, i:i+1]) for i in indices],
                    changed_bf16_elements_per_row=changed,
                    changed_payload_bytes=int(np.count_nonzero(b.view(np.uint8) != d.view(np.uint8))),
                    affected_bf16_storage_bytes=sum(changed)*2,
                    instrument_status='contextual-source-difference-observed' if any(changed) else 'selected-source-rows-identical',
                    source_assisted=True, no_new_encoder=True)
                predictions[cid] = dict(image=array(pipe.transformer.x_embedder(base.payload['latent'])),
                                         text=array(pipe.transformer.context_embedder(mixed)))
                del base, donor, bc, dc, mixed, b, d
        finally:
            active['forecast'] = False
        if counts != before_forecast:
            raise ValueError('native work occurred during projection forecast')
        return dict(predictions=dict(semantic=config['hypotheses'], counts_at_seal=before_forecast,
                                     projections=predictions, selections=selections, source_diagnostics=diagnostics))
    rows, fields = [], []
    with torch.inference_mode():
        tx.predict('context_control_predictions', forecast)
        save(out / 'source-row-diagnostics.json', diagnostics)
        seal = tx.seal()
        event('sealed', seal=seal)
        def execute(_):
            for context in config['contexts']:
                cid = context['id']
                active['case'] = cid
                parent = load_source(parents[cid]['primed'])
                donor = load_source(parents[cid]['source'])
                base_c = parent.payload['conditioning'].clone()
                source_c = donor.payload['conditioning'].clone()
                nav = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=56, on_receipt=receipt)
                del parent, donor
                marks = {}
                for step in range(56):
                    point = dict(nav.point)
                    if point['phase'] == 'project':
                        c = base_c.clone()
                        if point['logical_step'] == 2:
                            ix = selections[cid]['rows']
                            c[:, ix] = source_c[:, ix]
                        apply(nav, c, 'context-control.C' + str(point['logical_step']))
                        marks['project' + str(point['logical_step']) + '_input'] = capture(nav).fingerprint
                    if step == 1:
                        entry = capture(nav)
                        marks['joint0_entry'] = entry.fingerprint
                        for field in ('image', 'text'):
                            fields.append(dict(case=cid, arm=ARM, field=field,
                                prediction=predictions[cid][field], actual=array(entry.payload[field])))
                        del entry
                    nav.step()
                final = capture(nav)
                png = cid + '.context-controls.png'
                adapter.decode(nav.session, height=512, width=512).save(out / png)
                counts['images'] += 1
                row = dict(case=cid, arm=ARM, rows=selections[cid]['rows'], source_parent=roots_for_case(parents, cid),
                    png=png, png_sha256=sha(out / png), final_cut=final.fingerprint,
                    final_latent=array(final.payload['latent']), marks=marks)
                rows.append(row)
                save(out / ('completed-' + cid + '.json'), row)
                event('image_complete', case=cid, arm=ARM)
                del nav, base_c, source_c, final, c
            return dict(counts=counts.copy(), rows=len(rows))
        with tx:
            tx.execute('native_context_controls', execute)
    receipts.close()
    for hook in hooks:
        hook.remove()
    if counts != expected_counts():
        raise ValueError('finite execution census differs: ' + str(counts))
    for name, value in (('rows.json', rows), ('fields.json', fields), ('actions.json', actions)):
        save(out / name, value)
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json', 'retrieve_flux_encoder.py'):
        shutil.copyfile(payload / name, out / name)
    save(out / 'native-contract.json', dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
        diffusers=importlib.metadata.version('diffusers'), dtype='bfloat16', batch_size=1,
        bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        source_root=str(source), source_manifest_sha256=config['source_manifest_sha256']))
    save(out / 'external-source.json', dict(host='beast', root=str(source), files_manifest_sha256=config['source_manifest_sha256'],
        referenced_files={name: inventory[name] for name in sorted(verified)}, source_request_sha256=config['source_request_sha256']))
    save(out / 'summary.json', dict(schema='mdb-binding-context-controls-v1', job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=request['request_sha256'], output_root=str(out), source_job=config['source_job'],
        counts=counts, counts_at_seal=before_forecast, seal=seal, transaction=verify_transaction(tx.root),
        seconds=time.monotonic()-started, mechanics_status='pending-independent-audit', terminal_status='unassessed'))
    event('complete')
    save(out / 'files.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'],
        output_root=str(out), files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


def roots_for_case(parents, cid):
    return dict(primed=parents[cid]['primed'], source_conditioning=parents[cid]['source'])


def stage(run, main_run):
    from mdb_lanes.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    h = helpers()
    read, save, sha = [h[k] for k in ('read', 'save', 'sha')]
    old = read(main_run / 'request.json')
    jid = read(main_run / 'submission.json')['job_id']
    _, raw, _ = scheduler_api().request('GET', '/api/jobs/' + jid, timeout_s=20)
    job = json.loads(raw)
    if job['state'] != 'succeeded':
        raise ValueError('completed main native campaign required')
    source = Path(old['config']['output_scratch']) / jid / 'binding-minimal-result'
    code = '''import json,hashlib
from pathlib import Path
r=Path(ROOT);names=('plan.json','request.json','summary.json','parents.json','token-addresses.json');manifest=json.loads((r/'files.json').read_text());values={}
for n in names:
 p=r/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[n];values[n]=json.loads(p.read_text())
print(json.dumps(dict(manifest_sha256=hashlib.sha256((r/'files.json').read_bytes()).hexdigest(),values=values)))
'''.replace('ROOT', repr(str(source)))
    result = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code,
                                      text=True, capture_output=True, check=True).stdout)
    values = result['values']
    if values['request.json']['request_sha256'] != old['request_sha256'] or values['plan.json'] != old['config']:
        raise ValueError('source job/config binding differs')
    addresses, native_plan = values['token-addresses.json'], values['plan.json']
    selections = {c['id']: select_rows(c, addresses['tokens'][c['id']], addresses['selectors'][c['id']]) for c in native_plan['contexts']}
    config = dict(schema='mdb-binding-context-controls-plan-v1', source_job=jid, source_root=str(source),
        source_manifest_sha256=result['manifest_sha256'], source_request_sha256=old['request_sha256'],
        model_path=native_plan['model_path'], original_source_sha256=native_plan['original_source_sha256'],
        contexts=native_plan['contexts'], selections=selections, output_scratch='/mnt/big/mdb/research',
        expected_counts=expected_counts(), max_wall_seconds=540, directory_custody=True,
        hypotheses=dict(stage='Exploratory contextual unchanged-row control fixed before any new model work.',
            prediction='Attribute binding should remain the base assignment under the fixed changed-token-specific hypothesis; retain every counterexample.',
            alternative='Unchanged post-attribute contextual rows may carry a broader sustaining program and transfer the source assignment.',
            semantics='Native object-specific attribute/shape/order review is separate from pixel distance; identical selected source rows make the control uninformative, not a null mechanism.',
            numerical='Sixteen complete projected I/T fields seal before eight current native suffixes; no attention or RGB forecast.'),
        source_assistance='Actual main-run primed project2 latent/conditioning and source natural conditioner; no new encoder, noise draw or prefix.')
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'plan.json', config)
    extras = [(HERE / name, name) for name in ('flux_binding_generalization.py', 'flux_source_receiver.py', 'flux_family_custody.py', 'retrieve_flux_encoder.py')]
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json'):
        origin = main_run / 'payload' / name
        shutil.copyfile(origin, run / name)
        extras.append((run / name, name))
    pre = run / 'preflight.v1.json'
    save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name, *[name for _, name in extras], pre.name],
                   path_token_allowlist=[str(source), config['model_path'], config['output_scratch'], 'binding-minimal-result']))
    extras.append((pre, pre.name))
    req = dict(schema='mdb-binding-context-controls-request-v1', worker_script=Path(__file__).name, config=config,
        model=old['model'], execution_environment=old['execution_environment'], needs=dict(cuda=True, host='beast'),
        task_family='forward', backend='diffusers', device='cuda', dtype='bfloat16', batch_size=1, height=512, width=512,
        source_hashes={name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        resource_exception=dict(ram_mb=10240, vram_mb=10240, cpu_threads=4, disk_gb=2, est_wall_s=240,
            calibration_job=jid, measured_peak_rss_mb=6761.2, measured_peak_vram_mb=9210.0,
            reason='Main6008op/88VAE job measured426.12s/6761.2MBRSS/9210MBVRAM. Current eight saved-parent56-operation suffixes/400blocks/8VAEs,16typedCActs; no encoder/noise/prefix. Same resident512px profile and near-data cuts;10GiBRAM/10GiBVRAM reservations. Stream BF16 arrays/CAS;240sestimate/540sworker/600souter,no retries.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=24, logical_branches=8,
            logical_checkpoint_replays=8, max_local_work_items=32,
            checkpoint_policy='Eight source-assisted contextual row controls;0encodes/0noise/prefix;16projectI/Tforecasts seal before448navops/400blocks/16Acts/8VAEs. SerialB1 CUDA, no nestedjobs/retries.'))
    req['request_sha256'] = hashlib.sha256(json.dumps(req, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / 'request.json', req)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    stage_worker(run / 'payload', worker=Path(__file__), config=config, request=req, additional_files=extras)
    return dict(request_sha256=req['request_sha256'], source_job=jid, selections=selections, counts=expected_counts())


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    h = helpers()
    read, save, sha = [h[k] for k in ('read', 'save', 'sha')]
    req = read(run / 'request.json')
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    if (run / 'submission.json').exists() or any(sha(run / 'payload' / name) != digest for name, digest in req['source_hashes'].items()):
        raise ValueError('changed/already submitted control payload')
    command = worker_command(req['worker_script'])
    command.insert(command.index('python') + 1, '-B')
    resources = {key: req['resource_exception'][key] for key in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')}
    job = guarded_mrun_launch(command, request=req, model=req['model'], task='forward', experiment='mdb-binding-context-controls',
        payload=run / 'payload', gpu=True, needs=req['needs'], prefer_host='beast', timeout_s=600, preflight=True,
        retry_on_kill=False, **resources,
        note='8saved-parent contextual unchanged-row controls, first2validpromptrows after changedIDs.0encodes/noise/prefix;16projectI/Tforecasts seal before448navops/400blocks/16CActs/8VAEs. Source manifest/CAS bound; serialB1CUDA,240sestimate/540sworker/600souter,0retries. Retain all images and source-identical/controlwins as observations.')
    save(run / 'submission.json', dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1':
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('run', type=Path)
        parser.add_argument('--main-run', type=Path)
        parser.add_argument('--submit', action='store_true')
        args = parser.parse_args()
        print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(), args.main_run.resolve()), sort_keys=True))
