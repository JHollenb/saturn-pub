"""Cross actual post-joint0 image/text states and replay the native consumer."""
from __future__ import annotations
import argparse
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
CASES = ('anchor', 'seed', 'wording', 'objects')
ARMS = ('I.P_T.P', 'I.P_T.Q', 'I.Q_T.P', 'I.Q_T.Q')


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def expected_counts():
    return dict(loads=1, encodes=0, noise_draws=0, native_transitions=864,
                native_blocks=784, acts=48, receipts=912, images=16)


def payload_slot_descriptors(payload):
    """Keep typed tags inside slots, with ordinary names at the payload root."""
    from saturn_pub.values import describe
    return {name: describe(value) for name, value in payload.items()}


def landmark_record(cut, case, arm, label):
    """Convert the public immutable cut metadata at the JSON boundary."""
    payload = cut.payload
    return dict(case=case, arm=arm, label=label, cut=cut.fingerprint,
                point=dict(cut.execution_point),
                payload_descriptors=payload_slot_descriptors(
                    {key: payload[key] for key in ('image', 'text', 'latent', 'conditioning')}))


def compatible_payloads(left, right):
    """Guard the declared mutable closure; no scientific effect threshold."""
    if set(left) != set(right) or not {'image', 'text', 'latent', 'conditioning'} <= set(left):
        raise ValueError('paired state slot set differs')
    common = [key for key in left if key not in ('image', 'text')]
    if any(left[key] != right[key] for key in common):
        raise ValueError('source pairs differ outside image/text closure')
    for key, shape in (('image', [1, 1024, 3072]), ('text', [1, 512, 3072])):
        for descriptor in (left[key], right[key]):
            if descriptor.get('kind') != 'tensor' or descriptor.get('dtype') != 'torch.bfloat16' or descriptor.get('shape') != shape:
                raise ValueError('source stream geometry differs')
    return dict(common_nonstream_slots=sorted(common))


def worker():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    h = helpers(); read, save, sha, identity = [h[k] for k in ('read', 'save', 'sha', 'identity')]
    req, c = [read(os.environ[k]) for k in ('SATURN_DEBUG_REQUEST', 'SATURN_RUN_CONFIG')]
    if req['config'] != c or any(sha(Path(n)) != digest for n, digest in req['source_hashes'].items()):
        raise ValueError('request/source identity differs')
    prior = Path(c['prior_root']); inventory = read(prior / 'files.json'); verified = set()
    if sha(prior / 'files.json') != c['prior_manifest_sha256']: raise ValueError('prior manifest changed')
    def member(name):
        p = prior / name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(prior):
            raise ValueError('escaped source member')
        if name not in verified:
            if sha(p) != inventory[name]: raise ValueError('prior member changed: ' + name)
            verified.add(name)
        return p
    if read(member('request.json'))['request_sha256'] != c['prior_request_sha256']: raise ValueError('prior request changed')
    old_plan, prior_rows = read(member('plan.json')), read(member('rows.json'))
    if c['contexts'] != old_plan['contexts']: raise ValueError('historical context set differs')
    index = {(r['case'], r['arm']): r for r in prior_rows}
    parents = {cid: {role: index[cid, arm]['marks']['after_joint0'] for role, arm in (('P', 'A@A'), ('Q', 'A@B'))} for cid in CASES}
    out = Path(c['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'; out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', c); save(out / 'request.json', req); save(out / 'source-parents.json', parents)
    counts = {k: 0 for k in expected_counts()}; active = dict(case='setup', arm='setup', forecast=False); boot = time.monotonic()
    def event(kind, **fields):
        row = dict(kind=kind, seconds=time.monotonic()-boot, counts=counts.copy(), **fields)
        with (out / 'timeline.jsonl').open('a') as stream: stream.write(json.dumps(row, sort_keys=True) + '\n')
        print('MDB_CARRIER_CROSS ' + json.dumps(row, sort_keys=True), flush=True)
        if row['seconds'] > c['max_wall_seconds']: raise RuntimeError('finite carrier-cross wall budget')
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
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available(): raise ValueError('CUDA admission required')
    for name, digest in c['original_source_sha256'].items():
        if sha(Path(importlib.metadata.distribution('diffusers').locate_file(name))) != digest: raise ValueError('native source changed')
    runpy.run_path(str(HERE / 'flux_source_receiver.py'))['RecipeBank'](read('parameter-artifacts.json'), c['model_path'])
    pipe = Flux2KleinPipeline.from_pretrained(c['model_path'], torch_dtype=torch.bfloat16,
                                             local_files_only=True, text_encoder=None, tokenizer=None)
    pipe.set_progress_bar_config(disable=True); counts['loads'] += 1
    for component in (pipe.transformer, pipe.vae): component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity='operation', residency='resident', device='cuda')
    original, store = LocalStore(prior / 'store'), LocalStore(out / 'store')
    def parent(cid, role):
        fp = parents[cid][role]; meta = read(member('store/cuts/' + fp + '.json'))
        for page in meta['slots'].values(): member('store/pages/' + page + '.json')
        for blob in meta['physical_blobs']: member('store/blobs/' + blob + '.safetensors')
        cut = original.load(fp, device='cuda'); point = cut.execution_point
        if cut.model_identity != adapter.model_identity or point['logical_step'] != 2 or point['phase'] != 'joint' or point['index'] != 1:
            raise ValueError('source model/clock differs')
        store.save(cut); return cut
    def array(value):
        if value.dtype != torch.bfloat16: raise ValueError('BF16 native field required')
        a = value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()
        digest = identity(dict(shape=list(a.shape), bytes=hashlib.sha256(a.tobytes()).hexdigest()))
        p = out / 'arrays' / (digest + '.npy')
        if not p.exists(): p.parent.mkdir(exist_ok=True); np.save(p, a, allow_pickle=False)
        return dict(path=p.relative_to(out).as_posix(), sha256=sha(p), shape=list(a.shape), dtype='bfloat16', storage_dtype='uint16')
    def tensor(ref): return torch.from_numpy(np.load(out / ref['path'], allow_pickle=False).copy()).view(torch.bfloat16).to('cuda')
    def entered(module, args):
        if active['forecast']: raise RuntimeError('target block entered before seal')
        counts['native_blocks'] += 1
    handles = [b.register_forward_pre_hook(entered) for b in (*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks)]
    receipt_file = gzip.open(out / 'receipts.jsonl.gz', 'wt', compresslevel=1)
    def receipt(value):
        body = value.to_dict(); key = 'native_transitions' if body['operation'] == 'continue' else 'acts'
        counts[key] += body.get('steps', 1); counts['receipts'] += 1
        if counts[key] > expected_counts()[key]: raise RuntimeError('finite operation budget')
        receipt_file.write(json.dumps(dict(case=active['case'], arm=active['arm'], receipt=body), sort_keys=True) + '\n')
    def capture(nav): cut = nav.session.capture(retain=False); store.save(cut); return cut
    actions, trajectory = [], []
    def apply(nav, slot, value, label):
        before = capture(nav); result = nav.apply(Act.replace(slot, value, name=label)); after = capture(nav)
        if describe(after.payload) != describe({**before.payload, slot: value}): raise ValueError('undeclared state write')
        actions.append(dict(case=active['case'], arm=active['arm'], slot=slot, label=label,
                            before=before.fingerprint, after=after.fingerprint, receipt=result.to_dict()))
        return after
    def landmark(nav, label):
        cut = capture(nav)
        trajectory.append(landmark_record(cut, active['case'], active['arm'], label))
        return cut
    counts_at_seal = counts.copy(); recipes, pair_contracts = {}, {}
    tx = ForecastTransaction(out / 'transaction', model_identity=adapter.model_identity, input_identity=identity(c),
        source_identity=identity(req['source_hashes']),
        frontier=dict(kind='architectural', observed='Actual P=A@A and Q=A@B post-joint0 states and historical images; full paired source state supplied.'),
        execution_scope='Source-assisted I/T combination witnesses; no learned denoiser equation or pixel numerical forecasts.',
        budgets=dict(max_native_calls=1, max_wall_ms=480000))
    def forecast(_):
        active['forecast'] = True
        try:
            for cid in CASES:
                p, q = parent(cid, 'P'), parent(cid, 'Q')
                if p.execution_point != q.execution_point: raise ValueError('paired full execution points differ')
                pair_contracts[cid] = compatible_payloads(payload_slot_descriptors(p.payload), payload_slot_descriptors(q.payload))
                streams = {role: {slot: array(cut.payload[slot]) for slot in ('image', 'text')} for role, cut in (('P', p), ('Q', q))}
                recipes[cid] = {}
                for arm in ARMS:
                    i, t = arm[2], arm[-1]
                    recipes[cid][arm] = dict(arm=arm, image_source=i, text_source=t,
                        image_source_cut=parents[cid][i], text_source_cut=parents[cid][t],
                        entry_image=streams[i]['image'], entry_text=streams[t]['text'])
                del p, q
        finally: active['forecast'] = False
        if counts != counts_at_seal: raise ValueError('native work preceded global seal')
        return dict(predictions=dict(counts_at_seal=counts_at_seal, recipes=recipes, pair_contracts=pair_contracts, semantic=c['hypotheses']))
    rows, fields, ports = [], [], []
    with torch.inference_mode():
        tx.predict('carrier_cross_entry_witnesses', forecast); save(out / 'source-recipes.json', recipes)
        seal = tx.seal(); event('sealed', seal=seal)
        def execute(_):
            for cid in CASES:
                active['case'] = cid
                for arm in ARMS:
                    active['arm'] = arm; cut = parent(cid, 'Q'); base_c = cut.payload['conditioning'].clone()
                    nav = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=54, on_receipt=receipt); del cut
                    event('branch_started', case=cid, arm=arm); recipe = recipes[cid][arm]
                    apply(nav, 'image', tensor(recipe['entry_image']), arm + '.image')
                    apply(nav, 'text', tensor(recipe['entry_text']), arm + '.text')
                    entry = landmark(nav, 'entry'); marks = dict(entry=entry.fingerprint)
                    fields.append(dict(case=cid, arm=arm, expected_image=recipe['entry_image'], expected_text=recipe['entry_text'],
                        actual_image=array(entry.payload['image']), actual_text=array(entry.payload['text']), entry_cut=entry.fingerprint))
                    values = {}
                    def attention(module, args, output):
                        values['attention_image'] = array(output[0]); values['attention_text'] = array(output[1])
                    hook = pipe.transformer.transformer_blocks[1].attn.register_forward_hook(attention)
                    try: nav.step()
                    finally: hook.remove()
                    after = landmark(nav, 'after_joint1'); marks['after_joint1'] = after.fingerprint
                    if set(values) != {'attention_image', 'attention_text'}: raise ValueError('joint1 attention capture incomplete')
                    ports.append(dict(case=cid, arm=arm, before_entry=entry.fingerprint,
                        after_joint1=after.fingerprint, values=values, observed_only=True))
                    del entry, after, values
                    for _step in range(53):
                        point = dict(nav.point)
                        if point['phase'] == 'project':
                            if point['logical_step'] != 3: raise ValueError('unexpected conditioning reset clock')
                            marks['before_project3'] = landmark(nav, 'before_project3').fingerprint
                            apply(nav, 'conditioning', base_c, arm + '.C3-reset')
                            marks['project3_input'] = landmark(nav, 'project3_input').fingerprint
                        nav.step()
                        if point['phase'] == 'project': marks['after_project3'] = landmark(nav, 'after_project3').fingerprint
                        if point['logical_step'] == 3 and point['phase'] == 'joint' and point['index'] == 0:
                            marks['after_step3_joint0'] = landmark(nav, 'after_step3_joint0').fingerprint
                    final = landmark(nav, 'final'); marks['final'] = final.fingerprint
                    png = cid + '.' + arm + '.png'; adapter.decode(nav.session, height=512, width=512).save(out / png); counts['images'] += 1
                    rows.append(dict(case=cid, arm=arm, png=png, png_sha256=sha(out / png), source_parent=parents[cid]['Q'],
                        final_cut=final.fingerprint, final_latent=array(final.payload['latent']), marks=marks))
                    event('image_complete', case=cid, arm=arm); del nav, final, base_c
                save(out / ('completed-' + cid + '.json'), [r for r in rows if r['case'] == cid])
            return dict(counts=counts.copy(), images=len(rows))
        with tx: tx.execute('native_carrier_cross_suffixes', execute)
    receipt_file.close()
    for handle in handles: handle.remove()
    if counts != expected_counts(): raise ValueError('execution census differs: ' + repr(counts))
    for name, value in (('rows.json', rows), ('fields.json', fields), ('actions.json', actions), ('trajectory.json', trajectory), ('ports.json', ports)):
        save(out / name, value)
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json', 'retrieve_flux_encoder.py'): shutil.copyfile(name, out / name)
    save(out / 'external-source.json', dict(prior_root=str(prior), prior_manifest_sha256=c['prior_manifest_sha256'], referenced_files={n: inventory[n] for n in sorted(verified)}))
    save(out / 'native-contract.json', dict(device=torch.cuda.get_device_name(), dtype='bfloat16', batch_size=1,
        torch=torch.__version__, diffusers=importlib.metadata.version('diffusers'), bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    save(out / 'summary.json', dict(schema='mdb-binding-carrier-cross-v1', job_id=os.environ['MRUN_JOB_ID'], request_sha256=req['request_sha256'],
        output_root=str(out), counts=counts, counts_at_seal=counts_at_seal, seal=seal, transaction=verify_transaction(tx.root),
        seconds=time.monotonic()-boot, mechanics_status='pending-independent-audit', terminal_status='unassessed'))
    event('complete'); save(out / 'files.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=req['request_sha256'], output_root=str(out),
        files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


def stage(run, prior_run):
    from mdb.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb.payload import stage_worker
    h = helpers(); read, save, sha = [h[k] for k in ('read', 'save', 'sha')]
    old = read(prior_run / 'request.json'); jid = read(prior_run / 'submission.json')['job_id']
    _, raw, _ = scheduler_api().request('GET', '/api/jobs/' + jid, timeout_s=20)
    if json.loads(raw)['state'] != 'succeeded': raise ValueError('completed route/payload source required')
    root = str(Path(old['config']['output_scratch']) / jid / 'binding-minimal-result')
    code = 'import json,hashlib\nfrom pathlib import Path\np=Path(' + repr(root) + ')\nprint(json.dumps(dict(sha=hashlib.sha256((p/"files.json").read_bytes()).hexdigest())))\n'
    meta = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code, text=True, capture_output=True, check=True).stdout)
    c = dict(schema='mdb-binding-carrier-cross-plan-v1', prior_root=root, prior_manifest_sha256=meta['sha'], prior_request_sha256=old['request_sha256'],
        main_root=old['config']['main_root'], main_manifest_sha256=old['config']['main_manifest_sha256'], contexts=old['config']['contexts'],
        arms=list(ARMS), model_path=old['config']['model_path'], original_source_sha256=old['config']['original_source_sha256'],
        output_scratch='/mnt/big/mdb/research', directory_custody=True, max_wall_seconds=480,
        hypotheses=dict(stage='Exploratory hypotheses frozen before all new hybrid suffixes; paired source states and endpoint images already observed.',
            primary_cases=list(CASES), primary_rank='Per case signed natural-source-axis RGB increment of I.Q_T.P exceeds I.P_T.Q; retain all misses.',
            semantic='I.Q_T.P follows known P assignment (three clear swaps, partial wording); I.P_T.Q retains base Q assignment. Deviations retained per case.',
            replay='I.P_T.P exactly rejoins prior A@A and I.Q_T.Q rejoins prior A@B final payload, latent and PNG.',
            numerical_scope='32 exact supplied I/T entry witnesses, zero learned denoiser equation or pixel numerical forecasts.',
            alternatives=['working-text carrier suffices', 'image state carries the assignment', 'joint-state coherence or interaction is required', 'context-dependent repair or mixed carriers'],
            limitations=['Four dependent historical contexts, three paired conditioning inputs, two seeds; source assisted.',
                         'Sufficiency at this clock and recipient is not necessity at every clock, durable memory or a universal object symbol.',
                         'Entry-state witnesses and observed attention returns are distinct from prospective learned-equation forecasts.']))
    run.mkdir(parents=True, exist_ok=False); save(run / 'plan.json', c)
    names = ('flux_binding_generalization.py', 'flux_source_receiver.py', 'flux_family_custody.py', 'retrieve_flux_encoder.py')
    extras = [(HERE / n, n) for n in names]
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json'):
        shutil.copyfile(prior_run / 'payload' / name, run / name); extras.append((run / name, name))
    pre = run / 'preflight.v1.json'; save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name, *[n for _, n in extras], pre.name],
        path_token_allowlist=[root, c['main_root'], c['model_path'], c['output_scratch'], 'binding-minimal-result']))
    extras.append((pre, pre.name))
    req = dict(schema='mdb-binding-carrier-cross-request-v1', worker_script=Path(__file__).name, config=c, model=old['model'], execution_environment=old['execution_environment'],
        needs=dict(cuda=True, host='beast'), task_family='forward', backend='diffusers', device='cuda', dtype='bfloat16', batch_size=1, height=512, width=512,
        source_hashes={n: sha(p) for p, n in [(Path(__file__), Path(__file__).name), *extras]},
        resource_exception=dict(ram_mb=10240, vram_mb=10240, cpu_threads=4, disk_gb=3, est_wall_s=180,
            calibration_job=jid, measured_peak_rss_mb=7663.4, measured_peak_vram_mb=9222.,
            reason='Same original resident model; prior28suffixes212.02s/7663.4MBRSS/9222MBVRAM. Current16shorter54op suffixes, exact source stream cross, seven CAS landmarks and first reader attention returns.10GiB RAM/VRAM,3GBdisk,180sestimate/480sworker/540souter;0encoder/noise/prefix/retries.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=64, logical_branches=16, logical_checkpoint_replays=16,
            max_local_work_items=96, checkpoint_policy='32suppliedI/Twitnesses globally sealed before16native54op suffixes/784blocks/48Acts/16VAEs.8knownreplays8newhybrids,one CUDA B1lease;0encoder/noise/prefix/retries.'))
    req['request_sha256'] = hashlib.sha256(json.dumps(req, sort_keys=True, allow_nan=False).encode()).hexdigest(); save(run / 'request.json', req)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    stage_worker(run / 'payload', worker=Path(__file__), config=c, request=req, additional_files=extras)
    return dict(request_sha256=req['request_sha256'], counts=expected_counts())


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    h = helpers(); read, save, sha = [h[k] for k in ('read', 'save', 'sha')]; req = read(run / 'request.json')
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    if (run / 'submission.json').exists() or any(sha(run / 'payload' / name) != digest for name, digest in req['source_hashes'].items()):
        raise ValueError('changed/already submitted payload')
    command = worker_command(req['worker_script']); command.insert(command.index('python') + 1, '-B')
    resources = {k: req['resource_exception'][k] for k in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')}
    job = guarded_mrun_launch(command, request=req, model=req['model'], task='forward', experiment='mdb-binding-carrier-cross', payload=run / 'payload',
        gpu=True, needs=req['needs'], prefer_host='beast', timeout_s=540, preflight=True, retry_on_kill=False, **resources,
        note='Postjoint0 image×text:4contexts×4arms=16images/864navops/784blocks/48Acts.8knownselfreplays8newhybrids;32source-entrywitnessesgloballysealedfirst. SevenCASlandmarks and next-joint attentionreturns.1CUDA B1lease180sestimate/480sworker/540souter;0encoder/noise/prefix/retries.')
    save(run / 'submission.json', dict(job_id=str(job))); return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1': worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
        parser.add_argument('--prior-run', type=Path); parser.add_argument('--submit', action='store_true'); args = parser.parse_args()
        print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(), args.prior_run.resolve()), sort_keys=True))
