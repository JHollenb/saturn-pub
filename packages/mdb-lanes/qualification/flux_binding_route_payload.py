"""Direct projected-text payload x row-support factorial on saved MDB parents."""
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
ARMS = ('sham', 'replay-A.full', 'replay-B.full', 'A@A', 'A@B', 'B@A', 'B@B')


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def expected_counts():
    return dict(loads=1, encodes=0, noise_draws=0, native_transitions=1540,
                native_blocks=1400, acts=56, receipts=1596, images=28)


def worker():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    h = helpers(); read, save, sha, identity = [h[k] for k in ('read', 'save', 'sha', 'identity')]
    req, c = [read(os.environ[k]) for k in ('SATURN_DEBUG_REQUEST', 'SATURN_RUN_CONFIG')]
    if req['config'] != c or any(sha(Path(n)) != v for n, v in req['source_hashes'].items()):
        raise ValueError('request/source binding differs')
    prior = Path(c['prior_root']); inventory = read(prior / 'files.json'); verified = set()
    if sha(prior / 'files.json') != c['prior_manifest_sha256']:
        raise ValueError('prior inventory changed')
    def member(name):
        p = prior / name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(prior):
            raise ValueError('escaped prior evidence')
        if name not in verified:
            if sha(p) != inventory[name]: raise ValueError('prior custody differs: ' + name)
            verified.add(name)
        return p
    if read(member('request.json'))['request_sha256'] != c['prior_request_sha256']:
        raise ValueError('prior request changed')
    old_plan = read(member('plan.json')); old_rows = read(member('rows.json')); old_fields = read(member('fields.json'))
    if c['contexts'] != [x for x in old_plan['contexts'] if x['id'] in CASES]:
        raise ValueError('context identity differs')
    row_index = {(x['case'], x['arm']): x for x in old_rows}
    field_index = {(x['case'], x['arm'], x['field']): x for x in old_fields}
    out = Path(c['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    parents = {cid: row_index[cid, 'sham']['marks']['joint0_entry'] for cid in CASES}
    save(out / 'plan.json', c); save(out / 'request.json', req); save(out / 'source-parents.json', parents)
    counts = {k: 0 for k in expected_counts()}; active = dict(case='setup', arm='setup', forecast=False)
    boot = time.monotonic()
    def event(kind, **fields):
        row = dict(kind=kind, seconds=time.monotonic()-boot, counts=counts.copy(), **fields)
        with (out / 'timeline.jsonl').open('a') as stream: stream.write(json.dumps(row, sort_keys=True) + '\n')
        print('MDB_ROUTE_PAYLOAD ' + json.dumps(row, sort_keys=True), flush=True)
        if row['seconds'] > c['max_wall_seconds']: raise RuntimeError('finite route/payload wall budget')
    event('setup')
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
    math = runpy.run_path(str(HERE / 'flux_binding_route_payload_math.py'))
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available(): raise ValueError('CUDA admission required')
    for name, digest in c['original_source_sha256'].items():
        if sha(Path(importlib.metadata.distribution('diffusers').locate_file(name))) != digest:
            raise ValueError('native implementation changed')
    runpy.run_path(str(HERE / 'flux_source_receiver.py'))['RecipeBank'](read('parameter-artifacts.json'), c['model_path'])
    pipe = Flux2KleinPipeline.from_pretrained(c['model_path'], torch_dtype=torch.bfloat16,
                                             local_files_only=True, text_encoder=None, tokenizer=None)
    pipe.set_progress_bar_config(disable=True); counts['loads'] += 1
    for component in (pipe.transformer, pipe.vae): component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity='operation', residency='resident', device='cuda')
    original, store = LocalStore(prior / 'store'), LocalStore(out / 'store')
    def parent(cid):
        fp = parents[cid]; meta = read(member('store/cuts/' + fp + '.json'))
        for page in meta['slots'].values(): member('store/pages/' + page + '.json')
        for blob in meta['physical_blobs']: member('store/blobs/' + blob + '.safetensors')
        cut = original.load(fp, device='cuda'); point = cut.execution_point
        if cut.model_identity != adapter.model_identity or point['logical_step'] != 2 or point['phase'] != 'joint' or point['index'] != 0:
            raise ValueError('source model/clock changed')
        store.save(cut); return cut
    def bits(value):
        if value.dtype != torch.bfloat16: raise ValueError('non-BF16 captured native field')
        return value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()
    def array(value):
        a = bits(value) if hasattr(value, 'detach') else np.asarray(value)
        if a.dtype != np.uint16: raise ValueError('BF16 bits required')
        digest = identity(dict(shape=list(a.shape), bytes=hashlib.sha256(a.tobytes()).hexdigest()))
        p = out / 'arrays' / (digest + '.npy')
        if not p.exists(): p.parent.mkdir(exist_ok=True); np.save(p, a, allow_pickle=False)
        return dict(path=p.relative_to(out).as_posix(), sha256=sha(p), shape=list(a.shape), dtype='bfloat16', storage_dtype='uint16')
    def prior_bits(cid, arm):
        ref = field_index[cid, arm, 'text']['actual']; p = member(ref['path'])
        if sha(p) != ref['sha256']: raise ValueError('prior projected text hash differs')
        a = np.load(p, allow_pickle=False)
        if a.dtype != np.uint16 or list(a.shape) != [1, 512, 3072]: raise ValueError('prior text geometry differs')
        return a
    def from_bits(a): return torch.from_numpy(a.copy()).view(torch.bfloat16).to('cuda')
    def entered(module, args):
        if active['forecast']: raise RuntimeError('native block entered before seal')
        counts['native_blocks'] += 1
    handles = [block.register_forward_pre_hook(entered) for block in (*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks)]
    receipt_file = gzip.open(out / 'receipts.jsonl.gz', 'wt', compresslevel=1)
    def receipt(value):
        body = value.to_dict(); key = 'native_transitions' if body['operation'] == 'continue' else 'acts'
        counts[key] += body.get('steps', 1); counts['receipts'] += 1
        if counts[key] > expected_counts()[key]: raise RuntimeError('finite operation budget')
        receipt_file.write(json.dumps(dict(case=active['case'], arm=active['arm'], receipt=body), sort_keys=True) + '\n')
    def capture(nav): cut = nav.session.capture(retain=False); store.save(cut); return cut
    actions = []
    def apply(nav, slot, value, label):
        before = capture(nav); result = nav.apply(Act.replace(slot, value, name=label)); after = capture(nav)
        if describe(after.payload) != describe({**before.payload, slot: value}): raise ValueError('undeclared slot write')
        actions.append(dict(case=active['case'], arm=active['arm'], slot=slot, label=label,
                            before=before.fingerprint, after=after.fingerprint, receipt=result.to_dict()))
        return after
    counts_at_seal = counts.copy(); recipes = {}
    tx = ForecastTransaction(out / 'transaction', model_identity=adapter.model_identity, input_identity=identity(c),
        source_identity=identity(req['source_hashes']),
        frontier=dict(kind='architectural', observed='Four exact prior sham joint0-entry parents and source T arrays; old image outcomes known.'),
        execution_scope='Supplied-state BF16 transport and entry-I invariance witnesses. No neural-operation, attention, MLP or pixel numerical forecasts.',
        budgets=dict(max_native_calls=1, max_wall_ms=600000))
    def forecast(_):
        active['forecast'] = True
        try:
            for cid in CASES:
                cut = parent(cid); base = bits(cut.payload['text']); image = array(cut.payload['image'])
                if not np.array_equal(base, prior_bits(cid, 'sham')): raise ValueError('parent/source T join differs')
                supports = c['selectors'][cid]; ra, rb = supports['A'], supports['B']
                mid, bfull = prior_bits(cid, 'A.mid'), prior_bits(cid, 'B.full')
                for source, selected in ((mid, ra), (bfull, rb)):
                    complement = [i for i in range(512) if i not in selected]
                    if not np.array_equal(source[:, complement], base[:, complement]): raise ValueError('payload extends outside source rows')
                target = math['norm'](base[:, ra], mid[:, ra]); recipes[cid] = {}
                for arm in ARMS:
                    if arm == 'sham':
                        support, payload, source_arm, alpha = 'A', 'none', 'sham', 0.
                        value = base[:, ra].copy()
                    elif arm.startswith('replay-'):
                        support = payload = arm[7]; source_arm = support + '.full'; alpha = 1.
                        value = prior_bits(cid, source_arm)[:, supports[support]].copy()
                    else:
                        payload, support = arm.split('@'); source_arm = 'A.mid' if payload == 'A' else 'B.full'
                        srcrows, destrows = supports[payload], supports[support]
                        source = mid if payload == 'A' else bfull
                        alpha, value, _ = math['match'](base[:, destrows], base[:, srcrows], source[:, srcrows], target)
                    selected = supports[support]; full = base.copy(); full[:, selected] = value
                    landed = math['norm'](base[:, selected], value)
                    recipes[cid][arm] = dict(arm=arm, support=support, payload=payload, source_arm=source_arm,
                        alpha=alpha, rows=selected, value_rows=array(value), landed_delta_l2=landed,
                        target_norm=target if '@' in arm else None,
                        match_rel_error=(landed-target)/target if '@' in arm else None,
                        entry_text=array(full), entry_image=image)
                del cut, base, mid, bfull
        finally: active['forecast'] = False
        if counts != counts_at_seal: raise ValueError('native execution preceded seal')
        return dict(predictions=dict(counts_at_seal=counts_at_seal, recipes=recipes, semantic=c['hypotheses']))
    rows, fields, ports = [], [], []
    with torch.inference_mode():
        tx.predict('route_payload_entry_witnesses', forecast); save(out / 'source-recipes.json', recipes)
        seal = tx.seal(); event('sealed', seal=seal)
        def execute(_):
            for cid in CASES:
                active['case'] = cid
                for arm in ARMS:
                    active['arm'] = arm; cut = parent(cid); base_c = cut.payload['conditioning'].clone()
                    nav = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=55, on_receipt=receipt); del cut
                    event('branch_started', case=cid, arm=arm)
                    recipe = recipes[cid][arm]
                    supplied = np.load(out / recipe['entry_text']['path'], allow_pickle=False)
                    entry = apply(nav, 'text', from_bits(supplied), arm + '.T-entry')
                    marks = dict(entry=entry.fingerprint)
                    fields.append(dict(case=cid, arm=arm, expected_text=recipe['entry_text'], actual_text=array(entry.payload['text']),
                                       expected_image=recipe['entry_image'], actual_image=array(entry.payload['image']), entry_cut=entry.fingerprint))
                    values, hooks = {}, []
                    block = pipe.transformer.transformer_blocks[0]
                    def q_input(module, args): values['normalized_text'] = array(args[0])
                    def output_hook(name): return lambda module, args, output: values.__setitem__(name, array(output))
                    def attention(module, args, output):
                        values['attention_image'] = array(output[0]); values['attention_text'] = array(output[1])
                    hooks.append(block.attn.add_q_proj.register_forward_pre_hook(q_input))
                    for key in ('q', 'k', 'v'):
                        hooks.append(getattr(block.attn, 'add_' + key + '_proj').register_forward_hook(output_hook('text_' + key)))
                    hooks.append(block.attn.register_forward_hook(attention))
                    try: nav.step()
                    finally:
                        for hook in hooks: hook.remove()
                    after = capture(nav); marks['after_joint0'] = after.fingerprint
                    values.update(joint0_image=array(after.payload['image']), joint0_text=array(after.payload['text']))
                    if set(values) != {'normalized_text', 'text_q', 'text_k', 'text_v', 'attention_image', 'attention_text', 'joint0_image', 'joint0_text'}:
                        raise ValueError('first reader port capture incomplete')
                    ports.append(dict(case=cid, arm=arm, before_entry=entry.fingerprint,
                                      after_joint0=after.fingerprint, values=values, observed_only=True))
                    del entry, after, values
                    for _step in range(54):
                        point = nav.point
                        if point['phase'] == 'project':
                            if point['logical_step'] != 3: raise ValueError('unexpected reset clock')
                            marks['project3_input'] = apply(nav, 'conditioning', base_c, arm + '.C3-reset').fingerprint
                        nav.step()
                    final = capture(nav); png = cid + '.' + arm + '.png'
                    adapter.decode(nav.session, height=512, width=512).save(out / png); counts['images'] += 1
                    rows.append(dict(case=cid, arm=arm, png=png, png_sha256=sha(out / png), source_parent=parents[cid],
                                     final_cut=final.fingerprint, final_latent=array(final.payload['latent']), marks=marks))
                    event('image_complete', case=cid, arm=arm); del nav, final, base_c
                save(out / ('completed-' + cid + '.json'), [r for r in rows if r['case'] == cid])
            return dict(counts=counts.copy(), images=len(rows))
        with tx: tx.execute('native_route_payload_suffixes', execute)
    receipt_file.close()
    for handle in handles: handle.remove()
    if counts != expected_counts(): raise ValueError('execution census differs: ' + repr(counts))
    for name, value in (('rows.json', rows), ('fields.json', fields), ('actions.json', actions), ('ports.json', ports)): save(out / name, value)
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json', 'retrieve_flux_encoder.py'):
        shutil.copyfile(name, out / name)
    save(out / 'external-source.json', dict(prior_root=str(prior), prior_manifest_sha256=c['prior_manifest_sha256'], referenced_files={n: inventory[n] for n in sorted(verified)}))
    save(out / 'native-contract.json', dict(device=torch.cuda.get_device_name(), dtype='bfloat16', batch_size=1,
        torch=torch.__version__, diffusers=importlib.metadata.version('diffusers'), bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    save(out / 'summary.json', dict(schema='mdb-binding-route-payload-v1', job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=req['request_sha256'], output_root=str(out), counts=counts, counts_at_seal=counts_at_seal,
        seal=seal, transaction=verify_transaction(tx.root), seconds=time.monotonic()-boot,
        mechanics_status='pending-independent-audit', terminal_status='unassessed'))
    event('complete'); save(out / 'files.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=req['request_sha256'],
        output_root=str(out), files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


def stage(run, prior_run):
    from mdb_lanes.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    h = helpers(); read, save, sha = [h[k] for k in ('read', 'save', 'sha')]
    old = read(prior_run / 'request.json'); jid = read(prior_run / 'submission.json')['job_id']
    _, raw, _ = scheduler_api().request('GET', '/api/jobs/' + jid, timeout_s=20)
    if json.loads(raw)['state'] != 'succeeded': raise ValueError('completed prior required')
    root = str(Path(old['config']['output_scratch']) / jid / 'binding-minimal-result')
    code = 'import json,hashlib\nfrom pathlib import Path\np=Path(' + repr(root) + ')\nprint(json.dumps(dict(sha=hashlib.sha256((p/"files.json").read_bytes()).hexdigest())))\n'
    meta = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code, text=True, capture_output=True, check=True).stdout)
    c = dict(schema='mdb-binding-route-payload-plan-v1', prior_root=root, prior_manifest_sha256=meta['sha'],
        prior_request_sha256=old['request_sha256'], main_root=old['config']['main_root'], main_manifest_sha256=old['config']['main_manifest_sha256'],
        contexts=[x for x in old['config']['contexts'] if x['id'] in CASES], selectors={cid: old['config']['selectors'][cid] for cid in CASES},
        arms=list(ARMS), model_path=old['config']['model_path'], original_source_sha256=old['config']['original_source_sha256'],
        output_scratch='/mnt/big/mdb/research', directory_custody=True, max_wall_seconds=600,
        transport='Ordered two-row source-minus-sham actual T delta; FP64 destination-base + alpha*delta -> FP32 -> ties-even BF16. Equal actual landed T norm to A.mid. No token IDs or positions moved.',
        hypotheses=dict(stage='Frozen exploratory predictions; historical parents/source arrays/known replay images explicitly supplied.',
            primary_cases=list(CASES), primary_rank='Per case A@A signed RGB increment on fixed natural source-minus-base axis exceeds A@B, B@A and B@B; retain every miss.',
            semantic='Crossed/scaled cells A@B/B@A/B@B retain base assignment or mixed/collateral response rather than a clean swap; clean swaps refute this expectation.',
            replay='sham/replay-A.full/replay-B.full/A@A rejoin prior sham/A.full/B.full/A.mid final payload, latent and PNG exactly.',
            numerical_scope='28 complete supplied T-entry witnesses and invariant I fields; no neural equation, attention, MLP or pixel forecast.',
            alternatives=['payload direction dominates', 'receiving row/position dominates', 'direction-by-row compatibility', 'distributed contextual read with normalization-dependent potency'],
            limitations=['Source-assisted historical contexts, not held-out generalization.', 'Direct-T intervention tests this boundary, not a natural prompt encoder program.', 'Equal T norm need not imply equal normalized text or Q/K/V dose.']))
    run.mkdir(parents=True, exist_ok=False); save(run / 'plan.json', c)
    names = ('flux_binding_generalization.py', 'flux_binding_generalization_audit.py', 'flux_binding_route_payload_math.py',
             'flux_source_receiver.py', 'flux_family_custody.py', 'retrieve_flux_encoder.py')
    extras = [(HERE / n, n) for n in names]
    for n in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json'):
        shutil.copyfile(prior_run / 'payload' / n, run / n); extras.append((run / n, n))
    pre = run / 'preflight.v1.json'; save(pre, dict(schema_version='mrun-preflight-config-v1',
        hashed_roots=[Path(__file__).name, *[n for _, n in extras], pre.name],
        path_token_allowlist=[root, c['main_root'], c['model_path'], c['output_scratch'], 'binding-minimal-result']))
    extras.append((pre, pre.name))
    req = dict(schema='mdb-binding-route-payload-request-v1', worker_script=Path(__file__).name, config=c,
        model=old['model'], execution_environment=old['execution_environment'], needs=dict(cuda=True, host='beast'),
        task_family='forward', backend='diffusers', device='cuda', dtype='bfloat16', batch_size=1, height=512, width=512,
        source_hashes={n: sha(p) for p, n in [(Path(__file__), Path(__file__).name), *extras]},
        resource_exception=dict(ram_mb=10240, vram_mb=10240, cpu_threads=4, disk_gb=4, est_wall_s=240,
            calibration_job=jid, measured_peak_rss_mb=7059.6, measured_peak_vram_mb=9220.,
            reason='Same original resident model; prior64suffixes273.9s/7059.6MBRSS/9220MBVRAM. Current28shorter55op suffixes plus first-reader BF16 captures.10GiB RAM/VRAM,4GBdisk,240sestimate/600sworker/660souter;no encodes/noise/prefix/retries.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=84, logical_branches=28,
            logical_checkpoint_replays=28, max_local_work_items=128,
            checkpoint_policy='28suppliedT/Ientrywitnesses globally sealed before28native55op suffixes/1400blocks/56Acts/28VAEs;source-onlytextnormsolve;one CUDA B1lease,0encoder/noise/prefix/retries.'))
    req['request_sha256'] = hashlib.sha256(json.dumps(req, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / 'request.json', req)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    stage_worker(run / 'payload', worker=Path(__file__), config=c, request=req, additional_files=extras)
    return dict(request_sha256=req['request_sha256'], counts=expected_counts())


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    h = helpers(); read, save, sha = [h[k] for k in ('read', 'save', 'sha')]; req = read(run / 'request.json')
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](req['execution_environment'])
    if (run / 'submission.json').exists() or any(sha(run / 'payload' / n) != v for n, v in req['source_hashes'].items()):
        raise ValueError('changed/already submitted payload')
    command = worker_command(req['worker_script']); command.insert(command.index('python') + 1, '-B')
    resources = {k: req['resource_exception'][k] for k in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')}
    job = guarded_mrun_launch(command, request=req, model=req['model'], task='forward', experiment='mdb-binding-route-payload',
        payload=run / 'payload', gpu=True, needs=req['needs'], prefer_host='beast', timeout_s=660,
        preflight=True, retry_on_kill=False, **resources,
        note='DirectT direction×support:4contexts×7arms=28nativeimages/1540ops/1400blocks/56Acts.16knownreplays12newcross/scaledfutures. MatchlandedTnormtoA.mid;28T/Ientrywitnessessealedfirst;capturefirst-reader norm/QKV/attention.1CUDA B1lease240sestimate/600sworker/660souter;0encoder/noise/prefix/retries.')
    save(run / 'submission.json', dict(job_id=str(job))); return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1': worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
        parser.add_argument('--prior-run', type=Path); parser.add_argument('--submit', action='store_true'); args = parser.parse_args()
        print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(), args.prior_run.resolve()), sort_keys=True))
