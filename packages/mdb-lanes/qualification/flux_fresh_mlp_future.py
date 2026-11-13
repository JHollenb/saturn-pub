"""A fresh receiver, frozen MLP programs, own baselines and nine native image futures."""
import hashlib
import inspect
import json
import os
import runpy
import time
from pathlib import Path


def cases():
    return [dict(id='half', doses=[.5, .5]), dict(id='row4heavy', doses=[.875, .125]), dict(id='row11heavy', doses=[.125, .875])]


def expected_counts():
    return dict(loads=1, encodes=0, noise_draws=1, native_transitions=538, acts=15, receipts=37, images=9)


def worker():
    payload = Path.cwd()
    h = runpy.run_path(str(payload/'flux_binding_minimal.py'))
    sha, save, identity = [h[key] for key in ('sha', 'save', 'identity')]
    read = lambda p: json.loads(Path(p).read_text())
    request = read(os.environ['SATURN_DEBUG_REQUEST']); c = read(os.environ['SATURN_RUN_CONFIG'])
    if c != request['config'] or c['cases'] != cases() or any(sha(payload/name) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('frozen request/source differs')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    source = Path(c['source_collection'])
    remote = read(source/'remote-custody.json')
    if sha(source/'remote-custody.json') != c['source_record_sha256'] or runpy.run_path(str(payload/'flux_family_custody.py'))['verify'](source) != remote['verification']:
        raise ValueError('complete original source custody differs')
    data = source/'extracted/binding-minimal-result'; historical_plan = read(data/'plan.json')
    scratch = Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out = scratch/'binding-minimal-result'; out.mkdir(parents=True, exist_ok=False)
    save(out/'plan.json', c); save(out/'request.json', request); save(out/'source-binding.json', dict(remote_record=remote, source_record_sha256=c['source_record_sha256'], selection=c['selection']))
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers import transformer_flux2 as native_module
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.native_navigation import NativeCutNavigator
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.tensor_archive import load_tensor_archive
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available() or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError('original CUDA BF16 profile differs')
    started = time.monotonic(); counts = dict.fromkeys(expected_counts(), 0)
    entries = dict(blocks=0, target_blocks=0, dispatch=0)
    targets = dict(row4heavy=0, row11heavy=0)
    active = dict(phase='setup', branch='setup', target=False)
    arrays, receipts, rows = {}, [], []
    def event(kind, **fields):
        row = dict(kind=kind, seconds=time.monotonic()-started, counts=dict(counts), entries=dict(entries), targets=dict(targets), **fields)
        with (out/'timeline.jsonl').open('a') as f: f.write(json.dumps(row, sort_keys=True)+'\n')
        print('MDB_FRESH_MLP '+json.dumps(row, sort_keys=True), flush=True)
        if row['seconds'] > c['worker_wall_s']: raise RuntimeError('finite fresh-receiver budget')
    record = runpy.run_path(str(payload/'flux_hybrid_future.py'))['record_bf16']
    array = lambda value: record(out, value, arrays)
    conv = lambda value: value.detach().float().cpu().numpy()
    root = Path(c['model_path'])
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.safetensors', '.json', '.txt', '.model', '.tiktoken'}}
    if identity(inventory) != c['checkpoint_sha256']: raise ValueError('checkpoint identity differs')
    event('model_load_start')
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16, local_files_only=True, text_encoder=None, tokenizer=None)
    pipe.set_progress_bar_config(disable=True); counts['loads'] = 1
    for component in (pipe.transformer, pipe.vae): component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity='operation', residency='resident', device='cuda')
    store = LocalStore(out/'store'); original = LocalStore(data/'source/store')
    historical = {key: original.load(historical_plan['parents'][key], device='cuda') for key in ('base', 'edited')}
    if any(cut.model_identity != adapter.model_identity for cut in historical.values()): raise ValueError('source model differs')
    for cut in historical.values(): store.save(cut)
    block = pipe.transformer.transformer_blocks[1]
    r = runpy.run_path(str(payload/'flux_reentry_reference.py')); law = runpy.run_path(str(payload/'flux_receiver_norm_prediction.py'))
    math = runpy.run_path(str(payload/'flux_own_state_mlp_math.py')); program = runpy.run_path(str(payload/'flux_fresh_mlp_program.py'))
    for cls, key in ((type(block), 'Flux2TransformerBlock.source_sha256'), (type(block.attn.processor), 'Flux2AttnProcessor.source_sha256')):
        if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest() != r['SOURCE_CONTRACT'][key]: raise ValueError('native implementation differs')
    weights, _ = runpy.run_path(str(payload/'flux_response_campaign.py'))['parameter_recipes'](block, root, inventory, r, out, payload)
    def guard(module, args):
        if active['phase'] == 'forecast': raise ValueError('learned forward inside predictor')
    def entered(module, args):
        entries['blocks'] += 1
        if active['target']: entries['target_blocks'] += 1; targets[active['case']] += 1
    guards = [m.register_forward_pre_hook(guard) for component in (pipe.transformer, pipe.vae) for m in component.modules()]
    handles = [b.register_forward_pre_hook(entered) for b in [*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks]]
    def receipt(value):
        body = value.to_dict(); store.receipt(value); receipts.append(dict(branch=active['branch'], body=body))
        counts['receipts'] += 1; counts['native_transitions' if body['operation'] == 'continue' else 'acts'] += body.get('steps', 1)
        if any(counts[key] > expected_counts()[key] for key in ('native_transitions', 'acts', 'receipts')): raise ValueError('finite execution census exceeded')
    def capture(nav):
        cut = nav.session.capture(retain=False); store.save(cut); return cut
    def apply(nav, key, value, name):
        before = capture(nav); nav.apply(Act.replace(key, value, name=name)); after = capture(nav)
        if describe(after.payload) != describe({**before.payload, key: value}): raise ValueError('undeclared mutation')
        return after
    origin = historical['base'].payload
    active.update(phase='prefix', branch='fresh-prefix')
    with torch.inference_mode():
        latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels//4, 512, 512, torch.bfloat16, torch.device('cuda'), torch.Generator(device='cuda').manual_seed(c['seed']))
    counts['noise_draws'] = 1
    session = adapter.session(latent=latent.clone(), conditioning=origin['conditioning'].clone(), img_ids=img_ids.clone(), txt_ids=origin['txt_ids'].clone(), timesteps=origin['timesteps'].clone(), sigmas=origin['sigmas'].clone())
    initial = session.capture(retain=False); store.save(initial)
    nav = NativeCutNavigator(session, max_steps=58, on_receipt=receipt); nav.step(58); parent = capture(nav)
    if any(parent.execution_point[key] != value for key, value in dict(logical_step=2, phase='joint', index=1).items()) or entries['blocks'] != 51:
        raise ValueError('fresh prefix point/count differs')
    T = conv(parent.payload['text']); D = conv(historical['edited'].payload['text'])[:, [4, 11]].astype(np.float64)-conv(historical['base'].payload['text'])[:, [4, 11]].astype(np.float64)
    # D is wide FP64 and retained losslessly as a supplied operand, not a BF16 state.
    dpath = out/'historical-delta.npy'; np.save(dpath, D, allow_pickle=False)
    inputs = {spec['id']: program['make_text'](T, D, spec['doses'], r['bf16']) for spec in cases()}
    input_records = {name: array(value) for name, value in inputs.items()}
    save(out/'fresh-parent.json', dict(seed=c['seed'], initial=initial.fingerprint, parent=parent.fingerprint, historical_parents={key: cut.fingerprint for key, cut in historical.items()}, inputs=input_records, delta=dict(path=dpath.name, sha256=sha(dpath)), counts=dict(counts), entries=dict(entries)))
    event('fresh_prefix_complete')
    capture_ports = runpy.run_path(str(payload/'flux_precision_cut.py'))['capture_ports']
    def native_block(name, calibration=False):
        active.update(phase='calibration' if calibration else 'target', branch=name+'.native', case=name, target=not calibration)
        nav = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=54, on_receipt=receipt)
        entry = apply(nav, 'text', torch.from_numpy(inputs[name].copy()).to('cuda', torch.bfloat16), name+'.native.text')
        ports = {}
        try: capture_ports(block, native_module, T.shape[1], lambda key, value: ports.__setitem__(key, array(value)), entries, nav.step)
        finally: active['target'] = False
        exit = capture(nav)
        for role in ('image', 'text'): ports[role+'.exit'] = array(exit.payload[role])
        return nav, dict(entry=entry.fingerprint, exit=exit.fingerprint, stages=ports)
    anchor_nav, anchor = native_block('half', calibration=True)
    save(out/'calibration.json', anchor); event('calibration_frozen', calibration=anchor, programs=c['programs'])
    stages = {key: load_tensor_archive(out, item) for key, item in anchor['stages'].items()}
    context = {key: tuple(conv(v) for v in value) if isinstance(value, tuple) else conv(value) for key, value in parent.payload['context'].items()}
    active['phase'] = 'forecast'
    cache = law['prepare'](conv(parent.payload['image']), inputs['half'], context, weights, stages, r, absolute_ports=c['absolute_ports'], affine_policy=c['policy'], absolute_norms=('text.norm1', 'text.qk_rms'), attention_mode='anchored', receiver_norm='absolute')
    forecasts, costs = {}, {}
    for spec in cases():
        name = spec['id']; predicted, costs[name] = program['predict_pair'](cache, inputs[name], law, r, math)
        forecasts[name] = {mode: {role: {key: array(value) for key, value in fields.items()} for role, fields in roles.items()} for mode, roles in predicted.items()}
        event('calculated', case=name)
    del cache, stages, weights, predicted
    if entries != dict(blocks=52, target_blocks=0, dispatch=1) or any(targets.values()): raise ValueError('target ran before write seal')
    save(out/'forecasts.json', forecasts); save(out/'prediction-costs.json', costs)
    def transaction(directory, observed):
        return ForecastTransaction(out/directory, model_identity=adapter.model_identity, input_identity=identity(c), source_identity=identity(request['source_hashes']), frontier=dict(kind='architectural', observed=observed), execution_scope=c['scope'], budgets=dict(max_native_calls=1, max_wall_ms=c['worker_wall_s']*1000))
    tx = transaction('write-transaction', 'One native prefix and declared shared half-state calibration only; no two target blocks or target images observed.')
    tx.predict('writes', lambda _: dict(predictions=dict(forecasts=forecasts, counts_at_seal=dict(counts), entries_at_seal=dict(entries), targets=dict(targets))))
    write_seal = tx.seal(); event('writes_sealed', seal=write_seal); active['phase'] = 'consumer'
    steps = runpy.run_path(str(payload/'flux_response_validate.py'))['steps_to_commit']
    def fixed_readout(rgb):
        x = rgb.astype(np.float64)
        return {name: dict(mean_rgb=value.mean(axis=(0, 1)).tolist(), red_minus_blue=float((value[...,0]-value[...,2]).mean())) for name, value in [('whole', x), ('left', x[:, :256]), ('right', x[:, 256:]), ('border', np.concatenate([x[:32].reshape(-1, 1, 3), x[-32:].reshape(-1, 1, 3)]))]}
    def finish(nav, name, mode, installed, source_exit):
        active.update(branch=name+'.'+mode, target=False, phase='consumer')
        commits = []
        while nav.point['logical_step'] < 4:
            before = nav.point['logical_step']; nav.step(steps(nav.point, 5, 20))
            if nav.point['logical_step'] != before+1: raise ValueError('suffix commit differs')
            commits.append(capture(nav).fingerprint)
        final = capture(nav); png = out/(name+'.'+mode+'.png'); adapter.decode(nav.session, height=512, width=512).save(png); counts['images'] += 1
        rgb = np.asarray(Image.open(png).convert('RGB')).copy()
        row = dict(id=name, mode=mode, entry=installed, source_exit=source_exit, commits=commits, final=final.fingerprint, png=png.name, png_sha256=sha(png), readout=fixed_readout(rgb), targets_at_image=dict(targets))
        rows.append(row); save(out/('completed-'+name+'.'+mode+'.json'), row); event('image_complete', case=name, mode=mode)
    scaffold = store.load(anchor['exit'], device='cuda')
    def candidate_futures(_):
        for spec in cases():
            name = spec['id']
            for mode in ('old', 'new'):
                active.update(phase='consumer', branch=name+'.'+mode, target=False)
                nav = NativeCutNavigator(Session.from_cut(adapter, scaffold), max_steps=53, on_receipt=receipt)
                for role in ('image', 'text'):
                    value = load_tensor_archive(out, forecasts[name][mode][role]['exit'])
                    apply(nav, role, torch.from_numpy(value.copy()).to('cuda', torch.bfloat16), name+'.'+mode+'.'+role)
                installed = capture(nav)
                if any(describe(installed.payload[key]) != describe(scaffold.payload[key]) for key in scaffold.payload if key not in {'image', 'text'}): raise ValueError('candidate closure differs')
                finish(nav, name, mode, installed.fingerprint, scaffold.fingerprint)
        return dict(images=counts['images'], counts=dict(counts))
    with tx: tx.execute('six-candidate-futures', candidate_futures)
    pixel_forecasts = {row['id']+'.'+row['mode']: dict(png_sha256=row['png_sha256'], readout=row['readout']) for row in rows}
    save(out/'image-forecasts.json', pixel_forecasts)
    future = transaction('image-transaction', 'Six candidate images and fixed-region measurements known; native baseline image and two target blocks/images not observed. Later baseline-derived masks are scoring, not presealed numerical forecasts.')
    future.predict('pixels', lambda _: dict(predictions=dict(forecasts=pixel_forecasts, counts_at_seal=dict(counts), entries_at_seal=dict(entries), targets=dict(targets))))
    image_seal = future.seal(); event('image_forecasts_sealed', seal=image_seal)
    native = dict(half=anchor)
    def targets_and_baseline(_):
        active.update(branch='half.native', target=False)
        finish(anchor_nav, 'half', 'native', anchor['exit'], anchor['exit'])
        for spec in cases()[1:]:
            nav, value = native_block(spec['id']); native[spec['id']] = value
            finish(nav, spec['id'], 'native', value['exit'], value['exit'])
        return dict(images=counts['images'], counts=dict(counts))
    with future: future.execute('native-baseline-and-two-targets', targets_and_baseline)
    for handle in [*handles, *guards]: handle.remove()
    save(out/'native.json', native); save(out/'rows.json', rows); save(out/'receipts.json', receipts)
    if counts != expected_counts() or entries != dict(blocks=486, target_blocks=2, dispatch=3) or targets != dict(row4heavy=1, row11heavy=1): raise ValueError('final native census differs')
    summary = dict(schema='mdb-fresh-mlp-futures-v1', job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'], output_root=str(scratch), counts=counts, entries=entries, targets=targets,
        write_seal=write_seal, image_seal=image_seal, write_transaction=verify_transaction(tx.root), image_transaction=verify_transaction(future.root), seconds=time.monotonic()-started,
        unique_arrays=len(arrays), mechanics_status='pending-independent-audit', terminal_status='unassessed', scope=c['scope'])
    save(out/'summary.json', summary); save(out/'files.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'], request_sha256=request['request_sha256'], output_root=str(out), files_manifest_sha256=sha(out/'files.json'), summary_sha256=sha(out/'summary.json')), sort_keys=True), flush=True)


if __name__ == '__main__': worker()
