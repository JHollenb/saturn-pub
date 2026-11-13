"""Eight matched attribute-binding contexts, one native MDB execution lease.

Source-assisted causal transfer; observed conditioner/prefixes. No pixel forecast.
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
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def contexts():
    template = 'A {a} {x} to the {side} of a {b} {y} on a plain white background. Two separate objects, simple studio product photograph.'
    rows = []
    for name, seed, attrs, objects, side in (
        ('anchor', 109, ('red', 'blue'), ('sphere', 'cube'), 'left'),
        ('seed', 347, ('red', 'blue'), ('sphere', 'cube'), 'left'),
        ('wording', 109, ('red', 'blue'), ('sphere', 'cube'), 'left'),
        ('colors', 109, ('green', 'yellow'), ('sphere', 'cube'), 'left'),
        ('objects', 109, ('red', 'blue'), ('cone', 'cylinder'), 'left'),
        ('layout', 109, ('red', 'blue'), ('sphere', 'cube'), 'right'),
        ('material', 109, ('wooden', 'metal'), ('sphere', 'cube'), 'left'),
        ('texture', 109, ('striped', 'dotted'), ('sphere', 'cube'), 'left'),
    ):
        prompts = {}
        for role, (a, b) in zip(('base', 'source'), (attrs, attrs[::-1])):
            prompts[role] = (f'Studio photograph on white: a {b} cube stands to the right of a {a} sphere. Show both objects separately.'
                             if name == 'wording' else template.format(a=a, b=b, x=objects[0], y=objects[1], side=side))
        rows.append(dict(id=name, seed=seed, attributes=list(attrs), objects=list(objects), side=side,
                         prompts=prompts, attribute_class=name if name in ('material', 'texture') else 'color'))
    return rows


def arms():
    return [dict(id='base-replay', primed=False, early=False, late=False, restore=[]),
            dict(id='I', primed=True, early=False, late=False, restore=[]),
            dict(id='C', primed=False, early=True, late=False, restore=[]),
            dict(id='IC.early', primed=True, early=True, late=False, restore=[]),
            dict(id='IC.late', primed=True, early=False, late=True, restore=[]),
            dict(id='IC.control-rows', primed=True, early=True, late=False, control=True, restore=[]),
            dict(id='IC.restoreI', primed=True, early=True, late=False, restore=['image']),
            dict(id='IC.restoreT', primed=True, early=True, late=False, restore=['text']),
            dict(id='IC.restoreIT', primed=True, early=True, late=False, restore=['image', 'text'])]


def resolve_rows(base, source):
    """Deterministic paired-token resolver, frozen before any native image."""
    if len(base['ids']) != len(source['ids']):
        raise ValueError('padded token lengths differ')
    changed = [i for i, (a, b) in enumerate(zip(base['ids'], source['ids'])) if a != b]
    unchanged = [i for i, (a, b) in enumerate(zip(base['ids'], source['ids']))
                 if a == b and base['mask'][i] and source['mask'][i] and base['offsets'][i][1] > base['offsets'][i][0]]
    unchanged += [i for i, (a, b) in enumerate(zip(base['ids'], source['ids']))
                  if a == b and i not in unchanged]
    if not changed or len(unchanged) < len(changed):
        raise ValueError('changed/unchanged-row instrument incomplete')
    # First unchanged content rows, fixed before observing any image or carrier.
    return dict(changed=changed, control=unchanged[:len(changed)],
                valid_lengths=[sum(base['mask']), sum(source['mask'])],
                token_lengths_equal=sum(base['mask']) == sum(source['mask']),
                control_valid_rows=sum(bool(base['mask'][i] and source['mask'][i]) for i in unchanged[:len(changed)]),
                limitation='Changed-ID union may include shifted suffix tokens; unchanged-ID rows may carry contextual attributes. Control prefers valid content rows, then unchanged padding if insufficient.')


def expected_counts():
    return dict(loads=1, encodes=16, noise_draws=2, native_transitions=6008,
                native_blocks=5368, acts=184, receipts=6192, images=88)


def worker():
    boot = time.monotonic()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    payload = Path.cwd()
    request, config = [read(os.environ[k]) for k in ('SATURN_DEBUG_REQUEST', 'SATURN_RUN_CONFIG')]
    if request['config'] != config or any(sha(payload / n) != v for n, v in request['source_hashes'].items()):
        raise ValueError('source/request binding changed')
    if config['contexts'] != contexts() or config['arms'] != arms():
        raise ValueError('finite campaign changed')
    out = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', config)
    save(out / 'request.json', request)
    counts = {k: 0 for k in expected_counts()}
    active = {'case': 'setup', 'arm': 'setup', 'forecast': False}

    def event(kind, **fields):
        row = dict(kind=kind, seconds=time.monotonic() - boot, counts=counts.copy(), **fields)
        with (out / 'timeline.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        print('MDB_BINDING_GENERALIZATION ' + json.dumps(row, sort_keys=True), flush=True)
        if row['seconds'] > config['max_wall_seconds']:
            raise RuntimeError('finite campaign wall budget')

    event('setup')
    import importlib.metadata
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
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
            raise ValueError('original native source changed: ' + name)
    for name, digest in config['checkpoint_files'].items():
        if sha(Path(config['model_path']) / name) != digest:
            raise ValueError('original encoder/tokenizer changed: ' + name)
    # Validate original archive recipes without making another resident weight bank.
    runpy.run_path(str(payload / 'flux_source_receiver.py'))['RecipeBank'](read(payload / 'parameter-artifacts.json'), config['model_path'])

    def array(value):
        value = value.detach().contiguous().cpu()
        if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError('finite BF16 evidence required')
        bits = value.view(torch.uint16).numpy()
        digest = hashlib.sha256(bits.tobytes()).hexdigest()
        path = out / 'arrays' / (digest + '.npy')
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            np.save(path, bits, allow_pickle=False)
        return dict(path=path.relative_to(out).as_posix(), sha256=sha(path), shape=list(bits.shape),
                    dtype='bfloat16', storage_dtype='uint16')

    pipe = Flux2KleinPipeline.from_pretrained(config['model_path'], torch_dtype=torch.bfloat16, local_files_only=True)
    counts['loads'] += 1
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.eval().requires_grad_(False).to('cuda')
    encodings, tokens, selectors = {}, {}, {}
    with torch.inference_mode():
        for context in config['contexts']:
            cid = context['id']
            tokens[cid] = {}
            for role, prompt in context['prompts'].items():
                text = pipe.tokenizer.apply_chat_template([dict(role='user', content=prompt)], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False)
                t = pipe.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_offsets_mapping=True)
                spans = []
                for attr in context['attributes']:
                    start = text.index(attr, text.index(prompt))
                    spans.append(dict(attribute=attr, start=start, end=start+len(attr),
                                      rows=[i for i, (a, b) in enumerate(t['offset_mapping']) if a < start+len(attr) and b > start]))
                tokens[cid][role] = dict(ids=t['input_ids'], mask=t['attention_mask'], offsets=t['offset_mapping'],
                                         pieces=pipe.tokenizer.convert_ids_to_tokens(t['input_ids']), attribute_spans=spans)
            selectors[cid] = resolve_rows(tokens[cid]['base'], tokens[cid]['source'])
        save(out / 'token-addresses.json', dict(tokens=tokens, selectors=selectors, prediction='Paired-source changed-ID rows; no learned resolver.'))
        event('addresses_frozen')
        for context in config['contexts']:
            for role, prompt in context['prompts'].items():
                value, ids = pipe.encode_prompt(prompt, device=torch.device('cuda'), num_images_per_prompt=1, max_sequence_length=512)
                encodings[(context['id'], role)] = (value.cpu(), ids.cpu())
                counts['encodes'] += 1
            event('encoded', case=context['id'])
    pipe.text_encoder = None
    pipe.tokenizer = None
    del value, ids
    gc.collect()
    torch.cuda.empty_cache()
    for component in (pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False).to('cuda')
    pipe.vae.bn.to('cpu')
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity='operation', residency='resident', device='cuda')
    store = LocalStore(out / 'store')
    block_handles = []
    def block_entry(module, args):
        if active['forecast']:
            raise RuntimeError('target block entered during forecast')
        counts['native_blocks'] += 1
    for block in [*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks]:
        block_handles.append(block.register_forward_pre_hook(block_entry))
    rows, fields, writes, trajectories, actions = [], [], [], [], []
    roots, source_entries = {}, {}
    receipts = gzip.open(out / 'receipts.jsonl.gz', 'wt', compresslevel=1)

    def receipt(value):
        body = value.to_dict()
        key = 'native_transitions' if body['operation'] == 'continue' else 'acts'
        counts[key] += body.get('steps', 1)
        counts['receipts'] += 1
        if counts[key] > expected_counts()[key]:
            raise RuntimeError('finite native operation budget')
        receipts.write(json.dumps(dict(case=active['case'], arm=active['arm'], receipt=body), sort_keys=True) + '\n')

    def capture(nav):
        cut = nav.session.capture(retain=False)
        store.save(cut)
        return cut

    def apply(nav, slot, value, label):
        before = nav.session.capture(retain=False)
        act = Act.replace(slot, value, name=label)
        r = nav.apply(act)
        after = nav.session.capture(retain=False)
        if describe(after.payload) != describe({**before.payload, slot: value}):
            raise ValueError('undeclared typed write')
        actions.append(dict(case=active['case'], arm=active['arm'], slot=slot, label=label,
                            before=before.fingerprint, after=after.fingerprint, receipt=r.to_dict()))

    def finish(nav, cid, name, marks):
        final = capture(nav)
        png = cid + '.' + name + '.png'
        adapter.decode(nav.session, height=512, width=512).save(out / png)
        counts['images'] += 1
        row = dict(case=cid, arm=name, png=png, png_sha256=sha(out / png), final_cut=final.fingerprint,
                   final_latent=array(final.payload['latent']), marks=marks)
        rows.append(row)
        event('image_complete', case=cid, arm=name)
        return row

    with torch.inference_mode():
        noise = {}
        for seed in sorted({x['seed'] for x in config['contexts']}):
            latent, img_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                torch.bfloat16, torch.device('cuda'), torch.Generator(device='cuda').manual_seed(seed))
            noise[seed] = (latent.cpu(), img_ids.cpu())
            counts['noise_draws'] += 1
        timesteps = torch.tensor(config['schedule']['timesteps'], dtype=torch.float32, device='cuda')
        sigmas = torch.tensor(config['schedule']['sigmas'], dtype=torch.float32, device='cuda')
        for context in config['contexts']:
            cid = context['id']
            active.update(case=cid)
            for role in ('base', 'source'):
                active['arm'] = 'natural.' + role
                latent, img_ids = [v.to('cuda') for v in noise[context['seed']]]
                conditioning, txt_ids = [v.to('cuda') for v in encodings[(cid, role)]]
                nav = NativeCutNavigator(adapter.session(latent=latent.clone(), conditioning=conditioning, img_ids=img_ids,
                    txt_ids=txt_ids, timesteps=timesteps.clone(), sigmas=sigmas.clone()), max_steps=112, on_receipt=receipt)
                marks = {'initial': capture(nav).fingerprint}
                for step in range(112):
                    if step in (33, 56):
                        marks['prime_entry' if step == 33 else 'project2'] = capture(nav).fingerprint
                    nav.step()
                finish(nav, cid, 'natural.' + role, marks)
                roots[(cid, role)] = marks['project2']
                source_entries[(cid, role)] = marks['prime_entry']
                del nav
            active['arm'] = 'prime'
            parent = store.load(source_entries[(cid, 'base')], device='cuda')
            src = store.load(source_entries[(cid, 'source')], device='cuda')
            nav = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=23, on_receipt=receipt)
            apply(nav, 'image', src.payload['image'], 'prime.image')
            del parent, src
            for _ in range(23):
                nav.step()
            roots[(cid, 'primed')] = capture(nav).fingerprint
            if dict(nav.point)['phase'] != 'project' or nav.point['logical_step'] != 2:
                raise ValueError('primed clock differs')
            del nav
            event('context_prefixes_complete', case=cid)
        save(out / 'parents.json', {cid: {role: roots[(cid, role)] for role in ('base', 'source', 'primed')} for cid in selectors})
        before_forecast = counts.copy()
        predictions = {}
        tx = ForecastTransaction(out / 'transaction', model_identity=adapter.model_identity, input_identity=identity(config),
            source_identity=identity(request['source_hashes']), frontier=dict(kind='architectural',
            observed='16 original encoded contexts,16 honest natural paths and8 image-only primed prefixes; no new suffix entry.'),
            execution_scope='Externally admitted native CUDA BF16 B1; full projection forecasts use original Linear primitives only.',
            budgets=dict(max_native_calls=1, max_wall_ms=1080000))
        def forecast(_):
            active['forecast'] = True
            try:
                for context in config['contexts']:
                    cid = context['id']
                    base_c = encodings[(cid, 'base')][0].to('cuda')
                    source_c = encodings[(cid, 'source')][0].to('cuda')
                    for spec in config['arms']:
                        cut = store.load(roots[(cid, 'primed' if spec['primed'] else 'base')], device='cuda')
                        c = base_c.clone()
                        indices = selectors[cid]['control' if spec.get('control') else 'changed'] if spec['early'] else []
                        c[:, indices] = source_c[:, indices]
                        predictions[(cid, spec['id'])] = dict(image=array(pipe.transformer.x_embedder(cut.payload['latent'])),
                                                              text=array(pipe.transformer.context_embedder(c)))
                        del cut
            finally:
                active['forecast'] = False
            if counts != before_forecast:
                raise ValueError('native target entered forecast')
            return dict(predictions=dict(semantic=config['hypotheses'], counts_at_seal=before_forecast,
                projections={cid + '.' + arm: value for (cid, arm), value in predictions.items()},
                source_assisted_restore='I-only actual joint0 exit obtained after seal; declared fixed dependency, not a numerical forecast.'))
        tx.predict('all_future_predictions', forecast)
        seal = tx.seal()
        event('sealed', seal=seal)

        def execute(_):
            for context in config['contexts']:
                cid = context['id']
                active['case'] = cid
                matched = None
                base_c, source_c = [encodings[(cid, r)][0].to('cuda') for r in ('base', 'source')]
                for spec in config['arms']:
                    name = spec['id']
                    active['arm'] = name
                    parent = store.load(roots[(cid, 'primed' if spec['primed'] else 'base')], device='cuda')
                    nav = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=56, on_receipt=receipt)
                    del parent
                    marks = {}
                    for step in range(56):
                        point = dict(nav.point)
                        if point['phase'] == 'project':
                            c = base_c.clone()
                            pulse = spec['early'] if point['logical_step'] == 2 else spec['late']
                            indices = selectors[cid]['control' if spec.get('control') else 'changed'] if pulse else []
                            c[:, indices] = source_c[:, indices]
                            apply(nav, 'conditioning', c, name + '.C' + str(point['logical_step']))
                            if point['logical_step'] == 3:
                                marks['project3_reset'] = capture(nav).fingerprint
                        ports, hooks = {}, []
                        if step == 1:
                            entry = capture(nav)
                            marks['joint0_entry'] = entry.fingerprint
                            for field in ('image', 'text'):
                                fields.append(dict(case=cid, arm=name, field=field,
                                    prediction=predictions[(cid, name)][field], actual=array(entry.payload[field])))
                            block = pipe.transformer.transformer_blocks[0]
                            def attn_hook(module, args, output):
                                ports['attention'] = output[0].detach().clone()
                            def ff_hook(module, args, output):
                                ports['mlp_input'] = args[0].detach().clone()
                                ports['mlp'] = output.detach().clone()
                            hooks.extend((block.attn.register_forward_hook(attn_hook), block.ff.register_forward_hook(ff_hook)))
                            def qkv_hook(key):
                                return lambda module, args, output: ports.__setitem__(key, describe(output))
                            for key in ('q', 'k', 'v'):
                                hooks.append(getattr(block.attn, 'to_' + key).register_forward_hook(qkv_hook(key)))
                        try:
                            nav.step()
                        finally:
                            for hook in hooks:
                                hook.remove()
                        if ports:
                            actual = capture(nav)
                            marks['joint0_exit'] = actual.fingerprint
                            (_, _, ga), (_, _, gm) = Flux2Modulation.split(entry.payload['context']['img_mod'], 2)
                            values = dict(residual=entry.payload['image'], attention=ports['attention'], mlp=ports['mlp'],
                                          mlp_input=ports['mlp_input'], gate_attention=ga, gate_mlp=gm, actual=actual.payload['image'])
                            writes.append(dict(case=cid, arm=name, fields={k: array(v) for k, v in values.items()},
                                               qkv_descriptors={k: ports[k] for k in ('q', 'k', 'v')}))
                            if name == 'I':
                                matched = actual.fingerprint
                            if spec['restore']:
                                if matched is None:
                                    raise ValueError('declared matched restore dependency missing')
                                source_cut = store.load(matched, device='cuda')
                                for slot in spec['restore']:
                                    apply(nav, slot, source_cut.payload[slot], name + '.restore.' + slot)
                                marks['restoration_source'] = matched
                                marks['post_restore'] = capture(nav).fingerprint
                                del source_cut
                            del actual, entry, values, ports
                        if point['phase'] in ('project', 'scheduler'):
                            cut = nav.session.capture(retain=False)
                            trajectories.append(dict(case=cid, arm=name, point=point,
                                state={k: describe(cut.payload[k]) for k in ('latent', 'image', 'text', 'conditioning')}))
                    finish(nav, cid, name, marks)
                    del nav
                save(out / ('completed-' + cid + '.json'), dict(rows=[r for r in rows if r['case'] == cid]))
            return dict(counts=counts.copy(), rows=len(rows))

        with tx:
            tx.execute('native_futures', execute)
    receipts.close()
    for hook in block_handles:
        hook.remove()
    if counts != expected_counts():
        raise ValueError('physical execution census differs: ' + str(counts))
    for name, value in (('rows.json', rows), ('fields.json', fields), ('writes.json', writes),
                        ('actions.json', actions), ('trajectories.json', trajectories)):
        save(out / name, value)
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'source-binding.json', 'retrieve_flux_encoder.py'):
        shutil.copyfile(payload / name, out / name)
    save(out / 'native-contract.json', dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
         diffusers=importlib.metadata.version('diffusers'), dtype='bfloat16', batch_size=1,
         bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
         encoder_files=config['checkpoint_files'], shared_noise_seeds=sorted(noise)))
    save(out / 'summary.json', dict(schema='mdb-binding-generalization-v1', job_id=os.environ['MRUN_JOB_ID'],
         request_sha256=request['request_sha256'], output_root=str(out), counts=counts,
         counts_at_seal=before_forecast, transaction=verify_transaction(tx.root), seal=seal,
         seconds=time.monotonic()-boot, mechanics_status='pending-independent-audit', terminal_status='unassessed'))
    event('complete')
    save(out / 'files.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'],
          output_root=str(out), files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


if __name__ == '__main__':
    worker()
