"""Saved-only matched-dose audit, compact publisher and finite CPU launcher.

audit NATIVE_ROOT MAIN_ROOT CONTEXT_ROOT OUTPUT_JSON
stage RUN --native COMPLETED_NATIVE_RUN
submit RUN
No models, encoders, noise draws or new image generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ('sham', 'A.half-context', 'A.context', 'A.mid', 'A.full', 'B.half', 'B.full', 'B.attribute-norm')
COUNTS = dict(loads=1, encodes=0, noise_draws=0, native_transitions=3584,
              native_blocks=3200, acts=128, receipts=3712, images=64)


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization_audit.py'))


def source_helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def mixed_rows(base, source, rows, alpha, h):
    """Independent CPU float64 difference -> FP32 -> BF16, with exact endpoints."""
    import numpy as np
    h['require'](base.dtype == source.dtype == np.uint16 and base.shape == source.shape
                 and base.ndim == 3 and base.size > 0, 'invalid conditioning geometry')
    h['require'](isinstance(rows, list) and rows and len(rows) == len(set(rows))
                 and all(type(i) is int and 0 <= i < base.shape[1] for i in rows), 'invalid dose row addresses')
    h['require'](isinstance(alpha, (int, float)) and math.isfinite(alpha), 'nonfinite dose alpha')
    left, right = base[:, rows], source[:, rows]
    if alpha == 0:
        return left.copy()
    if alpha == 1:
        return right.copy()
    b, s = [h['decode_bf16'](value).astype(np.float64) for value in (left, right)]
    h['require'](bool(np.isfinite(b).all() and np.isfinite(s).all()), 'nonfinite conditioning dose evidence')
    return h['round_bf16']((b + float(alpha) * (s - b)).astype(np.float32))


def delta_metrics(base_bits, actual_bits, h):
    import numpy as np
    h['require'](base_bits.shape == actual_bits.shape, 'dose comparison shape differs')
    base, actual = [h['decode_bf16'](value).astype(np.float64) for value in (base_bits, actual_bits)]
    h['require'](bool(np.isfinite(base).all() and np.isfinite(actual).all()), 'nonfinite conditioning dose')
    delta = actual - base
    words = int(np.count_nonzero(base_bits != actual_bits))
    return dict(l2=float(np.sqrt(np.sum(delta * delta))), rms=float(np.sqrt(np.mean(delta * delta))),
                elements=int(delta.size), different_bf16_elements=words,
                actually_different_payload_bytes=int(np.count_nonzero(np.ascontiguousarray(base_bits).view(np.uint8)
                                                               != np.ascontiguousarray(actual_bits).view(np.uint8))),
                storage_bytes_occupied_by_different_bf16_elements=2 * words)


def projected_text_dose(sham_bits, actual_bits, conditioning_l2, h):
    """Full landed T2 difference; equal C norms need not survive projection."""
    import numpy as np
    h['require'](sham_bits.ndim == 3 and sham_bits.shape == actual_bits.shape,
                 'projected text dose geometry differs')
    metrics = delta_metrics(sham_bits, actual_bits, h)
    numeric_difference = h['decode_bf16'](actual_bits) != h['decode_bf16'](sham_bits)
    metrics.update(reference='same-case sham full actual T2 at joint0 entry',
        bit_changed_token_rows=int(np.count_nonzero(np.any(sham_bits != actual_bits, axis=(0, 2)))),
        numerically_changed_token_rows=int(np.count_nonzero(np.any(numeric_difference, axis=(0, 2)))),
        token_rows=int(actual_bits.shape[1]),
        projection_gain_l2=metrics['l2'] / conditioning_l2 if conditioning_l2 else None,
        gain_status='defined' if conditioning_l2 else 'undefined-zero-conditioning-delta',
        interpretation='Observed BF16 context-embedder response; a matched C dose does not imply a matched T dose.')
    return metrics


def endpoint_comparison(evidence, row, source, source_row, h):
    h['require'](evidence.hashes[row['png']] == row['png_sha256']
                 and source.hashes[source_row['png']] == source_row['png_sha256'], 'replay PNG reference differs')
    _, actual_payload = evidence.cut(row['final_cut'])
    _, source_payload = source.cut(source_row['final_cut'])
    return dict(full_payload_exact=actual_payload == source_payload,
                final_latent=h['comparison'](evidence.bits(row['final_latent']), source.bits(source_row['final_latent'])),
                png_bytes_exact=row['png_sha256'] == source_row['png_sha256'],
                source_arm=source_row['arm'], source_cut=source_row['final_cut'])


def rgb_response(image, reference, natural_base, natural_source):
    """Signed response to a reference on the fixed natural source-minus-base axis."""
    import numpy as np
    result = {}
    for name, selection in (('whole', slice(None)), ('left', slice(0, image.shape[1] // 2)),
                            ('right', slice(image.shape[1] // 2, None))):
        response = image[:, selection].astype(np.float64) - reference[:, selection].astype(np.float64)
        axis = natural_source[:, selection].astype(np.float64) - natural_base[:, selection].astype(np.float64)
        power = float(np.sum(axis * axis))
        coefficient = float(np.sum(response * axis)) / power if power else None
        remainder = response - coefficient * axis if coefficient is not None else None
        result[name] = dict(signed_increment_on_natural_axis=coefficient,
            response_rms_0_255=float(np.sqrt(np.mean(response * response))),
            axis_orthogonal_response_rms_0_255=float(np.sqrt(np.mean(remainder * remainder))) if remainder is not None else None,
            interpretation='Geometric RGB diagnostic; orthogonal response is not a semantic collateral label.')
    return result


def analyze(native_root, main_root, context_root, output):
    import numpy as np
    from PIL import Image
    h = helpers()
    read, require = h['read'], h['require']
    roots = {key: h['find_root'](value) for key, value in
             (('native', native_root), ('main', main_root), ('context', context_root))}
    output = Path(output).resolve()
    require(not output.exists() and all(not output.is_relative_to(root) for root in roots.values()),
            'audit output must be new and outside all immutable source roots')
    inventories = {key: h['inventory'](root) for key, root in roots.items()}
    owners = {key: h['Evidence'](root, inventories[key][0]) for key, root in roots.items()}
    evidence, main, context = [owners[key] for key in ('native', 'main', 'context')]
    plan, request, summary, rows, fields, actions, doses = [read(evidence.path(name)) for name in
        ('plan.json', 'request.json', 'summary.json', 'rows.json', 'fields.json', 'actions.json', 'source-doses.json')]
    main_plan, main_rows, parents, addresses = [read(main.path(name)) for name in
        ('plan.json', 'rows.json', 'parents.json', 'token-addresses.json')]
    context_plan, context_rows = [read(context.path(name)) for name in ('plan.json', 'rows.json')]
    require(request['config'] == plan and summary['request_sha256'] == request['request_sha256'], 'request/config/summary join differs')
    require(plan['main_manifest_sha256'] == inventories['main'][1]['files_manifest_sha256']
            and plan['context_manifest_sha256'] == inventories['context'][1]['files_manifest_sha256']
            and plan['main_request_sha256'] == read(main.path('request.json'))['request_sha256'], 'source manifest/request binding differs')
    require(context_plan['source_manifest_sha256'] == plan['main_manifest_sha256'], 'context control is not joined to this main source')
    require(summary['counts'] == COUNTS, 'matched-dose native census differs')
    contexts = plan['contexts']
    require(contexts == main_plan['contexts'] == context_plan['contexts'] and len(contexts) == 8, 'source context set differs')
    cases = [case['id'] for case in contexts]
    require(len(set(cases)) == 8, 'duplicate context IDs')
    index = {(row['case'], row['arm']): row for row in rows}
    expected_keys = {(case, arm) for case in cases for arm in ARMS}
    require(len(rows) == len(index) == 64 and set(index) == expected_keys, 'native arm inventory differs')
    field_index = {(row['case'], row['arm'], row['field']): row for row in fields}
    require(len(fields) == len(field_index) == 128
            and set(field_index) == {(case, arm, f) for case, arm in expected_keys for f in ('image', 'text')},
            'projection inventory differs')
    main_index = {(row['case'], row['arm']): row for row in main_rows}
    context_index = {row['case']: row for row in context_rows}
    slot_bits = runpy.run_path(str(HERE / 'flux_binding_context_controls_audit.py'))['slot_bits']
    forecast_paths = [path for path in (roots['native'] / 'transaction/forecasts').glob('*.json')]
    require(len(forecast_paths) == 1, 'one global matched-dose forecast payload required')
    frozen = h['untag'](read(evidence.path(forecast_paths[0].relative_to(roots['native']).as_posix()))['predictions'])
    prefix = {key: int(key == 'loads') for key in COUNTS}
    require(summary['counts_at_seal'] == frozen['counts_at_seal'] == prefix, 'new native work preceded global dose seal')
    require(frozen['semantic'] == plan['hypotheses'], 'frozen semantic predictions differ')
    recipes = frozen['conditioning_recipes']
    require(recipes == doses, 'source dose metadata is not the frozen recipe')
    require(set(recipes) == set(cases) and set(frozen['projections']) == set(cases), 'sealed recipe/projection cases differ')
    timeline = [json.loads(line) for line in evidence.path('timeline.jsonl').read_text().splitlines()]
    seals = [i for i, row in enumerate(timeline) if row['kind'] == 'sealed']
    require(len(seals) == 1 and timeline[seals[0]]['counts'] == prefix, 'matched-dose seal chronology differs')
    completed = [(i, row['case'], row['arm']) for i, row in enumerate(timeline) if row['kind'] == 'image_complete']
    require(len(completed) == 64 and {(case, arm) for _, case, arm in completed} == expected_keys
            and min(i for i, _, _ in completed) > seals[0], 'a target image preceded seal or is missing')
    transaction_dirs = [path.parent for path in roots['native'].rglob('seal.json') if (path.parent / 'events').is_dir()]
    require(transaction_dirs == [roots['native'] / 'transaction'], 'expected one global transaction')
    try:
        from mdb_lanes.forecast_transaction import verify_transaction
    except ModuleNotFoundError as exc:
        if exc.name not in ('mdb', 'mdb_lanes.forecast_transaction'):
            raise
        transaction = dict(status='installed-verifier-unavailable', local_inventory_and_forecast_joins=True)
    else:
        transaction = verify_transaction(roots['native'] / 'transaction', expected_seal_sha256=h['sha'](roots['native'] / 'transaction/seal.json'))
        require(transaction['verified'] and transaction['phase'] == 'closed', 'dose transaction is not closed')
    require(len(actions) == 128, 'expected 128 conditioning Acts')
    projections, joins, footprints, measured_doses, controls, diagnostics = [], [], {}, {}, {}, {}

    def image(owner, row):
        require(owner.hashes[row['png']] == row['png_sha256'], 'PNG reference changed')
        with Image.open(owner.path(row['png'])) as opened:
            require(opened.mode == 'RGB' and opened.size == (512, 512), 'PNG geometry differs')
            return np.asarray(opened).copy()

    for case in cases:
        require(set(recipes[case]) == set(ARMS) and set(frozen['projections'][case]) == set(ARMS), 'sealed arm set differs')
        base = slot_bits(main, parents[case]['primed'], 'conditioning')
        source = slot_bits(main, parents[case]['source'], 'conditioning')
        _, original_payload = main.cut(parents[case]['primed'])
        base_image = image(main, main_index[case, 'natural.base'])
        source_image = image(main, main_index[case, 'natural.source'])
        i_image = image(main, main_index[case, 'I'])
        sham_text = evidence.bits(field_index[case, 'sham', 'text']['actual'])
        measured_doses[case], diagnostics[case], footprints[case], controls[case] = {}, {}, {}, {}
        for arm in ARMS:
            row, recipe = index[case, arm], recipes[case][arm]
            selected, alpha = recipe['rows'], recipe['alpha']
            require(row['source_parent'] == parents[case]['primed'], 'row parent differs from supplied main parent')
            expected = base.copy()
            expected[:, selected] = mixed_rows(base, source, selected, alpha, h)
            expected_rows = addresses['selectors'][case]['changed'] if arm.startswith('A.') or arm == 'sham' else context_plan['selections'][case]['rows']
            require(selected == expected_rows and selected == plan['selectors'][case][recipe['support']], 'dose support differs from declared A/B addresses')
            require(np.array_equal(evidence.bits(recipe['value_rows']), expected[:, selected]), 'frozen BF16 replacement differs from independent interpolation')
            fixed_alpha = {'sham': 0., 'A.full': 1., 'B.full': 1., 'B.half': .5}
            if arm in fixed_alpha:
                require(alpha == fixed_alpha[arm], 'fixed endpoint/half-dose alpha differs')
            if arm == 'A.mid':
                require(alpha == (recipes[case]['A.context']['alpha'] + 1.) / 2., 'midpoint alpha differs')
            for field in ('image', 'text'):
                item = field_index[case, arm, field]
                require(item['prediction'] == frozen['projections'][case][arm][field], 'comparison is not sealed prediction')
                actual, predicted = evidence.bits(item['actual']), evidence.bits(item['prediction'])
                require(list(actual.shape) == [1, 1024 if field == 'image' else 512, 3072], 'projected field is not complete')
                evidence.witness_join(item['actual'], row['marks']['joint0_entry'], field)
                projections.append(dict(case=case, arm=arm, field=field, **h['comparison'](predicted, actual)))
            joins.append(dict(case=case, arm=arm, **evidence.latent_join(row)))
            local = [action for action in actions if action['case'] == case and action['arm'] == arm]
            require(len(local) == 2, 'wrong per-arm C Act count')
            steps, landed = {}, None
            for action in local:
                before, before_payload = evidence.cut(action['before'])
                after, after_payload = evidence.cut(action['after'])
                point = before['execution_point']
                step = point['logical_step']
                require(action['slot'] == 'conditioning' and point['phase'] == 'project' and step in (2, 3)
                        and step not in steps, 'conditioning Act slot/clock differs')
                require(set(before_payload) == set(after_payload)
                        and all(before_payload[key] == after_payload[key] for key in before_payload if key != 'conditioning'),
                        'dose Act changed another payload slot')
                require(action['receipt']['parent'] == action['before'] and action['receipt']['result'] == action['after'], 'Act receipt/cut differs')
                actual = slot_bits(evidence, action['after'], 'conditioning')
                require(np.array_equal(actual, expected if step == 2 else base), 'dose recipe/base C reset differs')
                require(row['marks']['project' + str(step) + '_input'] == action['after'], 'project input mark differs')
                if step == 2:
                    require(before_payload == original_payload, 'dose future does not start at saved main primed parent')
                    landed = actual
                steps[step] = dict(typed_footprint_exact=True, conditioning_bits_exact=True, before=action['before'], after=action['after'])
            require(set(steps) == {2, 3}, 'project3 base conditioning reset absent')
            footprints[case][arm] = steps
            metrics = delta_metrics(base[:, selected], landed[:, selected], h)
            declared = doses[case][arm]
            measured_doses[case][arm] = dict(recipe=recipe, declared=declared, measured=metrics,
                measured_minus_declared_l2=metrics['l2'] - declared['landed_delta_l2'],
                projected_text_dose=projected_text_dose(sham_text, evidence.bits(field_index[case, arm, 'text']['actual']), metrics['l2'], h))
            observed_image = image(evidence, row)
            diagnostics[case][arm] = dict(natural_axis=h['rgb_diagnostics'](observed_image, base_image, source_image),
                matched_I_axis=h['rgb_diagnostics'](observed_image, i_image, source_image),
                matched_I_fixed_natural_axis=rgb_response(observed_image, i_image, base_image, source_image),
                minus_I_rms_0_255=float(np.sqrt(np.mean((observed_image.astype(np.float64) - i_image.astype(np.float64)) ** 2))))
            if arm in ('sham', 'A.full', 'B.full'):
                owner, reference = (context, context_index[case]) if arm == 'B.full' else (main, main_index[case, 'I' if arm == 'sham' else 'IC.early'])
                controls[case][arm] = endpoint_comparison(evidence, row, owner, reference, h)
        for arm, target in (('A.half-context', 'B.half'), ('A.context', 'B.full'), ('B.attribute-norm', 'A.full')):
            require(recipes[case][arm]['matched_to'] == target, 'matched pair designation differs')
            actual, target_norm = measured_doses[case][arm]['measured']['l2'], measured_doses[case][target]['measured']['l2']
            measured_doses[case][arm]['actual_matched_pair'] = dict(target_arm=target, target_l2=target_norm,
                signed_l2_difference=actual - target_norm, relative_l2_difference=(actual - target_norm) / target_norm if target_norm else None)
            text, target_text = [measured_doses[case][name]['projected_text_dose'] for name in (arm, target)]
            measured_doses[case][arm]['actual_matched_pair']['projected_text'] = dict(
                target_arm=target, arm_l2=text['l2'], target_l2=target_text['l2'],
                l2_ratio=text['l2'] / target_text['l2'] if target_text['l2'] else None,
                rms_ratio=text['rms'] / target_text['rms'] if target_text['rms'] else None,
                signed_l2_difference=text['l2'] - target_text['l2'],
                ratio_status='defined' if target_text['l2'] else 'undefined-zero-target-text-delta',
                interpretation='C-matched pair observed after anisotropic context projection; no equality threshold or address-specificity certification.')
    exact = sum(row['raw_bits_exact'] for row in projections)
    replay = all(row['full_payload_exact'] and row['final_latent']['raw_bits_exact'] and row['png_bytes_exact']
                 for pairs in controls.values() for row in pairs.values())
    primary_rank = {}
    for case in cases:
        a, b = [diagnostics[case][arm]['matched_I_fixed_natural_axis']['whole']['signed_increment_on_natural_axis']
                for arm in ('A.context', 'B.full')]
        primary_rank[case] = dict(primary_case=case in plan['hypotheses']['primary_cases'],
            A_context=a, B_full=b, signed_difference=a - b if a is not None and b is not None else None,
            predicted_order_observed=a > b if a is not None and b is not None else None,
            status='RGB rank observation; semantic assignment review remains separate')
    result = dict(schema='mdb-binding-matched-dose-audit-v1', mechanics_status='verified',
        instrument_status='source-replays-verified' if replay else 'source-replay-discrepancy-observed',
        numerical_status='all-compared-bits-exact' if exact == 128 else 'nonzero-discrepancies-observed',
        trend_status='observations-retained', terminal_status='unassessed', source_job=summary['job_id'],
        custody={key: value[1] for key, value in inventories.items()}, counts=COUNTS, transaction=transaction,
        projection_comparisons=128, exact_projection_comparisons=exact, projections=projections,
        final_native_joins=joins, conditioning_footprints=footprints, replay_controls=controls,
        doses=measured_doses, rgb_diagnostics=diagnostics, primary_rank_observations=primary_rank,
        hypotheses=plan['hypotheses'], new_model_calls=0, new_images=0,
        limitations=['Only projected image/text fields are numerical forecasts; RGB and attribute assignments are native observations.',
                    'Landed BF16 norms and matching errors are diagnostics, never scientific pass/fail gates.',
                    'Dose scalar/coordinate support is supplied; equal norm does not equate direction, content, or semantic potency.',
                    'Conditioning norms are matched before the anisotropic context embedder. Full actual T2 dose/gain and matched-pair ratios are measured separately; this is not a pure address-specificity test.',
                    'RGB projections and left/right halves are not semantic classifications or causal percentages.',
                    'Eight contexts reuse prior noise and primed paths; source-assisted hypotheses are not held-out generalization.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def worker():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    h = source_helpers()
    read, save, sha = [h[key] for key in ('read', 'save', 'sha')]
    request = read(os.environ['SATURN_DEBUG_REQUEST'])
    config = request['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(sha(Path(name)) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('CPU audit isolation/source binding differs')
    roots = {key: Path(config[key + '_root']) for key in ('native', 'main', 'context')}
    if any(sha(root / 'files.json') != config[key + '_manifest_sha256'] for key, root in roots.items()):
        raise ValueError('source inventory identity differs')
    out = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = analyze(roots['native'], roots['main'], roots['context'], out / 'audit.json')
    native = roots['native']
    for name in ('rows.json', 'fields.json', 'actions.json', 'source-doses.json', 'source-parents.json',
                 'source-token-addresses.json', 'native-contract.json',
                 'parameter-artifacts.json', 'retrieve_weights.py', 'retrieve_flux_encoder.py', 'source-binding.json'):
        shutil.copyfile(native / name, out / name)
    for name in ('plan.json', 'request.json', 'summary.json', 'timeline.jsonl'):
        shutil.copyfile(native / name, out / ('source-' + name))
    for row in read(native / 'rows.json'):
        target = out / row['png']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native / row['png'], target)
    shutil.copytree(native / 'transaction', out / 'source-transaction')
    save(out / 'external-evidence.json', dict(host='beast', roots={key: str(root) for key, root in roots.items()},
        manifests={key: read(root / 'files.json') for key, root in roots.items()},
        scope='Complete native/main/context inventories bind raw BF16, CAS and receipts retained near compute; individual checksum-verified retrieval.'))
    shutil.copyfile(HERE / 'retrieve_reader_evidence.py', out / 'retrieve_evidence.py')
    save(out / 'plan.json', config)
    save(out / 'request.json', request)
    save(out / 'summary.json', dict(schema='mdb-binding-matched-dose-compact-v1', job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=request['request_sha256'], output_root=str(out), source_job=report['source_job'],
        mechanics_status=report['mechanics_status'], seconds=time.monotonic() - start,
        new_execution_counts=dict(model_loads=0, native_primitives=0, images=0), terminal_status='unassessed'))
    save(out / 'files.json', {path.relative_to(out).as_posix(): sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'],
        output_root=str(out), files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


def stage(run, native):
    from mdb_lanes.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    h = source_helpers()
    read, save, sha = [h[key] for key in ('read', 'save', 'sha')]
    old = read(native / 'request.json')
    jid = read(native / 'submission.json')['job_id']
    _, raw, _ = scheduler_api().request('GET', '/api/jobs/' + jid, timeout_s=20)
    if json.loads(raw)['state'] != 'succeeded':
        raise ValueError('completed matched-dose native campaign required')
    roots = dict(native=Path(old['config']['output_scratch']) / jid / 'binding-minimal-result',
                 main=Path(old['config']['main_root']), context=Path(old['config']['context_root']))
    code = 'import json,hashlib\nfrom pathlib import Path\nroots=' + repr({key: str(root) for key, root in roots.items()}) + '\nprint(json.dumps({k:dict(sha=hashlib.sha256((Path(v)/"files.json").read_bytes()).hexdigest(),bytes=sum(p.stat().st_size for p in Path(v).rglob("*") if p.is_file()))for k,v in roots.items()}))\n'
    metadata = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code, text=True, capture_output=True, check=True).stdout)
    config = dict(output_scratch='/mnt/big/mdb/research', directory_custody=True)
    for key, root in roots.items():
        config[key + '_root'] = str(root)
        config[key + '_manifest_sha256'] = metadata[key]['sha']
        if key != 'native' and key + '_manifest_sha256' in old['config'] and config[key + '_manifest_sha256'] != old['config'][key + '_manifest_sha256']:
            raise ValueError('native source manifest changed: ' + key)
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'plan.json', config)
    names = ('flux_binding_generalization.py', 'flux_binding_generalization_audit.py', 'flux_binding_context_controls.py',
             'flux_binding_context_controls_audit.py', 'flux_binding_matched_dose.py', 'flux_family_custody.py', 'retrieve_reader_evidence.py')
    extras = [(HERE / name, name) for name in names]
    pre = run / 'preflight.v1.json'
    save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name, *names, pre.name],
                  path_token_allowlist=[*[str(root) for root in roots.values()], config['output_scratch'], 'binding-minimal-result']))
    extras.append((pre, pre.name))
    request = dict(schema='mdb-binding-matched-dose-audit-request-v1', worker_script=Path(__file__).name, config=config,
        execution_environment=old['execution_environment'], needs=dict(cpu=True, host='beast'), task_family='analysis',
        backend='numpy', device='cpu', dtype='float32', source_hashes={name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        resource_exception=dict(ram_mb=2048, vram_mb=0, cpu_threads=2, disk_gb=1, est_wall_s=180,
            reason='Saved-only128projectionfields/128CActfootprints/64PNGs; stream per arm, three complete inventories, no weights/framework hydration. Prior main saved-audit profile motivates2GBRAM/2CPU/180sestimate/480souter.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=64, logical_branches=1,
            logical_checkpoint_replays=0, max_local_work_items=128, extra=dict(task_family='analysis',
                logical_operation_unit='one saved arm audit containing two field comparisons, two Act checks, and one PNG observation',
                primitive_observation_counts=dict(projected_fields=128, conditioning_acts=128, images=64)),
            checkpoint_policy='Saved-only128sealedI/Tcomparisons,128CActfootprints/baseCresets,64nativePNGs,24replaycontrols,threeinventories.0models/nativeforwards/retries.'))
    request['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / 'request.json', request)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    stage_worker(run / 'payload', worker=Path(__file__), config=config, request=request, additional_files=extras)
    return dict(request_sha256=request['request_sha256'], sources=metadata)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    h = source_helpers()
    read, save, sha = [h[key] for key in ('read', 'save', 'sha')]
    request = read(run / 'request.json')
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    if (run / 'submission.json').exists() or any(sha(run / 'payload' / name) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('changed/already submitted saved audit')
    command = worker_command(request['worker_script'])
    command.insert(1, 'CUDA_VISIBLE_DEVICES=')
    command.insert(command.index('python') + 1, '-B')
    resources = {key: request['resource_exception'][key] for key in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')}
    job = guarded_mrun_launch(command, request=request, model=None, task='analysis', experiment='mdb-binding-matched-dose-audit',
        payload=run / 'payload', gpu=False, needs=request['needs'], prefer_host='beast', timeout_s=480,
        preflight=True, retry_on_kill=False, **resources,
        note='Saved-only matched-dose audit:128sealedI/Tfields,128CActfootprints/baseCresets,64existingPNGs,24source-replaycontrols,nativeBF16norms/threeinventories.0models/nativeforwards/newimages;CPU2/RAM2GB/180sestimate/480souter,no retries.')
    save(run / 'submission.json', dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1':
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest='mode', required=True)
        a = commands.add_parser('audit')
        for key in ('native_root', 'main_root', 'context_root', 'output'):
            a.add_argument(key, type=Path)
        s = commands.add_parser('stage'); s.add_argument('run', type=Path); s.add_argument('--native', type=Path, required=True)
        u = commands.add_parser('submit'); u.add_argument('run', type=Path)
        args = parser.parse_args()
        if args.mode == 'audit':
            report = analyze(args.native_root, args.main_root, args.context_root, args.output)
            print(json.dumps({key: report[key] for key in ('mechanics_status', 'projection_comparisons', 'exact_projection_comparisons', 'new_model_calls')}, sort_keys=True))
        else:
            print(json.dumps(stage(args.run.resolve(), args.native.resolve()) if args.mode == 'stage' else submit(args.run.resolve()), sort_keys=True))
