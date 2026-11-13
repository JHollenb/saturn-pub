"""Saved-only direct-T route/payload factorial audit and compact CPU publisher.

audit NATIVE_ROOT MATCHED_ROOT MAIN_ROOT OUTPUT_JSON
stage RUN --native COMPLETED_NATIVE_RUN
submit RUN
Expected entry fields are source-assisted Act witnesses, not denoiser forecasts.
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
ARMS = ('sham', 'replay-A.full', 'replay-B.full', 'A@A', 'A@B', 'B@A', 'B@B')
CELLS = ('A@A', 'A@B', 'B@A', 'B@B')
REPLAYS = {'sham': 'sham', 'replay-A.full': 'A.full', 'replay-B.full': 'B.full', 'A@A': 'A.mid'}
COUNTS = dict(loads=1, encodes=0, noise_draws=0, native_transitions=1540,
              native_blocks=1400, acts=56, receipts=1596, images=28)


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization_audit.py'))


def source_helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def transport_rows(destination_base, source_base, source_value, alpha, h):
    """Independently transport ordered delta rows at the declared BF16 boundary."""
    import numpy as np
    values = (destination_base, source_base, source_value)
    h['require'](all(value.dtype == np.uint16 for value in values)
                 and destination_base.ndim == 3 and destination_base.size > 0
                 and all(value.shape == destination_base.shape for value in values), 'transport row geometry differs')
    h['require'](isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha >= 0, 'invalid payload amplitude')
    decoded = [h['decode_bf16'](value).astype(np.float64) for value in values]
    h['require'](all(bool(np.isfinite(value).all()) for value in decoded), 'nonfinite transport evidence')
    if alpha == 0:
        return destination_base.copy()
    if alpha == 1 and np.array_equal(destination_base, source_base):
        return source_value.copy()
    destination, baseline, source = decoded
    return h['round_bf16']((destination + float(alpha) * (source - baseline)).astype(np.float32))


def ordered_direction_comparison(left_base, left_value, right_base, right_value, h):
    """Compare decoded ordered-row deltas, rather than disjoint full supports."""
    import numpy as np
    values = (left_base, left_value, right_base, right_value)
    h['require'](all(value.dtype == np.uint16 and value.ndim == 3 for value in values)
                 and left_base.size > 0 and all(value.shape == left_base.shape for value in values),
                 'ordered direction comparison geometry differs')
    decoded = [h['decode_bf16'](value).astype(np.float64) for value in values]
    h['require'](all(bool(np.isfinite(value).all()) for value in decoded), 'nonfinite ordered direction evidence')
    left, right = decoded[1] - decoded[0], decoded[3] - decoded[2]
    norms = [float(np.sqrt(np.sum(value * value))) for value in (left, right)]
    denominator = norms[0] * norms[1]
    cosine = float(np.sum(left * right)) / denominator if denominator else None
    # Clip only numerical overshoot of the mathematical cosine range.
    if cosine is not None:
        cosine = max(-1., min(1., cosine))
    return dict(left_l2=norms[0], right_l2=norms[1],
        left_ordered_row_l2=[float(value) for value in np.sqrt(np.sum(left * left, axis=(0, 2)))],
        right_ordered_row_l2=[float(value) for value in np.sqrt(np.sum(right * right, axis=(0, 2)))],
        ordered_delta_cosine=cosine, cosine_status='defined' if denominator else 'undefined-zero-delta-direction',
        left_to_right_l2_ratio=norms[0] / norms[1] if norms[1] else None,
        scope='Ordered two-row deltas aligned row1-to-row1 and row2-to-row2; absolute support positions are not part of this cosine.')


def factorial_metrics(images, sham, natural_base, natural_source, rgb_response):
    """2x2 finite differences, with payload first and destination second."""
    import numpy as np
    responses = {cell: images[cell].astype(np.float64) - sham.astype(np.float64) for cell in CELLS}
    arrays = dict(
        payload_effect_at_A=responses['A@A'] - responses['B@A'],
        payload_effect_at_B=responses['A@B'] - responses['B@B'],
        destination_effect_for_A=responses['A@A'] - responses['A@B'],
        destination_effect_for_B=responses['B@A'] - responses['B@B'],
        interaction=responses['A@A'] - responses['A@B'] - responses['B@A'] + responses['B@B'])
    scalar_cells = {cell: rgb_response(images[cell], sham, natural_base, natural_source) for cell in CELLS}
    result = dict(cells=scalar_cells, contrasts={},
                  convention='First symbol is source payload direction; second is destination support. I=Y_AA-Y_AB-Y_BA+Y_BB.',
                  scope='Observed native RGB finite differences, not causal percentages or semantic assignment labels.')
    for name, delta in arrays.items():
        result['contrasts'][name] = rgb_response(delta, np.zeros_like(delta), natural_base, natural_source)
    return result


def analyze(native_root, prior_root, main_root, output):
    import numpy as np
    from PIL import Image
    h = helpers()
    read, require = h['read'], h['require']
    mh = runpy.run_path(str(HERE / 'flux_binding_matched_dose_audit.py'))
    slot_bits = runpy.run_path(str(HERE / 'flux_binding_context_controls_audit.py'))['slot_bits']
    roots = {key: h['find_root'](value) for key, value in
             (('native', native_root), ('prior', prior_root), ('main', main_root))}
    output = Path(output).resolve()
    require(not output.exists() and all(not output.is_relative_to(root) for root in roots.values()),
            'audit output must be new and outside all immutable source roots')
    inventories = {key: h['inventory'](root) for key, root in roots.items()}
    owners = {key: h['Evidence'](root, inventories[key][0]) for key, root in roots.items()}
    native, prior, main = [owners[key] for key in ('native', 'prior', 'main')]
    plan, request, summary, rows, fields, actions, recipes, parents, ports = [read(native.path(name)) for name in
        ('plan.json', 'request.json', 'summary.json', 'rows.json', 'fields.json', 'actions.json',
         'source-recipes.json', 'source-parents.json', 'ports.json')]
    prior_plan, prior_rows, prior_fields, prior_request = [read(prior.path(name)) for name in
        ('plan.json', 'rows.json', 'fields.json', 'request.json')]
    main_plan, main_rows = [read(main.path(name)) for name in ('plan.json', 'rows.json')]
    require(request['config'] == plan and summary['request_sha256'] == request['request_sha256'], 'request/config/summary join differs')
    require(plan['prior_manifest_sha256'] == inventories['prior'][1]['files_manifest_sha256']
            and plan['main_manifest_sha256'] == inventories['main'][1]['files_manifest_sha256']
            and plan['prior_request_sha256'] == prior_request['request_sha256'], 'source manifest/request binding differs')
    require(prior_plan['main_manifest_sha256'] == plan['main_manifest_sha256'], 'prior/main join differs')
    require(summary['counts'] == COUNTS, 'route/payload execution census differs')
    contexts = plan['contexts']
    cases = [case['id'] for case in contexts]
    require(len(contexts) == len(set(cases)) == 4 and set(cases) == {'anchor', 'seed', 'wording', 'objects'}, 'four declared historical cases required')
    prior_contexts, main_contexts = [{row['id']: row for row in item['contexts']} for item in (prior_plan, main_plan)]
    require(all(case == prior_contexts[case['id']] == main_contexts[case['id']] for case in contexts), 'historical contexts differ')
    expected_keys = {(case, arm) for case in cases for arm in ARMS}
    index = {(row['case'], row['arm']): row for row in rows}
    field_index = {(row['case'], row['arm']): row for row in fields}
    port_index = {(row['case'], row['arm']): row for row in ports}
    require(len(rows) == len(index) == len(fields) == len(field_index) == len(ports) == len(port_index) == 28
            and set(index) == set(field_index) == set(port_index) == expected_keys, 'row/entry-witness/port inventory differs')
    prior_index = {(row['case'], row['arm']): row for row in prior_rows}
    prior_field_index = {(row['case'], row['arm'], row['field']): row for row in prior_fields}
    main_index = {(row['case'], row['arm']): row for row in main_rows}
    require(set(parents) == set(recipes) == set(cases), 'source parent/recipe cases differ')
    forecast_paths = list((roots['native'] / 'transaction/forecasts').glob('*.json'))
    require(len(forecast_paths) == 1, 'one global entry-witness seal required')
    frozen = h['untag'](read(native.path(forecast_paths[0].relative_to(roots['native']).as_posix()))['predictions'])
    prefix = {key: int(key == 'loads') for key in COUNTS}
    require(summary['counts_at_seal'] == frozen['counts_at_seal'] == prefix, 'new native work preceded global witness seal')
    require(frozen['recipes'] == recipes and frozen['semantic'] == plan['hypotheses'], 'frozen recipe/hypothesis binding differs')
    timeline = [json.loads(line) for line in native.path('timeline.jsonl').read_text().splitlines()]
    seals = [i for i, row in enumerate(timeline) if row['kind'] == 'sealed']
    require(len(seals) == 1 and timeline[seals[0]]['counts'] == prefix, 'witness seal chronology differs')
    completed = [(i, row['case'], row['arm']) for i, row in enumerate(timeline) if row['kind'] == 'image_complete']
    require(len(completed) == 28 and {(case, arm) for _, case, arm in completed} == expected_keys
            and min(i for i, _, _ in completed) > seals[0], 'target image preceded seal or is missing')
    transaction_dirs = [path.parent for path in roots['native'].rglob('seal.json') if (path.parent / 'events').is_dir()]
    require(transaction_dirs == [roots['native'] / 'transaction'], 'expected one global transaction')
    try:
        from mdb_lanes.forecast_transaction import verify_transaction
    except ModuleNotFoundError as exc:
        if exc.name not in ('mdb', 'mdb_lanes.forecast_transaction'):
            raise
        transaction = dict(status='installed-verifier-unavailable', local_inventory_and_entry_joins=True)
    else:
        transaction = verify_transaction(roots['native'] / 'transaction', expected_seal_sha256=h['sha'](roots['native'] / 'transaction/seal.json'))
        require(transaction['verified'] and transaction['phase'] == 'closed', 'entry transaction is not closed')
    require(len(actions) == 56, 'expected 56 typed Acts')
    witnesses, joins, footprints, doses, controls, rgb, factorials, port_deltas, ranks, source_directions = [], [], {}, {}, {}, {}, {}, {}, {}, {}

    def load_image(owner, row):
        require(owner.hashes[row['png']] == row['png_sha256'], 'PNG reference changed')
        with Image.open(owner.path(row['png'])) as opened:
            require(opened.mode == 'RGB' and opened.size == (512, 512), 'PNG geometry differs')
            return np.asarray(opened).copy()

    def prior_text(case, arm):
        reference = prior_field_index[case, arm, 'text']['actual']
        prior.witness_join(reference, prior_index[case, arm]['marks']['joint0_entry'], 'text')
        return prior.bits(reference)

    for case in cases:
        require(set(recipes[case]) == set(ARMS), 'sealed arm set differs')
        parent = prior_index[case, 'sham']['marks']['joint0_entry']
        require(parents[case] == parent, 'supplied parent differs from prior sham joint0 entry')
        parent_cut, parent_payload = prior.cut(parent)
        require(parent_cut['execution_point']['logical_step'] == 2
                and parent_cut['execution_point']['phase'] == 'joint' and parent_cut['execution_point']['index'] == 0,
                'parent is not step2 joint0 entry')
        base_t, base_i, base_c = [slot_bits(prior, parent, slot) for slot in ('text', 'image', 'conditioning')]
        require(np.array_equal(base_t, prior_text(case, 'sham')), 'sham T field is not the source parent')
        selectors = prior_plan['selectors'][case]
        require(plan['selectors'][case] == selectors, 'source support selectors differ')
        source_texts = {'A': prior_text(case, 'A.mid'), 'B': prior_text(case, 'B.full')}
        for payload, value in source_texts.items():
            complement = [i for i in range(base_t.shape[1]) if i not in selectors[payload]]
            require(np.array_equal(value[:, complement], base_t[:, complement]), 'source payload extends outside declared rows')
        source_directions[case] = dict(A_source_arm='A.mid', B_source_arm='B.full',
            A_rows=selectors['A'], B_rows=selectors['B'],
            comparison=ordered_direction_comparison(base_t[:, selectors['A']], source_texts['A'][:, selectors['A']],
                                                   base_t[:, selectors['B']], source_texts['B'][:, selectors['B']], h))
        target = mh['delta_metrics'](base_t, source_texts['A'], h)['l2']
        natural_base = load_image(main, main_index[case, 'natural.base'])
        natural_source = load_image(main, main_index[case, 'natural.source'])
        sham_rgb = load_image(native, index[case, 'sham'])
        images = {}
        doses[case], footprints[case], controls[case], rgb[case], port_deltas[case] = {}, {}, {}, {}, {}
        for arm in ARMS:
            row, entry, recipe, port = index[case, arm], field_index[case, arm], recipes[case][arm], port_index[case, arm]
            require(row['source_parent'] == parent, 'branch does not start at common parent')
            selected, alpha = recipe['rows'], recipe['alpha']
            support = 'A' if arm == 'sham' or arm == 'replay-A.full' else 'B' if arm == 'replay-B.full' else arm[-1]
            require(recipe['arm'] == arm and recipe['support'] == support and selected == selectors[support], 'destination support differs')
            require(len(selected) == 2, 'expected two ordered destination rows')
            if arm in ('sham', 'replay-A.full', 'replay-B.full', 'A@A'):
                reference = REPLAYS[arm]
                require(recipe['source_arm'] == reference and recipe['payload'] == ('none' if arm == 'sham' else support), 'exact entry source designation differs')
                expected = prior_text(case, reference)
                require(alpha == (0. if arm == 'sham' else 1.), 'exact entry endpoint alpha differs')
            else:
                payload = arm[0]
                require(recipe['payload'] == payload and recipe['source_arm'] == ('A.mid' if payload == 'A' else 'B.full'), 'source payload designation differs')
                source_rows = selectors[payload]
                expected = base_t.copy()
                expected[:, selected] = transport_rows(base_t[:, selected], base_t[:, source_rows], source_texts[payload][:, source_rows], alpha, h)
            require(np.array_equal(expected[:, selected], native.bits(recipe['value_rows'])), 'frozen ordered replacement differs from independent delta transport')
            untouched = [i for i in range(base_t.shape[1]) if i not in selected]
            require(np.array_equal(expected[:, untouched], base_t[:, untouched]), 'entry changed undeclared text rows')
            require(entry['entry_cut'] == row['marks']['entry'], 'entry witness/cut mark differs')
            for field, expected_bits in (('text', expected), ('image', base_i)):
                ref, actual_ref = entry['expected_' + field], entry['actual_' + field]
                require(ref == recipe['entry_' + field], 'entry reference is not the frozen witness')
                sealed, actual = native.bits(ref), native.bits(actual_ref)
                require(list(actual.shape) == [1, 512 if field == 'text' else 1024, 3072], 'entry witness is not full shape')
                require(np.array_equal(sealed, expected_bits), 'sealed entry witness differs from supplied-state reconstruction')
                native.witness_join(actual_ref, entry['entry_cut'], field)
                witnesses.append(dict(case=case, arm=arm, field=field, evidence_kind='supplied_state_entry_witness', **h['comparison'](sealed, actual)))
            joins.append(dict(case=case, arm=arm, **native.latent_join(row)))
            local = [action for action in actions if action['case'] == case and action['arm'] == arm]
            require(len(local) == 2, 'expected entry T Act and project3 C reset')
            seen = set()
            for action in local:
                before, before_payload = native.cut(action['before'])
                after, after_payload = native.cut(action['after'])
                slot, point = action['slot'], before['execution_point']
                require(slot in ('text', 'conditioning') and slot not in seen, 'typed write slot/duplicate differs')
                require(set(before_payload) == set(after_payload)
                        and all(before_payload[key] == after_payload[key] for key in before_payload if key != slot), 'Act changed another payload slot')
                require(action['receipt']['parent'] == action['before'] and action['receipt']['result'] == action['after'], 'Act receipt/cut join differs')
                if slot == 'text':
                    require(before_payload == parent_payload and point == parent_cut['execution_point'], 'T Act is not at supplied common parent')
                    require(action['after'] == row['marks']['entry'], 'entry mark is not actual T Act result')
                    require(np.array_equal(slot_bits(native, action['after'], 'text'), expected), 'actual installed T differs from frozen recipe')
                else:
                    require(point['logical_step'] == 3 and point['phase'] == 'project', 'C reset is at wrong clock')
                    require(before_payload['conditioning'] == parent_payload['conditioning'], 'conditioning changed before its base reset')
                    require(action['after'] == row['marks']['project3_input'], 'C reset mark differs')
                    require(np.array_equal(slot_bits(native, action['after'], 'conditioning'), base_c), 'project3 conditioning is not base')
                seen.add(slot)
            require(seen == {'text', 'conditioning'}, 'missing T mutation/C reset')
            footprints[case][arm] = dict(text_only_entry=True, base_conditioning_preserved=True, base_conditioning_reset_at_project3=True)
            measured = mh['delta_metrics'](base_t, native.bits(entry['actual_text']), h)
            doses[case][arm] = dict(recipe=recipe, measured=measured,
                measured_minus_declared_l2=measured['l2'] - recipe['landed_delta_l2'],
                source_A_mid_target_l2=target, match_target_applies=arm in CELLS,
                relative_to_common_target_difference=(measured['l2'] - target) / target if target else None,
                factorial_match_relative_error=(measured['l2'] - target) / target if target and arm in CELLS else None)
            if arm in CELLS:
                payload = arm[0]
                source_rows = selectors[payload]
                doses[case][arm]['installed_vs_intended_ordered_direction'] = dict(
                    installed_destination_rows=selected, intended_source_rows=source_rows,
                    intended_source_arm='A.mid' if payload == 'A' else 'B.full',
                    comparison=ordered_direction_comparison(base_t[:, selected], native.bits(entry['actual_text'])[:, selected],
                                                           base_t[:, source_rows], source_texts[payload][:, source_rows], h),
                    interpretation='Observed unscaled source direction versus actual landed BF16 delta; amplitude and cosine distortion are diagnostics, not gates.')
            require(port['observed_only'] is True and port['before_entry'] == row['marks']['entry']
                    and port['after_joint0'] == row['marks']['after_joint0'], 'port capture/cut binding differs')
            after_joint0, after_joint0_payload = native.cut(row['marks']['after_joint0'])
            point = after_joint0['execution_point']
            require(point['logical_step'] == 2 and point['phase'] == 'joint' and point['index'] == 1,
                    'observed reader output is not the joint0 exit')
            require(after_joint0_payload['conditioning'] == parent_payload['conditioning'], 'reader changed base conditioning')
            required_ports = {'normalized_text', 'text_q', 'text_k', 'text_v', 'attention_image', 'attention_text', 'joint0_image', 'joint0_text'}
            require(required_ports <= set(port['values']), 'required observed port capture missing')
            for name, ref in port['values'].items():
                actual = native.bits(ref)
                sham = native.bits(port_index[case, 'sham']['values'][name])
                port_deltas[case].setdefault(arm, {})[name] = mh['delta_metrics'](sham, actual, h)
                if name in ('joint0_image', 'joint0_text'):
                    native.witness_join(ref, row['marks']['after_joint0'], name.removeprefix('joint0_'))
            observed_image = load_image(native, row)
            images[arm] = observed_image
            rgb[case][arm] = dict(natural_axis=h['rgb_diagnostics'](observed_image, natural_base, natural_source),
                relative_to_sham=mh['rgb_response'](observed_image, sham_rgb, natural_base, natural_source))
            if arm in REPLAYS:
                controls[case][arm] = mh['endpoint_comparison'](native, row, prior, prior_index[case, REPLAYS[arm]], h)
        factorials[case] = factorial_metrics(images, sham_rgb, natural_base, natural_source, mh['rgb_response'])
        cell_scalar = {cell: factorials[case]['cells'][cell]['whole']['signed_increment_on_natural_axis'] for cell in CELLS}
        ranks[case] = {cell: dict(A_at_A=cell_scalar['A@A'], alternative=cell_scalar[cell],
            signed_difference=cell_scalar['A@A'] - cell_scalar[cell] if cell_scalar['A@A'] is not None and cell_scalar[cell] is not None else None,
            predicted_order_observed=cell_scalar['A@A'] > cell_scalar[cell] if cell_scalar['A@A'] is not None and cell_scalar[cell] is not None else None)
            for cell in ('A@B', 'B@A', 'B@B')}
    exact = sum(row['raw_bits_exact'] for row in witnesses)
    replay = all(row['full_payload_exact'] and row['final_latent']['raw_bits_exact'] and row['png_bytes_exact'] for pairs in controls.values() for row in pairs.values())
    result = dict(schema='mdb-binding-route-payload-audit-v1', mechanics_status='verified',
        instrument_status='known-replays-verified' if replay else 'known-replay-discrepancy-observed',
        entry_witness_status='all-supplied-entry-bits-exact' if exact == 56 else 'entry-discrepancies-observed',
        trend_status='observations-retained', terminal_status='unassessed', source_job=summary['job_id'],
        custody={key: value[1] for key, value in inventories.items()}, counts=COUNTS, transaction=transaction,
        entry_witness_comparisons=56, exact_entry_witness_comparisons=exact, text_entry_witnesses=28, unchanged_image_witnesses=28,
        prospective_denoiser_equation_forecasts=0, entry_witnesses=witnesses, final_native_joins=joins,
        typed_footprints=footprints, doses=doses, replay_controls=controls, rgb_diagnostics=rgb,
        factorials=factorials, ordinal_predictions=ranks, observed_port_deltas=port_deltas,
        source_directions=source_directions, hypotheses=plan['hypotheses'],
        new_model_calls=0, new_images=0,
        limitations=['Sealed T fields are source-assisted entry Act witnesses; unchanged I is a supplied-state invariant. Neither forecasts a learned denoiser computation or RGB.',
                    '16 known continuation replays include all4 A@A factorial reference cells;12 crossed/scaled suffix futures are new.',
                    'Payload direction and destination support are crossed at one T boundary, with actual landed BF16 norm residuals retained.',
                    'A destination includes its own receiving baseline/context and positional treatment; location effects do not establish an exclusive semantic address.',
                    'Normalized/QKV/attention ports are observed native diagnostics, not independent numerical forecasts.',
                    'RGB finite differences, ordinal ranks and orthogonal responses are not semantic assignment classifiers or causal percentages.',
                    'Four source-assisted historical contexts reuse prior seeds/images; no held-out or automatic address-induction claim.'])
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
        raise ValueError('saved-only audit isolation/source binding differs')
    roots = {key: Path(config[key + '_root']) for key in ('native', 'prior', 'main')}
    if any(sha(root / 'files.json') != config[key + '_manifest_sha256'] for key, root in roots.items()):
        raise ValueError('source inventory identity differs')
    out = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = analyze(roots['native'], roots['prior'], roots['main'], out / 'audit.json')
    native = roots['native']
    # Keep compact source metadata by exact bytes, and leave raw arrays/CAS near compute.
    for path in sorted(native.iterdir()):
        if not path.is_file() or path.name == 'files.json':
            continue
        if path.suffix not in ('.json', '.jsonl'):
            continue
        name = 'source-' + path.name if path.name in ('plan.json', 'request.json', 'summary.json', 'timeline.jsonl') else path.name
        shutil.copyfile(path, out / name)
    for name in ('retrieve_weights.py', 'retrieve_flux_encoder.py'):
        shutil.copyfile(native / name, out / name)
    for row in read(native / 'rows.json'):
        target = out / row['png']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native / row['png'], target)
    shutil.copytree(native / 'transaction', out / 'source-transaction')
    save(out / 'external-evidence.json', dict(host='beast', roots={key: str(root) for key, root in roots.items()},
        manifests={key: read(root / 'files.json') for key, root in roots.items()},
        scope='Full native/prior/main inventories bind BF16 arrays, CAS and receipts retained near compute; individual checksum-verified retrieval.'))
    shutil.copyfile(HERE / 'retrieve_reader_evidence.py', out / 'retrieve_evidence.py')
    save(out / 'plan.json', config)
    save(out / 'request.json', request)
    save(out / 'summary.json', dict(schema='mdb-binding-route-payload-compact-v1', job_id=os.environ['MRUN_JOB_ID'],
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
        raise ValueError('completed route/payload native campaign required')
    roots = dict(native=Path(old['config']['output_scratch']) / jid / 'binding-minimal-result',
                 prior=Path(old['config']['prior_root']), main=Path(old['config']['main_root']))
    code = 'import json,hashlib\nfrom pathlib import Path\nroots=' + repr({key: str(root) for key, root in roots.items()}) + '\nprint(json.dumps({k:dict(sha=hashlib.sha256((Path(v)/"files.json").read_bytes()).hexdigest(),bytes=sum(p.stat().st_size for p in Path(v).rglob("*") if p.is_file()))for k,v in roots.items()}))\n'
    metadata = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code, text=True, capture_output=True, check=True).stdout)
    config = dict(output_scratch='/mnt/big/mdb/research', directory_custody=True)
    for key, root in roots.items():
        config[key + '_root'] = str(root)
        config[key + '_manifest_sha256'] = metadata[key]['sha']
        if key != 'native' and config[key + '_manifest_sha256'] != old['config'][key + '_manifest_sha256']:
            raise ValueError('native source inventory changed: ' + key)
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'plan.json', config)
    names = ('flux_binding_generalization.py', 'flux_binding_generalization_audit.py', 'flux_binding_context_controls.py',
             'flux_binding_context_controls_audit.py', 'flux_binding_matched_dose_audit.py',
             'flux_binding_route_payload.py', 'flux_binding_route_payload_math.py', 'flux_family_custody.py', 'retrieve_reader_evidence.py')
    extras = [(HERE / name, name) for name in names]
    pre = run / 'preflight.v1.json'
    save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name, *names, pre.name],
                  path_token_allowlist=[*[str(root) for root in roots.values()], config['output_scratch'], 'binding-minimal-result']))
    extras.append((pre, pre.name))
    request = dict(schema='mdb-binding-route-payload-audit-request-v1', worker_script=Path(__file__).name, config=config,
        execution_environment=old['execution_environment'], needs=dict(cpu=True, host='beast'), task_family='analysis',
        backend='numpy', device='cpu', dtype='float32', source_hashes={name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        resource_exception=dict(ram_mb=1536, vram_mb=0, cpu_threads=2, disk_gb=1, est_wall_s=90,
            calibration_job='job-b2c3dd9b831f', measured_peak_rss_mb=472.2, measured_wall_s=48.36,
            reason='Saved-only28arms:56entrywitnesses/56Acts/28PNGs/16replays plus observed port deltas, three inventories; stream per arm/port. Prior64arm saved-audit472.2MB/48.36s;1.5GBRAM/2CPU/90sestimate/300souter.0models.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=28, logical_branches=1,
            logical_checkpoint_replays=0, max_local_work_items=64, extra=dict(task_family='analysis',
                unit='one saved native arm, including two full entry witnesses, two typed Acts, final join, PNG and optional observed-port diagnostics'),
            checkpoint_policy='Saved-only28per-arm audit units include56entrywitnesses/56Acts/28PNGs/16knownreplays, source-only delta reconstruction and RGB2x2.0models/nativeforwards/retries.'))
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
    job = guarded_mrun_launch(command, request=request, model=None, task='analysis', experiment='mdb-binding-route-payload-audit',
        payload=run / 'payload', gpu=False, needs=request['needs'], prefer_host='beast', timeout_s=300,
        preflight=True, retry_on_kill=False, **resources,
        note='Saved-only route/payload audit28arm units:56suppliedentrywitnesses/56typedActs/28existingPNGs/16knownreplays; independent BF16transport/norms, RGB2x2 andobservedports, threeinventories.0models/nativeforwards/newimages;CPU2/RAM1.5GB/90sestimate/300souter,0retries.')
    save(run / 'submission.json', dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1':
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest='mode', required=True)
        a = commands.add_parser('audit')
        for key in ('native_root', 'prior_root', 'main_root', 'output'):
            a.add_argument(key, type=Path)
        s = commands.add_parser('stage'); s.add_argument('run', type=Path); s.add_argument('--native', type=Path, required=True)
        u = commands.add_parser('submit'); u.add_argument('run', type=Path)
        args = parser.parse_args()
        if args.mode == 'audit':
            report = analyze(args.native_root, args.prior_root, args.main_root, args.output)
            print(json.dumps({key: report[key] for key in ('mechanics_status', 'entry_witness_comparisons', 'exact_entry_witness_comparisons', 'new_model_calls')}, sort_keys=True))
        else:
            print(json.dumps(stage(args.run.resolve(), args.native.resolve()) if args.mode == 'stage' else submit(args.run.resolve()), sort_keys=True))
