"""Saved-only joint0-exit image/text carrier cross and finite CPU publisher.

audit NATIVE_ROOT PRIOR_ROOT MAIN_ROOT OUTPUT_JSON
stage RUN --native COMPLETED_NATIVE_RUN
submit RUN
Entry fields are supplied-state witnesses, not learned-equation forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ('I.P_T.P', 'I.P_T.Q', 'I.Q_T.P', 'I.Q_T.Q')
REPLAYS = {'I.P_T.P': 'A@A', 'I.Q_T.Q': 'A@B'}
COUNTS = dict(loads=1, encodes=0, noise_draws=0, native_transitions=864,
              native_blocks=784, acts=48, receipts=912, images=16)


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization_audit.py'))


def source_helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def source_letters(arm):
    if arm not in ARMS:
        raise ValueError('unknown carrier cross arm')
    image, text = arm.split('_')
    return image[-1], text[-1]


def reference_closeness(image, same_text, other_text):
    """Signed geometric closeness; no hybrid pixel prediction or semantics."""
    import numpy as np
    arrays = [np.asarray(value, dtype=np.float64) for value in (image, same_text, other_text)]
    if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
        raise ValueError('reference RGB geometry differs')
    same, other = [float(np.sqrt(np.mean((arrays[0] - value) ** 2))) for value in arrays[1:]]
    return dict(rms_to_same_text_reference_0_255=same, rms_to_other_text_reference_0_255=other,
                other_minus_same_rms_0_255=other - same,
                scope='Diagnostic distance to known endpoint images, not an exact hybrid image forecast or semantic classifier.')


def carrier_factorial(images, natural_base, natural_source, rgb_response):
    """Conditional I/T finite differences and interaction over complete RGB."""
    import numpy as np
    y = {key: images[key].astype(np.float64) for key in ARMS}
    contrasts = dict(text_effect_with_I_P=y['I.P_T.P'] - y['I.P_T.Q'],
        text_effect_with_I_Q=y['I.Q_T.P'] - y['I.Q_T.Q'],
        image_effect_with_T_P=y['I.P_T.P'] - y['I.Q_T.P'],
        image_effect_with_T_Q=y['I.P_T.Q'] - y['I.Q_T.Q'],
        interaction=y['I.P_T.P'] - y['I.P_T.Q'] - y['I.Q_T.P'] + y['I.Q_T.Q'])
    return dict(convention='Y_PP-Y_PQ-Y_QP+Y_QQ; first symbol selects image source, second text source.',
        contrasts={key: rgb_response(value, np.zeros_like(value), natural_base, natural_source) for key, value in contrasts.items()},
        scope='Observed finite RGB contrasts; no semantic mediation percentages or population effects.')


def state_field_comparison(evidence, reference_cut, actual_cut, slot, h, mh, slot_bits):
    """Native project boundaries contain None streams; preserve that distinction."""
    import numpy as np
    _, reference = evidence.cut(reference_cut)
    _, actual = evidence.cut(actual_cut)
    left, right = reference[slot], actual[slot]
    result = dict(reference_kind=left.get('kind'), actual_kind=right.get('kind'),
                  descriptors_exact=left == right, l2=None, rms=None, raw_bits_exact=None)
    if left.get('kind') != 'tensor' or right.get('kind') != 'tensor':
        result['status'] = ('not-applicable-nontensor' if left.get('kind') != 'tensor' and right.get('kind') != 'tensor'
                            else 'not-comparable-tensor-versus-nontensor')
        return result
    result.update(reference_shape=left.get('shape'), actual_shape=right.get('shape'),
                  reference_dtype=left.get('dtype'), actual_dtype=right.get('dtype'))
    if left.get('shape') != right.get('shape') or left.get('dtype') != 'torch.bfloat16' or right.get('dtype') != 'torch.bfloat16':
        result['status'] = 'not-comparable-tensor-geometry-or-dtype'
        return result
    result['status'] = 'compared-tensor'
    if left == right:
        result.update(raw_bits_exact=True, l2=0., rms=0., elements=int(np.prod(left['shape'], dtype=np.int64)))
    else:
        baseline, value = slot_bits(evidence, reference_cut, slot), slot_bits(evidence, actual_cut, slot)
        result.update(raw_bits_exact=bool(np.array_equal(baseline, value)), **mh['delta_metrics'](baseline, value, h))
    return result


def analyze(native_root, prior_root, main_root, output):
    import numpy as np
    from PIL import Image
    h = helpers()
    read, require = h['read'], h['require']
    mh = runpy.run_path(str(HERE / 'flux_binding_matched_dose_audit.py'))
    slot_bits = runpy.run_path(str(HERE / 'flux_binding_context_controls_audit.py'))['slot_bits']
    roots = {key: h['find_root'](value) for key, value in (('native', native_root), ('prior', prior_root), ('main', main_root))}
    output = Path(output).resolve()
    require(not output.exists() and all(not output.is_relative_to(root) for root in roots.values()), 'output must be new and outside immutable inputs')
    inventories = {key: h['inventory'](root) for key, root in roots.items()}
    owners = {key: h['Evidence'](root, inventories[key][0]) for key, root in roots.items()}
    native, prior, main = [owners[key] for key in ('native', 'prior', 'main')]
    plan, request, summary, rows, fields, actions, recipes, parents, trajectory, ports = [read(native.path(name)) for name in
        ('plan.json', 'request.json', 'summary.json', 'rows.json', 'fields.json', 'actions.json', 'source-recipes.json', 'source-parents.json', 'trajectory.json', 'ports.json')]
    prior_plan, prior_rows, prior_request = [read(prior.path(name)) for name in ('plan.json', 'rows.json', 'request.json')]
    main_plan, main_rows = [read(main.path(name)) for name in ('plan.json', 'rows.json')]
    require(request['config'] == plan and summary['request_sha256'] == request['request_sha256'], 'request/config/summary join differs')
    require(plan['prior_manifest_sha256'] == inventories['prior'][1]['files_manifest_sha256']
            and plan['main_manifest_sha256'] == inventories['main'][1]['files_manifest_sha256']
            and plan['prior_request_sha256'] == prior_request['request_sha256'], 'source manifest/request binding differs')
    require(prior_plan['main_manifest_sha256'] == plan['main_manifest_sha256'], 'prior/main source join differs')
    require(summary['counts'] == COUNTS, 'carrier cross census differs')
    contexts, cases = plan['contexts'], [case['id'] for case in plan['contexts']]
    require(len(contexts) == len(set(cases)) == 4 and set(cases) == {'anchor', 'seed', 'wording', 'objects'}, 'four fixed historical cases required')
    pc, mc = [{case['id']: case for case in item['contexts']} for item in (prior_plan, main_plan)]
    require(all(case == pc[case['id']] == mc[case['id']] for case in contexts), 'historical contexts differ')
    keys = {(case, arm) for case in cases for arm in ARMS}
    index, field_index = [{(row['case'], row['arm']): row for row in values} for values in (rows, fields)]
    require(len(rows) == len(index) == len(fields) == len(field_index) == 16 and set(index) == set(field_index) == keys, 'arm/witness inventory differs')
    port_index = {(row['case'], row['arm']): row for row in ports}
    port_status = dict(status='available', observed_only=True)
    if not ports:
        recovery = read(native.path('recovery.json'))
        require(recovery['optional_port_index_status'] == 'unavailable-not-persisted'
                and recovery['original_job_state'] == 'failed' and recovery['new_native_execution'] is False,
                'empty port index lacks explicit failed-serialization recovery')
        proof = read(native.path('source-physical-inventory.json'))
        require(native.hashes['source-physical-inventory.json'] == recovery['source_physical_inventory_sha256']
                and 'ports.json' not in proof['files'] and 'trajectory.json' in proof['files'], 'recovery source inventory proof differs')
        require(set(recovery['original_to_recovered']) == set(proof['files']), 'recovery omitted an original physical file')
        for original_name, item in proof['files'].items():
            retained = recovery['original_to_recovered'][original_name]
            require(native.hashes[retained] == item['sha256'] and native.path(retained).stat().st_size == item['bytes'],
                    'retained original physical file differs')
        job = read(native.path('source-failed-job.json'))
        require(native.hashes['source-failed-job.json'] == recovery['original_job_record_sha256']
                and native.hashes['source-failed-worker.log'] == recovery['original_log_sha256']
                and job['state'] == 'failed' and job['job_id'] == summary['job_id'] == recovery['original_job_id']
                and request['request_sha256'] == recovery['original_request_sha256'], 'original failed job/log/request proof differs')
        port_status = dict(status='unavailable-not-persisted', observed_only=True,
            reason='Native attention arrays remain retained but their per-arm index was never persisted; no ports attributed or manufactured.',
            recovery_sha256=native.hashes['recovery.json'], source_physical_inventory_sha256=native.hashes['source-physical-inventory.json'])
    else:
        require(len(ports) == len(port_index) == 16 and set(port_index) == keys, 'observed port inventory differs')
    labels = ('entry', 'after_joint1', 'before_project3', 'project3_input', 'after_project3', 'after_step3_joint0', 'final')
    trajectory_index = {(row['case'], row['arm'], row['label']): row for row in trajectory}
    require(len(trajectory) == len(trajectory_index) == 112
            and set(trajectory_index) == {(case, arm, label) for case, arm in keys for label in labels}, 'trajectory inventory differs')
    prior_index = {(row['case'], row['arm']): row for row in prior_rows}
    main_index = {(row['case'], row['arm']): row for row in main_rows}
    require(set(recipes) == set(parents) == set(cases), 'source recipe/parent cases differ')
    forecast_paths = list((roots['native'] / 'transaction/forecasts').glob('*.json'))
    require(len(forecast_paths) == 1, 'one global entry witness payload required')
    frozen = h['untag'](read(native.path(forecast_paths[0].relative_to(roots['native']).as_posix()))['predictions'])
    prefix = {key: int(key == 'loads') for key in COUNTS}
    require(summary['counts_at_seal'] == frozen['counts_at_seal'] == prefix, 'native work preceded seal')
    require(frozen['recipes'] == recipes and frozen['semantic'] == plan['hypotheses'], 'frozen source recipes/hypotheses differ')
    timeline = [json.loads(line) for line in native.path('timeline.jsonl').read_text().splitlines()]
    seals = [i for i, row in enumerate(timeline) if row['kind'] == 'sealed']
    require(len(seals) == 1 and timeline[seals[0]]['counts'] == prefix, 'seal chronology differs')
    images_after = [(i, row['case'], row['arm']) for i, row in enumerate(timeline) if row['kind'] == 'image_complete']
    require(len(images_after) == 16 and {(case, arm) for _, case, arm in images_after} == keys
            and min(i for i, _, _ in images_after) > seals[0], 'image preceded seal or is missing')
    transaction_dirs = [path.parent for path in roots['native'].rglob('seal.json') if (path.parent / 'events').is_dir()]
    require(transaction_dirs == [roots['native'] / 'transaction'], 'one global transaction required')
    try:
        from mdb.forecast_transaction import verify_transaction
    except ModuleNotFoundError as exc:
        if exc.name not in ('mdb', 'mdb.forecast_transaction'):
            raise
        transaction = dict(status='installed-verifier-unavailable', local_inventory_and_entry_joins=True)
    else:
        transaction = verify_transaction(roots['native'] / 'transaction', expected_seal_sha256=h['sha'](roots['native'] / 'transaction/seal.json'))
        require(transaction['verified'] and transaction['phase'] == 'closed', 'carrier transaction is not closed')
    require(len(actions) == 48, 'expected48 typed Acts')
    witnesses, joins, compatibility, footprints, controls, rgb, closeness, factorials, ranks, trajectories, port_deltas, text_states, complete_payloads = [], [], {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    def image(owner, row):
        require(owner.hashes[row['png']] == row['png_sha256'], 'PNG reference changed')
        with Image.open(owner.path(row['png'])) as opened:
            require(opened.mode == 'RGB' and opened.size == (512, 512), 'PNG geometry differs')
            return np.asarray(opened).copy()

    for case in cases:
        require(set(recipes[case]) == set(ARMS), 'frozen recipe arm set differs')
        source_fps = {letter: prior_index[case, arm]['marks']['after_joint0'] for letter, arm in (('P', 'A@A'), ('Q', 'A@B'))}
        require(parents[case] == source_fps, 'source P/Q cuts differ from prior joint0 exits')
        cuts = {letter: prior.cut(fp) for letter, fp in source_fps.items()}
        p, q = cuts['P'][0], cuts['Q'][0]
        pp, qp = cuts['P'][1], cuts['Q'][1]
        require(p['manifest']['model_identity'] == q['manifest']['model_identity'] and p['execution_point'] == q['execution_point'], 'P/Q specimen or clock differs')
        point = q['execution_point']
        require(point['logical_step'] == 2 and point['phase'] == 'joint' and point['index'] == 1, 'source is not joint1 entry')
        require(set(pp) == set(qp) and all(pp[key] == qp[key] for key in pp if key not in ('image', 'text')), 'P/Q differ outside working image/text; cross invalid')
        compatibility[case] = dict(sources=source_fps, same_model_and_clock=True, non_image_text_payload_exact=True,
                                    identical_slots=sorted(key for key in pp if key not in ('image', 'text')))
        sealed_slots = frozen['pair_contracts'][case]['common_nonstream_slots']
        require(len(sealed_slots) == len(set(sealed_slots)) and set(sealed_slots) == set(compatibility[case]['identical_slots']), 'sealed P/Q compatibility contract differs')
        values = {letter: {slot: slot_bits(prior, source_fps[letter], slot) for slot in ('image', 'text', 'conditioning')} for letter in ('P', 'Q')}
        natural_base = image(main, main_index[case, 'natural.base'])
        natural_source = image(main, main_index[case, 'natural.source'])
        reference_images = {'P': image(prior, prior_index[case, 'A@A']), 'Q': image(prior, prior_index[case, 'A@B'])}
        current = {}
        footprints[case], controls[case], rgb[case], closeness[case], trajectories[case], port_deltas[case], text_states[case], complete_payloads[case] = {}, {}, {}, {}, {}, {}, {}, {}
        for arm in ARMS:
            row, entry, recipe = index[case, arm], field_index[case, arm], recipes[case][arm]
            image_letter, text_letter = source_letters(arm)
            expected = {'image': values[image_letter]['image'], 'text': values[text_letter]['text']}
            require(recipe['arm'] == arm and recipe['image_source'] == image_letter and recipe['text_source'] == text_letter
                    and recipe['image_source_cut'] == source_fps[image_letter] and recipe['text_source_cut'] == source_fps[text_letter], 'declared carrier source differs')
            require(row['source_parent'] == source_fps['Q'], 'branch is not based at Q')
            require(row['marks']['final'] == row['final_cut'], 'final landmark/row differs')
            require(entry['entry_cut'] == row['marks']['entry'], 'entry mark differs')
            for slot in ('image', 'text'):
                frozen_ref, actual_ref = entry['expected_' + slot], entry['actual_' + slot]
                require(frozen_ref == recipe['entry_' + slot], 'entry reference is not sealed')
                expected_bits, actual_bits = native.bits(frozen_ref), native.bits(actual_ref)
                require(list(actual_bits.shape) == [1, 1024 if slot == 'image' else 512, 3072], 'entry witness is not full shape')
                require(np.array_equal(expected_bits, expected[slot]), 'entry witness differs from declared source cut')
                native.witness_join(actual_ref, entry['entry_cut'], slot)
                witnesses.append(dict(case=case, arm=arm, field=slot, evidence_kind='supplied_state_entry_witness', **h['comparison'](expected_bits, actual_bits)))
            joins.append(dict(case=case, arm=arm, **native.latent_join(row)))
            local = [action for action in actions if action['case'] == case and action['arm'] == arm]
            require([action['slot'] for action in local] == ['image', 'text', 'conditioning'], 'expected ordered image/text Acts then C reset')
            previous_after = None
            for action in local:
                before, before_payload = native.cut(action['before'])
                after, after_payload = native.cut(action['after'])
                slot = action['slot']
                require(set(before_payload) == set(after_payload)
                        and all(before_payload[key] == after_payload[key] for key in before_payload if key != slot), 'Act changed another payload slot')
                require(action['receipt']['parent'] == action['before'] and action['receipt']['result'] == action['after'], 'receipt/cut join differs')
                if slot in ('image', 'text'):
                    require(before['execution_point'] == point, 'working-state Act is at wrong clock')
                    if slot == 'image':
                        require(before_payload == qp and action['before'] == source_fps['Q'], 'first Act is not based at Q')
                    else:
                        require(action['before'] == previous_after and action['after'] == row['marks']['entry'], 'two entry Acts are not joined')
                    require(np.array_equal(slot_bits(native, action['after'], slot), expected[slot]), 'installed carrier field differs')
                    previous_after = action['after']
                else:
                    reset = before['execution_point']
                    require(reset['logical_step'] == 3 and reset['phase'] == 'project', 'C reset clock differs')
                    require(before_payload['conditioning'] == qp['conditioning'], 'base C changed before reset')
                    require(action['after'] == row['marks']['project3_input'], 'C reset mark differs')
                    require(np.array_equal(slot_bits(native, action['after'], 'conditioning'), values['Q']['conditioning']), 'C reset is not base')
            footprints[case][arm] = dict(image_then_text_only_at_entry=True, entry_parent=source_fps['Q'], base_C_preserved_and_reset=True)
            trajectories[case][arm] = {}
            clocks = dict(entry=(2, 'joint', 1), after_joint1=(2, 'joint', 2), before_project3=(3, 'project', None),
                          project3_input=(3, 'project', None), after_project3=(3, 'joint', 0), after_step3_joint0=(3, 'joint', 1), final=(4, 'project', None))
            for label in labels:
                traced = trajectory_index[case, arm, label]
                require(traced['cut'] == row['marks'][label], 'trajectory mark differs')
                cut, descriptors = native.cut(traced['cut'])
                require(traced['point'] == cut['execution_point']
                        and (traced['point']['logical_step'], traced['point']['phase'], traced['point']['index']) == clocks[label], 'trajectory clock differs')
                require(traced['payload_descriptors'] == {slot: descriptors[slot] for slot in ('image', 'text', 'latent', 'conditioning')}, 'trajectory descriptor/CAS join differs')
                references = {name: trajectory_index[case, reference_arm, label]['cut']
                              for name, reference_arm in (('PP', 'I.P_T.P'), ('QQ', 'I.Q_T.Q'))}
                metrics = {slot: {name: state_field_comparison(native, ref, traced['cut'], slot, h, mh, slot_bits)
                                 for name, ref in references.items()} for slot in ('image', 'text', 'latent', 'conditioning')}
                trajectories[case][arm][label] = dict(cut=traced['cut'], reference_cuts=references, fields=metrics)
                if label in ('before_project3', 'project3_input'):
                    complete_payloads[case].setdefault(arm, {})[label] = {}
                    for name, reference_cut in references.items():
                        _, reference_payload = native.cut(reference_cut)
                        same_slots = set(reference_payload) == set(descriptors)
                        different = sorted(key for key in set(reference_payload) | set(descriptors)
                                           if key not in reference_payload or key not in descriptors or reference_payload[key] != descriptors[key])
                        complete_payloads[case][arm][label][name] = dict(reference_cut=reference_cut,
                            slot_sets_exact=same_slots, differing_slots=different, all_payload_exact=same_slots and not different,
                            nonlatent_slots_exact=same_slots and all(key == 'latent' for key in different),
                            latent_only_variation=same_slots and different == ['latent'],
                            scope='Complete native payload descriptor comparison; latent-only variation is observed, not a semantic gate.')
            if ports:
                port = port_index[case, arm]
                require(port['observed_only'] is True and port['before_entry'] == row['marks']['entry']
                        and port['after_joint1'] == row['marks']['after_joint1'], 'reader port/cut binding differs')
                require(set(port['values']) == {'attention_image', 'attention_text'}, 'observed attention port set differs')
                port_deltas[case][arm] = {}
                for name, ref in port['values'].items():
                    value = native.bits(ref)
                    require(list(value.shape) == [1, 1024 if name == 'attention_image' else 512, 3072], 'observed attention field is not complete')
                    port_deltas[case][arm][name] = mh['delta_metrics'](native.bits(port_index[case, 'I.Q_T.Q']['values'][name]), value, h)
            observed = image(native, row)
            current[arm] = observed
            rgb[case][arm] = dict(natural_axis=h['rgb_diagnostics'](observed, natural_base, natural_source),
                relative_to_Q=mh['rgb_response'](observed, reference_images['Q'], natural_base, natural_source))
            closeness[case][arm] = reference_closeness(observed, reference_images[text_letter], reference_images['Q' if text_letter == 'P' else 'P'])
            if arm in REPLAYS:
                controls[case][arm] = mh['endpoint_comparison'](native, row, prior, prior_index[case, REPLAYS[arm]], h)
        factorials[case] = carrier_factorial(current, natural_base, natural_source, mh['rgb_response'])
        for label in labels:
            described = {arm: native.cut(trajectory_index[case, arm, label]['cut'])[1]['text'] for arm in ARMS}
            tensors = all(value.get('kind') == 'tensor' for value in described.values())
            same = all(value == described[ARMS[0]] for value in described.values())
            text_states[case][label] = dict(kinds={arm: value.get('kind') for arm, value in described.items()},
                all_descriptors_exact=same, all_tensor_text_states_bit_identical=same if tensors else None,
                status='identical-tensor-text' if tensors and same else 'different-tensor-text' if tensors else 'nontensor-or-mixed-text-state',
                interpretation='Observed carrier equality/presence at this clock; project-boundary None is not a numerical zero or a semantic absence claim.')
        t_p = rgb[case]['I.Q_T.P']['relative_to_Q']['whole']['signed_increment_on_natural_axis']
        t_q = rgb[case]['I.P_T.Q']['relative_to_Q']['whole']['signed_increment_on_natural_axis']
        ranks[case] = dict(I_Q_T_P=t_p, I_P_T_Q=t_q, signed_difference=t_p - t_q if t_p is not None and t_q is not None else None,
            predicted_order_observed=t_p > t_q if t_p is not None and t_q is not None else None,
            scope='One frozen ordinal native-outcome prediction; semantic outcomes remain separately reviewed.')
    exact = sum(row['raw_bits_exact'] for row in witnesses)
    replay = all(row['full_payload_exact'] and row['final_latent']['raw_bits_exact'] and row['png_bytes_exact'] for pairs in controls.values() for row in pairs.values())
    result = dict(schema='mdb-binding-carrier-cross-audit-v1', mechanics_status='verified',
        instrument_status='known-replays-verified' if replay else 'known-replay-discrepancy-observed',
        entry_witness_status='all-supplied-entry-bits-exact' if exact == 32 else 'entry-discrepancies-observed',
        trend_status='observations-retained', terminal_status='unassessed', source_job=summary['job_id'],
        custody={key: value[1] for key, value in inventories.items()}, counts=COUNTS, transaction=transaction,
        entry_witness_comparisons=32, exact_entry_witness_comparisons=exact, prospective_denoiser_equation_forecasts=0,
        source_compatibility=compatibility, entry_witnesses=witnesses, final_native_joins=joins, typed_footprints=footprints,
        replay_controls=controls, rgb_diagnostics=rgb, same_text_reference_closeness=closeness, factorials=factorials,
        ordinal_predictions=ranks, trajectory_comparisons_to_PP_QQ=trajectories, text_state_observations=text_states,
        complete_payload_comparisons_before_project3=complete_payloads,
        observed_joint1_attention_deltas_from_QQ=port_deltas, optional_port_index_status=port_status,
        hypotheses=plan['hypotheses'], new_model_calls=0, new_images=0,
        limitations=['Entry I/T witnesses are copied supplied source states and typed installation checks, not learned-equation or RGB forecasts.',
                    'Eight self-replays are known controls and eight hybrids are new native futures; four source-assisted historical contexts are dependent.',
                    'The experiment tests conditional carrier sufficiency/coupling at joint0 exit, not exclusive semantic memory or maintenance at arbitrary clocks.',
                    'Base-C projection can make T identical again while image/latent histories differ; subsequent native readers can recreate text differences. Trajectory equality is observed, not a semantic gate.',
                    'RGB source-axis, interaction and reference-distance diagnostics are not semantic mediation percentages, probabilities or exact hybrid pixel predictions.',
                    'Every ordinal miss and partial/collateral semantic outcome remains evidence; terminal certification is unassessed.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    return result


def worker():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    h = source_helpers()
    read, save, sha = [h[key] for key in ('read', 'save', 'sha')]
    request = read(os.environ['SATURN_DEBUG_REQUEST']); config = request['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(sha(Path(name)) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('saved-only isolation/source binding differs')
    roots = {key: Path(config[key + '_root']) for key in ('native', 'prior', 'main')}
    if any(sha(root / 'files.json') != config[key + '_manifest_sha256'] for key, root in roots.items()):
        raise ValueError('source inventory changed')
    out = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    report = analyze(roots['native'], roots['prior'], roots['main'], out / 'audit.json')
    native = roots['native']
    for path in sorted(native.iterdir()):
        if path.is_file() and path.name != 'files.json' and path.suffix in ('.json', '.jsonl'):
            name = 'source-' + path.name if path.name in ('plan.json', 'request.json', 'summary.json', 'timeline.jsonl') else path.name
            shutil.copyfile(path, out / name)
    for name in ('retrieve_weights.py', 'retrieve_flux_encoder.py'):
        shutil.copyfile(native / name, out / name)
    for row in read(native / 'rows.json'):
        target = out / row['png']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native / row['png'], target)
    shutil.copytree(native / 'transaction', out / 'source-transaction')
    save(out / 'external-evidence.json', dict(host='beast', roots={key: str(root) for key, root in roots.items()},
        manifests={key: read(root / 'files.json') for key, root in roots.items()},
        scope='Complete native/prior/main inventories bind CAS/arrays/receipts retained near compute; checksum-verified individual retrieval.'))
    shutil.copyfile(HERE / 'retrieve_reader_evidence.py', out / 'retrieve_evidence.py')
    save(out / 'plan.json', config); save(out / 'request.json', request)
    save(out / 'summary.json', dict(schema='mdb-binding-carrier-cross-compact-v1', job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=request['request_sha256'], output_root=str(out), source_job=report['source_job'],
        mechanics_status=report['mechanics_status'], seconds=time.monotonic() - start,
        new_execution_counts=dict(model_loads=0, native_primitives=0, images=0), terminal_status='unassessed'))
    save(out / 'files.json', {path.relative_to(out).as_posix(): sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'],
        output_root=str(out), files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


def stage(run, native):
    from mdb.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb.payload import stage_worker
    h = source_helpers(); read, save, sha = [h[key] for key in ('read', 'save', 'sha')]
    old = read(native / 'request.json'); jid = read(native / 'submission.json')['job_id']
    _, raw, _ = scheduler_api().request('GET', '/api/jobs/' + jid, timeout_s=20)
    if json.loads(raw)['state'] != 'succeeded':
        raise ValueError('completed carrier-cross native campaign required')
    roots = dict(native=Path(old['config']['output_scratch']) / jid / 'binding-minimal-result',
                 prior=Path(old['config']['prior_root']), main=Path(old['config']['main_root']))
    code = 'import json,hashlib\nfrom pathlib import Path\nroots=' + repr({key: str(root) for key, root in roots.items()}) + '\nprint(json.dumps({k:dict(sha=hashlib.sha256((Path(v)/"files.json").read_bytes()).hexdigest(),bytes=sum(p.stat().st_size for p in Path(v).rglob("*") if p.is_file()))for k,v in roots.items()}))\n'
    metadata = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code, text=True, capture_output=True, check=True).stdout)
    config = dict(output_scratch='/mnt/big/mdb/research', directory_custody=True)
    for key, root in roots.items():
        config[key + '_root'], config[key + '_manifest_sha256'] = str(root), metadata[key]['sha']
        if key != 'native' and config[key + '_manifest_sha256'] != old['config'][key + '_manifest_sha256']:
            raise ValueError('native source inventory changed: ' + key)
    run.mkdir(parents=True, exist_ok=False); save(run / 'plan.json', config)
    names = ('flux_binding_generalization.py', 'flux_binding_generalization_audit.py', 'flux_binding_context_controls.py',
             'flux_binding_context_controls_audit.py', 'flux_binding_matched_dose_audit.py', 'flux_binding_carrier_cross.py',
             'flux_family_custody.py', 'retrieve_reader_evidence.py')
    extras = [(HERE / name, name) for name in names]
    pre = run / 'preflight.v1.json'
    save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name, *names, pre.name],
        path_token_allowlist=[*[str(root) for root in roots.values()], config['output_scratch'], 'binding-minimal-result']))
    extras.append((pre, pre.name))
    request = dict(schema='mdb-binding-carrier-cross-audit-request-v1', worker_script=Path(__file__).name, config=config,
        execution_environment=old['execution_environment'], needs=dict(cpu=True, host='beast'), task_family='analysis',
        backend='numpy', device='cpu', dtype='float32', source_hashes={name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        resource_exception=dict(ram_mb=1536, vram_mb=0, cpu_threads=2, disk_gb=1, est_wall_s=120,
            calibration_job='job-6382a82b5455', measured_peak_rss_mb=446.4, measured_wall_s=60.46,
            reason='Saved-only16arms/32entrywitnesses/48Acts/16PNGs/8replays,112trajectorycuts+readerports,3inventories. Streamonefield; prior28arm audit446.4MB/60.46s.1.5GBRAM/2CPU/120sestimate/360souter;0models.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=16, logical_branches=1,
            logical_checkpoint_replays=0, max_local_work_items=32, extra=dict(task_family='analysis',
                unit='one saved native arm with two full entry witnesses, three Acts, seven trajectory marks, native join and PNG'),
            checkpoint_policy='Saved-only16arm units:32suppliedentrywitnesses/48typedActs/112trajectorycuts/16PNGs/8knownreplays; source P/Q compatibility and carrierRGB2x2.0models/nativeforwards/retries.'))
    request['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / 'request.json', request)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    stage_worker(run / 'payload', worker=Path(__file__), config=config, request=request, additional_files=extras)
    return dict(request_sha256=request['request_sha256'], sources=metadata)


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    h = source_helpers(); read, save, sha = [h[key] for key in ('read', 'save', 'sha')]
    request = read(run / 'request.json')
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    if (run / 'submission.json').exists() or any(sha(run / 'payload' / name) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('changed/already submitted saved audit')
    command = worker_command(request['worker_script']); command.insert(1, 'CUDA_VISIBLE_DEVICES='); command.insert(command.index('python') + 1, '-B')
    resources = {key: request['resource_exception'][key] for key in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')}
    job = guarded_mrun_launch(command, request=request, model=None, task='analysis', experiment='mdb-binding-carrier-cross-audit',
        payload=run / 'payload', gpu=False, needs=request['needs'], prefer_host='beast', timeout_s=360, preflight=True, retry_on_kill=False, **resources,
        note='Saved-only16carrier-crossarm units:32entrywitnesses/48Acts/112trajectorycuts/16PNGs/8knownreplays,3inventories,P/Qnonstreamcompatibility,RGBI×T contrasts.0models/nativeforwards/newimages;CPU2/RAM1.5GB/120sestimate/360souter,0retries.')
    save(run / 'submission.json', dict(job_id=str(job))); return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1':
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='mode', required=True)
        a = commands.add_parser('audit')
        for key in ('native_root', 'prior_root', 'main_root', 'output'): a.add_argument(key, type=Path)
        s = commands.add_parser('stage'); s.add_argument('run', type=Path); s.add_argument('--native', type=Path, required=True)
        u = commands.add_parser('submit'); u.add_argument('run', type=Path)
        args = parser.parse_args()
        if args.mode == 'audit':
            report = analyze(args.native_root, args.prior_root, args.main_root, args.output)
            print(json.dumps({key: report[key] for key in ('mechanics_status', 'entry_witness_comparisons', 'exact_entry_witness_comparisons', 'new_model_calls')}, sort_keys=True))
        else:
            print(json.dumps(stage(args.run.resolve(), args.native.resolve()) if args.mode == 'stage' else submit(args.run.resolve()), sort_keys=True))
