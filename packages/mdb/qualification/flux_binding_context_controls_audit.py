"""Saved-only contextual row control audit and finite CPU launcher.

audit CONTROL_ROOT MAIN_ROOT OUTPUT_JSON
stage RUN --controls COMPLETED_CONTROL_RUN
submit RUN
No model hydration, native model operation or new image generation occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import struct
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = 'IC.context-controls'
COUNTS = dict(loads=1, encodes=0, noise_draws=0, native_transitions=448,
              native_blocks=400, acts=16, receipts=464, images=8)


def helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization_audit.py'))


def source_helpers():
    return runpy.run_path(str(HERE / 'flux_binding_generalization.py'))


def slot_bits(evidence, fingerprint, slot):
    """Read original BF16 tensor bytes from its typed page without a framework."""
    import numpy as np
    cut, descriptors = evidence.cut(fingerprint)
    described = descriptors[slot]
    if described.get('kind') != 'tensor' or described.get('dtype') != 'torch.bfloat16':
        raise ValueError('expected BF16 native cut slot: ' + slot)
    page = json.loads(evidence.path('store/pages/' + cut['slots'][slot] + '.json').read_text())
    encoded = page['encoded']
    blob = evidence.path('store/blobs/' + page['tensor_blob'] + '.safetensors')
    with blob.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated tensor header')
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= 8 << 20:
            raise ValueError('invalid tensor header extent')
        item = json.loads(stream.read(size))[encoded['name']]
        start, stop = item['data_offsets']
        if item['dtype'] != 'BF16' or item['shape'] != described['shape'] or not (0 <= start <= stop <= blob.stat().st_size-8-size):
            raise ValueError('typed slot storage geometry differs')
        stream.seek(8 + size + start)
        raw = stream.read(stop-start)
    if hashlib.sha256(raw).hexdigest() != described['sha256']:
        raise ValueError('typed slot content hash differs')
    value = np.frombuffer(raw, dtype='<u2')
    if value.size != int(np.prod(described['shape'], dtype=np.int64)):
        raise ValueError('typed slot element count differs')
    return value.reshape(described['shape'])


def conditioning_source_dose(base, source, contextual_rows, changed_id_rows):
    """Measure source-minus-base dose at two row sets, without an effect gate."""
    import numpy as np
    if (base.dtype != np.uint16 or source.dtype != np.uint16
            or base.shape != source.shape or base.ndim != 3 or not base.size):
        raise ValueError('conditioning dose needs equally shaped rank-three BF16 bits')

    def measure(rows):
        if (not rows or len(set(rows)) != len(rows)
                or any(type(i) is not int or not 0 <= i < base.shape[1] for i in rows)):
            raise ValueError('conditioning dose row addresses are invalid')
        left, right = [np.ascontiguousarray(value[:, rows]) for value in (base, source)]
        floats = [(value.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                  for value in (left, right)]
        if not all(bool(np.isfinite(value).all()) for value in floats):
            raise ValueError('nonfinite conditioning source dose evidence')
        delta = floats[1] - floats[0]
        changed = int(np.count_nonzero(left != right))
        return dict(rows=list(rows), shape=list(left.shape), elements=int(left.size),
                    l2=float(np.sqrt(np.sum(delta * delta))),
                    rms=float(np.sqrt(np.mean(delta * delta))),
                    different_bf16_elements=changed,
                    actually_different_payload_bytes=int(np.count_nonzero(left.view(np.uint8) != right.view(np.uint8))),
                    storage_bytes_occupied_by_different_bf16_elements=2 * changed)

    contextual, original = measure(contextual_rows), measure(changed_id_rows)
    return dict(delta='source conditioning minus base conditioning, decoded from native BF16 CAS bytes',
                contextual_rows=contextual, original_changed_id_rows=original,
                contextual_to_original_l2_ratio=contextual['l2'] / original['l2'] if original['l2'] else None,
                contextual_to_original_rms_ratio=contextual['rms'] / original['rms'] if original['rms'] else None,
                ratio_status='defined' if original['l2'] else 'undefined-original-changed-id-dose-is-zero',
                interpretation='Diagnostic only; contextual and changed-ID row interventions are not energy matched.')


def analyze(control_root, main_root, output):
    import numpy as np
    from PIL import Image
    h = helpers()
    read, require = h['read'], h['require']
    control_root, main_root = h['find_root'](control_root), h['find_root'](main_root)
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(control_root) and not output.is_relative_to(main_root),
            'audit output must be new and outside both immutable input roots')
    control_hashes, custody = h['inventory'](control_root)
    main_hashes, main_custody = h['inventory'](main_root)
    evidence, main = h['Evidence'](control_root, control_hashes), h['Evidence'](main_root, main_hashes)
    plan, request, summary, rows, fields, actions = [read(evidence.path(name)) for name in
        ('plan.json', 'request.json', 'summary.json', 'rows.json', 'fields.json', 'actions.json')]
    main_plan, main_request, main_rows, main_fields, parents, addresses = [read(main.path(name)) for name in
        ('plan.json', 'request.json', 'rows.json', 'fields.json', 'parents.json', 'token-addresses.json')]
    require(request['config'] == plan and main_request['config'] == main_plan, 'request/config join differs')
    require(plan['source_manifest_sha256'] == main_custody['files_manifest_sha256']
            and plan['source_request_sha256'] == main_request['request_sha256'], 'main source identity differs')
    require(summary['request_sha256'] == request['request_sha256'] and summary['counts'] == COUNTS,
            'control request/census differs')
    contexts = plan['contexts']
    require(contexts == main_plan['contexts'] and len(contexts) == 8, 'source context set differs')
    cases = [c['id'] for c in contexts]
    require(len(set(cases)) == 8, 'duplicate context IDs')
    index = {(r['case'], r['arm']): r for r in rows}
    require(len(rows) == len(index) == 8 and set(index) == {(case, ARM) for case in cases}, 'control row inventory differs')
    field_index = {(r['case'], r['field']): r for r in fields}
    require(len(fields) == len(field_index) == 16
            and set(field_index) == {(case, f) for case in cases for f in ('image', 'text')}
            and all(r['arm'] == ARM for r in fields), 'control projection inventory differs')
    main_index = {(r['case'], r['arm']): r for r in main_rows}
    main_field_index = {(r['case'], r['arm'], r['field']): r for r in main_fields}
    context_h = runpy.run_path(str(HERE / 'flux_binding_context_controls.py'))
    selections = {c['id']: context_h['select_rows'](c, addresses['tokens'][c['id']], addresses['selectors'][c['id']]) for c in contexts}
    require(selections == plan['selections'], 'frozen contextual row resolver differs')
    frozen = h['untag'](read(evidence.path('transaction/forecasts/context_control_predictions.json'))['predictions'])
    prefix = {key: int(key == 'loads') for key in COUNTS}
    require(summary['counts_at_seal'] == frozen['counts_at_seal'] == prefix,
            'new native work/encoder/noise preceded global control seal')
    require(frozen['semantic'] == plan['hypotheses'] and frozen['selections'] == selections,
            'frozen source selectors/hypotheses differ')
    require(set(frozen['projections']) == set(cases), 'sealed projection cases differ')
    timeline = [json.loads(line) for line in evidence.path('timeline.jsonl').read_text().splitlines()]
    seals = [i for i, row in enumerate(timeline) if row['kind'] == 'sealed']
    require(len(seals) == 1 and timeline[seals[0]]['counts'] == prefix, 'control seal chronology differs')
    completed = [(i, r['case']) for i, r in enumerate(timeline) if r['kind'] == 'image_complete']
    require(len(completed) == 8 and set(c for _, c in completed) == set(cases)
            and min(i for i, _ in completed) > seals[0], 'native control future preceded seal or is missing')
    transaction_dirs = [p.parent for p in control_root.rglob('seal.json') if (p.parent/'events').is_dir()]
    require(transaction_dirs == [control_root/'transaction'], 'expected one global control transaction')
    try:
        from mdb.forecast_transaction import verify_transaction
    except ModuleNotFoundError as exc:
        if exc.name not in ('mdb', 'mdb.forecast_transaction'):
            raise
        transaction = dict(status='installed-verifier-unavailable', local_inventory_and_forecast_joins=True)
    else:
        transaction = verify_transaction(control_root/'transaction', expected_seal_sha256=h['sha'](control_root/'transaction/seal.json'))
        require(transaction['verified'] and transaction['phase'] == 'closed', 'control transaction is not closed')
    projections, joins, footprint, selected_changes, source_doses, weak_controls, diagnostics = [], [], {}, {}, {}, {}, {}
    require(len(actions) == 16, 'expected sixteen actual C Acts')
    for context in contexts:
        case = context['id']
        row = index[case, ARM]
        require(row['rows'] == selections[case]['rows'] and row['source_parent'] == context_h['roots_for_case'](parents, case),
                'control source parent/rows differ')
        for field in ('image', 'text'):
            item = field_index[case, field]
            require(item['prediction'] == frozen['projections'][case][field], 'comparison is not the sealed prediction')
            actual, predicted = evidence.bits(item['actual']), evidence.bits(item['prediction'])
            require(list(actual.shape) == [1, 1024 if field == 'image' else 512, 3072], 'projected field is not complete')
            evidence.witness_join(item['actual'], row['marks']['joint0_entry'], field)
            projections.append(dict(case=case, field=field, **h['comparison'](predicted, actual)))
        joins.append(dict(case=case, **evidence.latent_join(row)))
        base = slot_bits(main, parents[case]['primed'], 'conditioning')
        source = slot_bits(main, parents[case]['source'], 'conditioning')
        selected = selections[case]['rows']
        expected = base.copy()
        expected[:, selected] = source[:, selected]
        local_actions = [a for a in actions if a['case'] == case]
        require(len(local_actions) == 2, 'wrong per-context C Act count')
        by_step = {}
        for action in local_actions:
            before, before_payload = evidence.cut(action['before'])
            after, after_payload = evidence.cut(action['after'])
            point = before['execution_point']
            require(action['slot'] == 'conditioning' and point['phase'] == 'project'
                    and point['logical_step'] in (2, 3), 'C Act is at wrong slot/clock')
            step = point['logical_step']
            require(step not in by_step, 'duplicate C Act clock')
            require(set(before_payload) == set(after_payload)
                    and all(before_payload[k] == after_payload[k] for k in before_payload if k != 'conditioning'),
                    'C Act changed another payload slot')
            require(action['receipt']['parent'] == action['before'] and action['receipt']['result'] == action['after'],
                    'C Act receipt/cut join differs')
            actual = slot_bits(evidence, action['after'], 'conditioning')
            require(np.array_equal(actual, expected if step == 2 else base), 'C row transplant/base reset differs')
            require(row['marks']['project'+str(step)+'_input'] == action['after'], 'saved project-input mark differs from actual Act')
            by_step[step] = dict(typed_footprint_exact=True, conditioning_bits_exact=True,
                                 before=before['fingerprint'], after=after['fingerprint'])
            if step == 2:
                _, original_payload = main.cut(parents[case]['primed'])
                require(before_payload == original_payload, 'new control does not start at saved main primed parent')
        require(set(by_step) == {2, 3}, 'project3 base-C reset absent')
        footprint[case] = by_step
        left, right = base[:, selected], source[:, selected]
        changed = [int(np.count_nonzero(left[:, j] != right[:, j])) for j in range(2)]
        selected_changes[case] = dict(rows=selected, changed_bf16_elements_per_row=changed,
            changed_payload_bytes=int(np.count_nonzero(left.copy().view(np.uint8) != right.copy().view(np.uint8))),
            affected_bf16_storage_bytes=2 * sum(changed),
            byte_metric_definition='changed_payload_bytes counts actually unequal uint8 bytes; affected_bf16_storage_bytes counts both storage bytes of every unequal BF16 element.',
            actual_project2_conditioning_changed_elements=int(np.count_nonzero(base != expected)),
            source_rows_carry_a_difference=any(changed), status='contextual-difference-observed' if any(changed) else 'source-identical-control-uninformative')
        source_doses[case] = conditioning_source_dose(base, source, selected, addresses['selectors'][case]['changed'])
        declared = frozen['source_diagnostics'][case]
        require(declared['changed_bf16_elements_per_row'] == changed
                and declared['changed_payload_bytes'] == selected_changes[case]['changed_payload_bytes'], 'source-byte diagnostic differs')
        _, original_i = main.cut(main_index[case, 'I']['marks']['joint0_entry'])
        _, weak_i = main.cut(main_index[case, 'IC.control-rows']['marks']['joint0_entry'])
        weak_projection = {}
        for field in ('image', 'text'):
            a = main.bits(main_field_index[case, 'I', field]['actual'])
            b = main.bits(main_field_index[case, 'IC.control-rows', field]['actual'])
            weak_projection[field] = h['comparison'](b, a)
        weak_controls[case] = dict(original_prefix_rows=addresses['selectors'][case]['control'],
            conditioning_descriptor_exact_to_I=original_i['conditioning'] == weak_i['conditioning'],
            projected_fields=weak_projection,
            status='source-identical-prefix-sham' if original_i['conditioning'] == weak_i['conditioning'] else 'nonzero-prefix-control-observed')
        images = {}
        for name, owner, item in ((ARM, evidence, row),
                                  ('natural.base', main, main_index[case, 'natural.base']),
                                  ('natural.source', main, main_index[case, 'natural.source']),
                                  ('I', main, main_index[case, 'I']),
                                  ('old-prefix-control', main, main_index[case, 'IC.control-rows'])):
            path = owner.path(item['png'])
            require(owner.hashes[item['png']] == item['png_sha256'], 'PNG reference changed')
            with Image.open(path) as image:
                require(image.mode == 'RGB' and image.size == (512, 512), 'PNG shape/mode differs')
                images[name] = np.asarray(image).copy()
        diagnostics[case] = dict(
            contextual_control=h['rgb_diagnostics'](images[ARM], images['natural.base'], images['natural.source']),
            matched_I=h['rgb_diagnostics'](images['I'], images['natural.base'], images['natural.source']),
            new_control_minus_I_rms_0_255=float(np.sqrt(np.mean((images[ARM].astype(np.float64)-images['I'].astype(np.float64))**2))),
            old_prefix_control_minus_I_rms_0_255=float(np.sqrt(np.mean((images['old-prefix-control'].astype(np.float64)-images['I'].astype(np.float64))**2))),
            old_prefix_control_png_bytes_exact_to_I=main_index[case, 'IC.control-rows']['png_sha256'] == main_index[case, 'I']['png_sha256'])
    exact = all(r['raw_bits_exact'] for r in projections)
    result = dict(schema='mdb-binding-context-controls-audit-v1', mechanics_status='verified',
        instrument_status='contextual-row-source-differences-measured', numerical_status='all-compared-bits-exact' if exact else 'nonzero-discrepancies-observed',
        trend_status='observations-retained', terminal_status='unassessed', source_job=summary['job_id'], main_source_job=plan['source_job'],
        custody=custody, main_custody=main_custody, counts=COUNTS, transaction=transaction,
        projection_comparisons=16, exact_projection_comparisons=sum(r['raw_bits_exact'] for r in projections),
        projections=projections, final_native_joins=joins, conditioning_footprints=footprint,
        selections=selections, selected_source_changes=selected_changes, source_dose_diagnostics=source_doses,
        original_prefix_controls=weak_controls, rgb_diagnostics=diagnostics, new_model_calls=0, new_images=0,
        limitations=['Original shared Linear primitives forecast projected I/T only; native RGB is observed.',
                     'Pixel projections/halves are diagnostics, not semantic attribute classifications or causal percentages.',
                     'Source-identical selected rows are an uninformative control, not evidence of absent distributed attribute state.',
                     'Contextual-row and changed-ID-row source differences have measured, potentially unequal L2/RMS doses; this is not an energy-matched address-specificity test.',
                     'Eight dependent contexts reuse two main-run noise seeds; manual object/attribute competence review remains separate.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def worker():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    sh = source_helpers()
    read, save, sha = [sh[k] for k in ('read', 'save', 'sha')]
    request = read(os.environ['SATURN_DEBUG_REQUEST'])
    config = request['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(sha(Path(name)) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('CPU audit isolation/source binding differs')
    root, main = Path(config['control_root']), Path(config['main_root'])
    if sha(root/'files.json') != config['control_manifest_sha256'] or sha(main/'files.json') != config['main_manifest_sha256']:
        raise ValueError('source inventory identity differs')
    out = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID'] / 'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = analyze(root, main, out/'audit.json')
    for name in ('rows.json', 'fields.json', 'actions.json', 'source-row-diagnostics.json',
                 'source-token-addresses.json', 'source-parents.json', 'native-contract.json',
                 'parameter-artifacts.json', 'retrieve_weights.py', 'retrieve_flux_encoder.py', 'source-binding.json'):
        shutil.copyfile(root/name, out/name)
    for name in ('plan.json', 'request.json', 'summary.json', 'timeline.jsonl'):
        shutil.copyfile(root/name, out/('source-'+name))
    for row in read(root/'rows.json'):
        shutil.copyfile(root/row['png'], out/row['png'])
    shutil.copytree(root/'transaction', out/'source-transaction')
    save(out/'external-evidence.json', dict(host='beast', roots=dict(control=str(root), main=str(main)),
         manifests=dict(control=read(root/'files.json'), main=read(main/'files.json')),
         scope='Full native cuts, BF16 arrays and receipts stay near compute; source identity preserved by both complete inventories.'))
    shutil.copyfile(HERE/'retrieve_reader_evidence.py', out/'retrieve_evidence.py')
    save(out/'plan.json', config)
    save(out/'request.json', request)
    save(out/'summary.json', dict(schema='mdb-binding-context-controls-compact-v1', job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=request['request_sha256'], output_root=str(out), source_job=report['source_job'],
        mechanics_status=report['mechanics_status'], seconds=time.monotonic()-started,
        new_execution_counts=dict(model_loads=0, native_primitives=0, images=0), terminal_status='unassessed'))
    save(out/'files.json', {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'],
        output_root=str(out), files_manifest_sha256=sha(out/'files.json'), summary_sha256=sha(out/'summary.json')), sort_keys=True), flush=True)


def stage(run, controls):
    from mdb.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb.payload import stage_worker
    sh = source_helpers()
    read, save, sha = [sh[k] for k in ('read', 'save', 'sha')]
    old = read(controls/'request.json')
    jid = read(controls/'submission.json')['job_id']
    _, raw, _ = scheduler_api().request('GET', '/api/jobs/'+jid, timeout_s=20)
    if json.loads(raw)['state'] != 'succeeded':
        raise ValueError('completed native contextual controls required')
    root = Path(old['config']['output_scratch'])/jid/'binding-minimal-result'
    main = Path(old['config']['source_root'])
    code = '''import json,hashlib
from pathlib import Path
roots=dict(control=Path(CONTROL),main=Path(MAIN));print(json.dumps({k:dict(sha=hashlib.sha256((r/'files.json').read_bytes()).hexdigest(),bytes=sum(p.stat().st_size for p in r.rglob('*') if p.is_file()))for k,r in roots.items()}))
'''.replace('CONTROL',repr(str(root))).replace('MAIN',repr(str(main)))
    metadata = json.loads(subprocess.run(['ssh','beast','python3','-I','-'],input=code,text=True,capture_output=True,check=True).stdout)
    if metadata['main']['sha'] != old['config']['source_manifest_sha256']:
        raise ValueError('completed main source manifest differs')
    config = dict(control_root=str(root), main_root=str(main), control_manifest_sha256=metadata['control']['sha'],
        main_manifest_sha256=metadata['main']['sha'], output_scratch='/mnt/big/mdb/research', directory_custody=True)
    run.mkdir(parents=True,exist_ok=False)
    save(run/'plan.json',config)
    names = ('flux_binding_generalization.py','flux_binding_generalization_audit.py','flux_binding_context_controls.py','flux_family_custody.py','retrieve_reader_evidence.py')
    extras = [(HERE/name,name)for name in names]
    pre = run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*names,pre.name],
        path_token_allowlist=[str(root),str(main),config['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    request = dict(schema='mdb-binding-context-controls-audit-request-v1',worker_script=Path(__file__).name,config=config,
        execution_environment=old['execution_environment'],needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float32',
        source_hashes={name:sha(path)for path,name in [(Path(__file__),Path(__file__).name),*extras]},
        resource_exception=dict(ram_mb=1536,vram_mb=0,cpu_threads=2,disk_gb=1,est_wall_s=90,
            reason='Saved-only16projectionfields/16CActfootprints/8PNGs, both source inventories;largest3.93M BF16conditioner, streampercontext. No Torch/weights/model hydration.1.5GBRAM/2CPU/90sestimate/300souter.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=48,logical_branches=1,
            logical_checkpoint_replays=0,max_local_work_items=64,extra=dict(task_family='analysis'),
            checkpoint_policy='Saved-only contextualcontrol audit:16sealedprojectionfields,16CActfootprints,8nativePNGcomparisons,2completeinventories;0models/0nativeforwards/retries.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/'request.json',request)
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    stage_worker(run/'payload',worker=Path(__file__),config=config,request=request,additional_files=extras)
    return dict(request_sha256=request['request_sha256'],sources=metadata)


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    sh=source_helpers();read,save,sha=[sh[k]for k in ('read','save','sha')]
    request=read(run/'request.json')
    runpy.run_path(str(HERE/'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    if (run/'submission.json').exists()or any(sha(run/'payload'/n)!=v for n,v in request['source_hashes'].items()):
        raise ValueError('changed/already submitted saved audit')
    command=worker_command(request['worker_script']);command.insert(1,'CUDA_VISIBLE_DEVICES=');command.insert(command.index('python')+1,'-B')
    resources={k:request['resource_exception'][k]for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job=guarded_mrun_launch(command,request=request,model=None,task='analysis',experiment='mdb-binding-context-controls-audit',
        payload=run/'payload',gpu=False,needs=request['needs'],prefer_host='beast',timeout_s=300,preflight=True,retry_on_kill=False,**resources,
        note='Saved-only contextualcontrol audit:16sealedI/Tfields,16CActfootprints/baseCresets,8existingPNGs,sourcebytes andweakprefixsham vsI,dualcompleteinventories.0models/nativeforwards/newimages,CPU2/RAM1.5GB/90sestimate/300souter,no retries.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':
        worker()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        commands=parser.add_subparsers(dest='mode',required=True)
        a=commands.add_parser('audit');a.add_argument('control_root',type=Path);a.add_argument('main_root',type=Path);a.add_argument('output',type=Path)
        s=commands.add_parser('stage');s.add_argument('run',type=Path);s.add_argument('--controls',type=Path,required=True)
        u=commands.add_parser('submit');u.add_argument('run',type=Path)
        args=parser.parse_args()
        if args.mode=='audit':
            report=analyze(args.control_root,args.main_root,args.output)
            print(json.dumps({k:report[k]for k in ('mechanics_status','projection_comparisons','exact_projection_comparisons','new_model_calls')},sort_keys=True))
        else:
            print(json.dumps(stage(args.run.resolve(),args.controls.resolve())if args.mode=='stage'else submit(args.run.resolve()),sort_keys=True))
