"""Integrate one selected MLP law on stored own-state upstream predictions."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def helpers(payload):
    return {name: runpy.run_path(str(payload/name)) for name in
        ('flux_reentry_reference.py', 'flux_receiver_norm_prediction.py', 'flux_own_state_mlp_math.py',
         'flux_source_receiver.py', 'flux_attention_anchor.py', 'flux_numpy_archive.py', 'analyze_flux_binding_minimal.py')}


class Sources:
    """Bind saved predictions, native references and historical calibration separately."""
    def __init__(self, config, helper):
        self.config, self.helper = config, helper
        self.collection = Path(config['source_collection'])
        self.data = self.collection/'extracted/binding-minimal-result'
        self.record = read(self.collection/'remote-custody.json')
        if sha(self.collection/'remote-custody.json') != config['source_record_sha256'] or sha(config['source_audit']) != config['source_audit_sha256'] or read(config['source_audit'])['custody'] != self.record['verification']:
            raise ValueError('source scientific custody differs')
        self.checked = {}
        self.plan, self.native, self.forecasts, self.baselines, self.manifest = [self.metadata(name) for name in
            ('plan.json', 'native.json', 'forecasts.json', 'baselines.json', 'parameter-artifacts.json')]
        if self.plan['cases'] != config['cases']:
            raise ValueError('four-case cohort differs')
        self.previous = Path(config['previous_conditional_root'])
        if sha(self.previous/'conditional.json') != config['previous_conditional_sha256'] or sha(self.previous/'audit-v1.json') != config['previous_conditional_audit_sha256'] or read(self.previous/'audit-v1.json')['mechanics_status'] != 'verified':
            raise ValueError('previous conditional evidence differs')
        self.conditional = read(self.previous/'conditional.json')
        selection = Path(config['selection_collection'])
        if sha(selection/'remote-custody.json') != config['selection_record_sha256'] or sha(selection/'extracted/binding-minimal-result/panel-report.json') != config['selection_report_sha256']:
            raise ValueError('selected two-law panel differs')
        self.selection = read(selection/'extracted/binding-minimal-result/panel-report.json')
        if self.selection['instrument_status'] != 'qualified-control' or self.selection['request_sha256'] != config['selection_request_sha256']:
            raise ValueError('selected law unqualified')
        self.contexts = {}

    def checked_file(self, path):
        path = Path(path)
        relative = path.relative_to(self.data).as_posix()
        expected = self.record['files'][relative]
        if relative not in self.checked:
            if sha(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
                raise ValueError('source evidence changed: '+relative)
            self.checked[relative] = expected
        return path

    def metadata(self, name):
        return read(self.checked_file(self.data/name))

    def array(self, ref):
        from mdb.tensor_archive import load_tensor_archive
        root = ref['root']
        if root == 'conditional':
            return self.helper['flux_attention_anchor.py']['load_bits'](self.previous, ref['record'])
        base = self.data if root == 'source' else self.data/'source/prefixes' if root == 'prefixes' else None
        if base is None:
            raise ValueError('source root differs')
        self.checked_file(base/ref['record']['path'])
        return load_tensor_archive(base, ref['record'])

    def gate(self, fingerprint, role):
        import numpy as np
        if fingerprint not in self.contexts:
            store = self.data/'source/store'
            cut = read(self.checked_file(store/'cuts'/(fingerprint+'.json')))
            page = read(self.checked_file(store/'pages'/(cut['slots']['context']+'.json')))
            if page['tensor_blob'] is not None:
                self.checked_file(store/'blobs'/(page['tensor_blob']+'.safetensors'))
            self.contexts[fingerprint] = self.helper['analyze_flux_binding_minimal.py']['read_slot'](store, fingerprint, 'context')
        key = 'img_mod' if role == 'image' else 'txt_mod'
        return np.split(self.contexts[fingerprint][key].reshape(1, 1, 6*3072), 6, axis=-1)[5]

    def point(self, name, anchor, role):
        is0 = name == anchor
        own = (self.baselines[anchor] if is0 else self.forecasts[name])['hybrid']
        native = self.plan['anchors'][anchor]['stages'] if is0 else self.native[name]['stages']
        ref = lambda root, record: dict(root=root, record=record)
        native_root = 'prefixes' if is0 else 'source'
        return dict(anchor=anchor, endpoint='P0' if is0 else 'P1',
            own_input=ref('source', own['fields'][role]['mlp_input']),
            native_input=ref(native_root, native[role+'.mlp_input']),
            own_mlp=ref('source', own['fields'][role]['mlp']),
            own_exit=ref('source', own['fields'][role]['exit']),
            own_updated=ref('source', own['stages'][role+'.after_attention']),
            native_mlp=ref(native_root, native[role+'.mlp']),
            native_exit=ref(native_root, native[role+'.exit']),
            native_updated=ref(native_root, native[role+'.after_attention']),
            old_conditional=ref('conditional', self.conditional['anchors' if is0 else 'cases'][name][role]),
            own_context_cut=self.plan['anchors'][anchor]['entry'],
            native_context_cut=self.plan['anchors'][anchor]['entry'] if is0 else self.plan['parents']['base'])


def load_weights(source, bank, role):
    prefix = 'ff' if role == 'image' else 'ff_context'
    entries = {port: item for item in source.manifest['artifacts'] for port in item['parameter_ports']}
    result = {}
    for field, suffix in [('in', 'linear_in.weight'), ('out', 'linear_out.weight')]:
        item = entries[prefix+'.'+suffix]
        result[field] = bank.load(dict(path=item['path'], sha256=item['sha256'],
            descriptor=dict(sha256=item['source']['payload_sha256']), parameter_tensor=item['source']['tensor'], stored_shape=item['source']['shape']))
    return result


def worker():
    payload = Path.cwd()
    request = read(os.environ['SATURN_DEBUG_REQUEST'])
    c = request['config']
    if read(os.environ['SATURN_RUN_CONFIG']) != c or any(sha(payload/name) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('own-state request/source differs')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    import numpy as np
    h = helpers(payload)
    base = runpy.run_path(str(payload/'flux_binding_minimal.py'))
    save = base['save']
    source = Sources(c, h)
    prior_request = read(source.collection/'request.json')
    if any(request['source_hashes'][name] != prior_request['source_hashes'][name] for name in ('flux_reentry_reference.py', 'flux_receiver_norm_prediction.py', 'flux_arithmetic_reference.py')):
        raise ValueError('frozen upstream arithmetic changed')
    scratch = Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']
    out = scratch/'binding-minimal-result'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(out/'plan.json', c)
    save(out/'request.json', request)
    save(out/'parameter-artifacts.json', source.manifest)
    shutil.copyfile(payload/'retrieve_weights.py', out/'retrieve_weights.py')
    bank = h['flux_source_receiver.py']['RecipeBank'](source.manifest, c['model_path'])
    reference, law, math = [h[name] for name in ('flux_reentry_reference.py', 'flux_receiver_norm_prediction.py', 'flux_own_state_mlp_math.py')]
    B = reference['bf16']
    archive, states, checks = {}, {}, {}
    counts = dict(model_loads=0, native_calls=0, images=0, mlp_calculations=0, affine_calculations=0, swiglu_calculations=0, receiving_writes=0)
    def output(value):
        return dict(root='output', record=h['flux_numpy_archive.py']['record_bf16'](out, value, archive))
    for role in ('image', 'text'):
        weights = load_weights(source, bank, role)
        for anchor in ('half', 'half.image1'):
            stages = {field: source.array(dict(root='prefixes', record=source.plan['anchors'][anchor]['stages'][role+'.'+field])) for field in ('mlp_input','mlp_paired','mlp_activation','mlp')}
            for name in [anchor, *[spec['id'] for spec in c['cases'] if spec['anchor'] == anchor]]:
                point = source.point(name, anchor, role)
                own_gate, native_gate = [source.gate(point[key], role) for key in ('own_context_cut','native_context_cut')]
                own_updated, native_updated = [source.array(point[key]) for key in ('own_updated','native_updated')]
                old = math['mlp'](source.array(point['own_input']), stages, weights, 'anchored', reference, law)
                new = math['mlp'](source.array(point['own_input']), stages, weights, 'direct', reference, law)
                conditional = math['mlp'](source.array(point['native_input']), stages, weights, 'direct', reference, law)
                counts['mlp_calculations'] += 3
                counts['affine_calculations'] += 6
                counts['swiglu_calculations'] += 3
                values = {}
                for method, U, g, M in [('old_own',own_updated,own_gate,old), ('new_own',own_updated,own_gate,new),
                    ('old_conditional',native_updated,native_gate,source.array(point['old_conditional'])),
                    ('new_conditional',native_updated,native_gate,conditional),
                    ('native',native_updated,native_gate,source.array(point['native_mlp']))]:
                    values[method] = math['writeback'](U, g, M, B)
                    counts['receiving_writes'] += 1
                control_details = dict(old_own_mlp=math['equality'](old, source.array(point['own_mlp'])),
                    old_own_exit=math['equality'](values['old_own']['exit'], source.array(point['own_exit'])),
                    native_write_closure=math['equality'](values['native']['exit'], source.array(point['native_exit'])))
                # Direct-first result from the preselected native-input panel is
                # an exact source control, never a new validation specimen.
                if anchor == 'half' and name in ('half', 'row4seven_eighths'):
                    endpoint = 'P0' if name == anchor else 'P1'
                    item = source.selection['records'][role]['programs'][endpoint]['absolute.anchored']
                    root = Path(c['selection_collection'])/'extracted/binding-minimal-result'
                    from mdb.tensor_archive import load_tensor_archive
                    control_details['selected_conditional_join'] = math['equality'](conditional, load_tensor_archive(root, item))
                check = {key:value['raw_bits_exact'] for key,value in control_details.items()}
                point['control_details'] = control_details
                point['own_gate'], point['native_gate'] = output(own_gate), output(native_gate)
                point['states'] = {}
                for method, fields in values.items():
                    refs = dict(gated=output(fields['gated']))
                    refs['after_attention'] = point['own_updated' if method.endswith('_own') else 'native_updated']
                    # Existing fields keep their original source identity.
                    if method == 'old_own':
                        refs['mlp'], refs['exit'] = point['own_mlp'], point['own_exit']
                    elif method == 'old_conditional':
                        refs['mlp'] = point['old_conditional']
                        refs['exit'] = output(fields['exit'])
                    elif method == 'native':
                        refs['mlp'], refs['exit'] = point['native_mlp'], point['native_exit']
                    else:
                        refs['mlp'], refs['exit'] = output(fields['mlp']), output(fields['exit'])
                    point['states'][method] = refs
                states.setdefault(name, {})[role] = point
                checks.setdefault(name, {})[role] = check
                row = dict(kind='point_complete', point=name, role=role, checks=check, seconds=time.monotonic()-started, counts=dict(counts))
                with (out/'timeline.jsonl').open('a') as f:
                    f.write(json.dumps(row, sort_keys=True)+'\n')
                print('MDB_OWN_MLP '+json.dumps(row, sort_keys=True), flush=True)
                if time.monotonic()-started > c['max_wall_seconds']:
                    raise RuntimeError('finite saved own-state cut exceeded')
                del values, old, new, conditional
        del weights, stages
    expected = dict(model_loads=0, native_calls=0, images=0, mlp_calculations=36, affine_calculations=72, swiglu_calculations=36, receiving_writes=60)
    if counts != expected or set(states) != {'half','half.image1',*[spec['id'] for spec in c['cases']]}:
        raise ValueError('own-state cohort/work census differs')
    save(out/'states.json', states)
    save(out/'checks.json', checks)
    save(out/'source-binding.json', dict(source_custody=source.record['verification'], source_record_sha256=c['source_record_sha256'],
        previous_conditional_sha256=c['previous_conditional_sha256'], selection_report_sha256=c['selection_report_sha256'], scope=c['scope']))
    analysis = runpy.run_path(str(payload/'analyze_flux_own_state_mlp.py'))['analyze'](out, c)
    save(out/'analysis-v1.json', analysis)
    issues = [name+'.'+role+'.'+key for name,roles in checks.items() for role,items in roles.items() for key,value in items.items() if value is not True]
    if any(name in sys.modules for name in ('torch','diffusers','transformers')):
        raise ValueError('saved own-state calculation imported model framework')
    save(out/'source-reads.json', source.checked)
    summary = dict(schema='mdb-flux-own-state-mlp-v1', job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'],
        output_root=str(scratch), counts=counts, seconds=time.monotonic()-started, unique_arrays=len(archive),
        parameter_reads=bank.reads, parameter_cache_peak_bytes=bank.peak_retained, primary=analysis['primary'],
        mechanics_status='pending-independent-audit', instrument_status='qualified-control' if not issues else 'instrument-broken',
        instrument_issues=issues, terminal_status='unassessed', scope=c['scope'])
    save(out/'summary.json', summary)
    save(out/'files.json', {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'], request_sha256=request['request_sha256'], output_root=str(out),
        files_manifest_sha256=sha(out/'files.json'), summary_sha256=sha(out/'summary.json')), sort_keys=True), flush=True)


def stage(run, previous, primitive):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    here = Path(__file__).resolve().parent
    base = runpy.run_path(str(here/'flux_binding_minimal.py'))
    save = base['save']
    collector = runpy.run_path(str(here/'collect_flux_remote_directory.py'))
    # Both evidence studies are explicit inputs; no checkout discovery.
    if read(previous/'verification-v2.json')['mechanics_status'] != 'verified' or read(primitive/'verification-final-v2.json')['instrument_status'] != 'qualified':
        raise ValueError('verified predecessor studies required')
    source = collector['checked_pointer'](previous/'native-remote-pointer-v1.json')
    selection = collector['checked_pointer'](primitive/'panel/remote-pointer-v2.json')
    old = read(previous/'run-02/request.json')
    prim = read(primitive/'run-01/request.json')['config']
    base['bind_environment'](old['execution_environment'])
    c = dict(source_collection=source['collection'], source_record_sha256=read(previous/'native-remote-pointer-v1.json')['record_sha256'],
        source_audit=prim['source_audit'], source_audit_sha256=prim['source_audit_sha256'],
        previous_conditional_root=prim['previous_mlp_root'], previous_conditional_sha256=prim['previous_mlp_conditional_sha256'],
        previous_conditional_audit_sha256=prim['previous_mlp_audit_sha256'],
        selection_collection=selection['collection'], selection_record_sha256=read(primitive/'panel/remote-pointer-v2.json')['record_sha256'],
        selection_report_sha256=sha(primitive/'panel/panel-report-v2.json'), selection_request_sha256=selection['verification']['request_sha256'],
        model_path=prim['model_path'], model_identity=prim['model_identity'], cases=old['config']['cases'],
        output_scratch='/mnt/big/mdb/research', directory_custody=True, max_wall_seconds=480,
        primary_hypothesis='Direct-first/anchored-output lowers mean image complete-exit finite-response error against the existing hybrid on four saved states. Per-case/text/mixed countertrends retained; no accuracy gate.',
        scope='Saved own-state integration: unchanged stored hybrid upstream predictions feed old/new MLP programs; true P0/P1, gated write, complete exit and signed finite-addition account retained. Conditional native-input worlds are separate diagnostics. Four known targets/two known receivers,0new native/model/framework calls/images. Dense original weights and historical calibration remain supplied; no default, generalization or speed claim.')
    if any(sha(here/name) != old['source_hashes'][name] for name in ('flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py')):
        raise ValueError('frozen reference changed')
    run.mkdir(parents=True, exist_ok=False)
    save(run/'plan.json', c)
    names = ('flux_binding_minimal.py','flux_family_custody.py','flux_source_receiver.py','flux_reentry_reference.py','flux_receiver_norm_prediction.py',
        'flux_arithmetic_reference.py','flux_attention_anchor.py','flux_numpy_archive.py','analyze_flux_binding_minimal.py','flux_own_state_mlp_math.py','analyze_flux_own_state_mlp.py')
    extras = [(here/name,name) for name in names]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')]
    preflight = run/'preflight.v1.json'
    save(preflight, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name,*[name for _,name in extras],preflight.name],
        path_token_allowlist=[c['source_collection'],str(Path(c['source_audit']).parent),c['previous_conditional_root'],c['selection_collection'],c['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((preflight,preflight.name))
    request = dict(schema='mdb-own-state-mlp-request-v1', config=c, needs=dict(cpu=True,host='beast'), task_family='analysis', backend='numpy',device='cpu',
        dtype='float64-with-BF16-boundaries', execution_environment=old['execution_environment'], source_hashes={name:sha(p) for p,name in [(Path(__file__),Path(__file__).name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=36,logical_branches=4,logical_checkpoint_replays=0,max_local_work_items=36,
            extra=dict(task_family='analysis'),checkpoint_policy='Stored-own-input MLP replacement,12old-control reconstructions+12newown+12newconditionalMLPs;60finitewrites including source/native diagnostic worlds. TrueP0/P1and signedtransport,0native/framework/images. No scientific metric gates.'),
        resource_exception=dict(ram_mb=3072,cpu_threads=4,disk_gb=3,est_wall_s=120,calibration_job='job-d37341c851a5',measured_peak_rss_mb=1474.8,
            reason='This exact 36-MLP/60-write geometry measured1474.8MBRSS and73.33sprocess on job-d37341c851a5; separate recomputation measured1498MB.3GB/4CPU retains headroom without the previous6GB reservation. CPU-only, expectedunder3min,480worker/720outer,noretries; raw evidence staysBeast.'))
    request['request_sha256'] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/'request.json', request)
    stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    base = runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')))
    r = read(run/'request.json')
    base['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/name) != value for name,value in r['source_hashes'].items()):
        raise ValueError('already submitted or source changed')
    command = worker_command(Path(__file__).name)
    command.insert(1,'CUDA_VISIBLE_DEVICES=')
    command.insert(command.index('python')+1,'-B')
    job = guarded_mrun_launch(command,request=r,model=None,task='analysis',experiment='mdb-flux-own-state-mlp',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',
        timeout_s=720,retry_on_kill=False,preflight=True,**{key:r['resource_exception'][key] for key in ('ram_mb','cpu_threads','disk_gb','est_wall_s')},
        note='Saved own-state MLP integration, firstdirect/outputanchored vs existinghybrid.36NumPyMLPs,12exactoldcontrols,60gatedresidualwrites,4knowntargets/2anchors/2streams.TrueP0/P1,signedfinitewriteandmixedaccounts;0native/framework/images.3GB/4CPU,expectedunder3min,480worker/720outer,noretries.')
    base['save'](run/'submission.json',dict(job_id=str(job)))
    return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1':
        worker()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('run',type=Path)
        parser.add_argument('previous',type=Path,nargs='?')
        parser.add_argument('primitive',type=Path,nargs='?')
        parser.add_argument('--submit',action='store_true')
        args=parser.parse_args()
        print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(),args.previous.resolve(),args.primitive.resolve()),sort_keys=True))
