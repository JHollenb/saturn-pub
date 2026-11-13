"""Recover completed saved carrier-cross execution after metadata serialization.

stage RUN --failed FAILED_NATIVE_RUN
submit RUN
One finite CPU-only recovery/audit; never executes a model or new native future.
The failed root and original scheduler outcome remain immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ARMS = ('I.P_T.P', 'I.P_T.Q', 'I.Q_T.P', 'I.Q_T.Q')
LABELS = ('entry', 'after_joint1', 'before_project3', 'project3_input', 'after_project3', 'after_step3_joint0', 'final')
COUNTS = dict(loads=1, encodes=0, noise_draws=0, native_transitions=864, native_blocks=784, acts=48, receipts=912, images=16)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def relative(name):
    value = PurePosixPath(name)
    if not isinstance(name, str) or not name or '\\' in name or value.is_absolute() or '..' in value.parts or value.as_posix() != name:
        raise ValueError('unsafe original physical path')
    return value


def physical_inventory(root):
    """Hash all actual regular files, including malformed partial metadata."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('physical source must be a real directory')
    files, size = {}, 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirs):
            raise ValueError('symlink directory in failed source')
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('nonregular source file')
            key = path.relative_to(root).as_posix(); relative(key)
            size += path.stat().st_size
            if len(files) >= 50000 or size > 4 * 1024**3:
                raise ValueError('declared finite recovery inventory exceeded (50000files/4GiB)')
            files[key] = dict(sha256=sha(path), bytes=path.stat().st_size)
    return dict(schema='mdb-failed-native-physical-inventory-v1', root=str(root), files=dict(sorted(files.items())), file_count=len(files), total_bytes=size)


def request_identity(request):
    unsigned = dict(request); claimed = unsigned.pop('request_sha256')
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != claimed:
        raise ValueError('original request self seal differs')
    return claimed


def completion_rows(timeline, contexts):
    rows = [row for row in timeline if row['kind'] == 'image_complete']
    expected = {(case['id'], arm) for case in contexts for arm in ARMS}
    if len(rows) != 16 or {(row['case'], row['arm']) for row in rows} != expected or rows[-1]['counts'] != COUNTS:
        raise ValueError('source does not prove all16 original native images/census')
    return rows


def execution_environment_from_log(log):
    values = [json.loads(line[len('MDB_EXECUTION_ENVIRONMENT='):]) for line in log.splitlines()
              if line.startswith('MDB_EXECUTION_ENVIRONMENT=')]
    if len(values) > 1:
        raise ValueError('ambiguous original execution environment log records')
    return values[0] if values else None


def reconstruct_trajectory(rows, evidence):
    trajectory = []
    for row in rows:
        for label in LABELS:
            fingerprint = row['marks'][label]
            cut, descriptors = evidence.cut(fingerprint)
            trajectory.append(dict(case=row['case'], arm=row['arm'], label=label, cut=fingerprint,
                point=dict(cut['execution_point']), payload_descriptors={slot: descriptors[slot] for slot in ('image', 'text', 'latent', 'conditioning')}))
    return trajectory


def build_recovered(source, destination, config, payload):
    """Hardlink original bytes, reconstruct only metadata from original CAS."""
    source, destination, payload = Path(source), Path(destination), Path(payload)
    proof_path = payload / 'original-physical-inventory.json'
    proof = read(proof_path)
    if sha(proof_path) != config['source_physical_inventory_sha256'] or proof != physical_inventory(source):
        raise ValueError('failed source physical inventory changed after staging')
    original_request = read(source / 'request.json')
    if request_identity(original_request) != config['original_request_sha256'] or read(source / 'plan.json') != original_request['config']:
        raise ValueError('failed source request/config differs')
    job_path, log_path = payload / 'original-job.json', payload / 'original-worker.log'
    job = read(job_path)
    if (sha(job_path) != config['original_job_record_sha256'] or sha(log_path) != config['original_log_sha256']
            or job['state'] != 'failed' or job['job_id'] != config['original_job_id']):
        raise ValueError('original failed job/log identity differs')
    timeline = [json.loads(line) for line in (source / 'timeline.jsonl').read_text().splitlines()]
    images = completion_rows(timeline, original_request['config']['contexts'])
    log_events = [json.loads(line[len('MDB_CARRIER_CROSS '):]) for line in log_path.read_text().splitlines() if line.startswith('MDB_CARRIER_CROSS ')]
    completion_rows(log_events, original_request['config']['contexts'])
    seals = [row for row in timeline if row['kind'] == 'sealed']
    prefix = {key: int(key == 'loads') for key in COUNTS}
    if len(seals) != 1 or seals[0]['counts'] != prefix or 'ports.json' in proof['files'] or 'trajectory.json' not in proof['files']:
        raise ValueError('source failure/recovery scope differs')
    for name in ('rows.json', 'fields.json', 'actions.json', 'source-recipes.json', 'source-parents.json'):
        if name not in proof['files']:
            raise ValueError('missing core retained evidence: ' + name)
    rows = read(source / 'rows.json')
    if len(rows) != 16 or len(read(source / 'fields.json')) != 16 or len(read(source / 'actions.json')) != 48:
        raise ValueError('saved native core census incomplete')
    destination.mkdir(parents=True, exist_ok=False)
    mapping = {}
    for name in proof['files']:
        retained = 'source-partial-trajectory.json.txt' if name == 'trajectory.json' else 'source-original-files.json' if name == 'files.json' else name
        target = destination / relative(retained); target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / relative(name), target)
        mapping[name] = retained
    # Every write below is exclusive: no retained hardlinked original is altered.
    h = runpy.run_path(str(HERE / 'flux_binding_generalization_audit.py'))
    evidence = h['Evidence'](destination, {mapping[name]: item['sha256'] for name, item in proof['files'].items()})
    trajectory = reconstruct_trajectory(rows, evidence)
    save_new(destination / 'trajectory.json', trajectory)
    save_new(destination / 'ports.json', [])
    recovered_artifacts = {}
    for name in ('parameter-artifacts.json', 'retrieve_weights.py', 'retrieve_flux_encoder.py', 'source-binding.json'):
        expected = original_request['source_hashes'][name]
        original = payload / 'original-payload' / name
        if sha(original) != expected:
            raise ValueError('original request-sealed missing artifact differs: ' + name)
        if (destination / name).exists():
            if sha(destination / name) != expected:
                raise ValueError('retained artifact disagrees with original source seal')
        else:
            with original.open('rb') as inp, (destination / name).open('xb') as out:
                shutil.copyfileobj(inp, out)
            recovered_artifacts[name] = expected
    if not (destination / 'native-contract.json').exists():
        environment = execution_environment_from_log(log_path.read_text())
        packages = {} if environment is None else environment.get('packages', {})
        save_new(destination / 'native-contract.json', dict(schema='mdb-recovered-request-bound-native-contract-v1',
            device=None, device_backend=original_request.get('device'), dtype=original_request.get('dtype'), batch_size=original_request.get('batch_size'),
            torch=packages.get('torch'), diffusers=packages.get('diffusers'), bf16_reduced_reduction=None,
            cuda_driver=None if environment is None else environment.get('cuda_driver'),
            observed_execution_environment=environment, configured_execution_environment=original_request['execution_environment'],
            source_request_sha256=config['original_request_sha256'], original_log_sha256=config['original_log_sha256'],
            status='Versions/driver copied only from the original MDB_EXECUTION_ENVIRONMENT log; torch device-name/precision flag fields remain null rather than inferred.'))
    from mdb_lanes.forecast_transaction import verify_transaction
    transaction = verify_transaction(destination / 'transaction', expected_seal_sha256=sha(destination / 'transaction/seal.json'))
    if not transaction['verified'] or transaction['phase'] != 'closed':
        raise ValueError('original native transaction did not close')
    save_new(destination / 'summary.json', dict(schema='mdb-binding-carrier-cross-recovered-native-v1', job_id=config['original_job_id'],
        request_sha256=config['original_request_sha256'], output_root=str(destination), counts=images[-1]['counts'], counts_at_seal=prefix,
        seal=seals[0]['seal'], transaction=transaction, seconds_to_last_image=images[-1]['seconds'],
        original_scheduler_state='failed', mechanics_status='recovered-metadata-pending-independent-audit', terminal_status='unassessed',
        recovered_metadata=True, scope='All16 original images existed before metadata serialization failed; no native execution during recovery.'))
    for original, name in ((proof_path, 'source-physical-inventory.json'), (job_path, 'source-failed-job.json'), (log_path, 'source-failed-worker.log')):
        with original.open('rb') as inp, (destination / name).open('xb') as out:
            shutil.copyfileobj(inp, out)
    recovery = dict(schema='mdb-carrier-cross-saved-recovery-v1', original_job_id=config['original_job_id'], original_job_state='failed',
        original_request_sha256=config['original_request_sha256'], original_source_root=str(source),
        original_job_record_sha256=config['original_job_record_sha256'], original_log_sha256=config['original_log_sha256'],
        source_physical_inventory_sha256=config['source_physical_inventory_sha256'], original_to_recovered=mapping,
        optional_port_index_status='unavailable-not-persisted', raw_attention_arrays='retained-unattributed; no per-arm index manufactured',
        trajectory_reconstructed=dict(entries=len(trajectory), source='Complete rows.marks and retained native CAS manifests only; original partial bytes retained.'),
        recovered_request_sealed_artifacts=recovered_artifacts,
        native_contract_unknowns=[name for name in ('device', 'torch', 'diffusers', 'bf16_reduced_reduction') if read(destination / 'native-contract.json').get(name) is None],
        new_native_execution=False, new_images=0, source_original_request_retained=True)
    save_new(destination / 'recovery.json', recovery)
    save_new(destination / 'files.json', {path.relative_to(destination).as_posix(): sha(path) for path in sorted(destination.rglob('*')) if path.is_file()})
    if physical_inventory(source) != proof:
        raise ValueError('original failed evidence changed during recovery')
    return recovery


def worker():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '2'
    request = read(os.environ['SATURN_DEBUG_REQUEST']); config = request['config']; request_identity(request)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(sha(Path(name)) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('CPU isolation/source seal differs')
    start = time.monotonic(); job_root = Path(config['output_scratch']) / os.environ['MRUN_JOB_ID']
    recovered = job_root / 'recovered-native'
    recovery = build_recovered(config['failed_root'], recovered, config, Path.cwd())
    ah = runpy.run_path(str(HERE / 'flux_binding_carrier_cross_audit.py'))
    out = job_root / 'binding-minimal-result'; out.mkdir(parents=True, exist_ok=False)
    report = ah['analyze'](recovered, config['prior_root'], config['main_root'], out / 'audit.json')
    for path in sorted(recovered.iterdir()):
        if path.is_file() and path.name != 'files.json' and path.suffix in ('.json', '.jsonl'):
            name = 'source-' + path.name if path.name in ('plan.json', 'request.json', 'summary.json', 'timeline.jsonl') else path.name
            shutil.copyfile(path, out / name)
    for name in ('retrieve_weights.py', 'retrieve_flux_encoder.py', 'source-failed-worker.log'):
        shutil.copyfile(recovered / name, out / name)
    for row in read(recovered / 'rows.json'):
        target = out / row['png']; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(recovered / row['png'], target)
    shutil.copytree(recovered / 'transaction', out / 'source-transaction')
    roots = dict(native=str(recovered), prior=config['prior_root'], main=config['main_root'], original_failed=config['failed_root'])
    manifests = {key: read(Path(value) / 'files.json') for key, value in roots.items() if key != 'original_failed'}
    manifests['original_failed'] = {name: item['sha256'] for name, item in read('original-physical-inventory.json')['files'].items()}
    save_new(out / 'external-evidence.json', dict(host='beast', roots=roots, manifests=manifests,
        scope='Recovered native and original failed bytes remain near compute. Original partial trajectory/raw unindexed arrays preserved; checksum-verified retrieval.'))
    shutil.copyfile(HERE / 'retrieve_reader_evidence.py', out / 'retrieve_evidence.py')
    save_new(out / 'plan.json', config); save_new(out / 'request.json', request)
    save_new(out / 'summary.json', dict(schema='mdb-binding-carrier-cross-recovery-compact-v1', job_id=os.environ['MRUN_JOB_ID'],
        request_sha256=request['request_sha256'], output_root=str(out), source_job=config['original_job_id'], original_scheduler_state='failed',
        mechanics_status=report['mechanics_status'], optional_port_index_status='unavailable-not-persisted', seconds=time.monotonic() - start,
        recovered_native_root=str(recovered), new_execution_counts=dict(model_loads=0, native_primitives=0, images=0), terminal_status='unassessed'))
    save_new(out / 'files.json', {path.relative_to(out).as_posix(): sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    print('MDB_DIRECTORY_READY ' + json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'], request_sha256=request['request_sha256'],
        output_root=str(out), files_manifest_sha256=sha(out / 'files.json'), summary_sha256=sha(out / 'summary.json')), sort_keys=True), flush=True)


def stage(run, failed):
    from mdb_lanes.job_support import scheduler_api, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    old = read(failed / 'request.json'); request_identity(old); jid = read(failed / 'submission.json')['job_id']
    for name, expected in old['source_hashes'].items():
        relative(name)
        if sha(failed / 'payload' / name) != expected: raise ValueError('original staged payload differs: ' + name)
    api = scheduler_api(); status, raw, _ = api.request('GET', '/api/jobs/' + jid, timeout_s=20)
    raw = raw.encode() if isinstance(raw, str) else raw
    if status >= 400 or json.loads(raw)['state'] != 'failed': raise ValueError('terminal failed native source required')
    custody = runpy.run_path(str(HERE / 'flux_family_custody.py'))
    with custody['_open_logs'](api, '/api/jobs/' + jid + '/logs?offset=0&wait_s=1') as stream:
        log = stream.read(16 << 20)
        if stream.read(1): raise ValueError('original log exceeds declared16MiB metadata bound')
    events = [json.loads(line[len(b'MDB_CARRIER_CROSS '):]) for line in log.splitlines() if line.startswith(b'MDB_CARRIER_CROSS ')]
    completion_rows(events, old['config']['contexts'])
    failed_root = str(Path(old['config']['output_scratch']) / jid / 'binding-minimal-result')
    # Run only the read-only inventory helper over existing saved files on the data host.
    # __file__ binds definitions only; inventory uses no sibling imports or model code.
    code = 'import json\nfrom pathlib import Path\nns={"__file__":' + repr(str(Path(__file__))) + ',"__name__":"recovery_inventory"}\nexec(' + repr(Path(__file__).read_text()) + ',ns)\nprint(json.dumps(ns["physical_inventory"](Path(' + repr(failed_root) + '))))\n'
    physical = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code, text=True, capture_output=True, check=True).stdout)
    roots = {key: old['config'][key + '_root'] for key in ('prior', 'main')}
    code = 'import json,hashlib\nfrom pathlib import Path\nroots=' + repr(roots) + '\nprint(json.dumps({k:hashlib.sha256((Path(v)/"files.json").read_bytes()).hexdigest()for k,v in roots.items()}))\n'
    source_hashes = json.loads(subprocess.run(['ssh', 'beast', 'python3', '-I', '-'], input=code, text=True, capture_output=True, check=True).stdout)
    if any(source_hashes[key] != old['config'][key + '_manifest_sha256'] for key in roots): raise ValueError('prior/main source inventory changed')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'original-job.json').write_bytes(raw); (run / 'original-worker.log').write_bytes(log)
    save_new(run / 'original-physical-inventory.json', physical)
    config = dict(failed_root=failed_root, original_job_id=jid, original_request_sha256=old['request_sha256'],
        original_job_record_sha256=sha(run / 'original-job.json'), original_log_sha256=sha(run / 'original-worker.log'),
        source_physical_inventory_sha256=sha(run / 'original-physical-inventory.json'), prior_root=roots['prior'], main_root=roots['main'],
        prior_manifest_sha256=source_hashes['prior'], main_manifest_sha256=source_hashes['main'], output_scratch='/mnt/big/mdb/research', directory_custody=True)
    save_new(run / 'plan.json', config)
    names = ('flux_binding_generalization.py', 'flux_binding_generalization_audit.py', 'flux_binding_context_controls.py',
             'flux_binding_context_controls_audit.py', 'flux_binding_matched_dose_audit.py', 'flux_binding_carrier_cross_audit.py',
             'flux_family_custody.py', 'retrieve_reader_evidence.py')
    extras = [(HERE / name, name) for name in names]
    extras += [(run / name, name) for name in ('original-job.json', 'original-worker.log', 'original-physical-inventory.json')]
    extras += [(failed / 'payload' / name, 'original-payload/' + name) for name in old['source_hashes']]
    pre = run / 'preflight.v1.json'
    save_new(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[Path(__file__).name, *[name for _, name in extras], pre.name],
        path_token_allowlist=[failed_root, *roots.values(), old['config']['model_path'], config['output_scratch'], 'binding-minimal-result', 'recovered-native']))
    extras.append((pre, pre.name))
    request = dict(schema='mdb-binding-carrier-cross-saved-recovery-request-v1', worker_script=Path(__file__).name, config=config,
        execution_environment=old['execution_environment'], needs=dict(cpu=True, host='beast'), task_family='analysis', backend='numpy', device='cpu', dtype='float32',
        source_hashes={name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        resource_exception=dict(ram_mb=1536, vram_mb=0, cpu_threads=2, disk_gb=2, est_wall_s=180,
            reason='Model-free hardlink recovery of already completed16native suffixes;112CAS trajectory rows, retained failed files/partial metadata,32entrywitnesses/48Acts/8replays/PP-QQ trajectories.1.5GBRAM/2CPU/180sestimate/480souter;0CUDA/models/newfutures.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=16, logical_branches=1,
            logical_checkpoint_replays=0, max_local_work_items=32, extra=dict(task_family='analysis', unit='one retained arm plus its core proof/trajectory audit'),
            checkpoint_policy='Recover metadata only from original retained bytes/CAS; preserve failed source and original outcome.16arm units,32source-entrywitnesses,48typedActs,112landmarks,8replays.0models/nativeforwards/retries.'))
    request['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save_new(run / 'request.json', request)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    stage_worker(run / 'payload', worker=Path(__file__), config=config, request=request, additional_files=extras)
    return dict(request_sha256=request['request_sha256'], original_job=jid, physical_files=physical['file_count'], physical_bytes=physical['total_bytes'], new_model_calls=0)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    request = read(run / 'request.json'); request_identity(request)
    runpy.run_path(str(HERE / 'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    if (run / 'submission.json').exists() or any(sha(run / 'payload' / name) != digest for name, digest in request['source_hashes'].items()):
        raise ValueError('changed/already submitted recovery')
    command = worker_command(request['worker_script']); command.insert(1, 'CUDA_VISIBLE_DEVICES='); command.insert(command.index('python') + 1, '-B')
    resources = {key: request['resource_exception'][key] for key in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')}
    job = guarded_mrun_launch(command, request=request, model=None, task='analysis', experiment='mdb-binding-carrier-cross-recovery',
        payload=run / 'payload', gpu=False, needs=request['needs'], prefer_host='beast', timeout_s=480, preflight=True, retry_on_kill=False, **resources,
        note='Saved-only recovery of completed failed native carrier-cross metadata:16existingimages/112CASlandmarks/32entrywitnesses/48Acts/8replays; originalfailedroot/log/job preserved, ports unindexed.0models/CUDA/nativefutures;CPU2/RAM1.5GB/180sestimate/480souter,0retries.')
    save_new(run / 'submission.json', dict(job_id=str(job))); return dict(job_id=str(job))


if __name__ == '__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER') == '1': worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='mode', required=True)
        s = commands.add_parser('stage'); s.add_argument('run', type=Path); s.add_argument('--failed', type=Path, required=True)
        u = commands.add_parser('submit'); u.add_argument('run', type=Path)
        args = parser.parse_args()
        print(json.dumps(stage(args.run.resolve(), args.failed.resolve()) if args.mode == 'stage' else submit(args.run.resolve()), sort_keys=True))
