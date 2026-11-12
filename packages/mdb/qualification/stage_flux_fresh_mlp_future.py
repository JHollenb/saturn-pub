"""Stage one frozen, source-assisted fresh-receiver transfer with nine image futures."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run, own_study, hybrid_study):
    from mdb import parameter_artifacts
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here/'flux_binding_minimal.py')); sha, save = h['sha'], h['save']
    read = lambda p: json.loads(p.read_text())
    own = read(own_study/'run-02/request.json'); original = read(hybrid_study/'run-01/request.json')
    proof = read(own_study/'verification-v1.json')
    if proof['mechanics_status'] != 'verified' or proof['instrument_status'] != 'qualified': raise ValueError('own-state development unverified')
    for name in ('flux_reentry_reference.py', 'flux_receiver_norm_prediction.py', 'flux_arithmetic_reference.py', 'flux_own_state_mlp_math.py'):
        if sha(here/name) != own['source_hashes'][name]: raise ValueError('selected numerical program changed')
        if name in original['source_hashes'] and sha(here/name) != original['source_hashes'][name]: raise ValueError('upstream program changed')
    if own['config']['source_record_sha256'] != read(hybrid_study/'native-remote-pointer-v1.json')['record_sha256']: raise ValueError('historical source join differs')
    source_record = read(hybrid_study/'native-remote-pointer-v1.json')['record']
    old = original['config']; h['bind_environment'](original['execution_environment'])
    worker = here/'flux_fresh_mlp_future.py'; api = runpy.run_path(str(worker))
    run.mkdir(parents=True, exist_ok=False)
    c = dict(schema='mdb-fresh-mlp-futures-plan-v1', seed=81119, cases=api['cases'](), programs=dict(old='frozen hybrid anchored-first/anchored-output', new='same upstream with direct-first/anchored-output'),
        **{key: old[key] for key in ('model', 'model_path', 'checkpoint_sha256', 'absolute_ports', 'policy')},
        source_collection=own['config']['source_collection'], source_record_sha256=own['config']['source_record_sha256'], output_scratch='/mnt/big/mdb/research', worker_wall_s=720, directory_custody=True,
        selection=dict(own_request_sha256=own['request_sha256'], own_proof_sha256=sha(own_study/'verification-v1.json'), historical_request_sha256=original['request_sha256'], primitive_sources={name: own['source_hashes'][name] for name in ('flux_reentry_reference.py', 'flux_receiver_norm_prediction.py', 'flux_arithmetic_reference.py', 'flux_own_state_mlp_math.py')}),
        hypotheses=dict(primary='Mean image complete-exit paired-response relative RMS is lower for new than old on the two targets; retain exceptions and baseline bias without an all-row accuracy gate.',
            image='Measure installed RGB endpoint and each program own-P0 paired response; expected state gain may shrink or reverse downstream.',
            color='Frozen exploratory direction: row4heavy minus row11heavy decreases mean red-minus-blue on fixed left/right halves and later baseline-derived object supports; supplied historical D may include other changes.',
            mask_rule='Post-seal native half-image scoring only: foreground min(R,G,B)<220 or max-min>25, restrict y=[32,480), x=[32,256) for left and [256,480) for right. Empty/misidentified support is instrument-incomplete. Background is complement; retain fixed halves and top/bottom32border.',
            scope='New seed81119 native receiver under original cached conditioning. Historical D=Tedited−Tbase on rows4/11 is supplied, not freshly resolved from a prompt pair. Every T(d)=B(Tfresh+dD) formed independently from untouched fresh text. One new native half-state calibration shared by both already frozen laws. Two targets and six candidate images sealed before targets. Nine image futures; all native prefix, calibration, suffix and VAE work counted.'),
        scope='Source-assisted receiver transfer with explicit fresh calibration, original dense parameters and native suffix/VAE. New receiver, same historical two-row intervention and numerical laws. No new semantic-address discovery, calibration-free portability or speed claim.')
    save(run/'plan.json', c); save(run/'selection.json', dict(own_proof=proof, source_record=source_record))
    names = ('flux_binding_minimal.py', 'flux_family_custody.py', 'flux_hybrid_future.py', 'flux_reentry_reference.py', 'flux_receiver_norm_prediction.py', 'flux_arithmetic_reference.py', 'flux_own_state_mlp_math.py', 'flux_fresh_mlp_program.py', 'flux_precision_cut.py', 'flux_response_campaign.py', 'flux_response_validate.py')
    extras = [(here/name, name) for name in names]+[(Path(parameter_artifacts.__file__), 'retrieve_weights.py'), (run/'plan.json', 'plan.json'), (run/'selection.json', 'selection.json')]
    pre = run/'preflight.v1.json'; save(pre, dict(schema_version='mrun-preflight-config-v1', hashed_roots=[worker.name, *[name for _, name in extras], pre.name], path_token_allowlist=[c['source_collection'], c['model_path'], c['output_scratch'], 'binding-minimal-result']))
    extras.append((pre, pre.name))
    request = dict(schema='mdb-fresh-mlp-futures-request-v1', config=c, model=c['model'], task_family='forward', backend='diffusers', device='cuda', dtype='bfloat16', batch_size=1, height=512, width=512, needs=dict(cuda=True, host='beast'), execution_environment=original['execution_environment'],
        source_hashes={name: sha(p) for p, name in [(worker, worker.name), *extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=9, logical_branches=9, logical_checkpoint_replays=9, max_local_work_items=16,
            checkpoint_policy='58native prefixsteps+1shared calibration before prediction; six candidate ownP0/P1 image futures before two target blocks/images. Nineimages/538transitions/15Acts/37receipts/486nativeblocks.1load,1noise,0encoders,0retries. Freeze writes then PNG/fixed-region forecasts; baseline-derived masks are later scoring.'),
        resource_exception=dict(ram_mb=6144, vram_mb=10500, cpu_threads=4, disk_gb=10, est_wall_s=240,
            calibration_job='job-0b1961445df7', measured_peak_rss_mb=4126.2, measured_peak_vram_mb=9338.0,
            reason='This exact9image/486block geometry measured175.05sprocess,4126.2MBRSS/9338MBVRAM on job-0b1961445df7. Future6GBRAM retains headroom with10.5GBVRAM;240s estimate,720worker/900outer,noretries. Executed13GBrequest remains immutable.'))
    request['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run/'request.json', request); stage_worker(run/'payload', worker=worker, config=c, request=request, additional_files=extras)
    return dict(run=str(run), request_sha256=request['request_sha256'], expected_counts=api['expected_counts']())


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h = runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')))
    r = json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/name) != digest for name, digest in r['source_hashes'].items()): raise ValueError('changed or submitted payload')
    job = guarded_mrun_launch(worker_command('flux_fresh_mlp_future.py'), request=r, model=r['model'], task='forward', experiment='mdb-flux-fresh-mlp-futures', payload=run/'payload', prefer_host='beast', gpu=True, needs=r['needs'], timeout_s=900, retry_on_kill=False, preflight=True,
        **{key:r['resource_exception'][key] for key in ('ram_mb', 'vram_mb', 'cpu_threads', 'disk_gb', 'est_wall_s')},
        note='Fresh seed81119 receiver, fixed historical two-row displacement, explicit1half-state calibration shared by frozenold/newMLPlaws.58prefixsteps then six candidate ownP0/P1 images BEFORE2native targetJ1/images.9images/538transitions/486blocks/15Acts/37receipts;1load/1noise/0encoders/0retries.6GBRAM/10.5GBVRAM from measured4126/9338MB,240sestimate/720worker/900outer. Fullstates stayBeast; nativebaseline scoredafterimageseal.')
    h['save'](run/'submission.json', dict(job_id=str(job))); return dict(job_id=str(job))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('own_study',type=Path,nargs='?'); p.add_argument('hybrid_study',type=Path,nargs='?'); p.add_argument('--submit',action='store_true'); a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.own_study.resolve(),a.hybrid_study.resolve()),sort_keys=True))
