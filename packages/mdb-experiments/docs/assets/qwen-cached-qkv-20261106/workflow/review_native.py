"""Record the exact finite source/resource review after metadata-only staging."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
QUAL = BASE / 'saturn-pub/packages/mdb/qualification'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frontend = QUAL / 'stage_qwen_cached_qkv.py'
    if sha(frontend) != 'bf5ccf76dda3059d3b222981efec483c8fe5d57bb93beb182ca6592e0edaa460':
        raise ValueError('Qualified frontend changed')
    module = runpy.run_path(str(frontend))
    run = ROOT / 'run-01'
    request = module['inspect_run'](run)
    protocol = json.loads((ROOT / 'selection-v1/protocol.json').read_text())
    if sha(ROOT / 'selection-v1/protocol.json') != '8c53ae9417ba40420e806cb634d59931fb0131f95d0329e5b34e078429503bff':
        raise ValueError('Scientific selection changed')
    expected = {
        'producer-qualification-v2/qualification.json': '3f40ba8f445fb1fdaaf20a7f9aebe78909f35a3fdf4b821cf1d0a0215c1ffd86',
        'audit-qualification-v1/qualification.json': '1954a405a694a949e06290b13860abc53064bb82b8bcd5ac2cb4da22d715401a',
        'controller-qualification-v1/qualification.json': 'e2e1a0462cb6717df92ea04b9db8794311b98e39dfde064321d58814c43dc33c',
    }
    for relative, digest in expected.items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Qualification changed: ' + relative)
    command = module['command'](request)
    if command[-2:] != ['--config', 'saturn_run_config.json'] or request['needs'] != {'cuda': True, 'host': 'beast'}:
        raise ValueError('Actual execution entry/device changed')
    if request['workload_geometry'] != protocol['physical_plan'] or protocol['scientific_gates']:
        raise ValueError('Declared physical/scientific plan changed')
    source_arrays = protocol['source_arrays']
    if len(source_arrays) != 13 or any('/native/' in row['path'] or '/minus/' in row['path'] for row in source_arrays.values()):
        raise ValueError('Own-positive source selection differs')
    profile = request['config']
    review = dict(schema='mdb-cached-qkv-finite-source-review-v1', request_sha256=request['request_sha256'],
        review_status='ready-for-finite-run', blocking_issues=[], reviewer='root source/ABI/resource review; not an Astra admission gate',
        source_and_wheel_files_authenticated=len(request['source_hashes']) + len(request['runtime_wheels']),
        qualification_receipts=expected, command=command, selection_sha256=sha(ROOT / 'selection-v1/protocol.json'),
        source_scope=dict(arrays=13, archive_bytes=sum(row['bytes'] for row in source_arrays.values()),
            owned_functional_positive=True, no_negative_or_native_frontier=True,
            parent_forecast_sha256=profile['parent_forecast_sha256'], original_layout_restoration='Dense nonoverlapping empty_strided, logical raw-bit comparison, explicit same-stride sign output'),
        source_review=['Read complete predictor, shared evaluator, adapter and frontend. Candidate forms sQ/sK/sV and recomputes attention/O/R/postnorm/gate/up/SiLU/down; no stale R/G/U.',
            'One global seal covers complete cached/full outputs and response hashes before baseline and both edited native targets; actual consumed QKV arrays and geometry retained.',
            'Seven SDPA and43linear operations:2cached/2full/3native. Native modules guarded before seal; source and original parameters immutable before/after.',
            'Aggregate SDPA call sidecar is written post-target. Its actual-call joins are distinct from the pretarget array/response seal.',
            'Reader exercises actual installed Atlas/backend schema and exact fixture serializer, Tx, layouts and operands. Original recipe streaming remains the near-data audit task.'],
        resource_basis=request['resource_exception'], scheduling=request['scheduling_contract'],
        expected_wall_seconds=60, original_job_expected_over_15_minutes=False,
        stopping_policy='Identity/schema/layout/finite/source/parameter/custody/resource mechanics only. Scientific numerical/bit/signed-zero misses are retained; no accuracy gate.',
        narrative_review=dict(review='holistic-review-11.json', sha256=sha(BASE / 'artifacts/mdb-flux-understanding-20261106/holistic-review-11.json'),
            scope='Astra independently reviews cumulative evidence and endorses frozen cut15; not a substitute for this exact request review.'),
        engineering_diagnostic='Root observed one I001 import-group style finding in unchanged frozen turn14 producer fbd8; F checks pass. MDB CI runs tests without a Ruff gate. Frozen executed source is preserved; this does not affect numerical semantics or source identity.')
    with (run / 'finite-run-review-v1.json').open('x') as stream:
        json.dump(review, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(request_sha256=request['request_sha256'], review_sha256=sha(run / 'finite-run-review-v1.json'),
        source_and_wheel_files=review['source_and_wheel_files_authenticated'], scientific_accuracy_gates=0)))


if __name__ == '__main__':
    main()
