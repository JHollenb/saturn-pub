"""Compose qualified transport/metadata-scope tools for this saved-only audit."""
import hashlib
import json
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
QUAL = BASE / 'saturn-pub/packages/mdb/qualification'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = BASE / 'artifacts/mdb-qwen-row-sign-20261106/collect_compact.py'
    generic = QUAL / 'collect_flux_remote_directory.py'
    if sha(prior) != 'b621d8e12dc06476abf356643f22693a48a5a52bd43c7684d66e788126475015' or sha(generic) != '007098eb72842fe1f5d585f64eae272812303d1c18eb67a17ba43508c646b18d':
        raise ValueError('Qualified transport or scope helper changed')
    scope, core = runpy.run_path(str(prior)), runpy.run_path(str(generic))
    request = scope['canonical_request'](ROOT / 'audit-01/request.json')
    native = scope['canonical_request'](ROOT / 'run-01/request.json')
    if request['request_sha256'] != 'debd122ff812da52b15a91d3b856301060c7ddb752b4f4a026f8e8b35bcbe351' or request['source_hashes']['audit_qwen_cached_qkv.py'] != 'b473d178fd9825009c0c039dc23ab0f2208309f2d34ccc2d784ee5b5710385eb':
        raise ValueError('Saved reader/source request changed')
    if native['request_sha256'] != request['config']['native_request_sha256'] or request['config']['source_job'] != 'job-585e40c9614b':
        raise ValueError('Original source request/job changed')
    os.environ['MRUN_URL'] = 'http://192.168.68.68:9025'
    encoder_source = BASE / 'artifacts/mdb-flux-functional-decoder-20261106/recovery-collection-01'
    pointer = ROOT / 'audit-custody-pointer-v1.json'
    if not pointer.exists():
        core['collect'](ROOT / 'audit-01', pointer, encoder_source=encoder_source,
                        catalog='/mnt/hdd8tb/mdb/custody/qwen-cached-qkv-20261106')
    record = core['checked_pointer'](pointer)
    if record['verification']['request_sha256'] != request['request_sha256']:
        raise ValueError('Collected audit request differs')
    scope['compact_scope'](record, max_expanded_bytes=64 << 20, max_archive_bytes=32 << 20)
    encoder = runpy.run_path(str(encoder_source / 'payload/flux_family_custody.py'))
    output = ROOT / 'compact-collection-01'
    output.mkdir(exist_ok=False)
    archive = output / 'archive.tar.gz'
    core['retrieve'](pointer, 'archive.tar.gz', archive)
    bounds = encoder['_limits'](None)
    inventory = encoder['_scan_archive'](archive, bounds, 'binding-minimal-result')
    if inventory != record['files']:
        raise ValueError('Archive member inventory differs')
    extracted = output / 'extracted'
    extracted.mkdir()
    encoder['_scan_archive'](archive, bounds, 'binding-minimal-result', extracted)
    pack = extracted / 'binding-minimal-result'
    encoder['_check_manifest'](pack, inventory, bounds)
    summary = json.loads((pack / 'summary.json').read_text())
    if sha(pack / 'files.json') != 'c662757ab5f261b91f72006bd3de57291e1ff70e208c38e9e30631cb6c1a3ca7' or summary['schema'] != 'mdb-qwen-cached-qkv-compact-v1' or summary['request_sha256'] != request['request_sha256'] or summary['source_job'] != 'job-585e40c9614b':
        raise ValueError('Compact source/schema identity differs')
    receipt = dict(schema='mdb-qwen-cached-qkv-compact-retrieval-v1', job_id=summary['job_id'],
        request_sha256=request['request_sha256'], native_job_id=summary['source_job'],
        files_manifest_sha256=sha(pack / 'files.json'), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        file_count=len(inventory), expanded_bytes=sum(row['bytes'] for row in inventory.values()),
        pointer_sha256=sha(pointer), collector_sha256=sha(Path(__file__)), generic_transport_sha256=sha(generic),
        reused_metadata_scope_sha256=sha(prior), raw_tensor_arrays_collected=False, model_calls=0)
    with (output / 'collection-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
