"""Snapshot terminal partial evidence without rewriting its failed job history."""
import argparse
import gzip
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path


def recover(source, destination):
    read=lambda p:json.loads(p.read_text())
    job=read(source/('original-timeout-job.json'if(source/'original-timeout-job.json').exists()else'job.json'));request=read(source/'request.json')
    if job['state']not in{'timeout','failed'}:raise ValueError('only a terminal timeout/failed snapshot is supported')
    code=(source/'payload/flux_family_custody.py').read_text()
    if hashlib.sha256(code.encode()).hexdigest()!=request['source_hashes']['flux_family_custody.py']:raise ValueError('encoder changed')
    remote=Path(request['config']['output_scratch'])/job['job_id']/'binding-minimal-result'
    snapshot=remote.parent/(job['state']+'-snapshot-01')/'binding-minimal-result'
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'payload',destination/'payload')
    shutil.copyfile(source/'request.json',destination/'request.json')
    shutil.copyfile(source/('original-timeout-job.json'if(source/'original-timeout-job.json').exists()else'job.json'),destination/'job.json')
    script="""import json,os,shutil
from pathlib import Path
namespace={}
exec(ENCODER,namespace)
original=Path(ORIGINAL);output=Path(SNAPSHOT)
assert not (original/'summary.json').exists() and not (original/'files.json').exists()
inventory=namespace['_inventory'](original,namespace['_limits'](None))
output.mkdir(parents=True,exist_ok=False)
for name in inventory:
    src=original/name;dst=output/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
report=dict(schema='mdb-response-'+STATE+'-snapshot-v1',job_id=JOB,request_sha256=REQUEST,original_state=STATE,original_root=str(original),output_root=str(output.parent),completed=[p.stem[len('completed-'):] for p in sorted(original.glob('completed-*.json'))],new_model_calls=0,scope='Derived forensic snapshot; original terminal status retained. No original manifests or seals modified.')
namespace['_save'](output/'recovery.json',dict(report=report,original_inventory=inventory))
namespace['_save'](output/'summary.json',report)
namespace['_save'](output/'files.json',{p.relative_to(output).as_posix():namespace['_sha'](p)[0]for p in sorted(output.rglob('*'))if p.is_file()})
namespace['emit'](output,report,compression_level=1)
"""
    script=script.replace('ENCODER',repr(code)).replace('ORIGINAL',repr(str(remote))).replace('SNAPSHOT',repr(str(snapshot))).replace('JOB',repr(job['job_id'])).replace('REQUEST',repr(request['request_sha256'])).replace('STATE',repr(job['state']))
    m=runpy.run_path(str(source/'payload/flux_family_custody.py'));bounds=m['_limits'](None);h=hashlib.sha256();size=0
    with (destination/'stderr.txt').open('xb')as err,(destination/'archive.tar.gz').open('xb')as archive,(destination/'worker.log.gz').open('xb')as log:
        p=subprocess.Popen(['ssh','beast','python3','-I','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
        p.stdin.write(script.encode());p.stdin.close();parser=m['_Parts'](archive,bounds)
        with gzip.GzipFile(filename='',fileobj=log,mode='wb',mtime=0,compresslevel=1)as compressed:
            while block:=p.stdout.read(1<<20):
                size+=len(block)
                if size>bounds['max_log_bytes']:p.kill();raise ValueError('bounded transfer exceeded')
                h.update(block);compressed.write(block);parser.feed(block)
        if p.wait()!=0:raise RuntimeError('snapshot failed; inspect retained stderr')
        envelope=parser.finish()
    m['_binding'](envelope,job,request,job['job_id'])
    extracted=destination/'extracted';extracted.mkdir()
    inventory=m['_scan_archive'](destination/'archive.tar.gz',bounds,envelope['archive_root'],extracted)
    m['_check_manifest'](extracted/envelope['archive_root'],inventory,bounds)
    recovery=read(extracted/envelope['archive_root']/'recovery.json')
    for name,item in recovery['original_inventory'].items():
        if inventory[name]!=item:raise ValueError('snapshot differs from original file')
    m['_save'](destination/'report.json',envelope)
    result=dict(schema='mdb-'+job['state']+'-forensic-custody-v1',original_state=job['state'],job_id=job['job_id'],remote_snapshot=str(snapshot),original_files_verified=len(recovery['original_inventory']),archive_sha256=m['_sha'](destination/'archive.tar.gz')[0],transport_sha256=h.hexdigest(),transport_bytes=size,new_model_calls=0)
    m['_save'](destination/'forensic-custody.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    print(json.dumps(recover(a.source.resolve(),a.destination.resolve()),sort_keys=True))
