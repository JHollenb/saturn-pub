"""Integrity failures must be caught before fetching scientific evidence."""
import hashlib
import io
import json
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


SOURCE=Path(__file__).resolve().parents[1]/'qualification/collect_flux_remote_directory.py'


def test_pointer_tampering_is_rejected_before_network(tmp_path,monkeypatch):
    h=runpy.run_path(str(SOURCE)); calls=[]
    h['checked_pointer'].__globals__['bounded_fetch']=lambda *a,**kw:calls.append(a)
    record=dict(host='beast',collection='/mnt/hdd8tb/mdb/custody/test/job-abcd/collection-01',verification=dict(job_id='job-abcd'))
    p=tmp_path/'pointer.json';p.write_text(json.dumps(dict(schema='mdb-remote-custody-pointer-v1',record=record,record_sha256='0'*64)))
    with pytest.raises(ValueError,match='digest'):h['checked_pointer'](p)
    assert not calls


@pytest.mark.parametrize('member',['../file','/tmp/file','extracted/../file','extracted//file','extracted/./file','extracted\\file'])
def test_bad_member_is_rejected_before_remote_lookup(tmp_path,member):
    h=runpy.run_path(str(SOURCE))
    with pytest.raises(ValueError,match='safe new retrieval'):h['retrieve'](tmp_path/'missing-pointer',member,tmp_path/'output')


def test_oversize_remote_stream_never_promotes_or_writes_extra_byte(tmp_path,monkeypatch):
    h=runpy.run_path(str(SOURCE))
    class Process:
        def __init__(self):self.stdout=io.BytesIO(b'123456789');self.stopped=False
        def poll(self):return 0 if self.stopped else None
        def terminate(self):self.stopped=True
        def wait(self,timeout=None):self.stopped=True;return 0
    process=Process();monkeypatch.setattr(h['bounded_fetch'].__globals__['subprocess'],'Popen',lambda *a,**kw:process)
    p=tmp_path/'partial'
    with pytest.raises(ValueError,match='exceeded'):h['bounded_fetch']('/remote',p,size=3,digest=hashlib.sha256(b'123').hexdigest())
    assert p.stat().st_size<=3 and process.stopped and process.stdout.closed


def encoder_run(root,sealed=True):
    source=SOURCE.with_name('flux_family_custody.py').read_bytes();p=root/'payload/flux_family_custody.py';p.parent.mkdir(parents=True);p.write_bytes(source)
    r=dict(source_hashes={'flux_family_custody.py':hashlib.sha256(source).hexdigest()} if sealed else {})
    r['request_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest();(root/'request.json').write_text(json.dumps(r));return r


def test_explicit_encoder_has_separate_request_binding_and_does_not_mutate_source(tmp_path):
    h=runpy.run_path(str(SOURCE));original=tmp_path/'original';original.mkdir();external=tmp_path/'external';r=encoder_run(external)
    api,path,receipt=h['custody_encoder'](original,external)
    assert not list(original.iterdir()) and path==external/'payload/flux_family_custody.py'
    assert receipt['source_request_sha256']==r['request_sha256'] and receipt['sha256']==r['source_hashes']['flux_family_custody.py']
    assert api['_request'](external)==r


def test_unsealed_external_encoder_is_refused(tmp_path):
    h=runpy.run_path(str(SOURCE));external=tmp_path/'external';encoder_run(external,sealed=False)
    with pytest.raises(ValueError,match='not request sealed'):h['custody_encoder'](tmp_path/'original',external)


def test_near_data_emission_records_and_reuses_explicit_bounds(tmp_path):
    h=runpy.run_path(str(SOURCE));metadata=tmp_path/'metadata';request=encoder_run(metadata)
    output=tmp_path/'binding-minimal-result';output.mkdir()
    report=dict(job_id='job-abcd',request_sha256=request['request_sha256'])
    def save(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')
    save(output/'summary.json',report);(output/'result.bin').write_bytes(bytes(range(256))*3)
    save(output/'files.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    save(metadata/'job.json',dict(job_id='job-abcd',state='succeeded'))
    save(metadata/'publication.json',dict(output_root=str(output),files_manifest_sha256=hashlib.sha256((output/'files.json').read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((output/'summary.json').read_bytes()).hexdigest()))
    limits=dict(max_archive_bytes=32768,max_expanded_bytes=32768,max_files=16,max_member_bytes=8192,chunk_bytes=97,max_log_bytes=65536)
    save(metadata/'collection-limits.json',limits);bundle=tmp_path/'metadata.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for p in metadata.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(metadata),recursive=False)
    destination=tmp_path/'collection';r=subprocess.run([sys.executable,'-I','-c',h['REMOTE'],str(bundle),str(destination),str(output)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    record=json.loads(r.stdout)['record'];assert record['limits']==limits and record['verification']['part_count']>1
    assert h['custody_encoder'](metadata)[0]['verify'](destination,record['limits'])==record['verification']
    assert (destination/'extracted/binding-minimal-result/result.bin').read_bytes()==(output/'result.bin').read_bytes()
