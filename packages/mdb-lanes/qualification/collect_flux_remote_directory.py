"""Keep full legacy custody on Beast; retrieve explicitly selected exact files."""
import argparse
import hashlib
import json
import runpy
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def command(code,*args):
    return ['ssh','beast',shlex.join(['python3','-I','-c',code,*map(str,args)])]


REMOTE = r'''
import gzip,hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
bundle,destination,remote_output=map(Path,sys.argv[1:]);destination.mkdir(parents=True,exist_ok=False)
with tarfile.open(bundle,'r:gz') as tar:
    for item in tar:
        name=PurePosixPath(item.name)
        if name.is_absolute() or '..' in name.parts or not item.isfile():raise ValueError('unsafe metadata bundle')
        target=destination/item.name;target.parent.mkdir(parents=True,exist_ok=True)
        with tar.extractfile(item) as source,target.open('xb') as out:
            while block:=source.read(1<<20):out.write(block)
import runpy
encoder=destination/'custody-encoder.py'
if not encoder.exists():encoder=destination/'payload/flux_family_custody.py'
h=runpy.run_path(str(encoder));read=lambda p:json.loads(p.read_text());save=h['_save'];sha=h['_sha'];request=h['_request'](destination);job=read(destination/'job.json');publication=read(destination/'publication.json')
recovery_encoder=read(destination/'recovery-encoder.json') if (destination/'recovery-encoder.json').exists() else None
if recovery_encoder and sha(encoder)[0]!=recovery_encoder['sha256']:raise ValueError('explicit recovery encoder changed')
if job['state']!='succeeded':raise ValueError('completed job required')
summary=read(remote_output/'summary.json')
if summary['job_id']!=job['job_id'] or summary['request_sha256']!=request['request_sha256']:raise ValueError('directory identity differs')
if publication['files_manifest_sha256']!=sha(remote_output/'files.json')[0] or publication['summary_sha256']!=sha(remote_output/'summary.json')[0] or publication['output_root']!=str(remote_output):raise ValueError('native publication differs')
limits=read(destination/'collection-limits.json') if (destination/'collection-limits.json').exists() else None
bounds=h['_limits'](limits)
code="import json,runpy;from pathlib import Path;h=runpy.run_path("+repr(str(encoder))+");p=Path("+repr(str(remote_output))+");h['emit'](p,json.loads((p/'summary.json').read_text()),limits="+repr(limits)+",compression_level=1)"
raw_hash=hashlib.sha256();raw_bytes=0;started=time.monotonic()
with (destination/'recovery-stderr.txt').open('wb') as err,(destination/'archive.tar.gz').open('xb') as archive,(destination/'worker.log.gz').open('xb') as logfile:
    process=subprocess.Popen([sys.executable,'-I','-c',code],stdout=subprocess.PIPE,stderr=err);parser=h['_Parts'](archive,bounds)
    try:
        with gzip.GzipFile(filename='',fileobj=logfile,mode='wb',mtime=0,compresslevel=1) as log:
            while block:=process.stdout.read(1<<20):
                raw_bytes+=len(block)
                if raw_bytes>bounds['max_log_bytes']:raise ValueError('bounded transport exceeded')
                raw_hash.update(block);log.write(block);parser.feed(block)
        if process.wait()!=0:raise ValueError('custody encoder failed; stderr retained')
        envelope=parser.finish()
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        process.stdout.close()
h['_binding'](envelope,job,request,job['job_id']);inventory=h['_scan_archive'](destination/'archive.tar.gz',bounds,envelope['archive_root']);extracted=destination/'extracted';extracted.mkdir();h['_scan_archive'](destination/'archive.tar.gz',bounds,envelope['archive_root'],extracted);h['_check_manifest'](extracted/envelope['archive_root'],inventory,bounds)
save(destination/'report.json',envelope)
save(destination/'log-custody.json',dict(schema='mdb-flux-family-log-custody-v1',storage='gzip',job_id=job['job_id'],complete=True,job_json_sha256=sha(destination/'job.json')[0],uncompressed_sha256=raw_hash.hexdigest(),uncompressed_bytes=raw_bytes,compressed_sha256=sha(destination/'worker.log.gz')[0],compressed_bytes=sha(destination/'worker.log.gz')[1],scope='Exact request-qualified directory re-emission bytes; not original scheduler log',error=None))
save(destination/'archive-custody.json',dict(schema='mdb-flux-family-archive-custody-v1',job_id=job['job_id'],complete=True,archive_sha256=sha(destination/'archive.tar.gz')[0],archive_bytes=sha(destination/'archive.tar.gz')[1],parts=parser.parts,error=None))
verified=h['verify'](destination,limits)
if envelope['files_manifest_sha256']!=publication['files_manifest_sha256'] or sha(extracted/envelope['archive_root']/'summary.json')[0]!=publication['summary_sha256']:raise ValueError('emitted evidence differs from original scheduler publication')
record=dict(schema='mdb-remote-directory-custody-v1',host='beast',collection=str(destination),original_output=str(remote_output),verification=verified,log=read(destination/'log-custody.json'),files_manifest_sha256=envelope['files_manifest_sha256'],summary_sha256=sha(extracted/envelope['archive_root']/'summary.json')[0],files=inventory,seconds=time.monotonic()-started,scope='Full archive, exact duplicate transport log, payload, extracted scientific evidence and legacy offline verification retained on Beast. No model rerun. Local pointer is not a complete collection.')
if recovery_encoder:record['recovery_encoder']=recovery_encoder
if limits is not None:record['limits']=limits
save(destination/'remote-custody.json',record);bundle.unlink();print(json.dumps(dict(record=record,record_sha256=sha(destination/'remote-custody.json')[0]),sort_keys=True))
'''


def custody_encoder(source,encoder_source=None):
    """An absent worker encoder may come from an explicitly named sealed run.

    Keep recovery code separate from the original executed payload/request.
    Never silently add it to a historical source manifest.
    """
    root=source if encoder_source is None else Path(encoder_source)
    path=root/'payload/flux_family_custody.py';h=runpy.run_path(str(path));receipt=None
    if encoder_source is not None:
        request=h['_request'](root);digest=request['source_hashes'].get('flux_family_custody.py')
        if digest!=sha(path):raise ValueError('explicit custody encoder is not request sealed')
        receipt=dict(schema='mdb-explicit-recovery-encoder-v1',source_request_sha256=request['request_sha256'],sha256=digest,scope='Collection-only encoder supplied by a separately request-sealed run. Original worker payload/request remain unchanged.')
    return h,path,receipt


def collect(source,pointer,*,catalog='/mnt/hdd8tb/mdb/custody/hybrid-futures-20261105',encoder_source=None,limits=None):
    from mdb_lanes.job_support import scheduler_api
    if pointer.exists():raise ValueError('pointer exists')
    h,encoder,encoder_receipt=custody_encoder(source,encoder_source);read=lambda p:json.loads(p.read_text());request=h['_request'](source)
    if limits is not None:limits={key:value for key,value in h['_limits'](limits).items() if key in h['DEFAULTS']}
    jid=read(source/'submission.json')['job_id'] if (source/'submission.json').exists() else read(source/'job.json')['job_id']
    api=scheduler_api();status,raw,_=api.request('GET','/api/jobs/'+jid,timeout_s=30)
    if status>=400:raise ValueError('job lookup failed')
    job=json.loads(raw)
    if job['state']!='succeeded' or not jid.startswith('job-') or any(x not in '0123456789abcdef' for x in jid[4:]):raise ValueError('completed identified job required')
    with h['_open_logs'](api,'/api/jobs/'+jid+'/logs?offset=0&wait_s=1') as stream:
        log=stream.read(1<<20)
        if stream.read(1):raise ValueError('unexpectedly large metadata log')
    found=[json.loads(x[len(b'MDB_DIRECTORY_READY '):]) for x in log.splitlines() if x.startswith(b'MDB_DIRECTORY_READY ')]
    if len(found)!=1 or found[0]['job_id']!=jid or found[0]['request_sha256']!=request['request_sha256']:raise ValueError('native directory publication missing')
    publication=found[0];remote_output=Path(request['config']['output_scratch'])/jid/'binding-minimal-result'
    if publication['output_root']!=str(remote_output):raise ValueError('unexpected output')
    destination=Path(catalog)/jid/'collection-01';remote_bundle=Path(catalog)/(jid+'.'+uuid.uuid4().hex+'.metadata.tar.gz')
    subprocess.run(['ssh','beast',shlex.join(['mkdir','-p',str(Path(catalog))])],check=True)
    with tempfile.TemporaryDirectory(prefix='mdb-remote-custody-') as tmp:
        temp=Path(tmp);(temp/'job.json').write_text(json.dumps(job));(temp/'publication.json').write_text(json.dumps(publication));(temp/'scheduler-metadata.log').write_bytes(log);archive=temp/'metadata.tar.gz'
        files=[(p,p.relative_to(source).as_posix()) for p in sorted((source/'payload').rglob('*')) if p.is_file()]
        files.extend((source/n,n) for n in ('request.json','plan.json') if (source/n).exists());files.extend((temp/n,n) for n in ('job.json','publication.json','scheduler-metadata.log'))
        if limits is not None:
            (temp/'collection-limits.json').write_text(json.dumps(limits,sort_keys=True)+'\n');files.append((temp/'collection-limits.json','collection-limits.json'))
        if encoder_receipt:
            (temp/'recovery-encoder.json').write_text(json.dumps(encoder_receipt,sort_keys=True)+'\n');files.extend([(encoder,'custody-encoder.py'),(temp/'recovery-encoder.json','recovery-encoder.json')])
        with tarfile.open(archive,'w:gz') as tar:
            for p,name in files:
                if p.is_symlink():raise ValueError('source symlink refused')
                tar.add(p,arcname=name,recursive=False)
        subprocess.run(['scp','-q',str(archive),'beast:'+str(remote_bundle)],check=True)
    try:result=subprocess.run(command(REMOTE,remote_bundle,destination,remote_output),check=True,capture_output=True,text=True)
    except subprocess.CalledProcessError as error:
        diagnostic=pointer.with_name(pointer.stem+'.collection-error-'+uuid.uuid4().hex+'.json');diagnostic.parent.mkdir(parents=True,exist_ok=True)
        diagnostic.write_text(json.dumps(dict(job_id=jid,remote_collection=str(destination),returncode=error.returncode,stderr=error.stderr[-65536:],scope='Collection failure only. Partial remote custody retained; no model rerun.'),indent=2,sort_keys=True)+'\n')
        raise ValueError('remote collection failed; diagnostics retained at '+str(diagnostic)+': '+error.stderr[-2000:]) from None
    record=json.loads(result.stdout);pointer.parent.mkdir(parents=True,exist_ok=True)
    value=dict(schema='mdb-remote-custody-pointer-v1',**record,collector_source_sha256=sha(Path(__file__)),local_source=str(source),scope='Pointer to fully verified remote custody; scientific tensors are not locally collected. Use --retrieve to fetch exact retained bytes.')
    if encoder_source is not None:value['custody_encoder_source']=str(Path(encoder_source).resolve())
    with pointer.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    return dict(pointer=str(pointer),remote_collection=str(destination),verification=record['record']['verification'],local_scientific_bytes_downloaded=0)


READ_REMOTE = r'''
import sys
from pathlib import Path
path=Path(sys.argv[1]);limit=int(sys.argv[2])
if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError('remote symlink refused')
with path.open('rb') as f:
    remaining=limit+1
    while remaining:
        block=f.read(min(1<<20,remaining))
        if not block:break
        sys.stdout.buffer.write(block);remaining-=len(block)
'''


def bounded_fetch(remote,output,*,size,digest):
    if not isinstance(size,int) or size<0:raise ValueError('invalid retrieval size')
    process=subprocess.Popen(command(READ_REMOTE,remote,size),stdout=subprocess.PIPE)
    seen=0;hasher=hashlib.sha256()
    try:
        with output.open('xb') as f:
            while block:=process.stdout.read(min(1<<20,size-seen+1)):
                seen+=len(block)
                if seen>size:raise ValueError('retrieval exceeded inventoried byte count; partial retained')
                hasher.update(block);f.write(block)
        if process.wait()!=0 or seen!=size or hasher.hexdigest()!=digest:raise ValueError('retrieved bytes differ; partial retained')
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        process.stdout.close()


def checked_pointer(pointer):
    value=json.loads(pointer.read_text());record=value['record'];collection=Path(record['collection']);jid=record['verification']['job_id']
    if value['schema']!='mdb-remote-custody-pointer-v1' or record['host']!='beast' or not collection.is_absolute() or '..' in collection.parts or not collection.is_relative_to('/mnt/hdd8tb/mdb/custody') or collection.parts[-2:]!=(jid,'collection-01'):raise ValueError('untrusted remote custody location')
    if not jid.startswith('job-') or not jid[4:] or any(x not in '0123456789abcdef' for x in jid[4:]):raise ValueError('invalid remote job')
    canonical=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
    if hashlib.sha256(canonical).hexdigest()!=value['record_sha256']:raise ValueError('pointer record digest differs')
    source=Path(value['local_source']);h,_,receipt=custody_encoder(source,value.get('custody_encoder_source'));request=h['_request'](source)
    if record.get('recovery_encoder')!=receipt:raise ValueError('recovery encoder binding differs')
    trusted=json.loads((source/('submission.json' if (source/'submission.json').exists() else 'job.json')).read_text())
    if trusted['job_id']!=jid or record['verification']['request_sha256']!=request['request_sha256']:raise ValueError('pointer differs from staged request/job')
    with tempfile.TemporaryDirectory(prefix='mdb-record-verification-') as tmp:
        p=Path(tmp)/'record.json';bounded_fetch(collection/'remote-custody.json',p,size=len(canonical),digest=value['record_sha256'])
        if json.loads(p.read_text())!=record:raise ValueError('remote custody differs')
    return record


def retrieve(pointer,relative,output):
    parts=Path(relative)
    if parts.is_absolute() or any(p in {'','.','..'} for p in relative.split('/')) or '\\' in relative or output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents):raise ValueError('safe new retrieval path required')
    record=checked_pointer(pointer)
    if relative=='archive.tar.gz':digest=record['verification']['archive_sha256'];size=record['verification']['archive_bytes']
    elif relative=='worker.log.gz':digest=record['log']['compressed_sha256'];size=record['log']['compressed_bytes']
    else:
        prefix='extracted/binding-minimal-result/'
        if not relative.startswith(prefix):raise ValueError('only retained scientific members or transport containers may be retrieved')
        item=record['files'].get(relative[len(prefix):])
        if not item:raise ValueError('member not inventoried')
        digest,size=item['sha256'],item['bytes']
    output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_name(output.name+'.retrieving')
    if temp.exists():raise ValueError('partial retrieval exists')
    bounded_fetch(Path(record['collection'])/relative,temp,size=size,digest=digest)
    temp.replace(output);return dict(path=str(output),sha256=digest,bytes=size)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path,nargs='?');p.add_argument('pointer',type=Path,nargs='?');p.add_argument('--catalog',help='Explicit remote custody parent for this study');p.add_argument('--encoder-source',type=Path,help='Explicit separately request-sealed run supplying a missing collection encoder');p.add_argument('--limits-json',type=Path,help='Explicit finite custody bounds, recorded separately from the original worker request');p.add_argument('--retrieve',nargs=3,metavar=('POINTER','MEMBER','OUTPUT'));a=p.parse_args()
    options={} if a.catalog is None else dict(catalog=a.catalog)
    if a.limits_json is not None:options['limits']=json.loads(a.limits_json.read_text())
    print(json.dumps(retrieve(Path(a.retrieve[0]).resolve(),a.retrieve[1],Path(a.retrieve[2]).resolve()) if a.retrieve else collect(a.source.resolve(),a.pointer.resolve(),encoder_source=None if a.encoder_source is None else a.encoder_source.resolve(),**options),sort_keys=True))
