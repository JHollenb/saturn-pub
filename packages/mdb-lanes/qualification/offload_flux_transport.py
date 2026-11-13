"""Reversibly move verified duplicate transport containers, never raw results."""
import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime,timezone
from pathlib import Path


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def offload(collection,receipt):
    read=lambda p:json.loads(p.read_text()); rows=[]; remote_root='/mnt/hdd8tb/mdb/custody/attention-arithmetic-20261105/transport'
    if receipt.exists() or (collection/'transport-externalized-v1.json').exists(): raise ValueError('externalization already recorded')
    for filename,metadata,key in [('archive.tar.gz','archive-custody.json','archive_sha256'),('worker.log.gz','log-custody.json','compressed_sha256')]:
        path=collection/filename; digest=read(collection/metadata)[key]
        if path.is_symlink() or sha(path)!=digest:raise ValueError('original transport differs')
        rows.append(dict(local=str(path),sha256=digest,bytes=path.stat().st_size,remote=remote_root+'/'+digest+'.gz'))
    subprocess.run(['ssh','beast','mkdir','-p',remote_root],check=True)
    for row in rows:
        subprocess.run(['scp','-q',row['local'],'beast:'+row['remote']],check=True)
        code='import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); print(hashlib.file_digest(p.open("rb"),"sha256").hexdigest())'
        result=subprocess.run(['ssh','beast',shlex.join(['python3','-I','-c',code,row['remote']])],check=True,text=True,capture_output=True)
        if result.stdout.strip()!=row['sha256']:raise ValueError('remote copy differs; local retained')
        print('Verified remote transport',Path(row['local']).name,row['bytes'],flush=True)
    record=dict(schema='mdb-owned-transport-externalization-v1',created_at=datetime.now(timezone.utc).isoformat(),host='beast',records=rows,freed_bytes=sum(r['bytes'] for r in rows),
        source_sha256=sha(Path(__file__)),retrieval_command='python -I '+str(Path(__file__).resolve())+' --restore '+str(receipt),
        scope='Only duplicate compressed transport containers moved. Extracted numerical evidence, manifests, receipts, original custody metadata and audits unchanged. Full collection verification requires exact retrieval first.')
    with receipt.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n');f.flush()
    marker=collection/'transport-externalized-v1.json'
    with marker.open('x') as f:json.dump(dict(receipt=str(receipt),sha256=sha(receipt),scope=record['scope']),f,indent=2);f.write('\n')
    for row in rows:
        path=Path(row['local'])
        if sha(path)!=row['sha256']:raise ValueError('local source changed before removal')
        path.unlink()
    return dict(freed_bytes=record['freed_bytes'],receipt=str(receipt),raw_evidence_changed=False)


def restore(receipt):
    record=json.loads(receipt.read_text())
    for row in record['records']:
        target=Path(row['local'])
        if not target.exists():
            temp=target.with_name(target.name+'.retrieving')
            if temp.exists():raise ValueError('partial retrieval exists')
            subprocess.run(['scp','-q',record['host']+':'+row['remote'],str(temp)],check=True)
            if sha(temp)!=row['sha256']:raise ValueError('restored bytes differ')
            temp.replace(target)
        if sha(target)!=row['sha256']:raise ValueError('existing transport differs')
    return dict(restored_files=len(record['records']),bytes=sum(r['bytes'] for r in record['records']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path,nargs='?');p.add_argument('receipt',type=Path,nargs='?');p.add_argument('--restore',type=Path);a=p.parse_args()
    print(json.dumps(restore(a.restore.resolve()) if a.restore else offload(a.collection.resolve(),a.receipt.resolve()),sort_keys=True))
