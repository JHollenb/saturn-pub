"""Retrieve one external evidence file by alias/path and verify its exact SHA."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


def retrieve(manifest,alias,name,destination):
    m=json.loads(manifest.read_text());p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name:raise ValueError('relative evidence path required')
    expected=m['manifests'][alias][name];destination.mkdir(parents=True,exist_ok=True)
    target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temp=target.with_name(target.name+'.retrieving')
        subprocess.run(['scp','-q',m['host']+':'+m['roots'][alias]+'/'+name,str(temp)],check=True)
        if hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()!=expected:raise ValueError('retrieved evidence checksum differs')
        temp.replace(target)
    if hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=expected:raise ValueError('existing evidence checksum differs')
    return str(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('alias',choices=['native','reader','controls']);p.add_argument('path');p.add_argument('destination',type=Path)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('external-evidence.json'));a=p.parse_args()
    print(retrieve(a.manifest,a.alias,a.path,a.destination))
