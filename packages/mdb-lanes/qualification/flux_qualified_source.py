"""Fresh verification of consumed files against inherited audited full custody."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class QualifiedSource:
    def __init__(self,collection,record_sha256,audit_path,audit_sha256):
        self.collection=Path(collection);record=self.collection/'remote-custody.json'
        if sha(record)!=record_sha256 or sha(audit_path)!=audit_sha256:raise ValueError('inherited custody/audit digest differs')
        self.record=json.loads(record.read_text());self.audit=json.loads(Path(audit_path).read_text());self.checked={};self.audit_sha256=audit_sha256;self.record_sha256=record_sha256
        if self.audit['mechanics_status']!='verified' or self.audit['custody']!=self.record['verification']:raise ValueError('inherited full scientific qualification differs')
        self.root=self.collection/'extracted/binding-minimal-result'

    def member(self,name):
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or str(p)!=name or '\\' in name:raise ValueError('unsafe source member')
        item=self.record['files'][name];path=self.root/name
        if path.is_symlink() or not path.is_file() or any(x.is_symlink() for x in path.parents):raise ValueError('nonregular source member')
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('consumed source bytes differ')
        self.checked[name]=dict(item);return path

    def json(self,name):return json.loads(self.member(name).read_text())

    def cut(self,fp):
        item=self.json('store/cuts/'+fp+'.json')
        if item['fingerprint']!=fp:raise ValueError('source cut identity differs')
        for page in set(item['slots'].values()):
            body=self.json('store/pages/'+page+'.json')
            if body['tensor_blob'] is not None:self.member('store/blobs/'+body['tensor_blob']+'.safetensors')
        return item

    def report(self):
        return dict(schema='mdb-consumed-source-verification-v1',inherited_full_custody=self.record['verification'],
            inherited_audit_sha256=self.audit_sha256,inherited_record_sha256=self.record_sha256,files=dict(sorted(self.checked.items())),
            scope='Inherited full-custody and typed-store qualification; fresh exact checksum verification of every consumed source file. This is not another full archive/log verification.')
