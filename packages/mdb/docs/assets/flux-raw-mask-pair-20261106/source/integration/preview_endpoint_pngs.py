"""Retrieve only six original PNGs by their published native member hashes."""
import base64
import hashlib
import json
import pathlib
import subprocess

OUT = pathlib.Path(__file__).resolve().parent / 'endpoint-preview-v1'
NATIVE = 'job-0b6c4340d5e6'
MANIFEST = '89dafe939e0143596edf88b42e073f97cc5c945c893b5b8cf9897eff809c7154'
REQUEST = '783e8dc0c0a731c458eed739c5d8b68c932bf622ef5f426d7f484596a8c011dd'
REMOTE = r'''
import base64,hashlib,json
from pathlib import Path,PurePosixPath
root=Path('/mnt/big/mdb/research/job-0b6c4340d5e6/raw-mask-pair-result')
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(root/'files.json').read_bytes();m=json.loads(raw)
assert sha(raw)=='89dafe939e0143596edf88b42e073f97cc5c945c893b5b8cf9897eff809c7154'
assert m['job_id']=='job-0b6c4340d5e6' and m['request_sha256']=='783e8dc0c0a731c458eed739c5d8b68c932bf622ef5f426d7f484596a8c011dd'
files={};metadata={}
for case in ('short','long'):
 for lane in ('forecast','native-own','native-reference'):
  name=f'cases/{case}/decoder-{lane}.json';b=(root/name).read_bytes()
  assert sha(b)==m['files'][name]['sha256'] and len(b)==m['files'][name]['bytes']
  d=json.loads(b);png=d['png'];relative=PurePosixPath(png['path'])
  assert not relative.is_absolute() and '..' not in relative.parts
  name=f'cases/{case}/'+str(relative);p=root/name
  assert not p.is_symlink() and p.stat().st_size<=2<<20
  b=p.read_bytes();assert sha(b)==png['sha256']==m['files'][name]['sha256']
  assert len(b)==m['files'][name]['bytes']
  output=f'{case}-{lane}.png';files[output]=base64.b64encode(b).decode()
  metadata[output]={'native_path':name,'sha256':sha(b),'bytes':len(b),'decoder_metadata_sha256':m['files'][f'cases/{case}/decoder-{lane}.json']['sha256']}
print(json.dumps({'files':files,'metadata':metadata,'manifest_bytes':len(raw)}))
'''

result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'beast', 'python3', '-I', '-'],
                        input=REMOTE, text=True, capture_output=True, check=True, timeout=45)
packet = json.loads(result.stdout)
OUT.mkdir(exist_ok=False)
for name, encoded in packet['files'].items():
    raw = base64.b64decode(encoded, validate=True)
    row = packet['metadata'][name]
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    (OUT / name).write_bytes(raw)
receipt = {'schema': 'mdb-selected-endpoint-png-preview-v1', 'native_job': NATIVE,
           'native_request_sha256': REQUEST, 'native_manifest_sha256': MANIFEST,
           'native_manifest_bytes': packet['manifest_bytes'], 'files': packet['metadata'],
           'numerical_arrays_loaded': False, 'model_calls': 0,
           'scope': 'Selected original PNG hashes only; not a complete tensor audit.',
           'within_case_identical_png_bytes': {
               case: len({packet['metadata'][case + '-' + lane + '.png']['sha256']
                          for lane in ('forecast', 'native-own', 'native-reference')}) == 1
               for case in ('short', 'long')},
           'different_case_png_bytes': packet['metadata']['short-forecast.png']['sha256']
               != packet['metadata']['long-forecast.png']['sha256']}
with (OUT / 'receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt, indent=2))
