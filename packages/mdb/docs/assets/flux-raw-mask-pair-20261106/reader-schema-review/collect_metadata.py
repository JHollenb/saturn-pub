"""Read only explicitly selected JSON metadata; never open tensor payloads."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
BASE = Path('/Users/jakeholl/domains/artifacts/mdb-flux-raw-mask-pair-20261106')
request = json.loads((BASE/'run-02/request.json').read_text())
remote_script = r'''
import hashlib,json
from pathlib import Path
root=Path('/mnt/big/mdb/research/job-0b6c4340d5e6/raw-mask-pair-result')
names=['plan.json','request.json','source.json','summary.json','native-contract.json','runtime-contract.json','files.json','common-frontend.json','forecast-responses.json','sdpa-forecast.json','sdpa-native.json','timeline.jsonl','forecast-transaction/seal.json','forecast-transaction/forecasts/paired-raw-composition.json','forecast-transaction/observations/both_native_qwen_flux_vae_chains.json']
for case in ('short','long'):
 for name in ('input.json','forecast-qwen.json','native-qwen.json','frontend-forecast.json','frontend-native.json','frontend/forecast.json','frontend/native.json','decoder-forecast.json','decoder-native-own.json','decoder-native-reference.json','forecast-state-rows.json','native-state-rows.json'):
  names.append('cases/'+case+'/'+name)
names.extend(str(p.relative_to(root)) for p in sorted((root/'forecast-transaction/events').glob('*.json')))
manifest=json.loads((root/'files.json').read_text())
out={}
for name in names:
 path=root/name
 assert path.suffix in ('.json','.jsonl') and not path.is_symlink()
 data=path.read_bytes()
 digest=hashlib.sha256(data).hexdigest()
 if name!='files.json':
  row=manifest['files'][name]
  assert row['bytes']==len(data) and row['sha256']==digest,(name,'manifest mismatch')
 out[name]={'sha256':digest,'bytes':len(data),'value':[json.loads(line) for line in data.decode().splitlines()] if name.endswith('.jsonl') else json.loads(data)}
print(json.dumps({'schema':'mdb-mask-pair-actual-json-metadata-packet-v1','remote_root':str(root),'files':out,'tensor_array_reads':0,'checkpoint_reads':0,'tokenizer_calls':0,'model_calls':0},sort_keys=True))
'''
command=['ssh','beast','python3','-']
result=subprocess.run(command,input=remote_script,text=True,capture_output=True,check=False)
with (ROOT/'ssh-stdout.json').open('x') as stream:stream.write(result.stdout)
with (ROOT/'ssh-stderr.txt').open('x') as stream:stream.write(result.stderr)
assert result.returncode==0,result.stderr
packet=json.loads(result.stdout)
assert packet['files']['request.json']['value']['request_sha256']==request['request_sha256']
assert packet['files']['files.json']['value']['job_id']=='job-0b6c4340d5e6'
assert packet['files']['source.json']['value']['job_id']=='job-0b6c4340d5e6'
receipt={'schema':'mdb-mask-pair-json-metadata-collection-v1','command':command,
         'request_sha256':request['request_sha256'],'returncode':result.returncode,
         'json_packet_sha256':hashlib.sha256((ROOT/'ssh-stdout.json').read_bytes()).hexdigest(),
         'json_packet_bytes':(ROOT/'ssh-stdout.json').stat().st_size,
         'remote_manifest_sha256':packet['files']['files.json']['sha256'],
         'selected_json_files':len(packet['files']),'all_selected_files_join_original_manifest':True,
         'remote_code_sha256':hashlib.sha256(remote_script.encode()).hexdigest(),
         'scope':'actual authenticated-SSH JSON metadata only, no tensor payload reads or numerical reductions',
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (ROOT/'collection-receipt.json').open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(receipt,sort_keys=True))
