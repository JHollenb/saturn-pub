"""Validate JSON-only remaining ABI joins; never open an NPY or import Torch."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PACKAGE = Path('/Users/jakeholl/domains/saturn-pub/packages/mdb/qualification')
packet=json.loads((ROOT/'ssh-stdout.json').read_text())
files=packet['files']
value=lambda name:files[name]['value']
manifest=value('files.json')['files']
checks={}
schemas={}
array_occurrences=[]


def check(name, condition):
    checks[name]=bool(condition)
    assert condition,name


def decode(node):
    kind=node['kind']
    if kind=='scalar':return node['value']
    if kind=='mapping':return {key:decode(item) for key,item in node['items'].items()}
    if kind in ('list','tuple'):return [decode(item) for item in node['items']]
    raise AssertionError('Metadata tree unexpectedly requires payload decoding: '+kind)


def native_member(name):
    return any(part in name for part in ('/native-state-store/','/qwen/native/','/frontend/native/',
        '/decoder/native-own/','/decoder/native-reference/')) or name.endswith('/frontend/native.json') or Path(name).name in {
        'native-state-rows.json','native-qwen.json','frontend-native.json','decoder-native-own.json','decoder-native-reference.json'}


def visit_records(item, case, label):
    if isinstance(item,dict):
        if {'path','sha256','bytes','shape'} <= item.keys():
            name='cases/'+case+'/'+item['path']
            check('retained_record:'+label, name in manifest and manifest[name]['sha256']==item['sha256'] and manifest[name]['bytes']==item['bytes'])
            kind='tensor' if 'logical_dtype' in item else 'bytes'
            required={'path','sha256','bytes','shape'} | ({'logical_dtype','stored_dtype','content_sha256','stride','encoding','device_at_capture'} if kind=='tensor' else {'dtype'})
            check('record_schema:'+label,required<=item.keys())
            array_occurrences.append({'location':label,'kind':kind,'keys':sorted(item),'shape':item['shape']})
        for key,child in item.items():visit_records(child,case,label+'/'+str(key))
    elif isinstance(item,list):
        for index,child in enumerate(item):visit_records(child,case,label+'/'+str(index))


request=value('request.json')
body={key:item for key,item in request.items() if key!='request_sha256'}
check('request_selfhash',hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()==request['request_sha256'])
forecast=decode(value('forecast-transaction/forecasts/paired-raw-composition.json')['predictions'])
observed=decode(value('forecast-transaction/observations/both_native_qwen_flux_vae_chains.json')['value'])
summary=value('summary.json')
post={'source.json','summary.json','native-contract.json','sdpa-native.json','timeline.jsonl'}
expected_frontier={name for name in manifest if name not in post and not name.startswith('forecast-transaction/') and not native_member(name)}
check('v2_frontier_complete_native_exclusion',set(forecast['external_files'])==expected_frontier)
check('forecast_external_files_manifest_joins',all(manifest[name]==row for name,row in forecast['external_files'].items()))
for case in ('short','long'):
    prefix='cases/'+case+'/'
    check(case+':cohort_summary',summary['cases'][case]['forecast']==forecast['cases'][case] and summary['cases'][case]['native']==observed[case])
    for lane,tree in (('forecast',forecast['cases'][case]),('native',observed[case])):
        check(case+':'+lane+':qwen_sidecar',tree['qwen']==value(prefix+lane+'-qwen.json'))
        qwen=value(prefix+lane+'-qwen.json')
        check(case+':'+lane+':hidden_geometry',len(qwen['hidden_states'])==28 and all(row['shape']==[1,512,2560] for row in qwen['hidden_states']) and qwen['conditioning']['shape']==[1,512,7680])
        check(case+':'+lane+':position_role',('position_ids' in qwen)==(lane=='forecast'))
        if lane=='forecast':check(case+':forecast_position_geometry',qwen['position_ids']['shape']==[1,512])
        for name in (lane+'-qwen.json','frontend-'+lane+'.json'):
            visit_records(value(prefix+name),case,prefix+name)
        check(case+':'+lane+':frontend_duplicate',value(prefix+'frontend-'+lane+'.json')==value(prefix+'frontend/'+lane+'.json'))
        rows=value(prefix+lane+'-state-rows.json')
        check(case+':'+lane+':rows',len(rows)==113 and [row['number'] for row in rows]==list(range(-1,112)) and tree['flux']=={'rows':lane+'-state-rows.json','transitions':112})
        check(case+':'+lane+':row_shape',all({'cut','parent','boundary','execution_point','state','clock','reset','number'}<=row.keys() for row in rows))
    for lane in ('forecast','native-own','native-reference'):
        decoder=value(prefix+'decoder-'+lane+'.json')
        check(case+':'+lane+':decoder_sidecar',decoder==(forecast['cases'][case]['vae'] if lane=='forecast' else observed[case]['vae']['own' if lane=='native-own' else 'reference']))
        visit_records(decoder,case,prefix+'decoder-'+lane+'.json')
        check(case+':'+lane+':png',manifest[prefix+decoder['png']['path']]==decoder['png'] | {} if False else
              manifest[prefix+decoder['png']['path']]=={'sha256':decoder['png']['sha256'],'bytes':decoder['png']['bytes']})
        check(case+':'+lane+':endpoint_shapes',decoder['decoded']['shape']==[1,3,512,512] and decoder['pixels']['shape']==[512,512,3])
        if lane=='forecast':
            check(case+':functional_macros',len(decoder['macros'])==11 and 'post_quant_conv' in decoder['macros'])
        else:
            check(case+':'+lane+':actual_operand_keys',{'actual_before_post_quant','actual_after_post_quant'}<=decoder.keys() and decoder['operand_boundary']=='after_post_quant_conv_before_decoder')

response=value('forecast-responses.json')
check('response_qwen_coverage',set(response['qwen'])=={str(i) for i in range(28)}|{'conditioning'})
check('response_state_coverage',len(response['denoiser'])==113 and [row['number'] for row in response['denoiser']]==list(range(-1,112)))
for row in response['denoiser']:
    index=row['number']+1
    check('response_clock:'+str(index),row['clock']==value('cases/short/forecast-state-rows.json')[index]['clock']==value('cases/long/forecast-state-rows.json')[index]['clock'])
    fields={'/conditioning','/img_ids','/latent','/sigmas','/timesteps','/txt_ids'}
    if row['clock']['phase']!='project':fields|={'/image','/text','/context/img_mod','/context/txt_mod','/context/single_mod','/context/temb','/context/rope/0','/context/rope/1'}
    if row['clock']['phase']=='scheduler':fields.add('/noise_prediction')
    check('response_liveness:'+str(index),set(row['fields'])==fields)
descriptors=[*response['qwen'].values(),*response['decoder'].values(),*(entry for row in response['denoiser'] for entry in row['fields'].values())]
check('response_descriptor_uniform_shape',all(set(row)=={'shape','elements','dtype','orientation','sha256'} and row['elements']==math.prod(row['shape']) and row['dtype']=='float64' and row['orientation']=='long-minus-short' and len(row['sha256'])==64 for row in descriptors))
for lane,start,end in (('forecast',0,54),('native',54,126)):
    rows=value('sdpa-'+lane+'.json')
    check('sdpa:'+lane+':ordinals',[row['ordinal'] for row in rows]==list(range(start,end)))
    for case in ('short','long'):
        selected=[row for row in rows if row['case']==case]
        check('sdpa:'+lane+':'+case+':counts',len(selected)==(27 if lane=='forecast' else 36))
        check('sdpa:'+lane+':'+case+':schemas',all({'case','lane','ordinal','query','key','value','mask','is_causal','enable_gqa','configured_backend','selected_kernel'}==row.keys() and all(set(row[key])=={'shape','stride','dtype','device'} for key in ('query','key','value')) and row['query']['shape']==[1,32,512,128] and row['key']['shape']==row['value']['shape']==[1,8 if case=='long' else 32,512,128] for row in selected))
common=value('common-frontend.json')
front=value('cases/short/frontend-forecast.json')
check('common_all8_records',set(common)=={'raw_noise','packed_noise','img_ids','txt_ids','timesteps','sigmas','generator_before','generator_after'})
common_shapes={key:{'common_keys':sorted(row['descriptor']),'array_record_keys':sorted(front[key]),
                    'capture_stride_present': 'stride' in front[key]} for key,row in common.items()}
check('tensor_frontend_metadata_joins',all(common[key]['descriptor']['shape']==front[key]['shape'] and common[key]['descriptor']['stride']==front[key]['stride'] and common[key]['descriptor']['dtype']==front[key]['logical_dtype'] and common[key]['descriptor']['sha256']==front[key]['content_sha256'] for key in common if not key.startswith('generator_')))
check('generator_minimal_byte_schema',all(set(front[key])=={'path','sha256','bytes','shape','dtype'} for key in ('generator_before','generator_after')))
check('catalog_exact_count_budget_from_shapes',math.prod(value('cases/short/decoder-forecast.json')['decoded']['shape'])+math.prod(value('cases/short/decoder-forecast.json')['pixels']['shape'])==1572864)
events=value('timeline.jsonl')
entries=[row for row in events if row['kind']=='phase_entry']
check('timeline_phase_order',[(row['lane'],row['component'],row['case']) for row in entries]==[(lane,component,case) for lane in ('forecast','native') for component in ('qwen','flux','vae') for case in ('short','long')])
check('timeline_placement_order',[(row['component'],row['direction']) for row in events if row['kind']=='component_placement']==[(component,direction) for component in ('qwen','flux','vae')*2 for direction in ('cuda','cpu')])
eventnames=[files[name]['value']['kind'] for name in sorted(files) if name.startswith('forecast-transaction/events/')]
check('closed_transaction_event_abi',eventnames==['opened','forecast_persisted','globally_sealed','native_execution_released','observation_persisted','closed'])
copy_names=['plan.json','request.json','source.json','summary.json','native-contract.json','files.json','runtime-contract.json','timeline.jsonl','sdpa-forecast.json','sdpa-native.json','forecast-responses.json','common-frontend.json']
check('worker_packaging_root_metadata_exists',all(name in files for name in copy_names))
check('worker_packaging_source_metadata_exists',all(row['path'] in manifest for row in value('source.json')['source_files'].values()))
schemas={'array_record_occurrences':len(array_occurrences),'array_record_variants':dict(Counter((item['kind'],tuple(item['keys'])) for item in array_occurrences)),
         'common_frontend':common_shapes,'signed_response_descriptors':len(descriptors),'case_rows':113,
         'catalog_endpoint_elements_per_record':1572864,'catalog_records_planned':4,
         'native_last_hidden_retention':'reference-only metadata; native position IDs not independently retained',
         'worker_package_original_pngs':6}
schemas['array_record_variants']=[{'kind':kind,'keys':list(keys),'occurrences':count} for (kind,keys),count in schemas['array_record_variants'].items()]
issues=[{'id':'common-byte-generator-rich-descriptor-assumption','status':'known-author-v3-fix',
         'evidence':'Both generator records are _save_bytes minimal records; captured stride/logical_dtype/content_sha256 absent.',
         'action':'Compute raw content/type from decoded saved NPY in actual audit, compare stride only when retained, disclose serialization layout separately.'},
        {'id':'native-frontend-duplicate-frontier-classification','status':'already-fixed-v2',
         'evidence':'Actual frontend/native.json duplicates frontend-native.json and is native-only.'},
        {'id':'uncropped-count-metadata','status':'already-preserved-observation',
         'evidence':'Frozen producer counts BatchEncoding fields; reader reparses retained tokenizer with explicit input_ids, preserves original value.'}]
report={'schema':'mdb-mask-pair-remaining-reader-json-schema-review-v1','status':'no-additional-schema-blocker-found-beyond-known-v3-generator-fix',
        'actual_job':'job-0b6c4340d5e6','request_sha256':request['request_sha256'],
        'raw_manifest_sha256':files['files.json']['sha256'],
        'raw_file_count':len(manifest),'raw_expanded_bytes':value('files.json')['expanded_bytes'],
        'metadata_packet_sha256':hashlib.sha256((ROOT/'ssh-stdout.json').read_bytes()).hexdigest(),
        'reader_source_sha256_at_review':hashlib.sha256((PACKAGE/'audit_flux_raw_mask_pair.py').read_bytes()).hexdigest(),
        'original_producer_sha256':request['source_hashes']['flux_raw_mask_pair.py'],
        'checks':checks,'check_count':len(checks),'schemas':schemas,'issues':issues,
        'section_assessments':{
            'common_frontend':'Known byte-record fix needed; six tensor records match common descriptors entirely in JSON; generator raw content needs actual saved-array audit.',
            'qwen':'Forecast positions/profile and native last_hidden keys present; native positions not incorrectly required. Actual case SDPA schemas/counts/global ordinals match reader.',
            'responses':'Producer Torch and reader NumPy both convert to float64 before signed subtraction, flatten C order and hash little-endian float64. All actual descriptor keys, clocks and recursive liveness agree; array hashes not recomputed here.',
            'decoder':'Three lanes use rich tensor records for operands/decoded/macros, minimal byte records for pixels, minimal PNG records; inherited decoder/load_array supports both serializers. Full native intermediate macros not claimed.',
            'prefix':'IDs comparison gives a one-dimensional512 selector; zero common prefix explicitly returns None. Numeric common-prefix equality remains diagnostic.',
            'catalog':'Four exact-context dependent records; inherited catalog_comparison sets explicit1572864element budget per endpoint, not default1M. Native pixel uint8 differences cast float64 first.',
            'worker_packaging':'All required original JSON sidecars,14source metadata references and6PNG references exist and bind actual manifest. No renamed nativeorigin path assumption; source root wrapper intentionally copied separately from compact plain-map wrapper.'},
        'limits':['This is a JSON/schema/source-contract review, not a full-array or numerical audit.',
                  'No mask, tensor, generator, checkpoint, PNG or model asset payload opened; no tokenizer or framework imported.',
                  'Cannot establish native/forecast equality, response magnitudes, raw generator SHA or true capture stride from absent metadata.',
                  'After the v3 correction, actual full saved-only execution remains the required reader qualification on production geometry.'],
        'execution':{'tensor_array_reads':0,'checkpoint_reads':0,'model_calls':0,'tokenizer_calls':0,'CUDA':0,'jobs':0,'source_edits':0},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (ROOT/'review.json').open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'review_sha256':hashlib.sha256((ROOT/'review.json').read_bytes()).hexdigest(),'checks':len(checks),'status':report['status']},sort_keys=True))
