"""Audit fresh receiver custody, own-state forecasts, chronology and actual image responses."""
import hashlib
import json
import runpy
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from mdb_lanes.forecast_transaction import verify_transaction, load_forecast_tree
from mdb_lanes.response_closure import audit_response, measure_error
from mdb_lanes.tensor_archive import load_tensor_archive


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def tree(value):
    if isinstance(value, Mapping): return {k: tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [tree(v) for v in value]
    return value


def verify_consumed_state(row, chain, calibration_exit, native=None):
    """Named prediction/target evidence must be the actual native consumer input."""
    if len(chain)!=4 or row['commits']!=[body['result'] for body in chain[2:]] or row['commits'][-1]!=row['final']:
        raise ValueError('named image commits detached from continuation')
    expected=['apply','continue','continue','continue'] if native is not None else ['apply','apply','continue','continue']
    if [body['operation'] for body in chain]!=expected: raise ValueError('consumer operations differ')
    if native is None:
        if row['entry']!=chain[1]['result'] or chain[2]['parent']!=row['entry'] or row['source_exit']!=calibration_exit:
            raise ValueError('installed predicted state detached from consumer')
    elif (chain[0]['result']!=native['entry'] or chain[1]['steps']!=1
            or chain[1]['parent']!=native['entry'] or chain[1]['result']!=native['exit']
            or row['entry']!=native['exit'] or row['source_exit']!=native['exit'] or chain[2]['parent']!=native['exit']):
        raise ValueError('native target/calibration detached from image consumer')


def analyze(collection, output):
    here=Path(__file__).resolve().parent; data=collection/'extracted/binding-minimal-result'; payload=collection/'payload'
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection)
    request=read(collection/'request.json'); c=request['config']; summary=read(data/'summary.json')
    if read(data/'plan.json')!=c or read(data/'request.json')!=request or read(collection/'report.json')['report']!=summary: raise ValueError('request/report differs')
    for name,digest in request['source_hashes'].items():
        if sha(payload/name)!=digest: raise ValueError('executed source differs')
    for name,digest in read(payload/'payload-seal.json')['wheels'].items():
        if sha(payload/name)!=digest: raise ValueError('installed wheel differs')
    worker=runpy.run_path(str(payload/'flux_fresh_mlp_future.py'))
    if summary['counts']!=worker['expected_counts']() or summary['entries']!=dict(blocks=486,target_blocks=2,dispatch=3) or summary['targets']!=dict(row4heavy=1,row11heavy=1): raise ValueError('physical census differs')
    frozen={}; transactions={}
    for name,directory,forecast in [('write','write-transaction','writes'),('image','image-transaction','pixels')]:
        transactions[name]=verify_transaction(data/directory,expected_seal_sha256=summary[name+'_seal']['seal_sha256'])
        if not transactions[name]['verified'] or transactions[name]['phase']!='closed': raise ValueError('transaction differs')
        frozen[name]=tree(load_forecast_tree(data/directory,read(data/directory/'forecasts'/(forecast+'.json'))['predictions']))
    forecasts=read(data/'forecasts.json'); pixels=read(data/'image-forecasts.json')
    if forecasts!=frozen['write']['forecasts'] or pixels!=frozen['image']['forecasts']: raise ValueError('sealed forecasts differ')
    if frozen['write']['counts_at_seal']!=dict(worker['expected_counts'](),native_transitions=59,acts=1,receipts=3,images=0) or frozen['write']['entries_at_seal']!=dict(blocks=52,target_blocks=0,dispatch=1): raise ValueError('prewrite native work differs')
    if frozen['image']['counts_at_seal']!=dict(worker['expected_counts'](),native_transitions=377,acts=13,receipts=27,images=6) or frozen['image']['entries_at_seal']!=dict(blocks=340,target_blocks=0,dispatch=1): raise ValueError('preimage native work differs')
    if any(any(row['targets'].values()) for row in frozen.values()): raise ValueError('target observed before seal')
    timeline=[json.loads(line) for line in (data/'timeline.jsonl').read_text().splitlines()]
    at=next(i for i,row in enumerate(timeline) if row['kind']=='image_forecasts_sealed')
    if any(row['entries']['target_blocks'] for row in timeline[:at+1]) or sum(row['kind']=='image_complete' for row in timeline[:at])!=6: raise ValueError('target/image chronology differs')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py')); store=typed['audit_store'](data/'store')
    slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']
    field=lambda fp,key: slot(data/'store',fp,key)
    descriptor=lambda fp: {row['key']['value']:row['value'] for row in store['cuts'][fp]['payload']['items']}
    fresh=read(data/'fresh-parent.json'); native=read(data/'native.json'); calibration=read(data/'calibration.json')
    if native['half']!=calibration or c['seed']!=fresh['seed'] or set(native)!=set(forecasts) or set(native)!={'half','row4heavy','row11heavy'}: raise ValueError('cohort/calibration differs')
    source=Path(c['source_collection']); source_record=read(source/'remote-custody.json'); binding=read(data/'source-binding.json')
    if sha(source/'remote-custody.json')!=c['source_record_sha256'] or binding['remote_record']!=source_record or binding['source_record_sha256']!=c['source_record_sha256'] or binding['selection']!=c['selection']:
        raise ValueError('historical source authority differs')
    historical_plan=source/'extracted/binding-minimal-result/plan.json'
    if sha(historical_plan)!=source_record['files']['plan.json']['sha256'] or fresh['historical_parents']!={key:read(historical_plan)['parents'][key] for key in ('base','edited')}:
        raise ValueError('historical parent identities differ')
    def point(fp,step,phase,index):
        actual=typed['unpack'](store['cuts'][fp]['execution_point'])
        if any(actual[key]!=value for key,value in dict(logical_step=step,phase=phase,index=index).items()): raise ValueError('declared state clock differs')
    point(fresh['initial'],0,'project',None); point(fresh['parent'],2,'joint',1)
    initial,old=descriptor(fresh['initial']),descriptor(fresh['historical_parents']['base'])
    if any(initial[key]!=old[key] for key in ('conditioning','txt_ids','img_ids','timesteps','sigmas')):
        raise ValueError('fresh initial conditioning/IDs/schedule differ')
    r=runpy.run_path(str(payload/'flux_reentry_reference.py')); law=runpy.run_path(str(payload/'flux_receiver_norm_prediction.py')); math=runpy.run_path(str(payload/'flux_own_state_mlp_math.py'))
    program=runpy.run_path(str(payload/'flux_fresh_mlp_program.py'))
    checked=set()
    def arr(item):
        value=load_tensor_archive(data,item); checked.add(item['path'])
        if not math['same_bits'](r['bf16'](value),value): raise ValueError('nonfinite/nonBF16 field')
        return value
    T=field(fresh['parent'],'text'); I=field(fresh['parent'],'image')
    D=field(fresh['historical_parents']['edited'],'text')[:,[4,11]].astype(float)-field(fresh['historical_parents']['base'],'text')[:,[4,11]].astype(float)
    if sha(data/fresh['delta']['path'])!=fresh['delta']['sha256'] or not np.array_equal(np.load(data/fresh['delta']['path'],allow_pickle=False),D): raise ValueError('supplied historical delta differs')
    context=field(fresh['parent'],'context')
    gates={role:np.split(context[key].reshape(1,1,6*I.shape[-1]),6,axis=-1)[5] for role,key in [('image','img_mod'),('text','txt_mod')]}
    inputs={spec['id']:program['make_text'](T,D,spec['doses'],r['bf16']) for spec in c['cases']}
    realized={}
    for name,x in inputs.items():
        if not math['same_bits'](x,arr(fresh['inputs'][name])) or not math['same_bits'](x,field(native[name]['entry'],'text')) or not math['same_bits'](I,field(native[name]['entry'],'image')): raise ValueError('new input construction differs')
        if any(descriptor(native[name]['entry'])[key]!=descriptor(fresh['parent'])[key] for key in descriptor(fresh['parent']) if key!='text'): raise ValueError('native undeclared input closure differs')
        realized[name]=measure_error(x[:,[4,11]].astype(float)-inputs['half'][:,[4,11]].astype(float),np.zeros_like(D))
    for name,record in native.items():
        point(record['entry'],2,'joint',1); point(record['exit'],2,'joint',2)
        for item in record['stages'].values(): arr(item)
        for role in ('image','text'):
            if not math['same_bits'](arr(record['stages'][role+'.exit']),field(record['exit'],role)): raise ValueError('native port/state differs')
            written=math['writeback'](arr(record['stages'][role+'.after_attention']),gates[role],arr(record['stages'][role+'.mlp']),r['bf16'])
            if not math['same_bits'](written['exit'],field(record['exit'],role)): raise ValueError('native finite-write closure differs')
    # Replay the frozen pure predictor in this separate CPU process. Native target
    # ports above are evaluated, but are not passed into either candidate program.
    sources=runpy.run_path(str(payload/'flux_source_receiver.py')) if (payload/'flux_source_receiver.py').exists() else runpy.run_path(str(here/'flux_source_receiver.py'))
    bank=sources['RecipeBank'](read(data/'parameter-artifacts.json'),c['model_path'])
    weights={port:bank.load(dict(path=item['path'],sha256=item['sha256'],descriptor=dict(sha256=item['source']['payload_sha256']),parameter_tensor=item['source']['tensor'],stored_shape=item['source']['shape'])) for item in read(data/'parameter-artifacts.json')['artifacts'] for port in item['parameter_ports']}
    stages={key:arr(item) for key,item in calibration['stages'].items()}
    cache=law['prepare'](I,inputs['half'],context,weights,stages,r,absolute_ports=c['absolute_ports'],affine_policy=c['policy'],absolute_norms=('text.norm1','text.qk_rms'),attention_mode='anchored',receiver_norm='absolute')
    recomputed=0
    for spec in c['cases']:
        name=spec['id']; pair,_=program['predict_pair'](cache,inputs[name],law,r,math)
        for mode,roles in pair.items():
            for role,fields in roles.items():
                if set(forecasts[name][mode][role])!=set(fields): raise ValueError('forecast port census differs')
                for key,value in fields.items():
                    if not math['same_bits'](value,arr(forecasts[name][mode][role][key])): raise ValueError('separate forecast recomputation differs '+name+'.'+mode+'.'+role+'.'+key)
                    recomputed+=1
        del pair
    del cache,weights,stages
    if recomputed!=72 or checked!={p.relative_to(data).as_posix() for p in (data/'arrays').glob('*.npy')} or len(checked)!=summary['unique_arrays']: raise ValueError('array reference/recomputation census differs')
    rows=read(data/'rows.json'); receipts=read(data/'receipts.json'); branches=defaultdict(list)
    public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    for record in receipts:
        body=record['body']; public(body)
        if body!=read(data/'store/receipts'/(body['fingerprint']+'.json')): raise ValueError('durable receipt differs')
        branches[record['branch']].append(body)
    if len(receipts)!=37 or sum(body.get('steps',0) for chain in branches.values() for body in chain)!=538: raise ValueError('receipt work differs')
    if len(branches['fresh-prefix'])!=1 or branches['fresh-prefix'][0]['parent']!=fresh['initial'] or branches['fresh-prefix'][0]['result']!=fresh['parent'] or branches['fresh-prefix'][0]['steps']!=58: raise ValueError('prefix lineage differs')
    images={}; by={(row['id'],row['mode']):row for row in rows}
    if len(by)!=9 or len(rows)!=9: raise ValueError('image panel differs')
    for (name,mode),row in by.items():
        start=fresh['parent'] if mode=='native' else calibration['exit']; chain=branches[name+'.'+mode]
        for body in chain:
            if body['parent']!=start: raise ValueError('consumer lineage broken')
            start=body['result']
        if start!=row['final'] or sum(body.get('steps',0) for body in chain)!=(54 if mode=='native' else 53): raise ValueError('consumer horizon differs')
        verify_consumed_state(row,chain,calibration['exit'],native[name] if mode=='native' else None)
        point(row['final'],4,'project',None)
        if mode!='native':
            if any(row['targets_at_image'].values()) or len(chain)!=4 or [body['operation'] for body in chain[:2]]!=['apply','apply']: raise ValueError('candidate was not sealed before targets')
            installed=descriptor(row['entry']); scaffold=descriptor(calibration['exit'])
            if any(installed[key]!=scaffold[key] for key in scaffold if key not in {'image','text'}): raise ValueError('candidate undeclared closure differs')
            for role in ('image','text'):
                if not math['same_bits'](arr(forecasts[name][mode][role]['exit']),field(row['entry'],role)): raise ValueError('installed candidate differs')
            if pixels[name+'.'+mode]['png_sha256']!=row['png_sha256'] or pixels[name+'.'+mode]['readout']!=row['readout']: raise ValueError('presealed image/readout differs')
        if sha(data/row['png'])!=row['png_sha256']: raise ValueError('PNG identity differs')
        rgb=np.asarray(Image.open(data/row['png']).convert('RGB')).astype(float)
        fixed={label:dict(mean_rgb=value.mean(axis=(0,1)).tolist(),red_minus_blue=float((value[...,0]-value[...,2]).mean())) for label,value in [('whole',rgb),('left',rgb[:,:256]),('right',rgb[:,256:]),('border',np.concatenate([rgb[:32].reshape(-1,1,3),rgb[-32:].reshape(-1,1,3)]))]}
        if fixed!=row['readout']: raise ValueError('frozen fixed-region readout differs')
        images[name,mode]=rgb
    native0=images['half','native']; yy,xx=np.indices(native0.shape[:2]); foreground=(native0.min(axis=-1)<220)|((native0.max(axis=-1)-native0.min(axis=-1))>25)
    masks={name:foreground&(yy>=32)&(yy<480)&(xx>=lo)&(xx<hi) for name,lo,hi in [('left_object_support',32,256),('right_object_support',256,480)]}
    masks['background']=~(masks['left_object_support']|masks['right_object_support'])
    masks.update(left=xx<256,right=xx>=256,border=(yy<32)|(yy>=480))
    numerical={}; outcomes={}
    def measured(p,n):
        return dict(**measure_error(p,n),mae=float(np.abs(p-n).mean()))
    for name in ('row4heavy','row11heavy'):
        numerical[name]={}; outcomes[name]={}
        for mode in ('old','new'):
            numerical[name][mode]={}
            for role in ('image','text'):
                P0={key:arr(item) for key,item in forecasts['half'][mode][role].items()}; P1={key:arr(item) for key,item in forecasts[name][mode][role].items()}
                N0={key:arr(calibration['stages'][role+'.'+key]) for key in ('after_attention','mlp','exit')}; N1={key:arr(native[name]['stages'][role+'.'+key]) for key in N0}
                N0.update(gated=r['bf16'](gates[role]*N0['mlp'])); N1.update(gated=r['bf16'](gates[role]*N1['mlp']))
                numerical[name][mode][role]=dict(fields={key:audit_response(N0[key],N1[key],P0[key],P1[key]) for key in ('mlp','gated','exit')},finite_write=math['transport'](N0,N1,P0,P1))
            P0,P1=images['half',mode],images[name,mode]; N1=images[name,'native']; pd,nd=P1-P0,N1-native0
            regions={key:dict(pixels=int(mask.sum()),status='baseline-conditioned-scoring' if mask.any() else 'instrument-incomplete',predicted_delta_rgb=pd[mask].mean(axis=0).tolist() if mask.any() else None,
                native_delta_rgb=nd[mask].mean(axis=0).tolist() if mask.any() else None,delta_error=measured(pd[mask],nd[mask]) if mask.any() else None) for key,mask in masks.items()}
            outcomes[name][mode]=dict(endpoint=measured(P1,N1),response=measured(pd,nd),baseline=measured(P0,native0),regions=regions)
    ordering={region:{mode:float(((images['row4heavy',mode]-images['row11heavy',mode])[...,0]-(images['row4heavy',mode]-images['row11heavy',mode])[...,2])[mask].mean()) if mask.any() else None for mode in ('old','new','native')} for region,mask in masks.items()}
    sheet=Image.new('RGB',(3*384,3*408),'white'); draw=ImageDraw.Draw(sheet)
    for y,name in enumerate(('half','row4heavy','row11heavy')):
        for x,mode in enumerate(('native','old','new')):
            sheet.paste(Image.fromarray(images[name,mode].astype(np.uint8)).resize((384,384)),(x*384,y*408+24));draw.text((x*384+5,y*408+5),name+' | '+mode,fill='black')
    sheet.save(output.with_suffix('.png'))
    primary={mode:float(np.mean([numerical[name][mode]['image']['fields']['exit']['reference_response']['relative_rms'] for name in numerical])) for mode in ('old','new')}
    result=dict(schema='mdb-fresh-mlp-futures-audit-v1',mechanics_status='verified',instrument_status='qualified numerical/typed-consumer instrument; object-support identities need visual review',terminal_status='unassessed',custody=custody,counts=summary['counts'],entries=summary['entries'],transactions=transactions,
        primary=dict(mean_image_complete_response_error=primary),numerical=numerical,outcomes=outcomes,color_ordering=ordering,realized_text_displacements=realized,
        independent_forecast_field_comparisons=recomputed,unique_arrays=len(checked),typed_store=store['counts'],new_native_calls=0,scope=c['scope'],
        timing_scope='Executed prediction-costs.json seconds/affine_work describe the old program only; replacement MLP time is additionally paid. Complete native worker/process costs are reported separately. No forecast-only speed claim.',
        contact_sheet_sha256=sha(output.with_suffix('.png')),auditor_source_sha256=sha(Path(__file__)))
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return result
