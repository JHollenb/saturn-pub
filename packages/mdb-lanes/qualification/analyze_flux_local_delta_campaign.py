"""Audit prospective changed writes, pixel forecasts and actual native futures."""
import argparse
import json
import runpy
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from mdb_lanes.edit_search import file_digest as sha
from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
from mdb_lanes.response_closure import measure_error,audit_factorial
from mdb_lanes.tensor_archive import load_tensor_archive


def tree(value):
    if isinstance(value,Mapping): return {k:tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [tree(v) for v in value]
    return value


def compare(p,n):
    p,n=np.asarray(p,float),np.asarray(n,float); result=measure_error(p,n); nr=result['reference_rms']; pr=float(np.sqrt(np.mean(p*p)))
    return dict(**result,predicted_rms=pr,predicted_over_native_norm=None if nr==0 else pr/nr)


def analyze(collection,precision,output):
    here=Path(__file__).resolve().parent; read=lambda p:json.loads(p.read_text()); data=collection/'extracted/binding-minimal-result'
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection); request=read(collection/'request.json'); c=read(data/'plan.json'); summary=read(data/'summary.json')
    if c!=request['config'] or summary!=read(collection/'report.json')['report']: raise ValueError('report/request differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('worker source differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('runtime wheel differs')
    for name,item in c['input_files'].items():
        if sha(data/'source'/name)!=item['sha256'] or (data/'source'/name).stat().st_size!=item['bytes']: raise ValueError('source copy differs')
    worker=runpy.run_path(str(collection/'payload/flux_local_delta_campaign.py')); expected=worker['expected_counts']()
    if summary['counts']!=expected or summary['entries']!=dict(blocks=676,target_blocks=4) or any(v!=1 for v in summary['target_counts'].values()): raise ValueError('execution census differs')
    transactions={}; sealed={}
    for key,directory,forecast in [('write','write-transaction','affected_writes'),('image','image-transaction','image_effects')]:
        transactions[key]=verify_transaction(data/directory,expected_seal_sha256=summary[key+'_seal']['seal_sha256'])
        if not transactions[key]['verified'] or transactions[key]['phase']!='closed': raise ValueError('transaction not closed/verified')
        sealed[key]=tree(load_forecast_tree(data/directory,read(data/directory/'forecasts'/(forecast+'.json'))['predictions']))
    forecasts=read(data/'forecasts.json'); pixel_forecasts=read(data/'image-forecasts.json'); native=read(data/'native.json'); rows=read(data/'rows.json'); receipts=read(data/'receipts.json')
    if forecasts!=sealed['write']['forecasts'] or pixel_forecasts!=sealed['image']['forecasts'] or sealed['write']['entries_at_seal']!=dict(blocks=0,target_blocks=0): raise ValueError('forecast seal binding differs')
    if sealed['write']['counts_at_seal']!=dict(expected,loads=1,acts=0,receipts=0,native_transitions=0,images=0): raise ValueError('unexpected native work before writes seal')
    if sealed['image']['entries_at_seal']!=dict(blocks=480,target_blocks=0) or sealed['image']['counts_at_seal']!=dict(expected,native_transitions=530,acts=16,receipts=36,images=10): raise ValueError('unexpected consumer/target work before image seal')
    if any(any(sealed[key]['target_counts'].values()) for key in sealed): raise ValueError('native target observed before predictions')
    timeline=[json.loads(line) for line in (data/'timeline.jsonl').read_text().splitlines()]; kinds=[row['kind'] for row in timeline]
    at=kinds.index('image_forecasts_sealed')
    if any(row['entries']['target_blocks'] for row in timeline[:at+1]) or sum(row['kind']=='image_complete' for row in timeline[:at])!=10: raise ValueError('native target chronology differs')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py')); stores={key:typed['audit_store'](data/key) for key in ('store','source/store')}; B=typed['quantize_bf16']
    slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']; public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    branches=defaultdict(list)
    for record in receipts:
        body=record['body']; public(body)
        if body!=read(data/'store/receipts'/(body['fingerprint']+'.json')): raise ValueError('durable receipt differs')
        branches[record['branch']].append(body)
    if len(receipts)!=56 or sum(r['body'].get('steps',0) for r in receipts)!=746: raise ValueError('receipt work census differs')
    by={(row['id'],row['mode']):row for row in rows}
    if len(rows)!=14 or len(by)!=14: raise ValueError('image panel incomplete/duplicated')
    arrays_checked=set(); images={}; baselines={}
    def arr(record):
        p=data/record['path']; value=np.load(p,allow_pickle=False)
        if sha(p)!=record['sha256'] or list(value.shape)!=record['shape'] or str(value.dtype)!=record['dtype'] or not np.isfinite(value).all() or not np.array_equal(value,B(value)): raise ValueError('external forecast/port archive differs')
        arrays_checked.add(record['path']); return value
    def fields(which,fp): return {item['key']['value']:item['value'] for item in stores[which]['cuts'][fp]['payload']['items']}
    for anchor in ('half','half.image1'):
        row=by[anchor,'control']; p=data/row['png']
        if sha(p)!=c['anchors'][anchor]['png_sha256'] or row['png_sha256']!=sha(p): raise ValueError('historical suffix/image replay differs')
        baselines[anchor]=np.asarray(Image.open(p).convert('RGB')).copy()
    for row in rows:
        mode=row['mode']; name=row['id']; branch=name+'.'+mode; chain=branches[branch]; anchor=c['anchors'][row['anchor']]
        parent=c['parents']['base'] if mode=='native' else anchor['exit']
        for receipt in chain:
            if receipt['parent']!=parent: raise ValueError('branch lineage broken')
            parent=receipt['result']
        if parent!=row['final'] or sum(r.get('steps',0) for r in chain)!=(54 if mode=='native' else 53): raise ValueError('consumer horizon differs')
        if mode=='control':
            if any(r['operation']=='apply' for r in chain) or row['entry']!=anchor['exit']: raise ValueError('control changed state')
        else:
            if [r['operation'] for r in chain[:2]]!=['apply','apply'] or chain[1]['result']!=row['entry']: raise ValueError('typed writes/entry differ')
            entry=fields('store',row['entry'])
            scaffold=fields('source/store',c['parents']['base'] if mode=='native' else anchor['exit'])
            if any(entry[k]!=scaffold[k] for k in scaffold if k not in {'image','text'}): raise ValueError('undeclared closure changed')
            if mode!='native':
                for index,role in enumerate(('image','text')):
                    value=arr(forecasts[name][mode]['fields'][role]['exit'])
                    if not typed['matches'](value,entry[role],bf16=True) or chain[index]['act']['writes']!=[role]: raise ValueError('actual installed predicted write differs')
                if any(row['target_counts_at_image'].values()) or pixel_forecasts[name+'.'+mode]['png_sha256']!=row['png_sha256']: raise ValueError('candidate image not before targets')
        p=data/row['png']
        if sha(p)!=row['png_sha256']: raise ValueError('native/predicted image bytes differ')
        image=np.asarray(Image.open(p).convert('RGB')).copy(); images[name,mode]=image
        if worker['scene_readout'](image,baselines[row['anchor']])!=row['readout']: raise ValueError('source-support readout differs')
        if mode not in ('control','native') and row['readout']!=pixel_forecasts[name+'.'+mode]['readout']: raise ValueError('readout forecast not frozen')
    prior_collection=precision/'collection-01'
    prior_custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](prior_collection)
    prior=prior_collection/'extracted/binding-minimal-result'; old=read(prior/'forecasts.json'); old_native=read(prior/'native.json')
    for anchor,record in c['anchors'].items():
        if any(record[k]!=old_native[anchor][k] for k in ('entry','exit','stages')): raise ValueError('historical port/cut bank join differs')
    for filename in ('flux_response_reference.py','flux_reentry_reference.py'):
        if request['source_hashes'][filename]!=read(precision/'collection-01/request.json')['source_hashes'][filename]: raise ValueError('absolute baseline source contract changed')
    base=c['parents']['base']; edited=c['parents']['edited']; T0=slot(data/'source/store',base,'text'); T1=slot(data/'source/store',edited,'text')
    numerical={}; outcomes={}; exits={}; target_exits={}
    for spec in c['cases']:
        name=spec['id']; anchor=spec['anchor']; n=native[name]; entry=n['entry']; newtext=T0.copy()
        native_row=by[name,'native']; chain=branches[name+'.native']
        if entry!=native_row['entry'] or chain[2]['operation']!='continue' or chain[2]['steps']!=1 or chain[2]['parent']!=entry or chain[2]['result']!=n['exit']: raise ValueError('target numerical write disconnected from actual image continuation')
        point=typed['unpack'](stores['store']['cuts'][n['exit']]['execution_point'])
        if any(point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=2).items()): raise ValueError('numerical target exit clock differs')
        for row,dose in zip((4,11),spec['row_doses']): newtext[:,row]=B(T0[:,row].astype(float)+dose*(T1[:,row].astype(float)-T0[:,row].astype(float)))
        if not np.array_equal(newtext,slot(data/'store',entry,'text')) or not np.array_equal(slot(data/'store',entry,'image'),slot(data/'source/store',c['anchors'][anchor]['entry'],'image')): raise ValueError('fresh complete input construction differs')
        numerical[name]={}; outcomes[name]={}; exits[name]={}; target_exits[name]={}
        native_rgb=images[name,'native'].astype(float); rgb0=baselines[anchor].astype(float)
        for mode in ('local','absolute'):
            numerical[name][mode]={}; exits[name][mode]={}
            for role in ('image','text'):
                numerical[name][mode][role]={}
                for field,record in forecasts[name][mode]['fields'][role].items():
                    predicted=arr(record); actual=slot(data/'store',n['exit'],role) if field=='exit' else arr(n['fields'][field+'_'+role]); n0=load_tensor_archive(data/'source/prefixes',c['anchors'][anchor]['stages'][role+'.'+field])
                    p0=n0 if mode=='local' else load_tensor_archive(prior,old[anchor]['legacy']['fields'][role][field])
                    numerical[name][mode][role][field]=dict(endpoint=compare(predicted,actual),edit=compare(predicted.astype(float)-p0.astype(float),actual.astype(float)-n0.astype(float)))
                    if field=='exit': exits[name][mode][role]=predicted; target_exits[name][role]=actual
            prgb=images[name,mode].astype(float); difference=prgb-native_rgb
            frozen=by[name,mode]['readout']; observed=by[name,'native']['readout']; regions={}
            for region,pred in frozen['regions'].items():
                actual=observed['regions'][region]
                regions[region]=dict(predicted_delta_rgb=pred['delta_rgb'],native_delta_rgb=actual['delta_rgb'],delta_comparison=compare(pred['delta_rgb'],actual['delta_rgb']),
                    predicted_red_minus_blue_delta=pred['red_minus_blue_delta'],native_red_minus_blue_delta=actual['red_minus_blue_delta'],direction_matches=bool(np.sign(pred['red_minus_blue_delta'])==np.sign(actual['red_minus_blue_delta'])))
            outcomes[name][mode]=dict(rgb_mae=float(np.abs(difference).mean()),rgb_max_abs=float(np.abs(difference).max()),rgb_endpoint=compare(prgb,native_rgb),rgb_edit=compare(prgb-rgb0,native_rgb-rgb0),
                copy_anchor=compare(np.zeros_like(rgb0),native_rgb-rgb0),regions=regions,
                background_collateral=dict(predicted_mae=frozen['background_outside_source_boxes_mae'],native_mae=observed['background_outside_source_boxes_mae'],signed_error=observed['background_outside_source_boxes_mae']-frozen['background_outside_source_boxes_mae']))
    expected_color_hypothesis='For each fixed receiver, row4heavy is bluer (lower mean red-minus-blue) than row11heavy on both supplied source-object supports. This direction is frozen before candidate or target futures; retain every counterexample.'
    if c.get('hypotheses',{}).get('color_ordering')!=expected_color_hypothesis: raise ValueError('color ordering was not declared by this plan')
    color_ordering={}; factorial={}
    for suffix,receiver in [('', 'base'),('.image1','edited')]:
        a,b='row4heavy'+suffix,'row11heavy'+suffix; color_ordering[receiver]={}
        for region in ('left_source_object','right_source_object'):
            result={}
            for mode in ('local','absolute','native'):
                ra=by[a,mode]['readout']['regions'][region]['mean_rgb']; rb=by[b,mode]['readout']['regions'][region]['mean_rgb']; value=(ra[0]-ra[2])-(rb[0]-rb[2])
                result[mode]=dict(row4heavy_minus_row11heavy_red_blue=value,matches_frozen_negative_direction=value<0)
            color_ordering[receiver][region]=result
    order=['row11heavy','row4heavy','row11heavy.image1','row4heavy.image1']
    for mode in ('local','absolute'):
        factorial[mode]={}
        for role in ('image','text','RGB'):
            n={k:(images[name,'native'] if role=='RGB' else target_exits[name][role]) for k,name in zip(('00','01','10','11'),order)}
            p={k:(images[name,mode] if role=='RGB' else exits[name][mode][role]) for k,name in zip(('00','01','10','11'),order)}
            report=audit_factorial(n,p); report['axes']=dict(first='receiver',second='asymmetric dose pattern; not an independent row4 x row11 factorial')
            report['contrasts_with_norms']={key:compare(sum(p[k].astype(float)*sign for k,sign in coeff.items()),sum(n[k].astype(float)*sign for k,sign in coeff.items())) for key,coeff in {'receiver_by_pattern':{'11':1,'10':-1,'01':-1,'00':1}}.items()}
            factorial[mode][role]=report
    sheet=Image.new('RGB',(4*384,4*408),'white'); draw=ImageDraw.Draw(sheet)
    for y,spec in enumerate(c['cases']):
        name=spec['id']; panels=[('half baseline',baselines[spec['anchor']]),('local calculation',images[name,'local']),('absolute math',images[name,'absolute']),('native target',images[name,'native'])]
        for x,(label,rgb) in enumerate(panels): sheet.paste(Image.fromarray(rgb).resize((384,384)),(x*384,y*408+24)); draw.text((x*384+6,y*408+5),name+' | '+label,fill='black')
    sheet_path=output.with_suffix('.png'); sheet.save(sheet_path)
    result=dict(schema='mdb-local-delta-future-audit-v1',mechanics_status='verified',instrument_status='complete typed writes/lineage and pretarget PNG forecasts; fixed source supports supplied, visual semantics reviewed separately',terminal_status='unassessed',
        custody=custody,historical_calibration_custody=prior_custody,transactions=transactions,counts=expected,frozen_hypotheses=c['hypotheses'],numerical=numerical,outcomes=outcomes,color_ordering=color_ordering,factorial=factorial,external_array_files_verified=len(arrays_checked),contact_sheet=dict(path=str(sheet_path),sha256=sha(sheet_path)),
        worker_seconds=summary['worker_seconds'],auditor_source_sha256=sha(Path(__file__)),new_native_calls_in_audit=0,
        scope='Four previously unseen asymmetric text states on two fixed native-half calibrated receivers. Changed writes computed in NumPy, candidate images consume native suffix/VAE before native targets. Native encoder/prefix/calibration/suffix supplied; no new prompt/receiver-general law or independent whole-sampler image computation.')
    with output.open('x') as f: json.dump(result,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    return dict(mechanics_status='verified',rgb={name:{mode:dict(mae=row['rgb_mae'],edit_error=row['rgb_edit']['relative_rms']) for mode,row in modes.items()} for name,modes in outcomes.items()},color_ordering=color_ordering)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','precision','output'): p.add_argument(name,type=Path)
    a=p.parse_args(); print(json.dumps(analyze(*[getattr(a,n).resolve() for n in ('collection','precision','output')]),sort_keys=True))
