"""Independent, framework-free audit of sealed complete writes and native image futures."""
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
from mdb_lanes.response_closure import measure_error,audit_response,audit_factorial
from mdb_lanes.tensor_archive import load_tensor_archive


def tree(value):
    if isinstance(value,Mapping): return {k:tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [tree(v) for v in value]
    return value


def compare(p,n):
    p,n=np.asarray(p,float),np.asarray(n,float); result=measure_error(p,n); nr=result['reference_rms']; pr=float(np.sqrt(np.mean(p*p)))
    return dict(**result,predicted_rms=pr,predicted_over_native_norm=None if nr==0 else pr/nr)


def analyze(collection,output):
    here=Path(__file__).resolve().parent; read=lambda p:json.loads(p.read_text()); data=collection/'extracted/binding-minimal-result'
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection); request=read(collection/'request.json'); c=read(data/'plan.json'); summary=read(data/'summary.json')
    if c!=request['config'] or summary!=read(collection/'report.json')['report']: raise ValueError('report/request differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('worker source differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('runtime wheel differs')
    for name,item in c['input_files'].items():
        if sha(data/'source'/name)!=item['sha256'] or (data/'source'/name).stat().st_size!=item['bytes']: raise ValueError('source copy differs')
    worker=runpy.run_path(str(collection/'payload/flux_hybrid_future.py')); common=runpy.run_path(str(collection/'payload/flux_local_delta_campaign.py')); expected=worker['expected_counts'](); modes=worker['programs']()
    if c['cases']!=worker['cases']() or c['programs']!=modes or summary['counts']!=expected or summary['entries']!=dict(blocks=676,target_blocks=4,dispatch=4) or summary['target_counts']!={s['id']:1 for s in c['cases']}: raise ValueError('execution/program census differs')
    selected=read(collection/'payload/selection.json')
    if selected['selected_request']!=c['selection_request_sha256'] or selected['audit']['mechanics_status']!='verified': raise ValueError('selection provenance differs')
    transactions={}; sealed={}
    for key,directory,forecast in [('write','write-transaction','affected_writes'),('image','image-transaction','image_effects')]:
        transactions[key]=verify_transaction(data/directory,expected_seal_sha256=summary[key+'_seal']['seal_sha256'])
        if not transactions[key]['verified'] or transactions[key]['phase']!='closed': raise ValueError('transaction not closed/verified')
        sealed[key]=tree(load_forecast_tree(data/directory,read(data/directory/'forecasts'/(forecast+'.json'))['predictions']))
    forecasts=read(data/'forecasts.json'); baselines=read(data/'baselines.json'); pixel_forecasts=read(data/'image-forecasts.json'); native=read(data/'native.json'); rows=read(data/'rows.json'); receipts=read(data/'receipts.json')
    if forecasts!=sealed['write']['forecasts'] or baselines!=sealed['write']['baselines'] or pixel_forecasts!=sealed['image']['forecasts'] or sealed['write']['entries_at_seal']!=dict(blocks=0,target_blocks=0,dispatch=0): raise ValueError('forecast seal binding differs')
    if sealed['write']['counts_at_seal']!=dict(expected,loads=1,acts=0,receipts=0,native_transitions=0,images=0): raise ValueError('unexpected native work before writes seal')
    if sealed['image']['entries_at_seal']!=dict(blocks=480,target_blocks=0,dispatch=0) or sealed['image']['counts_at_seal']!=dict(expected,native_transitions=530,acts=16,receipts=36,images=10): raise ValueError('unexpected consumer/target work before image seal')
    if any(any(sealed[key]['target_counts'].values()) for key in sealed): raise ValueError('target observed before predictions')
    timeline=[json.loads(line) for line in (data/'timeline.jsonl').read_text().splitlines()]; at=[x['kind'] for x in timeline].index('image_forecasts_sealed')
    if any(row['entries']['target_blocks'] or row['entries']['dispatch'] for row in timeline[:at+1]) or sum(row['kind']=='image_complete' for row in timeline[:at])!=10: raise ValueError('target chronology differs')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py')); stores={key:typed['audit_store'](data/key) for key in ('store','source/store')}; B=typed['quantize_bf16']
    slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']; public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']; branches=defaultdict(list)
    for record in receipts:
        body=record['body']; public(body)
        if body!=read(data/'store/receipts'/(body['fingerprint']+'.json')): raise ValueError('durable receipt differs')
        branches[record['branch']].append(body)
    if len(receipts)!=56 or sum(r['body'].get('steps',0) for r in receipts)!=746: raise ValueError('receipt work differs')
    by={(row['id'],row['mode']):row for row in rows}; arrays_checked=set(); images={}; source_rgb={}
    if len(rows)!=14 or len(by)!=14: raise ValueError('image panel incomplete/duplicated')
    def arr(record,root=data):
        value=load_tensor_archive(root,record)
        if not np.isfinite(value).all() or not np.array_equal(value,B(value)): raise ValueError('finite BF16 evidence differs')
        arrays_checked.add(str(root/record['path'])); return value
    def fields(which,fp): return {item['key']['value']:item['value'] for item in stores[which]['cuts'][fp]['payload']['items']}
    required_stages={'attention.raw','image.after_attention','text.after_attention',*['text.'+x for x in ('modulated','q_projection','k_projection','v_projection','q_normalized','k_normalized','q_positioned','k_positioned')]}
    required_fields={'attention','mlp_input','mlp','exit'};required_native=required_stages|{role+'.'+field for role in ('image','text') for field in required_fields};logical_arrays=0
    if set(forecasts)!={s['id'] for s in c['cases']} or set(native)!=set(forecasts) or set(baselines)!={'half','half.image1'}:raise ValueError('prediction/native/anchor case census differs')
    for by_mode in [*forecasts.values(),*baselines.values()]:
        if set(by_mode)!=set(modes):raise ValueError('program census differs')
        for record in by_mode.values():
            if set(record['fields'])!={'image','text'} or any(set(v)!=required_fields for v in record['fields'].values()) or set(record['stages'])!=required_stages:raise ValueError('forecast port census differs')
            for item in [*record['stages'].values(),*[item for fields in record['fields'].values() for item in fields.values()]]:arr(item);logical_arrays+=1
    for record in native.values():
        if set(record['stages'])!=required_native:raise ValueError('native port census differs')
        for item in record['stages'].values():arr(item);logical_arrays+=1
    if logical_arrays!=304 or {str(p) for p in (data/'arrays').glob('*.npy')}!=arrays_checked or len(arrays_checked)!=summary['unique_array_files']:raise ValueError('array/reference census differs')
    for anchor in baselines:
        if baselines[anchor]['control']['stages']!=baselines[anchor]['hybrid']['stages'] or any(baselines[anchor]['control']['fields'][role]['attention']!=baselines[anchor]['hybrid']['fields'][role]['attention'] for role in ('image','text')):raise ValueError('baseline receiving-LN programs changed upstream stage')
    for anchor in ('half','half.image1'):
        row=by[anchor,'replay']; p=data/row['png']
        if sha(p)!=c['anchors'][anchor]['png_sha256'] or row['png_sha256']!=sha(p): raise ValueError('historical suffix/image replay differs')
        source_rgb[anchor]=np.asarray(Image.open(p).convert('RGB')).copy()
    for row in rows:
        mode=row['mode']; name=row['id']; chain=branches[name+'.'+mode]; anchor=c['anchors'][row['anchor']]; parent=c['parents']['base'] if mode=='native' else anchor['exit']
        for receipt in chain:
            if receipt['parent']!=parent: raise ValueError('branch lineage broken')
            parent=receipt['result']
        if parent!=row['final'] or sum(r.get('steps',0) for r in chain)!=(54 if mode=='native' else 53): raise ValueError('consumer horizon differs')
        finalpoint=typed['unpack'](stores['store']['cuts'][row['final']]['execution_point'])
        if finalpoint['logical_step']!=4: raise ValueError('image final clock differs')
        if mode=='replay':
            if any(r['operation']=='apply' for r in chain) or row['entry']!=anchor['exit']: raise ValueError('replay changed state')
        else:
            if [r['operation'] for r in chain[:2]]!=['apply','apply'] or chain[1]['result']!=row['entry']: raise ValueError('typed writes/entry differ')
            entry=fields('store',row['entry']); scaffold=fields('source/store',c['parents']['base'] if mode=='native' else anchor['exit'])
            if any(entry[k]!=scaffold[k] for k in scaffold if k not in {'image','text'}): raise ValueError('undeclared closure changed')
            if mode!='native':
                for index,role in enumerate(('image','text')):
                    value=arr(forecasts[name][mode]['fields'][role]['exit'])
                    if not typed['matches'](value,entry[role],bf16=True) or chain[index]['act']['writes']!=[role]: raise ValueError('installed predicted write differs')
                if any(row['target_counts_at_image'].values()) or pixel_forecasts[name+'.'+mode]['png_sha256']!=row['png_sha256']: raise ValueError('candidate image not before targets')
        p=data/row['png']
        if sha(p)!=row['png_sha256']: raise ValueError('image bytes differ')
        rgb=np.asarray(Image.open(p).convert('RGB')).copy(); images[name,mode]=rgb
        if common['scene_readout'](rgb,source_rgb[row['anchor']])!=row['readout']: raise ValueError('source-support readout differs')
        if mode not in ('replay','native') and row['readout']!=pixel_forecasts[name+'.'+mode]['readout']: raise ValueError('readout forecast not frozen')
    old_native=read(data/'source/prefixes/native.json')
    for anchor,record in c['anchors'].items():
        if any(record[k]!=old_native[anchor][k] for k in ('entry','exit','stages')): raise ValueError('historical calibration join differs')
        for role in ('image','text'):
            if not np.array_equal(arr(record['stages'][role+'.exit'],data/'source/prefixes'),slot(data/'source/store',record['exit'],role)): raise ValueError('anchor port/state differs')
    T0=slot(data/'source/store',c['parents']['base'],'text'); T1=slot(data/'source/store',c['parents']['edited'],'text'); numerical={}; stage_reports={}; outcomes={}; exits={}; actuals={}
    for spec in c['cases']:
        name=spec['id']; anchor=spec['anchor']; n=native[name]; chain=branches[name+'.native']; newtext=T0.copy()
        if n['entry']!=by[name,'native']['entry'] or chain[2]['operation']!='continue' or chain[2]['steps']!=1 or chain[2]['parent']!=n['entry'] or chain[2]['result']!=n['exit']: raise ValueError('target disconnected from future')
        point=typed['unpack'](stores['store']['cuts'][n['exit']]['execution_point'])
        if any(point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=2).items()): raise ValueError('target exit clock differs')
        for row,dose in zip((4,11),spec['row_doses']): newtext[:,row]=B(T0[:,row].astype(float)+dose*(T1[:,row].astype(float)-T0[:,row].astype(float)))
        if not np.array_equal(newtext,slot(data/'store',n['entry'],'text')) or not np.array_equal(slot(data/'store',n['entry'],'image'),slot(data/'source/store',c['anchors'][anchor]['entry'],'image')): raise ValueError('fresh input construction differs')
        numerical[name]={};stage_reports[name]={};outcomes[name]={};exits[name]={};actuals[name]={}
        nrgb=images[name,'native'].astype(float); rgb0=source_rgb[anchor].astype(float)
        for mode in modes:
            p=forecasts[name][mode]; p0=baselines[anchor][mode]; numerical[name][mode]={};stage_reports[name][mode]={};exits[name][mode]={}
            for key,record in p['stages'].items():
                x,x0=arr(record),arr(p0['stages'][key]); y=arr(n['stages'][key]); y0=arr(c['anchors'][anchor]['stages'][key],data/'source/prefixes')
                if key.startswith('text.') and key!='text.after_attention':y0=y0[:,[4,11]]
                stage_reports[name][mode][key]=audit_response(y0,y,x0,x)
            for role in ('image','text'):
                numerical[name][mode][role]={}
                for field,record in p['fields'][role].items():
                    x,x0=arr(record),arr(p0['fields'][role][field]);y=arr(n['stages'][role+'.'+field]);y0=arr(c['anchors'][anchor]['stages'][role+'.'+field],data/'source/prefixes')
                    report=audit_response(y0,y,x0,x);report['installed_edit']=compare(x.astype(float)-y0.astype(float),y.astype(float)-y0.astype(float));numerical[name][mode][role][field]=report
                    if field=='exit':
                        if not np.array_equal(y,slot(data/'store',n['exit'],role)): raise ValueError('native exit port/state differs')
                        exits[name][mode][role]=x;actuals[name][role]=y
            prgb=images[name,mode].astype(float);frozen=by[name,mode]['readout'];observed=by[name,'native']['readout'];regions={}
            for region,pred in frozen['regions'].items():
                actual=observed['regions'][region];regions[region]=dict(predicted_delta_rgb=pred['delta_rgb'],native_delta_rgb=actual['delta_rgb'],delta_comparison=compare(pred['delta_rgb'],actual['delta_rgb']),predicted_red_minus_blue_delta=pred['red_minus_blue_delta'],native_red_minus_blue_delta=actual['red_minus_blue_delta'],direction_matches=bool(np.sign(pred['red_minus_blue_delta'])==np.sign(actual['red_minus_blue_delta'])))
            outcomes[name][mode]=dict(rgb_mae=float(np.abs(prgb-nrgb).mean()),rgb_max_abs=float(np.abs(prgb-nrgb).max()),rgb_endpoint=compare(prgb,nrgb),rgb_edit=compare(prgb-rgb0,nrgb-rgb0),copy_anchor=compare(np.zeros_like(rgb0),nrgb-rgb0),regions=regions,background_collateral=dict(predicted_mae=frozen['background_outside_source_boxes_mae'],native_mae=observed['background_outside_source_boxes_mae'],signed_error=frozen['background_outside_source_boxes_mae']-observed['background_outside_source_boxes_mae']))
        for key in forecasts[name]['control']['stages']:
            if forecasts[name]['control']['stages'][key]!=forecasts[name]['hybrid']['stages'][key]: raise ValueError('receiving-LN programs changed upstream stage')
        for role in ('image','text'):
            if forecasts[name]['control']['fields'][role]['attention']!=forecasts[name]['hybrid']['fields'][role]['attention']: raise ValueError('attention differs between receiving-LN programs')
    color_ordering={};factorial={}
    for suffix,receiver in [('', 'base'),('.image1','edited')]:
        a,b='row4seven_eighths'+suffix,'row11seven_eighths'+suffix;color_ordering[receiver]={}
        for region in ('left_source_object','right_source_object'):
            result={}
            for mode in (*modes,'native'):
                ra=by[a,mode]['readout']['regions'][region]['mean_rgb'];rb=by[b,mode]['readout']['regions'][region]['mean_rgb'];value=(ra[0]-ra[2])-(rb[0]-rb[2]);result[mode]=dict(row4heavy_minus_row11heavy_red_blue=value,matches_frozen_negative_direction=value<0)
            color_ordering[receiver][region]=result
    order=['row11seven_eighths','row4seven_eighths','row11seven_eighths.image1','row4seven_eighths.image1']
    for mode in modes:
        factorial[mode]={}
        for role in ('image','text','RGB'):
            n={k:images[name,'native'] if role=='RGB' else actuals[name][role] for k,name in zip(('00','01','10','11'),order)};p={k:images[name,mode] if role=='RGB' else exits[name][mode][role] for k,name in zip(('00','01','10','11'),order)}
            report=audit_factorial(n,p);report['axes']=dict(first='known image receiver',second='complementary row-dose pattern; not independent row4 x row11 factorial');factorial[mode][role]=report
    primary={mode:float(np.mean([numerical[s['id']][mode]['image']['exit']['reference_response']['relative_rms'] for s in c['cases']])) for mode in modes}
    primary=dict(mean_image_response_error=primary,signed_hybrid_minus_control=primary['hybrid']-primary['control'],per_case_hybrid_minus_control={s['id']:numerical[s['id']]['hybrid']['image']['exit']['reference_response']['relative_rms']-numerical[s['id']]['control']['image']['exit']['reference_response']['relative_rms'] for s in c['cases']},interpretation='Signed observation for frozen mean hypothesis; no discovery gate.')
    output.parent.mkdir(parents=True,exist_ok=True); sheet=Image.new('RGB',(4*384,4*408),'white');draw=ImageDraw.Draw(sheet)
    for y,spec in enumerate(c['cases']):
        name=spec['id'];panels=[('half baseline',source_rgb[spec['anchor']]),('control LN',images[name,'control']),('hybrid LN',images[name,'hybrid']),('native target',images[name,'native'])]
        for x,(label,rgb) in enumerate(panels):sheet.paste(Image.fromarray(rgb).resize((384,384)),(x*384,y*408+24));draw.text((x*384+6,y*408+5),name+' | '+label,fill='black')
    sheet_path=output.with_suffix('.png');sheet.save(sheet_path)
    result=dict(schema='mdb-hybrid-future-audit-v1',mechanics_status='verified',instrument_status='complete paired P0/P1, typed native consumer lineage, pretarget image forecasts; supplied source supports, visual semantics reviewed separately',terminal_status='unassessed',custody=custody,transactions=transactions,counts=expected,frozen_hypotheses=c['hypotheses'],primary=primary,numerical=numerical,stages=stage_reports,outcomes=outcomes,color_ordering=color_ordering,factorial=factorial,typed_stores={k:v['counts'] for k,v in stores.items()},logical_prediction_native_array_references=logical_arrays,external_array_files_verified=len(arrays_checked),contact_sheet=dict(path=str(sheet_path),sha256=sha(sheet_path)),worker_seconds=summary['worker_seconds'],auditor_source_sha256=sha(Path(__file__)),new_native_calls_in_audit=0,scope=c['hypotheses']['scope'])
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',primary=primary,rgb={name:{mode:dict(mae=row['rgb_mae'],edit_error=row['rgb_edit']['relative_rms']) for mode,row in modes.items()} for name,modes in outcomes.items()},color_ordering=color_ordering)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(analyze(a.collection.resolve(),a.output.resolve()),sort_keys=True))
