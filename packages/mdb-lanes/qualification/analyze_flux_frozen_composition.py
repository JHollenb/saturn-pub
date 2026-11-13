"""Independent saved-only audit of target-quarantined contextual composition."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def read(p): return json.loads(p.read_text())


def analyze(run, destination, source):
    from mdb_lanes.forecast_transaction import verify_transaction, load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent
    u=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=u['sha'],u['metric']
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'))
    error=runpy.run_path(str(here/'analyze_flux_mlp_replacement.py'))['field_error']
    route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'))
    json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree']
    observer=runpy.run_path(str(here/'flux_frozen_composition.py'))['hue_observer']
    q=typed['quantize_bf16']
    data=run/'extracted/frozen-composition-result';origin=source/'run-01/extracted/conditioning-program-result'
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](run)
    summary,config,binding,rows,receipts=[read(data/n) for n in ('summary.json','plan.json','binding.json','rows.json','receipts.json')]
    request=read(run/'request.json');assert request['config']==config and summary==read(run/'report.json')['report']
    assert summary['request_sha256']==request['request_sha256']==custody['request_sha256']
    for name,h in request['source_hashes'].items(): assert sha(run/'payload'/name)==h
    assert sha(source/'analysis-v1.json')==config['source_audit_sha256']
    for name,m in config['input_files'].items(): assert (origin/name).stat().st_size==m['bytes'] and sha(origin/name)==m['sha256']
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    assert json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/frontiers/binding.json')['value']))==binding
    forecast=read(data/'forecasts.json')
    sealed=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/frozen_rules.json')['predictions']))
    assert sealed=={'initial_text':forecast,'hypotheses':config['hypotheses'],'rules':config['rules']}
    assert binding['new_joint_encoded'] is False and binding['native_transitions_before_seal']==0
    timeline=read(data/'timeline.json');seal_index=next(i for i,v in enumerate(timeline) if v['kind']=='sealed')
    assert timeline[seal_index]['counts']['encodes']==3 and timeline[seal_index]['counts']['native_transitions']==0
    assert [v['role'] for v in timeline[:seal_index] if v['kind']=='encoded']==['base','color','shape']
    assert [v['role'] for v in timeline[seal_index+1:] if v['kind']=='encoded']==['joint']
    arr=lambda r:load_tensor_archive(data,r)
    sources={k:arr(v) for k,v in binding['observed_sources'].items()};assert set(sources)=={'base','color','shape'}
    old={k:load_tensor_archive(origin,r) for k,r in binding['old_conditioning'].items()}
    additive=q((sources['color'].astype(np.float64)+sources['shape']-sources['base']).astype(np.float32))
    correction=old['joint'].astype(np.float64)-old['additive']
    values={'additive':additive,'plus_old':q((additive.astype(np.float64)+correction).astype(np.float32)),'minus_old':q((additive.astype(np.float64)-correction).astype(np.float32))}
    for k,v in values.items():assert np.array_equal(v,arr(binding['composed'][k]))
    joint=arr(read(data/'new-joint.json')['encoding']);assert read(data/'new-joint.json')['observed_after_seal']
    assert np.array_equal(arr(binding['source_ids']),arr(read(data/'new-joint.json')['ids']))
    weights=arr(binding['weight']);bias=None if binding['bias'] is None else arr(binding['bias'])
    native=read(data/'native-fields.json');equations={}
    for role,v in values.items():
        rebuilt=np.empty((*v.shape[:2],weights.shape[0]),np.float32)
        for start in range(0,weights.shape[0],128):
            stop=min(start+128,weights.shape[0]);x=v.astype(np.float64)@weights[start:stop].astype(np.float64).T
            if bias is not None:x+=bias[start:stop].astype(np.float64)
            rebuilt[:,:,start:stop]=q(x.astype(np.float32))
        equations[role]={'sealed_minus_independent':error(arr(forecast[role]),rebuilt),'sealed_minus_native':error(arr(forecast[role]),arr(native[role]))}
    store=typed['audit_store'](data/'store')
    def slot(cut,key):return route['slot_array'](data/'store',cut,key,typed)
    def fields(cut):return {i['key']['value']:i['value'] for i in store['cuts'][cut]['payload']['items']}
    public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    for r in receipts: public(r)
    by_receipt={r['fingerprint']:r for r in receipts}
    frontier=read(data/'frontier.json');previous=frontier['initial']
    assert len(frontier['path'])==84
    for r in frontier['path']:
        assert r==by_receipt[r['fingerprint']] and r['parent']==previous;previous=r['result']
    assert previous==frontier['common_project_frontier']
    images,latents={},{};raw={r['id']:r for r in rows}
    assert len(rows)==11 and len(raw)==11
    all_values={**sources,'joint':joint,**values}
    for row in rows:
        assert sha(data/row['png'])==row['png_sha256']
        with Image.open(data/row['png']) as im:
            assert im.mode=='RGB' and im.size==(512,512);images[row['id']]=np.asarray(im).copy()
        latents[row['id']]=arr(row['latent']);assert np.array_equal(latents[row['id']],slot(row['final_cut'],'latent'))
        previous=None
        for r in row['path']:
            assert r==by_receipt[r['fingerprint']]
            if previous: assert r['parent']==previous
            previous=r['result']
        assert previous==row['final_cut'] and len(row['path'])==(27 if row['id'].startswith('predicted.') else 28)
        for action in row['actions']:
            assert action['receipt']==by_receipt[action['receipt']['fingerprint']] and action['act']['writes']==[action['slot']]
            before,after=fields(action['parent']),fields(action['post_act'])
            assert all(before[k]==after[k] for k in before if k!=action['slot'])
            assert row['path'][0]['parent']==action['post_act']
        if row['id'].startswith('native.'):
            role=row['id'].split('.',1)[1];a=row['actions'][0]
            assert a['slot']=='conditioning' and a['parent']==frontier['common_project_frontier']
            assert np.array_equal(slot(a['post_act'],'conditioning'),all_values[role])
            assert np.array_equal(slot(row['path'][0]['result'],'text'),arr(native[role]))
        elif row['id'].startswith('predicted.'):
            role=row['id'].split('.',1)[1];a=row['actions'][0]
            assert a['slot']=='text' and a['parent']==raw['native.'+role]['path'][0]['result']
            assert np.array_equal(slot(a['post_act'],'text'),arr(forecast[role]))
        else:assert row['id']=='plain.base' and not row['actions'] and row['path'][0]['parent']==frontier['common_project_frontier']
    assert np.array_equal(images['plain.base'],images['native.base']) and np.array_equal(latents['plain.base'],latents['native.base'])
    assert len(receipts)==399 and len(by_receipt)==399 and sum(r['operation']=='continue' for r in receipts)==389 and sum(r['operation']=='apply' for r in receipts)==10
    observations={}
    for name,image in images.items():
        windows={'whole':image,**{k:image[box[1]:box[3],box[0]:box[2]] for k,box in config['observers']['windows'].items()}}
        observations[name]={}
        for label,value in windows.items():
            box=config['observers']['windows'].get(label)
            crop=lambda im:im if box is None else im[box[1]:box[3],box[0]:box[2]]
            observations[name][label]={'to_base':metric(value,crop(images['native.base'])),'to_joint':metric(value,crop(images['native.joint'])),'hue':observer(value)}
    pairs={k:{'rgb':metric(images['predicted.'+k],images['native.'+k]),'latent':metric(latents['predicted.'+k],latents['native.'+k])} for k in values}
    forecasts_e={k:metric(v,joint) for k,v in values.items()}
    eye={k:observations['native.'+k]['eyes']['to_joint']['mae'] for k in values}
    report={'schema':'mdb-frozen-composition-audit-v1','job_id':summary['job_id'],'request_sha256':request['request_sha256'],'mechanics_status':'verified','instrument_status':'diagnostic-native-controls-and-visual-review-required','trend_status':'preserve-per-case','terminal_status':'unassessed','custody':custody,'transaction':tx,'counts':summary['counts'],'quarantine':{'sources_encoded_before_seal':3,'new_joint_encoding_after_seal':True,'native_steps_before_seal':0},'full_projection_equations':equations,'encoding_to_new_joint':forecasts_e,'observations':observations,'prediction_pairs':pairs,'forecast_scores':{'eye_distance':eye,'plus_old_better_than_additive':eye['plus_old']<eye['additive'],'minus_old_worse_than_additive':eye['minus_old']>eye['additive'],'numerical_rgb_below1':{k:v['rgb']['mae']<1 for k,v in pairs.items()},'scope':'Scoring frozen forecasts; these booleans do not gate discovery or erase outcomes.'},'base_sham_exact':True,'worker_seconds':summary['worker_seconds'],'scope':config['scope']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        width,height=(384,415) if kind=='full' else (460,190)
        sheet=Image.new('RGB',(4*width,3*height),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for index,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);x,y=(index%4)*width,(index//4)*height
            im=im.resize((384,384)) if kind=='full' else im.crop((140,155,370,235)).resize((460,160))
            sheet.paste(im,(x,y+26));draw.text((x+5,y+7),row['id'],fill='black')
        sheet.save(destination.parent/('frozen-'+kind+'.png'))
    return {'job_id':summary['job_id'],'forecast_scores':report['forecast_scores'],'pairs':pairs,'encoding_l2':{k:v['l2'] for k,v in forecasts_e.items()},'native_relative_rms':{k:v['sealed_minus_native']['relative_rms'] for k,v in equations.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve()),sort_keys=True))
