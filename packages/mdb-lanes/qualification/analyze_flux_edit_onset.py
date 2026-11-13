"""Saved-only audit of onset futures, repeated text forecasts and clock traces."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def read(p):return json.loads(p.read_text())


def index_receipts(receipts):
    result={}
    for receipt in receipts:
        key=receipt['fingerprint']
        if key in result and result[key]!=receipt:
            raise ValueError('Receipt identity collision with different bodies')
        result[key]=receipt
    return result


def analyze(run,destination,source):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/'extracted/edit-onset-result';old=source/'run-03/extracted/frozen-composition-result'
    custody_helper=runpy.run_path(str(here/'flux_family_custody.py'))
    custody=custody_helper['verify'](run);source_custody=custody_helper['verify'](source/'run-03')
    u=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=u['sha'],u['metric']
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'))
    json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree']
    public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    hue=runpy.run_path(str(here/'flux_frozen_composition.py'))['hue_observer']
    config,summary,binding,rows,receipts=[read(data/n) for n in ('plan.json','summary.json','binding.json','rows.json','receipts.json')]
    request=read(run/'request.json');assert request['config']==config and summary==read(run/'report.json')['report']
    assert request['request_sha256']==summary['request_sha256']==custody['request_sha256']
    assert sha(source/'analysis-v1.json')==config['source_audit_sha256']
    for name,h in request['source_hashes'].items():assert sha(run/'payload'/name)==h
    for name,m in config['input_files'].items():assert sha(old/name)==m['sha256'] and (old/name).stat().st_size==m['bytes']
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    assert json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/frontiers/binding.json')['value']))==binding
    forecast=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/timing_and_repeated_reads.json')['predictions']))
    assert forecast=={'hypotheses':config['hypotheses'],'onsets':config['onsets'],'unchanged_predicted_additive_T':binding['predicted_additive_T']}
    assert binding['native_steps_before_seal']==0
    timeline=read(data/'timeline.json');sealed=next(v for v in timeline if v['kind']=='sealed');assert sealed['counts']['native_transitions']==0
    prediction=load_tensor_archive(data,binding['predicted_additive_T'])
    assert np.array_equal(prediction,load_tensor_archive(old,config['predicted_additive_T']))
    values={k:load_tensor_archive(old,v) for k,v in config['conditioning'].items()}
    old_native=load_tensor_archive(old,read(old/'native-fields.json')['additive'])
    store=typed['audit_store'](data/'store')
    def slot(cut,name):return route['slot_array'](data/'store',cut,name,typed)
    def fields(cut):return {i['key']['value']:i['value'] for i in store['cuts'][cut]['payload']['items']}
    for r in receipts:public(r)
    by_receipt=index_receipts(receipts);assert len(receipts)==2161
    # Identical canonical receipts can recur across physical branch attempts.
    # Retain all emissions and check repeated identities have identical bodies.
    assert all(r==by_receipt[r['fingerprint']] for r in receipts)
    assert len(rows)==25 and len({r['id'] for r in rows})==25
    assert summary['counts']==config['expected_counts']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=2128,acts=33,receipts=2161,images=25)
    assert sum(r['operation']=='continue' for r in receipts)==2128 and sum(r['operation']=='apply' for r in receipts)==33
    frontiers=read(data/'frontiers.json');assert set(frontiers)=={'0','1','2','3','4'}
    assert fields(frontiers['3'])==fields(config['historical_step3'])
    images,latents,field_reads={},{},{}
    for row in rows:
        assert sha(data/row['png'])==row['png_sha256']
        with Image.open(data/row['png']) as im:
            assert im.mode=='RGB' and im.size==(512,512);images[row['id']]=np.asarray(im).copy()
        latents[row['id']]=load_tensor_archive(data,row['latent']);assert np.array_equal(latents[row['id']],slot(row['final_cut'],'latent'))
        previous=config['initial_cut'] if row['id']=='base' else frontiers[str(row['onset'])]
        results=set()
        for receipt in row['path']:
            assert receipt==by_receipt[receipt['fingerprint']] and receipt['parent']==previous
            previous=receipt['result'];results.add(previous)
        assert previous==row['final_cut']
        steps=4-row['onset']
        assert len(row['actions'])==(0 if row['id']=='base' else 1+(steps if row['predicted'] else 0))
        assert len(row['path'])==28*steps+len(row['actions'])
        for index,action in enumerate(row['actions']):
            assert action['receipt']==by_receipt[action['receipt']['fingerprint']] and action['act']['writes']==[action['slot']]
            assert action['post_act'] in results
            before,after=fields(action['parent']),fields(action['post_act'])
            assert all(before[k]==after[k] for k in before if k!=action['slot'])
            if index==0:
                assert action['slot']=='conditioning' and action['parent']==frontiers[str(row['onset'])]
                assert np.array_equal(slot(action['post_act'],'conditioning'),values[row['role']])
            else:
                assert row['predicted'] and action['slot']=='text'
                assert np.array_equal(slot(action['post_act'],'text'),prediction)
        assert [int(v['step']) for v in row['trace']]==list(range(row['onset'],5))
        for v in row['trace']:
            assert v['cut'] in results or v['cut']==config['initial_cut']
            assert read(data/'store/cuts'/(v['cut']+'.json'))['execution_point']['logical_step']==int(v['step'])
            assert np.array_equal(load_tensor_archive(data,v['latent']),slot(v['cut'],'latent'))
        if row['role']=='additive':
            assert [r['step'] for r in row['project_reads']]==list(range(row['onset'],4))
        else:assert not row['project_reads']
        for r in row['project_reads']:
            native=load_tensor_archive(data,r['native_text']);assert r['native_descriptor']==r['native_text']['descriptor']
            assert r['project_receipt']==by_receipt[r['project_receipt']['fingerprint']] and r['project_receipt']['result'] in results
            joined=r['project_receipt']['result'] in store['cuts']
            if joined:assert np.array_equal(native,slot(r['project_receipt']['result'],'text'))
            residual=prediction.astype(np.float64)-native
            field_reads[row['id']+'/project'+str(r['step'])]={'onset':row['onset'],'clock':r['step'],'predicted_branch':row['predicted'],
                'predicted_minus_native':metric(prediction,native),'relative_rms':float(np.linalg.norm(residual)/np.linalg.norm(native.astype(np.float64))),
                'native_equals_historical_additive':bool(np.array_equal(native,old_native)),'typed_post_project_array_join':joined,
                'native_capture':'Full lossless native module return with descriptor and receipt; typed cut retained on predicted branches, archived observation on native branches'}
    assert sha(data/'base.png')==config['historical_pngs']['base']
    shams={}
    for onset in config['onsets']:
        name='onset'+str(onset)+'.base';shams[str(onset)]={'rgb_exact':bool(np.array_equal(images[name],images['base'])),'latent_exact':bool(np.array_equal(latents[name],latents['base']))}
        assert shams[str(onset)]['rgb_exact'] and shams[str(onset)]['latent_exact']
    # Historical onset3 is kept separate: these are reused outcomes, not new jobs.
    original={r['id']:r for r in read(old/'rows.json')}
    for role in (*values,'stacked_additive'):
        name='predicted.additive' if role=='stacked_additive' else 'native.'+role;r=original[name]
        assert sha(old/r['png'])==r['png_sha256']
        with Image.open(old/r['png']) as im:images['onset3.'+role]=np.asarray(im).copy()
        latents['onset3.'+role]=load_tensor_archive(old,r['latent'])
    observations={}
    for name,image in images.items():
        onset=int(name[5]) if name.startswith('onset') else 0
        matched=images['onset'+str(onset)+'.joint']
        observations[name]={}
        for label,box in {'whole':None,**config['observers']['windows']}.items():
            crop=lambda im:im if box is None else im[box[1]:box[3],box[0]:box[2]]
            value=crop(image)
            observations[name][label]={'to_base':metric(value,crop(images['base'])),'to_matched_joint':metric(value,crop(matched)),
                'to_full_joint':metric(value,crop(images['onset0.joint'])),'hue':hue(value)}
    pairs={str(k):{'rgb':metric(images['onset'+str(k)+'.stacked_additive'],images['onset'+str(k)+'.additive']),
        'latent':metric(latents['onset'+str(k)+'.stacked_additive'],latents['onset'+str(k)+'.additive'])} for k in range(4)}
    timing={role:{str(k):{'latent_to_full_joint':metric(latents['onset'+str(k)+'.'+role],latents['onset0.joint']),
        'eye_mae_to_matched_joint':observations['onset'+str(k)+'.'+role]['eyes']['to_matched_joint']['mae'],
        'whole_mae_to_matched_joint':observations['onset'+str(k)+'.'+role]['whole']['to_matched_joint']['mae']} for k in range(4)} for role in values}
    report={'schema':'mdb-edit-onset-audit-v1','job_id':summary['job_id'],'request_sha256':request['request_sha256'],'mechanics_status':'verified','instrument_status':'manualwindows-diagnostic-hue-unblinded-geometry','trend_status':'preserve-per-onset','terminal_status':'unassessed',
        'custody':custody,'source_custody':source_custody,'transaction':tx,'counts':summary['counts'],'replay':{'historical_step3_payload_exact':True,'historical_base_png_exact':True,'shams':shams},
        'projection_reads':field_reads,'observations':observations,'repeated_prediction_pairs':pairs,'timing':timing,
        'forecast_scores':{'repeated_prediction_rgb_below1':{k:v['rgb']['mae']<1 for k,v in pairs.items()},'scope':'Forecast scores are observations, not discovery gates. Geometry requires visual native controls; latent/RGB distance is not a shape label.'},
        'new_images':25,'historical_images_reused':8,'worker_seconds':summary['worker_seconds'],'scope':config['scope']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    roles=['base','color','shape','joint','additive','plus_old','minus_old','stacked_additive']
    for kind in ('full','eyes'):
        w,h=(256,284) if kind=='full' else (500,250)
        sheet=Image.new('RGB',(8*w,4*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for onset in range(4):
            for j,role in enumerate(roles):
                name='onset'+str(onset)+'.'+role;im=Image.fromarray(images[name])
                im=im.resize((256,256)) if kind=='full' else im.crop((130,180,380,290)).resize((500,220))
                sheet.paste(im,(j*w,onset*h+26));draw.text((j*w+4,onset*h+7),name,fill='black')
        sheet.save(destination.parent/('onset-'+kind+'.png'))
    return {'job_id':summary['job_id'],'new_images':25,'reused_images':8,'prediction_rgb_mae':{k:v['rgb']['mae'] for k,v in pairs.items()},
        'additive_eye_mae':{str(k):timing['additive'][str(k)]['eye_mae_to_matched_joint'] for k in range(4)},
        'native_project_reads':len(field_reads),'all_project_reads_equal_historical':all(v['native_equals_historical_additive'] for v in field_reads.values())}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve()),sort_keys=True))
