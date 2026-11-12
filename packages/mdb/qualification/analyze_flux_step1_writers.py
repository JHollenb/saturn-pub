"""Independent audit of pulse persistence, carrier handoffs and joint write fields."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination,source):
    from mdb.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/'extracted/step1-writers-result';old=source/'run-01/extracted/edit-onset-result'
    custody_tool=runpy.run_path(str(here/'flux_family_custody.py'));custody=custody_tool['verify'](run);source_custody=custody_tool['verify'](source/'run-01')
    util=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=util['sha'],util['metric']
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));q=typed['quantize_bf16'];route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'))
    json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree'];public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    index_receipts=runpy.run_path(str(here/'analyze_flux_edit_onset.py'))['index_receipts']
    error=runpy.run_path(str(here/'analyze_flux_mlp_replacement.py'))['field_error']
    config,summary,binding,rows,receipts=[read(data/n) for n in ('plan.json','summary.json','binding.json','rows.json','receipts.json')]
    request=read(run/'request.json');assert request['config']==config and summary==read(run/'report.json')['report']
    assert summary['request_sha256']==request['request_sha256']==custody['request_sha256']
    assert sha(source/'analysis-v1.json')==config['source_audit_sha256']
    for name,digest in request['source_hashes'].items():assert sha(run/'payload'/name)==digest
    for name,m in config['input_files'].items():assert sha(old/name)==m['sha256'] and (old/name).stat().st_size==m['bytes']
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    assert json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/frontiers/binding.json')['value']))==binding
    declared=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/pulse_and_writers.json')['predictions']))
    assert declared=={k:config[k] for k in ('hypotheses','handoffs','write_rows')}
    timeline=read(data/'timeline.json');assert next(r for r in timeline if r['kind']=='sealed')['counts']['native_transitions']==0
    numeric=read(data/'numeric-seal.json');nb=read(data/'numeric-binding.json');assert sha(data/'numeric-binding.json')==numeric['binding_sha256']
    assert sha(data/'numeric-seal.json')==summary['numeric_seal_sha256']
    assert numeric['counts_at_seal']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=112,acts=4,receipts=116,images=0,candidate_blocks=2)
    assert next(r for r in timeline if r['kind']=='numeric_sealed')['sha256']==summary['numeric_seal_sha256']
    store=typed['audit_store'](data/'store')
    def slot(cut,name):return route['slot_array'](data/'store',cut,name,typed)
    def fields(cut):return {v['key']['value']:v['value'] for v in store['cuts'][cut]['payload']['items']}
    def point(cut):return read(data/'store/cuts'/(cut+'.json'))['execution_point']
    arr=lambda r:load_tensor_archive(data,r)
    E={k:route['slot_array'](old/'store',v,'conditioning',typed) for k,v in config['program_cuts'].items()}
    ports={role:{k:arr(v) for k,v in values.items()} for role,values in nb['ports'].items()}
    assert all(np.array_equal(ports['base'][k],ports['shape'][k]) for k in ('residual','gate_attention','gate_mlp'))
    equations={};predicted={}
    for bits,record in numeric['predicted_exits'].items():
        b=ports['base'];A=ports['shape' if bits[0]=='1' else 'base']['attention'];M=ports['shape' if bits[1]=='1' else 'base']['mlp']
        a=q((b['gate_attention'].astype(np.float64)*A).astype(np.float32))
        mid=q((b['residual'].astype(np.float64)+a).astype(np.float32))
        m=q((b['gate_mlp'].astype(np.float64)*M).astype(np.float32))
        rebuilt=q((mid.astype(np.float64)+m).astype(np.float32));predicted[bits]=arr(record)
        equations[bits]={'sealed_minus_independent':error(predicted[bits],rebuilt)}
    validations=read(data/'write-validations.json')
    for bits,value in validations.items():
        equations[bits]['sealed_minus_native_fullblock']=error(predicted[bits],arr(value['native_image']))
    for bits,role in (('00','base'),('11','shape')):
        equations[bits]['sealed_minus_source_native_exit']=error(predicted[bits],slot(nb['post_joint'][role],'image'))
    for r in receipts:public(r)
    by_receipt=index_receipts(receipts);assert len(receipts)==1690
    assert sum(r['operation']=='continue' for r in receipts)==1646 and sum(r['operation']=='apply' for r in receipts)==44
    assert summary['counts']==config['expected_counts']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=1646,acts=44,receipts=1690,images=22,candidate_blocks=5)
    cuts=read(data/'cuts.json');raw={r['id']:r for r in rows};assert len(rows)==len(raw)==22
    for site in config['handoffs']['sites']:
        base,changed=[fields(cuts['step1'][k][site]) for k in ('base','shape')]
        assert all(base[k]==changed[k] for k in base if k not in ('conditioning','image','text'))
    images,latents={},{}
    remains={'after_joint0':82,'after_joint4':78,'after_single6':71,'after_single18':59}
    for row in rows:
        assert row==read(data/'rows'/(row['id']+'.json')) and sha(data/row['png'])==row['png_sha256']
        with Image.open(data/row['png']) as im:assert im.mode=='RGB' and im.size==(512,512);images[row['id']]=np.asarray(im).copy()
        latents[row['id']]=arr(row['latent']);assert np.array_equal(latents[row['id']],slot(row['final_cut'],'latent'))
        previous=row['parent'];results=set()
        for receipt in row['path']:
            assert receipt==by_receipt[receipt['fingerprint']] and receipt['parent']==previous
            previous=receipt['result'];results.add(previous)
        assert previous==row['final_cut']
        native_steps={'pulse':84,'continuous':56,'write-law':82}.get(row['kind'],remains.get(row.get('site')))
        assert sum(v['operation']=='continue' for v in row['path'])==native_steps
        assert len(row['path'])==native_steps+len(row['actions'])
        for action in row['actions']:
            assert action['receipt']==by_receipt[action['receipt']['fingerprint']] and action['act']['writes']==[action['slot']] and action['post_act'] in results
            before,after=fields(action['parent']),fields(action['post_act']);assert all(before[k]==after[k] for k in before if k!=action['slot'])
            p=point(action['parent'])
            if action['slot']=='conditioning':
                role=row['role'] if p['logical_step']==1 else 'base'
                assert p['phase']=='project' and p['logical_step'] in (1,2)
                assert np.array_equal(slot(action['post_act'],'conditioning'),E[role])
            elif row['kind']=='handoff':
                assert action['slot'] in ('image','text')
                assert np.array_equal(slot(action['post_act'],action['slot']),slot(row['source'],action['slot']))
            else:
                assert row['kind']=='write-law' and action['slot']=='image' and np.array_equal(slot(action['post_act'],'image'),predicted[row['bits']])
        assert [v['step'] for v in row['trace']]==[2,3,4]
        future_role=row['role'] if row['kind']=='continuous' else 'base'
        for t in row['trace']:
            assert t['cut'] in results or t['cut']==row['parent']
            assert point(t['cut'])['logical_step']==t['step'] and point(t['cut'])['phase']=='project'
            assert np.array_equal(arr(t['latent']),slot(t['cut'],'latent')) and np.array_equal(slot(t['cut'],'conditioning'),E[future_role])
    controls={}
    for row in rows:
        if 'expected_png' in row:
            value=row['png_sha256']==row['expected_png'];assert value==row['png_control_exact'];controls[row['id']]={'png_exact':value}
    for site in config['handoffs']['sites']:
        controls[site+'.IT']['latent_exact']=bool(np.array_equal(latents[site+'.IT'],latents['pulse.shape']))
    observations={}
    for name,image in images.items():
        observations[name]={}
        for label,box in {'whole':None,**config['observers']['windows']}.items():
            crop=lambda v:v if box is None else v[box[1]:box[3],box[0]:box[2]]
            value,base,target=crop(image),crop(images['pulse.base']),crop(images['pulse.shape'])
            d=target.astype(np.float64)-base;denom=float(np.sum(d*d))
            observations[name][label]={'to_base':metric(value,base),'to_pulse_shape':metric(value,target),'to_continuous_shape':metric(value,crop(images['continuous.shape'])),
                'pulse_shape_contrast_projection':None if not denom else float(np.sum((value.astype(np.float64)-base)*d)/denom)}
    cube={'00.base':'pulse.base','00.shape':'after_joint0.T','11.base':'after_joint0.I','11.shape':'pulse.shape',
        **{bits+'.'+role:'law.A'+bits[0]+'M'+bits[1]+'.T'+role for bits in ('10','01') for role in ('base','shape')}}
    effects={}
    for axis in ('A','M','T'):
        effects[axis]=[]
        for key,name in cube.items():
            bits,text=key.split('.')
            if (axis=='A' and bits[0]!='0') or (axis=='M' and bits[1]!='0') or (axis=='T' and text!='base'):continue
            other=('1'+bits[1]+'.'+text if axis=='A' else bits[0]+'1.'+text if axis=='M' else bits+'.shape')
            low=observations[name]['eyes']['pulse_shape_contrast_projection'];high=observations[cube[other]]['eyes']['pulse_shape_contrast_projection']
            effects[axis].append({'low':key,'high':other,'eye_projection_delta':None if low is None or high is None else high-low})
    forecast_scores={'early_T_better_than_I_eye_mae':observations['after_joint0.T']['eyes']['to_pulse_shape']['mae']<observations['after_joint0.I']['eyes']['to_pulse_shape']['mae'],
        'late_I_better_than_T_eye_mae':{s:observations[s+'.I']['eyes']['to_pulse_shape']['mae']<observations[s+'.T']['eyes']['to_pulse_shape']['mae'] for s in ('after_single6','after_single18')},
        'A1M0_better_than_A0M1_at_baseT':observations['law.A1M0.Tbase']['eyes']['to_pulse_shape']['mae']<observations['law.A0M1.Tbase']['eyes']['to_pulse_shape']['mae'],
        'scope':'Scoredprospectivepredictions, notdiscoverygates. RGBprojection/distance isnot a calibratedshape score; retainnativevisualcontrols.'}
    report={'schema':'mdb-step1-writers-audit-v1','job_id':summary['job_id'],'request_sha256':request['request_sha256'],'mechanics_status':'verified','instrument_status':'native-controls-plus-manual-region-observers','trend_status':'preserve-per-case','terminal_status':'unassessed',
        'custody':custody,'source_custody':source_custody,'transaction':tx,'numeric_seal':numeric,'counts':summary['counts'],'distinct_receipt_ids':len(by_receipt),'full_exit_equations':equations,'source_bound_text_validation_flags':{k:v['text_unchanged'] for k,v in validations.items()},
        'controls':controls,'observations':observations,'joint0_AMT_cube':cube,'conditional_write_effects':effects,'forecast_scores':forecast_scores,'worker_seconds':summary['worker_seconds'],
        'coverage':'All4completeimageexitfields independentlyrebuilt fromobservedR/A/M/gates;3fullnativefrozenwriteimagevalidations,2sourceexits,22nativeRGB; typedAct/closure/receipt/latentjoins. Nativevalidationtextflags source-bound, not independentfulltextreturnrebuild. No independentA/M formation orrawpromptRGB forecast.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        w,h=(320,350) if kind=='full' else (500,250);sheet=Image.new('RGB',(6*w,4*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);im=im.resize((320,320)) if kind=='full' else im.crop((130,180,380,290)).resize((500,220))
            x,y=(i%6)*w,(i//6)*h;sheet.paste(im,(x,y+26));draw.text((x+4,y+7),row['id'],fill='black')
        sheet.save(destination.parent/('step1-'+kind+'.png'))
    return {'job_id':summary['job_id'],'controls':controls,'forecast_scores':forecast_scores,
        'eye_projections':{k:v['eyes']['pulse_shape_contrast_projection'] for k,v in observations.items()},'numeric_fields_exact':{k:v['sealed_minus_independent']['bitwise_exact'] for k,v in equations.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve()),sort_keys=True))
