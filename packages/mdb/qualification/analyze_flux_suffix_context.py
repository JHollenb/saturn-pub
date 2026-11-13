"""Saved-only audit of fixed image pairs, native last-block ports and Euler forecasts."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination,source,trace):
    from mdb.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/'extracted/suffix-context-result';old=source/'run-01/extracted/receiver-cross-result';td=trace/'run-02/extracted/interaction-trace-result';cust=runpy.run_path(str(here/'flux_family_custody.py'));custody=cust['verify'](run);ancestor=cust['verify'](source/'run-01')
    util=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=util['sha'],util['metric'];typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));q=typed['quantize_bf16'];route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'));tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree'];index=runpy.run_path(str(here/'analyze_flux_edit_onset.py'))['index_receipts'];public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt'];H=runpy.run_path(str(here/'flux_suffix_context.py'));raw_error=runpy.run_path(str(here/'analyze_flux_mlp_replacement.py'))['field_error']
    def error(x,y):return {k:v for k,v in raw_error(np.asarray(x,np.float32),np.asarray(y,np.float32)).items() if not isinstance(v,(list,dict))}
    config,summary,binding,ports,forecasts,rows,receipts=[read(data/n) for n in ('plan.json','summary.json','binding.json','ports.json','forecasts.json','rows.json','receipts.json')];request=read(run/'request.json');source_plan=read(old/'plan.json')
    assert config==request['config'] and summary==read(run/'report.json')['report'] and summary['request_sha256']==request['request_sha256']==custody['request_sha256'];assert sha(source/'analysis-v1.json')==config['source_audit_sha256'] and sha(trace/'analysis-v1.json')==source_plan['source_audit_sha256']
    for name,digest in request['source_hashes'].items():assert sha(run/'payload'/name)==digest
    for name,m in config['input_files'].items():assert sha(old/name)==m['sha256'] and (old/name).stat().st_size==m['bytes']
    def frozen(path,name):return tree(load_forecast_tree(path,read(path/'forecasts'/(name+'.json'))['predictions']))
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);f=frozen(data/'transaction','frozen_suffix_hypotheses');assert tx['verified'] and tx['phase']=='closed' and f['hypotheses']==config['hypotheses'] and f['binding_sha256']==sha(data/'binding.json')
    assert f['counts_at_seal']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0,candidate_blocks=0,euler_forecasts=0)
    assert summary['counts']==config['expected_counts']==H['census']()
    for r in receipts:public(r)
    by=index(receipts);assert len(receipts)==80 and sum(r['operation']=='continue' for r in receipts)==48 and sum(r['operation']=='apply' for r in receipts)==32
    store=typed['audit_store'](data/'store')
    def slot(fp,s):return route['slot_array'](data/'store',fp,s,typed)
    def fields(fp):return {v['key']['value']:v['value'] for v in store['cuts'][fp]['payload']['items']}
    arr=lambda d:load_tensor_archive(data,d);images={};controls={};eulers={};equations={};postI={};postT={};velocities={};latents={};preI={};preT={};rowmap={r['id']:r for r in rows};seeds=list(config['parents']);first=fields(config['parents'][seeds[0]])
    for seed,fp in config['parents'].items():assert all(first[k]==fields(fp)[k] for k in first if k not in {'image','text','latent'})
    assert [{k:r[k] for k in ('id','pair','bit','text','latent')} for r in rows]==H['cases']()
    for row in rows:
        name=row['id'];parent=config['parents'][row['latent']];assert row['parent']==parent;previous=parent;result_ids=[]
        for r in row['path']:
            assert r==by[r['fingerprint']] and r['parent']==previous;previous=r['result'];result_ids.append(previous)
        assert len(row['actions'])==2 and len(row['path'])==5 and previous==row['final_cut'] and row['pre_cut']==result_ids[1] and row['post19_cut']==result_ids[2] and row['scheduler_cut']==result_ids[3]
        expected={'image':slot(config['pairs'][row['pair']][str(row['bit'])],'image'),'text':slot(config['parents'][row['text']],'text')}
        for action in row['actions']:
            assert action['receipt']==by[action['receipt']['fingerprint']] and action['act']['writes']==[action['slot']] and action['post_act'] in result_ids
            before,after=fields(action['parent']),fields(action['post_act']);assert all(before[k]==after[k] for k in before if k!=action['slot']);assert np.array_equal(slot(action['post_act'],action['slot']),expected[action['slot']])
        preI[name]=slot(row['pre_cut'],'image');preT[name]=slot(row['pre_cut'],'text');postI[name]=slot(row['post19_cut'],'image');postT[name]=slot(row['post19_cut'],'text');velocities[name]=slot(row['scheduler_cut'],'noise_prediction');latents[name]=slot(row['final_cut'],'latent')
        assert np.array_equal(preI[name],expected['image']) and np.array_equal(preT[name],expected['text'])
        for fp in (row['pre_cut'],row['post19_cut'],row['scheduler_cut']):assert np.array_equal(slot(fp,'latent'),slot(parent,'latent'))
        port=ports[row['port_key']];P,g=arr(port['P']),arr(port['gate']);ref=q((preI[name].astype(np.float64)+q((g.astype(np.float64)*P).astype(np.float32))).astype(np.float32));assert np.array_equal(ref,postI[name]);equations[name]={'post19_gated_exit_independently_exact':True,'candidate_parity_source_bound':row['candidate_native_parity']}
        p=data/'euler-transactions'/name;et=verify_transaction(p,expected_seal_sha256=row['euler_seal']['seal_sha256']);ef=frozen(p,'next_latent');assert et['verified'] and et['phase']=='closed' and ef==forecasts[name] and ef['scheduler_cut']==row['scheduler_cut'];assert ef['counts_at_seal']['native_transitions']==3*rows.index(row)+2 and ef['counts_at_seal']['images']==rows.index(row)
        sigma=slot(row['scheduler_cut'],'sigmas');dt=float(np.float32(sigma[4]-sigma[3]));assert dt==ef['sigma_delta'];forecast=arr(ef['latent']);reference=q(slot(parent,'latent')+q(q(np.float32(dt))*velocities[name]));assert np.array_equal(forecast,reference);eulers[name]={'transaction':et,'independent_rebuild':error(forecast,reference),'minus_native':error(forecast,latents[name])}
        assert sha(data/row['png'])==row['png_sha256']
        with Image.open(data/row['png']) as im:assert im.mode=='RGB' and im.size==(512,512);images[name]=np.asarray(im).copy()
        if 'expected_png' in row:assert row['png_control_exact']==(row['png_sha256']==row['expected_png']);controls[name]=row['png_control_exact']
    mechanics_predictions={};mlp_predictions={};port_changes={}
    for pair in seeds:
        for bit in (0,1):
            a,b=[ports[f'{pair}.I{bit}.T{t}'] for t in seeds];key=f'{pair}.I{bit}';M0,M1=arr(a['M']),arr(b['M']);A0,A1=arr(a['A']),arr(b['A']);g0,g1=arr(a['gate']),arr(b['gate'])
            mlp_predictions[key]={'M_text_independent':bool(np.array_equal(M0,M1)),'gate_text_independent':bool(np.array_equal(g0,g1))};port_changes[key]={'M':error(M1,M0),'A':error(A1,A0),'P':error(arr(b['P']),arr(a['P']))}
            for text in seeds:
                x,y=[f'{pair}.I{bit}.T{text}.Z{z}' for z in seeds];mechanics_predictions[f'{pair}.I{bit}.T{text}']={'post19_I_Z_independent':bool(np.array_equal(postI[x],postI[y])),'post19_T_Z_independent':bool(np.array_equal(postT[x],postT[y])),'readout_noise_Z_independent':bool(np.array_equal(velocities[x],velocities[y]))}
    old_rows={r['id']:r for r in read(td/'rows.json')};references={};axis_sources={}
    for seed in seeds:
        references[seed]={};axis_sources[seed]={}
        for bits in ('00','11'):
            row=old_rows[seed+'.'+bits];assert sha(td/row['png'])==row['png_sha256'];axis_sources[seed][bits]=row['png_sha256']
            with Image.open(td/row['png']) as im:references[seed][bits]=np.asarray(im).copy()
    paired={};scores={};individual={}
    for pair in seeds:
        for text in seeds:
            for latent in seeds:
                key=f'{pair}.T{text}.Z{latent}';lo,hi=[f'{pair}.I{b}.T{text}.Z{latent}' for b in (0,1)];paired[key]={'pair':pair,'text':text,'latent':latent,'previously_seen':pair==text==latent,'rows':[lo,hi],'regions':{},'state_changes':{}}
                for label,box in {'whole':None,**config['observers']['windows']}.items():
                    crop=lambda v:v if box is None else v[box[1]:box[3],box[0]:box[2]]
                    delta=crop(images[hi]).astype(np.float64)-crop(images[lo]);axes={}
                    for axis in seeds:
                        d=crop(references[axis]['11']).astype(np.float64)-crop(references[axis]['00']);den=float(np.sum(d*d));axes[axis]=None if not den else float(np.sum(delta*d)/den)
                    paired[key]['regions'][label]={'corrected_minus_base':metric(crop(images[hi]),crop(images[lo])),'fixed_axes':axes}
                for label,values in [('input_image',preI),('post19_image',postI),('readout_noise',velocities),('committed_latent',latents)]:paired[key]['state_changes'][label]=error(values[hi],values[lo])
                effect=paired[key]['regions']['eyes']['fixed_axes']['60667'];pred={'image_pair_led':1 if pair=='60666' else -1,'text_led':1 if text=='60666' else -1,'latent_led':1 if latent=='60666' else -1};scores[key]={'effect':effect,'previously_seen':pair==text==latent,'predicted_signs':pred,'matched':{k:bool(effect*v>0) for k,v in pred.items()}}
    for name,im in images.items():
        individual[name]={}
        for label,box in {'whole':None,**config['observers']['windows']}.items():
            crop=lambda v:v if box is None else v[box[1]:box[3],box[0]:box[2]]
            individual[name][label]=metric(crop(im),crop(references[rowmap[name]['pair']]['11']))
    interactions={}
    for pair in seeds:
        def delta(values,text,latent):return values[f'{pair}.I1.T{text}.Z{latent}'].astype(np.float64)-values[f'{pair}.I0.T{text}.Z{latent}']
        interactions[pair]={}
        for label,values in [('post19_image',postI),('readout_noise',velocities),('committed_latent',latents),('rgb',images)]:
            interactions[pair][label]={'T_effect_on_pair_delta_at_Z60666':error(delta(values,'60667','60666'),delta(values,'60666','60666')),'Z_effect_on_pair_delta_at_T60666':error(delta(values,'60666','60667'),delta(values,'60666','60666'))}
    report={'schema':'mdb-suffix-context-audit-v1','job_id':summary['job_id'],'request_sha256':summary['request_sha256'],'mechanics_status':'verified','instrument_status':'full-tail-cuts/native-ports/fixed-RGB-axes','trend_status':'factorial-effects-and-counterexamples-retained','terminal_status':'unassessed','custody':custody,'source_custody':ancestor,'transaction':tx,'counts':summary['counts'],'distinct_receipt_ids':len(by),'controls':controls,'post19_equations':equations,'euler_forecasts':eulers,'mechanical_prediction_scores':mechanics_predictions,'parallel_M_prediction_scores':mlp_predictions,'T_port_changes':port_changes,'paired_observations':paired,'signed_forecast_scores':scores,'observer_axis_png_sha256':axis_sources,'individual_to_pair_source_strong':individual,'pair_delta_interactions':interactions,'worker_seconds':summary['worker_seconds'],
        'coverage':'16nativefutures andfullpre19/post19/scheduler/finalcuts. Completeimagepairidentical acrossT/Z viaexacttypedI/TActs onnativeZparent.8separatenativeportcaptures; A/M/P/gate imagefields full.16localEulerforecasts after observedreadout before owncommit independently rebuilt. GlobalM/Z/signhypotheses frozenbeforetail.8pairedcontrasts2previouslyseen6new. RGBaxes supplied, notgeometry; no upstreamformation or fullpromptimageforecast.'}
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        w,h=(256,290) if kind=='full' else (400,210);sheet=Image.new('RGB',(4*w,4*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);im=im.resize((256,256)) if kind=='full' else im.crop((130,180,380,290)).resize((400,176));x,y=i%4*w,i//4*h;sheet.paste(im,(x,y+30));draw.text((x+4,y+6),row['id'],fill='black')
        sheet.save(destination.parent/('suffix-'+kind+'.png'))
    return {'controls':controls,'parallel_M':mlp_predictions,'Z_independence':mechanics_predictions,'signed_forecasts':scores,'euler_native_exact':sum(v['minus_native']['bitwise_exact'] for v in eulers.values()),'eye_effects':{k:v['regions']['eyes']['fixed_axes'] for k,v in paired.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve(),a.trace.resolve()),sort_keys=True))
