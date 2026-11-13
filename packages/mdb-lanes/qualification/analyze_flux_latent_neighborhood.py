"""Audit predicted-terminal neighborhoods and their native Euler checks offline."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination,source,trace):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;d=run/'extracted/latent-neighborhood-result';old=source/'run-01/extracted/suffix-context-result';td=trace/'run-02/extracted/interaction-trace-result';cust=runpy.run_path(str(here/'flux_family_custody.py'));custody=cust['verify'](run);ancestor=cust['verify'](source/'run-01');typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));q=typed['quantize_bf16'];route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'));u=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=u['sha'],u['metric'];tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree'];raw_error=runpy.run_path(str(here/'analyze_flux_mlp_replacement.py'))['field_error']
    def error(x,y):return {k:v for k,v in raw_error(np.asarray(x,np.float32),np.asarray(y,np.float32)).items() if not isinstance(v,(list,dict))}
    plan,request,summary,forecasts,rows,receipts=[read(p) for p in (d/'plan.json',run/'request.json',d/'summary.json',d/'forecasts.json',d/'rows.json',d/'receipts.json')];assert plan==request['config'] and summary==read(run/'report.json')['report'] and summary['request_sha256']==request['request_sha256']==custody['request_sha256'];assert sha(source/'analysis-v1.json')==plan['source_audit_sha256']
    for n,h in request['source_hashes'].items():assert sha(run/'payload'/n)==h
    for n,m in plan['input_files'].items():assert sha(old/n)==m['sha256'] and (old/n).stat().st_size==m['bytes']
    tx=verify_transaction(d/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);f=tree(load_forecast_tree(d/'transaction',read(d/'transaction/forecasts/latent_neighborhood.json')['predictions']));assert tx['verified'] and tx['phase']=='closed' and f['fields']==forecasts and f['hypotheses']==plan['hypotheses'] and f['counts_at_seal']['euler_forecasts']==24
    for k in ('images','native_transitions','acts','receipts','transformer_forwards','readouts','derived_scheduler_initializations','derived_terminal_initializations'):assert f['counts_at_seal'][k]==0
    expected=dict(loads=1,encodes=0,noise_draws=0,transformer_forwards=0,readouts=0,native_transitions=4,acts=0,receipts=4,images=24,euler_forecasts=24,derived_scheduler_initializations=4,derived_terminal_initializations=24);assert summary['counts']==expected
    store=typed['audit_store'](d/'store');slot=lambda fp,s:route['slot_array'](d/'store',fp,s,typed);arr=lambda x:load_tensor_archive(d,x)
    def fields(fp):return {v['key']['value']:v['value'] for v in store['cuts'][fp]['payload']['items']}
    images={};math={};controls={};validations={};by={r['fingerprint']:r for r in receipts};assert len(receipts)==len(by)==4 and len(rows)==len(forecasts)==24
    for row in rows:
        name=row['id'];v=forecasts[name];b=plan['sources'][v['source']];alpha=v['alpha'];scheduler=b['scheduler'];z0,z1=slot(scheduler,'latent'),slot(b['other_parent'],'latent');input=z0.copy() if alpha==0 else z1.copy() if alpha==1 else q(z0+np.float32(alpha)*(z1-z0));assert np.array_equal(input,arr(v['inputZ3']));sigma=slot(scheduler,'sigmas');dt=float(np.float32(sigma[4]-sigma[3]));assert dt==v['sigma_delta'];reference=q(input+q(q(np.float32(dt))*slot(scheduler,'noise_prediction')));pred=arr(v['predictedZ4']);assert np.array_equal(pred,reference) and np.array_equal(pred,slot(row['terminal_cut'],'latent'))
        math[name]={'input_independent_exact':True,'Euler_independent':error(pred,reference)};base,terminal=fields(b['reference_final']),fields(row['terminal_cut']);assert all(base[k]==terminal[k] for k in base if k!='latent')
        assert row['scheduler_source']==scheduler and row['lineage_kind']=='derived-terminal-initialization-from-sealed-Euler-prediction; notnativeforwardorAct'
        if row['native_validation'] is not None:
            n=row['native_validation'];r=n['receipt'];assert alpha==.1 and r==by[r['fingerprint']] and r['operation']=='continue' and r['parent']==n['derived_initial_cut'] and r['result']==n['native_final_cut'];a,c=fields(scheduler),fields(n['derived_initial_cut']);assert all(a[k]==c[k] for k in a if k!='latent');assert np.array_equal(slot(n['derived_initial_cut'],'latent'),input)
            actual=slot(n['native_final_cut'],'latent');assert n['predicted_exact']==bool(np.array_equal(pred,actual));validations[name]={'predicted_minus_native':error(pred,actual),'receipt':r['fingerprint']}
        assert sha(d/row['png'])==row['png_sha256']
        with Image.open(d/row['png']) as im:assert im.mode=='RGB' and im.size==(512,512);images[name]=np.asarray(im).copy()
        if 'expected_png' in row:assert row['png_control_exact']==(row['png_sha256']==row['expected_png']);controls[name]=row['png_control_exact']
    native_rows={r['id']:r for r in read(td/'rows.json')};axes={}
    source_a=read(source/'analysis-v1.json')
    for seed in ('60666','60667'):
        refs={}
        for bit in ('00','11'):
            r=native_rows[seed+'.'+bit];assert sha(td/r['png'])==r['png_sha256']==source_a['observer_axis_png_sha256'][seed][bit]
            with Image.open(td/r['png']) as im:refs[bit]=np.asarray(im).copy()
        axes[seed]=refs['11'].astype(np.float64)-refs['00']
    paired={};scores={};perturbations={}
    for pair in ('60666','60667'):
        for alpha in plan['alphas']:
            names=[f'{pair}.I{bit}.a{alpha:g}' for bit in (0,1)];name=f'{pair}.a{alpha:g}';paired[name]={'pair':pair,'alpha':alpha,'rows':names,'regions':{},'baseline_drift':{},'prediction':forecasts[names[0]]['predicted_pair_effects']}
            for label,box in {'whole':None,**plan['observers']['windows']}.items():
                crop=lambda x:x if box is None else x[box[1]:box[3],box[0]:box[2]];delta=crop(images[names[1]]).astype(np.float64)-crop(images[names[0]]);obs={}
                for seed,axis in axes.items():
                    a=crop(axis);den=float(np.sum(a*a));obs[seed]=None if not den else float(np.sum(delta*a)/den)
                paired[name]['regions'][label]={'corrected_minus_base':metric(crop(images[names[1]]),crop(images[names[0]])),'fixed_axes':obs};paired[name]['baseline_drift'][label]=metric(crop(images[names[0]]),crop(images[pair+'.I0.a0']))
            observed=paired[name]['regions']['eyes']['fixed_axes']['60667'];pred=paired[name]['prediction']['60667'];scores[name]={'previously_seen':alpha in (0.,1.),'observed_effect':observed,'predicted_effect':pred,'signed_error':observed-pred,'sign_matched':bool(observed*pred>0)}
            x0,x1=[arr(forecasts[n]['inputZ3']) for n in names];assert np.array_equal(x0,x1);native_input=slot(plan['sources'][pair+'.I0']['scheduler'],'latent');perturbations[name]=error(x0,native_input)
    out={'schema':'mdb-latent-neighborhood-audit-v1','job_id':summary['job_id'],'request_sha256':summary['request_sha256'],'mechanics_status':'verified','instrument_status':'sealedEuler/nativechecks/VAE/fixedRGBaxes/manual-quality-pending','trend_status':'small-dose-effects-and-forecast-errors-retained','terminal_status':'unassessed','custody':custody,'source_custody':ancestor,'transaction':tx,'counts':expected,'math_forecasts':math,'native_Euler_checks':validations,'controls':controls,'paired_observations':paired,'signed_forecast_scores':scores,'receiving_latent_perturbations':perturbations,'worker_seconds':summary['worker_seconds'],'coverage':'24completefinalZpredictions fromobservednativevelocity beforeallnewexecution, independentlyrebuilt;4nativeEulerchecks onexplicitderivedschedulerinputs;24VAEdecodes ofdeclaredderivedterminalstates. Zero new transformer/project/readout calls instrumented; no lateZActs or historicalrewindclaim.8oldendpointPNGs controls;8newintermediatepairedforecasts. Numericalfidelity separatefromlinearRGBinterpolation andsemanticquality.'}
    with destination.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        w,h=(256,290) if kind=='full' else (360,190);sheet=Image.new('RGB',(6*w,4*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);im=im.resize((256,256)) if kind=='full' else im.crop((130,180,380,290)).resize((360,158));x,y=i%6*w,i//6*h;sheet.paste(im,(x,y+28));draw.text((x+4,y+6),row['id'],fill='black')
        sheet.save(destination.parent/('neighborhood-'+kind+'.png'))
    return {'controls':controls,'native_Euler_exact':{k:v['predicted_minus_native']['bitwise_exact'] for k,v in validations.items()},'signed_forecasts':scores,'baseline_whole_MAE':{k:v['baseline_drift']['whole']['mae'] for k,v in paired.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve(),a.trace.resolve()),sort_keys=True))
