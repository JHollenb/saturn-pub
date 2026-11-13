"""Independent saved-only audit of fixed correction × receiver experiments."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination,source,writers):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/'extracted/receiver-cross-result';old=source/'run-02/extracted/interaction-trace-result';cust=runpy.run_path(str(here/'flux_family_custody.py'));custody=cust['verify'](run);ancestor=cust['verify'](source/'run-02');writer_custody=cust['verify'](writers/'run-01')
    util=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=util['sha'],util['metric'];typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));q=typed['quantize_bf16'];route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'));json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree'];index=runpy.run_path(str(here/'analyze_flux_edit_onset.py'))['index_receipts'];public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt'];H=runpy.run_path(str(here/'flux_receiver_cross.py'))
    raw_error=runpy.run_path(str(here/'analyze_flux_mlp_replacement.py'))['field_error']
    def error(x,y):return {k:v for k,v in raw_error(x,y).items() if not isinstance(v,(list,dict))}
    config,summary,binding,forecasts,validations,rows,receipts=[read(data/n) for n in ('plan.json','summary.json','binding.json','forecasts.json','validations.json','rows.json','receipts.json')];request=read(run/'request.json')
    assert config==request['config'] and summary==read(run/'report.json')['report'] and summary['request_sha256']==request['request_sha256']==custody['request_sha256']
    assert sha(source/'analysis-v1.json')==config['source_audit_sha256'] and sha(writers/'analysis-v1.json')==config['writer_audit_sha256']
    for name,digest in request['source_hashes'].items():assert sha(run/'payload'/name)==digest
    for name,m in config['input_files'].items():assert sha(old/name)==m['sha256'] and (old/name).stat().st_size==m['bytes']
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    frozen=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/fixed_correction_cross.json')['predictions']))
    assert frozen['fields']==forecasts and frozen['hypotheses']==config['hypotheses'] and frozen['binding_sha256']==sha(data/'binding.json')
    assert frozen['counts_at_seal']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0,candidate_blocks=0,cached_projections=0)
    expected=H['census'](config['selected']);assert summary['counts']==config['expected_counts']==expected
    for r in receipts:public(r)
    by=index(receipts);assert len(receipts)==100 and sum(r['operation']=='continue' for r in receipts)==60 and sum(r['operation']=='apply' for r in receipts)==40
    store=typed['audit_store'](data/'store')
    def slot(fp,s):return route['slot_array'](data/'store',fp,s,typed)
    def fields(fp):return {v['key']['value']:v['value'] for v in store['cuts'][fp]['payload']['items']}
    arr=lambda r:load_tensor_archive(data,r);deltas={};nativecats={};equations={};rounding={};source_checks={}
    for seed,b in binding.items():
        cats={k:load_tensor_archive(old,p['concat']) for k,p in config['selected'][seed]['ports'].items()};t,w=b['text_rows'],b['attention_width'];delta=q(cats['10'][:,t:,:w]+cats['01'][:,t:,:w]-cats['00'][:,t:,:w])-cats['11'][:,t:,:w]
        assert np.array_equal(delta,arr(b['delta'])) and np.array_equal(cats['11'],arr(b['native_concat']))
        deltas[seed]=delta;nativecats[seed]=cats['11'];source_checks[seed]={'delta_independently_rebuilt':True,'native_concat_source_exact':True}
    seeds=list(binding);a,b=seeds;cross_operands={'delta_cosine':float(np.sum(deltas[a].astype(np.float64)*deltas[b])/np.linalg.norm(deltas[a].astype(np.float64))/np.linalg.norm(deltas[b].astype(np.float64))),'gate_exact':bool(np.array_equal(arr(binding[a]['gate']),arr(binding[b]['gate']))),'weight_exact':bool(np.array_equal(arr(binding[a]['weight']),arr(binding[b]['weight'])))}
    for seed,b in binding.items():
        t,w=b['text_rows'],b['attention_width'];weight=arr(b['weight']);bias=None if b['bias'] is None else arr(b['bias']);gate=arr(b['gate']);R=slot(b['before'],'image');native=nativecats[seed]
        for case in H['cases'](seeds):
            if case['receiver']!=seed or case['donor'] is None or case['predicted']:continue
            name=case['id'];donor=case['donor'];dose=case['dose'];cat=arr(forecasts[name]['concat']);rebuilt=native.copy();rebuilt[:,t:,:w]=q(native[:,t:,:w]+np.float32(dose)*deltas[donor]);assert np.array_equal(cat,rebuilt)
            effective=cat[:,t:,:w]-native[:,t:,:w];rounding[name]=error(effective,np.float32(dose)*deltas[donor]);x=cat[:,t:].astype(np.float64);out=np.empty((*x.shape[:2],weight.shape[0]),np.float32)
            for start in range(0,weight.shape[0],192):
                stop=min(start+192,weight.shape[0]);v=x@weight[start:stop].astype(np.float64).T
                if bias is not None:v+=bias[start:stop].astype(np.float64)
                out[:,:,start:stop]=q(v.astype(np.float32))
            P=arr(forecasts[name]['P']);I=arr(forecasts[name]['I']);Iref=q((R.astype(np.float64)+q((gate.astype(np.float64)*P).astype(np.float32))).astype(np.float32));valid=validations[name];nP=arr(valid['P']);nI=arr(valid['I']);actual=arr(valid['fullblock_image']);nref=q((R.astype(np.float64)+q((gate.astype(np.float64)*nP).astype(np.float32))).astype(np.float32))
            assert np.array_equal(P,out) and np.array_equal(I,Iref) and np.array_equal(nI,nref)
            assert bool(np.array_equal(nI,actual))==valid['native_cached_exact'] and valid['native_cached_exact']
            equations[name]={'P_independent':error(P,out),'I_independent':error(I,Iref),'P_minus_native':error(P,nP),'I_minus_native':error(I,actual),'native_cached_exact':True,'source_bound_text_unchanged':valid['text_unchanged']}
        sham=validations[seed+'.sham'];assert np.array_equal(arr(sham['I']),slot(b['after'],'image')) and np.array_equal(arr(sham['I']),arr(sham['fullblock_image']))
    images={};latents={};old_rows={r['id']:r for r in read(old/'rows.json')};references={};controls={}
    for seed in binding:
        references[seed]={}
        for bits in ('00','11'):
            with Image.open(old/old_rows[seed+'.'+bits]['png']) as im:references[seed][bits]=np.asarray(im).copy()
    assert [{k:r[k] for k in ('id','receiver','donor','dose','predicted')} for r in rows]==H['cases'](seeds)
    for row in rows:
        name,seed=row['id'],row['receiver'];b=binding[seed];assert row['parent']==b['after'] and sha(data/row['png'])==row['png_sha256']
        with Image.open(data/row['png']) as im:assert im.size==(512,512) and im.mode=='RGB';images[name]=np.asarray(im).copy()
        latents[name]=arr(row['latent']);assert np.array_equal(latents[name],slot(row['final_cut'],'latent'));previous=row['parent'];result_ids=set()
        for r in row['path']:
            assert r==by[r['fingerprint']] and r['parent']==previous;previous=r['result'];result_ids.add(previous)
        assert previous==row['final_cut'] and len(row['actions'])==2 and len(row['path'])==5 and sum(r['operation']=='continue' for r in row['path'])==3
        key=row['validation_key'];I=arr(forecasts[key]['I'] if row['predicted'] else validations[key]['I']);T=slot(b['after'],'text')
        for action in row['actions']:
            assert action['receipt']==by[action['receipt']['fingerprint']] and action['act']['writes']==[action['slot']] and action['post_act'] in result_ids
            before,after=fields(action['parent']),fields(action['post_act']);assert all(before[k]==after[k] for k in before if k!=action['slot'])
            assert np.array_equal(slot(action['post_act'],action['slot']),I if action['slot']=='image' else T)
        if row['donor'] is None:
            assert row['png_control_exact']==(row['png_sha256']==row['expected_png']);controls[name]=row['png_control_exact']
        if row['receiver']==row['donor'] and abs(row['dose'])==1:
            historical=config['historical_pngs'][seed]['Alinear' if row['dose']==1 else 'Aopposite'];assert row['historical_png_exact']==(row['png_sha256']==historical);controls[name]=row['historical_png_exact']
    observations={};common_axes={};forecasts_scored={};dose_scores={};future_errors={};write_alignment={};response_curvature={}
    for row in rows:
        name,seed=row['id'],row['receiver'];image=images[name];observations[name]={}
        for label,box in {'whole':None,**config['observers']['windows']}.items():
            crop=lambda v:v if box is None else v[box[1]:box[3],box[0]:box[2]]
            v,lo,hi=crop(image),crop(references[seed]['00']),crop(references[seed]['11']);d=hi.astype(np.float64)-lo;den=float(np.sum(d*d))
            observations[name][label]={'to_weak':metric(v,lo),'to_strong':metric(v,hi),'source_contrast_projection':None if not den else float(np.sum((v.astype(np.float64)-lo)*d)/den)}
        # Exploratory common-axis diagnostic: all receivers against each saved RGB axis.
        x0,y0,x1,y1=config['observers']['windows']['eyes'];change=(image.astype(np.float64)-images[seed+'.sham'])[y0:y1,x0:x1];common_axes[name]={}
        for axis in seeds:
            d=(references[axis]['11'].astype(np.float64)-references[axis]['00'])[y0:y1,x0:x1];common_axes[name][axis]=float(np.sum(change*d)/np.sum(d*d))
    for row in rows:
        if row['donor'] is None or row['predicted']:continue
        name,receiver,donor,dose=row['id'],row['receiver'],row['donor'],row['dose'];effect=observations[name]['eyes']['source_contrast_projection']-1
        payload_sign=(1 if donor=='60666' else -1)*np.sign(dose);receiver_sign=(1 if receiver=='60666' else -1)*np.sign(dose)
        forecasts_scored[name]={'eye_projection_shift':effect,'payload_led_sign':int(payload_sign),'receiver_led_sign':int(receiver_sign),'payload_led_matched':bool(effect*payload_sign>0),'receiver_led_matched':bool(effect*receiver_sign>0),'new_cross':receiver!=donor,'historical_endpoint':receiver==donor and abs(dose)==1}
        if abs(dose)==.5:
            full=f'{receiver}.from{donor}.{2*dose:+g}';end=observations[full]['eyes']['source_contrast_projection']-1;dose_scores[name]={'half_shift':effect,'full_shift':end,'between_zero_and_full':bool(min(0,end)<=effect<=max(0,end))}
    for seed in seeds:
        donor=next(s for s in seeds if s!=seed);native=f'{seed}.from{donor}.+1';predicted=seed+'.predictedCross';future_errors[seed]={'rgb':metric(images[predicted],images[native]),'latent':error(latents[predicted],latents[native]),'forecast_RGB_MAE_below1':metric(images[predicted],images[native])['mae']<1}
    # Saved-only diagnostics; no fitted prediction or additional native execution.
    for donor in seeds:
        for dose in H['DOSES']:
            changes=[arr(validations[f'{receiver}.from{donor}.{dose:+g}']['I'])-arr(validations[receiver+'.sham']['I']) for receiver in seeds]
            write_alignment[f'{donor}.{dose:+g}']=error(changes[0],changes[1])
        for receiver in seeds:
            for dose in (.5,1.):
                plus=f'{receiver}.from{donor}.{dose:+g}';minus=f'{receiver}.from{donor}.{-dose:+g}';base=receiver+'.sham'
                response_curvature[f'{receiver}.from{donor}.{dose:g}']={}
                for label,values in [('exit',{k:arr(v['I']) for k,v in validations.items()}),('latent',latents),('rgb',images)]:
                    y=values[plus].astype(np.float64)+values[minus]-2*values[base].astype(np.float64)
                    response_curvature[f'{receiver}.from{donor}.{dose:g}'][label]={'l2':float(np.linalg.norm(y)),'mae':float(np.mean(np.abs(y))),'max_abs':float(np.max(np.abs(y)))}
    report={'schema':'mdb-receiver-cross-audit-v1','job_id':summary['job_id'],'request_sha256':summary['request_sha256'],'mechanics_status':'verified','instrument_status':'native-controls/fullfields/manual-RGB-regions','trend_status':'effects-and-counterexamples-retained','terminal_status':'unassessed','custody':custody,'source_custody':ancestor,'writer_custody':writer_custody,'transaction':tx,'counts':summary['counts'],'distinct_receipt_ids':len(by),'source_checks':source_checks,'cross_operands':cross_operands,'effective_delta_rounding':rounding,'equations':equations,'controls':controls,'observations':observations,'forecast_scores':forecasts_scored,'dose_scores':dose_scores,'predicted_future_errors':future_errors,'exploratory_common_eye_axes':common_axes,'exploratory_exit_write_alignment':write_alignment,'exploratory_signed_response_curvature':response_curvature,'worker_seconds':summary['worker_seconds'],
        'coverage':'16 complete P/I numerical predictions independently rebuilt;16mixed native fullblock image validations,2shams,20typednativefutures. FixedFP32attentiondelta before recipientBF16 composition. Ownendpoints previouslyseen;8cross signed/dose cases new. Native outgoingT source-bound flags; typedTAct independently checked. Receiver includes spatial alignment and observer axis; no universalsemantic editclaim.'}
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        w,h=(256,290) if kind=='full' else (400,210);sheet=Image.new('RGB',(5*w,4*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);im=im.resize((256,256)) if kind=='full' else im.crop((130,180,380,290)).resize((400,176));x,y=i%5*w,i//5*h;sheet.paste(im,(x,y+30));draw.text((x+4,y+6),row['id'],fill='black')
        sheet.save(destination.parent/('cross-'+kind+'.png'))
    return {'controls':controls,'cross_operands':cross_operands,'new_cross_scores':{k:v for k,v in forecasts_scored.items() if v['new_cross']},'dose_scores':dose_scores,'future_rgb_errors':{k:v['rgb'] for k,v in future_errors.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--writers',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve(),a.writers.resolve()),sort_keys=True))
