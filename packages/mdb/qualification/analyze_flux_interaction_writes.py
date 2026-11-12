"""Saved-only full-field math, typed causal edits and native image comparison."""
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
    here=Path(__file__).resolve().parent;data=run/'extracted/interaction-writes-result';old=source/'run-02/extracted/interaction-trace-result';custody_tool=runpy.run_path(str(here/'flux_family_custody.py'));custody=custody_tool['verify'](run);ancestor=custody_tool['verify'](source/'run-02')
    util=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=util['sha'],util['metric'];typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));q=typed['quantize_bf16'];route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'));json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree'];index=runpy.run_path(str(here/'analyze_flux_edit_onset.py'))['index_receipts'];public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    raw_error=runpy.run_path(str(here/'analyze_flux_mlp_replacement.py'))['field_error']
    def error(x,y):return {k:v for k,v in raw_error(x,y).items() if not isinstance(v,(list,dict))}
    config,summary,binding,forecasts,validations,rows,receipts=[read(data/n) for n in ('plan.json','summary.json','binding.json','forecasts.json','validations.json','rows.json','receipts.json')];request=read(run/'request.json')
    assert config==request['config'] and summary==read(run/'report.json')['report'] and summary['request_sha256']==request['request_sha256']==custody['request_sha256'];assert sha(source/'analysis-v1.json')==config['source_audit_sha256']
    for name,digest in request['source_hashes'].items():assert sha(run/'payload'/name)==digest
    for name,m in config['input_files'].items():assert sha(old/name)==m['sha256'] and (old/name).stat().st_size==m['bytes']
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    frozen=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/frozen_mixed_writes.json')['predictions']))
    assert frozen['fields']==forecasts and frozen['hypotheses']==config['hypotheses'] and frozen['binding_sha256']==sha(data/'binding.json')
    assert frozen['counts_at_seal']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0,candidate_blocks=0,cached_projections=0)
    expected=runpy.run_path(str(here/'flux_interaction_writes.py'))['census']({k:v['point'] for k,v in config['selected'].items()});assert summary['counts']==config['expected_counts']==expected
    for r in receipts:public(r)
    by=index(receipts);assert len(receipts)==expected['receipts'] and sum(r['operation']=='continue' for r in receipts)==expected['native_transitions'] and sum(r['operation']=='apply' for r in receipts)==32
    store=typed['audit_store'](data/'store')
    def slot(fp,s):return route['slot_array'](data/'store',fp,s,typed)
    def fields(fp):return {v['key']['value']:v['value'] for v in store['cuts'][fp]['payload']['items']}
    arr=lambda r:load_tensor_archive(data,r);equations={};nativeI={};predictedI={};text_linear={}
    for seed,b in binding.items():
        source_ports=config['selected'][seed]['ports'];cats={k:load_tensor_archive(old,p['concat']) for k,p in source_ports.items()};t,w=b['text_rows'],b['attention_width'];linear=q(cats['10']+cats['01']-cats['00']);weight=arr(b['weight']);bias=None if b['bias'] is None else arr(b['bias']);gate=arr(b['gate']);R=slot(b['before']['11'],'image');text=slot(b['after']['11'],'text')
        text_linear[seed]=q(slot(b['after']['10'],'text')+slot(b['after']['01'],'text')-slot(b['after']['00'],'text'));assert np.array_equal(text_linear[seed],arr(b['text_linear']))
        equations[seed]={};nativeI[seed]={};predictedI[seed]={}
        for case,descriptor in b['mixed_concats'].items():
            rebuilt=cats['11'].copy()
            if case in ('Alinear','AMlinear'):rebuilt[:,t:,:w]=linear[:,t:,:w]
            if case in ('Mlinear','AMlinear'):rebuilt[:,t:,w:]=linear[:,t:,w:]
            if case=='Aopposite':rebuilt[:,t:,:w]=q(2*cats['11'][:,t:,:w]-linear[:,t:,:w])
            cat=arr(descriptor);assert np.array_equal(cat,rebuilt);x=cat[:,t:].astype(np.float64);out=np.empty((*x.shape[:2],weight.shape[0]),np.float32)
            for start in range(0,weight.shape[0],192):
                stop=min(start+192,weight.shape[0]);v=x@weight[start:stop].astype(np.float64).T
                if bias is not None:v+=bias[start:stop].astype(np.float64)
                out[:,:,start:stop]=q(v.astype(np.float32))
            P=arr(forecasts[seed][case]['P']);I=arr(forecasts[seed][case]['I']);Iref=q((R.astype(np.float64)+q((gate.astype(np.float64)*P).astype(np.float32))).astype(np.float32));predictedI[seed][case]=I
            valid=validations[seed][case];nativeP=arr(valid['P']);nativeI[seed][case]=arr(valid['I']);native_exit=q((R.astype(np.float64)+q((gate.astype(np.float64)*nativeP).astype(np.float32))).astype(np.float32));actual=arr(valid['fullblock_image'])
            assert np.array_equal(native_exit,nativeI[seed][case]) and bool(np.array_equal(nativeI[seed][case],actual))==valid['native_cached_exact']
            equations[seed][case]={'P_independent':error(P,out),'I_independent':error(I,Iref),'P_minus_native':error(P,nativeP),'I_minus_native':error(I,actual),'native_cached_minus_fullblock':error(nativeI[seed][case],actual)}
        sham=validations[seed]['sham'];nativeI[seed]['sham']=arr(sham['I']);assert np.array_equal(nativeI[seed]['sham'],slot(b['after']['11'],'image')) and np.array_equal(nativeI[seed]['sham'],arr(sham['fullblock_image']))
    images={};latents={};old_rows={r['id']:r for r in read(old/'rows.json')};references={}
    for seed in binding:
        references[seed]={}
        for bits in ('00','11'):
            with Image.open(old/old_rows[seed+'.'+bits]['png']) as im:references[seed][bits]=np.asarray(im).copy()
    assert len(rows)==len({r['id'] for r in rows})==16
    for row in rows:
        seed,case=row['seed'],row['case'];b=binding[seed];assert row['parent']==b['after']['11'] and sha(data/row['png'])==row['png_sha256']
        with Image.open(data/row['png']) as im:assert im.size==(512,512) and im.mode=='RGB';images[row['id']]=np.asarray(im).copy()
        latents[row['id']]=arr(row['latent']);assert np.array_equal(latents[row['id']],slot(row['final_cut'],'latent'));previous=row['parent'];result_ids=set()
        for r in row['path']:
            assert r==by[r['fingerprint']] and r['parent']==previous;previous=r['result'];result_ids.add(previous)
        assert previous==row['final_cut'] and len(row['actions'])==2
        remain=(3-b['point']['logical_step'])*28+21-b['point']['index'];assert sum(r['operation']=='continue' for r in row['path'])==remain and len(row['path'])==remain+2
        expected_image=predictedI[seed]['AMlinear'] if case=='predictedAM' else nativeI[seed]['AMlinear'] if case=='AMTlinear' else nativeI[seed]['sham'] if case=='Tlinear' else nativeI[seed][case]
        expected_text=text_linear[seed] if case in ('Tlinear','AMTlinear') else slot(b['after']['11'],'text')
        for action in row['actions']:
            assert action['receipt']==by[action['receipt']['fingerprint']] and action['act']['writes']==[action['slot']] and action['post_act'] in result_ids
            before,after=fields(action['parent']),fields(action['post_act']);assert all(before[k]==after[k] for k in before if k!=action['slot'])
            assert np.array_equal(slot(action['post_act'],action['slot']),expected_image if action['slot']=='image' else expected_text)
        for record in row['trace']:assert record['cut'] in result_ids
    observations={};future_errors={};scores={};controls={}
    for name,image in images.items():
        seed=name.split('.')[0];observations[name]={}
        for label,box in {'whole':None,**config['observers']['windows']}.items():
            crop=lambda v:v if box is None else v[box[1]:box[3],box[0]:box[2]]
            v,low,high=crop(image),crop(references[seed]['00']),crop(references[seed]['11']);d=high.astype(np.float64)-low;den=float(np.sum(d*d))
            observations[name][label]={'to_weak':metric(v,low),'to_strong':metric(v,high),'source_contrast_projection':None if not den else float(np.sum((v.astype(np.float64)-low)*d)/den)}
    for seed in binding:
        alpha=lambda case:observations[seed+'.'+case]['eyes']['source_contrast_projection'];base=alpha('sham');negative,positive=alpha('Alinear')-base,alpha('Aopposite')-base
        future_errors[seed]={'rgb':metric(images[seed+'.predictedAM'],images[seed+'.AMlinear']),'latent':error(latents[seed+'.predictedAM'],latents[seed+'.AMlinear'])}
        scores[seed]={'Alinear_and_Aopposite_opposing_eye_shifts':negative*positive<0,'Alinear_eye_delta':negative,'Aopposite_eye_delta':positive,'Tlinear_shift_exceeds_AMlinear':abs(alpha('Tlinear')-base)>abs(alpha('AMlinear')-base),'predicted_AM_RGB_MAE_below1':future_errors[seed]['rgb']['mae']<1}
        row=next(r for r in rows if r['id']==seed+'.sham');assert row['png_control_exact']==(row['png_sha256']==row['expected_png']);controls[seed]=row['png_control_exact']
    report={'schema':'mdb-interaction-writes-audit-v1','job_id':summary['job_id'],'request_sha256':summary['request_sha256'],'mechanics_status':'verified','instrument_status':'native-shams/fullfields/manual-regions','trend_status':'signed-effects-and-misses-retained','terminal_status':'unassessed','custody':custody,'source_custody':ancestor,'transaction':tx,'counts':summary['counts'],'distinct_receipt_ids':len(by),'equations':equations,'controls':controls,'observations':observations,'predicted_future_errors':future_errors,'forecast_scores':scores,'source_bound_text_flags':{seed:{k:v['text_unchanged'] for k,v in values.items()} for seed,values in validations.items()},'worker_seconds':summary['worker_seconds'],
        'coverage':'8completeP+8Iindependentrebuilds;8mixedfullblockimagevalidations+2sourceshams;16nativeRGB. ExacttypedI/TActpayloads/untouchedclosure/receipt/latentjoins. SourceassistedfrozenA/M/Tcomposition; notformationforecast. Nativevalidationtextflags source-bound; textActs independentlychecked. Energy-selectedlatecandidates notminimal/necessarysemanticwriters.'}
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        w,h=(320,350) if kind=='full' else (500,250);sheet=Image.new('RGB',(4*w,4*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);im=im.resize((320,320)) if kind=='full' else im.crop((130,180,380,290)).resize((500,220));x,y=i%4*w,i//4*h;sheet.paste(im,(x,y+26));draw.text((x+4,y+7),row['id'],fill='black')
        sheet.save(destination.parent/('writes-'+kind+'.png'))
    return {'controls':controls,'forecast_scores':scores,'future_rgb_errors':{k:v['rgb'] for k,v in future_errors.items()},'eye_projections':{k:v['eyes']['source_contrast_projection'] for k,v in observations.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve()),sort_keys=True))
