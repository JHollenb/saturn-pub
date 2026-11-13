"""Saved-only full-field, typed-continuation and RAMT factorial audit."""
import argparse
import itertools
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination,source,onset):
    from mdb.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/'extracted/step1-overlap-result';old=source/'run-01/extracted/step1-writers-result'
    custody_tool=runpy.run_path(str(here/'flux_family_custody.py'));custody=custody_tool['verify'](run)
    ancestors={str(p):custody_tool['verify'](p/'run-01') for p in (source,onset)}
    util=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=util['sha'],util['metric']
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));q=typed['quantize_bf16'];route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'))
    error=runpy.run_path(str(here/'analyze_flux_mlp_replacement.py'))['field_error'];json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree']
    index=runpy.run_path(str(here/'analyze_flux_edit_onset.py'))['index_receipts'];public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    config,summary,binding,rows,receipts=[read(data/n) for n in ('plan.json','summary.json','binding.json','rows.json','receipts.json')]
    request=read(run/'request.json');assert request['config']==config and summary==read(run/'report.json')['report']
    assert summary['request_sha256']==request['request_sha256']==custody['request_sha256']
    assert sha(source/'analysis-v1.json')==config['source_audit_sha256'] and sha(onset/'analysis-v1.json')==config['onset_audit_sha256']
    for name,digest in request['source_hashes'].items():assert sha(run/'payload'/name)==digest
    for name,m in config['input_files'].items():assert sha(old/name)==m['sha256'] and (old/name).stat().st_size==m['bytes']
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    forecast=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/common_future_RAMT.json')['predictions']))
    assert forecast==config['hypotheses']
    timeline=read(data/'timeline.json');assert next(v for v in timeline if v['kind']=='sealed')['counts']['native_transitions']==0
    numeric=read(data/'numeric-seal.json');assert sha(data/'binding.json')==numeric['binding_sha256'] and sha(data/'numeric-seal.json')==summary['numeric_seal_sha256']
    assert numeric['counts_at_seal']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=14,acts=0,receipts=14,images=0,candidate_blocks=2,cached_projections=0)
    assert next(v for v in timeline if v['kind']=='numeric_sealed')['sha256']==summary['numeric_seal_sha256']
    store=typed['audit_store'](data/'store');arr=lambda v:load_tensor_archive(data,v)
    def slot(cut,name):return route['slot_array'](data/'store',cut,name,typed)
    def fields(cut):return {v['key']['value']:v['value'] for v in store['cuts'][cut]['payload']['items']}
    def point(cut):return read(data/'store/cuts'/(cut+'.json'))['execution_point']
    ports={role:{k:arr(v) for k,v in values.items()} for role,values in binding['fields'].items()}
    for role in ports:
        assert np.array_equal(ports[role]['R'],slot(binding['parents'][role],'image'))
        p=point(binding['parents'][role]);assert p['logical_step']==1 and p['phase']=='single' and p['index']==6
        path=binding['prefix_paths'][role];assert len(path)==7 and path[0]['parent']==config['input_cuts'][role] and path[-1]['result']==binding['posts'][role]
        assert path[-1]['parent']==binding['parents'][role]
    gate,weight=arr(binding['gate']),arr(binding['weight']);bias=None if binding['bias'] is None else arr(binding['bias'])
    nativeP={k:arr(v) for k,v in read(data/'native-projections.json').items()};valid=read(data/'validations.json');predI={};nativeI={};equations={}
    for bits in ('00','10','01','11'):
        x=np.concatenate((ports['shape' if bits[0]=='1' else 'base']['A'],ports['shape' if bits[1]=='1' else 'base']['M']),axis=2).astype(np.float64)
        rebuilt=np.empty((*x.shape[:2],weight.shape[0]),np.float32)
        # Different chunk geometry from worker, all operands/outputs audited.
        for start in range(0,weight.shape[0],192):
            stop=min(start+192,weight.shape[0]);v=x@weight[start:stop].astype(np.float64).T
            if bias is not None:v+=bias[start:stop].astype(np.float64)
            rebuilt[:,:,start:stop]=q(v.astype(np.float32))
        predicted=arr(numeric['P'][bits]);equations['P'+bits]={'sealed_minus_independent':error(predicted,rebuilt),'sealed_minus_native':error(predicted,nativeP[bits])}
        for r,role in enumerate(('base','shape')):
            key=str(r)+bits;R=ports[role]['R'];predI[key]=arr(numeric['I'][key])
            calculated=q((R.astype(np.float64)+q((gate.astype(np.float64)*predicted).astype(np.float32))).astype(np.float32))
            nativeI[key]=q((R.astype(np.float64)+q((gate.astype(np.float64)*nativeP[bits]).astype(np.float32))).astype(np.float32))
            observed=arr(valid[key]['image']);assert np.array_equal(observed,nativeI[key])==valid[key]['cached_exact']
            equations['I'+key]={'sealed_minus_independent':error(predI[key],calculated),'sealed_minus_native_fullblock':error(predI[key],observed),'native_cached_minus_fullblock':error(nativeI[key],observed)}
    for r,role in (('000','base'),('111','shape')):assert np.array_equal(nativeI[r],slot(binding['posts'][role],'image'))
    for receipt in receipts:public(receipt)
    by=index(receipts);assert len(receipts)==1328 and sum(r['operation']=='continue' for r in receipts)==1292 and sum(r['operation']=='apply' for r in receipts)==36
    assert summary['counts']==config['expected_counts']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=1292,acts=36,receipts=1328,images=18,candidate_blocks=8,cached_projections=4)
    for path in binding['prefix_paths'].values():
        for i,v in enumerate(path):assert v==by[v['fingerprint']] and (i==0 or v['parent']==path[i-1]['result'])
    assert len(rows)==len({r['id'] for r in rows})==18;images={};latents={};E=slot(config['input_cuts']['shape'],'conditioning')
    for row in rows:
        assert row==read(data/'rows'/(row['id']+'.json')) and sha(data/row['png'])==row['png_sha256']
        with Image.open(data/row['png']) as im:assert im.mode=='RGB' and im.size==(512,512);images[row['id']]=np.asarray(im).copy()
        latents[row['id']]=arr(row['latent']);assert np.array_equal(latents[row['id']],slot(row['final_cut'],'latent'))
        previous=row['parent'];results=set()
        for v in row['path']:
            assert v==by[v['fingerprint']] and v['parent']==previous;previous=v['result'];results.add(previous)
        assert previous==row['final_cut'] and len(row['path'])==73 and sum(v['operation']=='continue' for v in row['path'])==71
        assert len(row['actions'])==2 and row['parent']==binding['posts'][row['text_role']]
        for act in row['actions']:
            assert act['receipt']==by[act['receipt']['fingerprint']] and act['act']['writes']==[act['slot']] and act['post_act'] in results
            before,after=fields(act['parent']),fields(act['post_act']);assert all(before[k]==after[k] for k in before if k!=act['slot'])
            if act['slot']=='image':
                expected=predI[row['RAM']] if row['kind']=='predicted-exit' else nativeI[row['RAM']]
                assert np.array_equal(slot(act['post_act'],'image'),expected)
                if row['kind']=='predicted-exit':assert row['prediction']==numeric['I'][row['RAM']]
            else:
                assert act['slot']=='conditioning' and point(act['parent'])['logical_step']==2 and point(act['parent'])['phase']=='project'
                assert np.array_equal(slot(act['post_act'],'conditioning'),E)
        assert [t['step'] for t in row['trace']]==[2,3,4]
        for t in row['trace']:
            assert t['cut'] in results and point(t['cut'])['logical_step']==t['step'] and point(t['cut'])['phase']=='project'
            assert np.array_equal(arr(t['latent']),slot(t['cut'],'latent')) and np.array_equal(slot(t['cut'],'conditioning'),E)
    controls={r['id']:{'png_exact':r['png_sha256']==r['expected_png']} for r in rows if 'expected_png' in r}
    assert len(controls)==2
    for r in rows:
        if r['id'] in controls:assert controls[r['id']]['png_exact']==r['png_control_exact']
    weak='cube.R0A0M0.Tbase';strong='cube.R1A1M1.Tshape';observations={}
    for name,image in images.items():
        observations[name]={}
        for label,box in {'whole':None,**config['observers']['windows']}.items():
            crop=lambda v:v if box is None else v[box[1]:box[3],box[0]:box[2]]
            v,low,high=crop(image),crop(images[weak]),crop(images[strong]);d=high.astype(np.float64)-low;den=float(np.sum(d*d))
            observations[name][label]={'to_weak':metric(v,low),'to_strong':metric(v,high),'strong_contrast_projection':None if not den else float(np.sum((v.astype(np.float64)-low)*d)/den)}
    cube={(r,a,m,t):f'cube.R{r}A{a}M{m}.T'+('shape' if t else 'base') for r,a,m,t in itertools.product((0,1),repeat=4)};effects={}
    for i,axis in enumerate('RAMT'):
        effects[axis]=[]
        for key,name in cube.items():
            if key[i]:continue
            other=list(key);other[i]=1;changed=cube[tuple(other)];regions={}
            for region in observations[name]:
                lo,hi=observations[name][region],observations[changed][region]
                regions[region]={'projection_delta':hi['strong_contrast_projection']-lo['strong_contrast_projection'],'to_strong_mae_delta':hi['to_strong']['mae']-lo['to_strong']['mae']}
            effects[axis].append({'low':name,'high':changed,'regions':regions})
    future_errors={}
    for key,role in (('000','base'),('111','shape')):
        name='predicted.'+key+'.T'+role;native='cube.R'+key[0]+'A'+key[1]+'M'+key[2]+'.T'+role
        future_errors[name]={'rgb':metric(images[name],images[native]),'latent':error(latents[name],latents[native]),
            'trajectory':[{'step':a['step'],'latent':error(arr(a['latent']),arr(b['latent']))} for a,b in zip(next(r for r in rows if r['id']==name)['trace'],next(r for r in rows if r['id']==native)['trace'])]}
    report={'schema':'mdb-step1-overlap-audit-v1','job_id':summary['job_id'],'request_sha256':summary['request_sha256'],'mechanics_status':'verified','instrument_status':'native-controls-and-unblinded-region-observers','trend_status':'preserve-conditional-effects','terminal_status':'unassessed',
        'custody':custody,'ancestor_custody':ancestors,'transaction':tx,'numeric_seal':numeric,'counts':summary['counts'],'distinct_receipt_ids':len(by),'equations':equations,'source_bound_text_flags':{k:v['text_unchanged'] for k,v in valid.items()},'controls':controls,
        'observations':observations,'conditional_effects':effects,'predicted_future_errors':future_errors,'forecast_scores':{k:v['rgb']['mae']<1 for k,v in future_errors.items()},'worker_seconds':summary['worker_seconds'],
        'coverage':'All4P+8I fullindependentCPUrebuilds;8nativecachedfullblock imagechecks;16RAMT nativefutures+2CPUcornerfutures. SourceR/A/M observed. Mixed10/01P sealedbeforetheirfirstexecution;00/11sourceP alreadyobserved. FulltypedAct/receipt/latent/shapeEjoins. Textunchangedflags source-bound, not fulltextarrayaudit. RGBprojection isnot geometryclassifier; knownsourcecontrols notfreshpromptgeneralization.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        w,h=(320,350) if kind=='full' else (500,250);sheet=Image.new('RGB',(4*w,5*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);im=im.resize((320,320)) if kind=='full' else im.crop((130,180,380,290)).resize((500,220))
            x,y=i%4*w,i//4*h;sheet.paste(im,(x,y+26));draw.text((x+4,y+7),row['id'],fill='black')
        sheet.save(destination.parent/('overlap-'+kind+'.png'))
    return {'controls':controls,'predicted_future_rgb_errors':{k:v['rgb'] for k,v in future_errors.items()},'eye_projections':{k:v['eyes']['strong_contrast_projection'] for k,v in observations.items()},'conditional_eye_deltas':{k:[r['regions']['eyes']['projection_delta'] for r in v] for k,v in effects.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--onset-source',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve(),a.onset_source.resolve()),sort_keys=True))
