"""Independent saved-state audit of formation, clamp/rescue and native RGB."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination,source,overlap):
    from mdb.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/'extracted/shape-formation-result';old=source/'run-01/extracted/step1-writers-result'
    tools=runpy.run_path(str(here/'flux_family_custody.py'));custody=tools['verify'](run);ancestors={str(p):tools['verify'](p/'run-01') for p in (source,overlap)}
    util=runpy.run_path(str(here/'analyze_flux_scene_dependencies.py'));sha,metric=util['sha'],util['metric'];typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'))
    json_tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree'];public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt'];index=runpy.run_path(str(here/'analyze_flux_edit_onset.py'))['index_receipts']
    config,summary,binding,rows,receipts=[read(data/n) for n in ('plan.json','summary.json','binding.json','rows.json','receipts.json')];request=read(run/'request.json')
    assert config==request['config'] and summary==read(run/'report.json')['report'] and summary['request_sha256']==request['request_sha256']==custody['request_sha256']
    assert sha(source/'analysis-v1.json')==config['source_audit_sha256'] and sha(overlap/'analysis-v1.json')==config['overlap_audit_sha256']
    for name,digest in request['source_hashes'].items():assert sha(run/'payload'/name)==digest
    for name,meta in config['input_files'].items():assert sha(old/name)==meta['sha256'] and (old/name).stat().st_size==meta['bytes']
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified'] and tx['phase']=='closed'
    forecast=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/formation_roles.json')['predictions']));assert forecast==config['hypotheses']
    timeline=read(data/'timeline.json');assert next(r for r in timeline if r['kind']=='sealed')['counts']==dict(loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0)
    fresh=config.get('fresh_seed') is not None
    expected=dict(loads=1,encodes=0,noise_draws=int(fresh),native_transitions=1108 if fresh else 1068,acts=49 if fresh else 47,receipts=1157 if fresh else 1115,images=14)
    assert summary['counts']==config['expected_counts']==expected
    for r in receipts:public(r)
    by=index(receipts);assert len(receipts)==expected['receipts'] and sum(r['operation']=='continue' for r in receipts)==expected['native_transitions'] and sum(r['operation']=='apply' for r in receipts)==expected['acts']
    store=typed['audit_store'](data/'store')
    def slot(cut,name):return route['slot_array'](data/'store',cut,name,typed)
    def fields(cut):return {v['key']['value']:v['value'] for v in store['cuts'][cut]['payload']['items']}
    def point(cut):return read(data/'store/cuts'/(cut+'.json'))['execution_point']
    def chain(path,parent,final):
        for r in path:assert r==by[r['fingerprint']] and r['parent']==parent;parent=r['result']
        assert parent==final
    roots=config['input_cuts']
    if fresh:
        setup=read(data/'fresh-context.json');assert setup['seed']==config['fresh_seed'];chain(setup['base_path'],setup['initial'],setup['common_step1']);assert len(setup['base_path'])==28
        for role,value in setup['preparation'].items():
            chain(value['path'],setup['common_step1'],value['cut']);assert len(value['path'])==7 and len(value['actions'])==1
            act=value['actions'][0];assert act['slot']=='conditioning' and act['act']['writes']==['conditioning']
            assert np.array_equal(slot(act['post_act'],'conditioning'),route['slot_array'](old/'store',config['input_cuts'][role],'conditioning',typed))
            assert all(fields(act['parent'])[k]==fields(act['post_act'])[k] for k in fields(act['parent']) if k!='conditioning')
        roots={k:v['cut'] for k,v in setup['preparation'].items()}
    cuts=binding['cuts']
    for role in ('base','shape'):
        assert cuts[role]['0']==roots[role];chain(binding['prefix_paths'][role],roots[role],cuts[role]['7']);assert len(binding['prefix_paths'][role])==7
        for i in range(8):
            p=point(cuts[role][str(i)]);assert p['phase']=='single' and p['logical_step']==1 and p['index']==i
    for i in range(8):
        low,high=[fields(cuts[r][str(i)]) for r in ('base','shape')];assert all(low[k]==high[k] for k in low if k not in ('conditioning','image','text'))
    E=slot(roots['shape'],'conditioning');assert len(rows)==14 and [r['recipe'] for r in rows]==config['recipes'];images={};latents={};state_effects={}
    for row in rows:
        assert row==read(data/'rows'/(row['id']+'.json')) and sha(data/row['png'])==row['png_sha256']
        recipe=row['recipe'];entry=recipe['entry'];assert row['parent']==cuts[recipe['role']][str(entry)]
        chain(row['path'],row['parent'],row['final_cut']);assert sum(r['operation']=='continue' for r in row['path'])==78-entry
        assert len(row['path'])==78-entry+len(row['actions'])
        with Image.open(data/row['png']) as im:assert im.size==(512,512) and im.mode=='RGB';images[row['id']]=np.asarray(im).copy()
        latents[row['id']]=load_tensor_archive(data,row['latent']);assert np.array_equal(latents[row['id']],slot(row['final_cut'],'latent'))
        result_ids={r['result'] for r in row['path']}|{row['parent']};early=row['early'];assert [v['index'] for v in early]==list(range(entry,8));declared={}
        for observation in early:
            i=observation['index'];mode=recipe['mode'];selected={}
            if mode in ('I','T','IT') and i==entry:selected={s:('shape',i) for c,s in (('I','image'),('T','text')) if c in mode}
            elif mode=='Tseed0.reset6' and i in (0,6):selected={'text':('shape' if i==0 else 'base',i)}
            elif mode.startswith('Tclamp') and i<=6:selected={'text':('shape' if mode=='Tclamp0to5.rescue6' and i==6 else 'base',i)}
            elif mode=='Iclamp0to6' and i<=6:selected={'image':('base',i)}
            assert observation['writes']==list(selected) and observation['before'] in result_ids and observation['after'] in result_ids
            for fp in ('before','after'):
                p=point(observation[fp]);assert p['phase']=='single' and p['index']==i and p['logical_step']==1
            for s,value in selected.items():declared[(i,s)]=value
            if not selected:assert observation['before']==observation['after']
        for act in row['actions']:
            assert act['receipt']==by[act['receipt']['fingerprint']] and act['act']['writes']==[act['slot']] and act['post_act'] in result_ids
            before,after=fields(act['parent']),fields(act['post_act']);assert all(before[k]==after[k] for k in before if k!=act['slot']);p=point(act['parent'])
            if act['slot']=='conditioning':assert p['phase']=='project' and p['logical_step']==2 and np.array_equal(slot(act['post_act'],'conditioning'),E)
            else:
                assert p['phase']=='single' and p['logical_step']==1;role,i=declared.pop((p['index'],act['slot']))
                assert np.array_equal(slot(act['post_act'],act['slot']),slot(cuts[role][str(i)],act['slot']))
        assert not declared
        assert [t['step'] for t in row['trace']]==[2,3,4]
        for t in row['trace']:
            assert t['cut'] in result_ids and np.array_equal(slot(t['cut'],'conditioning'),E) and np.array_equal(load_tensor_archive(data,t['latent']),slot(t['cut'],'latent'))
        state_effects[row['id']]={}
        for obs in early:
            i=obs['index'];state_effects[row['id']][str(i)]={}
            for s in ('image','text'):
                base,high,value=[slot(fp,s).astype(np.float64) for fp in (cuts['base'][str(i)],cuts['shape'][str(i)],obs['after'])];d=high-base;den=float(np.sum(d*d))
                state_effects[row['id']][str(i)][s]={'source_contrast_projection':None if not den else float(np.sum((value-base)*d)/den),'to_base_l2':float(np.linalg.norm(value-base)),'to_shape_l2':float(np.linalg.norm(value-high))}
    controls={r['id']:{'png_exact':r['png_sha256']==r['expected_png']} for r in rows if 'expected_png' in r}
    for row in rows:
        if row['id'] in controls:assert controls[row['id']]['png_exact']==row['png_control_exact']
    for key in ('handoff0.IT','handoff6.IT'):
        controls[key]={'to_strong_png_exact':bool(np.array_equal(images[key],images['reference.shape'])),'to_strong_latent_exact':bool(np.array_equal(latents[key],latents['reference.shape']))}
    observations={}
    for name,image in images.items():
        observations[name]={}
        for label,box in {'whole':None,**config['observers']['windows']}.items():
            crop=lambda v:v if box is None else v[box[1]:box[3],box[0]:box[2]]
            v,low,high=crop(image),crop(images['reference.base']),crop(images['reference.shape']);d=high.astype(np.float64)-low;den=float(np.sum(d*d))
            observations[name][label]={'to_weak':metric(v,low),'to_strong':metric(v,high),'strong_contrast_projection':None if not den else float(np.sum((v.astype(np.float64)-low)*d)/den)}
    eye=lambda name:observations[name]['eyes'];alpha=lambda name:eye(name)['strong_contrast_projection'];mae=lambda name:eye(name)['to_strong']['mae']
    scores={'early_T_closer_than_I':mae('handoff0.T')<mae('handoff0.I'),'early_T_leaves_positive_image_future_after_reset6':alpha('Tseed0.reset6')>0,
        'text_clamp_weakens_strong_contrast':alpha('Tclamp0to6')<1,'text_rescue6_improves_matching':mae('Tclamp0to5.rescue6')<mae('Tclamp0to6')}
    report={'schema':'mdb-shape-formation-audit-v1','job_id':summary['job_id'],'request_sha256':summary['request_sha256'],'mechanics_status':'verified','instrument_status':'native-controls-and-supplied-regions','trend_status':'per-case-effects-retained','terminal_status':'unassessed',
        'custody':custody,'ancestor_custody':ancestors,'transaction':tx,'counts':summary['counts'],'distinct_receipt_ids':len(by),'controls':controls,'observations':observations,'state_effects':state_effects,'forecast_scores':scores,'fresh_seed':config.get('fresh_seed'),'worker_seconds':summary['worker_seconds'],
        'coverage':'Complete typedI/T captures atallsource/interventionboundaries0..7; exact declaredActpayloads/untouchedclosure, orderedreceipts, commonlaterEshape andlatent/RGBjoins. Nativeformation andnativefuture; sourceencodingsobserved. RGB/fieldcontrastprojections notsemanticpercentages. Freshseed RNGconfiguration source-bound; noCUDAnoise redrawn offline.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    for kind in ('full','eyes'):
        w,h=(320,350) if kind=='full' else (500,250);sheet=Image.new('RGB',(4*w,4*h),'#f5f5f5');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row['id']]);im=im.resize((320,320)) if kind=='full' else im.crop((130,180,380,290)).resize((500,220));x,y=i%4*w,i//4*h;sheet.paste(im,(x,y+26));draw.text((x+4,y+7),row['id'],fill='black')
        sheet.save(destination.parent/('formation-'+kind+'.png'))
    return {'job_id':summary['job_id'],'controls':controls,'forecast_scores':scores,'eye_projections':{k:v['eyes']['strong_contrast_projection'] for k,v in observations.items()},'state_at6':{k:v.get('6') for k,v in state_effects.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--overlap-source',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve(),a.overlap_source.resolve()),sort_keys=True))
