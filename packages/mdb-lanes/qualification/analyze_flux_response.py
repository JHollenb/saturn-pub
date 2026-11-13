"""Independent saved-evidence checks for the prospective response-law assay."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
from mdb_lanes.edit_search import file_digest
from mdb_lanes.parameter_artifacts import validate_manifest


def analyze(run,source,output):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text())
    composite=read(run/'analysis-view.json')if(run/'analysis-view.json').exists()else None
    custody=composite if composite else runpy.run_path(str(here/'flux_family_custody.py'))['verify'](run)
    data=run/'extracted/binding-minimal-result';config,summary,forecasts,rows,receipts=[read(data/n)for n in('plan.json','summary.json','forecasts.json','rows.json','receipts.json')]
    request=read(run/'request.json');assert config==request['config']and summary==read(run/'report.json')['report']
    worker=runpy.run_path(str(run/'payload/flux_response_campaign.py'));expected=worker['expected_counts'](len(worker['cases']()))
    if composite:expected=dict(expected,loads=2,receipts=96)
    assert summary['counts']==expected and len(rows)==expected['images']and len(receipts)==expected['receipts']
    assert summary['entries']==dict(blocks=97*len(worker['cases']()),target_blocks=len(worker['cases']()))
    for name,h in request['source_hashes'].items():assert file_digest(run/'payload'/name)==h
    for name,h in read(run/'payload/payload-seal.json')['wheels'].items():assert file_digest(run/'payload'/name)==h
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256']);assert tx['verified']and tx['phase']==('released'if composite else'closed')
    tree=runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree']
    frozen=tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/response_program.json')['predictions']))
    assert frozen['config']==config and frozen['forecasts']==forecasts
    assert frozen['entries_at_seal']==dict(blocks=0,target_blocks=0)and frozen['counts_at_seal']==dict(expected,loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0)
    assert validate_manifest(read(data/'parameter-artifacts.json'))==read(data/'parameter-artifacts.json')
    assert file_digest(data/'retrieve_weights.py')==request['source_hashes']['retrieve_weights.py']
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));store=typed['audit_store'](data/'store');B=typed['quantize_bf16']
    public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    for r in receipts:public(r['body'])
    readslot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot'];slot=lambda fp,k:readslot(data/'store',fp,k)
    def fields(fp):return {v['key']['value']:v['value']for v in store['cuts'][fp]['payload']['items']}
    def array(record):
        p=data/record['path'];assert file_digest(p)==record['sha256'];value=np.load(p,allow_pickle=False)
        assert list(value.shape)==record['shape']and str(value.dtype)==record['dtype']and np.isfinite(value).all()
        return value
    def error(predicted,native):
        p,n=predicted.astype(float),native.astype(float);delta=p-n;rms=float(np.sqrt(np.mean(delta**2)));scale=float(np.sqrt(np.mean(n**2)))
        pnorm,nnorm=float(np.linalg.norm(p)),float(np.linalg.norm(n))
        return dict(exact=bool(np.array_equal(predicted,native)),scalars=int(n.size),rms=rms,reference_rms=scale,relative_rms=None if scale==0 else rms/scale,max_abs=float(np.max(np.abs(delta))),cosine=None if pnorm==0 or nnorm==0 else float(np.sum(p*n)/(pnorm*nnorm)))
    by={(r['id'],r['mode']):r for r in rows};base_cut=config['parents']['base'];edited_cut=config['parents']['edited']
    scaffold=by['base','native']['target']['exit'];control_fields=fields(scaffold)
    assert typed['unpack'](store['cuts'][scaffold]['execution_point'])['index']==2
    branch_receipts={}
    for row in rows:
        label=row['id']+'.'+row['mode'];seq=[r['body']for r in receipts if r['branch']==label];branch_receipts[label]=seq
        start=base_cut if row['mode']=='native'else scaffold;native_steps=54 if row['mode']=='native'else 53
        grouped=any(r.get('steps',1)>1 for r in seq if r['operation']=='continue')
        receipt_count=(5 if row['mode']=='native'else 4)if grouped else native_steps+2
        assert len(seq)==receipt_count and sum(r.get('steps',1)for r in seq if r['operation']=='continue')==native_steps
        assert all(r.get('steps',1)==len(r['transitions'])for r in seq if r['operation']=='continue')
        for r in seq:
            assert r['parent']==start;start=r['result']
            assert r==read(data/'store/receipts'/(r['fingerprint']+'.json'))
        assert start==row['final']
        assert len(row['actions'])==2 and [a['slot']for a in row['actions']]==['image','text']
        for action in row['actions']:
            before,after=fields(action['before']),fields(action['after']);s=action['slot']
            assert all(before[k]==after[k]for k in before if k!=s)
            record=forecasts[row['id']]['input_fields'][s]if row['mode']=='native'else forecasts[row['id']]['fields'][s]['exit']
            assert np.array_equal(slot(action['after'],s),array(record))
            assert action['receipt']in seq
        assert file_digest(data/row['png'])==row['png_sha256']
        if row['mode']=='predicted':
            assert row['clock_scaffold']==scaffold
            assert row['target_entries_before_image']==(0 if composite else 1 if row['id']=='base'else 0)
            installed=fields(row['entry']);assert all(installed[k]==control_fields[k]for k in installed if k not in{'image','text'})
    prior=source/'collection-01/extracted/binding-minimal-result';prior_rows={r['id']:r for r in read(prior/'rows.json')}
    assert file_digest(source/'audit-v1.json')==config['source_audit_sha256']
    base_image=np.asarray(Image.open(prior/prior_rows['control']['png'])).copy();target_image=np.asarray(Image.open(prior/prior_rows['positive']['png'])).copy()
    assert by['base','native']['png_sha256']==config['baseline_png_sha256']==prior_rows['control']['png_sha256']
    compare=runpy.run_path(str(here/'analyze_flux_concept_routes.py'))['comparison']
    outcomes={};local={};causal_writes={};chronology={};images={}
    for spec in worker['cases']():
        name=spec['id'];native=by[name,'native'];predicted=by[name,'predicted'];f=forecasts[name];I=array(f['input_fields']['image']);T=array(f['input_fields']['text'])
        assert np.array_equal(I,slot(config['parents'][spec['receiver']],'image'))
        expected_T=slot(base_cut,'text').copy();ix=spec['rows'];donorT=slot(edited_cut,'text')
        if spec['dose']==1:expected_T[:,ix]=donorT[:,ix]
        elif ix:expected_T[:,ix]=B((expected_T[:,ix].astype(float)+spec['dose']*(donorT[:,ix].astype(float)-expected_T[:,ix])).astype(np.float32))
        assert np.array_equal(T,expected_T)
        target=native['target'];entry,exit=target['entry'],target['exit'];p={k:array(v)for k,v in target['fields'].items()}
        actual_fields=fields(exit);assert all(actual_fields[k]==control_fields[k]for k in actual_fields if k not in{'image','text'})
        local[name]={};causal_writes[name]={}
        for role in('image','text'):
            actual=dict(attention=p['attention_'+role],mlp=p['mlp_'+role],mlp_input=p['mlp_input_'+role],exit=slot(exit,role))
            local[name][role]={k:error(array(f['fields'][role][k]),v)for k,v in actual.items()}
            mod=slot(entry,'context')['img_mod'if role=='image'else'txt_mod'];parts=np.split(mod.reshape(1,1,-1),6,axis=-1)
            calculated=B(B(slot(entry,role)+B(parts[2]*actual['attention']))+B(parts[5]*actual['mlp']))
            assert np.array_equal(calculated,actual['exit'])
            reference=by['base','native']['target'];base_exit=slot(reference['exit'],role)
            predicted_effect=array(f['fields'][role]['exit'])-array(forecasts['base']['fields'][role]['exit'])
            actual_effect=actual['exit']-base_exit
            causal_writes[name][role]=error(predicted_effect,actual_effect)
            causal_writes[name][role]['by_field']={}
            for field in('attention','mlp_input','mlp','exit'):
                baseline=base_exit if field=='exit'else array(reference['fields'][field+'_'+role])
                pe=array(f['fields'][role][field])-array(forecasts['base']['fields'][role][field])
                causal_writes[name][role]['by_field'][field]=error(pe,actual[field]-baseline)
        native_rgb=np.asarray(Image.open(data/native['png'])).copy();predicted_rgb=np.asarray(Image.open(data/predicted['png'])).copy()
        images[name]=(native_rgb,predicted_rgb)
        pixel=error(predicted_rgb,native_rgb);pixel['mae']=float(np.mean(np.abs(predicted_rgb.astype(float)-native_rgb)))
        sides={}
        for mode,image in(('native',native_rgb),('predicted',predicted_rgb)):
            sides[mode]=[]
            for part in(image[:,:256],image[:,256:]):
                a=part.astype(float)/255;mask=(a.max(-1)-a.min(-1)>.18)&(a.max(-1)>.12)
                red=int(np.count_nonzero(mask&(a.argmax(-1)==0)));blue=int(np.count_nonzero(mask&(a.argmax(-1)==2)))
                sides[mode].append('red'if red>blue else'blue'if blue else None)
        signed={mode:compare(image,base_image,target_image)for mode,image in(('native',native_rgb),('predicted',predicted_rgb))}
        commits={str(c['step']):error(slot(next(x['cut']for x in predicted['commits']if x['step']==c['step']),'latent'),slot(c['cut'],'latent'))for c in native['commits']}
        outcomes[name]=dict(pixels=pixel,colors=sides,categorical_prediction=spec['categorical'],categorical_matched=None if spec['categorical']is None else sides['native']==spec['categorical'],
            rgb_donor_direction=signed,committed_latent_errors=commits,scope='Coarse half-frame hue diagnostics; native-image shape/relations/collateral review remains explicit. RGB contrast uses prior successful program, not a semantic score.')
        native_positions=[i for i,r in enumerate(receipts)if r['branch']==name+'.native'and r['body']['operation']=='continue']
        predicted_positions=[i for i,r in enumerate(receipts)if r['branch']==name+'.predicted']
        before=max(predicted_positions)<min(native_positions)
        assert before==(name!='base');chronology[name]=dict(predicted_future_before_corresponding_native_target=before,entries_before_predicted_image=predicted['target_entries_before_image'])
    linear_foil={}
    for role in('image','text'):
        linear_foil[role]={}
        native_half=by['half','native']['target']
        for field in('attention','mlp_input','mlp','exit'):
            p0,p4,p11=[array(forecasts[n]['fields'][role][field]).astype(float)for n in('base','row4','row11')]
            foil=B((p0+.5*(p4-p0)+.5*(p11-p0)).astype(np.float32))
            actual=(slot(native_half['exit'],role)if field=='exit'else array(native_half['fields'][field+'_'+role]))
            linear_foil[role][field]=error(foil,actual)
    result=dict(schema='mdb-response-program-audit-v1',mechanics_status='verified',instrument_status='Complete states/ports; coarse RGB observer plus separate image review',terminal_status='unassessed',
        auditor_source_sha256=file_digest(Path(__file__)),auditor_tools_manifest_sha256=file_digest(here/'tools-manifest.json')if(here/'tools-manifest.json').exists()else None,
        custody=custody,transaction=tx,counts=expected,entries=summary['entries'],local_errors=local,signed_write_errors=causal_writes,outcomes=outcomes,chronology=chronology,
        dense_equation_comparisons={n:f['dense_comparison']for n,f in forecasts.items()},work_geometry={n:f['work']for n,f in forecasts.items()},
        costs=dict(cache_seconds=summary['cache_seconds'],prediction_seconds={n:f['seconds']for n,f in forecasts.items()},dense_validation_seconds={n:f['dense_validation_seconds']for n,f in forecasts.items()},worker_seconds=summary['worker_seconds']),
        source_assistance='Observed J00 exits and unchanged time context, pinned joint1 weights; baseline supplies native joint2 clock scaffold. No new native ports used by predictor. All five forecasts seal before first target; new predicted futures execute before corresponding targets.',
        independent_prefix_scope='Mathematical FP64 reductions with explicit BF16 boundaries, not CUDA-kernel emulation; source encoder and earlier image history observed.',new_model_calls_in_audit=0)
    result['post_hoc_linear_foil']=dict(errors=linear_foil,formula='BF16(Ybase + .5*(Yrow4-Ybase) + .5*(Yrow11-Ybase))',
        scope='Saved-only diagnostic comparison using frozen independent single-row outputs against the native half-dose joint target. Foil was not prospectively frozen or installed; no RGB superiority claim and zero new model calls.')
    receiver={}
    assert np.array_equal(array(forecasts['half']['input_fields']['text']),array(forecasts['half.image1']['input_fields']['text']))
    for role in('image','text'):
        receiver[role]={}
        for field in('attention','mlp_input','mlp','exit'):
            predicted_difference=array(forecasts['half.image1']['fields'][role][field])-array(forecasts['half']['fields'][role][field])
            def observed(name):
                target=by[name,'native']['target']
                return slot(target['exit'],role)if field=='exit'else array(target['fields'][field+'_'+role])
            receiver[role][field]=error(predicted_difference,observed('half.image1')-observed('half'))
    receiver['RGB']=error(images['half.image1'][1].astype(float)-images['half'][1].astype(float),images['half.image1'][0].astype(float)-images['half'][0].astype(float))
    receiver['commits']={}
    for step in(3,4):
        def latent(name,mode):return slot(next(c['cut']for c in by[name,mode]['commits']if c['step']==step),'latent')
        receiver['commits'][str(step)]=error(latent('half.image1','predicted')-latent('half','predicted'),latent('half.image1','native')-latent('half','native'))
    result['receiving_state_contrast']=dict(errors=receiver,scope='Same frozen half-dose text, two observed image receivers. Saved-only contrast of prospective outputs and native futures; a small whole-state or RGB error does not establish accurate receiver-effect prediction.',new_model_calls=0)
    with output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    sheet=Image.new('RGB',(512,len(images)*280),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,pair)in enumerate(images.items()):
        for j,image in enumerate(pair):
            sheet.paste(Image.fromarray(image).resize((256,256)),(j*256,i*280+24));draw.text((j*256+4,i*280+6),name+(' native'if j==0 else' predicted write'),fill='black')
    sheet.save(output.with_suffix('.png'))
    return dict(mechanics_status=result['mechanics_status'],counts=expected,
        image_exit_relative_rms={k:v['image']['exit']['relative_rms']for k,v in local.items()},rgb_mae={k:v['pixels']['mae']for k,v in outcomes.items()},
        colors={k:v['colors']for k,v in outcomes.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in('run','source','output'):p.add_argument(n,type=Path)
    p.add_argument('--continuation',type=Path);p.add_argument('--view',type=Path)
    a=p.parse_args()
    if a.continuation:
        if a.view is None:raise ValueError('explicit derived view path required')
        runpy.run_path(str(Path(__file__).with_name('assemble_flux_response.py')))['assemble'](a.run.resolve(),a.continuation.resolve(),a.view.resolve());a.run=a.view
    print(json.dumps(analyze(*[getattr(a,n).resolve()for n in('run','source','output')]),sort_keys=True))
