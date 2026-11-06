"""Saved-only fresh reader parity, continuation and competing semantic contrasts."""
import argparse
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def region_effect(base,changed,mask):
    import numpy as np
    b,v=np.asarray(base,np.float64),np.asarray(changed,np.float64);n=int(mask.sum())
    energy=((v-b)**2).sum(axis=-1)
    def green(x):return (x[...,1]-x[...,0])/(x[...,1]+x[...,0]+1.)
    return dict(pixels=n,energy=float(energy[mask].sum()),mean_energy=float(energy[mask].mean()) if n else None,
        rgb_rms=float(np.sqrt(energy[mask].mean()/3)) if n else None,
        signed_green_score=float((green(v)-green(b))[mask].mean()) if n else None,
        status='observed' if n else 'instrument-incomplete-empty-mask')


def catalog_binding(context,parents,native_req,profile):
    return dict(model_identity=context['model'],input_identity={r:p['parent'] for r,p in parents.items()},
        execution=profile,source_identity=native_req['request_sha256'],frontier=context['clock'],horizon=1)


def operand_hypothesis(rows):
    k=rows['keys-only']['regions']['green']['signed_green_score'];v=rows['values-only']['regions']['green']['signed_green_score']
    return dict(keys_signed_green_score=k,values_signed_green_score=v,
        predicted_values_larger_absolute_effect=True,values_larger_matches=None if k is None or v is None else abs(v)>abs(k),
        predicted_values_negative=True,values_sign_matches=None if v is None else v<0,
        role='Frozen exploratory prediction; per-context agreement is an observation, not a scientific stopping gate.')


def stream_hypothesis(rows):
    t=rows['text-only']['regions']['green']['signed_green_score'];i=rows['image-only']['regions']['green']['signed_green_score']
    return dict(text_signed_green_score=t,image_signed_green_score=i,
        predicted_image_larger_absolute_effect=True,image_larger_matches=None if t is None or i is None else abs(i)>abs(t),
        predicted_image_negative=True,image_sign_matches=None if i is None else i<0,
        role='Frozen V-only mediation hypothesis; retain every context, miss and empty-mask abstention.')


def head_analysis(cases,rgb,load,root,conditions,specs,semantics):
    """Rank two consumers separately; keep signed complement interactions."""
    import numpy as np
    result={}
    for condition in conditions:
        r=condition['id'];result[r]={}
        def value(arm,field):
            key=r+'.'+arm
            return np.asarray(rgb(key) if field=='rgb' else load(root,cases[key]['first_image']),np.float64)
        for spec in specs:
            if spec['kind']!='image-head-add':continue
            name=spec['id'].removesuffix('-add');row=dict(heads=spec['heads'],consumers={})
            for field in ('image','rgb'):
                b,f,a,c=[value(x,field) for x in ('base','image-only',name+'-add',name+'-remove')]
                j=f-b;delta=a-b;remaining=c-b;den=float(np.vdot(j,j));interaction=j-delta-remaining
                row['consumers'][field]=dict(full_rms=float(np.sqrt(np.mean(j*j))),add_rms=float(np.sqrt(np.mean(delta*delta))),
                    add_projection=float(np.vdot(j,delta)/den) if den else None,
                    complement_projection=float(np.vdot(j,remaining)/den) if den else None,
                    interaction_relative_rms=float(np.sqrt(np.vdot(interaction,interaction)/den)) if den else None)
            row['signed_green_add']=semantics[r][name+'-add']['regions']['green']['signed_green_score']
            row['signed_green_complement']=semantics[r][name+'-remove']['regions']['green']['signed_green_score']
            result[r][name]=row
        for field in ('image','rgb'):
            result[r]['rank_'+field]=sorted([k for k in result[r] if not k.startswith('rank_')],key=lambda k:-(result[r][k]['consumers'][field]['add_projection'] or 0))
        groups=[k for k in result[r] if not k.startswith('rank_')]
        result[r]['rank_green_signed']=sorted(groups,key=lambda k:result[r][k]['signed_green_add'] if result[r][k]['signed_green_add'] is not None else float('inf'))
        result[r]['rank_rgb_rms']=sorted(groups,key=lambda k:-result[r][k]['consumers']['rgb']['add_rms'])
    return result


def worker():
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='2'
    import numpy as np
    from PIL import Image

    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction
    h=runpy.run_path('flux_source_receiver.py');read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    program=runpy.run_path('frozen-future-program.py');fresh=runpy.run_path('flux_reader_fresh.py')
    trajectory=runpy.run_path('flux_reader_route_future_audit.py')['trajectory']
    bf16=runpy.run_path('flux_reentry_reference.py')['bf16'];measure=runpy.run_path('analyze_flux_reader_arithmetic.py')['measure']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config'];root,source=Path(c['native_root']),Path(c['source_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('audit source/isolation differs')
    if sha(root/'files.json')!=c['native_manifest_sha256'] or sha(source/'files.json')!=c['source_manifest_sha256']:raise ValueError('input inventories changed')
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('native output changed')
    plan,summary,native_req=[read(root/n) for n in ('plan.json','summary.json','request.json')]
    if summary['counts']!=plan['planned_counts'] or native_req['config']!=plan or summary['request_sha256']!=native_req['request_sha256']:raise ValueError('native census/request differs')
    if any(sha(source/n)!=v['sha256'] for n,v in plan['inputs'].items()):raise ValueError('origin source changed')
    started=time.monotonic();cases=read(root/'cases.json');predictions=read(root/'predictions.json')
    prefix_root=root if (root/'prefixes.json').exists() else source
    parents=read(prefix_root/'fresh-parents.json')
    tx=read(root/'transaction-report.json');verify_transaction(root/'transaction',expected_seal_sha256=tx['seal_sha256'])
    timeline=[json.loads(x) for x in (root/'timeline.jsonl').read_text().splitlines()]
    seals=[i for i,x in enumerate(timeline) if x['kind']=='all-predictions-sealed'];futures=[i for i,x in enumerate(timeline) if x['kind']=='native-future-complete']
    ncases=len(cases)
    if len(seals)!=1 or len(futures)!=ncases or min(futures)<=seals[0]:raise ValueError('target suffix chronology differs')
    at_seal=timeline[seals[0]]['counts']
    expected_prefix=plan.get('expected_prefix_counts',dict(native_blocks=300,native_sdpa=300,native_transitions=340,prefix_stops=4))
    if any(at_seal.get(k,0)!=v for k,v in expected_prefix.items()):raise ValueError('pre-forecast prefix census differs')
    if len({x['initial_latent']['sha256'] for x in parents.values()})!=1:raise ValueError('fresh contexts do not share initial noise')
    prefix=read(prefix_root/'prefixes.json')['operands'];reports={};semantics={};exact=0;values=0
    formed={};formation_report={}
    if plan.get('reconstruct_values'):
        formation=read(root/'value-formation.json')
        for receiver,record in formation.items():
            formed[receiver]=load(root,record['values'])
            actual=load(prefix_root,prefix[receiver+'.base']['v'])[:,512:]
            formation_report[receiver]=dict(**measure(actual,formed[receiver]),complete_array_exact=np.array_equal(actual,formed[receiver]),values=int(actual.size),parent=record['parent'])
    def rgb(key):return np.asarray(Image.open(root/cases[key]['image']['path']).convert('RGB'))
    for condition in plan['conditions']:
        r=condition['id'];base=cases[r+'.base'];base_rgb=rgb(r+'.base');masks=fresh['semantic_masks'](base_rgb)
        b,d=[{f:load(prefix_root,a) for f,a in prefix[r+'.'+role].items()} for role in ('base','delta')]
        if formed:
            other=('c1-' if r.startswith('c0-') else 'c0-')+r.split('-')[1]
            b['v'][:,512:]=formed[r];d['v'][:,512:]=formed[other]
        semantic_rows={}
        for spec in plan['case_specs']:
            key=r+'.'+spec['id'];case=cases[key];prediction=predictions[key]
            frozen=load_forecast_tree(root/'transaction',read(root/'transaction/forecasts'/(key+'.json'))['predictions'])
            if json.dumps(frozen,sort_keys=True,default=dict)!=json.dumps(prediction,sort_keys=True):raise ValueError('published forecast differs from seal')
            kv=program['ports'](b,d,spec,bf16)
            if {f:program['payload_digest'](v) for f,v in kv.items()}!=prediction['ports'] or prediction['ports']!=case['footprint']['kv_sha256']:raise ValueError('sealed/installed KV differs')
            fields={}
            for f,a in prediction['fields'].items():
                actual,predicted=load(root,case['first_fields'][f]),load(root,a)
                fields[f]=dict(**measure(actual,predicted),archive_bytes_exact=a['sha256']==case['first_fields'][f]['sha256'])
                exact+=int(fields[f]['archive_bytes_exact']);values+=int(actual.size)
            image_path=trajectory(case,base,'image');text_path=trajectory(case,base,'text');latent_path=trajectory(case,base,'latent')
            candidate=rgb(key);regions={name:region_effect(base_rgb,candidate,mask) for name,mask in masks.items()}
            left=np.indices(base_rgb.shape[:2])[1]<base_rgb.shape[1]//2
            regions.update(left=region_effect(base_rgb,candidate,left),right=region_effect(base_rgb,candidate,~left))
            cone_color='green' if condition['color_assignment']==0 else 'yellow';cylinder_color='yellow' if cone_color=='green' else 'green'
            def difference(a,b):return None if regions[a]['mean_energy'] is None or regions[b]['mean_energy'] is None else regions[a]['mean_energy']-regions[b]['mean_energy']
            semantic_rows[spec['id']]=dict(regions=regions,prompt_assigned_cone_color=cone_color,
                cone_minus_cylinder_mean_energy=difference(cone_color,cylinder_color),green_minus_yellow_mean_energy=difference('green','yellow'),
                left_minus_right_mean_energy=difference('left','right'),
                green_to_yellow_expected_sign=-1,green_to_yellow_observed=regions['green']['signed_green_score'],
                yellow_to_green_expected_sign=1,yellow_to_green_observed=regions['yellow']['signed_green_score'],
                direction_applicable=spec['kind'] in ('band-cross','band-text','band-image','band-key','band-value','value-text','value-image'),shape_label_status='prompt-assigned; inspect unchanged native baseline before semantic interpretation')
            reports[key]=dict(spec=spec,reader_fields=fields,image_trajectory=image_path,text_trajectory=text_path,latent_trajectory=latent_path,
                expected_first_image_change=prediction['structural_prediction']['expected_first_image_change'],
                first_image_prediction_matches=prediction['structural_prediction']['expected_first_image_change']==image_path['first_value_divergence'],
                final_latent=measure(load(root,base['latent']),load(root,case['latent'])),final_rgb=measure(base_rgb,candidate))
            if spec['kind'] in ('band-text','band-image','value-text','value-image'):
                is_text=spec['kind'] in ('band-text','value-text')
                text_reference=r+('.band2' if is_text else '.base')
                image_reference=r+('.base' if is_text else '.band2')
                reports[key]['split_closure']=dict(text_reference=text_reference,image_reference=image_reference,
                    current_text_archive_exact=case['first_fields']['text']['sha256']==cases[text_reference]['first_fields']['text']['sha256'],
                    current_image_archive_exact=case['first_image']['sha256']==cases[image_reference]['first_image']['sha256'],
                    expected_first_text_change=0 if is_text else 1,
                    first_text_change_matches=text_path['first_value_divergence']==(0 if is_text else 1))
        semantics[r]=semantic_rows
    interactions={}
    for condition in plan['conditions']:
        r=condition['id']
        if r+'.text-only' not in cases:continue
        interactions[r]={}
        for field in ('rgb','latent'):
            vals=[rgb(r+'.'+arm) if field=='rgb' else load(root,cases[r+'.'+arm]['latent']) for arm in ('base','band2','text-only','image-only')]
            b,f,t,i=[np.asarray(v,np.float64) for v in vals];full=f-b;te=t-b;ie=i-b;den=float(np.vdot(full,full))
            report=measure(full,te+ie)
            report.update(text_projection=float(np.vdot(full,te)/den) if den else None,image_projection=float(np.vdot(full,ie)/den) if den else None,
                interpretation='Signed projections onto the complete band effect; not additive causal percentages or variance explained.')
            interactions[r][field]=report
        interactions[r]['green_score_interaction']={color:semantics[r]['band2']['regions'][color]['signed_green_score']-semantics[r]['text-only']['regions'][color]['signed_green_score']-semantics[r]['image-only']['regions'][color]['signed_green_score']
            if semantics[r]['band2']['regions'][color]['pixels'] else None for color in ('green','yellow')}
    operand_interactions={};operand_predictions={}
    for condition in plan['conditions']:
        r=condition['id']
        if r+'.keys-only' not in cases:continue
        operand_predictions[r]=operand_hypothesis(semantics[r]);operand_interactions[r]={}
        for field in ('rgb','latent'):
            vals=[rgb(r+'.'+arm) if field=='rgb' else load(root,cases[r+'.'+arm]['latent']) for arm in ('base','band2','keys-only','values-only')]
            b,f,k,v=[np.asarray(x,np.float64) for x in vals];full=f-b;ke=k-b;ve=v-b;den=float(np.vdot(full,full))
            report=measure(full,ke+ve)
            report.update(key_projection=float(np.vdot(full,ke)/den) if den else None,value_projection=float(np.vdot(full,ve)/den) if den else None,
                interpretation='Frozen K-only/V-only factorial; response interaction and signed projections, not additive causal percentages.')
            operand_interactions[r][field]=report
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    controls={}
    control_root=Path(plan.get('historical_control_root',str(source)))
    if plan.get('historical_controls'):
        inventory=read(control_root/'files.json')
        if plan.get('historical_control_manifest_sha256') and sha(control_root/'files.json')!=plan['historical_control_manifest_sha256']:raise ValueError('historical control inventory changed')
        if sha(control_root/'cases.json')!=inventory['cases.json']:raise ValueError('historical control rows changed')
        old=read(control_root/'cases.json')
        for key,oldkey in plan['historical_controls'].items():
            load(control_root,old[oldkey]['latent']);load(control_root,old[oldkey]['first_fields']['text'])
            if sha(control_root/old[oldkey]['image']['path'])!=old[oldkey]['image']['sha256']:raise ValueError('historical control image changed')
            controls[key]=dict(png_exact=cases[key]['image']['sha256']==old[oldkey]['image']['sha256'],
                latent_exact=cases[key]['latent']['sha256']==old[oldkey]['latent']['sha256'],
                text_exact=cases[key]['first_fields']['text']['sha256']==old[oldkey]['first_fields']['text']['sha256'])
    result=dict(schema='mdb-reader-fresh-audit-v1',source_job=summary['job_id'],mechanics_status='verified',instrument_status='reader-byte-parity' if exact==4*ncases else 'reader-discrepancies-observed',
        trend_status='observed',terminal_status='unassessed',counts=summary['counts'],exact_reader_archives=exact,reader_archives=4*ncases,compared_bf16_values=values,
        reader_forecasts=reports,semantic_contrasts=semantics,frozen_semantic_hypotheses=plan['semantic_hypotheses'],
        native_manifest_sha256=sha(root/'files.json'),seal_sha256=tx['seal_sha256'],historical_controls=controls,scope=plan['scope'],stream_interactions=interactions,
        operand_interactions=operand_interactions,operand_predictions=operand_predictions,
        stream_predictions={r:stream_hypothesis(rows) for r,rows in semantics.items()} if all('text-only' in rows and 'image-only' in rows for rows in semantics.values()) else {},
        limitations=['One fresh seed193, four previously encoded prompts; new color/order states but no new wording or encoder computation.',
            'Forecasts use native Flash/linear/norm/SwiGLU primitives; not independently implemented Flash arithmetic.',
            'Original fresh-parent acquisition executed300blocks; current pre-seal counts are retained in the frozen transaction. No target suffix executes before all forecasts seal.',
            'Final images are native observations, not numerical forecasts. All semantic alternatives and countertrends retained.',
            'Color masks are baseline-derived instruments; shape labels require visual conformance, and RGB sensitivity does not establish discrete concept identity.',
            'Original fresh panel includes a band1 spatial control without energy matching and a donor-free own-V edit. Stream panel instead separates the two complete attention output streams; read each plan for the actual cases.'])
    if plan.get('panel')=='heads':
        result['head_analysis']=head_analysis(cases,rgb,load,root,plan['conditions'],plan['case_specs'],semantics)
        result['limitations']=[x.replace('One fresh seed193','One previously observed seed193') for x in result['limitations']]
        result['limitations'].append('Head ranks are exploratory at image exit and native RGB separately. Signed projections are not causal percentages. Text forecasts are shared baseline predictions under protected text outputs; image endpoints are observed.')
    if formed:
        result['value_formation']=formation_report
        shutil.copyfile(root/'value-formation.json',out/'source-value-formation.json')
    if plan.get('new_prompt_encoding'):
        result['limitations'][0]='One fresh seed'+str(plan['seed'])+' and four fresh paraphrases encoded by the original conditioner; seed and wording change together, so their effects are not independently identified.'
        for name in ('encoder-source.json','retrieve_encoder.py'):shutil.copyfile(root/name,out/name)
        result['fresh_cohort_predictions']={}
        for r,rows in semantics.items():
            a,b=result['head_analysis'][r]['g6'],result['head_analysis'][r]['g7']
            ga,gb=a['signed_green_add'],b['signed_green_add'];full=rows['band2']['regions']['green']['signed_green_score']
            result['fresh_cohort_predictions'][r]=dict(selected_heads=a['heads'],outside_heads=b['heads'],selected_green=ga,outside_green=gb,
                full_values_negative=None if full is None else full<0,selected_more_negative=None if ga is None or gb is None else ga<gb,
                selected_more_rgb_aligned=a['consumers']['rgb']['add_projection']>b['consumers']['rgb']['add_projection'],
                role='Frozen sign/order predictions; per-context observations, no scientific stopping gate.')
    if plan.get('upstream_receivers'):
        bindings=read(root/'upstream-bindings.json');result['upstream_bindings']=bindings
        result['upstream_mediation']={}
        for r in plan['upstream_receivers']:
            keys=[r+'-own.base',r+'-own.band2',r+'-swap.band2',r+'-swap.base']
            base_rgb=rgb(keys[0]);masks=fresh['semantic_masks'](base_rgb);account={}
            for field in ('rgb','latent','image'):
                vals=[rgb(k) if field=='rgb' else load(root,cases[k]['latent' if field=='latent' else 'first_image']) for k in keys]
                b,v,z,both=[np.asarray(x,np.float64) for x in vals];total=both-b;vdelta=v-b;zdelta=z-b;interaction=both-z-v+b;den=float(np.vdot(total,total))
                account[field]=dict(total_rms=float(np.sqrt(np.mean(total*total))),band_only_rms=float(np.sqrt(np.mean(vdelta*vdelta))),
                    other_latent_paths_rms=float(np.sqrt(np.mean(zdelta*zdelta))),interaction_rms=float(np.sqrt(np.mean(interaction*interaction))),
                    band_projection=float(np.vdot(total,vdelta)/den) if den else None,
                    other_latent_paths_projection=float(np.vdot(total,zdelta)/den) if den else None,
                    band_removal_projection=float(np.vdot(total,both-z)/den) if den else None,
                    interaction_projection=float(np.vdot(total,interaction)/den) if den else None,
                    interaction_relative_rms=float(np.sqrt(np.vdot(interaction,interaction)/den)) if den else None)
            account['regions']={name:{key:region_effect(base_rgb,rgb(k),mask) for key,k in zip(('base','band-only','latent-band-restored','latent-natural'),keys,strict=True)} for name,mask in masks.items()}
            account['signed_green_interaction']={name:rows['latent-natural']['signed_green_score']-rows['latent-band-restored']['signed_green_score']-rows['band-only']['signed_green_score'] if rows['base']['pixels'] else None for name,rows in account['regions'].items()}
            result['upstream_mediation'][r]=account
        result['limitations']=['Two opposite-color seed241 recipients in one layout; original conditioning retained while native opposite-color latent is installed through typed project Act.',
            'Latent is a persistent source, and V is a recomputed mediator. Earlier readout/attention/MLP writer of the latent remains unidentified.',
            'Own versus swapped states differ in all latent consequences; the selected V band is one mediator. Signed projections are not causal percentages.',
            'Local reader equations use original primitives and observed positioned Q/K/V. Native image futures are observations, not numerical image forecasts.']
        shutil.copyfile(root/'upstream-bindings.json',out/'source-upstream-bindings.json')
    save(out/'audit.json',result);catalog=EquationCatalog(out/'catalog/equations');ids=[];profile=read(root/'native-contract.json')
    for key,case in cases.items():
        context=dict(model='flux2-klein-4b-distilled',seed=plan['seed'],case=key,source_job=summary['job_id'],horizon=1,clock=dict(step=3,block=0,phase='joint'))
        prediction=predictions[key]
        item=catalog.record(law_id='law.flux.reader.native-state-and-consumer-clock.v1',route_id='route.flux.image-KV.text-projection-residual-mlp.v1',alias='cmp.flux.reader-fresh.'+key,
            binding=catalog_binding(context,parents,native_req,profile),
            report=compare_equations({f:load(root,a) for f,a in prediction['fields'].items()},{f:load(root,a) for f,a in case['first_fields'].items()},max_elements=20_000_000),
            conditions=[dict(field=f,op='eq',value=v) for f,v in context.items()],tested_context=context,
            readers=['joint0.text-head','joint0.text-mlp','native-image-suffix'],evidence=dict(audit_sha256=sha(out/'audit.json'),seal_sha256=tx['seal_sha256'],new_state=key not in controls,seed=plan['seed'],
                semantic_status='competing hypotheses; see full signed per-context evidence',terminal_status='unassessed'))
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to={**context,'horizon':2}):raise ValueError('catalog applicability differs')
        if key==list(cases)[-1]:save(out/'catalog/context.json',context);save(out/'catalog/context-wrong-horizon.json',{**context,'horizon':2})
        ids.append(item['fingerprint'])
    save(out/'catalog/provenance.json',dict(records=ncases,fingerprints=ids,matching_queries=ncases,wrong_horizon_abstentions=ncases))
    for n in ('native-contract.json','summary.json','cases.json','predictions.json','plan.json','timeline.jsonl'):shutil.copyfile(root/n,out/('source-'+n))
    shutil.copyfile(prefix_root/'fresh-parents.json',out/'source-fresh-parents.json')
    for case in cases.values():shutil.copyfile(root/case['image']['path'],out/case['image']['path'])
    shutil.copytree(root/'transaction',out/'transaction')
    for n in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/n,out/n)
    save(out/'external-evidence.json',dict(host='beast',roots=dict(native=str(root),reader=str(source),controls=str(control_root)),manifests=dict(native=read(root/'files.json'),reader=read(source/'files.json'),controls=read(control_root/'files.json')),
        scope='Native tensors, fresh parent cuts and receipts remain remote; compact reports, native PNGs and sealed transaction retained locally.'))
    shutil.copyfile('retrieve_reader_evidence.py',out/'retrieve_evidence.py');save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(schema='mdb-reader-fresh-compact-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),source_job=summary['job_id'],seconds=time.monotonic()-started,
        exact_reader_archives=exact,records=ncases,mechanics_status='verified',terminal_status='unassessed'))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        here=Path(__file__).resolve().parent;helper=runpy.run_path(str(here/'flux_reader_attention_audit.py'))
        print(json.dumps(helper['submit'](a.run.resolve()) if a.submit else helper['stage'](a.run.resolve(),a.study.resolve(),worker_path=Path(__file__),
            extra_files=[(a.study.resolve()/'payload/flux_reader_route_future.py','frozen-future-program.py'),
                *[(here/n,n) for n in ('flux_reader_fresh.py','flux_reader_route_future_audit.py','flux_reentry_reference.py')]],audit_kind='reader-fresh',logical_operations=65)))
