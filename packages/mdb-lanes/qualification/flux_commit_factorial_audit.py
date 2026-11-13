"""Saved-only carry/flow factorial, signed consumers and sealed forecast audit."""
import argparse
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def same_record(a,b):
    """Forecast trees freeze JSON lists as tuples and maps as proxies."""
    return json.dumps(a,sort_keys=True,default=dict)==json.dumps(b,sort_keys=True,default=dict)


def finite_commit_numpy(z,n,delta,bf16):
    """Tested CUDA/BF16 tensor-scalar profile, then product, FP32 sum and cast.

    The initial audit used FP32 delta here and missed. The separately retained
    scalar diagnostic identifies BF16 scalar conversion before multiplication.
    This helper is qualified for that recorded profile, not all Torch backends.
    """
    import numpy as np
    scalar=bf16(np.asarray([delta],np.float32))[0]
    product=bf16(np.asarray(n,np.float32)*scalar)
    return bf16(np.asarray(z,np.float32)+product),product


def factorial(values):
    """Signed contrasts; interaction is retained rather than assigned to a part."""
    import numpy as np
    b,n,z,d=[np.asarray(values[k],np.float64) for k in ('00','01','10','11')]
    carry,flow,total=z-b,n-b,d-b
    interaction=total-carry-flow;den=float(np.vdot(total,total))
    rms=lambda a:float(np.sqrt(np.mean(a*a)))
    result=dict(total_rms=rms(total),carry_rms=rms(carry),flow_rms=rms(flow),interaction_rms=rms(interaction),
        carry_projection=float(np.vdot(total,carry)/den) if den else None,
        flow_projection=float(np.vdot(total,flow)/den) if den else None,
        interaction_projection=float(np.vdot(total,interaction)/den) if den else None,
        carry_flow_cross_energy=float(2*np.vdot(carry,flow)),
        total_energy=den,carry_energy=float(np.vdot(carry,carry)),flow_energy=float(np.vdot(flow,flow)),
        interaction_energy=float(np.vdot(interaction,interaction)),
        interaction_cross_energy=float(2*np.vdot(carry+flow,interaction)),
        interpretation='Signed projections and finite contrasts, not additive causal percentages.')
    result['energy_closure_remainder']=den-sum(result[k] for k in ('carry_energy','flow_energy','carry_flow_cross_energy','interaction_energy','interaction_cross_energy'))
    return result


def worker():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    import numpy as np
    from PIL import Image
    from mdb_lanes.equation_comparison import EquationCatalog,compare_equations
    from mdb_lanes.forecast_transaction import load_forecast_tree,verify_transaction
    h=runpy.run_path('flux_source_receiver.py');read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    measure=runpy.run_path('analyze_flux_reader_arithmetic.py')['measure'];bf16=runpy.run_path('flux_reentry_reference.py')['bf16']
    fresh=runpy.run_path('flux_reader_fresh.py');region=runpy.run_path('flux_reader_fresh_audit.py')['region_effect']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config'];root,source=Path(c['native_root']),Path(c['source_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('audit isolation/source differs')
    if sha(root/'files.json')!=c['native_manifest_sha256'] or sha(source/'files.json')!=c['source_manifest_sha256']:raise ValueError('input inventory differs')
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('native evidence changed')
    plan,summary,native_req=[read(root/n) for n in ('plan.json','summary.json','request.json')]
    if summary['counts']!=plan['planned_counts'] or native_req['config']!=plan or summary['request_sha256']!=native_req['request_sha256']:raise ValueError('native census/request differs')
    if any(sha(source/n)!=v['sha256'] for n,v in plan['inputs'].items()):raise ValueError('source inputs changed')
    started=time.monotonic();cases=read(root/'cases.json');predictions=read(root/'predictions.json');operands=read(root/'source-operands.json')
    tx=read(root/'transaction-report.json');verify_transaction(root/'transaction',expected_seal_sha256=tx['seal_sha256'])
    timeline=[json.loads(x) for x in (root/'timeline.jsonl').read_text().splitlines()]
    seal=[i for i,x in enumerate(timeline) if x['kind']=='all-predictions-sealed'];futures=[i for i,x in enumerate(timeline) if x['kind']=='native-future-complete']
    if len(cases)!=28 or len(predictions)!=16 or len(seal)!=1 or len(futures)!=28 or min(futures)<=seal[0]:raise ValueError('case/seal chronology differs')
    counts=timeline[seal[0]]['counts']
    if any(counts[k]!=v for k,v in plan['expected_prefix_counts'].items()) or counts['images'] or counts['native_scheduler_components']:raise ValueError('pretarget frontier census differs')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    errors={};ports={};controls={};factorials={};semantics={};ordering={};rounding={};exact=0;scalar_positions=0;ids=[]
    catalog=EquationCatalog(out/'catalog/equations');profile=read(root/'native-contract.json')
    def frozen(key):return load_forecast_tree(root/'transaction',read(root/'transaction/forecasts'/(key+'.json'))['predictions'])
    def rgb(key):return np.asarray(Image.open(root/cases[key]['png']['path']).convert('RGB'))
    for key,p in predictions.items():
        if not same_record(frozen(key),p):raise ValueError('published forecast differs from seal')
        row=cases[key];spec=row['spec'];r=spec['receiver'];zs,ns=operands[spec['z_source']],operands[spec['n_source']]
        z,n=load(root,zs['latent']),load(root,ns['noise']);sig=np.asarray(operands[r]['sigmas'],np.float32);dt=np.float32(sig[3]-sig[2])
        mathematical,product=finite_commit_numpy(z,n,dt,bf16);target=load(root,row['component']['latent'])
        native_fields=dict(latent=target,image=load(root,row['project_image']),values=load(root,row['actual_values']))
        predicted_fields={f:load(root,p[f]) for f in native_fields}
        actual_records=dict(latent=row['component']['latent'],image=row['project_image'],values=row['actual_values'])
        errors[key]={f:dict(**measure(a,predicted_fields[f]),archive_bytes_exact=p[f]['sha256']==actual_records[f]['sha256']) for f,a in native_fields.items()}
        errors[key]['independent_saved_commit']=measure(target,mathematical)
        errors[key]['saved_product']=measure(load(root,p['product']),product)
        errors[key]['clock_modulation_exact']=row['project_modulation']==operands[r]['modulation']
        # Retain scalar conversion, product rounding, FP32 add and final cast.
        actual_product=load(root,p['product']);scalar=bf16(np.asarray([dt],np.float32))[0]
        ideal_product=n.astype(np.float64)*float(dt);ideal=z.astype(np.float64)+ideal_product
        scalar_product=n.astype(np.float64)*float(scalar);wide_sum=z.astype(np.float64)+actual_product.astype(np.float64)
        added=(z+actual_product).astype(np.float32)
        round_defect=target.astype(np.float64)-ideal
        rounding[key]=dict(delta_sigma=float(dt),effective_delta_bf16=float(scalar),scalar_rounding_rms=float(np.sqrt(np.mean((scalar_product-ideal_product)**2))),
            product_rounding_rms=float(np.sqrt(np.mean((actual_product.astype(np.float64)-scalar_product)**2))),
            fp32_add_rounding_rms=float(np.sqrt(np.mean((added.astype(np.float64)-wide_sum)**2))),
            final_cast_rounding_rms=float(np.sqrt(np.mean((target.astype(np.float64)-added.astype(np.float64))**2))),
            complete_rounding_rms=float(np.sqrt(np.mean(round_defect**2))))
        for f in native_fields:exact+=int(errors[key][f]['archive_bytes_exact']);scalar_positions+=int(native_fields[f].size)
        context=dict(model='flux2-klein-4b-distilled',source_job=summary['job_id'],case=key,clock='step2.scheduler->step3.joint0.V',
            receiver=r,z_source=spec['z_source'],n_source=spec['n_source'],execution=profile)
        record=catalog.record(law_id='law.flux.euler-project-image-value.native.v1',route_id='route.flux.commit.project-image.joint0-value.v1',alias='cmp.flux.commit.'+key,
            binding=dict(model_identity=context['model'],input_identity=read(root/'source-cuts.json'),source_identity=native_req['request_sha256'],execution=profile,
                frontier=dict(step=2,phase='scheduler'),horizon=2),report=compare_equations(predicted_fields,native_fields,max_elements=20_000_000),
            conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,
            readers=['step3.project.image','step3.joint0.imageV','native.final-image'],
            evidence=dict(role='Sealed numerical commit/I/V forecast with separately executed original component targets; RGB observed',terminal_status='unassessed',new_fitted_parameters=0))
        ids.append(record['fingerprint'])
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to={**context,'clock':'step1.scheduler'}):raise ValueError('catalog supplied applicability differs')
    for key,row in cases.items():
        if row['spec']['kind']!='band':continue
        spec=row['spec'];r=spec['receiver'];p=frozen(key)
        if p['producer']!=spec['producer'] or not same_record(p['values'],predictions[spec['producer']]['values']):raise ValueError('band producer differs from seal')
        a=load(root,row['installed_band']);v=load(root,cases[spec['producer']]['actual_values'])[:,512:768]
        ports[key]=dict(actual_producer=spec['producer'],producer_band=measure(a,v),
            sealed_value_band=measure(a,load(root,p['values'])[:,512:768]),
            own_project_image_exact=row['project_image']==cases[r+'.latent-z0n0']['project_image'],
            own_project_text_exact=row['project_text']==cases[r+'.latent-z0n0']['project_text'],
            outside_band_values_exact=row['outside_band_values']['before']==row['outside_band_values']['after'],
            value_layout_exact=row['outside_band_values']['layout_before']==row['outside_band_values']['layout_after'],
            protected_query_key_text_values=row['protected_ports'])
    old=read(source/'cases.json');inventory=read(source/'files.json')
    for key,oldkey in plan['historical_controls'].items():
        prior=old[oldkey]
        if sha(source/prior['image']['path'])!=inventory[prior['image']['path']]:raise ValueError('historical PNG changed')
        load(source,prior['latent'])
        controls[key]=dict(png_exact=cases[key]['png']['sha256']==prior['image']['sha256'],latent_exact=cases[key]['latent']['sha256']==prior['latent']['sha256'])
    for r in operands:
        base=r+'.latent-z0n0';b=rgb(base);masks=fresh['semantic_masks'](b);semantics[r]={};factorials[r]={};ordering[r]={}
        for kind in ('latent','band'):
            keys={'00':base,**{z+ n:f'{r}.{kind}-z{z}n{n}' for z,n in (('0','1'),('1','0'),('1','1'))}}
            sem={corner:{name:region(b,rgb(k),mask) for name,mask in masks.items()} for corner,k in keys.items()};semantics[r][kind]=sem
            fields={'rgb':lambda k:rgb(k),'final_latent':lambda k:load(root,cases[k]['latent']),
                'joint0_image':lambda k:load(root,cases[k]['joint0_image'])}
            if kind=='latent':fields.update(commit=lambda k:load(root,cases[k]['component']['latent']),value_band=lambda k:load(root,cases[k]['actual_values'])[:,512:768])
            factorials[r][kind]={f:factorial({corner:get(k) for corner,k in keys.items()}) for f,get in fields.items()}
            z,n=[sem[x]['green']['signed_green_score'] for x in ('10','01')];fr=factorials[r][kind]['rgb']
            ordering[r][kind]=dict(carry_rgb_projection=fr['carry_projection'],flow_rgb_projection=fr['flow_projection'],
                predicted_carry_rgb_larger=fr['carry_projection']>fr['flow_projection'] if fr['carry_projection'] is not None else None,
                carry_signed_green=z,flow_signed_green=n,predicted_carry_green_larger=None if z is None or n is None else abs(z)>abs(n),
                predicted_negative_carry=None if z is None else z<0,predicted_negative_flow=None if n is None else n<0,
                role='Per-recipient frozen exploratory prediction outcomes; no scientific stopping gate.')
    result=dict(schema='mdb-flux-commit-factorial-audit-v1',source_job=summary['job_id'],mechanics_status='verified',instrument_status='complete-saved-port-comparisons',
        trend_status='observed',terminal_status='unassessed',counts=summary['counts'],scope=plan['scope'],native_manifest_sha256=sha(root/'files.json'),seal_sha256=tx['seal_sha256'],
        exact_forecast_archives=exact,forecast_archives=48,unique_forecast_archives=len({p[f]['sha256'] for p in predictions.values() for f in ('latent','image','values')}),
        saved_scalar_profile='cuda-bf16-tensor-scalar-v1; qualified by post-hoc source job-e9c202b9e714; original FP32-scalar audit retained',
        compared_scalar_positions=scalar_positions,forecast_errors=errors,band_ports=ports,historical_controls=controls,
        factorials=factorials,semantic_contrasts=semantics,frozen_ordering=ordering,rounding=rounding,semantic_hypotheses=plan['semantic_hypotheses'],new_model_calls=0,
        limitations=['Shared seed241/four known contexts; natural00/11 commits are observed replay controls,01/10 are new crosses, not independent samples.',
            'N2 is transplanted, not regenerated coherently after replacing Z2. No earlier attention/MLP writer identified.',
            'Original parameters and shared CUDA primitives predict commit/I/V. Full native image continuation is observed, not numerically forecast.',
            'Both consumers share natural00 RGB; they have different intervention supports and11 reference contrasts. The selected value band need not explain the whole latent effect.',
            'Baseline color masks are supplied instruments; no universal object or color symbol claim.'])
    save(out/'audit.json',result);save(out/'catalog/provenance.json',dict(records=len(ids),fingerprints=ids,matching_queries=len(ids),wrong_clock_abstentions=len(ids)))
    for key,row in cases.items():shutil.copyfile(root/row['png']['path'],out/row['png']['path'])
    for name in ('cases.json','predictions.json','source-operands.json','source-cuts.json','native-contract.json','summary.json','plan.json','timeline.jsonl'):
        shutil.copyfile(root/name,out/('source-'+name))
    shutil.copytree(root/'transaction',out/'transaction')
    for name in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/name,out/name)
    save(out/'external-evidence.json',dict(host='beast',roots=dict(native=str(root),reader=str(source)),manifests=dict(native=read(root/'files.json'),reader=inventory)))
    shutil.copyfile('retrieve_reader_evidence.py',out/'retrieve_evidence.py');save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),source_job=summary['job_id'],seconds=time.monotonic()-started,
        records=len(ids),mechanics_status='verified',instrument_status=result['instrument_status'],terminal_status='unassessed',new_execution_counts=dict(full_models=0,native_primitives=0,learned_forwards=0,saved_elementwise_commit_evaluations=16,images=0)))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        here=Path(__file__).resolve().parent;helper=runpy.run_path(str(here/'flux_reader_attention_audit.py'))
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        extras=[(here/n,n) for n in ('flux_reentry_reference.py','flux_reader_fresh.py','flux_reader_fresh_audit.py')]
        print(json.dumps(helper['submit'](a.run.resolve()) if a.submit else helper['stage'](a.run.resolve(),a.study.resolve(),worker_path=Path(__file__),extra_files=extras,audit_kind='commit-factorial',logical_operations=64)))
