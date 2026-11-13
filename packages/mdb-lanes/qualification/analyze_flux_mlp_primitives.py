"""Independent saved-array audit of an original-source two-MLP diagnostic.

Native intermediate operands are explicit diagnostic inputs. No target-assisted
quantity is counted as a prospective prediction or as a causal image edit.
"""
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from mdb_lanes.response_closure import audit_response, measure_error
from mdb_lanes.tensor_archive import load_tensor_archive


def signed_error(components, total):
    """Output-space error powers with every signed pairwise cross moment."""
    values={name:np.asarray(value,np.float64) for name,value in components.items()}
    total=np.asarray(total,np.float64)
    if not values or any(value.shape!=total.shape or not np.isfinite(value).all() for value in values.values()) or not total.size or not np.isfinite(total).all():
        raise ValueError('same-shaped finite error terms required')
    names=list(values)
    powers={name:float(np.mean(value*value)) for name,value in values.items()}
    cross={a+' × '+b:float(2*np.mean(values[a]*values[b])) for i,a in enumerate(names) for b in names[i+1:]}
    mse=float(np.mean(total*total))
    return dict(component_mse=powers,twice_signed_cross_moments=cross,total_mse=mse,
        summed_power=sum(powers.values())+sum(cross.values()),
        identity_residual_rms=float(np.sqrt(np.mean((total-sum(values.values()))**2))),
        scope='Ordered surrogate substitutions in a common output space, not unique positive causal shares. Signed cancellation retained.')


def substitutions(fields,native):
    C,Z,A=[np.asarray(fields[name],np.float64) for name in ('C','after_native_paired','after_native_activation')]
    Y=np.asarray(native,np.float64)
    return dict(first_affine_propagation=C-Z,nonlinear_propagation=Z-A,local_output_affine=A-Y),C-Y


def analyze(collection,output):
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    here=Path(__file__).resolve().parent;helper=lambda name:runpy.run_path(str(here/name))
    custody=helper('flux_family_custody.py')['verify'](collection);request=read(collection/'request.json');data=collection/'extracted/binding-minimal-result';c=request['config'];summary=read(data/'summary.json');rows=read(data/'rows.json')
    if read(data/'plan.json')!=c or read(data/'request.json')!=request or summary!=read(collection/'report.json')['report']:raise ValueError('primitive request/report differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('primitive payload changed')
    wheel_hashes=read(collection/'payload/payload-seal.json')['wheels']
    for name,digest in wheel_hashes.items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('primitive installed wheel changed')
    expected=dict(mlp=2,linear=4,silu=2,product=2,full_models=0,blocks=0,attention=0,images=0)
    if summary['counts']!=expected or set(rows)!={'image','text'}:raise ValueError('native primitive census differs')
    abi=read(data/'native-contract.json')
    if abi['original_module_source_sha256']!=c['original_module_source_sha256'] or type(abi['bf16_reduced_precision_reduction']) is not bool or not abi['bf16_reduced_precision_reduction'] or abi['tf32'] or not abi['deterministic']:raise ValueError('native numerical ABI differs')
    source=Path(c['source_collection']);source_record=read(source/'remote-custody.json');source_data=source/'extracted/binding-minimal-result';source_audit=Path(c['source_audit'])
    if sha(source/'remote-custody.json')!=c['source_record_sha256'] or sha(source_audit)!=c['source_audit_sha256'] or read(source_audit)['custody']!=source_record['verification']:raise ValueError('source scientific custody differs')
    binding=read(data/'source-binding.json')
    if binding['source_custody']!=source_record['verification'] or binding['record_sha256']!=c['source_record_sha256']:raise ValueError('source cohort differs')
    def source_json(name):
        p=source_data/name
        if sha(p)!=source_record['files'][name]['sha256']:raise ValueError('source metadata changed')
        return read(p)
    plan,native,forecasts,baselines,manifest=[source_json(name) for name in ('plan.json','native.json','forecasts.json','baselines.json','parameter-artifacts.json')]
    if manifest!=read(data/'parameter-artifacts.json') or native[c['case']]['entry']!=c['native_entry'] or native[c['case']]['exit']!=c['native_exit']:raise ValueError('source ports/checkpoint differ')
    previous=Path(c['previous_mlp_root'])
    if sha(previous/'conditional.json')!=c['previous_mlp_conditional_sha256'] or sha(previous/'audit-v1.json')!=c['previous_mlp_audit_sha256'] or read(previous/'audit-v1.json')['mechanics_status']!='verified':raise ValueError('prior conditional evidence differs')
    old_C=read(previous/'conditional.json');load_bits=helper('flux_attention_anchor.py')['load_bits']
    core=runpy.run_path(str(collection/'payload/flux_mlp_primitive_cut.py'));reference=runpy.run_path(str(collection/'payload/flux_reentry_reference.py'));law=runpy.run_path(str(collection/'payload/flux_receiver_norm_prediction.py'));mlp=runpy.run_path(str(collection/'payload/flux_hybrid_mlp_cut.py'))['mlp']
    bank=helper('flux_source_receiver.py')['RecipeBank'](manifest,c['model_path']);entries={port:item for item in manifest['artifacts'] for port in item['parameter_ports']};checked_arrays=set();reports={};checks={};calculations=0
    def source_array(root,item):
        name=(root/item['path']).relative_to(source_data).as_posix()
        if source_record['files'][name]['sha256']!=item['sha256']:raise ValueError('source array outside verified cohort')
        return load_tensor_archive(root,item)
    def array(item):
        checked_arrays.add(item['path']);return load_tensor_archive(data,item)
    native_names={'paired','silu','activation','mlp'};groups=dict(chained=native_names,local=native_names|{'product'},substitutions={'C','after_native_paired','after_native_activation'})
    for role,prefix,token_rows in [('image','ff',1024),('text','ff_context',512)]:
        row=rows[role]
        if set(row['native'])!=native_names or set(row['math'])!=set(groups) or any(set(row['math'][group])!=names for group,names in groups.items()):raise ValueError('primitive field census differs')
        if row!=read(data/('row-'+role+'.json')) or row['source_input']!=native[c['case']]['stages'][role+'.mlp_input'] or row['source_output']!=native[c['case']]['stages'][role+'.mlp']:raise ValueError('per-role/source join differs')
        geom=row['layouts']
        if geom['input']['shape']!=[1,token_rows,3072] or geom['chunks']['u']['shape']!=[1,token_rows,9216] or geom['chunks']['v']['storage_offset']!=9216 or geom['chunks']['u']['stride']!=[token_rows*18432,18432,1] or geom['chunks']['u']['contiguous']:raise ValueError('full native geometry/chunk layout differs')
        x=source_array(source_data,row['source_input']);stages={field:source_array(source_data/'source/prefixes',plan['anchors'][c['anchor']]['stages'][role+'.'+field]) for field in ('mlp_input','mlp_paired','mlp_silu','mlp_activation','mlp')}
        N={name:array(item) for name,item in row['native'].items()};N0={name:stages['mlp_'+name if name!='mlp' else 'mlp'] for name in native_names};parity=bool(np.array_equal(N['mlp'],source_array(source_data,row['source_output'])))
        checks[role]=dict(native_output_parity=parity,worker_parity_agrees=parity==row['native_output_parity'],owned_reconstruction={},saved_math={},previous_conditional={})
        weights={}
        for suffix in ('linear_in.weight','linear_out.weight'):
            port=prefix+'.'+suffix;item=entries[port];weights[port]=bank.load(dict(path=item['path'],sha256=item['sha256'],descriptor=dict(sha256=item['source']['payload_sha256']),parameter_tensor=item['source']['tensor'],stored_shape=item['source']['shape']))
        math=core['mathematical_fields'](x,stages,N,weights,role,reference,law);calculations+=1
        math0=core['mathematical_fields'](stages['mlp_input'],stages,N0,weights,role,reference,law);calculations+=1
        for group,fields in math.items():
            for name,value in fields.items():checks[role]['saved_math'][group+'.'+name]=bool(np.array_equal(value,array(row['math'][group][name])))
        checks[role]['previous_conditional']['target']=bool(np.array_equal(math['substitutions']['C'],load_bits(previous,old_C['cases'][c['case']][role])))
        checks[role]['previous_conditional']['anchor']=bool(np.array_equal(math0['substitutions']['C'],load_bits(previous,old_C['anchors'][c['anchor']][role])))
        full_stages={role+'.'+name:value for name,value in stages.items()};owned={}
        for name,by_mode in [('P1',forecasts[c['case']]),('P0',baselines[c['anchor']])]:
            for mode in ('control','hybrid'):
                fields=by_mode[mode]['fields'][role];P=source_array(source_data,fields['mlp']);actual=mlp(source_array(source_data,fields['mlp_input']),full_stages,weights,role,reference,law);calculations+=1;key=name+'.'+mode
                checks[role]['owned_reconstruction'][key]=bool(np.array_equal(actual,P)) and row['owned_reconstruction'][key] is True;owned[key]=P
        del weights
        local={};chained={}
        for group,dest in [('local',local),('chained',chained)]:
            for name,value in math[group].items():
                field='activation' if name=='product' else name
                dest[name]=dict(endpoint=measure_error(value,N[field]),paired_response=audit_response(N0[field],N[field],math0[group][name],value))
        terms,error=substitutions(math['substitutions'],N['mlp']);terms0,error0=substitutions(math0['substitutions'],N0['mlp'])
        decomposition=dict(endpoint=signed_error(terms,error),paired_response=signed_error({name:terms[name]-terms0[name] for name in terms},error-error0))
        C,C0=[values['substitutions']['C'].astype(float) for values in (math,math0)];Y,Y0=N['mlp'].astype(float),N0['mlp'].astype(float);inherited={}
        for mode in ('control','hybrid'):
            P,P0=[owned[name+'.'+mode].astype(float) for name in ('P1','P0')]
            inherited[mode]=dict(endpoint=signed_error(dict(inherited_operand=P-C,conditional_law=C-Y),P-Y),paired_response=signed_error(dict(inherited_operand=(P-P0)-(C-C0),conditional_law=(C-C0)-(Y-Y0)),(P-P0)-(Y-Y0)))
        reports[role]=dict(local=local,chained=chained,ordered_output_space=decomposition,inherited_vs_conditional=inherited,anchor_conditional_endpoint=measure_error(C0,Y0),conditional_response=audit_response(Y0,Y,C0,C),layouts=geom)
        print('MDB_MLP_PRIMITIVE_AUDIT '+json.dumps(dict(role=role,checks=checks[role],local_response_relative={name:value['paired_response']['reference_response']['relative_rms'] for name,value in local.items()},decomposition=decomposition),sort_keys=True),flush=True)
        del math,math0,N,N0,stages,owned,full_stages,terms,terms0,x
    if {p.relative_to(data).as_posix() for p in (data/'arrays').glob('*.npy')}!=checked_arrays or len(checked_arrays)!=summary['unique_arrays']:raise ValueError('primitive archive/reference census differs')
    # Only exact instrument checks have boolean authority. Numerical measurements
    # remain observations regardless of size, direction or cancellation.
    instrument=[]
    for role,check in checks.items():
        instrument.extend([(role+'.native_output_parity',check['native_output_parity']),(role+'.worker_parity_agrees',check['worker_parity_agrees'])])
        for group in ('owned_reconstruction','saved_math','previous_conditional'):instrument.extend((role+'.'+group+'.'+key,value) for key,value in check[group].items())
    issues=[name for name,value in instrument if value is not True]
    result=dict(schema='mdb-mlp-primitives-audit-v1',custody=custody,mechanics_status='verified',instrument_status='qualified' if not issues else 'instrument-broken',instrument_issues=issues,checks=checks,reports=reports,counts=summary['counts'],source_custody=source_record['verification'],source_binding=binding,native_contract=abi,unique_arrays=len(checked_arrays),independent_math_evaluations=calculations,parameter_reads=bank.reads,auditor_source_sha256=sha(Path(__file__)),terminal_status='unassessed',scope='Two known-input native MLP calls. Same-input local primitive comparisons and ordered conditional propagation, independently recalculated from pinned equations and original parameters. No new semantic edit/image, general accuracy floor, or prospective prediction claim.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return result
