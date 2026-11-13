"""Index prospectively tested numerical programs, retaining image counterexamples."""
import json
from pathlib import Path

from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog,compare_equations
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive


def publish(collection,audit_path,output):
    read=lambda p:json.loads(p.read_text());audit=read(audit_path);data=collection/'extracted/binding-minimal-result';request=read(collection/'request.json');c=request['config']
    if audit['mechanics_status']!='verified' or audit['custody']['request_sha256']!=request['request_sha256']:raise ValueError('verified prospective audit required')
    native=read(data/'native.json');forecasts=read(data/'forecasts.json');receipt=read(data/'receipts.json')[0]['body'];sources={**request['source_hashes'],**{'wheel:'+k:v for k,v in read(collection/'payload/payload-seal.json')['wheels'].items()}}
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas');records=[];contexts={};limits={}
    refs=[dict(path=str(p),sha256=sha(p)) for p in (audit_path,collection/'request.json',data/'parameter-artifacts.json',data/'rows.json')]
    for case in c['cases']:
        name=case['id'];actual={role:load_tensor_archive(data,native[name]['stages'][role+'.exit']) for role in ('image','text')}
        for mode,norm in c['programs'].items():
            calculated={role:load_tensor_archive(data,forecasts[name][mode]['fields'][role]['exit']) for role in ('image','text')}
            binding=dict(model_identity=receipt['model_identity'],source_identity=sources,input_identity=native[name]['entry'],execution=dict(backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',native_target=receipt['execution']),frontier=dict(kind='prospective-internal-state',scope='Complete P0/P1 sealed before fresh target; candidate native-suffix images sealed before target images. Historical calibration supplied.'),horizon=dict(equation='complete joint1 I/T exit',consumer='unchanged native suffix and VAE through denoising step4'))
            context=dict(**binding,clock=dict(step=2,joint=1),receiver=case['receiver'],row_doses=case['row_doses'],method=mode)
            report=dict(numerical=audit['numerical'][name][mode],outcome=audit['outcomes'][name][mode],factorial=audit['factorial'][mode],color_ordering=audit['color_ordering'])
            row=catalog.record(law_id='flux.joint1.receiver-law.anchored.'+norm+'.v1',route_id='flux.step2.joint1.contextual-write.v1',alias='cmp.flux.hybrid-future.'+name+'.'+mode,binding=binding,report=compare_equations(calculated,actual,max_elements=5_000_000),conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,readers=['image','text','native-suffix','VAE'],evidence=dict(artifacts=refs,measurements=report,source_native_states=4,additional_native_targets=0,selection='Two fixed supplied program IDs and exact applicability. Four new internal states, not semantic family induction.'))
            fp=row['fingerprint'];records.append(row);contexts[fp]=context;limits[fp]=[dict(claim_id='fresh-scope-and-counterexamples',status='limitation-retained',scope=c['hypotheses']['scope']+' All numerical/RGB/mixed errors retained; no universal semantic edit or speed claim.',measurements=report,provenance=refs)]
    atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance=dict(run_id=audit['custody']['job_id'],artifacts=refs),counterexamples=limits)
    for context in contexts.values():
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}):raise ValueError('exact applicability or abstention differs')
    result=dict(schema='mdb-hybrid-future-publication-v1',programs=2,equation_occurrences=8,source_native_states=4,new_native_calls_in_publication=0,exact_queries=8,changed_context_abstentions=16,tensor_bytes_copied=0,scope='Prospective numerical programs with observed image consumer outcomes. Supplied generic IDs/exact applicability, not automatically induced rules;8records from4native states.')
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result
