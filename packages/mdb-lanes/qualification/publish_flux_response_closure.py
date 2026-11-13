"""Index response limits and exact incremental execution without copying tensors."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb_lanes.computation_atlas import ComputationAtlas
from mdb_lanes.equation_comparison import EquationCatalog,compare_equations
from mdb_lanes.edit_search import file_digest as sha
from mdb_lanes.tensor_archive import load_tensor_archive


def publish(study,output):
    read=lambda p:json.loads(p.read_text()); here=Path(__file__).resolve().parent
    audit=read(study/'audit-v1.json'); incremental=read(study/'incremental/audit-v1.json')
    if audit['mechanics_status']!='verified' or incremental['mechanics_status']!='verified': raise ValueError('verified independent audits required')
    collection=study/'collection-01'; data=collection/'extracted/binding-minimal-result'; config=read(data/'plan.json'); forecasts=read(data/'forecasts.json'); native=read(data/'native.json')
    inc=study/'incremental/collection-01'; incdata=inc/'extracted/binding-minimal-result'; incsummary=read(incdata/'summary.json')
    output.mkdir(parents=True,exist_ok=False); catalog=EquationCatalog(output/'catalog'); atlas=ComputationAtlas(output/'atlas')
    refs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (study/'audit-v1.json',study/'incremental/audit-v1.json',collection/'request.json',inc/'request.json')]
    slots=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']; contexts={}; records=[]; limits={}; futures={}; effects={}
    for spec in config['cases']:
        name=spec['id']; actual={role:load_tensor_archive(data,native[name]['fields'][role]) for role in ('image','text')}
        for mode,law in [('absolute','flux.joint.changed-text.response.v1'),('anchored','flux.joint.receiver-anchor.response.v1'),('incremental','flux.joint.native-prefix-reuse.v1')]:
            source=inc if mode=='incremental' else collection; req=read(source/'request.json'); receipt=read((incdata if mode=='incremental' else data)/'receipts.json')[0]
            execution=dict(backend='diffusers',**receipt['execution']); model=receipt['model_identity']; sources={**req['source_hashes'],**{'wheel:'+k:v for k,v in read(source/'payload/payload-seal.json')['wheels'].items()}}
            if mode=='incremental':
                row=next(r for r in incsummary['reports'] if r['case']==name and r['mode']=='incremental')
                calculated={role:slots(incdata/'store',row['exit'],role) for role in actual}
                frontier=dict(kind='native-incremental-known-case',scope='Observed unchanged native image prefix plus full affected native text/attention/MLP execution; archived targets known. Actual prefix skips, not independent mathematical forecasting.')
            else:
                calculated={role:load_tensor_archive(data,forecasts[name]['fields'][role][mode]) for role in actual}
                frontier=dict(kind='sealed-new-internal-state',scope='Complete mathematical forecast before new target; observed parent and checkpoint. Anchored method additionally uses declared historical half-dose native exit.')
            report=compare_equations(calculated,actual,max_elements=5_000_000)
            horizon=dict(equation='complete joint1 I/T exit',consumer='next typed joint2 state; no RGB continuation in this assay')
            binding=dict(model_identity=model,source_identity=sources,execution=execution,input_identity=native[name]['entry'],frontier=frontier,horizon=horizon)
            context=dict(**binding,checkpoint_sha256=config['checkpoint_sha256'],clock=dict(step=2,joint=1),receiver=spec['receiver'],dose=spec['dose'],method=mode)
            record=catalog.record(law_id=law,route_id='flux.step2.joint1.working-text.receiver.v1',alias='cmp.flux.closure.'+name+'.'+mode,binding=binding,report=report,
                conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,readers=['native.joint1.image.exit','native.joint1.text.exit'],
                evidence=dict(artifacts=refs,measured=audit['local'][name],receiver_by_dose=audit['factorial'],independent_forecast_seal=audit['transaction']['seal_sha256'] if mode!='incremental' else None,
                    native_target_count=2,method=mode,incremental=incremental['counts'] if mode=='incremental' else None))
            fp=record['fingerprint']; records.append(record); contexts[fp]=context; limits[fp]=[]
            if mode!='incremental':
                limits[fp].append(dict(claim_id='fine-receiver-response-accuracy',status='limitation-retained',measurements=audit['factorial']['image']['absolute']['contrasts']['receiver_by_dose'],
                    scope='125.34percent image mixed-effect relativeRMS; zero-interaction foil100percent. Numerical limitation; no missing-semantic-variable inference.',provenance=refs))
            else:
                limits[fp].append(dict(claim_id='incremental-throughput',status='limitation-retained',scope='Actual native work skipped; fixed-order n2 calls do not establish speedup; affected native computation remains.',provenance=refs))
            futures[fp]=dict(status='complete-state-observed-no-image-continuation',exit=native[name]['exit'],provenance=refs)
            effects[fp]=[]
            for role in actual:
                N0=load_tensor_archive(data,forecasts[name]['fields'][role]['native_anchor'])
                effects[fp].append(dict(consumer_id='native.joint1.'+role+'.exit',contrast_id='3/8-minus-same-receiver-half',units='signed mean tensor value; not semantic score',value=float(np.mean(actual[role].astype(float)-N0.astype(float))),provenance=refs))
    atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance=dict(run_id='response-closure-and-incremental',artifacts=refs),signed_effects=effects,continuation=futures,counterexamples=limits)
    for fp,context in contexts.items():
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}): raise ValueError('applicability query mismatch')
    result=dict(schema='mdb-response-closure-publication-v1',programs=3,equation_occurrences=6,new_native_target_states=2,exact_queries=6,negative_queries=12,
        limits_retained=6,new_model_calls=0,tensor_bytes_copied=0,scope='Three supplied generic methods on the same two native targets, not six independent experiments. Exact-context applicability supplied, not learned. No image/semantic continuation or automatic family discovery.')
    with (output/'provenance.json').open('x') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('study',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); print(json.dumps(publish(a.study.resolve(),a.output.resolve()),sort_keys=True))
