"""Index a single response program across observed numerical/causal cases."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb_lanes.computation_atlas import ComputationAtlas
from mdb_lanes.equation_comparison import EquationCatalog,compare_equations
from mdb_lanes.edit_search import file_digest


def publish(view,audit,review,output):
    read=lambda p:json.loads(p.read_text());here=Path(__file__).resolve().parent;data=view/'extracted/binding-minimal-result'
    a=read(audit);inspection=read(review);provenance=read(view/'analysis-view.json')
    if a['mechanics_status']!='verified':raise ValueError('independent numerical audit required')
    rows={(r['id'],r['mode']):r for r in read(data/'rows.json')};forecasts=read(data/'forecasts.json');request=read(view/'request.json');config=read(data/'plan.json')
    source_identity={**request['source_hashes'],**{'wheel:'+k:v for k,v in read(view/'payload/payload-seal.json')['wheels'].items()}}
    receipts=read(data/'receipts.json');execution=dict(backend='diffusers',**next(r['body']['execution']for r in receipts if r['body']['operation']=='continue'));model=receipts[0]['body']['model_identity']
    slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas')
    refs=[dict(path=str(p.resolve()),sha256=file_digest(p))for p in(audit,review,view/'analysis-view.json',view/'request.json')]
    records=[];contexts={};effects={};futures={};counterexamples={}
    def array(record):
        p=data/record['path']
        if file_digest(p)!=record['sha256']:raise ValueError('equation array changed')
        return np.load(p,allow_pickle=False)
    for spec in config['cases']:
        name=spec['id'];native,predicted=rows[name,'native'],rows[name,'predicted'];f=forecasts[name];target=native['target']
        observed={};calculated={}
        for role in('image','text'):
            for field in('attention','mlp_input','mlp','exit'):
                key=role+'.'+field;calculated[key]=array(f['fields'][role][field])
                observed[key]=slot(data/'store',target['exit'],role)if field=='exit'else array(target['fields'][field+'_'+role])
        report=compare_equations(calculated,observed,max_elements=20_000_000)
        horizon=dict(equation='complete joint1 attention/MLP/exits',consumer='native suffix and VAE')
        context=dict(model_identity=model,checkpoint_sha256=config['checkpoint_sha256'],source_identity=source_identity,execution=execution,horizon=horizon,
            receiving_state=target['entry'],clock=dict(step=2,joint=1),mutable_rows=spec['rows'],dose=spec['dose'],receiver=spec['receiver'],forecast_seal_sha256=a['transaction']['seal_sha256'])
        for mode,row in(('native',native),('predicted',predicted)):
            if inspection['rows'][name][mode]['png_sha256']!=row['png_sha256']or file_digest(data/row['png'])!=row['png_sha256']:raise ValueError('visual review differs')
        record=catalog.record(law_id='flux.joint.changed-text.response.v1',route_id='flux.step2.joint1.working-text.receiver.v1',alias='cmp.flux.response.'+name,
            binding=dict(model_identity=model,source_identity=source_identity,execution=execution,input_identity=target['entry'],
                horizon=horizon,
                frontier=dict(kind='independent-observed-prefix',scope='Observed I/T/time context and pinned parameters. CPU mathematical reductions; no target ports used. Native baseline clock scaffold supplies unchanged closure.')),
            report=report,conditions=[dict(field=k,op='eq',value=v)for k,v in context.items()],tested_context=context,
            readers=['native.joint1.image.exit','native.joint1.text.exit','native.RGB','rgb.binding.donor-projection'],evidence=dict(artifacts=refs,seal=a['transaction']['seal_sha256'],
                numerical_errors=a['local_errors'][name],image_future_errors=a['outcomes'][name],visual=inspection['rows'][name],
                source_assistance=a['source_assistance'],original_timeout_retained=True))
        fp=record['fingerprint'];records.append(record);contexts[fp]=context
        effect=a['outcomes'][name]['rgb_donor_direction']['native']['target_projection']-a['outcomes']['base']['rgb_donor_direction']['native']['target_projection']
        effects[fp]=[dict(consumer_id='rgb.binding.donor-projection',contrast_id='branch-minus-fixed-I0-T0-control',units='RGB vector projection, not semantic correctness',value=effect,provenance=refs)]
        futures[fp]=dict(status='predicted-write-and-native-RGB-observed',native_png_sha256=native['png_sha256'],predicted_png_sha256=predicted['png_sha256'],
            categorical_prediction_matched=a['outcomes'][name]['categorical_matched'],provenance=refs)
        counters=[]
        if a['outcomes'][name]['categorical_matched']is False:counters.append(dict(claim_id='frozen-categorical-continuation',status='counterexample-retained',predicted=spec['categorical'],observed=a['outcomes'][name]['colors']['native'],provenance=refs))
        if spec['categorical']is None and effect<=0:counters.append(dict(claim_id='frozen-positive-RGB-direction',status='counterexample-retained',value=effect,scope='Fixed I0/T0 comparison; receiver change is included for half.image1.',provenance=refs))
        if name=='half.image1':counters.append(dict(claim_id='small-endpoint-error-implies-resolved-receiver-effect',status='limitation-retained',
            measurements=a['receiving_state_contrast'],scope='Saved-only diagnostic of prospective outputs. No calibrated accuracy gate; receiver-effect error exceeds measured RGB effect despite small endpoint errors.',provenance=refs))
        counterexamples[fp]=counters
    atlas.ingest(output/'catalog',[r['fingerprint']for r in records],provenance=dict(run_id='response-program-two-attempts',artifacts=refs),signed_effects=effects,continuation=futures,counterexamples=counterexamples)
    for fp,context in contexts.items():
        assert len(atlas.find(applicable_to=context))==1
        assert not atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)})
        assert not atlas.find(applicable_to={**context,'receiving_state':'unknown'})
    result=dict(schema='mdb-response-atlas-publication-v1',program_id='flux.joint.changed-text.response.v1',equation_occurrences=len(records),exact_queries=len(records),negative_queries=2*len(records),
        frozen_counterexamples=sum(c['status']=='counterexample-retained'for values in counterexamples.values()for c in values),limitations=sum(c['status']=='limitation-retained'for values in counterexamples.values()for c in values),
        actual_attempt_cost=provenance['actual_attempt_cost'],new_model_calls=0,scope='One generic program, five cases at one clock/checkpoint/seed. Exact-context qualification; automatic family induction and prompt-to-edit semantic compilation remain open.')
    with(output/'provenance.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in('view','audit','review','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(publish(*[getattr(a,n).resolve()for n in('view','audit','review','output')]),sort_keys=True))
