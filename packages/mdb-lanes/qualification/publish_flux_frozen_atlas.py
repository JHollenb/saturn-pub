"""Public atlas occurrences with frozen composition hypotheses and their misses."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def publish(run, audit_path, output):
    from mdb_lanes.computation_atlas import ComputationAtlas
    from mdb_lanes.equation_comparison import EquationCatalog, compare_equations
    from mdb_lanes.tensor_archive import load_tensor_archive
    data=run/'extracted/frozen-composition-result';audit=read(audit_path)
    config,summary,source,binding=[read(data/n) for n in ('plan.json','summary.json','source.json','binding.json')]
    assert audit['mechanics_status']=='verified' and audit['request_sha256']==summary['request_sha256']
    output.mkdir(parents=True,exist_ok=False)
    catalog,atlas=EquationCatalog(output/'catalog'),ComputationAtlas(output/'atlas')
    ref={'path':str(audit_path),'sha256':sha(audit_path)};req={'path':str(run/'request.json'),'sha256':sha(run/'request.json')}
    forecasts,native=read(data/'forecasts.json'),read(data/'native-fields.json')
    rows={r['id']:r for r in read(data/'rows.json')};frontier=read(data/'frontier.json')['common_project_frontier']
    records,contexts,effects,futures,counters=[],{},{},{},{}
    for role,pred in forecasts.items():
        context={'model_identity':source['model_identity'],'source_identity':read(run/'request.json')['source_hashes'],
            'execution':{'backend':'diffusers',**source['execution']},'clock':{'step':3,'phase':'project'},'seed':config['seed'],'program_role':role,
            'entry_cut':rows['native.'+role]['actions'][0]['post_act'],'receiving_state':frontier,
            'horizon':{'equation':'step3/conditioning-to-initialT','consumer':'native denoiser/readout/scheduler/VAE'}}
        report=compare_equations({'initial.T':load_tensor_archive(data,pred)},{'initial.T':load_tensor_archive(data,native[role])},max_elements=2_000_000)
        record=catalog.record(law_id='flux.context-projector.linear.reference.v1',route_id='flux.conditioning-to-text.step3.v1',alias='frozen-composition/source-projection',
            binding={'model_identity':source['model_identity'],'source_identity':context['source_identity'],'execution':context['execution'],'input_identity':context['entry_cut'],
                'horizon':context['horizon'],'frontier':{'kind':'source-only-composed-conditioning','receiving_state':frontier,'role':role,'source_encoding':binding['composed'][role],
                    'source_visibility':'Three observed new encodings and fixed historical interaction; new joint encoding unavailable until forecast seal'}},
            report=report,conditions=[{'field':k,'op':'eq','value':v} for k,v in context.items()],tested_context=context,
            readers=['initial.text.step3','rgb.supplied-eye-window'],evidence={'audit':ref,'request':req,'frozen_rule':config['rules'][role],
                'hypotheses':config['hypotheses'],'quarantine':audit['quarantine'],'encoding_forecast_error':audit['encoding_to_new_joint'][role],
                'projection_error':audit['full_projection_equations'][role],'native_future':audit['observations']['native.'+role],
                'predicted_T_future':audit['prediction_pairs'][role],'semantic_scope':'Manual windows/diagnostic hue bands/unblinded review; no automatic segmentation or complete semantic prediction'})
        key=record['fingerprint'];records.append(record);contexts[key]=context
        effects[key]=[{'consumer_id':'rgb.supplied-eye-window','contrast_id':'additive-error-minus-candidate-error-to-joint','value':audit['forecast_scores']['eye_distance']['additive']-audit['forecast_scores']['eye_distance'][role],'units':'RGB MAE improvement on0..255scale','provenance':[ref]}]
        futures[key]={'status':'predicted-initialT-native-future-observed','row_id':'predicted.'+role,'provenance':[ref],'scope':'Downstreamimage native; newjointnot supplied to candidate. Sourceencoder/composedE and basehistory observed.'}
        counters[key]=[{'claim_id':'fixed-old-interaction-improves-new-context','status':'not-rejected-in-this-context' if audit['forecast_scores']['plus_old_better_than_additive'] else 'counterexample-retained','provenance':[ref],
            'details':audit['forecast_scores']}]
    imported=atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance={'run_id':summary['job_id'],'artifacts':[ref,req]},signed_effects=effects,continuation=futures,counterexamples=counters)
    queries=[]
    for key,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1
        assert not atlas.find(applicable_to={**context,'clock':{'step':2,'phase':'project'}})
        assert not atlas.find(applicable_to={**context,'receiving_state':'different-history'})
        queries.append({'equation_id':key,'atlas_id':hit[0]['fingerprint'],'exact_hits':1,'changed_clock_hits':0,'changed_receiving_hits':0})
    for row in imported: assert atlas.show(row['fingerprint'])==row
    result={'schema':'mdb-frozen-composition-atlas-v1','source_job':summary['job_id'],'audit':ref,'records':len(records),'selected_occurrences':[r['fingerprint'] for r in records],'atlas_ids':[r['fingerprint'] for r in imported],'queries':queries,'new_model_calls':0,'scope':'Three complete projection occurrences and source-only semantic forecast outcomes, including failures. No automatic naming or generalization claim.'}
    with (output/'provenance.json').open('x') as f: json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'records':len(records),'exact_queries':len(queries),'negative_queries':2*len(queries),'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
