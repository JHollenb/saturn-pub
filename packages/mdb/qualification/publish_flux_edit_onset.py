"""Clock-qualified public equation occurrences with shared repeated-read futures."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def publish(run,audit_path,output):
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog,compare_equations
    from mdb.tensor_archive import load_tensor_archive
    data=run/'extracted/edit-onset-result';audit=read(audit_path)
    config,summary,binding,source=[read(data/n) for n in ('plan.json','summary.json','binding.json','source.json')]
    assert audit['mechanics_status']=='verified' and audit['request_sha256']==summary['request_sha256']
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas')
    ref={'path':str(audit_path),'sha256':sha(audit_path)};req={'path':str(run/'request.json'),'sha256':sha(run/'request.json')}
    predicted=load_tensor_archive(data,binding['predicted_additive_T']);records=[];contexts={};effects={};futures={};counters={}
    for row in read(data/'rows.json'):
        if row['role']!='additive' or row['predicted']:continue
        onset=row['onset'];entries={int(v['step']):v['cut'] for v in row['trace']}
        for observed in row['project_reads']:
            step=observed['step'];context={'model_identity':source['model_identity'],'execution':{'backend':'diffusers',**source['execution']},
                'source_identity':read(run/'request.json')['source_hashes'],'clock':{'step':step,'phase':'project'},'onset':onset,'program_role':'additive','seed':config['seed'],
                'entry_cut':entries[step],'receiving_state':entries[step],'horizon':{'equation':'conditioning-to-initialT','consumer':'RepeatedpredictedT/native remainingdenoiser/scheduler/VAE'}}
            comparison=compare_equations({'initial.T':predicted},{'initial.T':load_tensor_archive(data,observed['native_text'])},max_elements=2_000_000)
            record=catalog.record(law_id='flux.context-projector.linear.reference.v1',route_id='flux.conditioning-to-text.clocked.v1',alias='frozen-composition/repeated-project-read',
                binding={'model_identity':context['model_identity'],'execution':context['execution'],'source_identity':context['source_identity'],'input_identity':entries[step],'horizon':context['horizon'],
                    'frontier':{'kind':'observed-architectural-project-entry','onset':onset,'clock':context['clock'],'receiving_state':entries[step],
                        'source_prediction_seal':config['source_forecast_seal_sha256'],'scope':'SamepreviouslycalculatedT; no fit to newclocks. E/basehistory/nativesuffix observed. Sharedstackedfuture is not single-read causal attribution.'}},
                report=comparison,conditions=[{'field':k,'op':'eq','value':v} for k,v in context.items()],tested_context=context,
                readers=['initial.text.clocked','rgb.fixed-eye-window'],evidence={'audit':ref,'request':req,'native_read':audit['projection_reads'][row['id']+'/project'+str(step)],
                    'repeated_prediction_future':audit['repeated_prediction_pairs'][str(onset)],'target_collateral':audit['observations'][row['id']],
                    'timing_comparisons':{k:v[str(onset)] for k,v in audit['timing'].items()},'observer_scope':config['observers']['scope'],
                    'forecast_scope':'Prospectiveonset/repeated-read forecast onknownprompt. Not rawtextRGB or whollypredicteddenoiser.'})
            key=record['fingerprint'];records.append(record);contexts[key]=context
            candidate=audit['observations'][row['id']]['eyes']['to_matched_joint']['mae']
            baseline=audit['observations']['onset'+str(onset)+'.base']['eyes']['to_matched_joint']['mae']
            effects[key]=[{'consumer_id':'rgb.fixed-eye-window','contrast_id':'base-error-minus-additive-error-to-matched-joint','value':baseline-candidate,'units':'RGBMAEimprovement0..255','provenance':[ref]}]
            futures[key]={'status':'stacked-predictedT-native-future-observed','row_id':'onset'+str(onset)+'.stacked_additive','provenance':[ref],
                'shared_across_project_records_at_onset':onset,'scope':'AllremainingprojectTreads replaced. Multipleequationoccurrences shareonefuture; single-read marginaleffect not measured.'}
            counters[key]=[{'claim_id':'one-isolated-write-or-global-error-establishes-selective-edit','status':'not-established','provenance':[ref],
                'details':'Context/clock/receivinghistory and native shape/colorcontrols remainexplicit; target andcollateral measuredseparately. Geometry remainsunblinded and observerwindowsmaymissmovedfeatures.'}]
    assert len(records)==9
    imported=atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance={'run_id':summary['job_id'],'artifacts':[ref,req]},signed_effects=effects,continuation=futures,counterexamples=counters)
    queries=[]
    for key,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1
        assert not atlas.find(applicable_to={**context,'clock':{'step':(context['clock']['step']+1)%4,'phase':'project'}})
        assert not atlas.find(applicable_to={**context,'receiving_state':'different-history'})
        queries.append({'equation_id':key,'atlas_id':hit[0]['fingerprint'],'exact_hits':1,'changed_clock_hits':0,'changed_receiving_hits':0})
    for r in imported:assert atlas.show(r['fingerprint'])==r
    provenance={'schema':'mdb-edit-onset-atlas-v1','source_job':summary['job_id'],'audit':ref,'records':len(records),'selected_occurrences':[r['fingerprint'] for r in records],
        'atlas_ids':[r['fingerprint'] for r in imported],'queries':queries,'new_model_calls':0,'scope':'Nineclockedfullprojectionoccurrences; three sharedstackedpredictionfutures, not nineindependentRGBforecasts. Explicitcuration, no automaticsemanticnaming.'}
    with (output/'provenance.json').open('x') as f:json.dump(provenance,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'records':9,'exact_queries':9,'negative_queries':18,'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args();publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
