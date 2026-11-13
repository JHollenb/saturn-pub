"""Public equation/atlas records for frozen A/M writes and their T-conditioned futures."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def publish(run,audit_path,output):
    from mdb_lanes.computation_atlas import ComputationAtlas
    from mdb_lanes.equation_comparison import EquationCatalog,compare_equations
    from mdb_lanes.tensor_archive import load_tensor_archive
    data=run/'extracted/step1-writers-result';a=read(audit_path)
    summary,config,source,binding,numeric=[read(data/n) for n in ('summary.json','plan.json','source.json','numeric-binding.json','numeric-seal.json')]
    assert a['mechanics_status']=='verified' and a['request_sha256']==summary['request_sha256']
    validations=read(data/'write-validations.json')
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas')
    ref={'path':str(audit_path),'sha256':sha(audit_path)};req={'path':str(run/'request.json'),'sha256':sha(run/'request.json')}
    records=[];contexts={};effects={};futures={};counters={}
    from saturn_pub.store import LocalStore
    # Only the00 source return needs a typed tensor load; all other validation
    # returns are complete lossless arrays. No model is instantiated.
    store=LocalStore(data/'store')
    for bits,record in numeric['predicted_exits'].items():
        predicted=load_tensor_archive(data,record)
        observed=store.load(binding['post_joint']['base'],device='cpu').payload['image'].float().numpy() if bits=='00' else load_tensor_archive(data,validations[bits]['native_image'])
        execution={'backend':'diffusers',**source['execution']};context={'model_identity':source['model_identity'],'execution':execution,'clock':{'step':1,'phase':'joint','index':0},
            'selected_A_M':bits,'source_identity':read(run/'request.json')['source_hashes'],'receiving_state':binding['parents']['base'],'entry_cut':binding['parents']['base'],'seed':config['seed'],
            'horizon':{'equation':'joint0/image-exit-frozenwrites','consumer':'RetainbaseorshapeT throughreststep1; baseEfromstep2; nativesuffix/VAE'}}
        comparison=compare_equations({'image.exit':predicted},{'image.exit':observed},max_elements=4_000_000)
        record=catalog.record(law_id='flux.joint.frozen-sequential-write.reference.v1',route_id='flux.text-conditioned-image.step1.joint0.v1',alias='contextual-program/joint-write-occurrence',
            binding={'model_identity':source['model_identity'],'execution':execution,'input_identity':context['entry_cut'],'source_identity':context['source_identity'],'horizon':context['horizon'],
                'frontier':{'kind':'observed-R-A-M-gates','receiving_state':context['receiving_state'],'selected_A_M':bits,'writer':'B(B(R+B(ga*A))+B(gm*M))',
                    'source_ports':binding['ports'],'scope':'SourceA/M/gates observed; mathematical imageexitforecast, not sourceformationforecast. Tconditionedfuture pairs retain native T and unreadE difference erasedbefore nextEread.'}},
            report=comparison,conditions=[{'field':k,'op':'eq','value':v} for k,v in context.items()],tested_context=context,
            readers=['image.joint0.exit','rgb.supplied-eye-region'],evidence={'audit':ref,'request':req,'numeric_seal_sha256':summary['numeric_seal_sha256'],'equation_errors':a['full_exit_equations'][bits],
                'source_native_validation':{'image':'Completeactualimagearray; commonbaseparent or00source','text':'Source-boundworkerunchangedflag for mixedvalidations, not fulltextindependentaudit'},
                'T_conditioned_outcomes':{r:a['observations'][a['joint0_AMT_cube'][bits+'.'+r]] for r in ('base','shape')},
                'conditional_effects':a['conditional_write_effects'],'pulse_and_handoff_observations':a['observations'],'observer_scope':config['observers']['scope'],'coverage':a['coverage']})
        key=record['fingerprint'];records.append(record);contexts[key]=context
        effects[key]=[{'consumer_id':'rgb.supplied-eye-region','contrast_id':'shapeT-minus-baseT-at-fixed-imagewrite','value':value['eye_projection_delta'],'units':'Historicalpulse-shape RGBcontrastprojection','provenance':[ref]}
            for value in a['conditional_write_effects']['T'] if value['low'].startswith(bits+'.') and value['eye_projection_delta'] is not None]
        futures[key]={'status':'native-T-conditioned-futures-observed','row_ids':[a['joint0_AMT_cube'][bits+'.'+r] for r in ('base','shape')],'provenance':[ref],
            'scope':'Two dependent nativefutures atsamefrozenimageexit; downstreamneuralcalculation native. Current sourceports observed, imageexit mathematicallyforecast.'}
        counters[key]=[{'claim_id':'single-A-or-M-return-is-universal-shape-symbol','status':'not-established','provenance':[ref],
            'details':{'forecast_scores':a['forecast_scores'],'scope':'Fullfields andcontinuingT roles tested atoneclock/context; no minimal/necessary/globalwriter claim. Pulsepersistence andhandoff separate.'}}]
    imported=atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance={'run_id':summary['job_id'],'artifacts':[ref,req]},signed_effects=effects,continuation=futures,counterexamples=counters)
    queries=[]
    for key,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1
        assert not atlas.find(applicable_to={**context,'clock':{'step':2,'phase':'joint','index':0}})
        assert not atlas.find(applicable_to={**context,'receiving_state':'different-I-gates'})
        queries.append({'equation_id':key,'atlas_id':hit[0]['fingerprint'],'exact_hits':1,'changed_clock_hits':0,'changed_receiving_hits':0})
    for r in imported:assert atlas.show(r['fingerprint'])==r
    result={'schema':'mdb-step1-writers-atlas-v1','source_job':summary['job_id'],'audit':ref,'records':len(records),'selected_occurrences':[r['fingerprint'] for r in records],
        'atlas_ids':[r['fingerprint'] for r in imported],'queries':queries,'new_model_calls':0,'scope':'Fourcomplete frozenwriteequations; eightdependent AMT-cube futures from22newnativeimages. Explicit sourceassistance/clock/context; no automatic semanticnaming.'}
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'records':len(records),'exact_queries':len(queries),'negative_queries':2*len(queries),'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args();publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
