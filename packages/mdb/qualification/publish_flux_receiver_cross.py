"""Index fixed attention corrections and their receiver-qualified native effects."""
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
    data=run/'extracted/receiver-cross-result';a=read(audit_path);summary,plan,source,binding,forecasts,validations,rows=[read(data/n) for n in ('summary.json','plan.json','source.json','binding.json','forecasts.json','validations.json','rows.json')]
    assert a['mechanics_status']=='verified' and a['request_sha256']==summary['request_sha256'];output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas')
    ref={'path':str(audit_path),'sha256':sha(audit_path)};req={'path':str(run/'request.json'),'sha256':sha(run/'request.json')};records=[];contexts={};effects={};continuations={};counters={}
    by={r['id']:r for r in rows}
    for name,fields in forecasts.items():
        row=by[name];receiver,donor,dose=row['receiver'],row['donor'],row['dose'];b=binding[receiver];execution={'backend':'diffusers',**source['execution']}
        context={'model_identity':source['model_identity'],'execution':execution,'clock':{'step':3,'phase':'single','index':18},'receiving_state':b['before'],'entry_cut':b['before'],'receiver_seed':int(receiver),'delta_source_seed':int(donor),'dose':dose,'delta_sha256':binding[donor]['delta']['sha256'],'source_identity':read(run/'request.json')['source_hashes'],'horizon':{'equation':'single/image-exit-frozenobservedoperands','consumer':'NativeoutgoingT; single19/readout/scheduler/VAE'}}
        comparison=compare_equations({'image.exit':load_tensor_archive(data,fields['I'])},{'image.exit':load_tensor_archive(data,validations[name]['fullblock_image'])},max_elements=4_000_000)
        r=catalog.record(law_id='flux.single.frozen-parallel-gated-write.reference.v1',route_id='flux.text-conditioned-image.step3.single18.v1',alias='contextual-program/fixed-attention-correction-occurrence',
            binding={'model_identity':source['model_identity'],'execution':execution,'input_identity':b['before'],'source_identity':context['source_identity'],'horizon':context['horizon'],'frontier':{'kind':'observed-receiver-R-A-M-T-gate-and-actual-weights','receiving_state':b['before'],'writer':'Anew=B(Ar+dose*deltaAd); P=B([Anew;Mr]W^T+b); I=B(Rr+B(grP))','delta':binding[donor]['delta'],'scope':'FixedFP32source-assistedpayload before recipientBF16 add. Samepixel coordinates, notobjectalignment. CompleteP/Iforecast before mixednativeexecution.'}},
            report=comparison,conditions=[{'field':k,'op':'eq','value':v} for k,v in context.items()],tested_context=context,readers=['image.single.exit','rgb.supplied-eye-region'],
            evidence={'audit':ref,'request':req,'equation_errors':a['equations'][name],'rounding':a['effective_delta_rounding'][name],'native_outcome':a['observations'][name],'forecasts_scored':a['forecast_scores'][name],'common_axis_diagnostic':a['exploratory_common_eye_axes'][name],'coverage':a['coverage']})
        key=r['fingerprint'];records.append(r);contexts[key]=context;score=a['forecast_scores'][name]
        effects[key]=[{'consumer_id':'rgb.supplied-eye-region','contrast_id':'candidate-minus-native-strong','value':score['eye_projection_shift'],'units':'Withinreceiverweak-to-strong RGBcontrastprojection; notgeometrypercentage','provenance':[ref]}]
        related=[name]+([receiver+'.predictedCross'] if receiver!=donor and dose==1 else [])
        continuations[key]={'status':'native-futures-observed','row_ids':related,'provenance':[ref],'scope':'Nativecachedexitfuture; selectedcross+1 also independentCPUexitfuture. NativeTretained.'}
        counters[key]=[{'claim_id':claim,'status':'counterexample','provenance':[ref],'details':score} for claim in ('payload_led_matched','receiver_led_matched') if not score[claim]]
    imported=atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance={'run_id':summary['job_id'],'artifacts':[ref,req]},signed_effects=effects,continuation=continuations,counterexamples=counters);queries=[]
    for key,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1 and not atlas.find(applicable_to={**context,'clock':{**context['clock'],'step':2}}) and not atlas.find(applicable_to={**context,'receiving_state':'different-history'})
        queries.append({'equation_id':key,'atlas_id':hit[0]['fingerprint'],'exact_hits':1,'changed_clock_hits':0,'changed_receiving_hits':0})
    for r in imported:assert atlas.show(r['fingerprint'])==r
    result={'schema':'mdb-receiver-cross-atlas-v1','source_job':summary['job_id'],'audit':ref,'records':len(records),'selected_occurrences':[r['fingerprint'] for r in records],'atlas_ids':[r['fingerprint'] for r in imported],'queries':queries,'new_model_calls':0,'scope':'16context-qualified equations; fixedsource-assistedpayload, receiver/dose/clock explicit. Signedhypothesismisses retained. Notautomaticsemanticfamilydiscovery.'}
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'records':len(records),'exact_queries':len(queries),'negative_queries':2*len(queries),'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args();publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
