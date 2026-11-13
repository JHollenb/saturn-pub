"""Existing public equation/atlas APIs for eight context-qualified late writes."""
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
    data=run/'extracted/interaction-writes-result';a=read(audit_path);summary,plan,source,binding,forecasts,validations=[read(data/n) for n in ('summary.json','plan.json','source.json','binding.json','forecasts.json','validations.json')]
    assert a['mechanics_status']=='verified' and a['request_sha256']==summary['request_sha256'];output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas')
    ref={'path':str(audit_path),'sha256':sha(audit_path)};req={'path':str(run/'request.json'),'sha256':sha(run/'request.json')};records=[];contexts={};effects={};continuations={};counters={}
    for seed,b in binding.items():
        for case,fields in forecasts[seed].items():
            p=b['point'];execution={'backend':'diffusers',**source['execution']};context={'model_identity':source['model_identity'],'execution':execution,'clock':{'step':p['logical_step'],'phase':'single','index':p['index']},'seed':int(seed),'selected_operand_recipe':case,'receiving_state':b['before']['11'],'entry_cut':b['before']['11'],'source_identity':read(run/'request.json')['source_hashes'],'horizon':{'equation':'single/image-exit-frozenobservedoperands','consumer':'NativeoutgoingT unlessdeclaredTlinear; remainingnativeblocks/readout/scheduler/VAE'}}
            observed=load_tensor_archive(data,validations[seed][case]['fullblock_image']);predicted=load_tensor_archive(data,fields['I']);comparison=compare_equations({'image.exit':predicted},{'image.exit':observed},max_elements=4_000_000)
            r=catalog.record(law_id='flux.single.frozen-parallel-gated-write.reference.v1',route_id='flux.text-conditioned-image.step'+str(p['logical_step'])+'.single'+str(p['index'])+'.v1',alias='contextual-program/interaction-correction-occurrence',
                binding={'model_identity':source['model_identity'],'execution':execution,'input_identity':context['entry_cut'],'source_identity':context['source_identity'],'horizon':context['horizon'],'frontier':{'kind':'observed-four-corner-R-A-M-T-and-actual-weights','receiving_state':context['receiving_state'],'writer':'P=B([A;M]W^T+b); I=B(R+B(gP))','recipe':case,'source_cuts':b['before'],'scope':'Frozenobservedoperands/composedlinearizations; completeP/Iforecastbeforemixednativeexecution, notupstreamformationforecast.'}},
                report=comparison,conditions=[{'field':k,'op':'eq','value':v} for k,v in context.items()],tested_context=context,readers=['image.single.exit','rgb.supplied-eye-region'],
                evidence={'audit':ref,'request':req,'equation_errors':a['equations'][seed][case],'source_port_trace_audit_sha256':plan['source_audit_sha256'],'native_outcomes':{k:v for k,v in a['observations'].items() if k.startswith(seed+'.')},'forecast_scores':a['forecast_scores'][seed],'predicted_AM_future':a['predicted_future_errors'][seed],'coverage':a['coverage']})
            key=r['fingerprint'];records.append(r);contexts[key]=context
            delta=a['observations'][seed+'.'+case]['eyes']['source_contrast_projection']-a['observations'][seed+'.sham']['eyes']['source_contrast_projection']
            effects[key]=[{'consumer_id':'rgb.supplied-eye-region','contrast_id':'candidate-minus-native-strong','value':delta,'units':'Within-seedweak-to-strong RGBcontrastprojection; notgeometrypercentage','provenance':[ref]}]
            cases=[case]+(['AMTlinear','predictedAM'] if case=='AMlinear' else [])
            continuations[key]={'status':'native-futures-observed','row_ids':[seed+'.'+c for c in cases],'provenance':[ref],'scope':'Nativecachedimageexitfutures; predictedAM addsindependentCPUexitfuture undernativeoutgoingT. Consumerdetails/sourceassistanceexplicit.'}
            counters[key]=[{'claim_id':k,'status':'counterexample','provenance':[ref],'details':{'seed':seed,'score':v}} for k,v in a['forecast_scores'][seed].items() if type(v) is bool and not v]
    imported=atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance={'run_id':summary['job_id'],'artifacts':[ref,req]},signed_effects=effects,continuation=continuations,counterexamples=counters);queries=[]
    for key,context in contexts.items():
        changed_step=2 if context['clock']['step']==3 else 3
        hit=atlas.find(applicable_to=context);assert len(hit)==1 and not atlas.find(applicable_to={**context,'clock':{**context['clock'],'step':changed_step}}) and not atlas.find(applicable_to={**context,'receiving_state':'different-history'})
        queries.append({'equation_id':key,'atlas_id':hit[0]['fingerprint'],'exact_hits':1,'changed_clock_hits':0,'changed_receiving_hits':0})
    for r in imported:assert atlas.show(r['fingerprint'])==r
    result={'schema':'mdb-interaction-writes-atlas-v1','source_job':summary['job_id'],'audit':ref,'records':len(records),'selected_occurrences':[r['fingerprint'] for r in records],'atlas_ids':[r['fingerprint'] for r in imported],'queries':queries,'new_model_calls':0,'scope':'Eightfullimageexit equationsfrom8P/8Iforecasts,16dependentnativefutures. Existinggenericlawreused atnewclock/receiver; explicitcuration, notautomaticsemanticfamilydiscovery.'}
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'records':len(records),'exact_queries':len(queries),'negative_queries':2*len(queries),'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args();publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
