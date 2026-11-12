"""Publish clock/receiver-qualified RAM exits and their paired T continuations."""
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
    data=run/'extracted/step1-overlap-result';a=read(audit_path)
    summary,config,source,binding,numeric=[read(data/n) for n in ('summary.json','plan.json','source.json','binding.json','numeric-seal.json')]
    assert a['mechanics_status']=='verified' and a['request_sha256']==summary['request_sha256']
    validations=read(data/'validations.json');output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas')
    ref={'path':str(audit_path),'sha256':sha(audit_path)};req={'path':str(run/'request.json'),'sha256':sha(run/'request.json')}
    records=[];contexts={};effects={};futures={};counters={}
    for key,descriptor in numeric['I'].items():
        role='shape' if key[0]=='1' else 'base';execution={'backend':'diffusers',**source['execution']}
        context={'model_identity':source['model_identity'],'execution':execution,'clock':{'step':1,'phase':'single','index':6},'selected_R_A_M':key,
            'source_identity':read(run/'request.json')['source_hashes'],'receiving_state':binding['parents'][role],'entry_cut':binding['parents'][role],'seed':config['seed'],
            'horizon':{'equation':'single6/image-exit-frozenRAM','consumer':'Retainedbase/shape postblockT throughstep1; commonshapeEfrom2; native scheduler/VAE'}}
        observed=load_tensor_archive(data,validations[key]['image']);predicted=load_tensor_archive(data,descriptor)
        comparison=compare_equations({'image.exit':predicted},{'image.exit':observed},max_elements=4_000_000)
        row_id=lambda t:'cube.R'+key[0]+'A'+key[1]+'M'+key[2]+'.T'+t
        record=catalog.record(law_id='flux.single.frozen-parallel-gated-write.reference.v1',route_id='flux.text-conditioned-image.step1.single6.v1',alias='contextual-program/single-write-occurrence',
            binding={'model_identity':source['model_identity'],'execution':execution,'input_identity':context['entry_cut'],'source_identity':context['source_identity'],'horizon':context['horizon'],
                'frontier':{'kind':'observed-R-A-M-gate-actual-weights','receiving_state':context['receiving_state'],'selected_R_A_M':key,'writer':'P=B([A;M]W^T+b); I=B(R+B(gP))','source_fields':binding['fields'],
                    'scope':'NativeA/M observed. FourP andeightI completeCPUforecasts; mixed10/01P beforefirstnativeevaluation, source00/11P alreadyobserved. Numericalexit doesnot predictformation orrawpromptRGB.'}},
            report=comparison,conditions=[{'field':k,'op':'eq','value':v} for k,v in context.items()],tested_context=context,
            readers=['image.single6.exit','rgb.supplied-eye-region'],evidence={'audit':ref,'request':req,'numeric_seal_sha256':summary['numeric_seal_sha256'],'equation_errors':{k:a['equations'][k] for k in ('P'+key[1:],'I'+key)},
                'T_conditioned_outcomes':{r:a['observations'][row_id(r)] for r in ('base','shape')},'conditional_effects':a['conditional_effects'],'predicted_corner_futures':a['predicted_future_errors'],'coverage':a['coverage'],
                'observer_scope':'Unblinded knownportrait; RGBprojection againstweak/strongnativecontrols, not calibratedgeometry; coarsepreservation plusfinecollateral.'})
        identifier=record['fingerprint'];records.append(record);contexts[identifier]=context
        selected=next(r for r in a['conditional_effects']['T'] if r['low']==row_id('base'))
        effects[identifier]=[{'consumer_id':'rgb.supplied-eye-region','contrast_id':'shapeT-minus-baseT-at-fixed-RAM','value':selected['regions']['eyes']['projection_delta'],'units':'weak-to-strong RGBcontrastprojection','provenance':[ref]}]
        futures[identifier]={'status':'native-T-conditioned-futures-observed','row_ids':[row_id(r) for r in ('base','shape')],'provenance':[ref],'scope':'Two dependent nativecachedfutures undercommonshapeE. Only000/111 alsohaveCPU-predicted imageexitfutures; sourcefields observed.'}
        counters[identifier]=[{'claim_id':'one-write-is-a-context-free-shape-symbol','status':'not-established','provenance':[ref],'details':{'scope':'RAMT conditionalcontexts oneportrait/clock. No semanticminimality, freshpromptgeneralization orupstreamformation prediction.'}}]
    imported=atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance={'run_id':summary['job_id'],'artifacts':[ref,req]},signed_effects=effects,continuation=futures,counterexamples=counters)
    queries=[]
    for key,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1
        assert not atlas.find(applicable_to={**context,'clock':{'step':2,'phase':'single','index':6}})
        assert not atlas.find(applicable_to={**context,'receiving_state':'different-I-history'})
        queries.append({'equation_id':key,'atlas_id':hit[0]['fingerprint'],'exact_hits':1,'changed_clock_hits':0,'changed_receiving_hits':0})
    for r in imported:assert atlas.show(r['fingerprint'])==r
    result={'schema':'mdb-step1-overlap-atlas-v1','source_job':summary['job_id'],'audit':ref,'records':len(records),'selected_occurrences':[r['fingerprint'] for r in records],
        'atlas_ids':[r['fingerprint'] for r in imported],'queries':queries,'new_model_calls':0,'scope':'EightfullRAMexitequations;16dependentnativecachedRAMT futures+2CPUcornerfutures. Observedsourceassistance andscopeexplicit; noautomaticsemanticnaming.'}
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'records':len(records),'exact_queries':len(queries),'negative_queries':2*len(queries),'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args();publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
