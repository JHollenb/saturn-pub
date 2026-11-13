"""Public Euler equation records with separately curated conditional tail evidence."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def publish(run,audit_path,output):
    from mdb_lanes.computation_atlas import ComputationAtlas
    from mdb_lanes.equation_comparison import EquationCatalog,compare_equations
    from mdb_lanes.tensor_archive import load_tensor_archive
    data=run/'extracted/suffix-context-result';a=read(audit_path);summary,plan,source,forecasts,rows=[read(data/n) for n in ('summary.json','plan.json','source.json','forecasts.json','rows.json')]
    assert a['mechanics_status']=='verified' and a['request_sha256']==summary['request_sha256'];output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas')
    here=Path(__file__).resolve().parent;typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'));route=runpy.run_path(str(here/'analyze_flux_concept_routes.py'));slot=lambda fp,s:route['slot_array'](data/'store',fp,s,typed)
    ref={'path':str(audit_path),'sha256':sha(audit_path)};req={'path':str(run/'request.json'),'sha256':sha(run/'request.json')};records=[];contexts={};effects={};continuations={};counters={};curated=[]
    for pair_key,observation in a['paired_observations'].items():
        score=a['signed_forecast_scores'][pair_key];curated.append({'schema':'mdb-curated-computation-record-v1','id':'flux.suffix-context.'+pair_key,'record_kind':'Fixed-complete-image-pair × native-text × native-latent causal occurrence','applicability':{'model_identity':source['model_identity'],'execution':source['execution'],'clock':{'step':3,'phase':'single','index':19},'image_pair_cuts':plan['pairs'][observation['pair']],'text_source_cut':plan['parents'][observation['text']],'latent_source_cut':plan['parents'][observation['latent']],'observer_axes':a['observer_axis_png_sha256']},'observation':observation,'frozen_directional_scores':score,'audit':ref,'mechanics_status':'verified','terminal_status':'unassessed','scope':'Curatedcausalrecord, notautomaticallyinducedsemanticprogram or equation.'})
    for row in rows:
        name=row['id'];pair_key=f"{row['pair']}.T{row['text']}.Z{row['latent']}";execution={'backend':'diffusers',**source['execution']};context={'model_identity':source['model_identity'],'execution':execution,'clock':{'step':3,'phase':'scheduler'},'entry_cut':row['scheduler_cut'],'receiving_state':row['scheduler_cut'],'image_pair':row['pair'],'image_choice':row['bit'],'text_source':row['text'],'latent_source':row['latent'],'source_identity':read(run/'request.json')['source_hashes'],'horizon':{'equation':'complete-final-Euler-commit','consumer':'VAE-on-native-committed-latent'}}
        predicted=load_tensor_archive(data,forecasts[name]['latent']);comparison=compare_equations({'latent.exit':predicted},{'latent.exit':slot(row['final_cut'],'latent')},max_elements=200_000)
        r=catalog.record(law_id='flux.euler.commit.bf16-reference.v1',route_id='flux.final-flow-commit.v1',alias='contextual-program/observed-readout-euler-occurrence',
            binding={'model_identity':source['model_identity'],'execution':execution,'input_identity':row['scheduler_cut'],'source_identity':context['source_identity'],'horizon':context['horizon'],'frontier':{'kind':'observed-native-last-block-and-readout','source_cut':row['scheduler_cut'],'writer':'Z4=B(Z3+B(B(dt)*V3))','scope':'CompleteEulerprediction sealedbeforeownnativecommit/RGB, afterobservednativevelocity. Noupstreamformationforecast.'}},
            report=comparison,conditions=[{'field':k,'op':'eq','value':v} for k,v in context.items()],tested_context=context,readers=['latent.final','rgb.fixed60667-eye-window'],
            evidence={'audit':ref,'request':req,'euler_errors':a['euler_forecasts'][name],'causal_pair_id':'flux.suffix-context.'+pair_key,'paired_outcome':a['paired_observations'][pair_key],'frozen_directional_scores':a['signed_forecast_scores'][pair_key],'coverage':a['coverage']})
        key=r['fingerprint'];records.append(r);contexts[key]=context;score=a['signed_forecast_scores'][pair_key]
        effects[key]=[{'consumer_id':'rgb.fixed60667-eye-window','contrast_id':'I1-minus-I0-at-fixed-T-Z','value':score['effect'],'units':'Fixedsaved60667 RGBcontrastprojection; associatedpairedcontrast, sharedbytworows; notgeometrypercentage','provenance':[ref]}]
        continuations[key]={'status':'native-futures-observed','row_ids':a['paired_observations'][pair_key]['rows'],'provenance':[ref],'scope':'Bothfixed-imagepairmembers retained underidenticalT/Z; this equation record describesoneEulerrow.'}
        counters[key]=[{'claim_id':claim,'status':'counterexample','provenance':[ref],'details':score} for claim,matched in score['matched'].items() if not matched]
    imported=atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance={'run_id':summary['job_id'],'artifacts':[ref,req]},signed_effects=effects,continuation=continuations,counterexamples=counters);queries=[]
    for key,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1 and not atlas.find(applicable_to={**context,'clock':{**context['clock'],'step':2}}) and not atlas.find(applicable_to={**context,'receiving_state':'different-history'})
        queries.append({'equation_id':key,'atlas_id':hit[0]['fingerprint'],'exact_hits':1,'changed_clock_hits':0,'changed_receiving_hits':0})
    for r in imported:assert atlas.show(r['fingerprint'])==r
    (output/'causal-records.json').write_text(json.dumps({'records':curated},indent=2,sort_keys=True)+'\n')
    result={'schema':'mdb-suffix-context-atlas-v1','source_job':summary['job_id'],'audit':ref,'records':len(records),'curated_causal_records':len(curated),'selected_occurrences':[r['fingerprint'] for r in records],'atlas_ids':[r['fingerprint'] for r in imported],'queries':queries,'new_model_calls':0,'scope':'16observed-readoutEuler occurrences;8separatecuratedcausalpairs share16nativefutures. Pairedsignforecasts distinctfromlocalnumericalfidelity. Noautomaticsemanticfamilydiscovery.'}
    (output/'provenance.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'records':len(records),'curated_causal_records':len(curated),'exact_queries':len(queries),'negative_queries':2*len(queries),'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('output',type=Path);a=p.parse_args();publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
