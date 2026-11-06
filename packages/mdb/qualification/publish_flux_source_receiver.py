"""Publish scoped suffix predictions and retained source/receiver effect errors."""
import argparse
import runpy
from pathlib import Path


def publish(forecast,native,audit,output):
    from mdb.equation_comparison import EquationCatalog, compare_equations
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
    read,sha=h['read'],h['sha'];a=read(audit);cfg=read(native/'plan.json');c=read(forecast/'cohort.json')
    if a['mechanics_status']!='verified' or a['forecast_job']!=cfg['forecast_job']:
        raise ValueError('verified source/receiver audit required')
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'equations');ids=[];pred={};actual={}
    for arm in h['arms']():
        if arm['kind']=='sham':continue
        key=arm['id'];p=c['programs'][key]['records'][-1]['outputs']['latent']['archive']
        n=read(native/'rows'/(key+'.json'))['records'][-1]['outputs']['latent']['archive']
        pred[key]=h['load_array'](forecast,p);actual[key]=h['load_array'](native,n)
    def record(key,law,p,n,evidence):
        context=dict(model=cfg['model'],source_job=cfg['source_job'],occurrence=key,
            clock={'step':2,'phase':'single','block':0},horizon=50,numerical={'backend':'cuda','dtype':'bfloat16'},
            input_identity=cfg['parents'],cohort_sha256=cfg['forecast_cohort_sha256'])
        report=compare_equations(p,n,max_elements=20_000_000)
        row=catalog.record(law_id=law,route_id='route.flux.source.single-writes.latent.later-joint-reader.v1',
            alias='cmp.flux.source-receiver.'+key,binding=dict(model_identity=cfg['model'],execution=context['numerical'],
                input_identity=cfg['parents'],source_identity=a['native_job'],frontier=context['clock'],horizon=50),
            report=report,conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,
            readers=['native.final-latent','native.vae','step2.single0.parallel-attention-mlp','step3.joint0'],
            evidence=dict(audit={'path':str(audit),'sha256':sha(audit)},semantic_labels='unassigned',terminal_status='unassessed',
                scope=cfg['scope'],new_fitted_parameters=0,**evidence))
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to={**context,'horizon':49}):
            raise ValueError('catalog applicability scope mismatch')
        ids.append(row['fingerprint'])
    for key in pred:record(key,'law.flux.complete-suffix.v1',{'latent':pred[key]},{'latent':actual[key]},dict(rgb_error=a['rgb_endpoint_errors'][key]))
    for receiver in ('left','right'):
        record(receiver+'.effect','law.flux.conditional-source-response.v1',
            {'latent':pred[receiver+'.delta'].astype('float64')-pred[receiver+'.base']},
            {'latent':actual[receiver+'.delta'].astype('float64')-actual[receiver+'.base']},dict(rgb_effect_error=a['rgb_effect_errors'][receiver]))
    record('receiver.interaction','law.flux.source-receiver-interaction.v1',
        {'latent':(pred['right.delta'].astype('float64')-pred['right.base'])-(pred['left.delta'].astype('float64')-pred['left.base'])},
        {'latent':(actual['right.delta'].astype('float64')-actual['right.base'])-(actual['left.delta'].astype('float64')-actual['left.base'])},
        dict(rgb_interaction_error=a['rgb_interaction_error'],interpretation='Finite complete-receiver difference of conditional source effects; no semantic or causal-percentage interpretation.'))
    result=dict(schema='mdb-source-receiver-publication-v1',occurrences=len(ids),ids=ids,exact_queries=len(ids),changed_horizon_abstentions=len(ids),
        audit_sha256=sha(audit),new_model_calls=0,publisher_sha256=sha(Path(__file__)))
    h['save'](output/'provenance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('forecast','native','audit','output'):p.add_argument(k,type=Path)
    a=p.parse_args();print(publish(a.forecast.resolve(),a.native.resolve(),a.audit.resolve(),a.output.resolve()))
