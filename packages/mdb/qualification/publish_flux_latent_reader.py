"""Index scoped reader equations and factorial contrasts, retaining their errors."""
import argparse
import runpy
from pathlib import Path


def publish(root,audit,output):
    from mdb.equation_comparison import EquationCatalog, compare_equations
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
    read,load,sha=h['read'],h['load_array'],h['sha'];a=read(audit);c=read(root/'plan.json')
    if a['mechanics_status']!='verified' or a['native_manifest_sha256']!=sha(root/'files.json'):
        raise ValueError('verified matching audit required')
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'equations')
    cohort=read(root/'cohort.json');rows=read(root/'readers.json');ids=[]
    def record(key,p,n,law):
        context=dict(model=c['model'],job=a['job'],occurrence=key,clock=dict(step=3,phase='joint',block=0),
            frontier='observed-native-positioned-QKV',horizon=1,backend='cuda',dtype='bfloat16',batch_size=1)
        report=compare_equations(p,n,max_elements=20_000_000)
        row=catalog.record(law_id=law,route_id='route.flux.latent.image-KV.text-write.v1',alias='cmp.flux.latent-reader.'+key,
            binding=dict(model_identity=c['model'],input_identity=c['projects'],source_identity=a['job'],
                execution=dict(backend='cuda',dtype='bfloat16'),frontier=context['clock'],horizon=1),report=report,
            conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,
            readers=['joint0.text-head','joint0.text-mlp','joint0.text-exit'],
            evidence=dict(audit_sha256=sha(audit),scope=c['scope'],terminal_status='unassessed',semantic_labels='unassigned',new_fitted_parameters=0))
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to={**context,'horizon':2}):
            raise ValueError('applicability scope differs')
        ids.append(row['fingerprint'])
        if key=='right.key':h['save'](output/'context-right-key.json',context)
    p,n={},{}
    for key,row in rows.items():
        p[key]={f:load(root,r) for f,r in cohort['predictions'][key].items()}
        n[key]={f:load(root,r) for f,r in row['fields'].items()}
        record(key,p[key],n[key],'law.flux.text-reader-write.v1')
    for receiver in ('left','right'):
        def contrasts(values):
            b,k,v,d=[values[f'{receiver}.k{ki}v{vi}']['text'].astype('float64') for ki,vi in ((0,0),(1,0),(0,1),(1,1))]
            return dict(key=k-b,value=v-b,interaction=d-k-v+b)
        pc,nc=contrasts(p),contrasts(n)
        for name in pc:record(receiver+'.'+name,{'text_effect':pc[name]},{'text_effect':nc[name]},'law.flux.text-reader-factorial.v1')
    result=dict(occurrences=len(ids),ids=ids,exact_queries=len(ids),changed_horizon_abstentions=len(ids),new_model_calls=0,
        publisher_sha256=sha(Path(__file__)),audit_sha256=sha(audit))
    h['save'](output/'provenance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','audit','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(publish(a.root.resolve(),a.audit.resolve(),a.output.resolve()))
