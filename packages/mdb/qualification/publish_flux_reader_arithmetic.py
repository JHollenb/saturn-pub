"""Index known-input program parity and local/carried arithmetic discrepancies."""
import argparse
import runpy
from pathlib import Path


def publish(root,source,audit,output):
    from mdb.equation_comparison import EquationCatalog, compare_equations
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
    read,sha,load=h['read'],h['sha'],h['load_array'];a=read(audit)
    if a['mechanics_status']!='verified' or a['output_manifest_sha256']!=sha(root/'files.json'):raise ValueError('matching audit required')
    c=read(root/'plan.json');rows=read(root/'cases.json');model=read(source/'plan.json')['model']
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'equations');ids=[]
    flag=a['native_contract']['default_bf16_reduced_reduction']
    profile=dict(backend='cuda',dtype='bfloat16',bf16_reduced_reduction=flag if isinstance(flag,bool) else 'source-runtime-default-not-retained',
        device=a['native_contract']['device'],shape=[1,c['text_rows'],c['width']])
    def record(key,law,p,n,kind):
        context=dict(model=model,source_job=a['job'],occurrence=key,clock=dict(step=3,phase='joint',block=0),
            frontier='observed native component inputs',numerical=profile,comparison=kind,horizon=1)
        row=catalog.record(law_id=law,route_id='route.flux.image-KV.text-projection-residual-mlp.v1',alias='cmp.flux.reader-arithmetic.'+key,
            binding=dict(model_identity=model,input_identity=c['parents'],source_identity=a['job'],execution=profile,
                frontier=context['clock'],horizon=1),report=compare_equations(p,n,max_elements=20_000_000),
            conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,
            readers=['joint0.text-projection','joint0.text-residual','joint0.text-mlp'],
            evidence=dict(audit_sha256=sha(audit),instrument_status=a['instrument_status'],scope=c['scope'],
                role='known-target numerical diagnosis; not prospective prediction',new_fitted_parameters=0,terminal_status='unassessed'))
        wrong={**context,'numerical':{**profile,'bf16_reduced_reduction':not flag if isinstance(flag,bool) else 'opposite-of-source-runtime-default'}}
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to=wrong):raise ValueError('numerical applicability differs')
        if key=='right.projection-effect':
            h['save'](output/'context-right-projection.json',context);h['save'](output/'context-wrong-reduction.json',wrong)
        ids.append(row['fingerprint'])
    for key,row in rows.items():
        p={f:load(root,row['native']['native-input'][f]) for f in ('after_attention','mlp','text')}
        n={f:load(source,row['source_observed'][f]) for f in p}
        record(key,'law.flux.reader-tail.native-program.v1',p,n,'source-return component replay')
    for receiver in ('left','right'):
        b,d=[rows[receiver+'.'+name] for name in ('k0v0','k1v1')]
        for field in ('projection','paired','mlp'):
            p=[load(root,row['math']['local'][field]).astype('float64') for row in (b,d)]
            n=[load(root,row['native']['native-input'][field]).astype('float64') for row in (b,d)]
            record(receiver+'.'+field+'-effect','law.flux.reader-affine.reference.v1',{'effect':p[1]-p[0]},{'effect':n[1]-n[0]},'local operator effect')
        p=[load(root,row['native']['reference-head']['text']).astype('float64') for row in (b,d)]
        n=[load(source,row['source_observed']['text']).astype('float64') for row in (b,d)]
        record(receiver+'.carried-head-effect','law.flux.reader-tail.carried-head.v1',{'effect':p[1]-p[0]},{'effect':n[1]-n[0]},'frozen reference head through native arithmetic')
    result=dict(occurrences=len(ids),ids=ids,exact_queries=len(ids),changed_reduction_abstentions=len(ids),
        audit_sha256=sha(audit),publisher_sha256=sha(Path(__file__)),new_model_calls=0)
    h['save'](output/'provenance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','audit','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(publish(a.root.resolve(),a.source.resolve(),a.audit.resolve(),a.output.resolve()))
