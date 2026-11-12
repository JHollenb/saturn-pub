"""Index bounded arithmetic programs and keep their counterexamples attached."""
import argparse
import json
from pathlib import Path
import numpy as np
from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog,compare_equations
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive


def publish(study,output):
    read=lambda p:json.loads(p.read_text()); audit=read(study/'fresh/audit-v1.json'); transfer=read(study/'transfer/audit-v1.json')
    if audit['mechanics_status']!='verified' or transfer['mechanics_status']!='verified': raise ValueError('verified evidence required')
    collection=study/'fresh/collection-01'; data=collection/'extracted/binding-minimal-result'; c=read(data/'plan.json'); req=read(collection/'request.json'); native=read(data/'native.json'); forecasts=read(data/'forecasts.json'); receipt=read(data/'receipts.json')[0]['body']
    output.mkdir(parents=True,exist_ok=False); catalog=EquationCatalog(output/'catalog'); atlas=ComputationAtlas(output/'atlas')
    refs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (study/'fresh/audit-v1.json',study/'transfer/audit-v1.json',collection/'request.json')]
    sources={**req['source_hashes'],**{'wheel:'+k:v for k,v in read(collection/'payload/payload-seal.json')['wheels'].items()}}; contexts={}; records=[]; limits={}; futures={}
    for spec in c['cases']:
        name=spec['id']; actual={r:load_tensor_archive(data,native[name]['stages'][r+'.exit']) for r in ('image','text')}
        for method,law in [('local','flux.joint.primitive-anchor-wide.v1'),('arithmetic','flux.joint.attention-affine-scan1024.v1')]:
            calculated={r:load_tensor_archive(data,forecasts[name][method]['fields'][r]['exit']) for r in actual}
            report=compare_equations(calculated,actual,max_elements=5_000_000)
            binding=dict(model_identity=receipt['model_identity'],source_identity=sources,execution=dict(backend='diffusers',**receipt['execution']),input_identity=native[name]['entry'],
                frontier=dict(kind='sealed-new-internal-state',scope='Observed prefix and declared native calibration; candidate own operands sealed before targets. Native fallback diagnostics excluded from this equation.'),horizon=dict(equation='joint1 complete I/T exit',consumer='next typed joint2 state; no image continuation'))
            context=dict(**binding,checkpoint_sha256=c['checkpoint_sha256'],clock=dict(step=2,joint=1),receiver=spec['receiver'],row_doses=spec['row_doses'],method=method)
            row=catalog.record(law_id=law,route_id='flux.step2.joint1.contextual-color-state.v1',alias='cmp.flux.arithmetic.'+name+'.'+method,binding=binding,report=report,
                conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,readers=['native.joint1.image.exit','native.joint1.text.exit'],
                evidence=dict(artifacts=refs,measurements=audit['state'][name][method],native_target_count=4,forecast_seal=audit['transaction']['seal_sha256'],
                    rule=c['policy'] if method=='arithmetic' else 'native-anchor-wide-delta',absolute_ports=c['absolute_ports'] if method=='arithmetic' else []))
            fp=row['fingerprint']; records.append(row); contexts[fp]=context
            limits[fp]=[dict(claim_id='fine-interaction-and-geometry',status='limitation-retained',scope='Receiver-by-pattern errors retained;1024carry does not transfer to sampled MLP matrices. Native anchors/receiver/source remain supplied; no automatic semantic addressing, image continuation or speed claim.',
                measurements=dict(factorial=audit['factorial'][method],mixed_attention=audit['mixed_attention']),provenance=refs)]
            futures[fp]=dict(status='native-next-state-observed-no-image-continuation',exit=native[name]['exit'],provenance=refs)
    atlas.ingest(output/'catalog',[r['fingerprint'] for r in records],provenance=dict(run_id='fresh-affine-arithmetic',artifacts=refs),counterexamples=limits,continuation=futures)
    for context in contexts.values():
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}): raise ValueError('applicability query differs')
    result=dict(schema='mdb-arithmetic-publication-v1',programs=2,equation_occurrences=8,new_target_states=4,exact_queries=8,negative_queries=16,tensor_bytes_copied=0,new_model_calls=0,
        scope='Two supplied programs on four prospective native states; exact-context applicability indexed, not learned. Numerical law and limitations explicit, no semantic discovery or image-future qualification.')
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(publish(a.study.resolve(),a.output.resolve()),sort_keys=True))
