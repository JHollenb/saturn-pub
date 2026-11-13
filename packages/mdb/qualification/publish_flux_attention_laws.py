"""Index supplied attention/receiver numerical programs without semantic overclaim."""
import argparse
import json
from pathlib import Path
from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog,compare_equations
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive
import runpy


def publish(study,fresh,output):
    read=lambda p:json.loads(p.read_text());boundary=read(study/'audit-v1.json');receiver=read(study/'receiver/audit-v1.json')
    if any(a['mechanics_status']!='verified' for a in (boundary,receiver)):raise ValueError('verified audits required')
    rawroot=study/'collection-01/extracted/binding-minimal-result';fullroot=study/'receiver/collection-01/extracted/binding-minimal-result';target=fresh/'extracted/binding-minimal-result';native=read(target/'native.json');receipt=read(target/'receipts.json')[0]['body'];cases=read(rawroot/'plan.json')['cases'];raw=read(rawroot/'predictions.json');full=read(fullroot/'predictions.json');bits=runpy.run_path(str(study/'collection-01/payload/flux_attention_anchor.py'))['load_bits']
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas');rows=[];contexts={};limits={}
    refs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (study/'audit-v1.json',study/'full/audit-v1.json',study/'receiver/audit-v1.json',fresh/'request.json')]
    for spec in cases:
        name=spec['id']
        programs=[('raw',m,{'raw_attention':bits(rawroot,raw['cases'][name]['own'][m])},{'raw_attention':load_tensor_archive(target,native[name]['stages']['attention.raw'])}) for m in ('absolute','anchored')]
        programs.extend(('complete',m,{r:bits(fullroot,record['fields'][r]['exit']) for r in ('image','text')},{r:load_tensor_archive(target,native[name]['stages'][r+'.exit']) for r in ('image','text')}) for m,record in full['cases'][name].items())
        for scope,method,calculated,actual in programs:
            collection=study/('collection-01' if scope=='raw' else 'receiver/collection-01');request=read(collection/'request.json');sources={**request['source_hashes'],**{'wheel:'+k:v for k,v in read(collection/'payload/payload-seal.json')['wheels'].items()}}
            binding=dict(model_identity=receipt['model_identity'],source_identity=sources,input_identity=native[name]['entry'],execution=dict(backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',native_target=receipt['execution']),frontier=dict(kind='saved-state-development',scope='Known targets scored after independent own-state calculation; conditional-native-operands diagnostics excluded.'),horizon=dict(equation='raw attention' if scope=='raw' else 'complete joint1 I/T exit',consumer='saved native array comparison; no new continuation'))
            context=dict(**binding,clock=dict(step=2,joint=1),receiver=spec['receiver'],row_doses=spec['row_doses'],scope=scope,method=method)
            law='flux.joint1.attention.'+method+'.v1' if scope=='raw' else 'flux.joint1.receiver-law.'+method+'.v1'
            evidence=boundary['reports'][name]['own'][method] if scope=='raw' else dict(measurements=receiver['reports'][name][method],error_transport=receiver['error_transport'][name][method])
            row=catalog.record(law_id=law,route_id='flux.step2.joint1.contextual-write.v1',alias='cmp.flux.attention-law.'+name+'.'+scope+'.'+method,binding=binding,report=compare_equations(calculated,actual,max_elements=5_000_000),conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,readers=list(actual),evidence=dict(artifacts=refs,measurements=evidence,additional_native_targets=0,source_native_states=4,selection='Supplied fixed program and applicability, not induced semantic name'))
            fp=row['fingerprint'];rows.append(row);contexts[fp]=context
            limits[fp]=[dict(claim_id='development-and-composition-limits',status='limitation-retained',scope='Four known states/two unique text inputs; direct attention is locally better but its complete response can worsen. Error cross moments and weak mixed contrasts retained. No fresh-state, image, automatic naming/applicability or speed claim.',measurements=dict(mixed=boundary['factorial']['own'][method] if scope=='raw' else receiver['factorial'][method],law_interaction=None if scope=='raw' else receiver['law_interaction'][name]),provenance=refs)]
    atlas.ingest(output/'catalog',[r['fingerprint'] for r in rows],provenance=dict(run_id='saved-attention-receiver-laws',artifacts=refs),counterexamples=limits)
    for context in contexts.values():
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}):raise ValueError('applicability query differs')
    result=dict(schema='mdb-attention-law-publication-v1',programs=6,equation_occurrences=24,source_native_states=4,new_native_targets=0,exact_queries=24,changed_context_abstentions=48,tensor_bytes_copied=0,new_native_calls=0,scope='Two supplied attention-boundary laws and four complete numerical programs on the same four known states. Exact supplied applicability, not learned families or new discoveries per occurrence.')
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','fresh','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(publish(a.study.resolve(),a.fresh.resolve(),a.output.resolve()),sort_keys=True))
