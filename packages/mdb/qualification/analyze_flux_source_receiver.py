"""Saved-only source/receiver, port, continuation and prediction-error audit."""
import argparse
import json
import runpy
from pathlib import Path


def audit(forecast,native,output):
    import numpy as np
    from PIL import Image
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import verify_transaction
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
    read,sha=h['read'],h['sha']
    f=h['helper']('flux_prompt_factorial.py')
    def checked(root):
        manifest=read(root/'files.json')
        if any(sha(root/name)!=value for name,value in manifest.items()):raise ValueError('evidence file changed')
        return read(root/'summary.json')
    fs,ns=checked(forecast),checked(native)
    cfg=read(native/'plan.json');cohort=read(forecast/'cohort.json')
    nr,fr=read(native/'request.json'),read(forecast/'request.json')
    if (nr['config']!=cfg or nr['request_sha256']!=ns['request_sha256']
            or fr['request_sha256']!=fs['request_sha256']
            or any(fr['config'][k]!=cfg[k] for k in ('parents','point','source_job','source_files_sha256'))):
        raise ValueError('worker request/source binding differs')
    if (fs['job_id']!=cfg['forecast_job'] or sha(forecast/'cohort.json')!=cfg['forecast_cohort_sha256']
            or sha(forecast/'files.json')!=cfg['forecast_files_sha256']
            or ns['forecast_job']!=fs['job_id'] or ns['cohort_sha256']!=cfg['forecast_cohort_sha256']):
        raise ValueError('cross-lease forecast binding differs')
    if fs['counts']['mathematical_transitions']!=300 or fs['counts']['native_model_forwards'] or fs['native_module_attempts']:
        raise ValueError('forecast census differs')
    for key,seal in cohort['seals'].items():verify_transaction(forecast/'transactions'/key,expected_seal_sha256=seal['seal_sha256'])
    def metric(actual,pred):
        d=f['comparison'](actual,pred)
        d['sign']='observed minus predicted'
        d['actual_rms']=float(np.sqrt(np.mean(np.asarray(actual,np.float64)**2)))
        d['reference_rms']=float(np.sqrt(np.mean(np.asarray(pred,np.float64)**2)))
        d['relative_to_actual_rms']=d['rms']/d['actual_rms'] if d['actual_rms'] else None
        return d
    def contrast_error(pred,actual):
        d=metric(pred,actual);d['sign']='predicted minus observed effect';d['relative_rms_denominator']='observed effect RMS';return d
    rows={a['id']:read(native/'rows'/(a['id']+'.json')) for a in h['arms']()}
    original=LocalStore(forecast/'store')
    texts={key:original.load(fp).payload['text'].detach().float().cpu().numpy() for key,fp in cfg['parents'].items()}
    doses={}
    for arm in h['arms']():
        key='right.base' if arm['kind']=='sham' else arm['id']
        value=h['load_array'](forecast,cohort['programs'][key]['input_text'])
        expected=h['text_input'](texts,arm)
        if not np.array_equal(value,expected):raise ValueError('declared source dose differs')
        change=value.astype(np.float64)-texts[arm['receiver']]
        doses[arm['id']]=dict(rms=float(np.sqrt(np.mean(change**2))),l2=float(np.linalg.norm(change)),
            max_abs=float(np.max(np.abs(change))),nonzero_elements=int(np.count_nonzero(change)),
            input_sha256=cohort['programs'][key]['input_text']['sha256'])
    transferred=h['load_array'](forecast,cohort['programs']['right.delta']['input_text'])
    doses['right_delta_vs_contextual_donor']=metric(transferred,texts['c1-o1'])
    doses['left_delta_equals_observed_color_source']=bool(np.array_equal(
        h['load_array'](forecast,cohort['programs']['left.delta']['input_text']),texts['c1-o0']))
    field_errors=[];port_errors=[];locality=[];effects=[]
    ix=np.asarray(cfg['sample_indices']);image_sample=ix>=512
    for arm in h['arms']():
        key=arm['id'];pkey='right.base' if arm['kind']=='sham' else key
        p=cohort['programs'][pkey];n=rows[key]
        if len(p['records'])!=50 or len(n['records'])!=50:raise ValueError('incomplete suffix')
        for pr,nr in zip(p['records'],n['records']):
            if (pr['logical'],pr['local'])!=(nr['logical'],nr['local']):raise ValueError('clock join differs')
            for name,record in pr['outputs'].items():
                if record['archive']:
                    v=h['load_array'](native,nr['outputs'][name]['archive']);w=h['load_array'](forecast,record['archive'])
                    value=metric(v,w)
                    # Raw array hashes are the integrity authority. Cross-host
                    # FP64 reduction differences remain instrument diagnostics.
                    drift={k:value[k]-nr['errors'][name]['full'][k] for k in ('rms','mae','max_abs','squared_l2','relative_rms')
                        if value[k] is not None and nr['errors'][name]['full'][k] is not None}
                    field_errors.append(dict(arm=key,logical=pr['logical'],local=pr['local'],field=name,error=value,worker_reduction_difference=drift))
            if pr['ports']:
                for name,record in pr['ports'].items():
                    native_name='mlp_full' if name=='mlp' else name
                    if native_name in nr['ports']:
                        w=h['load_array'](forecast,record);v=h['load_array'](native,nr['ports'][native_name])
                        if name=='mlp':
                            neurons=np.unique(np.linspace(0,v.shape[-1]-1,min(32,v.shape[-1])).astype(np.int64))
                            v=v[:,:,neurons]
                        if v.ndim==3 and w.ndim==4 and v.shape[:2]==w.shape[:2] and v.shape[-1]==w.shape[-2]*w.shape[-1]:
                            v=v.reshape(w.shape)  # Declared feature -> head/dim view; no value change.
                        port_errors.append(dict(arm=key,local=pr['local'],port=name,error=metric(v,w)))
        if arm['kind']=='sham':
            base=rows['right.base']
            if n['final_payload']!=base['final_payload'] or n['image']['sha256']!=base['image']['sha256']:
                raise ValueError('same-parent sham fails exact native replay')
    for receiver in ('left','right'):
        base,changed=rows[receiver+'.base'],rows[receiver+'.delta']
        bproject=next(r for r in base['records'] if (r['logical'],r['local'])==(3,0))
        cproject=next(r for r in changed['records'] if (r['logical'],r['local'])==(3,0))
        for name in ('image','text'):
            bv,cv=[h['load_array'](native,r['outputs'][name]['archive']) for r in (bproject,cproject)]
            locality.append(dict(receiver=receiver,logical=3,local=0,port=name,exact=bool(np.array_equal(bv,cv)),effect=metric(cv,bv),
                prediction='text exactly resets from unchanged conditioning; image can carry prior latent difference'))
        bj=next(r for r in base['records'] if (r['logical'],r['local'])==(3,1))
        cj=next(r for r in changed['records'] if (r['logical'],r['local'])==(3,1))
        for name in ('text.norm','text.q','text.k','text.v','text.head','text.attention','text.mlp_input','text.mlp','image.head','image.mlp'):
            bv,cv=[h['load_array'](native,r['ports'][name]) for r in (bj,cj)]
            locality.append(dict(receiver=receiver,logical=3,local=1,port=name,exact=bool(np.array_equal(bv,cv)),effect=metric(cv,bv),
                prediction='text prefix local operands unchanged; attention can read changed image K/V before receiving text MLP'))
        for i in (0,1):
            bp,cp=base['records'][i]['ports'],changed['records'][i]['ports']
            for name in ('layernorm','modulated','q_raw','k_raw','q_norm','k_norm','q_rope','k_rope','v','mlp_full','attention','projection'):
                b,c=[h['load_array'](native,r[name])[:,image_sample] for r in (bp,cp)]
                locality.append(dict(receiver=receiver,local=6+i,port=name,exact=bool(np.array_equal(b,c)),effect=metric(c,b)))
        for bn,cn,bp,cp in zip(base['records'],changed['records'],cohort['programs'][receiver+'.base']['records'],cohort['programs'][receiver+'.delta']['records']):
            for name in bn['outputs']:
                if not bn['outputs'][name]['archive']:continue
                values=[h['load_array'](root,r['outputs'][name]['archive']) for root,r in ((native,bn),(native,cn),(forecast,bp),(forecast,cp))]
                observed=values[1].astype(np.float64)-values[0];predicted=values[3].astype(np.float64)-values[2]
                effects.append(dict(receiver=receiver,logical=bn['logical'],local=bn['local'],field=name,error=contrast_error(predicted,observed)))
    rgb={k:np.asarray(Image.open(native/row['image']['path']).convert('RGB'),np.float64) for k,row in rows.items()}
    predicted_images=read(native/'predicted-images.json')
    if predicted_images['counts']['blocks'] or predicted_images['counts']['native_transitions'] or predicted_images['counts']['images']!=6:
        raise ValueError('predicted images must precede native denoising')
    rp={key:np.asarray(Image.open(native/record['path']).convert('RGB'),np.float64) for key,record in predicted_images['images'].items()}
    png_errors={key:metric(rgb[key],rp[key]) for key in rp}
    rgb_effects={receiver:contrast_error(rp[receiver+'.delta']-rp[receiver+'.base'],rgb[receiver+'.delta']-rgb[receiver+'.base']) for receiver in ('left','right')}
    interaction=contrast_error((rp['right.delta']-rp['right.base'])-(rp['left.delta']-rp['left.base']),
        (rgb['right.delta']-rgb['right.base'])-(rgb['left.delta']-rgb['left.base']))
    controls={}
    for cell,key in [('c0-o0','left.base'),('c0-o1','right.base')]:
        expected=cfg['calibration'][cell]
        controls[key]=dict(png_exact=rows[key]['image']['sha256']==expected['image']['sha256'])
    result=dict(schema='mdb-source-receiver-audit-v1',mechanics_status='verified',terminal_status='unassessed',
        scope=cfg['scope'],forecast_job=fs['job_id'],native_job=ns['job_id'],forecast_sha256=sha(forecast/'files.json'),native_sha256=sha(native/'files.json'),
        source_covariates='Left/right receivers differ in complete observed state; not an isolated image-only receiver contrast.',
        counts=dict(forecast=fs['counts'],native=ns['counts']),controls=controls,
        input_doses=doses,field_errors=field_errors,port_errors=port_errors,locality=locality,effect_errors=effects,
        rgb_endpoint_errors=png_errors,rgb_effect_errors=rgb_effects,rgb_interaction_error=interaction,
        new_model_calls=0,auditor_sha256=sha(Path(__file__)),
        limitations=['Saved-only source/custody/reduction audit, not independent reconstruction of all300 mathematical transitions.',
            'Known seed127 source contexts; new intervention combinations, no new natural prompt/seed.',
            'Fixed-coordinate port witnesses and six full landmark exits per branch; no full-field errors elsewhere.',
            'Raw pixel effects are not calibrated object segmentation or semantic causal percentages.'])
    h['save'](output,result)
    return dict(mechanics_status='verified',forecast_job=fs['job_id'],native_job=ns['job_id'],controls=controls,
        rgb_endpoint_mae={k:v['mae'] for k,v in png_errors.items()},rgb_effect_relative={k:v['relative_rms'] for k,v in rgb_effects.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('forecast',type=Path);p.add_argument('native',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.forecast.resolve(),a.native.resolve(),a.output.resolve()),sort_keys=True))
