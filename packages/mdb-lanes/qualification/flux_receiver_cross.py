"""Cross a fixed source-assisted attention correction with two receiving states."""
DOSES=(-1.0,-0.5,0.5,1.0)


def cases(seeds):
    result=[]
    for receiver in seeds:
        result.append(dict(id=receiver+'.sham',receiver=receiver,donor=None,dose=0.0,predicted=False))
        for donor in seeds:
            for dose in DOSES:
                result.append(dict(id=f'{receiver}.from{donor}.{dose:+g}',receiver=receiver,donor=donor,dose=dose,predicted=False))
        donor=next(s for s in seeds if s!=receiver)
        result.append(dict(id=receiver+'.predictedCross',receiver=receiver,donor=donor,dose=1.0,predicted=True))
    return result


def compose(native,delta,*,text_rows,width,dose):
    import torch
    if native.ndim!=3 or delta.shape!=native[:,text_rows:,:width].shape or delta.dtype!=torch.float32 or not bool(torch.isfinite(delta).all()):
        raise ValueError('Declared finite FP32 image-attention correction required')
    value=native.clone()
    value[:,text_rows:,:width]=(native[:,text_rows:,:width].float()+dose*delta).to(native.dtype)
    if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite composed operand')
    return value


def census(seeds):
    n=len(seeds)
    if n!=2:raise ValueError('This assay requires exactly two seeds')
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=60,acts=40,receipts=100,images=20,candidate_blocks=16,cached_projections=18)
