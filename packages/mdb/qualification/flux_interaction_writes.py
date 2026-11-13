"""Frozen source-assisted operand compositions and bounded native futures."""
MIXES=('Alinear','Mlinear','AMlinear','Aopposite')
CASES=('sham','Alinear','Mlinear','AMlinear','Tlinear','AMTlinear','Aopposite','predictedAM')


def compose(concats,*,text_rows,width):
    import torch
    linear=(concats['10'].float()+concats['01'].float()-concats['00'].float()).to(concats['11'].dtype)
    result={}
    for case in MIXES:
        value=concats['11'].clone()
        if case in ('Alinear','AMlinear'):value[:,text_rows:,:width]=linear[:,text_rows:,:width]
        if case in ('Mlinear','AMlinear'):value[:,text_rows:,width:]=linear[:,text_rows:,width:]
        if case=='Aopposite':value[:,text_rows:,:width]=(2*concats['11'][:,text_rows:,:width].float()-linear[:,text_rows:,:width].float()).to(value.dtype)
        if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite composed operand')
        result[case]=value
    return result


def remaining(point):
    if point['phase']!='single' or point['logical_step'] not in (2,3) or not 0<=point['index']<19:raise ValueError('Writable single node required')
    return (3-point['logical_step'])*28+21-point['index']


def census(points):
    native=8*sum(remaining(p) for p in points.values())
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=native,acts=32,receipts=native+32,images=16,candidate_blocks=8,cached_projections=10)
