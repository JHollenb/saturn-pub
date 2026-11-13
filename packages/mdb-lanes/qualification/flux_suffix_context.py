"""Fixed complete image pairs crossed with native text and latent contexts."""
import itertools


def cases(seeds=('60666','60667')):
    return [dict(id=f'{pair}.I{bit}.T{text}.Z{latent}',pair=pair,bit=bit,text=text,latent=latent)
        for pair,bit,text,latent in itertools.product(seeds,(0,1),seeds,seeds)]


def census():
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=48,acts=32,receipts=80,images=16,candidate_blocks=8,euler_forecasts=16)


def shared_closure(cuts):
    from saturn_pub.values import describe
    seeds=list(cuts);first=cuts[seeds[0]]
    for cut in cuts.values():
        if cut.model_identity!=first.model_identity or cut.execution!=first.execution:raise ValueError('Receiver lane differs')
        if any(cut.execution_point[k]!=v for k,v in {'logical_step':3,'phase':'single','index':19}.items()):raise ValueError('Requires native pre-single19')
        if set(cut.payload)!=set(first.payload):raise ValueError('Receiver schema differs')
        for key in first.payload:
            if key not in {'image','text','latent'} and describe(cut.payload[key])!=describe(first.payload[key]):raise ValueError('Undeclared receiver difference: '+key)
    return True
