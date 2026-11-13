"""Explicit live-state interaction observables for a four-corner native replay."""
BITS=('00','10','01','11')


def live_slots(point):
    result=['latent']
    if point['phase'] in ('joint','single','readout'):result+=['image','text']
    if point['phase']=='scheduler':result+=['noise_prediction']
    return result


def field(values):
    return values['11'].double()-values['10'].double()-values['01'].double()+values['00'].double()


def measure(values,previous=None):
    import torch
    c=field(values);d=values['11'].double()-values['00'].double();den=float((d*d).sum());flat=c.reshape(-1)
    indices=torch.linspace(0,flat.numel()-1,64,device=c.device).long().unique()
    row={'l2':float(torch.linalg.vector_norm(c)),'max_abs':float(c.abs().max()),'signed_mean':float(c.mean()),
        'source_contrast_l2':float(torch.linalg.vector_norm(d)),'source_contrast_projection':None if den==0 else float((c*d).sum())/den,
        'elements':c.numel(),'indices':indices.cpu().tolist(),'witnesses':{k:v.reshape(-1)[indices].float().cpu().tolist() for k,v in values.items()},'interaction_witnesses':flat[indices].cpu().tolist()}
    if previous is not None:
        delta=c-previous;row['write_interaction_l2']=float(torch.linalg.vector_norm(delta));row['write_interaction_signed_mean']=float(delta.mean())
    return row,c


def candidate(point):
    return point['logical_step'] in (2,3) and point['phase']=='single' and point['index']<19


def census():return dict(loads=1,encodes=0,noise_draws=0,native_transitions=448,acts=0,receipts=448,images=8,candidate_blocks=8)
