"""Finite stream-factorial response accounts, without model frameworks."""
import numpy as np


def factorial(worlds,native,case,power):
    keys=('00','10','01','11');responses={k:np.asarray(worlds[case,k],np.float64)-np.asarray(worlds['half',k],np.float64) for k in keys}
    nd=np.asarray(native[case],np.float64)-np.asarray(native['half'],np.float64)
    if len({x.shape for x in [nd,*responses.values()]})!=1:raise ValueError('same factorial response coordinates required')
    image=responses['10']-responses['00'];text=responses['01']-responses['00'];interaction=responses['11']-responses['10']-responses['01']+responses['00']
    terms=dict(image_switch=image,text_switch=text,image_text_interaction=interaction)
    return dict(program_change=power(terms,responses['11']-responses['00']),
        revised_error=power(dict(old_error=responses['00']-nd,**terms),responses['11']-nd),
        native_effect_rms=float(np.sqrt(np.mean(nd*nd))),scope='Exact finite input-corner intervention account on these states; signed terms are not positive causal shares or independent predictions.')
