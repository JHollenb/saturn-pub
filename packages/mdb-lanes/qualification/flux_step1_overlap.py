"""Clock-explicit single6 R/A/M image exits and a common continuing shape program."""


def census():return dict(loads=1,encodes=0,noise_draws=0,native_transitions=1292,acts=36,receipts=1328,images=18,candidate_blocks=8,cached_projections=4)


def gated_exit(residual,gate,projection):
    import runpy
    from pathlib import Path
    B=runpy.run_path(str(Path(__file__).with_name('flux_conditioning_program.py')))['bf16']
    return B(residual.astype('float64')+B(gate.astype('float64')*projection))


def selected_concat(base,shape,*,bits,text_rows,width):
    result=base.clone()
    if bits[0]=='1':result[:,text_rows:,:width]=shape[:,text_rows:,:width]
    if bits[1]=='1':result[:,text_rows:,width:]=shape[:,text_rows:,width:]
    return result
