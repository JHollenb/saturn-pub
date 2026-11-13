"""Paired finite-write and Euler error accounts, in explicitly matching coordinates."""
import numpy as np


def response_error(worlds, case, mode):
    arrays=[np.asarray(worlds[key],np.float64) for key in [('half','native'),(case,'native'),('half',mode),(case,mode)]]
    if len({x.shape for x in arrays})!=1 or not arrays[0].size or any(not np.isfinite(x).all() for x in arrays):raise ValueError('finite identical response coordinates required')
    n0,n1,p0,p1=arrays
    return (p1-p0)-(n1-n0)


def finite_write(before, after, case, mode, power):
    carried=response_error(before,case,mode)
    final=response_error(after,case,mode)
    if carried.shape!=final.shape:raise ValueError('residual coordinate change requires a separate readout map')
    return power(dict(carried_response_error=carried,finite_write_response_error=final-carried),final)


def euler_account(inputs, noise, outputs, h, case, mode, power):
    if not np.isfinite(h) or set(inputs)!=set(noise) or set(inputs)!=set(outputs):raise ValueError('one finite Euler coefficient and matching worlds required')
    remainder={key:np.asarray(outputs[key],np.float64)-np.asarray(value,np.float64)-h*np.asarray(noise[key],np.float64) for key,value in inputs.items()}
    return power(dict(input_latent=response_error(inputs,case,mode),step_scaled_noise=h*response_error(noise,case,mode),
        finite_scheduler=response_error(remainder,case,mode)),response_error(outputs,case,mode))
