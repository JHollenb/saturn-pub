"""Guard against losing signed cancellation or silently widening the flow product."""
import runpy
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]/'qualification'
A=runpy.run_path(str(HERE/'flux_commit_factorial_audit.py'))
B=runpy.run_path(str(HERE/'flux_reentry_reference.py'))['bf16']
C=runpy.run_path(str(HERE/'flux_commit_factorial.py'))


def test_actual_transaction_frontier_and_record_contract(tmp_path):
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction,load_forecast_tree
    import json
    root=tmp_path/'transaction'
    tx=ForecastTransaction(root,model_identity='tiny',input_identity={'source':'cut'},source_identity='script',
        frontier=C['forecast_frontier'](),execution_scope='Model-free runner contract check',
        budgets=dict(max_native_calls=28,max_bytes=32<<20,max_wall_ms=600000))
    tx.frontier('source-cuts',{'scheduler':'cut'})
    p={'latent':{'sha256':'tiny'},'values':{'shape':[1,1024,24,128]}}
    tx.predict('full',lambda _:C['forecast_record'](p))
    tx.predict('band',lambda _:C['forecast_record']({'producer':'full','values':p['values'],'band':[512,768],'receiving_parent':'cut'}))
    seal=tx.seal();verify_transaction(root,expected_seal_sha256=seal['seal_sha256'])
    frozen=load_forecast_tree(root,json.loads((root/'forecasts/band.json').read_text())['predictions'])
    assert A['same_record'](frozen['values'],p['values'])
    tx.execute('full',lambda _:dict(case='full',component='dummy'))
    verify_transaction(root,expected_seal_sha256=seal['seal_sha256']);tx.close()


def test_cancelling_parts_and_interaction_keep_energy_closure():
    z=np.array([1.,2.,3.]);n=-.75*z;j=np.array([.5,0.,.25]);b=np.zeros(3)
    r=A['factorial']({'00':b,'10':z,'01':n,'11':z+n+j})
    assert r['carry_flow_cross_energy']==-21.
    assert r['carry_projection']>1 and r['flow_projection']<0
    assert r['interaction_energy']==np.vdot(j,j)
    assert abs(r['energy_closure_remainder'])<1e-12


def test_noise_product_rounding_precedes_sum():
    z=np.array([-.30078125],np.float32);n=np.ones(1,np.float32);delta=np.float32(.3)
    out,product=A['finite_commit_numpy'](z,n,delta,B)
    assert product[0]==.30078125 and out[0]==0
    wrong=B(z+delta*n)
    assert wrong[0]!=out[0]


def test_cuda_scalar_conversion_is_part_of_qualified_profile():
    dt=np.float32(-.16648530960083008);z=np.array([.2490234375],np.float32);n=np.array([1.5],np.float32)
    out,product=A['finite_commit_numpy'](z,n,dt,B)
    assert product[0]==-.2490234375 and out[0]==0
    assert B(z+B(dt*n))[0]!=0


def test_zero_total_keeps_parts_and_abstains_from_projection():
    z=np.array([1.,-2.]);b=np.zeros(2)
    r=A['factorial']({'00':b,'10':z,'01':-z,'11':b})
    assert r['total_energy']==0 and r['carry_energy']==5 and r['flow_energy']==5
    assert r['carry_projection'] is None and r['flow_projection'] is None
    assert r['carry_flow_cross_energy']==-10
