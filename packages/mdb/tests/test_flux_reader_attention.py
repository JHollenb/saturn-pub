"""Small mechanics only; real CUDA/source parity is measured by the assay."""
import runpy
from pathlib import Path

import torch

C = runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_reader_attention.py'))


def test_synthetic_query_rows_preserve_actual_queries_and_independent_outputs():
    generator = torch.Generator().manual_seed(13)
    q = torch.randn(1,3,2,4,generator=generator,dtype=torch.float64)
    k,v = [torch.randn(1,7,2,4,generator=generator,dtype=torch.float64) for _ in range(2)]
    results = []
    for mode in ('text','zero','repeat'):
        queries = C['queries'](q,7,mode)
        assert torch.equal(queries[:,:3],q)
        result = torch.nn.functional.scaled_dot_product_attention(
            queries.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3))
        results.append(result[:,:,:3])
    for result in results[1:]:torch.testing.assert_close(result,results[0],rtol=1e-12,atol=1e-12)


def test_lanes_retain_inputs_and_count_actual_kernel_or_explicit_operations():
    generator = torch.Generator().manual_seed(14)
    q = torch.randn(1,3,2,4,generator=generator,dtype=torch.bfloat16)
    k,v = [torch.randn(1,7,2,4,generator=generator,dtype=torch.bfloat16) for _ in range(2)]
    original = [x.clone() for x in (q,k,v)]
    counts = dict(sdpa=0,staged_matmuls=0,staged_softmax=0)
    for lane in ('auto-text','auto-full-zero','auto-full-repeat','math-text','staged-fp32','staged-probability-bf16'):
        output,profile = C['attention'](q,k,v,lane,counts)
        assert output.shape == (1,3,8) and output.dtype == q.dtype and torch.isfinite(output).all()
        assert profile['selected_aten_operations']
    assert counts == dict(sdpa=4,staged_matmuls=4,staged_softmax=2)
    for actual,expected in zip((q,k,v),original):assert torch.equal(actual,expected)
