"""Stable exact-arithmetic attention updates for explicitly changed key rows.

NumPy is optional and loaded on use. The cache contains calculated unchanged
contributions, never inferred semantic labels or native attention observations.
Callers bind model/state/position/dtype provenance and qualify kernel rounding.
"""
from dataclasses import dataclass
import math


def _require(ok,message):
    if not ok:raise ValueError(message)


def _indices(values,size,label):
    result=tuple(values)
    _require(all(type(i)is int and 0<=i<size for i in result)and tuple(sorted(set(result)))==result,"invalid "+label)
    return result


def _array(value):
    import numpy as np
    result=np.asarray(value,dtype=np.float64)
    _require(result.ndim==4 and result.size>0 and np.isfinite(result).all(),"finite nonempty B/H/row/feature arrays required")
    return result


def _frozen(value):
    import numpy as np
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


def dense_attention(q,k,v,*,scale=None,query_chunk=64):
    """Independent dense reference, with a stable softmax for every query."""
    import numpy as np
    q,k,v=map(_array,(q,k,v))
    _require(q.shape[:2]==k.shape[:2]==v.shape[:2]and q.shape[-1]==k.shape[-1]and k.shape[-2]==v.shape[-2],"attention geometry differs")
    scale=1/math.sqrt(q.shape[-1])if scale is None else scale
    _require(type(scale)in(int,float)and math.isfinite(scale)and scale>0 and type(query_chunk)is int and query_chunk>0,"invalid attention execution settings")
    result=np.empty((*q.shape[:-1],v.shape[-1]),np.float64)
    for first in range(0,q.shape[-2],query_chunk):
        sl=slice(first,first+query_chunk);scores=(q[:,:,sl]@k.swapaxes(-1,-2))*scale
        p=np.exp(scores-scores.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
        result[:,:,sl]=p@v
    return result


@dataclass(frozen=True)
class AttentionPartition:
    q:object
    k:object
    v:object
    mutable_rows:tuple
    maximum:object
    denominator:object
    numerator:object
    scale:float
    query_chunk:int

    def evaluate(self,key_rows,value_rows,*,query_rows=None,query_indices=()):
        """Replace all declared mutable K/V rows and any declared query rows.

        Unchanged contributions are combined using a shared maximum. No nearly
        equal numerator subtraction occurs. Changed queries are recomputed
        against every key, rather than incorrectly reusing their old scores.
        """
        import numpy as np
        k,v=map(_array,(key_rows,value_rows));ix=_indices(query_indices,self.q.shape[-2],"query rows")
        _require(k.shape==(*self.k.shape[:2],len(self.mutable_rows),self.k.shape[-1])and v.shape==(*self.v.shape[:2],len(self.mutable_rows),self.v.shape[-1]),"replacement key/value geometry differs")
        _require((query_rows is None)==(not ix),"query values and indices must be supplied together")
        replacement_q=None if query_rows is None else _array(query_rows)
        if replacement_q is not None:_require(replacement_q.shape==(*self.q.shape[:2],len(ix),self.q.shape[-1]),"replacement query geometry differs")
        result=np.empty_like(self.numerator)
        for first in range(0,self.q.shape[-2],self.query_chunk):
            sl=slice(first,first+self.query_chunk);scores=(self.q[:,:,sl]@k.swapaxes(-1,-2))*self.scale
            maximum=np.maximum(self.maximum[:,:,sl],scores.max(axis=-1,keepdims=True))
            factor=np.exp(self.maximum[:,:,sl]-maximum)
            p=np.exp(scores-maximum)
            numerator=self.numerator[:,:,sl]*factor+p@v
            denominator=self.denominator[:,:,sl]*factor+p.sum(axis=-1,keepdims=True)
            _require(np.all(denominator>0)and np.isfinite(numerator).all(),"invalid partition sum")
            result[:,:,sl]=numerator/denominator
        if ix:
            all_k=self.k.copy();all_v=self.v.copy()
            all_k[:,:,self.mutable_rows,:]=k;all_v[:,:,self.mutable_rows,:]=v
            result[:,:,ix,:]=dense_attention(replacement_q,all_k,all_v,scale=self.scale,query_chunk=self.query_chunk)
        return result,dict(queries=self.q.shape[-2],keys=self.k.shape[-2],mutable_keys=len(self.mutable_rows),changed_queries=len(ix),
            query_key_pairs_per_head=self.q.shape[-2]*len(self.mutable_rows)+len(ix)*self.k.shape[-2],
            dense_query_key_pairs_per_head=self.q.shape[-2]*self.k.shape[-2],
            scope="Work geometry excludes cache preparation, QKV/output projections, MLPs, rounding, custody and native validation; not a measured speedup.")


def prepare_attention_partition(q,k,v,mutable_rows,*,scale=None,query_chunk=64):
    """Calculate the unchanged-key numerator, denominator and maximum once."""
    import numpy as np
    q,k,v=map(_array,(q,k,v))
    _require(q.shape[:2]==k.shape[:2]==v.shape[:2]and q.shape[-1]==k.shape[-1]and k.shape[-2]==v.shape[-2],"attention geometry differs")
    ix=_indices(mutable_rows,k.shape[-2],"mutable key rows");_require(bool(ix),"mutable rows required")
    scale=1/math.sqrt(q.shape[-1])if scale is None else scale
    _require(type(scale)in(int,float)and math.isfinite(scale)and scale>0 and type(query_chunk)is int and query_chunk>0,"invalid attention execution settings")
    retained=[i for i in range(k.shape[-2])if i not in ix]
    numerator=np.zeros((*q.shape[:-1],v.shape[-1]),np.float64)
    denominator=np.zeros((*q.shape[:-1],1),np.float64);maximum=np.full_like(denominator,-np.inf)
    if retained:
        keys=k[:,:,retained,:];values=v[:,:,retained,:]
        for first in range(0,q.shape[-2],query_chunk):
            sl=slice(first,first+query_chunk);scores=(q[:,:,sl]@keys.swapaxes(-1,-2))*scale
            m=scores.max(axis=-1,keepdims=True);p=np.exp(scores-m)
            numerator[:,:,sl]=p@values;denominator[:,:,sl]=p.sum(axis=-1,keepdims=True);maximum[:,:,sl]=m
    return AttentionPartition(*map(_frozen,(q,k,v)),ix,*map(_frozen,(maximum,denominator,numerator)),float(scale),query_chunk)
