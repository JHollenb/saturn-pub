"""Branch-minus-one-item ablation: revert one speedup to main's behaviour by monkeypatch
(ABLATE env: comma list of 1,2,3,6,7,8), then exec bench.py with the remaining argv."""
import os, runpy, sys
import torch
which = set(filter(None, os.environ.get("ABLATE", "").split(",")))
import saturn_pub.values as sv
import saturn_pub.core as sc
import saturn_pub.adapters.decoder as dec
from transformers import DynamicCache
if "1" in which:
    sc.describe_remembered = sv.describe
if "2" in which:
    def _all_finite(self, tensors):
        if not all(isinstance(t, torch.Tensor) and t.device == self.device for t in tensors):
            return None
        if not tensors:
            return True
        return bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())
    dec.DecoderAdapter._all_finite = _all_finite
if "3" in which:
    def _cache(self, keys, values):
        cache = DynamicCache(config=self.model.config)
        for index, (key, value) in enumerate(zip(keys, values)):
            if key.shape[-2]:
                cache.update(key.to(self.device), value.to(self.device), index)
        return cache
    dec.DecoderAdapter._cache = _cache
if "6" in which:
    def _versions(value):
        out = []
        for path, item in sc._tensor_leaves(value):
            if sc._versioned(item):
                out.append((path, item, item._version))
        return tuple(out)
    sc._versions = _versions
if "7" in which:
    sc.Session._sealed_again = lambda self, *a: None
    sv._PLAIN_KEY = __import__("re").compile(r"(?!x)x")  # never matches: canonical sort
if "8" in which:
    dec._FrozenSnapshot.holds = lambda self: False
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
