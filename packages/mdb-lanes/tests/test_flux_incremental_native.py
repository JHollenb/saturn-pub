"""Framework-free proof that cache hits skip bodies and restore on errors."""
from pathlib import Path
import runpy
import pytest


class Module:
    def forward(self,*args,**kwargs): raise AssertionError('original body must not run')


class Value:
    def clone(self): return 'cached-output'


def test_cache_bypasses_original_body_and_restores_class_descriptor():
    api=runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_incremental_native.py'))
    m=Module(); rows=[]
    def check(name,args,kwargs,expected):
        if args!=(expected,) or kwargs: raise ValueError('changed call')
    with api['reuse_prefix']([('q',m,'expected',Value())],check,rows):
        assert m.forward('expected')=='cached-output'
        with pytest.raises(ValueError,match='changed call'): m.forward('expected',surprise=True)
    assert 'forward' not in m.__dict__
    assert len(rows)==1 and rows[0]['original_forward_executed'] is False
    with pytest.raises(AssertionError): m.forward('expected')


def test_original_instance_forward_restored_after_failure():
    api=runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_incremental_native.py'))
    m=Module(); original=lambda x:'instance'; m.forward=original
    def reject(*args): raise ValueError('mismatch')
    with pytest.raises(ValueError):
        with api['reuse_prefix']([('q',m,'expected',Value())],reject,[]): m.forward('wrong')
    assert m.forward is original
