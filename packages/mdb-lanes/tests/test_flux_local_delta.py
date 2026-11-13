from pathlib import Path
import runpy
import numpy as np


def api(): return runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_local_delta_reference.py'))


def test_affine_increment_preserves_sub_bf16_delta_and_zero_rows():
    h=api(); x0=np.array([[[-1/256,2.],[3.,4.]]],np.float32); x=x0.copy(); x[0,0,0]=1
    w=np.array([[256.,0.],[0.,1.]],np.float32); y0=np.array([[[300.,2.],[800.,4.]]],np.float32)
    y,rows=h['affine_delta'](x,x0,y0,w,lambda a:a.astype(np.float32),chunk=1)
    np.testing.assert_array_equal(y,[[[557.,2.],[800.,4.]]]); assert rows==1
    z,rows=h['affine_delta'](x0,x0,y0,w,lambda a:a.astype(np.float32)); np.testing.assert_array_equal(z,y0); assert rows==0


def test_normalization_uses_new_operand_and_preserves_native_anchor_defect():
    h=api(); x0=np.array([[1.,2.],[3.,4.]],np.float32); x=x0.copy(); x[0]=[2,3]; y0=x0*x0+7
    y=h['normalized_delta'](x,x0,y0,lambda a:a*a,lambda a:a.astype(np.float32))
    np.testing.assert_array_equal(y,[[11,16],[16,23]])


def test_image_forecast_uses_fixed_source_supports_and_separate_background():
    h=runpy.run_path(str(Path(__file__).parents[1]/'qualification/flux_local_delta_campaign.py'))
    source=np.full((512,512,3),255,np.uint8); source[100:200,100:200]=[255,0,0]; source[100:200,300:400]=[0,0,255]
    target=source.copy(); target[100:200,100:200]=[0,0,255]
    row=h['scene_readout'](target,source)
    assert row['regions']['left_source_object']['red_minus_blue_delta']==-510
    assert row['regions']['right_source_object']['red_minus_blue_delta']==0
    assert row['background_outside_source_boxes_mae']==0
    target[0,0]=0
    assert h['scene_readout'](target,source)['background_outside_source_boxes_mae']>0
