"""Derived fixed-scale response-error maps from authenticated original PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def plot(data,output):
    audit=json.loads((data/'audit-v1.json').read_text())
    rows=json.loads((data/'native-proof/rows.json').read_text());images={}
    for row in rows:
        p=data/'images'/row['png']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['png_sha256']:raise ValueError('source PNG differs')
        images[row['id'],row['mode']]=np.asarray(Image.open(p).convert('RGB')).astype(float)
    board=Image.new('RGB',(4*320,2*350+48),'white');draw=ImageDraw.Draw(board)
    draw.text((8,6),'Per-pixel channel RMS, 0-255 units. First three: white=0, dark blue>=8. Last: blue=new better, red=worse; saturates at +/-1.',fill='black')
    def magnitude(v):
        t=np.clip(v/8,0,1)[...,None];return (255*(1-t)+np.array([25,70,170])*t).astype(np.uint8)
    def signed(v):
        t=np.clip(np.abs(v),0,1)[...,None];color=np.where((v>0)[...,None],np.array([200,40,40]),np.array([35,85,195]));return (255*(1-t)+color*t).astype(np.uint8)
    counts={}
    for y,name in enumerate(('row4heavy','row11heavy')):
        native=images[name,'native']-images['half','native']
        old=(images[name,'old']-images['half','old'])-native
        new=(images[name,'new']-images['half','new'])-native
        nr,ore,nre=[np.sqrt(np.mean(value*value,axis=-1)) for value in (native,old,new)]
        counts[name]=dict(improved=int((nre<ore).sum()),worsened=int((nre>ore).sum()),unchanged=int((nre==ore).sum()))
        panels=[('native edit RMS',magnitude(nr)),('old response error',magnitude(ore)),('new response error',magnitude(nre)),('new minus old error RMS',signed(nre-ore))]
        for x,(label,rgb) in enumerate(panels):
            origin=(x*320,y*350+48);board.paste(Image.fromarray(rgb).resize((320,320)),(origin[0],origin[1]+22));draw.text((origin[0]+4,origin[1]+4),name+' | '+label,fill='black')
    output.parent.mkdir(parents=True,exist_ok=True);board.save(output)
    metadata=dict(schema='mdb-fresh-mlp-response-figure-v1',audit_sha256=hashlib.sha256((data/'audit-v1.json').read_bytes()).hexdigest(),figure_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),pixel_error_counts=counts,native_calls=0,
        scope='Derived display of original presealed candidate and native PNGs. Fixed common scales, no new model execution. Pixel counts compare complete per-pixel squared response error, not semantic quality.')
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n');return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(plot(a.data,a.output),sort_keys=True))
