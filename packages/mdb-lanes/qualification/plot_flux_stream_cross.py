"""Plot source-qualified stream corner errors with per-row comparison scales."""
import argparse
import hashlib
import json
from pathlib import Path


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    a=json.loads(report.read_text());fields=[('input','image','Incoming image'),('input','text','Incoming text'),('attention','image','J2 attention image'),('attention','text','J2 attention text'),('joint_exit','image','J2 exit image'),('joint_exit','text','J2 exit text'),('noise','noise_prediction','Projected noise'),('latent','latent','Next latent commit')]
    corners=('00','10','01','11');fig,axes=plt.subplots(1,2,figsize=(13,6.3),layout='constrained')
    for ax,case in zip(axes,('row4heavy','row11heavy')):
        values=np.array([[100*a['measurements'][label][role][case]['corners'][k]['reference_response']['relative_rms'] for k in corners] for label,role,_ in fields]);delta=values-values[:,0:1]
        im=ax.imshow(delta,cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto')
        for y in range(8):
            for x in range(4):ax.text(x,y,f'{values[y,x]:.3f}%',ha='center',va='center',fontsize=11,color='white' if abs(delta[y,x])>1.35 else '#111')
        ax.set_xticks(range(4),['Old I\nOld T','New I\nOld T','Old I\nNew T','New I\nNew T']);ax.set_yticks(range(8),[x[2] for x in fields]);ax.set_title(case);ax.tick_params(length=0)
        for y in (1.5,3.5,5.5,6.5):ax.axhline(y,color='#333',linewidth=.6)
    colorbar=fig.colorbar(im,ax=axes,shrink=.85);colorbar.set_label('Change from old I / old T (percentage points)\nBlue improves, red worsens; compare within each row')
    shams=sum(x['status']=='qualified' for x in a['shams'].values());fig.suptitle('J2 image × text input interventions\nNumbers: paired-response error; each corner uses its own half baseline · '+str(shams)+'/3 sham controls exact',fontsize=14)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);fig.savefig(output.with_suffix('.svg'));plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();output.with_suffix('.provenance.json').write_text(json.dumps(dict(report=str(report),report_sha256=sha(report),png_sha256=sha(output),svg_sha256=sha(output.with_suffix('.svg')),source_sha256=sha(Path(__file__)),
        scope='Derived scalar plot; no model or image synthesis. Relative denominators differ by row/case. Color scale is a visualization range, not a scientific threshold. Source00shams and cached11/native qualification are preserved in the report.'),indent=2,sort_keys=True)+'\n')
    return dict(png=str(output),svg=str(output.with_suffix('.svg')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(plot(a.report.resolve(),a.output.resolve()),sort_keys=True))
