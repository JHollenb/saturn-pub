"""Plot saved paired-response measurements; no model execution or image synthesis."""
import argparse
import hashlib
import json
from pathlib import Path


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(report.read_text());rows=data['curves'][:27]
    labels=[r['id'].replace('joint','J').replace('single','S').replace('.attention',' A').replace('.exit','') for r in rows]
    figure,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,layout='constrained')
    for column,case in enumerate(('row4heavy','row11heavy')):
        for y,role in enumerate(('image','text')):
            ax=axes[y,column]
            for mode,color in [('old','#b85c00'),('new','#1464b5')]:
                ax.plot(range(27),[100*r['roles'][role][case][mode]['reference_response']['relative_rms'] for r in rows],label=mode,color=color,marker='o',markersize=3,linewidth=1.8)
            ax.axvspan(-.3,6.5,color='#e9edf2',alpha=.6,zorder=-2);ax.set_title(case+' · '+role+' residual')
            reversal=data['first_observed_reversals'][role][case]['first_observed_new_worse']
            if reversal and reversal['id'] in [r['id'] for r in rows]:
                index=[r['id'] for r in rows].index(reversal['id']);ax.axvline(index,color='#555',linestyle=':',linewidth=1)
                ax.text(index+.2,.98,'first new > old: '+labels[index],transform=ax.get_xaxis_transform(),va='top',fontsize=8,color='#333')
            ax.set_ylabel('Paired-response RMS error / native edit RMS (%)');ax.grid(alpha=.2);ax.legend(frameon=False,loc='lower right');ax.set_xlim(-.3,26.3)
            ax.set_xticks(range(27),labels,rotation=65,ha='right',fontsize=8);ax.tick_params(labelbottom=True)
    qualified=sum(value=='qualified' for value in data['replay_status'].values());total=len(data['replay_status'])
    figure.suptitle('Native consumer trace of the original futures\nEach program uses its own baseline; '+str(qualified)+'/'+str(total)+' complete next-commit payloads replay exactly',fontsize=15)
    output.parent.mkdir(parents=True,exist_ok=True);figure.savefig(output,dpi=170);figure.savefig(output.with_suffix('.svg'));plt.close(figure)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    output.with_suffix('.provenance.json').write_text(json.dumps(dict(report=str(report),report_sha256=sha(report),png_sha256=sha(output),svg_sha256=sha(output.with_suffix('.svg')),source_sha256=sha(Path(__file__)),
        scope='Derived plot of saved retrospective response measurements. Relative denominators vary by boundary; curves compare program rankings, not amplification across coordinates. Joint A denotes captured post-attention residual. Single blocks retain complete-block resolution. Norm, projection and Euler have separate reported coordinate systems.'),indent=2,sort_keys=True)+'\n')
    return dict(png=str(output),svg=str(output.with_suffix('.svg')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(plot(a.report.resolve(),a.output.resolve()),sort_keys=True))
