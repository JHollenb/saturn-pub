"""Figures from authenticated scalar/QKV reports; no model or raw arrays."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PACK = ROOT / 'compact-collection-01/extracted/binding-minimal-result'
MANIFEST = 'c662757ab5f261b91f72006bd3de57291e1ff70e208c38e9e30631cb6c1a3ca7'
VERIFIER = '733788bb8602639868592223bfeb50dea0d4998f0807e040398e2481fb1f057e'
ARMS = ('last-row-negative', 'alternating64')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    if sha(PACK / 'files.json') != MANIFEST or sha(ROOT / 'saved-report-verification-v1/verification.json') != VERIFIER:
        raise ValueError('External report authentication differs')
    manifest = json.loads((PACK / 'files.json').read_text())
    for name, digest in manifest.items():
        path = PACK / name
        if path.is_symlink() or sha(path) != digest or path.suffix in ('.npy', '.npz', '.pt', '.pth', '.safetensors'):
            raise ValueError('Changed or nonmetadata member: ' + name)
    audit = json.loads((PACK / 'audit.json').read_text())
    output = ROOT / 'publication-v1'
    output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    plotted = {}
    for ax, arm, title in zip(axes, ARMS, ('Only input row 511 negative', 'Alternating 64-row input signs')):
        fields = audit['arms'][arm]['channels']['cached-qkv-minus-native']
        windows = fields['Hnext']['right_effect']['windows']
        x = np.arange(len(windows))
        labels = [f"{w['bounds'][0]}:{w['bounds'][1]}" for w in windows]
        h = [w['comparison']['rms'] for w in windows]
        m = [w['comparison']['rms'] for w in fields['M']['right_effect']['windows']]
        ax.bar(x - .18, h, .34, color='#3979a6', label='H19 state response')
        ax.bar(x + .18, m, .34, color='#37846b', label='MLP write response')
        ax.set(title=title, xticks=x, xticklabels=labels, ylabel='Native edited-minus-baseline RMS', xlabel='Query window [start:stop)')
        ax.set_yscale('symlog', linthresh=.1)
        ax.set_ylim(0, 45 if arm == 'alternating64' else 3)
        ax.tick_params(axis='x', labelrotation=30 if len(x) > 2 else 0, labelsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.text(.5, -.27, 'All complete predicted outputs and signed responses are bit-exact.',
                transform=ax.transAxes, ha='center', fontsize=9)
        plotted[arm] = dict(windows=[w['bounds'] for w in windows], native_H19_response_rms=h, native_M_response_rms=m)
    axes[0].legend(loc='upper left')
    fig.suptitle('We predicted the changed states, including unchanged input rows that read edits', fontsize=13)
    fig.savefig(output / 'predicted-native-responses.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    zero_counts = {}
    x = np.arange(2)
    for port, offset, color in (('q', -.19, '#3979a6'), ('k', .19, '#df913a')):
        values = [audit['arms'][arm]['positioned_QKV']['cached-qkv-minus-native'][port]['whole']['signed_zero']['defects'] for arm in ARMS]
        axes[0].bar(x + offset, values, .36, color=color, label=port.upper())
        for index, value in enumerate(values):
            axes[0].text(index + offset, value + 6, str(value), ha='center')
        zero_counts[port] = values
    axes[0].set(title='Q/K zero-sign differences are retained', ylabel='Raw-bit defects with zero numerical error',
                xticks=x, xticklabels=['Last row', 'Alternating 64'], ylim=(0, 420))
    axes[0].grid(axis='y', alpha=.2)
    axes[0].set_axisbelow(True)
    axes[0].legend()
    axes[1].axis('off')
    rows = []
    for arm in ARMS:
        for port in ('head', 'R', 'M', 'Hnext'):
            field = audit['arms'][arm]['channels']['cached-qkv-minus-native'][port]
            rows.append(['Last row' if arm == ARMS[0] else 'Alt. 64', port,
                         f"{field['endpoint']['whole']['rms']:.0f}", f"{field['response_error']['whole']['rms']:.0f}",
                         'Yes' if field['endpoint']['whole']['bitwise_exact'] else 'No'])
    table = axes[1].table(cellText=rows, colLabels=['Edit', 'Port', 'State\nerror', 'Response\nerror', 'Bit exact'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    axes[1].set_title('Observed downstream reconciliation', pad=15)
    fig.suptitle('Numerical operand parity does not imply raw-bit parity at every port', fontsize=13)
    fig.savefig(output / 'finite-zero-sign-boundary.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    save(output / 'publication.json', dict(schema='mdb-qwen-cached-qkv-publication-v1', native_job=audit['job_id'],
        audit_sha256=sha(PACK / 'audit.json'), compact_manifest_sha256=MANIFEST, verifier_sha256=VERIFIER,
        native_response_windows=plotted, QK_signed_zero_defects=zero_counts,
        scope='Plot coordinates come from authenticated complete saved scalar accounts. Windows/ports/channels share operands. Input doses differ; no intrinsic gain, semantic object, independent-kernel or speedup claim.',
        arrays_read=0, learned_calls=0, model_calls=0))
    save(output / 'publication-receipt.json', dict(schema='mdb-qwen-cached-qkv-publication-receipt-v1',
        source_sha256=sha(Path(__file__)), outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in output.iterdir() if p.is_file()},
        source_manifest_sha256=MANIFEST, verifier_sha256=VERIFIER, visual_inspection='Pending root view_image.'))
    print(json.dumps({p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in output.iterdir() if p.is_file()}))


if __name__ == '__main__':
    main()
