"""Render two figures from authenticated scalar reports, without model arrays."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
PACK = ROOT / 'compact-collection-01/extracted/binding-minimal-result'
MANIFEST = '82032b7ad875735d14a3c3f2ac2cc7dbb9bc342b3ae9373f2a8fbe349f06906f'
VERIFICATION = 'd3a5a75542be135a0e788eb3f6c90239193cdff78b4d5c27b67a4ea461c70108'
PORTS = ('head', 'R', 'M', 'Hnext')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    if sha(PACK / 'files.json') != MANIFEST or sha(ROOT / 'saved-report-verification-v1/verification.json') != VERIFICATION:
        raise ValueError('Authenticated report identity differs')
    manifest = read(PACK / 'files.json')
    if {p.relative_to(PACK).as_posix() for p in PACK.rglob('*') if p.is_file()} != set(manifest) | {'files.json'}:
        raise ValueError('Compact inventory differs')
    for name, digest in manifest.items():
        path = PACK / name
        if path.is_symlink() or sha(path) != digest or path.suffix in ('.npy', '.pt', '.pth', '.safetensors'):
            raise ValueError('Nonmetadata or changed member: ' + name)
    audit = read(PACK / 'audit.json')
    output = ROOT / 'publication-v1'
    output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, arm, title in zip(axes, ('all-negative', 'prefix-negative'), ('All rows negative', 'Only rows 0–255 negative')):
        signs = -np.ones(512, dtype=np.int8)
        if arm == 'prefix-negative':
            signs[256:] = 1
        allowed = np.tri(512, dtype=bool)
        cross = allowed & (signs[:, None] != signs[None, :])
        grid = allowed.astype(np.int8) + cross.astype(np.int8)
        ax.imshow(grid, origin='upper', interpolation='nearest', extent=(-.5, 511.5, 511.5, -.5),
                  cmap=ListedColormap(['#f1f3f5', '#3979a6', '#df913a']), vmin=0, vmax=2)
        ax.set(title=title, xlabel='Key row j', ylabel='Query row i', xticks=[0, 255, 511], yticks=[0, 255, 511])
        ax.axhline(255.5, color='white', lw=1)
        ax.axvline(255.5, color='white', lw=1)
        count = audit['arms'][arm]['support']['closed_rows']
        ax.text(.5, -.18, f'{count}/512 support-closed queries; {int(cross.sum()):,} cross-sign edges',
                transform=ax.transAxes, ha='center', fontsize=9)
    fig.suptitle('Permitted causal reads and their row-sign relation', fontsize=14)
    fig.legend(handles=[Patch(color='#f1f3f5', label='Forbidden'), Patch(color='#3979a6', label='Permitted, same sign'),
                        Patch(color='#df913a', label='Permitted, opposite signs')], loc='outside lower center', ncols=3)
    fig.savefig(output / 'support-boundary.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fields = audit['arms']['prefix-negative']['channels']['shortcut-minus-native']['fields']
    native = [fields[name]['right_effect']['windows']['suffix']['rms'] for name in PORTS]
    error = [fields[name]['response_error']['windows']['suffix']['rms'] for name in PORTS]
    endpoint = [100 * fields[name]['endpoint']['whole']['relative_rms'] for name in PORTS]
    response = [100 * fields[name]['response_error']['windows']['suffix']['relative_rms'] for name in PORTS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), constrained_layout=True)
    x = np.arange(4)
    axes[0].bar(x - .18, native, .34, label='Actual native response', color='#3979a6')
    axes[0].bar(x + .18, error, .34, label='Shortcut response error', color='#df913a')
    for index, value in enumerate(native):
        axes[0].text(index, max(value, error[index]) + .008, f'{value:.4f}', ha='center', fontsize=9)
    axes[0].set(title='Later rows 256–511: the whole response is missed', ylabel='RMS in each port’s coordinate units',
                xticks=x, xticklabels=PORTS, ylim=(0, .34))
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.12), fontsize=9)
    axes[1].bar(x - .18, endpoint, .34, label='Whole-state endpoint error / native state', color='#648979')
    axes[1].bar(x + .18, response, .34, label='Later-row response error / native response', color='#df913a')
    for index, value in enumerate(endpoint):
        axes[1].text(index - .18, value + 2, f'{value:.2f}%', ha='center', fontsize=9)
    axes[1].set(title='Different denominators answer different questions', ylabel='Relative RMS (%)',
                xticks=x, xticklabels=PORTS, ylim=(0, 115))
    axes[1].legend(loc='upper left', bbox_to_anchor=(0, -.12), fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('The unchanged rows carry a real attention-mediated state update', fontsize=14)
    fig.savefig(output / 'missed-response.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    rows = {name: {term: fields[name][term] for term in ('endpoint', 'response_error', 'left_effect', 'right_effect')} for name in PORTS}
    save(output / 'publication.json', dict(schema='mdb-qwen-row-sign-publication-v1', native_job=audit['job_id'],
        native_request_sha256=audit['request_sha256'], compact_manifest_sha256=MANIFEST, audit_sha256=sha(PACK / 'audit.json'),
        verification_sha256=VERIFICATION, source_baseline_errors=audit['source_baseline_errors'],
        prefix_arm_fields=rows, transaction=audit['transaction'],
        scope='Figures from authenticated scalar reports. No activation, checkpoint, tokenizer, neural recurrence or original model call.',
        structural_graph='Permitted mask and sign relations only; no observed attention weights or min-cut capacities.',
        region_scope='Whole and fixed windows overlap; charts are descriptive, not independent trials or accuracy gates.'))
    save(output / 'publication-receipt.json', dict(schema='mdb-qwen-row-sign-publication-receipt-v1',
        source_sha256=sha(Path(__file__)), compact_manifest_sha256=MANIFEST, verification_sha256=VERIFICATION,
        outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in output.iterdir() if p.is_file()},
        raw_arrays_read=0, learned_calls=0, model_calls=0, visual_inspection='Pending root view_image after rendering.'))
    print(json.dumps({p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in output.iterdir() if p.is_file()}, sort_keys=True))


if __name__ == '__main__':
    main()
