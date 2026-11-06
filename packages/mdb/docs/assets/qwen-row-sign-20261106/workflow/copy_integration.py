"""Copy the authenticated compact and exact owned sources, without arrays."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
DEST = BASE / 'saturn-pub/packages/mdb/docs/assets/qwen-row-sign-20261106'
VAULT = BASE / 'obsidian/images/mdb-qwen-row-sign-20261106'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    VAULT.mkdir(parents=True, exist_ok=False)
    copied = {}
    def copy(source, relative):
        if not source.is_file() or source.is_symlink() or source.suffix in ('.npy', '.npz', '.pt', '.pth', '.safetensors', '.whl', '.bin'):
            raise ValueError('Nonmetadata source refused: ' + str(source))
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Integration destination already exists')
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise ValueError('Integration copy changed bytes')
        copied[str(relative)] = dict(source=str(source.relative_to(BASE)), bytes=source.stat().st_size, sha256=sha(source))
    for directory, prefix in (('compact-collection-01/extracted/binding-minimal-result', 'compact'),
                              ('publication-v1', 'publication'), ('saved-report-verification-v1', 'verification')):
        for source in sorted((ROOT / directory).rglob('*')):
            if source.is_file():
                copy(source, Path(prefix) / source.relative_to(ROOT / directory))
    for run in ('run-02', 'audit-01', 'audit-02'):
        for name in ('request.json', 'staged-command.json', 'admission-command.json', 'submission.json',
                     'finite-run-review-v1.json', 'job-completed.json', 'job-failed.json', 'job.log',
                     'source-job.json', 'source-cohort.json', 'source-worker.log', 'parent-metadata-verification.json'):
            source = ROOT / run / name
            if source.is_file():
                copy(source, Path('execution') / run / name)
        request = json.loads((ROOT / run / 'request.json').read_text())
        for name, expected in request['source_hashes'].items():
            source = ROOT / run / 'payload' / name
            if sha(source) != expected:
                raise ValueError('Executed source changed: ' + name)
            copy(source, Path('executed-source') / run / name)
    for directory in ('helper-qualification-v1', 'producer-qualification-v3', 'audit-qualification-v1',
                      'audit-qualification-v2', 'controller-qualification-v1', 'controller-qualification-v2',
                      'controller-qualification-v3', 'saved-report-verifier-qualification-v1'):
        for source in sorted((ROOT / directory).iterdir()):
            if source.is_file() and source.suffix in ('.json', '.log', '.txt', '.py', '.md'):
                copy(source, Path('qualification') / directory / source.name)
    for name in ('freeze_selection.py', 'select_parameter_recipes.py', 'review_native.py', 'collect_compact.py',
                 'verify_saved_reports.py', 'publish.py', 'copy_integration.py', 'visual-verification-v1.json',
                 'compact-collection-01/collection-receipt.json', 'controller-qualification-v2/prior-collector-b21e.py'):
        source = ROOT / name
        if source.is_file():
            copy(source, Path('workflow') / name)
    for name in ('holistic-review-10.json', 'direction-10.md', 'holistic-review-11.json', 'direction-11.md'):
        source = BASE / 'artifacts/mdb-flux-understanding-20261106' / name
        if source.is_file():
            copy(source, Path('advisory') / name)
    for name in ('support-boundary.png', 'missed-response.png'):
        source = ROOT / 'publication-v1' / name
        shutil.copyfile(source, VAULT / name)
        if sha(source) != sha(VAULT / name):
            raise ValueError('Vault figure differs')
    report = dict(schema='mdb-row-sign-integration-copy-v1', copied=copied, files=len(copied),
        bytes=sum(row['bytes'] for row in copied.values()),
        vault_figures={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in VAULT.iterdir()},
        arrays_or_parameter_exports=False)
    for path in (ROOT / 'integration-copy-v1.json', DEST / 'integration-copy.json'):
        with path.open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: report[key] for key in ('files', 'bytes', 'arrays_or_parameter_exports')}))


if __name__ == '__main__':
    main()
