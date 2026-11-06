"""Verify the copied bytes, documentation links and actual portable MDB queries."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
ASSET = BASE / 'saturn-pub/packages/mdb/docs/assets/qwen-cached-qkv-20261106'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    copies = json.loads((ROOT / 'integration-copy-v1.json').read_text())
    for name, row in copies['copied'].items():
        path = ASSET / name
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Copied bytes differ: ' + name)
    blog = BASE / 'obsidian/blog/2026-11-06-090851-predicting-attention-from-transformed-operands.md'
    docs = [blog, BASE / 'saturn-pub/packages/mdb/docs/QWEN-CACHED-QKV.md', BASE / 'saturn-pub/packages/mdb/docs/QWEN-ROW-SIGN.md']
    links = []
    for path in docs:
        for _, target in re.findall(r'\[([^\]]*)\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http://', 'https://', '#')):
                continue
            if not (path.parent / target.split('#')[0]).resolve().exists():
                raise ValueError('Missing link: ' + target)
            links.append(dict(document=str(path.relative_to(BASE)), target=target))
    metadata = yaml.safe_load(blog.read_text().split('---', 2)[1])
    for field in ('source_docs', 'source_code', 'source_artifacts'):
        for target in metadata[field]:
            if not (blog.parent / target).resolve().is_file():
                raise ValueError('Missing source frontmatter: ' + target)
    commands = [[sys.executable, '-I', '-B', '-m', 'mdb', 'equations', 'find', '--catalog', str(ASSET / 'compact/catalog/equations'),
                 '--alias', 'cmp.qwen.cached-qkv.alternating64', '--applicable-to', str(ASSET / 'verification/cmp.qwen.cached-qkv.alternating64.exact.context.json')],
                [sys.executable, '-I', '-B', '-m', 'mdb.computation_atlas', '--atlas', str(ASSET / 'compact/atlas'), 'find',
                 '--law-id', 'law.qwen.positioned-QKV-row-sign.bf16.v1']]
    calls = []
    for index, command in enumerate(commands):
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        records = json.loads(result.stdout)
        name = f'documented-browse-{index + 1}.json'
        with (ROOT / name).open('x') as stream:
            stream.write(result.stdout)
        calls.append(dict(command=command, returncode=result.returncode, matches=len(records), output_sha256=sha(ROOT / name)))
        shutil.copyfile(ROOT / name, ASSET / 'workflow' / name)
    report = dict(schema='mdb-cached-qkv-docs-root-verification-v1', copied_files=copies['files'], copied_bytes=copies['bytes'],
        links=links, frontmatter_source_links=sum(len(metadata[field]) for field in ('source_docs', 'source_code', 'source_artifacts')),
        actual_documented_queries=calls, blog_sha256=sha(blog), figure_scope='Complete output fields mean head/R/M/H19. Q/K zero-sign differences are shown separately.',
        arrays_or_weights_copied=False)
    with (ROOT / 'docs-verification-v1.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    for name in ('verify_docs.py', 'docs-verification-v1.json'):
        shutil.copyfile(ROOT / name, ASSET / 'workflow' / name)
    print(json.dumps(dict(copied_files=copies['files'], links=len(links), frontmatter=report['frontmatter_source_links'],
        query_matches=[row['matches'] for row in calls], receipt_sha256=sha(ROOT / 'docs-verification-v1.json'))))


if __name__ == '__main__':
    main()
