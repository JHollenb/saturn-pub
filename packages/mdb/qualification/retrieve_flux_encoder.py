"""Verify or copy the exact original encoder/tokenizer files used by a run.

No parameter payload is stored in the evidence. Supply the original checkpoint
directory; hashes, rather than a mutable model name, define the requested files.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def retrieve(checkpoint,manifest,output=None):
    files=json.loads(manifest.read_text())['checkpoint_files'];result={}
    for name,expected in files.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('text_encoder','tokenizer'):
            raise ValueError('encoder path outside original checkpoint')
        source=checkpoint/relative
        with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=expected:raise ValueError('original encoder file differs: '+name)
        if output is not None:
            target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError(target)
            shutil.copyfile(source,target)
        result[name]=dict(sha256=digest,bytes=source.stat().st_size)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--manifest',type=Path,default=Path('encoder-source.json'));p.add_argument('--output',type=Path);a=p.parse_args()
    print(json.dumps(retrieve(a.checkpoint,a.manifest,a.output),sort_keys=True))
