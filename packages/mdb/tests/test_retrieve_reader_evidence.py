"""CLI aliases come from the evidence manifest; cached files still need custody."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).parents[1] / 'qualification/retrieve_reader_evidence.py'


def test_companion_manifest_alias_works_and_corruption_is_refused(tmp_path):
    destination = tmp_path / 'retrieved'
    destination.mkdir()
    target = destination / 'ports.bin'
    target.write_bytes(b'original model evidence')
    manifest = tmp_path / 'external-evidence.json'
    manifest.write_text(json.dumps(dict(host='not-contacted', roots=dict(control='/remote/original'),
        manifests=dict(control={'ports.bin':hashlib.sha256(target.read_bytes()).hexdigest()}))))
    args = [sys.executable, '-I', str(SCRIPT), 'control', 'ports.bin', str(destination), '--manifest', str(manifest)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0 and str(target) in result.stdout
    target.write_bytes(b'changed bytes')
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0 and 'existing evidence checksum differs' in result.stderr
