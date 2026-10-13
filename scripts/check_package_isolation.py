"""Check distribution boundaries and independent bare-wheel imports."""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


def one_artifact(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0].resolve()


def check_artifacts(directory: Path, namespace: str, sibling: str) -> Path:
    wheel = one_artifact(directory, "*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    if f"{namespace}/__init__.py" not in names:
        raise ValueError(f"{wheel.name} does not contain {namespace}")
    if any(name.startswith(f"{sibling}/") for name in names):
        raise ValueError(f"{wheel.name} contains sibling runtime {sibling}")
    if any(
        not name.startswith((f"{namespace}/", f"{wheel.name.split('-')[0]}-")) for name in names
    ):
        raise ValueError(f"{wheel.name} contains files outside its runtime/metadata")
    sdist = one_artifact(directory, "*.tar.gz")
    with tarfile.open(sdist) as archive:
        paths = [Path(member.name) for member in archive.getmembers() if member.isfile()]
    if not any(path.parts[1:] == ("src", namespace, "__init__.py") for path in paths):
        raise ValueError(f"{sdist.name} does not contain its source runtime")
    if any(sibling in path.parts or path.parts[1:3] == ("packages", "mdb") for path in paths):
        raise ValueError(f"{sdist.name} contains sibling package files")
    print(f"Verified {wheel.name} and {sdist.name}")
    return wheel


def check_installed(wheels: list[Path]) -> None:
    with tempfile.TemporaryDirectory(prefix="saturn-packages-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                *map(str, wheels),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
        for namespace, sibling in (("saturn_pub", "mdb"), ("mdb", "saturn_pub")):
            code = f"""
import importlib.abc
import pathlib
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'{sibling}', 'saturn', 'torch', 'transformers', 'transformer_lens',
                                      'diffusers', 'mrun', 'manalysis', 'common_harness'}}:
            raise ImportError('bare package attempted forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
import {namespace} as package
assert pathlib.Path(package.__file__).is_relative_to(pathlib.Path(sys.prefix))
if '{namespace}' == 'mdb':
    from mdb import BenchSession
    from mdb import ModelRunner, RunnerCapabilities, ForwardResult, TransformerLensRunner, TransformersRunner
    class JsonRunner(ModelRunner):
        def capabilities(self): return RunnerCapabilities('json')
        def forward(self, inputs, **kwargs): return ForwardResult(inputs + 1)
    with BenchSession.from_backend(JsonRunner(), identity={{'model_id': 'json'}}) as session:
        assert session.forward(2).output == 3
    from mdb.cli import main
    assert main(['version']) == 0
    assert main(['capabilities']) == 0
else:
    from saturn_pub import Act, Session, StateCut
"""
            result = subprocess.run(
                [str(python), "-I", "-c", code],
                cwd=root,
                text=True,
                capture_output=True,
            )
            if result.returncode:
                raise RuntimeError(f"{namespace} isolated import failed:\n{result.stderr}")
            print(f"Verified installed {namespace} with model frameworks and {sibling} forbidden")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saturn-dist", type=Path, default=repository / "dist/saturn")
    parser.add_argument("--mdb-dist", type=Path, default=repository / "packages/mdb/dist")
    args = parser.parse_args()
    wheels = [
        check_artifacts(args.saturn_dist, "saturn_pub", "mdb"),
        check_artifacts(args.mdb_dist, "mdb", "saturn_pub"),
    ]
    check_installed(wheels)


if __name__ == "__main__":
    main()
