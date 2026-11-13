"""Plot source-verified native RGB futures; no generated replacement pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def publish(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    audit = json.loads((root / "analysis-v1.json").read_text())
    if audit["mechanics_status"] != "valid":
        raise ValueError("requires independent audit")
    data = root / "extracted/flux-joint-writes-result"
    rows = {p.stem: json.loads(p.read_text()) for p in (data / "rows").glob("*.json")}
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(6, 4, figsize=(15, 22))
    groups = [["base", "changed", "sham", "reconstruct.base"]]
    groups += [[f"j{i}.{v}" for v in ("A", "M", "AM")] for i in range(1, 5)]
    groups += [["all." + v for v in ("A", "M", "AM")]]
    inputs = {}
    for line, names in enumerate(groups):
        for column in range(4):
            ax = axes[line, column]
            ax.axis("off")
            if column >= len(names):
                continue
            name = names[column]
            record = rows[name]["image"]
            path = data / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("PNG digest differs")
            inputs[name] = record
            ax.imshow(Image.open(path))
            p = audit["comparisons"][name]["scheduler_trajectory"]["4"]["target_projection"]
            ax.set_title(f"{name} | latent p={p:.3f}", fontsize=12)
    fig.suptitle("Which joint attention and image-MLP writes establish a persistent shape?", fontsize=16)
    fig.text(.5, .025, "A: replace image attention return; M: replace image FF return; AM: both | donor: matched base history\nAll branches use base text after joint4 and base conditioning from step2. Seed47, FLUX.2 Klein 4B, 4steps.\nOriginal native PNGs; p is latent geometry, not a semantic score. Informal visual review is separate.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .975))
    path = output / "joint-writes.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    manifest = {"schema": "mdb-flux-joint-writes-figures-v1", "native_inputs": inputs,
                "figures": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                "audit_sha256": hashlib.sha256((root / "analysis-v1.json").read_bytes()).hexdigest()}
    (output / "figures.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(publish(a.root, a.output)))
