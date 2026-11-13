"""Show checksum-verified native futures for the joint4 source factorial."""
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
        raise ValueError("requires independent source audit")
    data = root / "extracted/flux-joint-sources-result"
    rows = {p.stem: json.loads(p.read_text()) for p in (data / "rows").glob("*.json")}
    groups = [["base", "changed", "A.base.recompute", "A.base.M.changed"],
              ["qkv." + x for x in ("000", "001", "010", "011")],
              ["qkv." + x for x in ("100", "101", "110", "111")], ["textQ.base"]]
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
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
                raise ValueError("native PNG differs")
            inputs[name] = record
            ax.imshow(Image.open(path))
            p = audit["comparisons"][name]["scheduler_trajectory"]["4"]["target_projection"]
            ax.set_title(f"{name}\nlatent p={p:.3f}", fontsize=11)
    fig.suptitle("What supplies joint4's image write—and does its receiving MLP need to adapt?", fontsize=15)
    fig.text(.5, .025, "QKV bits: base image Q / base text K+V / base image K+V | A: base image attention return\nM.changed: hold original changed MLP return fixed | otherwise MLP recomputes on edited attention\nBase text after joint4; base conditioning from step2 | Klein4B, seed47, 512px, foursteps\nOriginal native PNGs. p is latent geometry, not a semantic score; no independent RGB forecasts.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .97))
    path = output / "joint4-sources.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    manifest = {"schema": "mdb-flux-joint-sources-figures-v1", "native_inputs": inputs,
        "figures": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
        "audit_sha256": hashlib.sha256((root / "analysis-v1.json").read_bytes()).hexdigest()}
    (output / "figures.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = publish(a.root, a.output)
    print(json.dumps({"figures": result["figures"], "images": len(result["native_inputs"])}))
