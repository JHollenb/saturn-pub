"""Plot original native PNGs and audited latent geometry; no generated imagery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def publish(run_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    audit = json.loads((run_dir / "analysis-v1.json").read_text())
    if audit["mechanics_status"] != "valid":
        raise ValueError("requires valid mechanics audit")
    data = run_dir / "extracted/flux-shape-timing-result"
    output.mkdir(parents=True, exist_ok=False)
    panels = [[("base", "Base: cube / sphere"), ("early", "Shape T; base C throughout"),
        ("early.switch1", "Shape C from step 1"), ("early.switch2", "Shape C from step 2"), ("early.switch3", "Shape C from step 3")],
        [("source_only", "Base T; shape C later"), ("coherent", "Shape T; shape C throughout"),
        ("coherent.switch1", "Return to base C at step 1"), ("coherent.switch2", "Return to base C at step 2"), ("coherent.switch3", "Return to base C at step 3")]]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    sources = {}
    for row, entries in enumerate(panels):
        for col, (name, title) in enumerate(entries):
            item = audit["receipt_paths"][name]["image"]
            path = data / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("audited image differs")
            sources[name] = item
            axes[row, col].imshow(Image.open(path))
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle("Same seed and initial image state: which later prompt reads shape the objects?", fontsize=16)
    fig.text(.5, .025, "FLUX.2 Klein 4B | seed 47 | 4 steps | native causal futures, not numerical image forecasts\nT = initial working text; C = persistent conditioning; switch occurs before that step's projection", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(output / "shape-timing-native-futures.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label in (("early", "Base C → shape C"), ("coherent", "Shape C → base C")):
        ax.plot([1, 2, 3], [audit["geometry"][name + f".switch{k}"]["shape_direction_projection"] for k in (1, 2, 3)], marker="o", label=label)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set(xticks=[1, 2, 3], xlabel="Source switch before diffusion step", ylabel="Final latent projection (early=0; coherent=1)", title="Timing of source reads changes the final state")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "shape-timing-latent-geometry.png", dpi=150)
    plt.close(fig)
    record = {"schema": "mdb-flux-shape-timing-figures-v1", "inputs": sources,
        "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.png")},
        "limits": "Original native PNGs; latent projection is not a semantic shape score; no shape classification or image synthesis by publisher."}
    (output / "figures.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(publish(a.run_dir.resolve(), a.output.resolve())))
