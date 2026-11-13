"""Plot original saved lifecycle images without executing a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def render(compact, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    compact, destination = Path(compact), Path(destination)
    report = json.loads((compact / "audit.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    panels = {
        "binding": ([("binding-" + c, c) for c in ("anchor", "seed", "wording", "objects")],
                    ("whole_Q", "support_P", "complement_P", "whole_P")),
        "fixed-payload": ([("cd-fresh0s613", "cone / cylinder"), ("cd-foxcat0s821", "fox / cat")],
                          ("baseline", "sham", "common-only", "differential-only", "both")),
    }
    provenance = []
    for name, (contexts, arms) in panels.items():
        fig, axes = plt.subplots(len(contexts), len(arms), squeeze=False,
                                 figsize=(3 * len(arms), 3.2 * len(contexts)))
        for row, (rid, label) in enumerate(contexts):
            for column, arm in enumerate(arms):
                key = rid + "/" + arm
                item = report["original_pngs"][key]["native"]
                path = compact / item["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
                    raise ValueError("saved original image differs: " + key)
                axes[row, column].imshow(mpimg.imread(path))
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if row == 0:
                    axes[row, column].set_title(arm, fontsize=12)
                if column == 0:
                    axes[row, column].set_ylabel(label, fontsize=12)
                provenance.append(dict(case=key, original=item, panel=name))
        fig.suptitle("Original native continuations — " + name, fontsize=16)
        fig.tight_layout(rect=(0, 0.015, 1, 0.96))
        fig.savefig(destination / (name + ".png"), dpi=110)
        plt.close(fig)
    receipt = dict(schema="mdb-lifecycle-image-panels-v1", model_calls=0,
                   scope="Display panels from hash-verified original native PNGs; numerical equality is established separately by the saved-only audit.",
                   audit_sha256=hashlib.sha256((compact / "audit.json").read_bytes()).hexdigest(),
                   originals=provenance)
    (destination / "render-provenance.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"panels": len(panels), "originals": len(provenance), "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.compact, args.destination), sort_keys=True))
