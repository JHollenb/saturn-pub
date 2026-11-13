"""Display hash-verified original residual/write continuation PNGs."""
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
    audit_path = compact / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["schema"] != "mdb-binding-residual-write-saved-audit-v1":
        raise ValueError("saved residual/write audit required")
    contexts = ("anchor", "wording")
    arms = ("whole_Q", "residual_only", "write_only", "both_RW")
    titles = ("Base Q", "Incoming contrast only", "Net write contrast only", "Both contrasts")
    destination.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.7), squeeze=False)
    originals = []
    for row, context in enumerate(contexts):
        for column, arm in enumerate(arms):
            key = "binding-" + context + "/" + arm
            record = audit["original_pngs"][key]["native"]
            path = compact / record["path"]
            if (path.stat().st_size != record["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
                raise ValueError("original native PNG differs: " + key)
            axes[row, column].imshow(mpimg.imread(path))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=11)
            if column == 0:
                axes[row, column].set_ylabel(context.capitalize(), fontsize=12)
            originals.append(dict(case=key, lane="native", record=record))
    fig.suptitle("Original native continuations from the four binding rows", fontsize=15)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(destination / "residual-write.png", dpi=120)
    plt.close(fig)
    receipt = dict(schema="mdb-binding-residual-write-figure-v1", model_calls=0,
        audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(), originals=originals,
        scope="Display of unchanged original native PNGs; the saved-only audit separately compares complete predicted and native arrays.")
    (destination / "render-provenance.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return dict(panels=1, original_images=len(originals), model_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.compact, args.destination), sort_keys=True))
