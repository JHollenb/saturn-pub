"""Publish original native handoff images and source-qualified carrier geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def publish(run_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    audit = json.loads((run_dir / "analysis-v1.json").read_text())
    if audit["mechanics_status"] != "valid":
        raise ValueError("requires completed mechanics audit")
    data = run_dir / "extracted/flux-handoff-result"
    plan = json.loads((data / "plan.json").read_text())
    observed = {p.stem: json.loads(p.read_text()) for p in (data / "rows").glob("*.json")}
    output.mkdir(exist_ok=False, parents=True)
    panels = [[("base", "Early-history reference"), ("changed", "Shape exposure at step 1"),
        ("late.sham", "Late copy sham"), ("late.restore_both", "Late restore both carriers")]]
    for slot in ("text", "image"):
        panels.append([(b["id"] + ".restore_" + slot,
            b["id"].replace("after_", "After ") + " | restore " + slot) for b in plan["boundaries"]])
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    inputs = {}
    for row, entries in enumerate(panels):
        for col, (name, title) in enumerate(entries):
            item = observed[name]["image"]
            path = data / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("native PNG differs")
            inputs[name] = item
            axes[row, col].imshow(Image.open(path))
            p = audit["comparisons"][name]["scheduler_trajectory"]["4"]["target_projection"]
            axes[row, col].set_title(title + f"\nfinal latent p={p:.3f}", fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle("Where does the second step commit a persistent shape change?", fontsize=16)
    fig.text(.5, .018, "Original native PNGs | FLUX.2 Klein 4B, seed47, 512px, 4steps | every branch receives base conditioning from step2\np is latent geometry (reference=0; shape-step1=1), not a semantic score; no numerical RGB forecasts", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .96))
    fig.savefig(output / "shape-step1-handoff.png", dpi=150)
    plt.close(fig)
    q = Path(__file__).parent
    typed = runpy.run_path(str(q / "analyze_flux_mlp_forecast.py"))
    helper = runpy.run_path(str(q / "analyze_flux_concept_routes.py"))
    geometry = {}
    for b in plan["boundaries"]:
        name = b["id"]
        values = {}
        for slot in ("text", "image"):
            arrays = [helper["slot_array"](data / "typed-store", audit["frontiers"][ref][name], slot, typed).astype(np.float64)
                      for ref in ("base", "changed")]
            delta = arrays[1] - arrays[0]
            values[slot] = {"delta_l2": float(np.linalg.norm(delta.ravel())),
                "relative_to_base_l2": float(np.linalg.norm(delta.ravel()) / np.linalg.norm(arrays[0].ravel())),
                "signed_mean": float(delta.mean())}
        geometry[name] = values
    result = {"schema": "mdb-flux-shape-handoff-figures-v1", "inputs": inputs, "carrier_geometry": geometry,
        "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.png")},
        "limits": "Full-field geometry is descriptive; original native PNGs need separate visual review; no discovered object mask or semantic classifier."}
    (output / "figures.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(publish(a.run_dir.resolve(), a.output.resolve())))
