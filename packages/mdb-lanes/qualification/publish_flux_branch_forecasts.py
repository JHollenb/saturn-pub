"""Build a scientific comparison panel from audited original model PNGs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def publish(run_dir, audit_path, output):
    audit = json.loads(audit_path.read_text())
    if audit["mechanics_status"] != "valid":
        raise ValueError("requires mechanically valid audited evidence")
    output.mkdir(parents=True, exist_ok=False)
    names = audit["branch_ids"]
    titles = {"base": "Base state", "relation": "Ordering text edit", "color": "Color text edit", "shape": "Shape text edit"}
    figures, arrays, rows = {}, {}, {}
    fig, axes = plt.subplots(2, len(names), figsize=(15, 8.5), squeeze=False)
    for column, name in enumerate(names):
        images = audit["branches"][name]["images"]
        arrays[name] = {}
        for row, kind in enumerate(("forecast", "native")):
            path = run_dir / images[kind]["path"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != images[kind]["sha256"]:
                raise ValueError("source PNG differs from audit")
            pixels = np.asarray(Image.open(path))
            arrays[name][kind] = pixels
            ax = axes[row, column]
            ax.imshow(pixels, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(titles[name], fontsize=13, pad=10)
            else:
                error = audit["branches"][name]["pixel_error"]
                ax.set_xlabel(f"Forecast error: max {error['max_abs']:g}, MAE {error['mean_abs']:g}", fontsize=10)
            if column == 0:
                ax.set_ylabel("Sealed equations" if row == 0 else "Native continuation", fontsize=12)
            figures[name + "." + kind] = {"path": str(path), "sha256": images[kind]["sha256"]}
        native_delta = arrays[name]["native"].astype(np.float64) - arrays["base"]["native"].astype(np.float64)
        rows[name] = {"pixel_error": audit["branches"][name]["pixel_error"],
            "png_bytes_exact": images["forecast"]["sha256"] == images["native"]["sha256"],
            "native_difference_from_base": {"mean_abs": float(np.abs(native_delta).mean()), "max_abs": float(np.abs(native_delta).max()),
                "changed_channels": int(np.count_nonzero(native_delta))}}
    fig.suptitle(f"Four predicted image futures from one base state   |   FLUX.2 Klein 4B   |   seed {audit['seed']}", fontsize=16, y=.98)
    fig.text(.5, .035, "All four forecasts sealed before any native image continuation. Only initial working text changes; later stored conditioning stays base.", ha="center", fontsize=10)
    fig.text(.5, .012, "Observed prompt carriers and full weights supplied. Functional forecasts share low-level CUDA kernels; every future consumes its own state.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.045, right=.99, top=.91, bottom=.09, wspace=.025, hspace=.08)
    path = output / "forecast-native-comparison.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    summary = {"schema": "mdb-flux-branch-forecast-figure-v1", "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "figure": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, "original_images": figures,
        "branches": rows, "scope": audit["scope"], "image_semantics": "not classified by this publisher; inspect original images separately"}
    with (output / "figures.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"figure": str(path), "branches": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir.resolve(), args.audit.resolve(), args.output.resolve())))
