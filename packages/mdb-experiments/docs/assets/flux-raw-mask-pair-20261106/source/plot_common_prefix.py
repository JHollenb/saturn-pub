"""Plot authenticated saved cross-case diagnostics; no arrays or model execution."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "compact-collection-01/extracted/binding-minimal-result/audit.json"
EXPECTED_AUDIT = "896a8111e1b0d4caeeec53a731cffd2c8ee4087e2708aea3ef6cc75a392d59db"
OUT = ROOT / "common-prefix-drift-v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    if sha(AUDIT) != EXPECTED_AUDIT:
        raise ValueError("Saved audit differs from authenticated collection")
    audit = json.loads(AUDIT.read_text())
    diagnostics = audit["common_token_prefix"]["diagnostics"]
    layers = sorted(map(int, diagnostics["native"]))
    if layers != list(range(28)):
        raise ValueError("Unexpected retained prefix horizon")
    rows = []
    fields = ("rms", "relative_rms", "native_rms", "maxabs", "changed_values", "sample_count")
    for layer in layers:
        native = diagnostics["native"][str(layer)]
        forecast = diagnostics["forecast"][str(layer)]
        if any(native[key] != forecast[key] for key in fields):
            raise ValueError("Lane summaries differ; revise the plotted scope")
        rows.append({"boundary": layer, **{key: native[key] for key in fields}})
    OUT.mkdir(exist_ok=False)
    data = {
        "schema": "mdb-common-prefix-drift-plot-data-v1",
        "audit_sha256": EXPECTED_AUDIT,
        "comparison": "long minus short within each lane, shared first 30 token IDs",
        "relative_denominator": "RMS of short-case hidden state on the same 30 rows",
        "raw_generic_orientation_label_preserved": "functional-minus-native",
        "orientation_note": "The generic raw label does not describe this cross-case diagnostic.",
        "forecast_native_summary_metrics_equal": True,
        "observational_scope": "Dependent states from one paired context contrast; no gain or semantic attribution claim.",
        "rows": rows,
        "new_neural_calls": 0,
        "raw_arrays_loaded": False,
    }
    write(OUT / "plot-data.json", data)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.0), sharex=True,
                             gridspec_kw={"hspace": 0.26})
    fig.suptitle("Same 30-token prefix, different finite states", fontsize=18, x=0.09, ha="left", y=0.97)
    fig.text(0.09, 0.93, "Observed long − short within each lane; forecast and native traces coincide.", fontsize=11)
    specifications = (
        ("rms", "Cross-case difference RMS", "activation units", "#275d91"),
        ("native_rms", "Short-case state RMS (relative-difference denominator)", "activation units · log scale", "#677d2f"),
        ("relative_rms", "Difference RMS / short-case state RMS", "dimensionless", "#b36526"),
    )
    for ax, (field, title, ylabel, color) in zip(axes, specifications):
        ax.plot(layers, [row[field] for row in rows], color=color, marker="o", markersize=3.2, linewidth=1.7)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.axvspan(6, 7, color="#c8c8c8", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.4, 27.4)
    axes[1].set_yscale("log")
    axes[0].annotate("H1: 0.0001784", xy=(1, rows[1]["rms"]), xytext=(2.5, 0.010),
                     arrowprops={"arrowstyle": "-", "color": "#777777"}, fontsize=9)
    axes[2].annotate("H6 → H7: absolute drift rises,\nrelative drift falls as state RMS jumps",
                     xy=(7, rows[7]["relative_rms"]), xytext=(10, 0.0028),
                     arrowprops={"arrowstyle": "-", "color": "#777777"}, fontsize=9)
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].set_xlabel("Qwen hidden-state boundary (H0 = token embedding)")
    axes[2].set_xticks([0, 1, 3, 6, 7, 9, 12, 15, 18, 21, 24, 27])
    fig.text(0.09, 0.015,
             "A context/execution-profile contrast, not forecast error. Mask/GQA lowering remains a hypothesis pending matched operands.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.075)
    fig.savefig(OUT / "common-prefix-drift.png", dpi=160)
    plt.close(fig)
    write(OUT / "receipt.json", {
        "schema": "mdb-common-prefix-drift-figure-v1",
        "source_sha256": sha(Path(__file__)),
        "audit_sha256": EXPECTED_AUDIT,
        "outputs": {name: sha(OUT / name) for name in ("plot-data.json", "common-prefix-drift.png")},
        "new_neural_calls": 0,
        "raw_arrays_loaded": False,
    })
    print(json.dumps({"out": str(OUT), "receipt_sha256": sha(OUT / "receipt.json")}))


if __name__ == "__main__":
    main()
