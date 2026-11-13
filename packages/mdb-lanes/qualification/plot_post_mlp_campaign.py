"""Plot retained scalar observations; this script never executes a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    data = args.run / "extracted/post-mlp-result"
    screen = json.loads((data / "screen.json").read_text())["rows"]
    rows = json.loads((data / "followups.json").read_text())["rows"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plot-source.py").write_bytes(Path(__file__).read_bytes())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    fig = plt.figure(figsize=(13.5, 8.8), layout="constrained")
    fig.get_layout_engine().set(w_pad=.08, h_pad=.08, wspace=.12, hspace=.08)
    grid = fig.add_gridspec(2, 2, height_ratios=(1., 1.2))
    colors = {"post.0": "#ac3b25", "post.1": "#267a91", "post.2": "#7356a4"}
    for column, (context, bit, label) in enumerate((("context_03", 8, "Initial RDI 255 · bit 8"),
                                                   ("context_00", 0, "Initial RDI 252 · bit 0"))):
        ax = fig.add_subplot(grid[0, column])
        for site in colors:
            selected = sorted((row for row in screen if row["site"] == site and row["context"] == context),
                              key=lambda row: row["dose"])
            ax.plot([r["dose"] for r in selected], [r["scalar_heads"]["regs"][0][7][bit] for r in selected],
                    marker="o", color=colors[site], lw=2, label=site)
        ax.axhline(0, color="#555555", linestyle="--", lw=1)
        ax.set(title=label, xlabel="MLP output scale", ylabel="Scalar native register logit",
               xticks=(0, .5, 1, 1.5))
        ax.grid(axis="y", color="#e4e6e8", lw=.7)
        ax.legend(frameon=False, ncol=3, fontsize=10)
    ax = fig.add_subplot(grid[1, 0])
    by_arm = {row["arm"]: row for row in rows if row["context"] == "context_03" and row["source"] == "post.0"}
    arms = (("zero", "projection_native"), ("skip_native", "carrier_native"))
    values = [[by_arm[arm]["heads"]["regs"][0][7][8] for arm in row] for row in arms]
    image = ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(vmin=-24, vcenter=0, vmax=24), aspect="auto")
    for i, row in enumerate(arms):
        for j, arm in enumerate(row):
            logit = values[i][j]
            rdi = int(by_arm[arm]["decode"]["register_words"][7], 16)
            ax.text(j, i, f"{logit:+.3f}\nRDI {rdi}", ha="center", va="center", fontsize=14,
                    color="white" if abs(logit) > 16 else "#141820", fontweight="semibold")
    ax.set(title="After zeroing post.0 · initial RDI 255",
           xticks=(0, 1), xticklabels=("Edited update U", "Native update U"),
           yticks=(0, 1), yticklabels=("Edited skip S", "Native skip S"))
    fig.colorbar(image, ax=ax, shrink=.8, label="Bit 8 logit", pad=.03)
    ax = fig.add_subplot(grid[1, 1])
    selected = {row["arm"]: row for row in rows if row["context"] == "context_00" and row["source"] == "post.0"}
    labels = ("Zero", "Restore U", "Restore S", "Restore both")
    names = ("zero", "projection_native", "skip_native", "carrier_native")
    values = [selected[name]["heads"]["regs"][0][7][0] for name in names]
    native = selected["identity"]["heads"]["regs"][0][7][0]
    ax.bar(labels, values, color=("#ac3b25", "#267a91", "#b08a40", "#4b8063"), width=.65)
    ax.axhline(0, color="#555555", lw=1)
    ax.axhline(native, color="#232b32", linestyle="--", lw=1, label=f"Native {native:+.2f}")
    for index, value in enumerate(values):
        rdi = int(selected[names[index]]["decode"]["register_words"][7], 16)
        ax.annotate(f"{value:+.2f}\nRDI {rdi}", (index, value), xytext=(0, 6 if value >= 0 else -6),
                    textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    ax.set(title="RDI 252 · same result, different margins",
           ylabel="Bit 0 logit", ylim=(min(values)-10, max(values)+9))
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle("Three MLPs under matched inputs, then a skip/update restoration test", fontsize=16, weight="semibold")
    fig.supxlabel("Frozen R3 · CPU FP32 · scalar evidence · other register and flag decisions unchanged in these followups",
                  fontsize=10, color="#46515b")
    fig.savefig(args.output / "post-mlp-campaign.png")
    fig.savefig(args.output / "post-mlp-campaign.pdf")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "figures": ["post-mlp-campaign.png", "post-mlp-campaign.pdf"]}))


if __name__ == "__main__":
    main()
