"""Plot already sealed dose regions and measured native choices; no model work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    publication = json.loads(args.publication.read_text())
    if publication["schema"] != "mdb-qwen-full-readout-publication-v1" or publication["mechanics_status"] != "valid":
        parser.error("a saved mechanics-valid full-readout publication is required")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompts = {"capital": "The capital of France is", "planet": "The largest planet in our solar system is"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True, layout="constrained")
    for ax, prompt_id in zip(axes, prompts, strict=True):
        envelope = publication["envelopes"][prompt_id]["envelope"]
        rows = [row for row in publication["rows"].values() if row["prompt_id"] == prompt_id]
        tokens = list(dict.fromkeys(segment["token_id"] for segment in envelope["segments"]))
        labels = {row["actual_token_id"]: row["observed_first_token_text"] for row in rows}
        for segment in envelope["segments"]:
            y = tokens.index(segment["token_id"])
            ax.plot([segment["left"], segment["right"]], [y, y], linewidth=11,
                    color=("#62b6b2", "#cf9bde")[y % 2], solid_capstyle="butt", zorder=1)
        for row in rows:
            ax.scatter(row["dose"], tokens.index(row["actual_token_id"]), color="#172e40", s=45, zorder=3)
        for point in envelope["breakpoints"][1:-1]:
            ax.axvline(point["dose"], color="#617386", linestyle="--", linewidth=1)
            ax.text(point["dose"] + .012, .5, f"predicted switch\n{point['dose']:.6f}", fontsize=9,
                    color="#42566b", va="center")
        ax.set(yticks=range(len(tokens)), yticklabels=[f"{token}  {labels.get(token, '?')!r}" for token in tokens],
               ylim=(-.45, len(tokens) - .55), xlim=(-.025, 1.025), title=prompts[prompt_id])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e5eaf0", linewidth=.8)
    axes[-1].set(xticks=[0, .25, .5, .75, 1], xlabel="Final-MLP write dose (1 = native)")
    fig.suptitle("Sealed first-token forecasts, tested through native Qwen", fontsize=15)
    fig.supxlabel("Colored bars: forecast winner across all 151,936 rows. Dots: native tests; sham overlaps dose 1.\n"
                  "Observed-prefix forecast. Exact switch locations and later text were not tested as forecasts.", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "new_model_calls": 0, "equation_module_evaluations": 0}))


if __name__ == "__main__":
    main()
