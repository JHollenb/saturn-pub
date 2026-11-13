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
    combined = publication["schema"] == "mdb-qwen-prompt-generalization-publication-v1"
    if not combined and (publication["schema"] != "mdb-qwen-full-readout-publication-v1" or publication["mechanics_status"] != "valid"):
        parser.error("a saved mechanics-valid full-readout publication or combined prompt campaign is required")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if combined:
        records = []
        for panel in publication["panels"].values():
            saved = panel["publication"]
            if saved["mechanics_status"] != "valid":
                parser.error("a panel is not mechanics-valid")
            for prompt_id, prefix in saved["prefixes"].items():
                count = len(prefix["token_ids"][0])
                records.append((prompt_id, f"{prompt_id} ({count} prompt tokens)", saved))
        if len(records) != 6:
            parser.error("the combined chart requires six prompt records")
        fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), sharex=True, layout="constrained")
        axes = axes.flat
    else:
        prompts = {"capital": "The capital of France is", "planet": "The largest planet in our solar system is"}
        records = [(key, text, publication) for key, text in prompts.items()]
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True, layout="constrained")
    axes = list(axes)
    for ax, (prompt_id, title, saved) in zip(axes, records, strict=True):
        envelope = saved["envelopes"][prompt_id]["envelope"]
        rows = [row for row in saved["rows"].values() if row["prompt_id"] == prompt_id]
        tokens = list(dict.fromkeys([segment["token_id"] for segment in envelope["segments"]]
                                   + [row["actual_token_id"] for row in rows]))
        labels = {row["actual_token_id"]: row["observed_first_token_text"] for row in rows}
        for segment in envelope["segments"]:
            y = tokens.index(segment["token_id"])
            ax.plot([segment["left"], segment["right"]], [y, y], linewidth=11,
                    color=("#62b6b2", "#cf9bde", "#ecb267")[y % 3], solid_capstyle="butt", zorder=1)
        for row in rows:
            ax.scatter(row["dose"], tokens.index(row["actual_token_id"]), color="#172e40", s=45, zorder=3)
        for point in envelope["breakpoints"][1:-1]:
            ax.axvline(point["dose"], color="#617386", linestyle="--", linewidth=1)
            text = f"{point['dose']:.3f}" if combined else f"predicted switch\n{point['dose']:.6f}"
            ax.text(point["dose"] + .012, .42 * (len(tokens) - 1) if combined else .5, text, fontsize=8 if combined else 9,
                    color="#42566b", va="center")
        ticklabels = [f"{token}  {labels[token]!r}" if token in labels else f"{token}  (untested)" for token in tokens]
        ax.set(yticks=range(len(tokens)), yticklabels=ticklabels, xticks=[0, .25, .5, .75, 1],
               ylim=(-.45, len(tokens) - .55), xlim=(-.025, 1.025), title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e5eaf0", linewidth=.8)
    for ax in axes[-(2 if combined else 1):]:
        ax.set_xlabel("Final-MLP write dose (1 = native)")
    fig.suptitle("Six fresh prompts: sealed forecasts and native tests" if combined else
                 "Sealed first-token forecasts, tested through native Qwen", fontsize=15)
    fig.supxlabel("Colored bars: forecast winner across all 151,936 rows. Dots: native tests; sham overlaps dose 1.\n"
                  "Dashed lines: predicted switches. Exact switch locations and later text were not tested as forecasts.",
                  fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "new_model_calls": 0, "equation_module_evaluations": 0}))


if __name__ == "__main__":
    main()
