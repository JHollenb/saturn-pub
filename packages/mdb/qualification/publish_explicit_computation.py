"""Publish compact views of completed evidence; never execute a model or equations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def publish(analysis_path: Path, output: Path) -> None:
    raw = analysis_path.read_bytes()
    data = json.loads(raw)
    rows = data["rows"]
    output.mkdir(parents=True, exist_ok=True)
    scopes = {}
    for scope in sorted({row["scope"] for row in rows}):
        selected = [row for row in rows if row["scope"] == scope]
        scopes[scope] = {
            "counts": data["counts"]["by_scope"][scope],
            "maximum_head_absolute_residual": {
                name: max(row["head_residuals"][name]["max_abs"] for row in selected)
                for name in selected[0]["head_residuals"]
            },
            "maximum_stage_absolute_residual": {
                name: max(row["stage_residuals"][name]["max_abs"] for row in selected)
                for name in selected[0]["stage_residuals"]
            },
            "head_decision_agreement_rows": {
                name: sum(row["all_head_decisions"][name]["equal"] for row in selected)
                for name in selected[0]["all_head_decisions"]
            },
        }
    compact_rows = []
    for row in rows:
        decoded = row["decoded"]
        compact_rows.append({
            "case_id": row["case_id"], "arm": row["arm"], "scope": row["scope"],
            "forecast": decoded["forecast"], "observed": decoded["observed"],
            "decoded_field_agreement": decoded["fields"],
            "all_head_decisions_equal": all(value["equal"] for value in row["all_head_decisions"].values()),
            "native_audit_all_heads_exact": row["native_audit_all_heads_exact"],
            "native_audit_all_stages_exact": row["native_audit_all_stages_exact"],
            "observed_successor": row["continuation"]["second_decode"],
            "successor_forecast_status": row["continuation"]["forecast_status"],
        })
    cases = []
    for case in data["case_results"]:
        readers = []
        for reader in case["readers"]:
            hidden = reader["hidden_unit_account"]
            units = hidden["all_96_units"]
            terms = [unit["U_swap_minus_zero"] for unit in units]
            readers.append({
                **{key: reader[key] for key in ["reader", "table", "contrasts", "route_distances"]},
                "hidden_unit_account": {
                    "authority": hidden["authority"], "limits": hidden["limits"],
                    "post_bias_projection": hidden["post_bias_projection"],
                    "positive_U_swap_minus_zero": sum(value for value in terms if value > 0),
                    "negative_U_swap_minus_zero": sum(value for value in terms if value < 0),
                    "net_U_swap_minus_zero": sum(terms),
                    "largest_absolute_changes": [
                        {"unit": unit["unit"], "U_swap_minus_zero": unit["U_swap_minus_zero"]}
                        for unit in sorted(units, key=lambda unit: abs(unit["U_swap_minus_zero"]), reverse=True)[:12]
                    ],
                    "complete_unit_account": "analysis-v3.json retains all 96 contributions per reader",
                },
            })
        cases.append({
            "case_id": case["case_id"], "scope": case["scope"],
            "family": case["family"], "payloads": case["payloads"],
            "prefix_proof": case["prefix_proof"], "readers": readers,
            "sham_heads_exact": all(x["exact_numeric"] for x in case["same_value_sham"]["heads"].values()),
        })
    summary = {
        "schema": "mdb-explicit-computation-publication-v1",
        "analysis_sha256": hashlib.sha256(raw).hexdigest(),
        "publication_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **{key: data[key] for key in [
            "source", "protocol", "formula", "weights", "global_forecast_exclusion",
            "receipt_counts", "mechanics_status", "instrument_status", "trend_status",
            "terminal_status", "limits", "own_state_continuations", "counterexamples",
        ]},
        "scopes": scopes, "rows": compact_rows, "cases": cases,
    }
    (output / "explicit-computation-20261030.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot(summary, output / "explicit-computation-20261030.png")
    print(json.dumps({"rows": len(rows), "analysis_sha256": summary["analysis_sha256"], "output": str(output)}))


def plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def reader(case_id: str, register: int, bit: int) -> dict:
        case = next(case for case in summary["cases"] if case["case_id"] == case_id)
        return next(r for r in case["readers"] if r["reader"]["register"] == register and r["reader"]["bit"] == bit)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    sub = reader("sub_258_new", 6, 1)
    arms = ["native", "zero", "U_swap", "U_swap_P2native"]
    labels = ["Native", "Early MLP\nzeroed", "U restored", "U + common\npredicted P2"]
    values = [sub["table"][arm]["actual_logit"] for arm in arms]
    forecasts = [sub["table"][arm]["forecast_logit"] for arm in arms]
    ax = axes[0]
    ax.bar(range(4), values, color=["#526b85", "#a0b3c3", "#ba5353", "#2b887b"], width=.62)
    ax.scatter(range(4), forecasts, marker="_", s=190, linewidths=2.2, color="black", label="Sealed forecast", zorder=3)
    ax.axhline(0, color="#444444", linewidth=.8)
    for i, value in enumerate(values):
        ax.text(i, value + (.3 if value >= 0 else -.35), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top")
    ax.set(xticks=range(4), xticklabels=labels, ylabel="RSI bit 1 logit", ylim=(-2.5, 11.4), title="Fresh SUB-258: predicted collateral and repair")
    ax.legend(frameon=False, loc="upper left")
    xor = reader("xor_255_anchor", 7, 0)
    x = np.arange(2)
    for offset, route, color in [(-.18, "U", "#526b85"), (.18, "S", "#2b887b")]:
        values = [xor["route_distances"][kind]["actual"][f"{route}_distance"] for kind in ["natural_post", "common_frozen_post"]]
        bars = axes[1].bar(x + offset, values, width=.34, color=color, label=f"{route} restoration")
        axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].set(xticks=x, xticklabels=["Natural P2 response", "Common predicted P2"], ylabel="Distance from native RDI bit 0 logit", ylim=(0, 5), title="XOR-255 anchor: S stays closer with P2 fixed")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Independent equations predicted the native intervention outcomes", fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    publish(args.analysis, args.output)


if __name__ == "__main__":
    main()
