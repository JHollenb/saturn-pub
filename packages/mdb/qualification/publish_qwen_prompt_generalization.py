"""Compose two predeclared Qwen prompt panels from validated saved evidence.

No model, equation module, tokenizer or replay execution. Panel publications
retain their own reference bases; campaign references use the supplied root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}


def publish(root):
    root = Path(root).resolve()
    manifest_path = root / "campaign-plan.json"
    manifest = read(manifest_path)
    require(set(manifest["panels"]) == {"language-a", "language-b"}, "both predeclared language panels are required")
    helper = runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))
    panels, prompt_rows, refs, total_rows, shams = {}, {}, {}, 0, 0
    for name, plan in manifest["panels"].items():
        panel_root = root / name
        run_dir = panel_root / "run-01"
        run = helper["source_run"](run_dir, "qwen-mlp-result")
        request = run["request"]
        require(request["config"] == plan, "actual request differs from the pre-submission compound plan: " + name)
        for filename in ("qwen_mlp_forecast_campaign.py", "qwen_mlp_equations.py", "qwen_readout_law.py"):
            require(request["source_hashes"][filename] == manifest["source_hashes"][filename], "worker/equation source differs from compound manifest: " + filename)
        publication_path = panel_root / "publication-v1.json"
        publication = read(publication_path)
        require(publication["mechanics_status"] == "valid", "panel publication is not mechanics-valid: " + name)
        require(all(publication["source"][key] == run["source"][key] for key in
                    ("job_id", "request_sha256", "archive_sha256", "transaction")), "panel publication custody differs")
        for ref in publication["references"].values():
            require(sha(helper["file_path"](panel_root, ref["path"])) == ref["sha256"], "panel publication reference changed: " + ref["path"])
        audit = read(run_dir / "analysis-v1.json")
        require(sha(run_dir / "analysis-v1-source.py") == audit["auditor_source_sha256"], "panel audit source snapshot differs")
        require(publication["rows"] == audit["rows"] and publication["envelopes"] == audit["envelopes"], "publication differs from the independently audited rows/envelopes")
        forecast = helper["read"](run["transaction"] / "forecasts/campaign.json")["predictions"]
        for prompt in plan["prompts"]:
            prompt_id = prompt["id"]
            envelope = run["tree"].select(forecast, ["full_vocabulary", prompt_id, "envelope"])
            require(envelope == publication["envelopes"][prompt_id]["envelope"], "published dose thresholds differ from sealed forecast")
            selected = {arm: publication["rows"][prompt_id + "." + arm] for arm in plan["arms"]}
            native = selected["native"]
            contrasts = {}
            for arm in ("zero", "half"):
                row = selected[arm]
                contrasts[arm] = {"dose": row["dose"], "forecast_token_id": row["forecast_token_id"],
                    "actual_token_id": row["actual_token_id"], "actual_token_changed_vs_native": row["actual_token_id"] != native["actual_token_id"],
                    "forecast_token_changed_vs_native": row["forecast_token_id"] != native["forecast_token_id"],
                    "actual_top2gap": row["actual_top2gap"], "forecast_top2gap": row["forecast_top2gap"],
                    "observed_generated_text": row["observed_generated_text"],
                    "observed_generated_token_ids": row["observed_generated_token_ids"], "continuation_check": row["continuation_check"]}
            thresholds = [{"dose": segment["left"], "before_token_id": envelope["segments"][index - 1]["token_id"],
                           "after_token_id": segment["token_id"]} for index, segment in enumerate(envelope["segments"]) if index]
            prompt_rows[name + "." + prompt_id] = {"panel": name, "prompt": prompt,
                "prompt_token_count": len(publication["prefixes"][prompt_id]["token_ids"][0]),
                "native_token_id": native["actual_token_id"], "native_forecast_token_id": native["forecast_token_id"],
                "observed_native_text": native["observed_generated_text"], "contrasts_vs_native": contrasts,
                "sealed_winner_thresholds": thresholds, "sealed_envelope": envelope,
                "threshold_scope": "analytical winner changes of the sealed affine law; native tested only at declared doses"}
        panels[name] = {"artifact_reference_base": name + "/", "publication": publication}
        refs[name] = {"publication": reference(publication_path, root), "request": reference(run_dir / "request.json", root),
                      "audit": reference(run_dir / "analysis-v1.json", root), "collector_report": reference(run_dir / "report.json", root)}
        total_rows += len(publication["rows"])
        shams += sum(row["arm"] == "sham" for row in publication["rows"].values())
    expected = manifest["denominators"]
    require(total_rows == expected["native_branches_including_shams"] and shams == expected["sham_branches"]
            and len(prompt_rows) == expected["new_prompt_contexts"] and total_rows - shams == expected["distinct_prompt_dose_pairs"],
            "completed panel denominators differ from the compound pre-submission declaration")
    return {"schema": "mdb-qwen-prompt-generalization-publication-v1", "artifact_reference_base": "supplied campaign root",
        "campaign_manifest": reference(manifest_path, root), "predeclared_campaign": manifest, "panels": panels,
        "references": refs, "prompt_contrasts": prompt_rows, "denominators": expected,
        "summary": {"full_vocab_argmax_matches": sum(panel["publication"]["scientific_summary"]["full_vocab_argmax_matches"] for panel in panels.values()),
            "full_vocab_tested_branches": total_rows, "physical_attempts": sum(panel["publication"]["counts"]["physical_attempts"] for panel in panels.values()),
            "max_full_logit_abs_residual": max(panel["publication"]["scientific_summary"]["max_full_logit_abs_residual"] for panel in panels.values()),
            "first_token_changes_vs_native": {arm: sum(prompt["contrasts_vs_native"][arm]["actual_token_changed_vs_native"] for prompt in prompt_rows.values()) for arm in ("zero", "half")}},
        "publisher_source_sha256": sha(Path(__file__)), "model_calls": 0, "equation_module_evaluations": 0,
        "terminal_status": "unassessed", "limits": ["Six prompts selected before either panel ran; two separate conditional-prefix forecast seals.",
            "All 24 branches are retained: 18 distinct prompt-dose pairs plus six shams; these are not 24 independent prompts.",
            "Unchanged supplied source equations, full checkpoint weights and observed native prefixes; no semantic-law learning or parameter compression claim.",
            "First-token predictions cover every vocabulary row; subsequent cached text is a native observation only.",
            "Envelope thresholds are sealed analytical forecasts, not extra native dose observations.",
            "The stager SHA is recorded by the pre-submission manifest; worker/equation SHAs are additionally verified against actual request source maps."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    result = publish(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"denominators": result["denominators"], "summary": result["summary"], "model_calls": 0}))


if __name__ == "__main__":
    main()
