"""Separate complete Qwen fields from their overlapping row diagnostics."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    account = runpy.run_path(str(ROOT / "verify_saved_reports.py"))
    audit = json.loads((ROOT / "compact-collection-01/extracted/binding-minimal-result/audit.json").read_text())
    full = [row["full"] for row in audit["qwen"]["fields"].values()]
    regions = [metric for row in audit["qwen"]["fields"].values() for key in ("attended_query_rows", "padded_query_rows") for metric in account["metrics"](row[key])]
    denoiser = [field for case in audit["cases"].values() for row in case["denoiser"]["boundaries"] for field in row["fields"].values()]
    summary = account["summarize"]
    result = {"schema": "mdb-e18-primary-scope-recount-v1", "source_verification_sha256": sha(ROOT / "actual-saved-report-verification-v1.json"),
              "primary_full_qwen_fields": summary(full), "regional_qwen_diagnostics": summary(regions),
              "primary_full_denoiser_fields": summary(denoiser), "primary_full_qwen_and_denoiser": summary(full + denoiser),
              "clarification": "The earlier root recursive count2047 includes29 complete Qwen fields AND58 overlapping attended/padded diagnostics. The1989 whole-field primary count excludes those58 subfield comparisons. Numeric measurements and all source/audit artifacts are unchanged.",
              "all_positions_are_dependent_and_overlap": True, "scientific_measurements_changed": False,
              "new_model_calls": 0, "new_installed_CLI_queries": 0}
    target = ROOT / "primary-scope-recount-v1.json"
    with target.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": sha(target), "primary": result["primary_full_qwen_and_denoiser"], "regional": result["regional_qwen_diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
