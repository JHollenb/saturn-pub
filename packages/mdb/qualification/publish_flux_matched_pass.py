"""Publish authenticated saved retrospective Q=G_1(N1) observations only.

No scheduler requests, checkpoint hydration, learned-weight math, model/native
calls or inferred missing trajectories. All signed account rows are preserved.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CUSTODY_HELPER_SHA256 = "04299a7830f19e536bd03946a20f39b329160c263ca94fdcac48b345c334b3af"
CORE_SHA256 = "583bb90bf489df8ba14ec118cdae8316855523d033548b2cd62d85be0999705d"
ANALYZER_SHA256 = "86084b8c751519dcb0ce4cbd5e8640a5cc84ccf12a986e6074c2e694002d40e5"
MEMOIZATION_SHA256 = "977a6df3717a714dec6d73b63b5cb3d2bfe35e10bd361aafecc3a12787cae214"
CELLS = ("c1-o0", "c0-o1")
LANDMARKS = (0, 5, 25, 26, 27)
TERMS = ("A_Q_minus_N", "B_P_minus_Q", "E_P_minus_N")
COLORS = ("#23855c", "#326cb4", "#171717")
SCOPE = ("Retrospective selected two-context logical-pass1 observations, Q=G_1(N1), after targets were known. "
    "P is the original full own-state forecast at the same cursor and N the actual clock-aligned native field. "
    "A=Q-N, B=P-Q, E=P-N; E=A+B. These signed vector accounts are not causal shares, isolated kernel errors, "
    "or prospective Q accuracy. The selected contexts differ in both supplied prompt factors. "
    "B is the difference between two whole-pass programs with different incoming latents; no G transport decomposition is inferred.")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def helpers(path):
    require(sha(path) == CUSTODY_HELPER_SHA256, "qualified generic saved-custody helper differs")
    return runpy.run_path(str(path))


def verify_capsule(h, directory, collection, job, filename):
    """Bind exact JSON bytes to the one bounded capsule in the actual job log."""
    log = h["file_path"](directory, "worker.log")
    expected = job["result"]["artifacts"]["job_log"]
    require(log.stat().st_size == expected["bytes"] <= 24 << 20 and sha(log) == expected["sha256"],
        "actual scheduler log differs")
    marker = "MDB_FACTORIAL_AUDIT_CAPSULE="
    capsules = [line[len(marker):] for line in log.read_text().splitlines() if line.startswith(marker)]
    require(len(capsules) == 1, "exactly one actual log capsule required")
    encoded = json.loads(capsules[0])
    metadata = collection["capsule"]
    require({k: v for k, v in encoded.items() if k != "base64_gzip"} == metadata
        and metadata["schema"] == "mdb-actual-audit-capsule-v1"
        and 0 < metadata["bytes"] <= 32 << 20 and 0 < metadata["gzip_bytes"] <= 8 << 20,
        "actual capsule metadata/bounds differ")
    compressed = base64.b64decode(encoded["base64_gzip"], validate=True)
    require(len(compressed) == metadata["gzip_bytes"] and hashlib.sha256(compressed).hexdigest() == metadata["gzip_sha256"],
        "actual compressed capsule differs")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read((32 << 20) + 1)
    report = h["file_path"](directory, filename)
    require(len(raw) == metadata["bytes"] and hashlib.sha256(raw).hexdigest() == metadata["sha256"]
        and raw == report.read_bytes(), "saved result is not its actual scheduler-log capsule")


def verify_parent(h, run, request):
    """Check staged, source-hashed parent GPU/accepted CPU metadata only."""
    parent, audit = run / "payload/parent-run", run / "payload/parent-audit"
    pr, pc, pj = [read(h["file_path"](parent, name)) for name in ("request.json", "collection.json", "collection-job.json")]
    ar, ac, aj, accepted = [read(h["file_path"](audit, name)) for name in ("request.json", "collection.json", "job.json", "audit.json")]
    for directory, req, collection, job in ((parent, pr, pc, pj), (audit, ar, ac, aj)):
        h["verify_request"](req)
        h["verify_job"](job, req)
        h["verify_bindings"](directory, collection)
        require(collection["job_id"] == job["job_id"] and collection["request_sha256"] == req["request_sha256"]
            and job["result"]["diagnostics"]["command"] == read(h["file_path"](directory, "admission-command.json"))["command"],
            "staged parent job/admission/collection differs")
    require(pc["schema"] == "mdb-prompt-factorial-ssh-collection-v1"
        and ac["schema"] == "mdb-actual-near-data-audit-collection-v1", "parent custody lanes differ")
    verify_capsule(h, audit, ac, aj, "audit.json")
    config = request["config"]
    require(pj["job_id"] == config["parent_job_id"] and pr["request_sha256"] == config["parent_request_sha256"]
        and aj["job_id"] == config["parent_audit_job_id"] and sha(audit / "audit.json") == config["parent_audit_sha256"]
        and accepted["mechanics_status"] == "valid" and accepted["auditor_source_sha256"] == ANALYZER_SHA256
        and accepted["source"]["job_id"] == pj["job_id"] and accepted["source"]["request_sha256"] == pr["request_sha256"]
        and accepted["source"]["collection_sha256"] == sha(parent / "collection.json"), "accepted parent audit identity differs")
    execution = accepted["near_data_execution"]
    require(execution["audit_job_id"] == aj["job_id"] and execution["audit_request_sha256"] == ar["request_sha256"]
        and execution["parent_job_id"] == pj["job_id"] and execution["parent_request_sha256"] == pr["request_sha256"]
        and execution["native_framework_forwards"] == 0 and execution["mathematical_transitions"] == 448
        and execution["parent_inventory_unchanged"] is True, "accepted parent numerical execution differs")
    inventory = read(h["file_path"](parent, "collection-local-inventory.json"))
    require(inventory == read(parent / "collection-remote-before.json") == read(parent / "collection-remote-after.json"),
        "source-hashed parent inventory snapshots differ")
    return inventory, {"parent_job_id": pj["job_id"], "parent_request_sha256": pr["request_sha256"],
        "parent_audit_job_id": aj["job_id"], "parent_audit_sha256": sha(audit / "audit.json"),
        "parent_inventory_sha256": sha(parent / "collection-local-inventory.json"),
        "scope": "Authenticated metadata staged in actual matched-job payload; original P/N arrays are not duplicated in this publication."}


def verify_run(run, h):
    run = Path(run).resolve()
    request, job, collection, capsule = [read(h["file_path"](run, name)) for name in
        ("request.json", "job.json", "collection.json", "result.json")]
    h["verify_request"](request)
    h["verify_job"](job, request)
    h["verify_bindings"](run, collection)
    require(collection["schema"] == "mdb-matched-pass-collection-v1" and collection["archive_sha256"] is None
        and collection["job_id"] == job["job_id"] and collection["request_sha256"] == request["request_sha256"]
        and collection["collector_sha256"] == request["launcher_sha256"]
        and job["result"]["diagnostics"]["command"] == read(run / "admission-command.json")["command"],
        "actual matched collection/admission identity differs")
    config = request["config"]
    require(config["schema"] == "mdb-matched-pass-plan-v1" and config["conditions"] == list(CELLS)
        and config["logical_step"] == 1 and config["native_framework_forwards"] == 0 and config["mathematical_transitions"] == 56,
        "retrospective finite plan differs")
    for name, digest in request["source_hashes"].items():
        require(sha(h["file_path"](run / "payload", name)) == digest, "actual staged source changed: " + name)
    for name, digest in (("flux_matched_pass_replay.py", CORE_SHA256), ("analyze_flux_prompt_factorial.py", ANALYZER_SHA256),
        ("memoize_flux_audit_parameters.py", MEMOIZATION_SHA256)):
        require(request["source_hashes"][name] == digest, "qualified numerical/I/O source differs")
    verify_capsule(h, run, collection, job, "result.json")
    require(capsule["schema"] == "mdb-matched-pass-execution-v1" and capsule["output_directory"] == "matched-pass-output",
        "actual result capsule lane differs")
    data = run / "extracted/matched-pass-output"
    actual = h["inventory"](data)
    require(actual == read(run / "collection-local-inventory.json") == read(run / "collection-remote-before.json")
        == read(run / "collection-remote-after.json") and actual["file_count"] == collection["file_count"]
        and actual["expanded_bytes"] == collection["expanded_bytes"], "authenticated matched inventories differ")
    manifest = read(data / "remote-manifest.json")
    require(manifest["schema"] == "mdb-matched-pass-manifest-v1" and manifest["job_id"] == job["job_id"]
        and manifest["request_sha256"] == request["request_sha256"]
        and manifest["files"] == {k: v for k, v in actual["files"].items() if k != "remote-manifest.json"}
        and capsule["remote_manifest_sha256"] == sha(data / "remote-manifest.json"), "actual matched manifest differs")
    source, plan, report, execution = [read(data / name) for name in ("source.json", "plan.json", "report.json", "execution.json")]
    require(plan == config and source == {"job_id": job["job_id"], "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"]} and report == capsule["report"] and execution == capsule["near_data_execution"],
        "actual matched report/source/plan differs")
    require(execution["job_id"] == job["job_id"] and execution["request_sha256"] == request["request_sha256"]
        and execution["worker_sha256"] == request["source_hashes"]["flux_matched_pass_worker.py"]
        and execution["parent_job_id"] == config["parent_job_id"]
        and execution["parent_audit_job_id"] == config["parent_audit_job_id"]
        and execution["parent_audit_sha256"] == config["parent_audit_sha256"]
        and execution["parent_inventory_unchanged"] is True and execution["mathematical_transitions"] == 56
        and execution["native_framework_forwards"] == 0, "actual matched execution differs")
    parent_inventory, parent = verify_parent(h, run, request)
    require(report["schema"] == "mdb-retrospective-matched-pass-replay-v1"
        and report["mechanics_status"] == "numeric-core-valid-custody-unassessed"
        and report["cells"] == list(CELLS) and report["logical_step"] == 1 and report["math_transitions"] == 56
        and report["native_forwards"] == 0 and report["source_job_id"] == config["parent_job_id"]
        and report["source_request_sha256"] == config["parent_request_sha256"]
        and report["sources"] == {"helper": CORE_SHA256, "analyzer": ANALYZER_SHA256, "memoization": MEMOIZATION_SHA256},
        "actual matched core scope/source differs")
    seal = read(data / "Q-seal.json")
    require(report["Q_seal_sha256"] == sha(data / "Q-seal.json")
        and seal["fingerprint"] == identity({k: v for k, v in seal.items() if k != "fingerprint"})
        and seal["sources"] == report["sources"] and seal["math_transitions"] == 56
        and seal["target_arrays_read_before_Q_seal"] == seal["native_forwards"] == 0
        and seal["kind"] == "retrospective-completed-Q-custody-seal", "completed retrospective Q custody seal differs")
    return data, report, execution, seal, actual, parent_inventory, parent


def descriptor(value):
    value = np.ascontiguousarray(value)
    require(value.dtype == np.float32 and value.size and np.isfinite(value).all(), "saved Q array ABI differs")
    result = {"shape": list(value.shape), "dtype": "float32",
        "fp32_le_sha256": hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()}
    if not np.any(value.view(np.uint32) & np.uint32(65535)):
        result["bf16_le_sha256"] = hashlib.sha256((value.view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest()
    return result


def validate_accounts(h, data, report, seal, parent_inventory):
    """Verify saved Q bytes and target archive custody; never execute equations."""
    count = 0
    for cell in CELLS:
        declared = seal["programs"][cell]
        program_path = h["file_path"](data, declared["path"])
        require(sha(program_path) == declared["sha256"], "Q program custody seal differs")
        program = read(program_path)
        rows = report["comparisons"][cell]["rows"]
        require(program["condition_id"] == cell and program["chain_length"] == len(program["transitions"]) == 28
            and program["logical_step"] == 1 and program["native_input_cut"] == seal["inputs"][cell]["cut"]
            and [row["local_index"] for row in rows] == list(LANDMARKS), "complete Q program/landmark census differs")
        for local, transition in enumerate(program["transitions"]):
            expected = ("project", None) if local == 0 else ("joint", local - 1) if local <= 5 else (
                "single", local - 6) if local <= 25 else ("readout", None) if local == 26 else ("commit", None)
            require(transition["logical_step"] == 1 and transition["local_index"] == local
                and (transition["kind"], transition["block_index"]) == expected
                and all((v["archive"] is not None) == (local in LANDMARKS) for v in transition["outputs"].values()),
                "Q complete operation/archived coverage differs")
        for row in rows:
            local = row["local_index"]
            qrow = program["transitions"][local]
            require(row["logical_step"] == 1 and row["transition_index"] == 28 + local
                and (row["kind"], row["block_index"]) == (qrow["kind"], qrow["block_index"])
                and set(row["fields"]) == set(qrow["outputs"]), "signed account landmark join differs")
            for name, field in row["fields"].items():
                q = field["Q"]
                require(q == qrow["outputs"][name], "signed account Q is not its completed sealed Q field")
                archive = q["archive"]
                path = h["file_path"](data, archive["path"])
                require(sha(path) == archive["sha256"] and path.stat().st_size == archive["bytes"], "Q archive bytes differ")
                value = np.load(path, allow_pickle=False)
                require(descriptor(value) == q["descriptor"] == archive["descriptor"], "Q archive full descriptor differs")
                del value
                for target in ("P", "N"):
                    record = field[target]["archive"]
                    original = parent_inventory["files"][str(h["relative"](record["path"]))]
                    require(record["sha256"] == original["sha256"] and original["filetype"] == "regular",
                        "P/N target archive is not in authenticated parent inventory")
                account = field["account"]
                require(account["signs"] == {"A": "Q-N", "B": "P-Q", "E": "P-N"}
                    and len({account[t]["native_rms"] for t in TERMS}) == 1
                    and len({account[t]["elements"] for t in TERMS}) == 1
                    and account["A_Q_minus_N"]["elements"] == int(np.prod(q["descriptor"]["shape"])),
                    "signed account orientation/common native denominator differs")
                count += 1
    require(count == 28, "all28 signed full-field accounts required")
    return {"full_field_accounts": count, "complete_Q_transitions": 56, "Q_full_archives_verified": count,
        "P_N_archive_refs_bound_to_parent_inventory": 2 * count,
        "scope": "Q archive bytes/descriptors verified locally. P/N scalar accounts are the authenticated actual replay result; their arrays are not duplicated or independently reduced again here."}


def relative_cross(account):
    native = account[TERMS[0]]["native_rms"]
    size = account[TERMS[0]]["elements"]
    return account["twice_dot_A_B"] / (size * native * native) if native else np.nan


def plot_working(report, stream, output):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharey="col", squeeze=False)
    all_crosses = []
    for i, cell in enumerate(CELLS):
        rows = [r for r in report["comparisons"][cell]["rows"] if stream in r["fields"]]
        require([r["local_index"] for r in rows] == [0, 5, 25], "working-state landmark coverage differs")
        x = [r["local_index"] for r in rows]
        accounts = [r["fields"][stream]["account"] for r in rows]
        for term, color, label, marker, size in zip(TERMS, COLORS, ("A = Q − N", "B = P − Q", "E = P − N"),
                ("s", "^", "o"), (8, 6, 4), strict=True):
            style = {"color": color, "label": label, "marker": marker, "markersize": size,
                "linestyle": "--", "markerfacecolor": "none" if term != TERMS[2] else color}
            axes[i, 0].plot(x, [a[term]["rms"] for a in accounts], **style)
            axes[i, 1].plot(x, [100 * a[term]["native_relative_rms"] if a[term]["native_relative_rms"] is not None
                else np.nan for a in accounts], **style)
        crosses = [relative_cross(a) for a in accounts]
        all_crosses.extend(v for v in crosses if np.isfinite(v))
        axes[i, 2].bar(x, crosses, width=1.6, color=["#b64040" if v < 0 else "#754c9c" for v in crosses])
        axes[i, 2].plot(x, crosses, "o", color="#171717", markersize=4)
        for column, title in enumerate(("Absolute signed-vector magnitude", "Common native-relative magnitude", "Signed normalized cross term")):
            ax = axes[i, column]
            ax.set(title=cell + ": " + title, xlabel="Logical pass1 observed exit")
            ax.set_xticks(x, ["Project", "After joint4", "After single19"])
            ax.grid(axis="y", alpha=.2)
            ax.axhline(0, color="#777777", linewidth=.7)
        axes[i, 0].set_ylabel("RMS of A/B/E, " + stream + " units")
        axes[i, 1].set_ylabel("RMS(A/B/E) / same native N RMS (%)")
        axes[i, 2].set_ylabel("2⟨A,B⟩ / (n × RMS(N)²)")
    # Include every term from both contexts; no limits derived from one series.
    for column in (0, 1):
        axes[0, column].set_ylim(bottom=0)
    span = max((abs(v) for v in all_crosses), default=0)
    axes[0, 2].set_ylim((-1.1 * span, 1.1 * span) if span else (-1, 1))
    if not span:
        for ax in axes[:, 2]:
            ax.text(.5, .85, "All cross terms exactly zero", ha="center", transform=ax.transAxes, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle("Retrospective logical pass1: " + stream + " signed error account", fontsize=17, y=.985)
    fig.text(.5, .015, "Q = G₁(N₁); P = original own-state forecast; N = clock-aligned native. E = A + B.\n"
        "Both prompt factors differ between selected contexts. Dashed segments guide across unobserved exits; no causal shares or isolated kernel attribution.",
        ha="center", fontsize=10, linespacing=1.5)
    fig.tight_layout(rect=(0, .08, 1, .95), h_pad=2.5, w_pad=2.5)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_commits(report, output):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    for i, (stream, local) in enumerate((("noise", 26), ("latent", 27))):
        accounts = [next(row for row in report["comparisons"][cell]["rows"] if row["local_index"] == local)["fields"][stream]["account"]
            for cell in CELLS]
        x = np.arange(2)
        for offset, (term, color, label) in enumerate(zip(TERMS, COLORS, ("A = Q − N", "B = P − Q", "E = P − N"), strict=True)):
            axes[i, 0].bar(x + (offset - 1) * .22, [a[term]["rms"] for a in accounts], width=.21, color=color, label=label)
            axes[i, 1].bar(x + (offset - 1) * .22, [100 * a[term]["native_relative_rms"] if a[term]["native_relative_rms"] is not None
                else np.nan for a in accounts], width=.21, color=color, label=label)
        cross = [relative_cross(a) for a in accounts]
        axes[i, 2].bar(x, cross, width=.5, color=["#b64040" if v < 0 else "#754c9c" for v in cross])
        for column, title in enumerate(("absolute magnitude", "common native-relative magnitude", "signed normalized cross term")):
            ax = axes[i, column]
            ax.set(title=stream + ": " + title)
            ax.set_xticks(x, CELLS)
            ax.grid(axis="y", alpha=.2)
            ax.axhline(0, color="#777777", linewidth=.7)
        axes[i, 0].set_ylabel("RMS of A/B/E, " + stream + " units")
        axes[i, 1].set_ylabel("RMS(A/B/E) / same native N RMS (%)")
        axes[i, 2].set_ylabel("2⟨A,B⟩ / (n × RMS(N)²)")
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle("Retrospective pass1 readout and commit: separate fields / units", fontsize=17, y=.985)
    fig.text(.5, .015, "A = Q − N; B = P − Q; E = P − N. Common denominator uses each same-cursor native field.\n"
        "Noise and committed latent are separate observables. These are whole-pass signed accounts, not isolated readout/Euler errors or causal shares.",
        ha="center", fontsize=10, linespacing=1.5)
    fig.tight_layout(rect=(0, .08, 1, .95), h_pad=2.5, w_pad=2.5)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def publish(run, output, *, custody_helper=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(run / "extracted"), "fresh output outside original result required")
    helper_path = Path(custody_helper or HERE / "publish_flux_prompt_factorial.py").resolve()
    h = helpers(helper_path)
    data, report, execution, seal, before, parent_inventory, parent = verify_run(run, h)
    accounts = validate_accounts(h, data, report, seal, parent_inventory)
    output.mkdir(parents=True)
    save(output / "signed-accounts.json", {"schema": "mdb-matched-pass-signed-accounts-v1", "comparisons": report["comparisons"],
        "definitions": {"Q": "G_1(actual native N1)", "P": "Original sealed own-state full program, same cursor",
            "N": "Actual clock-aligned native field", "A": "Q-N", "B": "P-Q", "E": "P-N=A+B"},
        "scope": SCOPE, "source_report_sha256": sha(data / "report.json"), "verification": accounts})
    for stream in ("image", "text"):
        plot_working(report, stream, output / (stream + "-account.png"))
    plot_commits(report, output / "readout-commit-account.png")
    lines = ["# Retrospective matched-input pass1 observations", "", SCOPE, "",
        "All RMS terms below use the same native denominator within each row. Cross is signed; negative does not imply total error reduction.", "",
        "| Context | Exit | Field | RMS(A) | RMS(B) | RMS(E) | N RMS | 2 dot(A,B) |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for cell in CELLS:
        for row in report["comparisons"][cell]["rows"]:
            exit_name = row["kind"] + (str(row["block_index"]) if row["block_index"] is not None else "")
            for name, field in row["fields"].items():
                a = field["account"]
                values = [a[t]["rms"] for t in TERMS] + [a[TERMS[0]]["native_rms"], a["twice_dot_A_B"]]
                lines.append("| " + " | ".join([cell, exit_name, name] + [f"{v:.9g}" for v in values]) + " |")
    with (output / "summary.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    summary = {"schema": "mdb-matched-pass-publication-summary-v1", "source": parent, "account_verification": accounts,
        "execution": execution, "conditions": list(CELLS), "logical_step": 1, "scope": SCOPE,
        "cross_term_normalization": "twice_dot_A_B / (elements × reported native_rms²); common native energy reconstructed from saved RMS",
        "coverage": "All28 signed fields at10 retained landmark transitions; complete56 Q transition descriptors. Other46 native exits not inferred.",
        "collection": {"files": before["file_count"], "expanded_bytes": before["expanded_bytes"], "collection_sha256": sha(run / "collection.json")},
        "source_report_sha256": sha(data / "report.json"), "Q_seal_sha256": sha(data / "Q-seal.json"),
        "parameter_cache_cleanup": execution["parameter_cache_cleanup"], "native_forwards": 0, "learned_weight_math": False}
    save(output / "numerical-summary.json", summary)
    require(h["inventory"](data) == before, "actual result changed during saved-only publication")
    outputs = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    receipt = {"schema": "mdb-matched-pass-publication-receipt-v1", "source_sha256": sha(__file__),
        "custody_helper_sha256": sha(helper_path), "inputs": {name: sha(h["file_path"](run, name)) for name in
            ("request.json", "job.json", "collection.json", "result.json", "worker.log", "admission-command.json",
             "collection-local-inventory.json", "collection-remote-before.json", "collection-remote-after.json")},
        "original_result_inventory_sha256": sha(run / "collection-local-inventory.json"), "original_result_unchanged": True,
        "outputs": outputs, "native_forwards": 0, "learned_weight_math": False, "scheduler_requests": 0, "scope": SCOPE}
    save(output / "publication-receipt.json", receipt)
    return {"output": str(output), "publication_receipt_sha256": sha(output / "publication-receipt.json"),
        "signed_accounts_sha256": sha(output / "signed-accounts.json"), "full_field_accounts": 28,
        "figures": [name for name in outputs if name.endswith(".png")]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--custody-helper", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.run, args.out, custody_helper=args.custody_helper), sort_keys=True))
