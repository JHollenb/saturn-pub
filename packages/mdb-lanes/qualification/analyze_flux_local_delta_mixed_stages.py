"""Saved-only mixed contrasts, inheriting the campaign's verified custody audit."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


COEFFICIENTS = {
    "row11heavy": 1,
    "row4heavy": -1,
    "row11heavy.image1": -1,
    "row4heavy.image1": 1,
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(predicted, native):
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    pr, nr = rms(predicted), rms(native)
    error = rms(predicted - native)
    return dict(
        scalars=int(native.size), native_rms=nr, predicted_rms=pr,
        error_rms=error, relative_rms=None if nr == 0 else error / nr,
        predicted_over_native_norm=None if nr == 0 else pr / nr,
        cosine=None if pr == 0 or nr == 0 else float(np.mean(predicted * native) / pr / nr),
    )


def analyze(collection, audit_path, output):
    read = lambda p: json.loads(p.read_text())
    audit = read(audit_path)
    if audit.get("mechanics_status") != "verified":
        raise ValueError("verified campaign audit prerequisite missing")
    data = collection / "extracted/binding-minimal-result"
    envelope = read(collection / "report.json")
    custody = audit["custody"]
    for key in ("job_id", "request_sha256", "archive_sha256"):
        if custody[key] != envelope[key]:
            raise ValueError("audit belongs to another collection")
    if sha(data / "files.json") != envelope["files_manifest_sha256"]:
        raise ValueError("audited file manifest changed")
    inventory = read(data / "files.json")
    consumed = {}

    def bound(name):
        path = data / name
        digest = sha(path)
        if inventory[name] != digest:
            raise ValueError("audited source bytes changed: " + name)
        consumed[name] = digest
        return path

    forecasts = read(bound("forecasts.json"))
    native = read(bound("native.json"))
    summary = read(bound("summary.json"))
    if summary != envelope["report"]:
        raise ValueError("audited summary differs")
    reader_path = Path(__file__).with_name("analyze_flux_binding_minimal.py")
    read_slot = runpy.run_path(str(reader_path))["read_slot"]

    def array(record):
        path = bound(record["path"])
        value = np.load(path, allow_pickle=False)
        if (consumed[record["path"]] != record["sha256"]
                or list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]
                or not np.isfinite(value).all()):
            raise ValueError("audited array descriptor differs")
        return value.astype(np.float64)

    def exit_slot(fingerprint, role):
        cut = read(bound("store/cuts/" + fingerprint + ".json"))
        page = read(bound("store/pages/" + cut["slots"][role] + ".json"))
        if page["tensor_blob"] is not None:
            bound("store/blobs/" + page["tensor_blob"] + ".safetensors")
        value = read_slot(data / "store", fingerprint, role).astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite audited exit")
        return value

    reports = {}
    for role in ("image", "text"):
        reports[role] = {}
        for field in ("attention", "mlp_input", "mlp", "exit"):
            predicted = actual = None
            for name, sign in COEFFICIENTS.items():
                p = array(forecasts[name]["local"]["fields"][role][field])
                n = (exit_slot(native[name]["exit"], role) if field == "exit"
                     else array(native[name]["fields"][field + "_" + role]))
                if p.shape != n.shape:
                    raise ValueError("mixed contrast geometry differs")
                if predicted is None:
                    predicted, actual = np.zeros_like(p), np.zeros_like(n)
                predicted += sign * p
                actual += sign * n
            reports[role][field] = compare(predicted, actual)
        existing = audit["factorial"]["local"][role]["contrasts_with_norms"]["receiver_by_pattern"]
        for key in ("relative_rms", "cosine", "predicted_over_native_norm"):
            if not np.isclose(reports[role]["exit"][key], existing[key], rtol=1e-12, atol=1e-12):
                raise ValueError("exit contrast does not reproduce original audit")
    result = dict(
        schema="mdb-local-delta-mixed-stages-v1", mechanics_status="verified-saved-calculation",
        prerequisite=dict(path=str(audit_path), sha256=sha(audit_path),
                          inherited="Source/entry custody, typed writes and chronology from verified campaign audit",
                          custody=custody),
        input_manifest_sha256=sha(data / "files.json"),
        collection_report_sha256=sha(collection / "report.json"),
        consumed_files=consumed, coefficients=COEFFICIENTS, fields=reports,
        calculation_source_sha256=sha(Path(__file__)), slot_reader_sha256=sha(reader_path),
        numpy_version=np.__version__, native_calls=0, model_loads=0,
        scope="Post-hoc diagnostic of local predictions at saved projected-attention, MLP-input, MLP-output and exit boundaries. Earliest captured error is not a first-primitive attribution. No new forecast, model execution or target-port substitution.",
    )
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return dict(output=str(output), sha256=sha(output), fields=reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection", "audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.collection.resolve(), args.audit.resolve(), args.output.resolve()), sort_keys=True))
