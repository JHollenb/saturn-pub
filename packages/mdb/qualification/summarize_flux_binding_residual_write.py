"""Reduce the saved residual/write audit without running a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(compact):
    import numpy as np
    from PIL import Image

    compact = Path(compact)
    audit_path = compact / "audit.json"
    report = json.loads(audit_path.read_text())
    if report["schema"] != "mdb-binding-residual-write-saved-audit-v1":
        raise ValueError("saved residual/write audit required")
    cases = report["cases"]

    def pixels(key):
        item = report["original_pngs"][key]["native"]
        path = compact / item["path"]
        if (path.stat().st_size != item["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]):
            raise ValueError("original PNG differs: " + key)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float64)

    def difference(left, right):
        delta = left - right
        return dict(rms=float(np.sqrt(np.mean(delta * delta))), signed_mean=float(delta.mean()),
                    max_abs=float(np.abs(delta).max()), changed=int(np.count_nonzero(delta)))

    contexts = {}
    for context in ("anchor", "wording"):
        prefix = "binding-" + context + "/"
        arms = {arm: pixels(prefix + arm) for arm in ("whole_Q", "residual_only", "write_only", "both_RW")}
        source = report["residual_write_sources"]["cases"][context]
        contexts[context] = dict(support=source["support"], geometry=source["declared_analysis"]["geometry"],
            source_arithmetic=source["numerical"], arms={})
        for arm, image in arms.items():
            case = cases[prefix + arm]
            contexts[context]["arms"][arm] = dict(
                png_rgb_vs_Q=difference(image, arms["whole_Q"]),
                png_rgb_vs_RW=difference(image, arms["both_RW"]),
                signed_image_effects=case["signed_image_effects"]["native"],
                first_consumer={name: port["signed_effects"]["native"]
                                for name, port in case["first_consumer"].items()},
                landed_l2=case["acts"]["native"][0]["landed_l2"])
    known = [case["known_replay"] for case in cases.values() if case["known_replay"] is not None]
    return dict(schema="mdb-binding-residual-write-observations-v1", model_calls=0,
        audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        native_job_id=report["native_job_id"], native_request_sha256=report["native_request_sha256"],
        coverage=report["coverage"], chronology=report["chronology"], contexts=contexts,
        errors=dict(image_pairs=len(cases),
            decoded_and_pixel_max_abs=max(m["max_abs"] for case in cases.values() for m in case["image_errors"].values()),
            image_nonzero_fields=sum(bool(m["changed"]) for case in cases.values() for m in case["image_errors"].values()),
            first_consumer_max_abs=max(port["endpoint_error"]["max_abs"] for case in cases.values()
                                       for port in case["first_consumer"].values()),
            first_consumer_nonzero_fields=sum(bool(port["endpoint_error"]["changed"]) for case in cases.values()
                                              for port in case["first_consumer"].values()),
            known_pngs_exact=sum(case["png_bytes_exact"] for case in known),
            known_payload_max_abs=max(m["max_abs"] for case in known for m in case["payload"]["tensors"].values())),
        interactions={key: dict(native=value["magnitudes"]["native"], regions=value["regional_interactions"],
                                forecast_native_errors=value["forecast_native_errors"])
                      for key, value in report["interactions"].items()},
        limitations=["Two inherited source contexts, fixed Q nontext and observed joint0 sources.",
            "The write is the whole finite block output minus input, including both learned paths and residual additions.",
            "RGB distances and interaction norms are numerical observations, not semantic probabilities or contributions.",
            "Functional predictions share native Torch primitives; the saved-only reader performs no learned forward."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.compact)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(errors=result["errors"], coverage=result["coverage"]), sort_keys=True))
