"""Summarize audited lifecycle arrays and original PNGs, with zero model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(compact):
    import numpy as np
    from PIL import Image

    compact = Path(compact)
    report_path = compact / "audit.json"
    report = json.loads(report_path.read_text())
    cases = report["cases"]

    def pixels(key):
        item = report["original_pngs"][key]["native"]
        path = compact / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("original PNG hash differs")
        with Image.open(path) as source:
            return np.asarray(source.convert("RGB"), dtype=np.float64)

    def distance(left, right):
        value = left - right
        return dict(rms=float(np.sqrt(np.mean(value * value))),
                    max_abs=float(np.max(np.abs(value))), changed=int(np.count_nonzero(value)))

    binding, fixed_cd = {}, {}
    for cid in ("anchor", "seed", "wording", "objects"):
        rid = "binding-" + cid
        whole_p, whole_q = pixels(rid + "/whole_P"), pixels(rid + "/whole_Q")
        binding[cid] = dict(support=report["binding_sources"][cid]["support"],
                            source=report["binding_sources"][cid]["carriage"], arms={})
        for arm in ("whole_Q", "support_P", "complement_P", "whole_P"):
            key = rid + "/" + arm
            case = cases[key]
            binding[cid]["arms"][arm] = dict(
                vs_whole_Q=distance(pixels(key), whole_q),
                vs_whole_P=distance(pixels(key), whole_p),
                posthoc_whole_endpoint_projection=case["conditional_endpoint_axis"]["projections"]["native"],
                landed_l2=case["acts"]["native"][0]["landed_l2"],
            )
    for cid in ("fresh0s613", "foxcat0s821"):
        rid = "cd-" + cid
        fixed_cd[cid] = {}
        for arm in ("sham", "common-only", "differential-only", "both"):
            case = cases[rid + "/" + arm]
            fixed_cd[cid][arm] = dict(
                pixel_effect=case["signed_image_effects"]["native"]["pixels"],
                decoded_effect=case["signed_image_effects"]["native"]["decoded"],
                landed_l2=case["acts"]["native"][0]["landed_l2"],
                first_consumer_rms={name: value["signed_effects"]["native"]["rms"]
                                    for name, value in case["first_consumer"].items()},
            )
    known = [case["known_replay"] for case in cases.values() if case["known_replay"] is not None]
    return dict(
        schema="mdb-lifecycle-observation-summary-v1",
        audit_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(), model_calls=0,
        coverage=report["coverage"], chronology=report["chronology"],
        errors=dict(
            decoded_and_pixels_max_abs=max(metric["max_abs"] for case in cases.values()
                                          for metric in case["image_errors"].values()),
            image_pairs_bit_exact=sum(all(m["bit_exact"] for m in case["image_errors"].values())
                                      for case in cases.values()),
            first_consumer_max_abs=max(port["endpoint_error"]["max_abs"] for case in cases.values()
                                       for port in case["first_consumer"].values()),
            known_pngs_exact=sum(case["png_bytes_exact"] for case in known),
            known_payload_max_abs=max(metric["max_abs"] for case in known
                                      for metric in case["payload"]["tensors"].values()),
        ),
        binding=binding, fixed_cd=fixed_cd,
        interaction_rms={rid: {field: value["rms"] for field, value in record["magnitudes"]["native"].items()}
                         for rid, record in report["interactions"].items()},
        limitations=[
            "Supplied historical states and original shared numerical primitives; source writer states were observed.",
            "RGB norms and posthoc endpoint projections are numerical diagnostics, not semantic probabilities or causal percentages.",
            "Fixed C/D payload bytes do not make landed BF16 ADD differences identical across receiving states.",
            "Four dependent binding contexts and two receivers do not establish arbitrary-prompt generalization.",
        ],
    )


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
