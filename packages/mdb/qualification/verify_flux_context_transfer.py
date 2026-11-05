"""Close audited context-transfer evidence and propose a bounded family record."""
import argparse
import json
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from mdb.computation_atlas import ComputationAtlas


def verify(root, previous):
    here = Path(__file__).resolve().parent
    helper = runpy.run_path(str(here / "analyze_flux_context_transfer.py"))
    read, sha = helper["read"], helper["sha"]
    audit = read(root / "audit-v2.json")
    review = read(root / "image-review-v1.json")
    data = root / "collection-01/extracted/binding-minimal-result"
    assert audit["mechanics_status"] == "verified"
    custody = runpy.run_path(str(here / "flux_family_custody.py"))["verify"](root / "collection-01")
    assert custody == audit["custody"]
    for name, fingerprint in read(root / "audit-source-seal-v2.json")["sources"].items():
        assert sha(root / "audit-sources-v2" / name) == fingerprint == audit["audit_sources"][name]
    rows = read(data / "rows.json")
    fields = lambda fp: {v["key"]["value"]: v["value"] for v in read(data / "store/cuts" / (fp + ".json"))["manifest"]["payload"]["items"]}
    control = fields(rows[0]["writer"]["entry"])
    bindings, halves = {}, {}
    for row in rows:
        name = row["id"]
        assert sha(data / row["png"]) == row["png_sha256"] == review["rows"][name]["png_sha256"]
        current = fields(row["writer"]["entry"])
        bindings[name] = [key for key in current if current[key] != control[key]]
        image = np.asarray(Image.open(data / row["png"]), dtype=np.float64)
        chroma = (image[..., 2] - image[..., 0]) / (image.sum(-1) + 1.0)
        halves[name] = dict(left=float(chroma[:, :256].mean()), right=float(chroma[:, 256:].mean()))
    for name, value in halves.items():
        value["delta_left"] = value["left"] - halves["control"]["left"]
        value["delta_right"] = value["right"] - halves["control"]["right"]
        assert abs(value["delta_right"] - value["delta_left"] - audit["outcomes"][name]["signed_effect"]) < 1e-12
    atlas = ComputationAtlas(root / "public-records-v2/atlas")
    records = atlas.find()
    publication = read(root / "public-records-v2/provenance.json")
    assert len(records) == publication["equation_occurrences"] == 8
    for record in records:
        context = record["equation"]["tested_context"]
        assert len(atlas.find(applicable_to=context)) == 1
        assert not atlas.find(applicable_to={**context, "source_seed": 110})
        for artifact in record["provenance"]["artifacts"]:
            assert sha(artifact["path"]) == artifact["sha256"]
    suites = ET.parse(root / "installed-instrument-tests-v2.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites) == 0
    assert len(read(data / "parameter-artifacts.json")["artifacts"]) == 169
    assert read(previous / "audit-v1.json")["mechanics_status"] == "verified"
    refs = [dict(path=str(p), sha256=sha(p)) for p in (
        root / "audit-v2.json", root / "image-review-v1.json", root / "run-01/request.json",
        root / "public-records-v2/provenance.json", previous / "audit-v1.json")]
    family = dict(schema="mdb-context-transfer-family-candidate-v1", generic_id="family.flux.working-carrier.0001",
        alias="Working text can regenerate image coloration after image restoration",
        status="family-candidate; manually interpreted, not an induced or validated semantic rule",
        supporting_contexts=[dict(relation="sphere left of cube", source=str(previous)),
                             dict(relation="sphere right of cube", source=str(root))],
        applicability=dict(checkpoint=read(data / "plan.json")["checkpoint_sha256"], seed=109,
            backend="CUDA BF16 B1 512px four passes", receiver="donor image pulse step1/joint4 then common project2",
            edit="conditioning rows4/11 before project2, reset before project3",
            restoration="working I or T after step2/joint0", addresses="Only these source tokenizer/encoder contexts qualified"),
        mechanism="Carrier asymmetry recurs across layouts. Right layout directly shows image-facing joint1 attention rewrite under equal incoming I; baseline A restores local I/MLP but edited T permits later regeneration.",
        continuation="Partial right-layout coloration survives I restoration and one-site A or M return restoration; T restoration suppresses visible edit with nonzero residual; IT restoration exactly restores baseline.",
        counterexamples=[dict(claim="complete color swap transfers", rows=["positive", "restore-I", "restore-I-A", "restore-I-M"],
                             observed="Both halves move chromatically but sphere remains strongly red with blue band and cube remains visibly blue.")],
        next_cut="Keep the two edited C rows at project3 versus current reset, matching restore-I and replaying its control.",
        unresolved=["duration/refresh requirement", "contextual conditioning outside the lexical slice", "receiver dependence", "new seeds/token addresses", "object-level semantic generalization"],
        provenance=refs)
    result = dict(schema="mdb-context-transfer-verification-v1", mechanics_status="verified", instrument_status="applicable",
        trend_status="recurring carrier asymmetry; incomplete semantic transfer", terminal_status="unassessed",
        counts=audit["counts"], installed_instrument_tests=sum(int(s.attrib["tests"]) for s in suites),
        categorical_arms=dict(matched=sum(review["rows"][r["id"]]["categorical_prediction_matched"] for r in rows if r["id"] != "validation"), total=7,
                              validation="Separate exact replay, not semantic test"),
        signed_predictions=dict(matched=sum(audit["outcomes"][n]["signed_effect"] > 0 for n in ("positive", "restore-I", "restore-I-A", "restore-I-M")), total=4),
        writer_entry_different_fields=bindings, half_frame_chromaticity=halves,
        atlas=publication, new_model_calls_in_verification=0, provenance=refs,
        qualification_sources={p.name: sha(p) for p in [Path(__file__), here / "analyze_flux_context_transfer.py", here / "publish_flux_context_transfer.py", here / "flux_context_transfer.py"]})
    for name, value in (("family-candidate-v1.json", family), ("verification-v1.json", result)):
        with (root / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return {k: result[k] for k in ("mechanics_status", "categorical_arms", "signed_predictions", "installed_instrument_tests", "new_model_calls_in_verification")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), args.previous.resolve()), sort_keys=True))
