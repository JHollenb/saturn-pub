"""Freeze a conditional own-functional frontier and two row-sign edits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "mdb-qwen-sign-shortcut-20261106"
PACK = PARENT / "compact-collection-01/extracted/binding-minimal-result"
METADATA = PARENT / "frontier-metadata-v1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(PACK / "source-files.json") == "d573bffe5c0c3dfc3d1c353721f5696909f73b75fb6ff293aed2e905f4ec0c33"
    assert sha(METADATA / "transaction/seal.json") == "a6d61b11fa67bfa0eb3a50fee66f26584b452874a590c255163102362460e652"
    forecast = read(METADATA / "cases/long-description/forecast.json")
    inputs = read(METADATA / "cases/long-description/inputs.json")
    source = read(PACK / "source-source.json")
    raw_manifest = read(PACK / "source-files.json")["files"]
    positive = forecast["ports"]["plus"]
    selected = {name: positive[name] for name in ("h", "head", "R", "gate", "up", "product", "M", "Hnext")}
    selected.update(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], position_ids=forecast["position_ids"])
    # These are nonpersistent FP32 constructor buffers archived before forecasts.
    for name, record in source["rope_buffers"].items():
        selected["rope_buffer:" + name] = record
    for record in selected.values():
        assert raw_manifest[record["path"]] == dict(bytes=record["bytes"], sha256=record["sha256"])
        assert not record["path"].startswith("cases/long-description/native/")
    protocol = dict(
        schema="mdb-qwen-row-sign-selection-v1", parent_job="job-8c949b1c0278",
        parent_request_sha256="1a9b11e4bc739410dcd165b31cd0c0452aa5f0583e7731598bc1e2b8a87bdd5a",
        parent_raw_root="/mnt/big/mdb/research/job-8c949b1c0278/qwen-sign-shortcut-result",
        parent_raw_manifest_sha256=sha(PACK / "source-files.json"),
        parent_global_seal_sha256=sha(METADATA / "transaction/seal.json"),
        source_case="long-description", frontier="authenticated own-functional block18 input and positive ports; conditional, not a new raw-prefix forecast",
        edit_index=18, horizon=19, sequence_rows=512, fixed_split=256,
        source_arrays=selected, attention_dispatch=dict(backend="sdpa", gqa_policy="enable_gqa"),
        actual_native_mask="None", actual_native_is_causal=True, cache=False,
        arms=[dict(id="all-negative", negative_rows=[0, 512], support_closed_windows=[[0, 256], [256, 512]]),
              dict(id="prefix-negative", negative_rows=[0, 256], support_closed_windows=[[0, 256]], mixed_support_windows=[[256, 512]])],
        predictor="row-sign own-positive R/G/U; P=SiLU(s*G)*(s*U),M=linear(P,Wd,None),H=s*R+M; headhat=s*positive_head",
        expectations=["Global control and prefix support-closed queries are candidate numerical equalities; record raw-bit/signedzero separately.",
                      "Mixed-support rows need not satisfy the shortcut. Cancellation can produce equality; no all-rows-must-miss gate.",
                      "Freeze complete S/F edited states and full own-positive response hashes before all3native component targets."],
        comparisons=["shortcut-minus-full-functional", "full-functional-minus-native", "shortcut-minus-native"],
        baseline="Archived functional positive basis versus fresh standalone native baseline, reported separately.",
        original_component=dict(parameter_prefix="model.layers.18.", selected_parameter_count=11, selected_parameter_payload_bytes=201861632),
        physical_plan=dict(functional_full_negative_blocks=2, shortcut_down_projections=2, native_block_calls=3,
                           tokenizer=0, embedding=0, full_model=0, LM=0, FLUX=0, VAE=0, noise=0),
        scientific_gates=[], limited_to="Source-qualified component-native primitive profile; no public Qwen Act, semantic route or independent kernel claim.",
    )
    output = ROOT / "selection-v1"
    output.mkdir(exist_ok=False)
    with (output / "protocol.json").open("x") as stream:
        json.dump(protocol, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"protocol_sha256": sha(output / "protocol.json"), "source_arrays": len(selected), "source_array_bytes": sum(r["bytes"] for r in selected.values()), "actual_model_calls": 0}))


if __name__ == "__main__":
    main()
