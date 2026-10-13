import json
import shutil

import pytest
import torch
from transformers import MambaConfig, MambaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchSession
from mdb.cli import main as cli_main
from mdb.model_os_contracts import content_digest
from mdb.state_annotations import StateAnnotationError, StateAnnotationStore


@pytest.fixture
def model():
    torch.manual_seed(103)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attn_implementation="eager",
    )).eval().to(torch.bfloat16)


def _bench(model, root, **kwargs):
    return BenchSession.from_resident(model, root=root, continuation_capacity=8, **kwargs)


def test_journal_annotations_preserve_cut_and_manifest_identities(model, tmp_path, capsys):
    root = tmp_path / "bench"
    bench = _bench(model, root)
    parent = bench.prefill([1, 2, 3])
    bench.step("layer")
    child = bench.checkpoint("post-write")
    selected_before = json.loads((root / "selected.json").read_text())
    child_bytes = bench.artifacts._path(child).read_bytes()

    first = bench.annotate_state(
        "recipient ready",
        cut_id=child,
        namespace="frontier-v1",
        role="recipient-parent",
        stage="postwrite",
        scientific_source_cut_ids=[parent],
        recipient_parent_cut_id=parent,
        tags=["kv", "frontier"],
        native_act_refs=["act://replace/frontier"],
        consumer_refs=["bench.qwen2.native"],
        geometry={"components": ["key", "value"], "layers": [0, 1],
                  "rows": {"start": 48, "stop": 75}},
        details={"arm": "coherent-frontier"},
        producer={"kind": "experiment", "id": "frontier-reduction-v1"},
    )
    selected_after = json.loads((root / "selected.json").read_text())
    assert selected_after["manifest"] == selected_before["manifest"]
    assert bench.artifacts._path(child).read_bytes() == child_bytes
    assert first["provenance"]["authority"] == "descriptive-only"
    assert first["provenance"]["semantic_claim"] is False
    assert first["session_binding"]["session_id"] == "bench"
    assert first["model_input_binding"]["token_count"] == 3
    assert first["native_writer_clock"]["boundary"] == "layer"
    assert first["immutable_parent_cut_id"] == parent
    assert first["qualified_label"].startswith("frontier-v1::recipient ready@sha256:")

    # An identical annotation is idempotent. A reused human label at another
    # immutable cut is legal and must be resolved through its qualified label.
    assert bench.annotate_state(
        "recipient ready", cut_id=child, namespace="frontier-v1",
        role="recipient-parent", stage="postwrite",
        scientific_source_cut_ids=[parent], recipient_parent_cut_id=parent,
        tags=["frontier", "kv"], native_act_refs=["act://replace/frontier"],
        consumer_refs=["bench.qwen2.native"],
        geometry={"components": ["key", "value"], "layers": [0, 1],
                  "rows": {"start": 48, "stop": 75}},
        details={"arm": "coherent-frontier"},
        producer={"kind": "experiment", "id": "frontier-reduction-v1"},
    )["annotation_id"] == first["annotation_id"]
    second = bench.annotate_state(
        "recipient ready", cut_id=parent, namespace="frontier-v1",
        role="scientific-source", stage="parent", tags=["source"],
    )
    assert second["annotation_id"] != first["annotation_id"]
    assert len(bench.list_state_annotations(label="recipient ready")) == 2
    with pytest.raises(StateAnnotationError, match="ambiguous label"):
        bench.show_state_annotation("recipient ready")
    assert bench.show_state_annotation(first["qualified_label"])["cut_id"] == child
    lineage = bench.state_lineage(child)
    assert [row["cut_id"] for row in lineage["cuts"]][-2:] == [parent, child]
    assert bench.verify_state_annotations()["annotation_count"] == 2

    exported = tmp_path / "exported"
    bench.export(exported)
    bench.close()
    offline = StateAnnotationStore.open(exported)
    assert offline.verify()["annotation_count"] == 2
    assert offline.show(first["annotation_id"])["geometry"]["rows"] == {"start": 48, "stop": 75}
    with pytest.raises(StateAnnotationError, match="read-only"):
        offline.annotate("offline mutation", cut_id=child, role="invalid", stage="offline")
    assert cli_main(["states", "--root", str(exported), "list", "--tag", "frontier"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [row["annotation_id"] for row in listed] == [first["annotation_id"]]


def test_legacy_sidecar_is_crash_safe_and_tamper_evident(model, tmp_path):
    root = tmp_path / "legacy"
    bench = _bench(model, root, publication_mode="legacy")
    cut = bench.prefill([4, 5, 6])
    selected_before = (root / "selected.json").read_bytes()
    annotation = bench.annotate_state(
        "native parent", cut_id=cut, namespace="legacy-test",
        role="native-parent", stage="prefill",
        geometry={"components": ["value"], "layers": [0, 1], "rows": [1, 2, 3]},
    )
    assert (root / "selected.json").read_bytes() == selected_before
    bench.close()

    store = StateAnnotationStore.open(root)
    assert store.verify()["annotation_count"] == 1
    orphan = root / "state-annotations/objects" / ("f" * 64 + ".json")
    orphan.write_text("{}")
    with pytest.raises(StateAnnotationError, match="orphan annotation object"):
        store.verify()
    orphan.unlink()

    object_path = root / store.list()[0]["object_path"]
    original = object_path.read_bytes()
    object_path.write_bytes(original + b"\n")
    with pytest.raises(StateAnnotationError, match="object bytes changed"):
        store.verify()
    object_path.write_bytes(original)

    copied = tmp_path / "tampered-index"
    shutil.copytree(root, copied)
    index = json.loads((copied / "state-annotations/index.json").read_text())
    index["entries"][0]["role"] = "invented-role"
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    index["content_sha256"] = content_digest(body)
    (copied / "state-annotations/index.json").write_text(
        json.dumps(index, sort_keys=True, separators=(",", ":")))
    with pytest.raises(StateAnnotationError, match="summary differs"):
        StateAnnotationStore.open(copied).verify()
    assert store.show(annotation["annotation_id"])["role"] == "native-parent"

    escaped = tmp_path / "escaped-index"
    shutil.copytree(root, escaped)
    index = json.loads((escaped / "state-annotations/index.json").read_text())
    index["entries"][0]["object_path"] = "state-annotations/objects/../../selected.json"
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    index["content_sha256"] = content_digest(body)
    (escaped / "state-annotations/index.json").write_text(
        json.dumps(index, sort_keys=True, separators=(",", ":")))
    with pytest.raises(StateAnnotationError, match="outside the session root"):
        StateAnnotationStore.open(escaped).verify()


def test_annotation_validation_refuses_fictional_bindings_and_unsafe_details(model, tmp_path):
    bench = _bench(model, tmp_path / "bench")
    cut = bench.prefill([7, 8, 9])
    with pytest.raises(StateAnnotationError, match="not registered"):
        bench.annotate_state("bad", cut_id="cas://sha256/" + "0" * 64,
                             role="source", stage="prefill")
    with pytest.raises(StateAnnotationError, match="unsupported type"):
        bench.annotate_state("bad", cut_id=cut, role="source", stage="prefill",
                             details={"tensor": object()})
    with pytest.raises(StateAnnotationError, match="0 <= start < stop"):
        bench.annotate_state("bad", cut_id=cut, role="source", stage="prefill",
                             geometry={"rows": {"start": 9, "stop": 2}})
    with pytest.raises(TypeError):
        # Model and input bindings are derived from the cut and are deliberately
        # absent from the caller-controlled API.
        bench.annotate_state("bad", cut_id=cut, role="source", stage="prefill",
                             model_input_binding={"token_count": 999})
    bench.close()


def test_journal_interruption_retains_only_an_unindexed_object(model, tmp_path, monkeypatch):
    root = tmp_path / "bench"
    bench = _bench(model, root)
    cut = bench.prefill([10, 11, 12])
    original = StateAnnotationStore._write

    def fail_index(self, path, raw, *, immutable):
        if path == self.index_path:
            raise OSError("injected index publication failure")
        return original(self, path, raw, immutable=immutable)

    monkeypatch.setattr(StateAnnotationStore, "_write", fail_index)
    with pytest.raises(OSError, match="injected index"):
        bench.annotate_state("interrupted", cut_id=cut, role="recipient", stage="prefill")
    monkeypatch.setattr(StateAnnotationStore, "_write", original)
    report = bench.verify_state_annotations()
    assert report["annotation_count"] == 0
    assert report["orphan_object_count"] == 1
    bench.close()


def test_family_session_derives_native_state_binding(tmp_path):
    torch.manual_seed(107)
    model = MambaForCausalLM(MambaConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=2,
        state_size=4, expand=2, conv_kernel=4,
    )).eval()
    bench = BenchSession.from_resident(model, root=tmp_path / "mamba")
    cut = bench.prefill([1, 2, 3])
    row = bench.annotate_state(
        "recurrent parent", cut_id=cut, namespace="family-test",
        role="recipient-parent", stage="prefill",
        geometry={"components": ["ssm-state"], "layers": [0, 1]},
    )
    assert row["session_binding"]["family"] == "mamba"
    assert row["model_input_binding"]["kind"] == "family-native-state"
    assert row["native_writer_clock"]["boundary"] == "token"
    assert bench.verify_state_annotations()["annotation_count"] == 1
    bench.close()
