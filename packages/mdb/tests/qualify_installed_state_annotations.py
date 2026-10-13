"""Installed-wheel state-annotation qualification with upstream imports blocked."""
from __future__ import annotations

import argparse
import importlib.abc
import json
import sys
from pathlib import Path


class _UpstreamBlocker(importlib.abc.MetaPathFinder):
    blocked = {"saturn", "common", "mrun"}

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname.split(".", 1)[0] in self.blocked:
            raise ImportError(f"qualification blocked upstream import: {fullname}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    target = args.target.resolve()
    output = args.out.resolve()
    sys.path.insert(0, str(target))
    sys.meta_path.insert(0, _UpstreamBlocker())

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import mdb
    from mdb.bench import BenchSession
    from mdb.state_annotations import StateAnnotationStore

    if not Path(mdb.__file__).resolve().is_relative_to(target):
        raise AssertionError(f"MDB did not import from installed target: {mdb.__file__}")
    if any(name.split(".", 1)[0] in _UpstreamBlocker.blocked for name in sys.modules):
        raise AssertionError("an upstream package was imported")

    torch.manual_seed(20261008)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attn_implementation="eager",
    )).eval().to(torch.bfloat16)
    root = output / "session"
    bench = BenchSession.from_resident(model, root=root, continuation_capacity=8)
    parent = bench.prefill([1, 2, 3])
    before = json.loads((root / "selected.json").read_text())
    row = bench.annotate_state(
        "installed recipient",
        cut_id=parent,
        namespace="installed-qualification",
        role="recipient-parent",
        stage="prefill",
        geometry={"components": ["value"], "layers": [0, 1], "rows": [0, 1]},
    )
    after = json.loads((root / "selected.json").read_text())
    assert after["manifest"] == before["manifest"]
    exported = output / "exported"
    bench.export(exported)
    bench.close()

    store = StateAnnotationStore.open(exported)
    report = store.verify()
    assert report["mechanics_status"] == "valid" and report["annotation_count"] == 1
    assert store.show(row["qualified_label"])["model_input_binding"]["token_count"] == 3
    print(json.dumps({"annotation_id": row["annotation_id"], "verification": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
