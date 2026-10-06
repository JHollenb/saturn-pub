"""Installed-wheel hidden-memory qualification with upstream imports blocked."""

from __future__ import annotations

import argparse
import importlib.abc
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


class _UpstreamBlocker(importlib.abc.MetaPathFinder):
    blocked = {"saturn", "common", "mrun"}

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname.split(".", 1)[0] in self.blocked:
            raise ImportError(f"qualification blocked upstream import: {fullname}")
        return None


class _TinyTokenizer:
    eos_token_id = 0

    def get_vocab(self):
        return {f"t{token}": token for token in range(64)}

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [1 + (ord(character) % 61) for character in text][:64]

    def decode(self, token_ids, skip_special_tokens=False):  # noqa: ARG002
        return " ".join(f"t{int(token)}" for token in token_ids)


def _selector():
    return {"layers": [0], "row_start": 1, "row_stop": 2, "components": ["key"]}


def _probe_spec():
    return {
        "name": "installed-wheel-probe",
        "intervention": {"mode": "scale", "selector": _selector(), "value": 0.5},
        "query_token_ids": [7],
        "continuation_tokens": 1,
        "observable": {"kind": "token_margin", "token_id": 2, "against_token_id": 3},
        "max_native_operations": 64,
        "max_capture_bytes": 64 << 20,
    }


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
    from mdb.bench_debugger import BenchDebuggerAdapter
    from mdb.bench_verify import verify
    from mdb.debugger_cli import DebugShell, _memory_table
    from mdb.debugger_protocol import BreakpointSpec, DebugCapability, InMemoryDebugTransport
    from mdb.debugger_session import DebugSession
    from mdb.debugger_worker import DebugWorkerController

    if not Path(mdb.__file__).resolve().is_relative_to(target):
        raise AssertionError(f"MDB did not import from installed target: {mdb.__file__}")
    forbidden = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in _UpstreamBlocker.blocked
    )
    if forbidden:
        raise AssertionError(f"upstream modules imported: {forbidden}")

    torch.manual_seed(20261005)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attn_implementation="eager",
        )
    ).eval().to(device="cpu", dtype=torch.bfloat16)
    bench = BenchSession.from_resident(
        model,
        _TinyTokenizer(),
        root=output / "session",
        continuation_capacity=8,
    )
    try:
        parent = bench.prefill([1, 2, 3])
        read = bench.memory_read(
            {"cut_id": parent, "row_start": 0, "row_stop": 3, "max_rows": 1}
        )
        assert read["truncated"] and read["next_row_start"] == 1
        assert len(read["rows"]) == 1 and read["rows"][0]["components"]

        write = bench.memory_write({"mode": "zero", "selector": _selector()})
        assert write["cut_id"] != parent and write["act_ref"].startswith("cas://sha256/")
        difference = bench.memory_diff(
            {
                "left_cut": parent,
                "right_cut": write["cut_id"],
                **_selector(),
                "values": False,
            }
        )
        cell = next(iter(difference["rows"][0]["components"].values()))
        assert not cell["equal"] and cell["delta"]["norm"] > 0

        bench.restore(parent)
        direct_probe = bench.memory_probe(_probe_spec())
        assert direct_probe["cut_id"] == parent
        assert direct_probe["report"]["selected_parent_restored_exact"]
        assert "signed_raw_logit_contrast" in _memory_table(direct_probe)

        adapter = BenchDebuggerAdapter(bench)
        capabilities = adapter.capabilities
        capability = DebugCapability(
            job_id="installed-wheel",
            session_id=bench.session_id,
            family=capabilities.family,
            model_identity=bench.identity,
            boundaries=capabilities.boundaries,
            inspectable_slots=capabilities.inspectable_slots,
            writable_slots=capabilities.writable_slots,
            exact_replay=True,
            optional_operations=adapter.optional_operations(),
        )
        worker = DebugWorkerController(capability, adapter)
        session = DebugSession(InMemoryDebugTransport(worker), job_id=capability.job_id)
        session.add_breakpoint(BreakpointSpec("token", 0))
        worker.safe_point("token", 0, {"phase": "committed-token"})
        shell = DebugShell(session)
        stream = io.StringIO()
        with redirect_stdout(stream):
            assert shell.execute(
                "memory read "
                + json.dumps(
                    {"cut_id": parent, "layers": [0], "row_start": 0, "max_rows": 1}
                )
            )
            assert shell.execute("memory diff " + json.dumps({
                "left_cut": parent, "right_cut": write["cut_id"], **_selector(),
                "values": False,
            }))
            assert shell.execute("memory probe " + json.dumps(_probe_spec()))
        rendered = stream.getvalue()
        assert "layer row token" in rendered
        assert "left_norm" in rendered
        assert "native" in rendered and "candidate" in rendered and "restored" in rendered
        assert "signed_raw_logit_contrast" in rendered

        exported = bench.export(output / "export")
        verification = verify(exported["root"])
        assert verification["mechanics_status"] == "valid"
        assert verification["counts"]["memory_probes"] >= 2
        receipt = {
            "mdb_file": str(Path(mdb.__file__).resolve()),
            "parent_cut": parent,
            "write_cut": write["cut_id"],
            "read_truncated": read["truncated"],
            "diff_equal": cell["equal"],
            "probe_report_ref": direct_probe["report_ref"],
            "debugger_optional_operations": list(adapter.optional_operations()),
            "debugger_output": rendered,
            "export_root": exported["root"],
            "verification": verification,
            "blocked_import_roots": sorted(_UpstreamBlocker.blocked),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "qualification.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(receipt, sort_keys=True))
    finally:
        bench.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
