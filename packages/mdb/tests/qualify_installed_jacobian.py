"""Two-process installed-wheel qualification for bounded J-space mechanics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import sys
from pathlib import Path


class _UpstreamBlocker(importlib.abc.MetaPathFinder):
    blocked = {"saturn", "saturn_pub", "common", "common_harness", "mrun"}

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname.split(".", 1)[0] in self.blocked:
            raise ImportError(f"qualification blocked upstream import: {fullname}")
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_runtime(target: Path):
    sys.path.insert(0, str(target))
    sys.meta_path.insert(0, _UpstreamBlocker())

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import mdb

    mdb_file = Path(mdb.__file__).resolve()
    if not mdb_file.is_relative_to(target):
        raise AssertionError(f"MDB did not import from installed target: {mdb_file}")
    forbidden = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in _UpstreamBlocker.blocked
    )
    if forbidden:
        raise AssertionError(f"upstream modules imported: {forbidden}")
    return torch, Qwen2Config, Qwen2ForCausalLM, mdb_file


def _resident(torch, Qwen2Config, Qwen2ForCausalLM):
    torch.manual_seed(91)
    return (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=64,
                attn_implementation="eager",
            )
        )
        .eval()
        .to(device="cpu", dtype=torch.bfloat16)
    )


def _lens(bench, torch):
    from mdb.bench_jacobian import _header_hash, _model_identity, application_profile
    from mdb.jacobian_workspace import metadata_sha256, tensor_payload_identity

    matrix = torch.eye(16, dtype=torch.bfloat16)
    profile = application_profile(bench)
    body = {
        "schema": "saturn-jacobian-lens-v1",
        "model_identity": {"tokenizer_identity_policy": "token-ids-only-v1"},
        "execution_compatibility": {
            "fitting_lane": profile["application_lane"],
            "permitted_application_lanes": [profile["application_lane"]],
        },
        "coordinate_map": {
            "residual_width": 16,
            "source_layer": 0,
            "completed_layer_boundary": 1,
            "hook_convention": "decoder-block-output",
            "source_position_semantics": "absolute source token",
        },
        "transport": {
            "frozen": True,
            "shape": [16, 16],
            "dtype": str(matrix.dtype),
            "sha256": tensor_payload_identity(matrix)["sha256"],
        },
        "decoder": {
            "residual_width": 16,
            "vocab_size": 32,
            "final_norm": "model.norm",
            "output_head": "lm_head",
            "final_norm_weight_sha256": _header_hash(bench.model.model.norm.weight),
            "unembedding_weight_sha256": _header_hash(bench.model.lm_head.weight),
        },
        "estimator": {"name": "target-sum/source-mean"},
        "dictionary": {
            "atom_derivation": "decoder-weight-times-transport",
            "norm_scale_convention": "unit-l2-after-transport",
            "sparse_interpretation": "unperformed",
        },
    }
    body["model_identity"] = _model_identity(bench, body)
    return {**body, "fingerprint": metadata_sha256(body)}, matrix


def _standalone_reader(bench, lens, source):
    from mdb.bench_jacobian import _model_identity, application_profile
    from mdb.jacobian_io import JacobianReader

    profile = application_profile(bench)
    return JacobianReader(
        lens,
        bench.model.lm_head.weight.detach(),
        model_identity=_model_identity(bench, lens.manifest),
        application_lane=profile["application_lane"],
        rms_weight=bench.model.model.norm.weight.detach(),
        final_norm=bench.model.model.norm,
        output_head=bench.model.lm_head,
        execution_binding={
            "model_abi_sha256": source["model_abi_sha256"],
            "numerical_contract": source["numerical_contract"],
            "session_id": source["session_id"],
        },
    )


def _process_record(mdb_file: Path, target: Path) -> dict:
    return {
        "pid": os.getpid(),
        "python": sys.executable,
        "mdb_file": str(mdb_file),
        "installed_target": str(target),
        "import_root_verified": str(mdb_file).startswith(str(target) + os.sep),
        "blocked_import_roots": sorted(_UpstreamBlocker.blocked),
    }


def _capture(args, torch, Qwen2Config, Qwen2ForCausalLM, mdb_file: Path) -> dict:
    from mdb.bench import BenchSession
    from mdb.bench_jacobian import application_profile
    from mdb.jacobian_io import FrozenJacobianLens, JacobianWriter, PreparedJWrite

    model = _resident(torch, Qwen2Config, Qwen2ForCausalLM)
    session_root = args.output / "capture-session"
    export_root = args.output / "export"
    bench = BenchSession.from_resident(
        model, root=session_root, lane="native", continuation_capacity=8
    )
    try:
        bench.prefill([1, 2, 3])
        manifest, transport = _lens(bench, torch)
        if application_profile(bench)["application_lane"] != "cpu-bfloat16-eager":
            raise AssertionError(f"unexpected application lane: {application_profile(bench)}")
        pin = bench.pin_jacobian_lens(manifest, transport)
        bench.step("layer")
        read = bench.read_jacobian(
            pin["lens_ref"],
            token_subset=[3, 7, 11],
            top_k=2,
            logical_address={"symbol": "fixture.animal", "instance": "one"},
        )
        source_residual = bench.resolve_value("residual")
        read_artifact = bench.artifacts.get(read["observation_ref"])
        selected = bench._tensor_load(
            read_artifact["readout_refs"]["selected_normalized_scores"]
        )
        selected_ids = bench._tensor_load(read_artifact["readout_refs"]["selected_token_ids"])
        with torch.inference_mode():
            native_scores = model.lm_head(model.model.norm(source_residual))
        if not torch.equal(selected, native_scores.gather(-1, selected_ids)):
            raise AssertionError("identity-lens read differs from the native normalized readout")

        standalone_root = args.output / "standalone"
        lens_root = standalone_root / "lens"
        FrozenJacobianLens(manifest, transport).save(lens_root)
        reopened_lens = FrozenJacobianLens.open(lens_root)
        reader = _standalone_reader(bench, reopened_lens, read_artifact["source"])
        standalone_observation = reader.read(
            source_residual,
            source=read_artifact["source"],
            token_subset=[3, 7, 11],
            top_k=2,
            logical_address={"symbol": "fixture.animal", "instance": "one"},
        )
        if not torch.equal(standalone_observation.selected_normalized_scores, selected):
            raise AssertionError("standalone reopened lens differs from Bench read")
        writer = JacobianWriter(reader)
        consumer = {"name": "fixture-later-query", "horizon": 2}
        steer = writer.prepare_steer(
            source_residual,
            source=read_artifact["source"],
            token_id=7,
            dose=0.5,
            consumer=consumer,
        )
        steer_root = standalone_root / "steer-candidate"
        steer.save(steer_root)
        reopened_steer = PreparedJWrite.open(steer_root)
        if not torch.equal(
            writer.validate(
                reopened_steer, residual=source_residual, source=read_artifact["source"]
            ),
            steer.candidate,
        ):
            raise AssertionError("reopened standalone steer candidate changed")
        span = writer.prepare_span(
            source_residual,
            source=read_artifact["source"],
            token_ids=[3, 7],
            target_coordinates=[0.25, -0.5],
            dose=1.0,
            consumer=consumer,
        )
        span_root = standalone_root / "span-candidate"
        span.save(span_root)
        reopened_span = PreparedJWrite.open(span_root)
        if not torch.equal(
            writer.validate(
                reopened_span, residual=source_residual, source=read_artifact["source"]
            ),
            span.candidate,
        ):
            raise AssertionError("reopened standalone span candidate changed")

        recipe = {
            "operation": "steer",
            "token_id": 7,
            "dose": 0.5,
            "consumer": consumer,
        }
        preview = bench.preview_jacobian_edit(read["observation_ref"], recipe)
        if not torch.equal(bench.resolve_value("residual"), source_residual):
            raise AssertionError("Bench preview mutated the native residual")
        candidate = bench._tensor_load(
            bench.artifacts.get(preview["jacobian_candidate_ref"])["candidate_ref"]
        )
        committed = bench.commit_edit(preview["act_id"])
        if not torch.equal(bench.resolve_value("residual"), candidate):
            raise AssertionError("Bench commit did not install the exact prepared candidate")
        bench.restore(read["source_cut"])
        if not torch.equal(bench.resolve_value("residual"), source_residual):
            raise AssertionError("source-cut restore was not exact")
        zero = bench.preview_jacobian_edit(read["observation_ref"], {**recipe, "dose": 0.0})
        if not zero["writer_metadata"]["recipe"]["noop_exact"]:
            raise AssertionError("zero-dose preview did not declare an exact no-op")
        bench.abort_edit(zero["act_id"])
        if not torch.equal(bench.resolve_value("residual"), source_residual):
            raise AssertionError("zero-dose abort did not restore the exact source")
        exported = bench.export(export_root)
        handles = {
            "schema": "mdb-installed-jacobian-handles-v1",
            "wheel_sha256": args.wheel_sha256,
            "processes": {"capture": _process_record(mdb_file, args.target)},
            "application_lane": application_profile(bench)["application_lane"],
            "source_cut": read["source_cut"],
            "observation_ref": read["observation_ref"],
            "lens_ref": pin["lens_ref"],
            "lens_fingerprint": pin["lens_fingerprint"],
            "committed_candidate_ref": committed["jacobian_candidate_ref"],
            "committed_cut": committed["cut_id"],
            "source_residual_sha256": read_artifact["source"]["activation"]["sha256"],
            "standalone": {
                "lens_manifest_sha256": _sha256(lens_root / "manifest.json"),
                "steer_manifest_sha256": _sha256(steer_root / "manifest.json"),
                "span_manifest_sha256": _sha256(span_root / "manifest.json"),
                "native_decoder_view_hash_bound": True,
                "execution_binding": reader.execution_binding,
                "save_open_validate": ["lens", "steer", "span"],
            },
            "export": exported,
        }
        _write_json(args.handles, handles)
        return handles
    finally:
        bench.close()


def _reopen(args, torch, Qwen2Config, Qwen2ForCausalLM, mdb_file: Path) -> dict:
    from mdb.bench import BenchSession
    from mdb.bench_verify import verify
    from mdb.errors import BenchError
    from mdb.jacobian_workspace import tensor_payload_identity

    handles = json.loads(args.handles.read_text(encoding="utf-8"))
    if handles["wheel_sha256"] != args.wheel_sha256:
        raise AssertionError("capture and reopen wheel hashes differ")
    model = _resident(torch, Qwen2Config, Qwen2ForCausalLM)
    export_root = Path(handles["export"]["root"])
    bench = BenchSession.resume(model, root=export_root)
    stale_rejected = False
    try:
        bench.restore(handles["source_cut"])
        source_residual = bench.resolve_value("residual")
        if tensor_payload_identity(source_residual)["sha256"] != handles["source_residual_sha256"]:
            raise AssertionError("fresh-process source restore changed residual bytes")
        preview = bench.preview_jacobian_edit(
            handles["observation_ref"],
            {
                "operation": "span",
                "token_ids": [3, 7],
                "target_coordinates": [0.0, 0.0],
                "dose": 0.0,
                "consumer": {"name": "fixture-query", "horizon": 1},
            },
        )
        if not preview["writer_metadata"]["recipe"]["noop_exact"]:
            raise AssertionError("fresh-process zero span is not an exact no-op")
        committed = bench.commit_edit(preview["act_id"])
        if not torch.equal(bench.resolve_value("residual"), source_residual):
            raise AssertionError("fresh-process zero-span commit changed native residual bytes")
        try:
            bench.preview_jacobian_edit(
                handles["observation_ref"],
                {
                    "operation": "steer",
                    "token_id": 7,
                    "dose": 0.0,
                    "consumer": {"name": "fixture-query", "horizon": 1},
                },
            )
        except BenchError as exc:
            if "stale" not in str(exc):
                raise
            stale_rejected = True
        if not stale_rejected:
            raise AssertionError("old source-bound observation was accepted after a new cut")
    finally:
        bench.close()

    audit = verify(export_root)
    if audit["mechanics_status"] != "valid" or audit["counts"].get(
        "jacobian_artifacts", 0
    ) < 5:
        raise AssertionError(f"Jacobian sidecar verification was incomplete: {audit}")
    report = {
        "schema": "mdb-installed-jacobian-qualification-v1",
        "mechanics_status": "valid",
        "instrument_status": "valid",
        "trend_status": "not-applicable",
        "terminal_status": "installed-wheel-mechanics-qualified",
        "semantic_writer_status": "unqualified",
        "claim_boundary": (
            "tiny seeded CPU BF16 identity-lens mechanics only; no learned semantic lens, "
            "useful writer, real-model behavior, or cross-lane portability claim"
        ),
        "wheel_sha256": args.wheel_sha256,
        "installed_target": str(args.target),
        "application_lane": handles["application_lane"],
        "processes": {
            **handles["processes"],
            "reopen": _process_record(mdb_file, args.target),
        },
        "fresh_process": (
            handles["processes"]["capture"]["pid"] != os.getpid()
        ),
        "source_cut": handles["source_cut"],
        "reopen_commit_cut": committed["cut_id"],
        "fresh_restore_exact": True,
        "fresh_zero_span_commit_exact": True,
        "old_observation_stale_after_new_cut": stale_rejected,
        "standalone": handles["standalone"],
        "verification": audit,
        "selection_note": (
            "The old source cut is restored explicitly; the selected cut may advance during "
            "export, commit, and close and is not required to remain the old source cut."
        ),
        "blocked_import_roots": sorted(_UpstreamBlocker.blocked),
    }
    _write_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capture", "reopen"), required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handles", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    args = parser.parse_args()
    args.target = args.target.resolve()
    args.output = args.output.resolve()
    args.handles = args.handles.resolve()
    args.report = args.report.resolve()
    torch, Qwen2Config, Qwen2ForCausalLM, mdb_file = _load_runtime(args.target)
    result = (
        _capture(args, torch, Qwen2Config, Qwen2ForCausalLM, mdb_file)
        if args.mode == "capture"
        else _reopen(args, torch, Qwen2Config, Qwen2ForCausalLM, mdb_file)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
