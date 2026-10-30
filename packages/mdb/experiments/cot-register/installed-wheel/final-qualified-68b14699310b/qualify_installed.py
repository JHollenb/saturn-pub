"""Fresh-process installed-wheel mechanics on an untrained, tiny Qwen fixture."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import sys
from pathlib import Path


class NoUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"saturn", "saturn_pub", "common", "common_harness", "mrun"}:
            raise ImportError("installed qualification attempted upstream import: " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "resume"))
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--optimized", action="store_true")
    args = parser.parse_args()
    sys.meta_path.insert(0, NoUpstream())

    import torch

    import mdb
    from mdb.bench import BenchSession
    from mdb.bench_jacobian import _header_hash, _model_identity, application_profile
    from mdb.bench_verify import verify
    from mdb.jacobian_workspace import metadata_sha256, tensor_payload_identity
    from mdb.local_debugger import tiny_qwen

    package = Path(mdb.__file__).resolve()
    if not package.is_relative_to(args.installed.resolve()):
        raise RuntimeError("qualification did not import the independently installed wheel")
    model = tiny_qwen(seed=91)
    if args.optimized:
        model.requires_grad_(False)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "capture":
        bench = BenchSession.from_resident(model, root=root / "source", lane="native",
                                          model_id="cot-register-mechanics-fixture", continuation_capacity=8)
        try:
            bench.prefill([1, 2, 3])
            refs = []
            for boundary in (1, 2):
                matrix = torch.eye(16, dtype=torch.bfloat16)
                profile = application_profile(bench)
                manifest = {
                    "schema": "saturn-jacobian-lens-v1",
                    "model_identity": {"tokenizer_identity_policy": "token-ids-only-v1"},
                    "execution_compatibility": {"fitting_lane": profile["application_lane"],
                        "permitted_application_lanes": [profile["application_lane"]]},
                    "coordinate_map": {"residual_width": 16, "source_layer": boundary-1,
                        "completed_layer_boundary": boundary, "hook_convention": "decoder-block-output",
                        "source_position_semantics": "absolute source token"},
                    "transport": {"frozen": True, "shape": [16, 16], "dtype": str(matrix.dtype),
                        "sha256": tensor_payload_identity(matrix)["sha256"]},
                    "decoder": {"residual_width": 16, "vocab_size": 32, "final_norm": "model.norm",
                        "output_head": "lm_head", "final_norm_weight_sha256": _header_hash(bench.model.model.norm.weight),
                        "unembedding_weight_sha256": _header_hash(bench.model.lm_head.weight)},
                    "estimator": {"name": "target-sum/source-mean"},
                    "dictionary": {"atom_derivation": "decoder-weight-times-transport",
                        "norm_scale_convention": "unit-l2-after-transport", "sparse_interpretation": "unperformed"},
                }
                manifest["model_identity"] = _model_identity(bench, manifest)
                manifest["fingerprint"] = metadata_sha256(manifest)
                refs.append(bench.pin_jacobian_lens(manifest, matrix)["lens_ref"])
            bench.configure_cot_register(refs, top_k=3, optimized=args.optimized)
            if args.optimized:
                bench.fork("unused")
                bench.checkout("main")
            bench.step("layer")
            register = bench.cot_register()
            if register["entry_count"] != 1 or not register["current_boundary_observed"]:
                raise RuntimeError("automatic observation did not run at the native layer boundary")
            source_digest = bench.inspect("residual")["sha256"]
            bench.export(root / "export")
            pointer = bench.cot_pointer(root / "pointer") if args.optimized else None
            report = {"mode": args.mode, "installed_package": str(package),
                      "source_residual_sha256": source_digest,
                      "register_entries": register["entry_count"], "mechanics_status": "valid",
                      "semantic_cot_qualified": False, "specimen": "untrained tiny CPU BF16 Qwen",
                      "optimized": args.optimized, "pointer": pointer}
            (root / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            bench.close()
    else:
        captured = json.loads((root / "capture.json").read_text())
        hydration = None
        if args.optimized:
            from mdb.bench_pointer import BenchPointer
            pointer = BenchPointer.open(captured["pointer"]["path"])
            browse = pointer.browse()
            plan = pointer.native_closure()
            if plan["hydration_audit"]["read_attempts"] != 0:
                raise RuntimeError("metadata/native planning unexpectedly hydrated tensors")
            hydrated = pointer.residual()
            hydration = {"browse_audit": browse["hydration_audit"], "closure_bytes": plan["payload_bytes"],
                         "residual_bytes": len(hydrated.payload), "audit": pointer.payloads.audit_summary()}
            bench = pointer.resume(model)
            if set(bench.vm._sessions) != {"main"}:
                raise RuntimeError("pointer resume hydrated an unselected branch")
            bench.configure_cot_register(top_k=3, optimized=True)
        else:
            bench = BenchSession.resume(model, root=root / "export")
        plain = BenchSession.from_resident(model, root=root / "native-reference", lane="native",
                                          model_id="cot-register-mechanics-fixture", continuation_capacity=8)
        try:
            restored = bench.inspect("residual")["sha256"] == captured["source_residual_sha256"]
            plain.prefill([1, 2, 3])
            native, observed = plain.step("token"), bench.step("token")
            equal = native["logits_sha256"] == observed["logits_sha256"]
            kv_equal = all(torch.equal(plain.resolve_value(f"kv_cache.layer_{layer}.{component}.private"),
                                      bench.resolve_value(f"kv_cache.layer_{layer}.{component}.private"))
                           for layer in range(2) for component in ("key", "value"))
            register = bench.cot_register()
            if not restored or not equal or not kv_equal or register["entry_count"] != 2 or register.get("panel_error"):
                raise RuntimeError("installed fresh-process register/native closure differs")
            audit = verify(bench.root)
            report = {"mode": args.mode, "installed_package": str(package),
                      "source_residual_restored_exact": restored, "native_logits_equal": equal,
                      "native_kv_equal": kv_equal, "register_entries": register["entry_count"],
                      "cot_artifacts": audit["counts"]["cot_artifacts"], "mechanics_status": "valid",
                      "semantic_cot_qualified": False, "specimen": "untrained tiny CPU BF16 Qwen",
                      "panel": str(bench.root / "cot-register.html"), "optimized": args.optimized,
                      "hydration": hydration, "performance": bench.inspect("performance")}
            (root / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            plain.close()
            bench.close()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
