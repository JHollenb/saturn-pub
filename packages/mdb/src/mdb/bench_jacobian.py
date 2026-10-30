"""J-space sidecars over exact Qwen Bench cuts and admitted residual Acts."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

import torch

from .errors import BenchError
from .jacobian_io import FrozenJacobianLens, JacobianReader, JacobianWriter
from .jacobian_workspace import metadata_sha256, tensor_payload_identity
from .model_os_contracts import content_digest

PIN_SCHEMA = "mdb-bench-jacobian-pin-v1"
READ_SCHEMA = "mdb-bench-jacobian-read-v1"
CANDIDATE_SCHEMA = "mdb-bench-jacobian-candidate-v1"


def _header_hash(tensor):
    identity = tensor_payload_identity(tensor)
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return sha256(f"{identity['dtype']}|{tuple(identity['shape'])}|".encode() + raw).hexdigest()


def application_profile(bench):
    device = bench.vm.runtime.device.type
    dtype = str(next(bench.model.parameters()).dtype).removeprefix("torch.")
    backend = bench.vm.runtime.backend
    implementation = "eager" if backend == "native-eager" else f"virtual-{backend}"
    suffix = ""
    if device == "cuda" and dtype == "float32":
        suffix = "-tf32on" if torch.backends.cuda.matmul.allow_tf32 else "-tf32off"
    return {
        "application_lane": f"{device}-{dtype}-{implementation}{suffix}",
        "model_abi_sha256": bench.model_abi,
        "numerical_contract": bench.vm.package.model_abi.fingerprint,
        "bench_lane": bench.lane,
        "attention_backend": backend,
        "dtype": "torch." + dtype,
    }


def _model_identity(bench, manifest):
    policy = manifest["model_identity"].get("tokenizer_identity_policy", "backend-json-v1")
    tokenizer = bench.tokenizer
    if policy == "token-ids-only-v1":
        if tokenizer is not None:
            raise BenchError("token-ids-only lens requires a token-ID-only resident")
        tokenizer_hash = metadata_sha256(
            {"policy": policy, "vocab_size": bench.model.config.vocab_size}
        )
    elif policy == "vocabulary-v1":
        if tokenizer is None:
            raise BenchError("vocabulary lens requires a tokenizer")
        tokenizer_hash = metadata_sha256(tokenizer.get_vocab())
    elif policy == "backend-json-v1":
        if tokenizer is None or not hasattr(tokenizer, "backend_tokenizer"):
            raise BenchError("lens requires the fitted backend tokenizer contract")
        tokenizer_hash = metadata_sha256(
            {
                "backend_json": tokenizer.backend_tokenizer.to_str(),
                "special_tokens_map": {
                    key: str(value) for key, value in tokenizer.special_tokens_map.items()
                },
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "chat_template": getattr(tokenizer, "chat_template", None),
                "fit_add_special_tokens": True,
                "assay_add_special_tokens": False,
            }
        )
    else:
        raise BenchError("unsupported lens tokenizer identity policy")
    result = {
        "model_id": bench.identity["model_id"],
        "revision": bench.identity["revision"],
        "weight_sha256": bench.identity["weights_sha256"],
        "config_sha256": metadata_sha256(bench.model.config.to_dict()),
        "tokenizer_sha256": tokenizer_hash,
    }
    if "tokenizer_identity_policy" in manifest["model_identity"]:
        result["tokenizer_identity_policy"] = policy
    return result


def atomic_sidecar(bench, operation, *args):
    """Publish metadata and previews atomically without dropping an in-flight token."""
    if bench.publication.mode != "journal" or not bench.branches or bench._edit:
        raise BenchError("Jacobian operations require journal mode, a branch, and no pending Act")
    branch = bench.branch
    session = bench.vm._session(bench.active)
    previous = (
        branch.cut,
        branch.node,
        deepcopy(bench.cuts),
        deepcopy(bench.events),
        session.checkpoint_ref,
        bench.source_frames.current,
        getattr(bench, "_frozen_readers", None),
    )
    try:
        with bench.publication.atomic_selection():
            return operation(bench, *args)
    except BaseException:
        (branch.cut, branch.node, bench.cuts, bench.events,
         session.checkpoint_ref, frame, bench._frozen_readers) = previous
        bench.source_frames._checkout(frame)
        bench._edit = None
        bench._jacobian_pending = None
        raise


def _reader(bench, pin_ref):
    binding = getattr(bench, "_frozen_readers", None)
    if binding is not None:
        binding.guard()
        if pin_ref in binding.readers:
            binding.stats["cache_hits"] += 1
            reader, pin = binding.readers[pin_ref]
            if pin["profile"] != application_profile(bench):
                raise BenchError("frozen reader numerical profile changed")
            return reader, pin
    pin = bench.artifacts.get(pin_ref)
    if pin.get("schema") != PIN_SCHEMA or pin["profile"] != application_profile(bench):
        raise BenchError("Jacobian lens pin is incompatible with this resident")
    if pin["model_identity"] != bench.identity:
        raise BenchError("Jacobian lens checkpoint identity changed")
    lens = FrozenJacobianLens(pin["lens_manifest"], bench._tensor_load(pin["transport_ref"]))
    decoder = lens.manifest["decoder"]
    _decoder_convention(bench, decoder)
    if decoder.get("final_norm_weight_sha256") != _header_hash(bench.model.model.norm.weight):
        raise BenchError("Jacobian final-normalization identity changed")
    reader = JacobianReader(
        lens,
        bench.model.lm_head.weight.detach(),
        model_identity=_model_identity(bench, lens.manifest),
        application_lane=pin["profile"]["application_lane"],
        rms_weight=bench.model.model.norm.weight.detach(),
        final_norm=bench.model.model.norm,
        output_head=bench.model.lm_head,
        execution_binding={
            "model_abi_sha256": bench.model_abi,
            "numerical_contract": bench.vm.package.model_abi.fingerprint,
            "session_id": bench.session_id,
        },
        _frozen_guard=None if binding is None else binding.guard,
    )
    if binding is not None:
        binding.readers[pin_ref] = reader, pin
        binding.stats["reader_builds"] += 1
    return reader, pin


def _decoder_convention(bench, decoder):
    if decoder["final_norm"] not in {"model.norm", "model.norm RMSNorm"}:
        raise BenchError("Jacobian Qwen decoder requires the native model.norm RMSNorm")
    if decoder["output_head"] not in {"lm_head", "lm_head linear unembedding"}:
        raise BenchError("Jacobian Qwen decoder requires the native lm_head linear unembedding")
    if decoder.get("logit_softcap") is not None:
        raise BenchError("Jacobian Qwen decoder does not use logit softcapping")
    if "tied_embeddings" in decoder:
        tied = (
            bench.model.lm_head.weight.data_ptr()
            == bench.model.model.embed_tokens.weight.data_ptr()
        )
        if type(decoder["tied_embeddings"]) is not bool or decoder["tied_embeddings"] != tied:
            raise BenchError("Jacobian Qwen embedding tying convention changed")


def pin_lens(bench, manifest, transport):
    lens = FrozenJacobianLens(manifest, transport)
    profile = application_profile(bench)
    if str(transport.dtype) != profile["dtype"]:
        raise BenchError("Jacobian lens dtype differs from the native numerical lane")
    decoder = lens.manifest["decoder"]
    _decoder_convention(bench, decoder)
    if decoder.get("final_norm_weight_sha256") != _header_hash(bench.model.model.norm.weight):
        raise BenchError("lens must bind the resident final-normalization weight")
    if decoder.get("unembedding_weight_sha256") != _header_hash(bench.model.lm_head.weight):
        raise BenchError("lens must bind the resident output-head weight")
    if decoder["vocab_size"] != bench.model.config.vocab_size:
        raise BenchError("lens vocabulary differs from resident")
    JacobianReader(
        lens,
        bench.model.lm_head.weight.detach(),
        model_identity=_model_identity(bench, lens.manifest),
        application_lane=profile["application_lane"],
        rms_weight=bench.model.model.norm.weight.detach(),
        final_norm=bench.model.model.norm,
        output_head=bench.model.lm_head,
        execution_binding={
            "model_abi_sha256": bench.model_abi,
            "numerical_contract": bench.vm.package.model_abi.fingerprint,
            "session_id": bench.session_id,
        },
    )
    pin = {
        "schema": PIN_SCHEMA,
        "lens_manifest": lens.manifest,
        "transport_ref": bench._tensor_ref(lens.transport_for(bench.vm.runtime.device)),
        "profile": profile,
        "model_identity": bench.identity,
        "semantic_writer_qualified": False,
    }
    ref = bench.artifacts.put(pin)
    bench.events.append({"kind": "jacobian-lens-pinned", "pin_ref": ref})
    bench._publish()
    return {"lens_ref": ref, "lens_fingerprint": lens.manifest["fingerprint"], "profile": profile}


def _source(bench, reader):
    execution = bench._backend().execution
    expected_boundary = reader.lens.manifest["coordinate_map"]["completed_layer_boundary"]
    if (
        execution is None
        or execution.logits is not None
        or execution.next_layer != expected_boundary
    ):
        raise BenchError(
            "Jacobian source requires the lens's completed layer boundary before readout"
        )
    residual = bench.resolve_value("residual")
    source = {
        "model_abi_sha256": bench.model_abi,
        "cut_ref": bench.branch.cut,
        "branch": bench.active,
        "session_id": bench.session_id,
        "token_position": len(bench.branch.token_ids),
        "completed_layer_boundary": execution.next_layer,
        "activation": tensor_payload_identity(residual),
        "input_tokens_sha256": content_digest(bench.branch.token_ids),
        "intervention_lineage": list(bench.branch.intervention_lineage),
        "cursor": bench.cursor(),
        "numerical_contract": bench.vm.package.model_abi.fingerprint,
    }
    return residual, source


def read(bench, pin_ref, options):
    reader, pin = _reader(bench, pin_ref)
    _source(bench, reader)  # Refuse bad coordinates before capturing anything.
    bench.checkpoint("jacobian-source")
    residual, source = _source(bench, reader)
    with bench.metrics.measure("jacobian_read"):
        observation = reader.read(residual, source=source, **options)
    tensors = {
        name: bench._tensor_ref(getattr(observation, name))
        for name in (
            "transported",
            "selected_token_ids",
            "selected_raw_scores",
            "selected_normalized_scores",
        )
    }
    receipt = {
        "schema": READ_SCHEMA,
        "pin_ref": pin_ref,
        "source": source,
        "source_ref": bench._tensor_ref(residual),
        "observation": observation.metadata,
        "readout_refs": tensors,
        "options": options,
        "model_identity": pin["model_identity"],
        "native_mutation": False,
        "semantic_writer_qualified": False,
    }
    ref = bench.artifacts.put(receipt)
    bench.events.append(
        {"kind": "jacobian-read", "observation_ref": ref, "source_cut": source["cut_ref"]}
    )
    bench._publish()
    return {
        "observation_ref": ref,
        "source_cut": source["cut_ref"],
        "selected_token_ids": observation.selected_token_ids.cpu().tolist(),
        "selected_scores": observation.selected_normalized_scores.cpu().tolist(),
        "metadata": observation.metadata,
    }


def preview(bench, observation_ref, spec):
    observed = bench.artifacts.get(observation_ref)
    if observed.get("schema") != READ_SCHEMA:
        raise BenchError("writer requires a bound Jacobian read receipt")
    reader, _ = _reader(bench, observed["pin_ref"])
    residual, source = _source(bench, reader)
    if source != observed["source"] or not torch.equal(
        residual, bench._tensor_load(observed["source_ref"])
    ):
        raise BenchError("Jacobian read is stale for the current cut, branch, or residual")
    allowed = {
        "operation",
        "token_id",
        "token_ids",
        "target_coordinates",
        "dose",
        "consumer",
        "logical_address",
    }
    if set(spec) - allowed or spec.get("operation") not in {"steer", "span"}:
        raise BenchError("Jacobian writer requires an explicit steer or span recipe")
    writer = JacobianWriter(reader)
    shared = {
        "source": source,
        "dose": spec.get("dose", 1.0),
        "consumer": spec.get("consumer"),
        "logical_address": spec.get("logical_address"),
    }
    if spec["operation"] == "steer":
        if set(spec) & {"token_ids", "target_coordinates"}:
            raise BenchError("steer recipe uses one token_id")
        prepared = writer.prepare_steer(residual, token_id=spec.get("token_id"), **shared)
    else:
        if "token_id" in spec:
            raise BenchError("span recipe uses token_ids and target_coordinates")
        target = torch.as_tensor(
            spec.get("target_coordinates"), device=residual.device, dtype=torch.float32
        )
        prepared = writer.prepare_span(
            residual, token_ids=spec.get("token_ids"), target_coordinates=target, **shared
        )
    candidate = writer.validate(prepared, residual=residual, source=source)
    receipt = {
        "schema": CANDIDATE_SCHEMA,
        "attempt_id": uuid4().hex,
        "observation_ref": observation_ref,
        "pin_ref": observed["pin_ref"],
        "source": source,
        "write": prepared.metadata,
        "candidate_ref": bench._tensor_ref(candidate),
        "delta_ref": bench._tensor_ref(prepared.delta),
        "semantic_writer_qualified": False,
    }
    candidate_ref = bench.artifacts.put(receipt)
    native = bench.preview_edit("residual", candidate, mode="replace")
    bench._jacobian_pending = {"act_id": native["act_id"], "candidate_ref": candidate_ref}
    bench.events.append(
        {
            "kind": "jacobian-preview",
            "act_id": native["act_id"],
            "jacobian_candidate_ref": candidate_ref,
            "observation_ref": observation_ref,
            "parent_cut": native["parent_cut"],
        }
    )
    bench._publish()
    return {**native, "jacobian_candidate_ref": candidate_ref, "writer_metadata": prepared.metadata}


__all__ = ["application_profile", "atomic_sidecar", "pin_lens", "read", "preview"]
