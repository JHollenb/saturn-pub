"""Conditional first-block sign shortcut in the original BF16 primitive order.

Only positive R/gate/up and the original down weight are numerical inputs. The
caller binds their source/frontier and measures finite oddness separately. This
helper neither assumes bitwise oddness as a gate nor reads negative targets.
It executes no attention, normalization, G/U projection or module forward.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import contextmanager

SCHEMA = "mdb-qwen-sign-shortcut-v1"
PROFILE_SCHEMA = "mdb-qwen-sign-shortcut-profile-v1"
INPUTS = {"R", "gate", "up"}
FRONTIERS = {"observed-positive-ports", "own-functional-positive-ports"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tensor_descriptor(value):
    """Exact BF16 content bytes plus live layout, without persisting a tensor."""
    import torch

    require(isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16 and value.layout == torch.strided,
            "operand must be a strided BF16 tensor")
    require(sys.byteorder == "little", "BF16 byte descriptor requires little-endian host")
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device),
                bf16_le_sha256=hashlib.sha256(memoryview(raw).cast("B")).hexdigest())


def execution_profile():
    """Observe policy; never change flags or claim the selected GEMM kernel."""
    import torch

    return dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cuda_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cuda_allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def comparison_element_budget(shapes):
    """Exact named-field budget for callers of compare_equations; no allocation."""
    require(isinstance(shapes, dict) and shapes, "comparison shapes must be a nonempty named map")
    total = 0
    for name, shape in shapes.items():
        require(isinstance(name, str) and name and isinstance(shape, (list, tuple)) and shape and
                all(type(size) is int and size > 0 for size in shape), "invalid comparison shape")
        total += math.prod(shape)
    return total


def make_profile(positive, down_weight, *, source_identity, frontier_kind, attention_profile):
    """Capture a caller-declared source and actual positive operand identities.

    Source/checkpoint authenticity and context sameness are caller obligations.
    attention_profile describes the producing Qwen call; no attention runs here.
    """
    require(isinstance(positive, dict) and set(positive) == INPUTS, "exact positive ports R/gate/up required")
    return dict(schema=PROFILE_SCHEMA, source_identity=source_identity, frontier_kind=frontier_kind,
        sign_scope="all-query/all-feature", assumptions=dict(bias=False, dropout_p=0., use_cache=False,
            plus_minus_same_context=True, finite_oddness="unestablished; measure R/G/U separately, including signed zero"),
        attention_profile=attention_profile, execution=execution_profile(),
        operands={**{name: tensor_descriptor(value) for name, value in positive.items()},
                  "down_weight": tensor_descriptor(down_weight)})


def validate_inputs(positive, down_weight, profile):
    import torch

    require(isinstance(positive, dict) and set(positive) == INPUTS, "exact positive ports R/gate/up required")
    require(isinstance(profile, dict) and set(profile) == {"schema", "source_identity", "frontier_kind", "sign_scope",
                "assumptions", "attention_profile", "execution", "operands"}, "finite profile fields differ")
    try:
        json.dumps(profile, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("profile must contain finite JSON metadata only") from error
    require(profile["schema"] == PROFILE_SCHEMA and profile["frontier_kind"] in FRONTIERS and
            profile["sign_scope"] == "all-query/all-feature", "unsupported frontier/sign scope")
    sources = profile["source_identity"]
    require(isinstance(sources, dict) and sources and all(isinstance(name, str) and name and isinstance(sha, str)
            and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha) for name, sha in sources.items()),
            "explicit source SHA256 identities required")
    require(profile["assumptions"] == dict(bias=False, dropout_p=0., use_cache=False, plus_minus_same_context=True,
            finite_oddness="unestablished; measure R/G/U separately, including signed zero"), "source assumptions differ")
    attention = profile["attention_profile"]
    require(isinstance(attention, dict) and set(attention) == {"backend", "gqa_policy", "query_rows", "mask_identity",
             "positions_identity", "rope_identity"} and attention["backend"] == "sdpa" and
            attention["query_rows"] in ("partially-padded", "all-attended") and
            attention["gqa_policy"] == ("repeat_kv" if attention["query_rows"] == "partially-padded" else "enable_gqa"),
            "unsupported source attention metadata")
    require(all(isinstance(attention[key], str) and len(attention[key]) == 64 and
                all(c in "0123456789abcdef" for c in attention[key])
                for key in ("mask_identity", "positions_identity", "rope_identity")), "exact source context identities required")
    require(profile["execution"] == execution_profile(), "actual primitive policy differs from finite profile")
    values = {**positive, "down_weight": down_weight}
    descriptors = {name: tensor_descriptor(value) for name, value in values.items()}
    require(descriptors == profile["operands"], "operand content/dtype/device/layout differs from bound profile")
    require(all(bool(torch.isfinite(value).all()) for value in values.values()), "nonfinite input operand")
    r, gate, up = (positive[name] for name in ("R", "gate", "up"))
    require(r.ndim == gate.ndim == up.ndim == 3 and r.shape[0] == 1 and r.shape[:2] == gate.shape[:2] and
            gate.shape == up.shape and all(size > 0 for size in r.shape + gate.shape), "positive port shapes differ")
    require(down_weight.ndim == 2 and tuple(down_weight.shape) == (r.shape[-1], gate.shape[-1]), "down weight shape differs")
    require(len({str(value.device) for value in values.values()}) == 1 and r.device.type in ("cpu", "cuda"), "mixed/unsupported operand devices")
    return descriptors


@contextmanager
def no_module_forwards():
    import torch

    def reject(module, args):
        raise RuntimeError("native module forward forbidden in sign shortcut: " + type(module).__name__)

    handle = torch.nn.modules.module.register_module_forward_pre_hook(reject)
    try:
        yield
    finally:
        handle.remove()


def predict_negative(positive, down_weight, *, profile):
    """Return the negative tail without inspecting a negative target/premise.

    This is conditional on finite upstream oddness, which is an external
    observation, not a helper gate. Guarding module forwards does not profile
    the selected kernels. Full source/checkpoint custody belongs to the caller.
    """
    import torch
    from torch.nn import functional as F

    before = validate_inputs(positive, down_weight, profile)
    with torch.inference_mode(), no_module_forwards():
        negative_gate = torch.neg(positive["gate"])
        negative_up = torch.neg(positive["up"])
        activation = F.silu(negative_gate)
        product = activation * negative_up
        mlp = F.linear(product, down_weight, bias=None)
        negative_r = torch.neg(positive["R"])
        output = negative_r + mlp
    ports = dict(negative_gate=negative_gate, negative_up=negative_up, activation=activation,
                 product=product, M=mlp, negative_R=negative_r, Hnext=output)
    require(all(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()) for value in ports.values()),
            "output dtype/finite mechanics differ")
    after = {**{name: tensor_descriptor(value) for name, value in positive.items()}, "down_weight": tensor_descriptor(down_weight)}
    require(after == before and profile["execution"] == execution_profile(), "input/weight/policy changed during shortcut")
    descriptors = {name: tensor_descriptor(value) for name, value in ports.items()}
    return dict(schema=SCHEMA, ports=ports, input_descriptors=before, output_descriptors=descriptors,
        profile=json.loads(json.dumps(profile, sort_keys=True, allow_nan=False)),
        counts=dict(negations=3, SiLU=1, elementwise_products=1, linear=1, residual_additions=1,
                    attention=0, RMSNorm=0, gate_projection=0, up_projection=0, module_forwards=0),
        census_scope="Prediction-formula Python primitives only; validation/copy/hash operations excluded; no selected-kernel profiler claim",
        comparison_max_elements=comparison_element_budget({name: descriptors[name]["shape"] for name in ("product", "M", "Hnext")}),
        formula="Gm=neg(Gp);Um=neg(Up);Pm=SiLU(Gm)*Um;Mm=linear(Pm,Wd,None);Ym=neg(Rp)+Mm",
        status="computed; numerical relation to a negative target remains untested here",
        limitations=["No negative target or native module was an input.",
                     "No -Mplus+Wd(GU) regrouping or extra learned RHS projection.",
                     "Finite R/G/U oddness and zero-sign effects must be measured separately; misses do not reject this computation.",
                     "Observed-positive ports imply a conditional frontier, not a held-out raw-input prediction."])
