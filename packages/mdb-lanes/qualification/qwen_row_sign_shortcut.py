"""Conditional per-query Qwen sign shortcut from positive R/G/U only.

Permitted-key support is a separate diagnostic. Every row is predicted even if
its allowed keys contain both signs. No negative target, attention, RMSNorm,
G/U projection or module forward is an input to the prediction formula.
"""
from __future__ import annotations

import hashlib
import runpy
from functools import lru_cache
from pathlib import Path

BASE_SHA256 = "8e59051a34bc8e89c9376fd29aec966df9cbe84560c81f86910504172caaa457"
SCHEMA = "mdb-qwen-row-sign-shortcut-v1"
PROFILE_SCHEMA = "mdb-qwen-row-sign-shortcut-profile-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


@lru_cache(maxsize=1)
def base():
    path = Path(__file__).with_name("qwen_sign_shortcut.py")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA256, "frozen primitive-profile helper differs")
    return runpy.run_path(str(path))


def mask_descriptor(value):
    import torch

    require(isinstance(value, torch.Tensor) and value.dtype == torch.bool and value.ndim == 2,
            "explicit dense bool permitted query-key mask required")
    raw = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device),
                content_sha256=hashlib.sha256(memoryview(raw).cast("B")).hexdigest())


def validate_signs(row_signs, permitted):
    import torch

    require(isinstance(row_signs, torch.Tensor) and row_signs.dtype == torch.bfloat16 and row_signs.ndim == 1 and
            row_signs.numel() > 0 and bool(((row_signs == 1) | (row_signs == -1)).all()), "one exact BF16 +/-1 sign per query required")
    descriptor = mask_descriptor(permitted)
    require(descriptor["shape"] == [row_signs.numel(), row_signs.numel()] and permitted.device == row_signs.device,
            "same-context self-attention sign/mask geometry or device differs")


def support_closed(row_signs, permitted):
    """All permitted keys have the query's sign; empty support is reported false.

    The caller authenticates this exact mask against the actual attention
    contract. This function does not inspect model attention or predict accuracy.
    """
    validate_signs(row_signs, permitted)
    disagreement = row_signs[:, None] != row_signs[None, :]
    return permitted.any(dim=-1) & ~(permitted & disagreement).any(dim=-1)


def _head_descriptor(positive_head, query_axis, row_signs, device):
    import torch

    if positive_head is None:
        require(query_axis is None, "head query axis requires an explicit positive head")
        return None
    require(isinstance(positive_head, torch.Tensor) and positive_head.ndim == 4 and query_axis in (1, 2) and
            positive_head.shape[0] == 1 and positive_head.shape[query_axis] == row_signs.numel() and
            positive_head.device == device and bool(torch.isfinite(positive_head).all()), "positive head/query axis geometry or finite contract differs")
    return base()["tensor_descriptor"](positive_head)


def make_profile(positive, down_weight, row_signs, permitted, *, source_identity, frontier_kind,
                 attention_profile, positive_head=None, head_query_axis=None):
    validate_signs(row_signs, permitted)
    ordinary = base()["make_profile"](positive, down_weight, source_identity=source_identity,
        frontier_kind=frontier_kind, attention_profile=attention_profile)
    return dict(schema=PROFILE_SCHEMA, positive_profile=ordinary,
        row_signs=base()["tensor_descriptor"](row_signs), permitted_mask=mask_descriptor(permitted),
        positive_head=_head_descriptor(positive_head, head_query_axis, row_signs, down_weight.device), head_query_axis=head_query_axis,
        sign_scope="Per-query sign across every feature; same row signs for Q/K/V source states",
        permitted_scope="Caller-authenticated actual allowed query-key set, not inferred from a token mask alone",
        premises="Attention-prefix oddness and finite rounding are measured after targets; support_closed is diagnostic, never a row selector")


def validate_inputs(positive, down_weight, row_signs, permitted, profile, *, positive_head=None, head_query_axis=None):
    validate_signs(row_signs, permitted)
    require(isinstance(profile, dict) and set(profile) == {"schema", "positive_profile", "row_signs", "permitted_mask", "positive_head", "head_query_axis", "sign_scope", "permitted_scope", "premises"}
            and profile["schema"] == PROFILE_SCHEMA, "row finite profile schema differs")
    ordinary = base()["validate_inputs"](positive, down_weight, profile["positive_profile"])
    require(positive["R"].shape[1] == row_signs.numel() and down_weight.device == row_signs.device,
            "complete positive rows/device differ from row signs")
    require(base()["tensor_descriptor"](row_signs) == profile["row_signs"] and mask_descriptor(permitted) == profile["permitted_mask"],
            "bound row signs/permitted-mask identity or layout changed")
    require(head_query_axis == profile["head_query_axis"] and
            _head_descriptor(positive_head, head_query_axis, row_signs, down_weight.device) == profile["positive_head"], "bound positive head or query axis changed")
    require(profile["sign_scope"] == "Per-query sign across every feature; same row signs for Q/K/V source states" and
            profile["permitted_scope"] == "Caller-authenticated actual allowed query-key set, not inferred from a token mask alone" and
            profile["premises"] == "Attention-prefix oddness and finite rounding are measured after targets; support_closed is diagnostic, never a row selector", "declared row scope changed")
    return ordinary


def predict_row_sign(positive, down_weight, row_signs, permitted, *, profile, positive_head=None, head_query_axis=None):
    """Compute every signed row with the source's BF16 primitive ordering.

    Optional headhat is only a signed head diagnostic and never supplies R/G/U
    or the MLP prediction. Native head B,H,T,D needs axis2; functional B,T,H,D
    needs axis1. The helper does not silently transpose either layout.
    """
    import torch
    from torch.nn import functional as F

    before = validate_inputs(positive, down_weight, row_signs, permitted, profile,
                             positive_head=positive_head, head_query_axis=head_query_axis)
    with torch.inference_mode(), base()["no_module_forwards"]():
        signed = row_signs.view(1, -1, 1)
        rhat, ghat, uhat = (signed * positive[name] for name in ("R", "gate", "up"))
        activation = F.silu(ghat)
        product = activation * uhat
        mlp = F.linear(product, down_weight, bias=None)
        output = rhat + mlp
        ports = dict(Rhat=rhat, Ghat=ghat, Uhat=uhat, activation=activation, product=product, M=mlp, Hnext=output)
        if positive_head is not None:
            shape = [1] * 4
            shape[head_query_axis] = row_signs.numel()
            ports["headhat"] = row_signs.view(shape) * positive_head
        closed = support_closed(row_signs, permitted)
    require(all(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()) for value in ports.values()), "forecast dtype/finite mechanics differ")
    after = validate_inputs(positive, down_weight, row_signs, permitted, profile,
                            positive_head=positive_head, head_query_axis=head_query_axis)
    require(before == after, "positive learned operands changed during row shortcut")
    return dict(schema=SCHEMA, ports=ports, support_closed=closed,
        support=dict(closed_rows=int(closed.sum()), mixed_or_empty_rows=int((~closed).sum()), total_rows=closed.numel(),
            empty_rows=int((~permitted.any(dim=-1)).sum()), use="diagnostic only; all complete output rows retained"),
        profile=profile, output_descriptors={name: base()["tensor_descriptor"](value) for name, value in ports.items()},
        counts=dict(row_sign_multiplications=3 + int(positive_head is not None), SiLU=1, elementwise_products=1,
            linear=1, residual_additions=1, attention=0, RMSNorm=0, gate_projection=0, up_projection=0, module_forwards=0),
        comparison_max_elements=base()["comparison_element_budget"]({name: list(ports[name].shape) for name in ("product", "M", "Hnext")}),
        formula="Rhat=s*R;Ghat=s*G;Uhat=s*U;P=SiLU(Ghat)*Uhat;M=linear(P,Wd,None);Y=Rhat+M",
        status="computed on every query; numerical/bit/sign-zero relations to targets remain observations",
        limitations=["The only learned prediction primitive is the original down projection; no attention or target feedback.",
            "Support closure is a real-arithmetic sufficient sign condition, not a finite-precision certificate or accuracy gate.",
            "Mixed-support rows are predicted and retained even when the shortcut premise does not hold.",
            "Named primitives/flags do not identify selected kernels or an independent implementation."])
