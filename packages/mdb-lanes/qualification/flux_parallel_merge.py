"""Recombine image-row residual/attention/MLP operands at a native parallel merge.

Use the original full output projection, preserving its shape and rounding.
This is an intervention recipe, not an independent prediction of a whole image.
"""
from __future__ import annotations


def merge_candidate(*, recipient_residual, donor_residual, recipient_operands, donor_operands,
                    gate, projection, attention_width, text_rows, mask):
    import torch
    from torch.nn import functional as F

    if len(mask) != 3 or any(bit not in "01" for bit in mask):
        raise ValueError("mask must select residual, attention, MLP with three binary digits")
    values = (recipient_residual, donor_residual, recipient_operands, donor_operands, gate)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 3 for value in values):
        raise ValueError("parallel merge requires rank-three tensors")
    if recipient_residual.shape != donor_residual.shape or recipient_operands.shape != donor_operands.shape:
        raise ValueError("matched donor/recipient geometry required")
    if recipient_residual.shape[:2] != recipient_operands.shape[:2] or gate.shape != (1, 1, recipient_residual.shape[-1]):
        raise ValueError("incompatible residual/operand/gate geometry")
    if not 0 < text_rows < recipient_residual.shape[1] or not 0 < attention_width < recipient_operands.shape[-1]:
        raise ValueError("invalid text/image rows or attention/MLP column split")
    if any(value.dtype != recipient_residual.dtype or value.device != recipient_residual.device for value in values):
        raise ValueError("mixed device/dtype operands are not qualified")
    residual, operands = recipient_residual.clone(), recipient_operands.clone()
    if mask[0] == "1":
        residual[:, text_rows:] = donor_residual[:, text_rows:]
    if mask[1] == "1":
        operands[:, text_rows:, :attention_width] = donor_operands[:, text_rows:, :attention_width]
    if mask[2] == "1":
        operands[:, text_rows:, attention_width:] = donor_operands[:, text_rows:, attention_width:]
    projected = F.linear(operands, projection.weight, projection.bias)
    return {"operands": operands, "projection": projected, "updated": residual + gate * projected}


def recipient_candidate(*, residual, operands, gate, projection, attention_width, text_rows, gains):
    """Scale only own image A/M operands; no donor tensors or donor lookup.

    gains=(A,M), each in [0,1]. Keep own residual, own text operands, full-shaped
    original F.linear and its bias. Returned full text rows therefore reconstruct
    the unchanged native text output. This is a local intervention, not an
    independent raw-prompt account or a semantic shape prediction.
    """
    import math

    import torch
    from torch.nn import functional as F

    if (not isinstance(gains, (tuple, list)) or len(gains) != 2
            or any(type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1 for value in gains)):
        raise ValueError("recipient gains must be two finite values in [0,1]")
    values = (residual, operands, gate)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 3 for value in values):
        raise ValueError("recipient merge requires rank-three tensors")
    if (residual.shape[:2] != operands.shape[:2] or gate.shape != (1, 1, residual.shape[-1])
            or type(text_rows) is not int or not 0 < text_rows < residual.shape[1]
            or type(attention_width) is not int or not 0 < attention_width < operands.shape[-1]):
        raise ValueError("invalid recipient residual/operand/gate geometry")
    weight, bias = projection.weight, projection.bias
    if weight.shape != (residual.shape[-1], operands.shape[-1]) or (bias is not None and bias.shape != (residual.shape[-1],)):
        raise ValueError("original recipient projection geometry differs")
    if any(value.dtype != residual.dtype or value.device != residual.device or not bool(torch.isfinite(value).all())
           for value in (*values, weight, *((bias,) if bias is not None else ()))):
        raise ValueError("recipient arithmetic profile or finite-value contract differs")
    selected = operands.clone()
    selected[:, text_rows:, :attention_width] = operands[:, text_rows:, :attention_width] * gains[0]
    selected[:, text_rows:, attention_width:] = operands[:, text_rows:, attention_width:] * gains[1]
    projected = F.linear(selected, weight, bias)
    return {"operands": selected, "projection": projected, "updated": residual + gate * projected}
