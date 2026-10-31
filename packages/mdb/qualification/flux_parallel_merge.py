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
