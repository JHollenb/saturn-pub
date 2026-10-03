"""Fail-closed certificates for bounded autoregressive induction circuits.

A researcher freezes a :class:`InductionCircuitPolicy` (candidate circuit family,
control arms, held-out panel thresholds, native-consumer contract) *in advance*. A
worker supplies the measured per-example outcomes of a controlled relational-induction
workload. :func:`certify_induction_route` then returns a signed (content-hashed)
certificate, or -- the instant a gate fails -- the same object with the failing gate
named and ``certified: False``. Nothing is tuned after the outcomes arrive.

Layers:

* :mod:`~saturn_pub.certificate.gate` -- the stdlib-only decision rule (imports no model
  framework and nothing else from ``saturn_pub``); it only reads outcome vectors.
* :mod:`~saturn_pub.certificate.workload` -- the stdlib-only arbitrary-token
  relational-induction assay builder.
* :mod:`~saturn_pub.certificate.panel` -- the measurement helper that runs the causal
  battery on a native ``saturn_pub`` decoder adapter (imports torch on demand).
* :mod:`~saturn_pub.certificate.register` -- wraps a certificate as a
  :class:`saturn_pub.core.Receipt` and registers it in a
  :class:`saturn_pub.evidence.ClaimsRegistry`.

``panel`` is imported lazily so importing this package never imports PyTorch.
"""

from __future__ import annotations

from typing import Any

from .gate import (
    INDUCTION_CERTIFICATE_SCHEMA,
    INDUCTION_PANEL_SCHEMA,
    NATIVE_CONSUMER,
    CertificateError,
    ConsumerContract,
    InductionCircuitPolicy,
    certify_induction_route,
)
from .register import (
    certificate_receipt,
    register_certificate_claim,
    write_certificate,
)
from .workload import (
    InductionPanelSpec,
    build_relational_induction_workload,
    relational_induction_positions,
    wrong_donor_indices,
)

_LAZY = {"measure_induction_panel", "certify_decoder_induction"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import panel

        return getattr(panel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # gate
    "INDUCTION_CERTIFICATE_SCHEMA",
    "INDUCTION_PANEL_SCHEMA",
    "NATIVE_CONSUMER",
    "CertificateError",
    "ConsumerContract",
    "InductionCircuitPolicy",
    "certify_induction_route",
    # workload
    "InductionPanelSpec",
    "build_relational_induction_workload",
    "relational_induction_positions",
    "wrong_donor_indices",
    # measurement (lazy, torch)
    "measure_induction_panel",
    "certify_decoder_induction",
    # custody
    "certificate_receipt",
    "write_certificate",
    "register_certificate_claim",
]
