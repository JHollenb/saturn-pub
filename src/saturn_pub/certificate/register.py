"""Bind an induction certificate to saturn-pub's custody plane.

A certificate dict is already receipt-shaped: it carries top-level ``gates`` and a
``content_sha256`` seal. These helpers wrap it as a :class:`saturn_pub.core.Receipt`,
write it to disk receipt-style, and register it as a row in a
:class:`saturn_pub.evidence.ClaimsRegistry` so a certificate drift-detects like any
other claim (if the certificate JSON on disk changes, :func:`saturn_pub.evidence.verify`
marks the claim stale). This module imports no model framework.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core import Receipt
from ..evidence.claims import ClaimRow, ClaimsRegistry, ingest_receipt
from .gate import INDUCTION_CERTIFICATE_SCHEMA


def _check_certificate(certificate: Mapping[str, Any]) -> None:
    if certificate.get("schema") != INDUCTION_CERTIFICATE_SCHEMA:
        raise ValueError("object is not an induction certificate")
    if "content_sha256" not in certificate or "gates" not in certificate:
        raise ValueError("certificate is missing its seal or gate summary")


def certificate_receipt(certificate: Mapping[str, Any]) -> Receipt:
    """Wrap a certificate as a content-sealed saturn-pub Receipt."""

    _check_certificate(certificate)
    return Receipt.make(kind="induction-circuit-certificate", certificate=dict(certificate))


def write_certificate(certificate: Mapping[str, Any], path: str | Path) -> Path:
    """Write the certificate to ``path`` as indented JSON (ingest-ready)."""

    _check_certificate(certificate)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def register_certificate_claim(
    registry: ClaimsRegistry,
    certificate: Mapping[str, Any],
    claim_id: str,
    text: str,
    path: str | Path,
    *,
    status: str | None = None,
    worker: str | None = None,
    submit_command: str | None = None,
) -> ClaimRow:
    """Write the certificate and register it as a chained claim row.

    ``status`` defaults to ``"measured"`` when the certificate passed and ``"refuted"``
    when any gate failed -- a failing certificate is a legitimate recorded observation,
    not an error.
    """

    _check_certificate(certificate)
    resolved_status = status or ("measured" if certificate.get("certified") else "refuted")
    certificate_path = write_certificate(certificate, path)
    return ingest_receipt(
        registry,
        certificate_path,
        claim_id,
        text,
        resolved_status,
        worker=worker,
        submit_command=submit_command,
    )


__all__ = [
    "certificate_receipt",
    "write_certificate",
    "register_certificate_claim",
]
