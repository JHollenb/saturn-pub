"""Mathematical analysis layer: proof-series identities over captured ports.

``finite`` holds one function per identity (proof IDs of obsidian/model-math/proof-index.md) and
the ``IDENTITIES`` catalogue; ``explain`` holds ``explain_margin`` and ``explain_port``;
``forecast`` seals predictions of native values at a run's cursor and verifies them;
``propagate`` predicts and then measures the change an edit makes in a downstream consumer;
``reads`` lists who reads a source row (mass, exact contribution, native causal verdict);
``symbols`` builds the symbol table; ``circuits`` builds and natively tests circuit records;
``compare`` measures a consumer under several edits or checkpoints.
Every number leaving ``IdentityResult.summary()``, ``explain_margin`` or ``explain_port`` is a
labelled number (authority ``measured | derived-exact | derived-local | bound | supplied``) and
every result has a state (``resolved | unresolved | refused``) with a reason when not resolved. A
refused or unresolved ``IdentityResult.summary()`` makes no claim: its top-level authority is
``None`` and its numbers are inspection-only records ``{value, authority: None, inspection_only}``.
Importing this package does not import torch; functions import it when called.
"""

from . import circuits, compare, explain, finite, forecast, propagate, reads, symbols
from .finite import IDENTITIES, IdentityResult, list_identities

__all__ = ["IDENTITIES", "IdentityResult", "list_identities", "circuits", "compare", "explain",
           "finite", "forecast", "propagate", "reads", "symbols"]
