"""mdb <-> TransformerLens and nnsight.

- ``names``: one name table, mdb ports <-> TL hook names <-> nnsight paths, and the layout
  conversions (torch-free).
- ``tl``: a TransformerBridge around the attached module; TL caches imported as mdb origins;
  ``mdb.trace`` captures exported as TL caches.
- ``nn``: nnsight reads by mdb address and ``batch_whatif`` (mdb edits measured over many
  prompts in batched traces; cell edits in mdb's ``reader`` scope or ``sequence`` scope);
  ``grads`` (values and input gradients of a margin, one batched forward + backward).
- ``verify``: a claimed effect from another tool re-measured by mdb natively, with mdb's
  sealed prediction and P20 verdict beside the claim.

Importing ``mdb.interop`` loads neither torch nor the tools; each submodule imports its tool
when used.
"""

__all__ = ["names", "tl", "nn", "verify"]
