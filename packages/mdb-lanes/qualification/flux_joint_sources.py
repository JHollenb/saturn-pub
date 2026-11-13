"""Scoped surgery at native normalized, positioned Q/K/V in one candidate block."""
from __future__ import annotations

import runpy
from pathlib import Path


def source_candidate(block, cut, *, donor=None, mask="000", text_query=False,
                     attention=None, mlp=None):
    from diffusers.models.transformers import transformer_flux2 as native

    helper = runpy.run_path(str(Path(__file__).with_name("flux_joint_writes.py")))
    if len(mask) != 3 or any(v not in "01" for v in mask):
        raise ValueError("requires image-Q/text-KV/image-KV mask")
    if (mask != "000" or text_query) and donor is None:
        raise ValueError("source replacement requires observed donor")
    if type(block.attn.processor).__name__ != "Flux2AttnProcessor":
        raise ValueError("requires qualified native joint attention processor")
    original, captured = native.dispatch_attention_fn, {}
    text_rows = cut.payload["text"].shape[1]

    def dispatch(q, k, v, **kwargs):
        if captured:
            raise ValueError("candidate dispatched attention more than once")
        captured["observed"] = {name: value.detach().clone() for name, value in (("q", q), ("k", k), ("v", v))}
        effective = {name: value.clone() for name, value in (("q", q), ("k", k), ("v", v))}
        if donor is not None:
            for name, value in effective.items():
                supplied = donor[name]
                if supplied.shape != value.shape or supplied.dtype != value.dtype or supplied.device != value.device:
                    raise ValueError("Q/K/V donor contract differs")
            if mask[0] == "1":
                effective["q"][:, text_rows:] = donor["q"][:, text_rows:]
            if text_query:
                effective["q"][:, :text_rows] = donor["q"][:, :text_rows]
            for bit, rows in ((mask[1], slice(None, text_rows)), (mask[2], slice(text_rows, None))):
                if bit == "1":
                    for name in ("k", "v"):
                        effective[name][:, rows] = donor[name][:, rows]
        captured["effective"] = effective
        captured["dispatch"] = {"backend": str(kwargs.get("backend")), "parallel_config": str(kwargs.get("parallel_config")),
                                "mask_is_none": kwargs.get("attn_mask") is None}
        return original(effective["q"], effective["k"], effective["v"], **kwargs)

    try:
        native.dispatch_attention_fn = dispatch
        result = helper["joint_candidate"](block, cut, attention=attention, mlp=mlp)
        if set(captured) != {"observed", "effective", "dispatch"}:
            raise ValueError("native positioned Q/K/V were not observed")
        result.update(qkv=captured, text_rows=text_rows)
        return result
    finally:
        native.dispatch_attention_fn = original
