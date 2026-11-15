"""SC1 check: SAELens's own qwen_scope loader (via the local-path shim) vs mdb's FeatureBasis.

usage: python sc1_check.py SAE_DIR LAYER [RELEASE]
  SAE_DIR  directory holding layer{L}.sae.pt (e.g. /mnt/big/llm-models/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50)
  LAYER    layer index
  RELEASE  SAELens release (default: from the file's config repo, e.g. qwen-scope-3-1.7b-base-w32k-l50)

Compares, for the same file, three objects: (a) saelens_from_local (SAELens's converter),
(b) FeatureBasis.to_saelens() (mdb's builder), (c) the FeatureBasis. W_enc/W_dec/b_enc/b_dec are
compared bitwise; k from each config; active sets on 64 gaussian residuals (seed 0) scaled to the
file's own b_dec norm (a stand-in for real residual scale). Prints one JSON object."""
import json
import sys

import torch

from mdb.analysis import sae


def main(sae_dir: str, layer: int, release: str) -> dict:
    basis = sae.open_set("sc1", sae_dir).get(layer)
    ref = sae.saelens_ref(basis.source.get("repo"), layer, basis.port)
    release = release or ref.get("release")
    if not release:
        sys.exit("no SAELens release for this repo; pass RELEASE")
    theirs = sae.saelens_from_local(release, f"layer{layer}", sae_dir)
    mine = basis.to_saelens()
    out = {"release": release, "layer": layer, "file": basis.source.get("file"),
           "sha256_manifest": basis.source.get("sha256_manifest"), "d_sae": basis.d_sae,
           "k_file": basis.k, "k_saelens_loader": int(theirs.cfg.k),
           "k_mdb_to_saelens": int(mine.cfg.k), "tensors_bitwise": {}}
    for name, native in (("W_enc", basis.W_enc.T), ("W_dec", basis.W_dec.T),
                         ("b_enc", basis.b_enc), ("b_dec", basis.b_dec)):
        out["tensors_bitwise"][name] = {
            "loader_vs_basis": bool(torch.equal(getattr(theirs, name), native)),
            "to_saelens_vs_basis": bool(torch.equal(getattr(mine, name), native)),
            "loader_vs_to_saelens": bool(torch.equal(getattr(theirs, name), getattr(mine, name)))}
    g = torch.Generator().manual_seed(0)
    scale = float(basis.b_dec.norm()) or 1.0
    xs = torch.randn(64, basis.d_model, generator=g) * (scale / basis.d_model ** 0.5)
    with torch.no_grad():
        a_theirs, a_mine = theirs.encode(xs), mine.encode(xs)
    equal_loader = equal_mine = same_acts = 0
    for i in range(64):
        ids = sorted(basis.encode(xs[i])["ids"])
        equal_loader += sorted(a_theirs[i].nonzero().reshape(-1).tolist()) == ids
        equal_mine += sorted(a_mine[i].nonzero().reshape(-1).tolist()) == ids
        same_acts += bool(torch.equal(a_theirs[i], a_mine[i]))
    out["active_sets_of_64"] = {"loader_equals_basis": equal_loader,
                                "to_saelens_equals_basis": equal_mine,
                                "loader_acts_bitwise_equal_to_saelens": same_acts,
                                "input_scale_b_dec_norm": scale}
    out["pass"] = (all(all(v.values()) for v in out["tensors_bitwise"].values())
                   and equal_loader == 64 and equal_mine == 64
                   and out["k_file"] == out["k_saelens_loader"] == out["k_mdb_to_saelens"])
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    print(json.dumps(main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else ""),
                     indent=2))
