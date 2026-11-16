"""Item 4: hop_edge (spectral, shared cache, twice) and gate_switch_forecast on a small random
Qwen2 (CPU bf16): full JSON output for a bit-identity diff, plus the process peak RSS and the
analysis' own RSS growth. usage: python sym_check.py OUT.json [L d d_ff]"""
import json
import resource
import sys

import torch

from mdb.analysis import symbols
from mdb.analysis.forecast import Margin
from mdb.analysis.symbols import readout
from mdb.engine.run import Run
from saturn_pub.adapters.decoder import DecoderAdapter
from transformers import Qwen2Config, Qwen2ForCausalLM

torch.manual_seed(0)
torch.set_num_threads(2)
L, d, dff = (int(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (4, 256, 1024)))
cfg = Qwen2Config(vocab_size=512, hidden_size=d, intermediate_size=dff, num_hidden_layers=L,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  attn_implementation="eager")
model = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
ad = DecoderAdapter(model, granularity="operation")
PROMPT = [1, 5, 9, 3]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS: bytes


run = Run(ad, PROMPT)
run.until("t3.logits")
reader = readout(run, Margin(3, 4, 9))
donor = Run(ad, [2, 7, 9, 3])
value = donor.read_cache(0, "v", staged=False).clone()[:, 0:1, 1:2]
edited = Run(ad, PROMPT)
edited.set_cell(0, "v", value, rows=[1], kv=[0])
edited.until("t3.logits")
before = rss_mb()
cache: dict = {}
out = {
    "edge1": symbols.hop_edge(run, edited, reader, cell_layer=0, top_neurons=4, spectral=True,
                              cache=cache),
    "edge2": symbols.hop_edge(run, edited, reader, cell_layer=0, top_neurons=4, spectral=True,
                              cache=cache),
    "gsf": symbols.gate_switch_forecast(run, edited, reader, cell_layer=0, ks=(1, 4),
                                        cache=cache),
}
after = rss_mb()
cached = sum(v.numel() * v.element_size() for v in cache.values() if isinstance(v, torch.Tensor))
cached += sum(t.numel() * t.element_size() for v in cache.values() if isinstance(v, tuple)
              for t in v if isinstance(t, torch.Tensor))
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True, default=str)
print(json.dumps({"L": L, "d": d, "d_ff": dff, "peak_rss_mb": round(after, 1),
                  "rss_growth_mb": round(after - before, 1), "cache_bytes": cached}))
