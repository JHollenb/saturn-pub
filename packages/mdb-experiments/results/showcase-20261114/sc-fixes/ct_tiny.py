"""SC8: circuit-tracer 0.5.0 attribution on a tiny random Qwen3 (CPU, fp32)."""
import sys, traceback, glob, os
import torch; torch.set_num_threads(4); torch.manual_seed(0)
import huggingface_hub.constants as c; c.HF_HUB_ENABLE_HF_TRANSFER = False
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
S = "/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/sc"
D = f"{S}/tiny-qwen3"
TOKSRC = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"))[0]
if not os.path.exists(f"{D}/config.json"):
    tok = AutoTokenizer.from_pretrained(TOKSRC)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                      tie_word_embeddings=False, attn_implementation="eager")
    m = Qwen3ForCausalLM(cfg); m.save_pretrained(D); tok.save_pretrained(D)
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
d, F, L = 64, 32, 2
tcs = {}
for l in range(L):
    t = SingleLayerTranscoder(d, F, torch.nn.ReLU(), l, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        t.W_enc.normal_(0, 0.3); t.W_dec.normal_(0, 0.3); t.b_enc.fill_(0.0)
    tcs[l] = t
ts = TranscoderSet(tcs, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out", scan="tiny")
from circuit_tracer import ReplacementModel, attribute
backend = sys.argv[1] if len(sys.argv) > 1 else "nnsight"
try:
    rm = ReplacementModel.from_pretrained_and_transcoders(D, ts, backend=backend, device="cpu", dtype=torch.float32)
    print("loaded", type(rm).__name__)
except Exception:
    traceback.print_exc(); print("CALL: ReplacementModel.from_pretrained_and_transcoders"); sys.exit(1)
try:
    g = attribute("The capital of France is", rm, max_n_logits=3, batch_size=16, verbose=False)
    print("graph", g.adjacency_matrix.shape, g.active_features.shape, g.logit_tokens if hasattr(g,'logit_tokens') else None)
    g.to_pt(f"{S}/tiny-graph-{backend}.pt")
    torch.save({"adjacency_matrix": g.adjacency_matrix, "active_features": g.active_features,
                "activation_values": g.activation_values, "selected_features": g.selected_features,
                "input_tokens": g.input_tokens, "logit_token_ids": g.logit_token_ids,
                "n_layers": L, "circuit_tracer": "0.5.0"}, f"{S}/tiny-graph-plain.pt")
    torch.save({"W_enc": [t.W_enc.detach() for t in ts], "W_dec": [t.W_dec.detach() for t in ts],
                "b_enc": [t.b_enc.detach() for t in ts], "b_dec": [t.b_dec.detach() for t in ts]}, f"{S}/tiny-tc.pt")
    import transformers, nnsight; print("versions", transformers.__version__, nnsight.__version__, torch.__version__)
except Exception:
    traceback.print_exc(); print("CALL: circuit_tracer.attribute"); sys.exit(1)
