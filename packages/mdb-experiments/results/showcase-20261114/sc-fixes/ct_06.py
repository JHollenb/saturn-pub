"""SC8: circuit-tracer 0.5.0 attribution on local Qwen3-0.6B (CPU fp32), random 16-feature transcoders."""
import glob, os, time
import torch; torch.set_num_threads(4); torch.manual_seed(1)
S = "/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/sc"
M = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*"))[0]
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(M); d, L, F = cfg.hidden_size, cfg.num_hidden_layers, 16
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
tcs = {}
for l in range(L):
    t = SingleLayerTranscoder(d, F, torch.nn.ReLU(), l, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        t.W_enc.normal_(0, d ** -0.5); t.W_dec.normal_(0, 0.05)
    tcs[l] = t
ts = TranscoderSet(tcs, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out", scan="rand16")
from circuit_tracer import ReplacementModel, attribute
t0 = time.time()
rm = ReplacementModel.from_pretrained_and_transcoders(M, ts, backend="nnsight", device="cpu", dtype=torch.float32)
g = attribute("The capital of France is", rm, max_n_logits=3, batch_size=64)
print("graph", g.adjacency_matrix.shape, g.active_features.shape, g.logit_token_ids.tolist(), "%.1fs" % (time.time() - t0))
torch.save({"adjacency_matrix": g.adjacency_matrix, "active_features": g.active_features,
            "activation_values": g.activation_values, "input_tokens": g.input_tokens,
            "logit_token_ids": g.logit_token_ids, "n_layers": L, "model": M}, f"{S}/q06-graph-plain.pt")
torch.save({"W_enc": [t.W_enc.detach() for t in ts], "W_dec": [t.W_dec.detach() for t in ts],
            "b_enc": [t.b_enc.detach() for t in ts], "b_dec": [t.b_dec.detach() for t in ts]}, f"{S}/q06-tc.pt")
