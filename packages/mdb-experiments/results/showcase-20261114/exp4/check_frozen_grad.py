# TODO(mdb-feature P4): run.hold('...p') / hold('...norm_scale') makes this bitwise check of the detach-based frozen gradient a debugger-mode test — hand-built nnsight freeze hooks live here and in exp4 (gap audit §2 P4).
import torch, nnsight
from transformers import Qwen3Config, Qwen3ForCausalLM
from mdb.interop import nn
cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attn_implementation="eager")
torch.manual_seed(0); model=Qwen3ForCausalLM(cfg).eval().to(torch.float32)
NL=4; ids=[1,2,3,4,5]; a,b=3,7; layer=1
def rho(x,eps): return (x.float().pow(2).mean(-1,keepdim=True)+eps).rsqrt()
nm=nn.wrap(model)
with torch.no_grad(), nm.trace(torch.tensor([ids])):
    saved=nnsight.save({})
    for x2 in range(layer+1,NL):
        L2=nm.model.layers[x2]; saved[("a",x2)]=L2.input_layernorm.input.clone(); saved[("n",x2)]=L2.post_attention_layernorm.input.clone()
    saved[("F",0)]=nm.model.norm.input.clone()
sc={"a":{},"n":{}}
for x2 in range(layer+1,NL):
    L2=model.model.layers[x2]
    sc["a"][x2]=(rho(saved[("a",x2)],L2.input_layernorm.variance_epsilon)*L2.input_layernorm.weight.float()).detach()
    sc["n"][x2]=(rho(saved[("n",x2)],L2.post_attention_layernorm.variance_epsilon)*L2.post_attention_layernorm.weight.float()).detach()
sc["F"]=(rho(saved[("F",0)],model.model.norm.variance_epsilon)*model.model.norm.weight.float()).detach()
t=torch.tensor([ids])
def frozen(delta=None):
    with nn._Frozen(model), torch.enable_grad(), nm.trace(t):
        emb=nm.model.embed_tokens.output; emb.requires_grad_(True)
        mout=nm.model.layers[layer].mlp.output
        if delta is not None:
            mout[0,-1]=mout[0,-1]+delta
        for x in range(layer+1,NL):
            L_=nm.model.layers[x]
            L_.input_layernorm.output=(L_.input_layernorm.input*sc["a"][x])
            L_.self_attn.q_proj.output=L_.self_attn.q_proj.output.detach()
            L_.self_attn.k_proj.output=L_.self_attn.k_proj.output.detach()
            L_.post_attention_layernorm.output=(L_.post_attention_layernorm.input*sc["n"][x])
        nm.model.norm.output=(nm.model.norm.input*sc["F"])
        lg=nm.lm_head.output[0,-1].float(); M=lg[a]-lg[b]
        g=nnsight.save([]); Ms=M.detach().save()
        with M.backward():
            g.append(mout.grad[0,-1].detach().double())
    return g[0], float(Ms)
g,M0=frozen()
lg=model(t).logits[0,-1]; print("frozen M", M0, "native M", float(lg[a]-lg[b]))
torch.manual_seed(1); d=torch.randn(32)*1e-6; model.double() if False else None
_,M1=frozen(d)
print("grad.d", float(g@d.double()), "finite", M1-M0)

# plain torch reference with hooks
store = {}
hs = []
def mk_norm(scale):
    return lambda mod, inp, out: inp[0] * scale
def det(mod, inp, out): return out.detach()
for x in range(layer + 1, NL):
    L_ = model.model.layers[x]
    hs.append(L_.input_layernorm.register_forward_hook(mk_norm(sc["a"][x])))
    hs.append(L_.post_attention_layernorm.register_forward_hook(mk_norm(sc["n"][x])))
    hs.append(L_.self_attn.q_proj.register_forward_hook(det))
    hs.append(L_.self_attn.k_proj.register_forward_hook(det))
hs.append(model.model.norm.register_forward_hook(mk_norm(sc["F"])))
def mlp_hook(mod, inp, out):
    out = out.clone()
    if store.get("delta") is not None:
        out[0, -1] = out[0, -1] + store["delta"]
    out = out.detach().requires_grad_(True) if not out.requires_grad else out
    out.retain_grad(); store["m"] = out
    return out
hs.append(model.model.layers[layer].mlp.register_forward_hook(mlp_hook))
with torch.enable_grad():
    for p in model.parameters(): p.requires_grad_(False)
    lg = model(t).logits[0, -1].float(); M = lg[a] - lg[b]; M.backward()
gt = store["m"].grad[0, -1].double()
store["delta"] = d
with torch.no_grad():
    lg2 = model(t).logits[0, -1].float()
print("torch grad.d", float(gt @ d.double()), "torch finite", float(lg2[a] - lg2[b]) - float(M), "grad diff", float((gt - g).abs().max()))
for h in hs: h.remove()
clean = {}
def keep(name):
    def fn(mod, inp, out):
        clean[name] = out.detach().clone()
    return fn
hk = [model.model.layers[x].self_attn.q_proj.register_forward_hook(keep(("q", x))) for x in range(layer + 1, NL)]
hk += [model.model.layers[x].self_attn.k_proj.register_forward_hook(keep(("k", x))) for x in range(layer + 1, NL)]
with torch.no_grad(): model(t)
for h in hk: h.remove()
hs = []
for x in range(layer + 1, NL):
    L_ = model.model.layers[x]
    hs.append(L_.input_layernorm.register_forward_hook(mk_norm(sc["a"][x])))
    hs.append(L_.post_attention_layernorm.register_forward_hook(mk_norm(sc["n"][x])))
    hs.append(L_.self_attn.q_proj.register_forward_hook(lambda m, i, o, x=x: clean[("q", x)]))
    hs.append(L_.self_attn.k_proj.register_forward_hook(lambda m, i, o, x=x: clean[("k", x)]))
hs.append(model.model.norm.register_forward_hook(mk_norm(sc["F"])))
store["delta"] = None
hs.append(model.model.layers[layer].mlp.register_forward_hook(mlp_hook))
with torch.no_grad():
    l0 = model(t).logits[0, -1].float()
    store["delta"] = d
    l1 = model(t).logits[0, -1].float()
print("frozen-forward finite", float(l1[a] - l1[b]) - float(l0[a] - l0[b]), "vs grad.d", float(g @ d.double()))
