"""CPU check of bridge.py on saturn_pub's tiny random Flux2KleinAdapter (no pretrained weights).

Run (laptop, one process, CPU, ~seconds):
  W=<worktree>; PY=$W/.venv-or-any-python-with-torch-diffusers
  uv run --no-project --python $PY --with nnsight==0.8.2 env PYTHONPATH=$W/src:$W/packages/mdb/src:$W/packages/mdb-lanes/src \
      python packages/mdb-lanes/experiments/interp_bridge_20261114/toy_check.py

Checks: replay step == adapter.advance (bitwise); nnsight capture == port capture (bitwise);
nnsight zero/mask cuts == port cuts (bitwise); nnsight-wrapped native == native; first-order
attribution vs measured single-head cut effect; nnsight gradient == probe gradient (one step).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge  # noqa: E402

torch.manual_seed(0)
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from saturn_pub.adapters.flux2 import Flux2KleinAdapter  # noqa: E402
from mdb_lanes import attention_port as ap  # noqa: E402

STEPS, T, N_IMG = 4, 3, 4
adapter = Flux2KleinAdapter.tiny(granularity="operation")
tr = adapter.model
JOINT = len(tr.transformer_blocks)
BLOCKS = JOINT + len(tr.single_transformer_blocks)
H = tr.transformer_blocks[0].attn.heads
g = torch.Generator().manual_seed(11)
sched = FlowMatchEulerDiscreteScheduler()
sched.set_timesteps(STEPS)
inputs = dict(latent=torch.randn(1, 8, N_IMG, generator=g).transpose(1, 2),
              conditioning=torch.randn(1, T, 24, generator=g),
              img_ids=torch.arange(N_IMG * 4).reshape(1, N_IMG, 4).float() % 3,
              txt_ids=torch.zeros(1, T, 4), timesteps=sched.timesteps.clone(), sigmas=sched.sigmas.clone())
rows = {"image": list(range(T, T + N_IMG)), "text": list(range(T)), "own:w": [1], "others:w": [0, 2]}
results = {}


def run_adapter():
    sess = adapter.session(**{k: v.clone() for k, v in inputs.items()})
    state = sess._state
    while state["step"] < STEPS:
        state = adapter.advance(state)
    return state["latent"]


native = run_adapter()

# replay step == advance
x = inputs["latent"].clone()
xs = [x]
with torch.no_grad():
    for s in range(STEPS):
        x = bridge.step_forward(adapter, x, inputs["conditioning"], inputs["img_ids"], inputs["txt_ids"],
                                inputs["timesteps"], inputs["sigmas"], s)
        xs.append(x)
results["replay_equals_advance"] = bool(torch.equal(xs[-1], native))

# port capture + cuts
backend = ap.FluxBackend(tr, text_rows=T)
cells = [(1, 0), (JOINT, 1), (BLOCKS - 1, 0)]
obs = [ap.observe(bridge.port_observer_reduce(h), steps=[0, 2], blocks=[b], name=f"b{b}") for b, h in cells]
port = ap.AttentionPort(backend, rows=rows, zero_method="delta")
with port.apply(observers=obs, forwards=STEPS):
    port_native = run_adapter()
results["port_observe_equals_native"] = bool(torch.equal(port_native, native))
cell_cut = (JOINT + 1, 1)
cuts = {"zero": ap.cut(["image", "others:w"], ["own:w"], heads=[cell_cut], renormalize=False),
        "mask": ap.cut(["image", "others:w"], ["own:w"], heads=[cell_cut], renormalize=True)}
port_cut = {}
for name, c in cuts.items():
    with port.apply(c, forwards=STEPS):
        port_cut[name] = run_adapter()

# nnsight
from nnsight import NNsight  # noqa: E402
nn_tr = NNsight(tr)
results["nnsight_wrapped_native_equal"] = bool(torch.equal(run_adapter(), native))
store, final = bridge.nnsight_capture(nn_tr, run_adapter, cells, [0, 2], joint=JOINT)
results["nnsight_trace_native_equal"] = bool(torch.equal(final, native))
cmp = {}
for (b, h), o in zip(cells, obs):
    for s in (0, 2):
        pr, nr = o.records[(s, b)], store[(s, b)]
        p_nn = bridge.probs_of(nr["q"], nr["k"], nr["scale"])
        cmp[f"s{s}b{b}h{h}"] = {f: bridge.compare_tensors(pr[f], nr[f])["bitwise_equal"] for f in ("q", "k", "v", "out")}
        cmp[f"s{s}b{b}h{h}"]["p"] = bridge.compare_tensors(pr["p"], p_nn)["bitwise_equal"]
results["capture_bitwise"] = cmp
for name, c in cuts.items():
    out = bridge.nnsight_cut(nn_tr, run_adapter, [c], rows, range(STEPS), [cell_cut[0]], joint=JOINT)
    results[f"nnsight_cut_{name}_equals_port"] = bool(torch.equal(out, port_cut[name]))
    results[f"cut_{name}_changes_output"] = not bool(torch.equal(out, native))
results["nnsight_after_traces_native_equal"] = bool(torch.equal(run_adapter(), native))

# attribution vs measured single-head cut (iso-free toy: base = native, direction = random)
base = torch.zeros_like(native[0])
direction = torch.randn(native[0].shape, generator=g)
probe = bridge.Probe()
terms = {}
recv = torch.tensor(rows["image"] + rows["others:w"])
src = torch.tensor(rows["own:w"])


def on_step(s, pr):
    for b in range(BLOCKS):
        q, k, v, out, scale = pr.seen[b]
        w_img, w_txt = bridge.out_weights(tr, b)
        terms[(s, b)] = bridge.head_terms(q, k, v, out, pr.zeros[b].grad, recv, src,
                                          scale or q.shape[-1] ** -0.5, w_img, w_txt, T)


inputs_bw = dict(conditioning=inputs["conditioning"], img_ids=inputs["img_ids"], txt_ids=inputs["txt_ids"],
                 timesteps=inputs["timesteps"], sigmas=inputs["sigmas"], shape=(1, T + N_IMG, H, 16))
rep = bridge.backward_trajectory(adapter, xs, inputs_bw, base, direction, probe, checkpoint=True, on_step=on_step)
results["backward_report"] = rep
y0 = float(bridge.outcome(native, base, direction))
lin = []
for b, h in [(0, 0), (1, 1), (JOINT, 0), (JOINT + 1, 1), (BLOCKS - 1, 1)]:
    for kind, renorm in (("zero", False), ("mask", True)):
        c = ap.cut(["image", "others:w"], ["own:w"], heads=[(b, h)], renormalize=renorm)
        with port.apply(c, forwards=STEPS):
            y = float(bridge.outcome(run_adapter(), base, direction))
        pred = sum(float(terms[(s, b)][f"att_{kind}_img"][h] + terms[(s, b)][f"att_{kind}_txt"][h]) for s in range(STEPS))
        lin.append(dict(block=b, head=h, kind=kind, measured=y - y0, predicted=pred))
results["linearity"] = lin

# nnsight gradient for one step (last), one block, versus the probe gradient
s_last, b_g = STEPS - 1, JOINT
x = xs[s_last].detach().clone().requires_grad_(True)


def one_step():
    return bridge.step_forward(adapter, x, inputs["conditioning"], inputs["img_ids"], inputs["txt_ids"],
                               inputs["timesteps"], inputs["sigmas"], s_last)


try:
    with nn_tr.trace(fn=one_step) as tracer:
        with tracer.invoke():
            out = bridge.dispatch_op(nn_tr, b_g, JOINT).output
            res = tracer.result
            y = bridge.outcome(res, base, direction)
            with y.backward():
                gn = out.grad.save()
    probe_grad = None
    probe.arm(record=True, grads=True, blocks=range(BLOCKS), like=((1, T + N_IMG, H, 16), adapter.dtype, x.device))
    x2 = xs[s_last].detach().clone().requires_grad_(True)
    with torch.enable_grad(), probe:
        y2 = bridge.outcome(bridge.step_forward(adapter, x2, inputs["conditioning"], inputs["img_ids"],
                                                inputs["txt_ids"], inputs["timesteps"], inputs["sigmas"], s_last,
                                                probe=probe), base, direction)
        y2.backward()
    results["nnsight_grad_vs_probe"] = bridge.compare_tensors(gn, probe.zeros[b_g].grad)
except Exception as e:  # recorded, not hidden
    results["nnsight_grad_error"] = f"{type(e).__name__}: {e}"

print(json.dumps(results, indent=1, default=str))

# extra access points (same calls as interp_bridge_worker.extras)
jenv, senv = nn_tr.transformer_blocks[0], nn_tr.single_transformer_blocks[0]
names_j = [op.name for op in jenv.source]
names_s = [op.name for op in senv.source]
found = {}
with nn_tr.trace(fn=run_adapter) as tracer:
    with tracer.invoke():
        mod = getattr(jenv.source, "Flux2Modulation_split_0").output
        found["joint_gate_msa"] = mod[0][2].detach().clone()
        d_out = bridge.dispatch_op(nn_tr, 0, JOINT).output
        found["joint_dispatch_out"] = d_out.detach().clone()
        found["joint_to_add_out_in"] = jenv.attn.to_add_out.input.detach().clone()
        found["joint_to_out_in"] = jenv.attn.to_out[0].input.detach().clone()
        found["joint_ff_linear_in_out"] = jenv.ff.linear_in.output.detach().clone()
        found["single_dispatch_out"] = bridge.dispatch_op(nn_tr, JOINT, JOINT).output.detach().clone()
        found["single_mlp_act_in"] = senv.attn.mlp_act_fn.input.detach().clone()
        found["single_to_out_in"] = senv.attn.to_out.input.detach().clone()
        done = tracer.result.save()
inner = found["single_dispatch_out"].shape[2] * found["single_dispatch_out"].shape[3]
print(json.dumps(dict(
    names_j=names_j, names_s=names_s, shapes={k: list(v.shape) for k, v in found.items()},
    to_out_identity=bool(torch.equal(found["joint_to_out_in"], found["joint_dispatch_out"][:, T:].flatten(2, 3))),
    to_add_out_identity=bool(torch.equal(found["joint_to_add_out_in"], found["joint_dispatch_out"][:, :T].flatten(2, 3))),
    single_identity=bool(torch.equal(found["single_to_out_in"][..., :inner], found["single_dispatch_out"].flatten(2, 3))),
    extras_native_equal=bool(torch.equal(done, native))), indent=0))
