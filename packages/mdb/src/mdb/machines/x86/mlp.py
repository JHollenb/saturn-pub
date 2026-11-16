"""torchx86's ALU dispatch as a gated MLP, read as mdb sites (run 6 of the binary-ground-truth PREREG).

torchx86's ``Dispatch`` computes every ALU class on every lane (``up``), gates by the decoded class
(``gate``), ``f = gate * up``, ``down = sum_k f[k]``. This module records those four sites per
lane-step and gives them addresses and verbs::

    i<t>.mlp.up[k]     class k's outputs as 134 bits: value (64) | value2 (64) | CF PF AF ZF SF OF
    i<t>.mlp.gate[k]   class k's gate (hard: one-hot of the decoded class; soft: softmax(onehot / tau))
    i<t>.mlp.f[k]      gate[k] * up[k]
    i<t>.mlp.down      the dispatch output as 134 bits (hard: exact; soft: sum_k f[k] > 0.5)

``k`` is a class index or a torchx86 class name (``add``, ``xor``, ...). ``t`` is the tick (the
instruction index, from 0) during which the dispatch ran.

``attribute`` names the class unit responsible for ``down`` from ``f`` and ``down`` alone (it never
reads the decoded class): the exact contribution ``c_k = sum_{j: down_j} f[k, j]`` and the ablation
``a_k`` = number of output bits that flip when ``f[k]`` is zeroed. A unit is named when its score
is the unique maximum and > 0; otherwise nothing is named (-1).

The SAE here is a minimal top-k SAE in the layout mdb's ``FeatureBasis`` stores (``W_enc (w, d)``,
``b_enc``, ``W_dec (d, w)``, ``b_dec``, active set = top-k of ``relu(W_enc x + b_enc)``). mdb has no
SAE trainer, and ``FeatureBasis`` only accepts residual ports (``h``, ``h_out``) of a language model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from mdb.core.records import seal_record

from .binary import Binary, torchx86
from .lanes import Lanes
from .trace import default_input

NB = 134                              # bits per class unit: value 64, value2 64, six flags
_ADDR = re.compile(r"^i(?P<t>\d+)\.mlp\.(?P<site>up|gate|f|down)(?:\[(?P<k>\w+)\])?$")


def classes() -> tuple:
    torchx86()
    from torchx86.decode import CLASSES  # noqa: PLC0415

    return CLASSES


def bits_of(words: torch.Tensor) -> torch.Tensor:
    """[..., 8] int64 (value, value2, 6 flags) -> [..., 134] uint8."""
    sh = torch.arange(64, device=words.device)
    v = (words[..., 0, None] >> sh) & 1
    v2 = (words[..., 1, None] >> sh) & 1
    return torch.cat([v, v2, (words[..., 2:] != 0).long()], -1).to(torch.uint8)


def words_of(bits: torch.Tensor) -> tuple[int, int, int]:
    """134 bits -> (value, value2, flag bits CF..OF as an int), unsigned."""
    b = [int(x) for x in bits.reshape(-1).tolist()]
    w = lambda s: sum(bit << i for i, bit in enumerate(b[s:s + 64]))
    return w(0), w(64), sum(bit << i for i, bit in enumerate(b[128:]))


# --------------------------------------------------------------------------- capture


@dataclass
class Capture:
    """One batched run in a gated dispatch mode. Per lane-step (``[B, T]``): ``pc`` (-1 once the
    lane has stopped), ``up`` uint8 ``[B, T, K, 134]``, ``gate`` float ``[B, T, K]``, ``down`` uint8
    ``[B, T, 134]`` (the machine's own output). ``result`` is the lanes' ``Result``."""
    pc: torch.Tensor
    up: torch.Tensor
    gate: torch.Tensor
    down: torch.Tensor
    result: object

    @property
    def live(self) -> torch.Tensor:
        return self.pc >= 0

    def f(self) -> torch.Tensor:
        return self.gate[..., None] * self.up.float()


def capture(binary: Binary, inputs: list, *, mode: str = "hard", tau=1.0, device="cpu", cap: int = 20000,
            edit: tuple | None = None) -> Capture:
    """Run every input on its own lane with ``Dispatch(mode, tau)`` and record the four sites.

    ``tau`` may be a float or a tensor ``[B, 1]`` (one temperature per lane). ``edit = (t, k, value)``
    replaces ``gate[k]`` with ``value`` on every lane while it executes tick ``t`` (the gate is the
    machine's own: int64 one-hot under ``hard``)."""
    torchx86()
    from torchx86.machine import Dispatch  # noqa: PLC0415

    lanes = Lanes(binary, inputs, device=device, cap=cap)
    m = lanes.m
    if torch.is_tensor(tau):
        tau = tau.to(m.dev)
    m.dispatch = Dispatch(mode, tau)
    d = m.dispatch
    ups, gates, downs = [], [], []

    def on_up(_m, _i, out):
        ups.append(bits_of(out) if mode == "hard" else out.to(torch.uint8))

    def on_gate(_m, _i, out):
        if edit is not None:
            t, k, v = edit
            hit = m.steps == t
            out = out.clone()
            out[:, k] = torch.where(hit, torch.as_tensor(v, dtype=out.dtype, device=out.device), out[:, k])
        gates.append(out.float())
        return out

    def on_down(_m, _i, out):
        downs.append(bits_of(out) if mode == "hard" else (out > 0.5).to(torch.uint8))

    hooks = [d.up.register_forward_hook(on_up), d.gate.register_forward_hook(on_gate),
             d.down.register_forward_hook(on_down)]
    try:
        res = lanes.run(record="pc")
    finally:
        for h in hooks:
            h.remove()
    st = lambda xs: torch.stack(xs, 1)
    return Capture(res.pc, st(ups), st(gates), st(downs), res)


def decoded_class(binary: Binary, pc: torch.Tensor) -> torch.Tensor:
    """The decoded ALU class per tick (-1 where pc < 0). Scoring only: attribution never reads it."""
    sorted_pcs = torch.tensor(binary.pcs, dtype=torch.int64, device=pc.device)
    j = torch.searchsorted(sorted_pcs, pc.clamp(min=0)).clamp(max=len(binary.pcs) - 1)
    row = torch.where(sorted_pcs[j] == pc, j + 1, 0)
    cls = torch.tensor(binary.rows[:, binary.F["cls"]], device=pc.device)[row]
    return torch.where(pc >= 0, cls, -1)


# --------------------------------------------------------------------------- attribution


def attribute(f: torch.Tensor, down: torch.Tensor) -> dict:
    """Blind attribution of ``down`` [N, 134] to the class units of ``f`` [N, K, 134].

    ``contribution[n, k] = sum_{j: down_j} f[n, k, j]`` (exact share of the firing bits' pre-threshold
    mass); ``ablation[n, k]`` = number of bits of ``(sum f - f[k]) > 0.5`` that differ from ``down``.
    ``named_*`` is the unique argmax with a positive score, else -1."""
    on = down.bool()
    contrib = (f * on[:, None, :]).sum(-1)
    pre = f.sum(1)
    abl = (((pre[:, None, :] - f) > 0.5) != on[:, None, :]).sum(-1).float()
    return {"contribution": contrib, "ablation": abl,
            "named_contribution": _unique_argmax(contrib), "named_ablation": _unique_argmax(abl)}


def _unique_argmax(s: torch.Tensor) -> torch.Tensor:
    top = s.max(-1)
    unique = (s == top.values[:, None]).sum(-1) == 1
    return torch.where(unique & (top.values > 0), top.indices, -1)


def attribute_capture(cap: Capture, chunk: int = 1 << 14) -> dict:
    """``attribute`` over every live lane-step of a capture (flattened, lane-major)."""
    live = cap.live
    up, gate, down = cap.up[live], cap.gate[live], cap.down[live]
    out = {"named_contribution": [], "named_ablation": []}
    for s in range(0, len(up), chunk):
        r = attribute(gate[s:s + chunk, :, None] * up[s:s + chunk].float(), down[s:s + chunk])
        for key in out:
            out[key].append(r[key])
    return {k: torch.cat(v) if v else torch.zeros(0, dtype=torch.long) for k, v in out.items()}


# --------------------------------------------------------------------------- usage


def usage(cap: Capture, cls: torch.Tensor, loop_heads) -> list[dict]:
    """Per lane: class units gated on per step, units with nonzero f, nonzero f coordinates, the union
    of gated units over the run and the tick at which it saturates, and the end of the first loop
    iteration (the first return to a loop head already visited)."""
    K = cap.gate.shape[-1]
    heads = torch.tensor(sorted(loop_heads) or [-2], device=cap.pc.device)
    out = []
    for lane in range(cap.pc.shape[0]):
        live = cap.live[lane]
        T = int(live.sum())
        g = cap.gate[lane, :T] != 0                                      # [T, K]
        fnz = (cap.gate[lane, :T, :, None] * cap.up[lane, :T].float()) != 0
        first = torch.where(g.any(0), g.float().argmax(0), -1)           # first tick each unit is on
        used = first >= 0
        coord_first = torch.where(fnz.any(0), fnz.float().argmax(0), -1)
        pcs = cap.pc[lane, :T]
        at_head = (pcs[:, None] == heads).any(1)
        loop1 = -1
        seen = set()
        for t in torch.nonzero(at_head).flatten().tolist():
            p = int(pcs[t])
            if p in seen:
                loop1 = t
                break
            seen.add(p)
        sat = int(first[used].max()) if used.any() else -1
        out.append({
            "T": T, "K": K,
            "units_on_per_step_mean": float(g.sum(1).float().mean()),
            "units_on_per_step_max": int(g.sum(1).max()),
            "units_f_nonzero_per_step_mean": float(fnz.any(-1).sum(1).float().mean()),
            "coords_f_nonzero_per_step_mean": float(fnz.sum((1, 2)).float().mean()) / (K * NB),
            "union_units": int(used.sum()), "union_units_frac": int(used.sum()) / K,
            "union_classes_decoded": len(set(cls[lane, :T].tolist())),
            "union_saturation_tick": sat,
            "union_coords_frac": int((coord_first >= 0).sum()) / (K * NB),
            "union_coords_saturation_tick": int(coord_first.max()),
            "first_loop_iteration_end_tick": loop1,
            "union_units_at_loop1_end": int((used & (first <= loop1)).sum()) if loop1 >= 0 else 0,
            "units": sorted(torch.nonzero(used).flatten().tolist()),
        })
    return out


# --------------------------------------------------------------------------- SAE


def train_topk_sae(X: torch.Tensor, groups: list, *, width: int, k: int, steps: int = 2000, batch: int = 512,
                   lr: float = 1e-3, seed: int = 0) -> dict:
    """Train ``len(groups)`` independent top-k SAEs at once; group g sees rows ``X[groups[g]]``.

    Returns float32 tensors ``W_enc [G, w, d]``, ``b_enc [G, w]``, ``W_dec [G, d, w]``, ``b_dec [G, d]``
    (``FeatureBasis`` layout per group) and the final batch MSE per group. Decoder columns are kept
    at unit norm; ``b_dec`` starts at the group mean, ``W_enc`` at ``W_dec^T``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev, G, d = X.device, len(groups), X.shape[1]
    W_dec = torch.randn(G, d, width, generator=gen).to(dev)
    W_dec /= W_dec.norm(dim=1, keepdim=True)
    W_dec.requires_grad_(True)
    W_enc = W_dec.detach().transpose(1, 2).clone().requires_grad_(True)
    b_enc = torch.zeros(G, width, device=dev, requires_grad=True)
    b_dec = torch.stack([X[g].float().mean(0) for g in groups]).requires_grad_(True)
    sizes = torch.tensor([len(g) for g in groups], device=dev)
    flat = torch.cat([torch.as_tensor(g, device=dev) for g in groups])
    offs = torch.cumsum(sizes, 0) - sizes
    opt = torch.optim.Adam([W_enc, b_enc, W_dec, b_dec], lr=lr)
    gdev = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        r = (torch.rand(G, batch, device=dev, generator=gdev) * sizes[:, None]).long()
        xb = X[flat[offs[:, None] + r]].float()                        # [G, b, d]
        a = _encode(xb, W_enc, b_enc, b_dec, k)
        recon = torch.einsum("gbw,gdw->gbd", a, W_dec) + b_dec[:, None]
        loss_g = ((recon - xb) ** 2).sum(-1).mean(-1)
        opt.zero_grad()
        loss_g.sum().backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    p = {n: t.detach() for n, t in (("W_enc", W_enc), ("b_enc", b_enc), ("W_dec", W_dec), ("b_dec", b_dec))}
    p["k"] = k
    p["final_mse"] = loss_g.detach()
    return p


def _encode(x, W_enc, b_enc, b_dec, k):  # noqa: ARG001  (b_dec is not applied to the input, as in FeatureBasis)
    pre = torch.relu(torch.einsum("gbd,gwd->gbw", x, W_enc) + b_enc[:, None])     # FeatureBasis.select rule
    v, i = pre.topk(k, -1)
    return torch.zeros_like(pre).scatter(-1, i, v)


def sae_encode(p: dict, g: int, x: torch.Tensor) -> torch.Tensor:
    """Activations [N, w] of group g's SAE on rows x [N, d] (top-k rule)."""
    with torch.no_grad():
        return _encode(x.float()[None], p["W_enc"][g:g + 1], p["b_enc"][g:g + 1], p["b_dec"][g:g + 1], p["k"])[0]


def purity(acts: torch.Tensor, cls: torch.Tensor, K: int) -> list[dict]:
    """Per feature: firing count, modal class, class purity among the rows where it is active."""
    on = acts > 0
    onehot = torch.nn.functional.one_hot(cls.clamp(min=0), K).float()
    counts = on.float().T @ onehot                                      # [w, K]
    n = counts.sum(1)
    mode = counts.argmax(1)
    return [{"feature": i, "fires": int(n[i]), "mode_class": int(mode[i]),
             "purity": float(counts[i, mode[i]] / n[i]) if n[i] > 0 else None} for i in range(acts.shape[1])]


def value_split(acts: torch.Tensor, X: torch.Tensor, cls: torch.Tensor, feats: list[dict], K: int) -> dict:
    """Does a class's data split across features by value? For each live feature with purity >= 0.9:
    the bit agreement of its active rows (mean over the class block's bits of max(p, 1-p)) minus the
    same over all rows of its class. Also: per class, how many pure features it has."""
    lifts, per_class = [], {}
    for ft in feats:
        if not ft["fires"] or ft["purity"] < 0.9:
            continue
        c = ft["mode_class"]
        blk = X[:, c * NB:(c + 1) * NB].float()
        rows = (acts[:, ft["feature"]] > 0) & (cls == c)
        allc = cls == c
        agree = lambda m: float(torch.maximum(blk[m].mean(0), 1 - blk[m].mean(0)).mean())
        lifts.append(agree(rows) - agree(allc))
        per_class[c] = per_class.get(c, 0) + 1
    return {"pure_features": len(lifts), "mean_bit_agreement_lift": sum(lifts) / len(lifts) if lifts else None,
            "classes_with_2plus_pure_features": sum(v >= 2 for v in per_class.values()),
            "pure_features_per_class": {str(k): v for k, v in sorted(per_class.items())}}


# --------------------------------------------------------------------------- a debugger view


class MlpRun:
    """One input's run with the dispatch as sites: ``print``/``read`` an ``i<t>.mlp.*`` address,
    ``attribute(t)``, and ``whatif`` on a gate unit (sealed prediction of ``down``, then native)."""

    def __init__(self, program, input: bytes | None = None, *, device="cpu", cap: int | None = None):
        self.binary = program if isinstance(program, Binary) else Binary.load(program)
        self.input = input if input is not None else default_input(self.binary)
        self.device = device
        self.cap = cap or 20000
        self.cap_ = capture(self.binary, [self.input], device=device, cap=self.cap)
        self.T = int(self.cap_.live[0].sum())
        self.cap = cap or 4 * self.T + 64
        self.names = classes()

    def _unit(self, k: str | None) -> int | None:
        if k is None:
            return None
        return int(k) if k.isdigit() else self.names.index(k)

    def _parse(self, text: str) -> tuple[int, str, int | None]:
        m = _ADDR.match(text.strip())
        if not m:
            raise ValueError(f"not an mlp address: {text!r} (i<t>.mlp.up|gate|f[k] or i<t>.mlp.down)")
        t, site, k = int(m["t"]), m["site"], self._unit(m["k"])
        if not 0 <= t < self.T:
            raise ValueError(f"tick {t} is outside 0..{self.T - 1}")
        if (site == "down") != (k is None):
            raise ValueError("up/gate/f take a unit [k]; down takes none")
        return t, site, k

    def read(self, text: str) -> dict:
        t, site, k = self._parse(text)
        c = self.cap_
        if site == "gate":
            g = float(c.gate[0, t, k])
            return {"address": text, "value": g, "authority": "measured"}
        bits = {"up": lambda: c.up[0, t, k], "f": lambda: c.gate[0, t, k] * c.up[0, t, k].float(),
                "down": lambda: c.down[0, t]}[site]()
        v, v2, fl = words_of((bits != 0).long())
        return {"address": text, "value": v, "value2": v2, "flags": fl, "nonzero_bits": int((bits != 0).sum()),
                "authority": "measured"}

    def print(self, text: str) -> str:
        r = self.read(text)
        if "value2" not in r:
            return f"{r['address']} = {r['value']:g} [{r['authority']}]"
        return (f"{r['address']} = value {r['value']:#x} value2 {r['value2']:#x} flags {r['flags']:#08b} "
                f"({r['nonzero_bits']} bits on) [{r['authority']}]")

    def attribute(self, t: int) -> dict:
        c = self.cap_
        r = attribute(c.f()[0, t:t + 1], c.down[0, t:t + 1])
        name = lambda i: self.names[i] if i >= 0 else None
        return {"tick": t, "named_contribution": name(int(r["named_contribution"][0])),
                "named_ablation": name(int(r["named_ablation"][0])),
                "contribution": r["contribution"][0].tolist(), "ablation": r["ablation"][0].tolist()}

    def whatif(self, text: str, value: int) -> dict:
        """Set ``i<t>.mlp.gate[k]`` to ``value`` during tick t. Sealed first: the dispatch output at t,
        ``down' = down + (value - gate_k) * up_k`` in the machine's int64 words (exact, derived). Then
        native: the run re-executes with the edit; the record compares down at t and y."""
        t, site, k = self._parse(text)
        if site != "gate":
            raise ValueError("whatif edits a gate unit: i<t>.mlp.gate[k]")
        c = self.cap_
        g0 = int(c.gate[0, t, k])
        up_w = words_of(c.up[0, t, k].long())
        down_w = words_of(c.down[0, t].long())
        M = (1 << 64) - 1
        dv = int(value) - g0
        pred = ((down_w[0] + dv * up_w[0]) & M, (down_w[1] + dv * up_w[1]) & M,
                sum(((((down_w[2] >> i) & 1) + dv * ((up_w[2] >> i) & 1)) != 0) << i for i in range(6)))
        ident = (f"x86:{self.binary.name}:{self.binary.image_sha256[:12]}",
                 {"kind": "x86-64-binary", "program": self.binary.name, "image_sha256": self.binary.image_sha256},
                 {"backend": "torchx86", "dispatch": "hard"})
        hx = lambda w: f"value {w[0]:#x} value2 {w[1]:#x} flags {w[2]:#08b}"
        pbody = {"schema": "mdb-machine-whatif-v1", "edit": {"address": text, "value": f"{int(value)}"},
                 "consumer": "i<t>.mlp.down, then y = last write bytes + exit code",
                 "prediction": {"method": "down + (value - gate_k) * up_k (exact, int64 words)",
                                "down": hx(pred)},
                 "sealed_prediction": "sealed before the native lane runs"}
        sealed = seal_record(session_id=ident[0], kind="machine-whatif", model_identity=ident[1],
                             execution=ident[2], authority="derived-exact", body=pbody)
        e = capture(self.binary, [self.input], device=self.device, cap=self.cap, edit=(t, k, int(value)))
        native = words_of(e.down[0, t].long())
        base, new = self.cap_.result, e.result
        T_new = int(e.live[0].sum())
        pcb, pcn = self.cap_.pc[0], e.pc[0]
        n = min(self.T, T_new)
        diff = torch.nonzero(pcb[:n] != pcn[:n]).flatten()
        ctrl = int(diff[0]) if len(diff) else (None if T_new == self.T else n)
        y_changed = (base.y[0], int(base.exit_code[0])) != (new.y[0], int(new.exit_code[0])) \
            or new.stop_kind(0) != "exit"
        from .session import verdict_of  # noqa: PLC0415

        body = {"schema": "mdb-machine-whatif-result-v1", "whatif": sealed["id"],
                "verdict": verdict_of(new.stop_kind(0), ctrl is not None, y_changed),
                "down_native": hx(native), "down_predicted": hx(pred),
                "down_prediction": "confirmed" if native == pred else "not confirmed",
                "total_effect": "effect" if y_changed else "no effect",
                "frozen_effect": "not measured", "path_diverged_at": f"{ctrl}", "stop": new.stop_kind(0),
                "y": new.y[0].hex(), "exit_code": f"{int(new.exit_code[0])}"}
        res = seal_record(session_id=ident[0], kind="machine-whatif-result", model_identity=ident[1],
                          execution=ident[2], authority="measured", body=body)
        return {**body, "records": [sealed["id"], res["id"]]}


__all__ = ["MlpRun", "Capture", "attribute", "attribute_capture", "bits_of", "capture", "classes",
           "decoded_class", "purity", "sae_encode", "train_topk_sae", "usage", "value_split", "words_of"]
