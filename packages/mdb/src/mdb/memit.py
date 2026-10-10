"""MEMIT (Meng et al., ICLR 2023) and AlphaEdit (Fang et al., ICLR 2025) as removable MDB overlays.

Port of the reference algorithm (github.com/kmeng01/memit: ``compute_z``, ``compute_ks``,
``execute_memit``) with EasyEdit's Qwen2.5 hyperparameters as defaults:

1. Target values. For each request, optimize a residual delta added to the output of the last
   edit layer at the subject's last token, over the bare prompt plus model-generated context
   prefixes. The loss is the target NLL, plus ``kl_factor`` x KL on ``"{subject} is a"`` (read at
   the subject's last token), plus weight decay. The delta is norm-clamped to
   ``clamp_norm_factor`` x the original hidden norm. Target ``z = h_L + delta``.
2. Spread over layers. For each edit layer ``l`` in ascending order:
   - the keys are the ``down_proj`` input at the subject's last token, averaged over contexts,
     with the lower layers' updates already applied;
   - the residual is ``(z - h_L) / (remaining layers)``;
   - the update is ``dW = R K^T (lambda C0 + K K^T)^-1``, with ``C0 = E[k k^T]`` over generic text.

AlphaEdit (reference: github.com/jianghoucheng/AlphaEdit, ``AlphaEdit_main.py`` and ``get_project``)
uses the same targets and keys. Its update is projected onto the near-null space of the
preserved-knowledge key covariance, so preserved keys are (approximately) untouched:
``P = U_0 U_0^T`` over singular directions of ``C0`` with singular value ``< nullspace_threshold``,
and ``dW = X^T`` with ``X = solve(P (K K^T + C_prev) + L2 I, P K R^T)``. ``C_prev`` accumulates the
keys of earlier edit batches (sequential editing).

Deviations from the reference, by design:
- ``dW`` is applied as a forward hook on ``down_proj``, so removal is exact undo; the reference
  writes the weight in place.
- ``C0`` is estimated from a caller-supplied corpus and token budget; the reference uses 100k
  Wikipedia records.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from mdb.mlp_memory import MLPMemoryError, _tensor_sha, decoder_layers
from mdb.model_os_contracts import content_digest

MEMIT_SCHEMA = "mdb-memit-patch-v1"


@dataclass
class MEMITHyperParams:
    """Defaults: EasyEdit ``hparams/MEMIT/qwen2.5-7b.yaml`` (Qwen2.5 has 28 layers at 1.5B and 7B)."""
    layers: tuple = (4, 5, 6, 7, 8)
    v_num_grad_steps: int = 25
    v_lr: float = 0.5
    v_weight_decay: float = 1e-3
    clamp_norm_factor: float = 4.0
    kl_factor: float = 0.0625
    mom2_update_weight: float = 15000.0
    context_template_length_params: tuple = ((5, 10), (10, 10))
    seed: int = 0


@dataclass
class AlphaEditHyperParams(MEMITHyperParams):
    """EasyEdit ``hparams/AlphaEdit/qwen2.5-7b.yaml``: MEMIT's settings plus the projection terms."""
    nullspace_threshold: float = 2e-2
    L2: float = 1.0


# ----------------------------------------------------------------------------------- helpers
def _ids(tok, text):
    return list(tok(text, add_special_tokens=False).input_ids)


def subject_last_index(tok, template, subject):
    """Index of the subject's last token in ``template.format(subject)`` (reference "subject_last")."""
    prefix = template[: template.index("{}")]
    return len(_ids(tok, prefix + subject)) - 1


def _batch(rows, pad_id, device):
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, device=device)
        mask[i, : len(r)] = 1
    return ids, mask


def _pad_id(tok):
    pad = getattr(tok, "pad_token_id", None)
    return pad if pad is not None else getattr(tok, "eos_token_id", 0) or 0


@torch.no_grad()
def generate_context_templates(model, tok, length_params=((5, 10), (10, 10)), seed=0):
    """Reference context templates: ``["{}"]`` plus sampled prefixes (top-k 5) ending in ``". {}"``."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    seeds = ["The", "Therefore", "Because", "I", "You"]
    out = ["{}"]
    for length, n_gen in length_params:
        for prompt in seeds:
            ids = torch.tensor([_ids(tok, prompt)], device=device)
            for _ in range(max(1, n_gen // len(seeds))):
                gen = model.generate(ids, do_sample=True, top_k=5, max_length=length,
                                     pad_token_id=_pad_id(tok), attention_mask=torch.ones_like(ids))
                text = tok.decode(gen[0], skip_special_tokens=True)
                out.append(text.replace("{", " ").replace("}", " ") + ". {}")
    return out


class _DownProjHooks:
    """Internal (unguarded) down_proj deltas used while fitting; also captures down_proj inputs."""

    def __init__(self, model, deltas, capture_layers=()):
        self.handles, self.inputs = [], {}
        stack = decoder_layers(model)
        for layer in sorted(set(deltas) | set(capture_layers)):
            def hook(module, inputs, output, layer=layer):
                if layer in capture_layers:
                    self.inputs[layer] = inputs[0].detach()
                if layer in deltas:
                    dW = deltas[layer].to(device=output.device, dtype=torch.float32)
                    return output + (inputs[0].float() @ dW.T).to(output.dtype)
            self.handles.append(stack[layer].mlp.down_proj.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


# ----------------------------------------------------------------------------------- stats
@torch.no_grad()
def layer_second_moments(model, tok, texts, layers, *, n_tokens=1_000_000, max_len=512, batch=8):
    """Uncentered second moment ``E[k k^T]`` of each layer's ``down_proj`` input over ``texts``."""
    device = next(model.parameters()).device
    stack = decoder_layers(model)
    d_ff = stack[layers[0]].mlp.down_proj.in_features
    moments = {layer: torch.zeros(d_ff, d_ff, device=device, dtype=torch.float32) for layer in layers}
    count = 0
    chunks, buf = [], []
    for text in texts:
        buf.extend(_ids(tok, text))
        while len(buf) >= max_len:
            chunks.append(buf[:max_len])
            buf = buf[max_len:]
        if len(chunks) * max_len >= n_tokens:
            break
    chunks = chunks[: max(1, n_tokens // max_len)]
    hooks = _DownProjHooks(model, {}, capture_layers=layers)
    try:
        for i in range(0, len(chunks), batch):
            ids = torch.tensor(chunks[i:i + batch], device=device)
            model(input_ids=ids, use_cache=False)
            for layer in layers:
                k = hooks.inputs[layer].reshape(-1, d_ff).float()
                moments[layer] += k.T @ k
            count += ids.numel()
    finally:
        hooks.remove()
    return {layer: (m / count) for layer, m in moments.items()}, count


# ----------------------------------------------------------------------------------- algorithm
def compute_z(model, tok, request, layer, templates, hp, base_deltas=None):
    """Reference ``compute_z``: target residual at ``layer`` output, subject-last token.

    ``base_deltas`` are earlier patches' ``down_proj`` deltas, active during the optimization.
    """
    device = next(model.parameters()).device
    target = _ids(tok, request["target_new"])
    prompt, subject = request["prompt"], request["subject"]
    rewriting = [c.format(prompt) for c in templates]
    kl_prompts = ["{} is a"]
    rows, lookups, targets = [], [], []
    for p in rewriting:
        ids = _ids(tok, p.format(subject)) + target[:-1]
        rows.append(ids)
        lookups.append(subject_last_index(tok, p, subject))
        targets.append(len(ids))
    for p in kl_prompts:
        rows.append(_ids(tok, p.format(subject)))
        lookups.append(subject_last_index(tok, p, subject))
    ids, mask = _batch(rows, _pad_id(tok), device)
    n_rw = len(rewriting)
    tgt = torch.full(ids.shape, -100, dtype=torch.long, device=device)
    for i, ex_len in enumerate(targets):
        tgt[i, ex_len - len(target): ex_len] = torch.tensor(target, device=device)
    stack = decoder_layers(model)
    delta = torch.zeros(model.config.hidden_size, device=device, requires_grad=True)
    state = {"init": None}

    def edit(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if state["init"] is None:
            state["init"] = hidden[0, lookups[0]].detach().clone()
        hidden = hidden.clone()
        for i, idx in enumerate(lookups):
            hidden[i, idx, :] = hidden[i, idx, :] + delta.to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    handle = stack[layer].register_forward_hook(edit)
    base_hooks = _DownProjHooks(model, base_deltas or {})
    opt = torch.optim.Adam([delta], lr=hp.v_lr)
    kl_init, losses = None, []
    try:
        for it in range(hp.v_num_grad_steps):
            with torch.enable_grad():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                kl_logits = torch.stack([logits[n_rw + i, idx] for i, idx in enumerate(lookups[n_rw:])])
                kl_logp = torch.log_softmax(kl_logits, -1)
                if kl_init is None:
                    kl_init = kl_logp.detach().clone()
                logp = torch.log_softmax(logits[:n_rw], -1)
                t = tgt[:n_rw]
                gathered = torch.gather(logp, 2, torch.where(t != -100, t, 0).unsqueeze(2)).squeeze(2)
                m = (t != -100).float()
                nll = (-(gathered * m).sum(1) / len(target)).mean()
                kl = hp.kl_factor * torch.nn.functional.kl_div(kl_init, kl_logp, log_target=True,
                                                                reduction="batchmean")
                wd = hp.v_weight_decay * (torch.norm(delta) / torch.norm(state["init"]) ** 2)
                loss = nll + kl + wd
                losses.append(float(loss.detach()))
                if float(loss.detach()) < 5e-2 or it == hp.v_num_grad_steps - 1:
                    break
                (grad,) = torch.autograd.grad(loss, [delta])
            opt.zero_grad()
            delta.grad = grad
            opt.step()
            with torch.no_grad():
                max_norm = hp.clamp_norm_factor * state["init"].norm()
                if delta.norm() > max_norm:
                    delta.mul_(max_norm / delta.norm())
    finally:
        handle.remove()
        base_hooks.remove()
    return (state["init"] + delta).detach(), {"losses": losses, "nll_final": float(nll.detach()),
                                               "delta_norm": float(delta.detach().norm()),
                                               "init_norm": float(state["init"].norm())}


@torch.no_grad()
def _keys_and_outputs(model, tok, requests, templates, layer, z_layer, deltas):
    """Mean down_proj input over contexts at subject-last, and bare-prompt z_layer output."""
    device = next(model.parameters()).device
    stack = decoder_layers(model)
    keys, zs = [], []
    for req in requests:
        ks = []
        for c in templates:
            p = c.format(req["prompt"])
            idx = subject_last_index(tok, p, req["subject"])
            hooks = _DownProjHooks(model, deltas, capture_layers=[layer])
            try:
                model(input_ids=torch.tensor([_ids(tok, p.format(req["subject"]))], device=device), use_cache=False)
            finally:
                hooks.remove()
            ks.append(hooks.inputs[layer][0, idx].float())
        keys.append(torch.stack(ks).mean(0))
        out = {}
        h = stack[z_layer].register_forward_hook(
            lambda m, a, o: out.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        hooks = _DownProjHooks(model, deltas)
        try:
            model(input_ids=torch.tensor([_ids(tok, req["prompt"].format(req["subject"]))], device=device),
                  use_cache=False)
        finally:
            h.remove()
            hooks.remove()
        zs.append(out["h"][0, subject_last_index(tok, req["prompt"], req["subject"])].float())
    return torch.stack(keys, 1), torch.stack(zs, 1)


def null_space_projection(second_moment, threshold):
    """AlphaEdit ``get_project``: projector onto singular directions of ``C0`` below ``threshold``."""
    U, S, _ = torch.linalg.svd(second_moment.float(), full_matrices=False)
    small = (S < threshold).nonzero(as_tuple=True)[0]
    U0 = U[:, small]
    return U0 @ U0.T, int(len(small))


def alphaedit_solve(P, K, R, K_prev=None, l2=1.0):
    """Solve ``(P (K K^T + K_prev K_prev^T) + l2 I) X = P K R^T`` without a d_ff x d_ff solve.

    The system matrix is ``l2 I`` plus a rank-n term ``(P J) J^T`` with ``J = [K, K_prev]``, so
    Woodbury gives ``X = (B - P J (l2 I_n + J^T P J)^{-1} J^T B) / l2`` with ``B = P K R^T``.
    Identical to AlphaEdit's dense solve; memory is one projector plus d_ff x n terms.
    """
    J = K if K_prev is None else torch.cat([K, K_prev], 1)
    PJ = P @ J
    B = PJ[:, :K.shape[1]] @ R.T
    small = l2 * torch.eye(J.shape[1], dtype=J.dtype, device=J.device) + J.T @ PJ
    return (B - PJ @ torch.linalg.solve(small, J.T @ B)) / l2


def _locate_and_edit(algorithm, model, tok, requests, second_moments, hp, *, context_templates, base,
                     base_identity, mom2_info, projections):
    if getattr(model, "_mdb_mlp_memory", None) is not None:
        raise MLPMemoryError(f"{algorithm} needs the native model; pass a live patch as ``base`` instead")
    layers = sorted(int(x) for x in hp.layers)
    if base is not None and base.layers != layers:
        raise MLPMemoryError("base patch was fitted on different layers")
    base_deltas = dict(base.deltas) if base is not None else {}
    z_layer = layers[-1]
    templates = context_templates or (base.templates if base is not None and base.templates else
                                      generate_context_templates(model, tok, hp.context_template_length_params,
                                                                 hp.seed))
    diag = {"requests": {}}
    z_list = []
    for req in requests:
        z, d = compute_z(model, tok, req, z_layer, templates, hp, base_deltas=base_deltas)
        z_list.append(z.float())
        diag["requests"][req.get("id", req["subject"])] = d
    zs = torch.stack(z_list, 1)
    new = {}
    if algorithm == "AlphaEdit" and projections is None:
        projections, diag["null_space_dims"] = {}, {}
        for layer in layers:
            projections[layer], diag["null_space_dims"][str(layer)] = null_space_projection(
                second_moments[layer].to(zs.device), hp.nullspace_threshold)
    for i, layer in enumerate(layers):
        live = {l: base_deltas.get(l, 0) + new.get(l, 0) for l in set(base_deltas) | set(new)}
        K, cur = _keys_and_outputs(model, tok, requests, templates, layer, z_layer, live)
        K64, R64 = K.double(), ((zs - cur) / (len(layers) - i)).double()
        if algorithm == "MEMIT":
            C = second_moments[layer].to(K.device).double()  # reference MEMIT: no cache of earlier edits
            adj = torch.linalg.solve(hp.mom2_update_weight * C + K64 @ K64.T, K64)
            new[layer] = (R64 @ adj.T).float().cpu()
        else:
            P = projections[layer].to(K.device).double()
            Kp = base.keys.get(layer) if base is not None else None
            X = alphaedit_solve(P, K64, R64, None if Kp is None else Kp.to(K.device).double(), hp.L2)
            del P
            new[layer] = X.T.float().cpu()
    final = {l: base_deltas.get(l, 0) + new[l] for l in layers}
    keys = {}
    for layer in (layers if algorithm == "AlphaEdit" else ()):  # reference: cache_c += K K^T after the batch
        K, _ = _keys_and_outputs(model, tok, requests, templates, layer, z_layer, final)
        prev = base.keys.get(layer) if base is not None else None
        keys[layer] = K.float().cpu() if prev is None else torch.cat([prev, K.float().cpu()], 1)
    all_requests = (base.requests if base is not None else []) + list(requests)
    patch = MEMITPatch(layers, final, hparams=hp, requests=all_requests, base_identity=base_identity,
                       templates=templates, mom2_info=mom2_info or {}, algorithm=algorithm, keys=keys)
    diag["update_norms"] = {str(l): float(new[l].norm()) for l in layers}
    return patch, diag


def memit(model, tok, requests, second_moments, hp=None, *, context_templates=None, base=None,
          base_identity=None, mom2_info=None):
    """Apply MEMIT to ``requests`` (``{"prompt": "... {} ...", "subject", "target_new"}``).

    Returns a ``MEMITPatch`` (not installed) and diagnostics. The model must be native. Pass an
    earlier patch as ``base`` to edit sequentially on top of it; the result contains both.
    """
    return _locate_and_edit("MEMIT", model, tok, requests, second_moments, hp or MEMITHyperParams(),
                            context_templates=context_templates, base=base, base_identity=base_identity,
                            mom2_info=mom2_info, projections=None)


def alphaedit(model, tok, requests, second_moments, hp=None, *, context_templates=None, base=None,
              base_identity=None, mom2_info=None, projections=None):
    """Apply AlphaEdit (null-space-projected MEMIT) to ``requests``; same contract as ``memit``.

    ``projections`` (``{layer: P}``) may be passed to reuse ``null_space_projection`` results.
    """
    return _locate_and_edit("AlphaEdit", model, tok, requests, second_moments, hp or AlphaEditHyperParams(),
                            context_templates=context_templates, base=base, base_identity=base_identity,
                            mom2_info=mom2_info, projections=projections)


# ----------------------------------------------------------------------------------- overlay
class MEMITPatch:
    """MEMIT weight deltas held as a removable overlay on ``down_proj`` (same API as ``MLPMemory``)."""

    def __init__(self, layers, deltas, *, hparams, requests, base_identity=None, templates=(), mom2_info=None,
                 algorithm="MEMIT", keys=None):
        self.layers = sorted(int(x) for x in layers)
        self.deltas = {int(k): v.float().cpu().contiguous() for k, v in deltas.items()}
        self.hparams = hparams if isinstance(hparams, dict) else asdict(hparams)
        self.requests = [{k: v for k, v in r.items()} for r in requests]
        self.base_identity = base_identity
        self.templates = list(templates)
        self.mom2_info = dict(mom2_info or {})
        self.algorithm = algorithm
        self.keys = {int(k): v.float().cpu().contiguous() for k, v in (keys or {}).items()}

    def manifest(self):
        method = {"MEMIT": "MEMIT (Meng et al. 2023)", "AlphaEdit": "AlphaEdit (Fang et al. 2025)"}[self.algorithm]
        return {"schema": MEMIT_SCHEMA, "method": method, "algorithm": self.algorithm, "layers": self.layers,
                "hparams": json.loads(json.dumps(self.hparams)), "requests": self.requests,
                "base_identity": self.base_identity, "context_templates": self.templates,
                "mom2": self.mom2_info,
                "deltas": {str(k): {"sha256": _tensor_sha(v), "shape": list(v.shape)}
                           for k, v in sorted(self.deltas.items())},
                "keys": {str(k): {"sha256": _tensor_sha(v), "shape": list(v.shape)}
                         for k, v in sorted(self.keys.items())}}

    @property
    def digest(self):
        return content_digest(self.manifest())

    def install(self, model):
        return DownProjInstallation(model, self.deltas, self.digest)

    def save(self, path):
        from safetensors.torch import save_file
        tensors = {f"{k}.dW": v for k, v in self.deltas.items()}
        tensors.update({f"{k}.keys": v for k, v in self.keys.items()})
        save_file(tensors, str(path),
                  metadata={"mdb_memit": json.dumps(self.manifest(), sort_keys=True)})
        return self.digest

    @classmethod
    def load(cls, path, *, expected_digest=None):
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as handle:
            manifest = json.loads(handle.metadata()["mdb_memit"])
            if manifest.get("schema") != MEMIT_SCHEMA:
                raise MLPMemoryError("MEMIT patch schema changed")
            deltas = {int(k): handle.get_tensor(f"{k}.dW") for k in manifest["deltas"]}
            keys = {int(k): handle.get_tensor(f"{k}.keys") for k in manifest.get("keys", {})}
        patch = cls(manifest["layers"], deltas, hparams=manifest["hparams"], requests=manifest["requests"],
                    base_identity=manifest["base_identity"], templates=manifest["context_templates"],
                    mom2_info=manifest["mom2"], algorithm=manifest.get("algorithm", "MEMIT"), keys=keys)
        if patch.manifest() != manifest:
            raise MLPMemoryError("MEMIT patch content does not match its manifest")
        if expected_digest is not None and patch.digest != expected_digest:
            raise MLPMemoryError("MEMIT patch digest mismatch")
        return patch


@dataclass
class DownProjInstallation:
    model: object
    deltas: dict
    digest: str
    _handles: list = field(default_factory=list)

    def __post_init__(self):
        if getattr(self.model, "_mdb_mlp_memory", None) is not None:
            raise MLPMemoryError("another MLP memory overlay is already installed on this model")
        stack = decoder_layers(self.model)
        cache = {}
        for layer, dW in self.deltas.items():
            def hook(module, inputs, output, dW=dW, layer=layer):
                key = (layer, output.device)
                if key not in cache:
                    cache[key] = dW.to(device=output.device, dtype=torch.float32)
                return output + (inputs[0].float() @ cache[key].T).to(output.dtype)
            self._handles.append(stack[layer].mlp.down_proj.register_forward_hook(hook))
        self.model._mdb_mlp_memory = self.digest

    @property
    def active(self):
        return bool(self._handles)

    def remove(self):
        for h in self._handles:
            h.remove()
        if self._handles:
            self.model._mdb_mlp_memory = None
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


def load_overlay(path, *, expected_digest=None):
    """Load either an ``MLPMemory`` or a ``MEMITPatch`` file by its metadata."""
    from safetensors import safe_open

    from mdb.mlp_memory import MLPMemory
    with safe_open(str(Path(path)), framework="pt") as handle:
        meta = handle.metadata() or {}
    if "mdb_memit" in meta:
        return MEMITPatch.load(path, expected_digest=expected_digest)
    return MLPMemory.load(path, expected_digest=expected_digest)
