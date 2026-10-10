"""Linear associative MLP memory: an additive, removable readout on native MLP outputs.

An ``MLPMemory`` adds ``delta_l(x) = normalize(x) @ B_l`` to the output of the MLP at each
selected decoder layer ``l``, at every position, where ``x`` is that MLP's own input. Each
``B_l`` is a weighted ridge readout fitted sequentially from the first selected layer to the
last: layer ``l`` is fitted on keys and outputs captured with the already-fitted layers below
it installed. Edit rows ask the MLP output at one position to become a target (a donor
prompt's native MLP output, or an explicit vector); every other captured row asks for no
change. The model's parameters are never written. Removing the hooks restores the native
model bit for bit.

The readout keeps its regularized inverse Gram matrix and right-hand side in float64, so a
group of rows can be inserted or deleted exactly (Sherman-Morrison-Woodbury) without a refit.
Deletion uses the rows stored at insertion, so it is exact for the stored readout. A
sequential refit can differ from an incremental insert: inserted rows at a layer are captured
under the overlay that already exists below it, while earlier rows keep their original keys.

Measured basis (Qwen2.5-1.5B fp32, 12 country->capital edits at layers 0-4 and 19, job
job-2eb150e954e4): efficacy 11/12, generalization 24/36, locality 1.00, exact undo, and
Sherman-Morrison writes matching a batch refit to about 1e-10. Quadratic (Hebbian) features
did not beat this linear readout at that scale.

Feature maps other than the default linear one are available for keys that must be
conjunctive (fire for subject AND relation, not either alone): ``QuadraticFeatures`` (random
Hebbian products), ``BoundFeatures`` (outer product of fitted subject and relation reader
coordinates: tensor-product binding) and ``PhasorFeatures`` (unit-phasor binding, the
spectral-address rule ``phase(symbol) + phase(role)``). Edits may also give explicit per-layer
output deltas, for example from ``attribute_delta``, which optimizes a relation-specific value
instead of transplanting a donor's whole subject representation.

Measured guidance (Qwen2.5-1.5B, job-75565544f1a2 / job-1243e5c83d95):
- A donor target is a subject swap.
- Zero rows for the edited subject's other relations under a linear key collapse held-out
  behavior: store keys are about 97% additive, so the requirement is an XOR solvable only
  through interaction residue.
- The fix is a linear key with ``attribute_delta`` values trained on a few phrasings plus KL on
  the subject's other relations: 12/12 edits, 23/24 paraphrases, locality 0.97.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import torch

from mdb.model_os_contracts import content_digest

MEMORY_SCHEMA = "mdb-mlp-memory-v1"
ACT_SCHEMA = "mdb-mlp-memory-act-v1"
FEATURES = "linear-normalized-mlp-input"


class MLPMemoryError(ValueError):
    pass


def decoder_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not all(hasattr(layer, "mlp") for layer in layers):
        raise MLPMemoryError("MLP memory needs a decoder exposing model.model.layers[i].mlp")
    return layers


def _features(x):
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _tensor_sha(tensor):
    data = tensor.detach().to("cpu").contiguous().reshape(-1)
    return sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


class FeatureMap:
    """Key feature map ``phi(x)`` of an MLP input; stateless apart from its fixed tensors."""
    kind = "linear"

    def __init__(self, **tensors):
        self.tensors = {k: v.detach().float().cpu().contiguous() for k, v in tensors.items()}
        self._cache = {}

    def spec(self):
        return {"kind": self.kind, "tensors": {k: _tensor_sha(v) for k, v in sorted(self.tensors.items())}}

    def _t(self, name, device):
        key = (name, str(device))
        if key not in self._cache:
            self._cache[key] = self.tensors[name].to(device)
        return self._cache[key]

    def __call__(self, x):
        return _features(x)

    @staticmethod
    def from_spec(kind, tensors):
        cls = {c.kind: c for c in (FeatureMap, QuadraticFeatures, BoundFeatures, PhasorFeatures,
                                   AddressFeatures)}[kind]
        return cls(**tensors)


class QuadraticFeatures(FeatureMap):
    """``(A x^) * (G x^) / sqrt(m)``: the random Hebbian product construction."""
    kind = "quadratic"

    @classmethod
    def random(cls, d, m, seed):
        g = torch.Generator().manual_seed(int(seed))
        return cls(A=torch.randn(m, d, generator=g), G=torch.randn(m, d, generator=g))

    def __call__(self, x):
        xh = _features(x)
        A, G = self._t("A", xh.device), self._t("G", xh.device)
        return (xh @ A.T) * (xh @ G.T) / A.shape[0] ** 0.5


class _Readers(FeatureMap):
    def _coords(self, x):
        xh = _features(x) - self._t("mu", x.device)
        return xh @ self._t("Ps", x.device).T, xh @ self._t("Pr", x.device).T


class BoundFeatures(_Readers):
    """Tensor-product binding ``[s, 1] (x) [r, 1]`` of whitened subject/relation reader coordinates."""
    kind = "bound"

    def __call__(self, x):
        s, r = self._coords(x)
        one = torch.ones_like(s[..., :1])
        s1, r1 = torch.cat([s, one], -1), torch.cat([r, one], -1)
        out = (s1.unsqueeze(-1) * r1.unsqueeze(-2)).flatten(-2)
        return out / out.shape[-1] ** 0.5


class PhasorFeatures(_Readers):
    """Spectral-address binding: unit phasors ``exp(i (theta_symbol + theta_role))``.

    ``theta_symbol = arg(E_s s)`` and ``theta_role = arg(E_r r)`` for fixed complex expansions of
    the reader coordinates, so binding is elementwise phase addition (``SpectralLogicalAddress``).
    """
    kind = "phasor"

    @staticmethod
    def expansions(k_subject, k_relation, k, seed):
        g = torch.Generator().manual_seed(int(seed))
        return {name: torch.randn(k, n, generator=g) for name, n in
                (("Es_re", k_subject), ("Es_im", k_subject), ("Er_re", k_relation), ("Er_im", k_relation))}

    def __call__(self, x):
        s, r = self._coords(x)
        d = x.device
        ta = torch.atan2(s @ self._t("Es_im", d).T, s @ self._t("Es_re", d).T)
        tb = torch.atan2(r @ self._t("Er_im", d).T, r @ self._t("Er_re", d).T)
        theta = ta + tb
        return torch.cat([torch.cos(theta), torch.sin(theta)], -1) / theta.shape[-1] ** 0.5


class AddressFeatures(FeatureMap):
    """Spectral address of calibrated subject and role posteriors, bound by phase addition.

    ``p_s = softmax(W_s x^ + b_s)`` and ``p_r = softmax(W_r x^ + b_r)`` are class posteriors.
    Each class has canonical phases from the spectral-address ABI hash and an amplitude (0 for
    the ``<none>`` class, so "no subject" or "no role" means no address). The superposed codes
    ``a = sum_c p_s[c] amp_s[c] exp(2 pi i theta_s[c])`` and ``b`` (same for roles) are bound
    elementwise, ``psi = a * b``: phase addition ``theta_symbol + theta_role`` as in
    ``SpectralLogicalAddress.phases``. In real coordinates this is the gated bilinear (Hebbian)
    product of two reader outputs. Phase carries the address and magnitude carries reader
    confidence. ``mode`` 0 = phasor, 1 = unit-modulus phasor (magnitude discarded, ablation),
    2 = exact one-hot tensor product ``p_s (x) p_r`` without phase compression (ablation).
    """
    kind = "address"

    def __call__(self, x):
        xh = _features(x)
        d = xh.device
        ps = torch.softmax(xh @ self._t("Ws", d).T + self._t("bs", d), -1) * self._t("amp_s", d)
        pr = torch.softmax(xh @ self._t("Wr", d).T + self._t("br", d), -1) * self._t("amp_r", d)
        mode = int(self.tensors["mode"].item())
        if mode == 2:
            out = (ps.unsqueeze(-1) * pr.unsqueeze(-2)).flatten(-2)
            return out
        ts, tr = 2 * torch.pi * self._t("theta_s", d), 2 * torch.pi * self._t("theta_r", d)
        ar, ai = ps @ torch.cos(ts), ps @ torch.sin(ts)
        br, bi = pr @ torch.cos(tr), pr @ torch.sin(tr)
        if mode == 1:
            na = torch.sqrt(ar ** 2 + ai ** 2).clamp_min(1e-12)
            nb = torch.sqrt(br ** 2 + bi ** 2).clamp_min(1e-12)
            ar, ai, br, bi = ar / na, ai / na, br / nb, bi / nb
        out = torch.cat([ar * br - ai * bi, ar * bi + ai * br], -1)
        return out / ts.shape[-1] ** 0.5 * 2 ** 0.5


LINEAR = FeatureMap()


@dataclass
class LayerReadout:
    """Ridge readout ``B = P @ C`` with ``P = (Phi^T W Phi + alpha I)^-1`` and ``C = Phi^T W T``."""
    layer: int
    alpha: float
    P: torch.Tensor
    C: torch.Tensor
    B: torch.Tensor | None = None
    rows: dict = field(default_factory=dict)  # group -> (Phi, T, w), float64 on CPU

    def __post_init__(self):
        if self.B is None:
            self.B = (self.P @ self.C).float()

    def _woodbury(self, Phi, T, w, sign):
        U = Phi.to(self.P)
        PU = self.P @ U.T
        inner = torch.diag(1.0 / w.to(self.P)) + sign * (U @ PU)
        self.P = self.P - sign * PU @ torch.linalg.solve(inner, PU.T)
        self.P = 0.5 * (self.P + self.P.T)
        self.C = self.C + sign * U.T @ (T.to(self.C) * w.to(self.C).view(-1, 1))
        self.B = (self.P @ self.C).float()

    def insert(self, group, Phi, T, w):
        if group in self.rows:
            raise MLPMemoryError(f"memory group already present at layer {self.layer}: {group}")
        self._woodbury(Phi, T, w, +1.0)
        self.rows[group] = (Phi.double().cpu(), T.double().cpu(), w.double().cpu())

    def delete(self, group):
        if group not in self.rows:
            raise MLPMemoryError(f"memory group not present at layer {self.layer}: {group}")
        self._woodbury(*self.rows.pop(group), -1.0)


class MLPMemory:
    """Fitted per-layer linear readouts plus the rows they were fitted on."""

    def __init__(self, layers, readouts=None, *, base_identity=None, alpha=1e-4, edit_weight=1.0,
                 features=None):
        self.layers = sorted(int(layer) for layer in layers)
        if len(set(self.layers)) != len(self.layers) or not self.layers:
            raise MLPMemoryError("MLP memory needs distinct layers")
        features = features or {}
        if not isinstance(features, dict):
            features = {layer: features for layer in self.layers}
        self.features = {layer: features.get(layer, LINEAR) for layer in self.layers}
        self.readouts: dict[int, LayerReadout] = dict(readouts or {})
        self.base_identity = base_identity
        self.alpha, self.edit_weight = float(alpha), float(edit_weight)
        self.edits: dict[str, dict] = {}

    # -- identity -------------------------------------------------------------------------
    def manifest(self):
        return {"schema": MEMORY_SCHEMA, "features": FEATURES, "layers": self.layers,
                "base_identity": self.base_identity, "alpha": self.alpha,
                "edit_weight": self.edit_weight, "edits": self.edits,
                "readouts": {str(layer): {"alpha": r.alpha, "B_sha256": _tensor_sha(r.B),
                                          "shape": list(r.B.shape), "groups": sorted(r.rows)}
                             for layer, r in sorted(self.readouts.items())},
                **({"feature_maps": {str(layer): f.spec() for layer, f in sorted(self.features.items())}}
                   if any(f.kind != "linear" for f in self.features.values()) else {})}

    @property
    def digest(self):
        return content_digest(self.manifest())

    # -- application ----------------------------------------------------------------------
    def _deltas(self, below=None):
        return {layer: (self.features[layer], r.B) for layer, r in self.readouts.items()
                if below is None or layer < below}

    def install(self, model):
        if set(self.readouts) != set(self.layers):
            raise MLPMemoryError("MLP memory is not fitted at every selected layer")
        return Installation(model, self._deltas(), self.digest)

    # -- fitting --------------------------------------------------------------------------
    @classmethod
    @torch.inference_mode()
    def fit(cls, model, edits, preserve=(), *, layers, alpha=1e-4, edit_weight=1.0,
            base_identity=None, features=None):
        """Fit sequentially, lowest selected layer first.

        ``edits``: ``{"id", "ids", "position"}`` plus one of ``"donor_ids"``/``"donor_position"``
        (target = the donor's native MLP output there), ``"targets": {layer: vector}`` (absolute
        MLP output) or ``"deltas": {layer: vector}`` (change added to the MLP output).
        ``features``: a ``FeatureMap`` or ``{layer: FeatureMap}``; default linear.
        ``preserve``: token-id lists whose every row should keep its native MLP output.
        ``alpha`` is relative to the mean diagonal of each layer's weighted Gram matrix and is
        frozen into the readout at fit time.
        """
        memory = cls(layers, base_identity=base_identity, alpha=alpha, edit_weight=edit_weight,
                     features=features)
        edits = memory._normalize(edits)
        donors = memory._donor_targets(model, edits)
        for layer in memory.layers:
            groups = memory._rows(model, layer, edits, donors, preserve)
            Phi = torch.cat([g[0] for g in groups.values()])
            T = torch.cat([g[1] for g in groups.values()])
            w = torch.cat([g[2] for g in groups.values()])
            gram = Phi.T @ (Phi * w.view(-1, 1))
            a = alpha * float(torch.diagonal(gram).mean())
            eye = torch.eye(gram.shape[0], dtype=gram.dtype)
            P = torch.linalg.inv(gram + a * eye)
            P = 0.5 * (P + P.T)
            readout = LayerReadout(layer, a, P, Phi.T @ (T * w.view(-1, 1)))
            readout.rows = groups
            memory.readouts[layer] = readout
        for edit in edits:
            memory.edits[edit["id"]] = memory._edit_record(edit)
        return memory

    @torch.inference_mode()
    def insert(self, model, edits, preserve=()):
        """Add edit (and optional preservation) rows without refitting, layer by layer."""
        edits = self._normalize(edits)
        if any(edit["id"] in self.edits for edit in edits):
            raise MLPMemoryError("edit id already present")
        donors = self._donor_targets(model, edits)
        offset = len([g for g in self.readouts[self.layers[0]].rows if g.startswith("preserve:")])
        for layer in self.layers:
            groups = self._rows(model, layer, edits, donors, preserve, preserve_offset=offset)
            for group, (Phi, T, w) in groups.items():
                self.readouts[layer].insert(group, Phi, T, w)
        for edit in edits:
            self.edits[edit["id"]] = self._edit_record(edit)

    def delete(self, edit_id):
        if edit_id not in self.edits:
            raise MLPMemoryError(f"unknown edit id: {edit_id}")
        for readout in self.readouts.values():
            readout.delete(f"edit:{edit_id}")
        del self.edits[edit_id]

    # -- internals ------------------------------------------------------------------------
    def _normalize(self, edits):
        out, seen = [], set()
        for edit in edits:
            item = dict(edit)
            if not isinstance(item.get("id"), str) or item["id"] in seen:
                raise MLPMemoryError("each edit needs a distinct string id")
            seen.add(item["id"])
            ids = [int(t) for t in item["ids"]]
            if not 0 <= int(item["position"]) < len(ids):
                raise MLPMemoryError(f"edit {item['id']} position outside its prompt")
            kinds = [k for k in ("donor_ids", "targets", "deltas") if k in item]
            if len(kinds) != 1:
                raise MLPMemoryError(f"edit {item['id']} needs exactly one of donor_ids, targets or deltas")
            if kinds[0] != "donor_ids" and set(map(int, item[kinds[0]])) != set(self.layers):
                raise MLPMemoryError(f"edit {item['id']} {kinds[0]} must cover every selected layer")
            out.append({**item, "ids": ids, "position": int(item["position"])})
        return out

    def _edit_record(self, edit):
        record = {"ids_sha256": content_digest(edit["ids"]), "position": edit["position"]}
        if "donor_ids" in edit:
            record.update(donor_ids_sha256=content_digest([int(t) for t in edit["donor_ids"]]),
                          donor_position=int(edit["donor_position"]))
        else:
            kind = "targets" if "targets" in edit else "deltas"
            record[f"{kind}_sha256"] = {str(k): _tensor_sha(torch.as_tensor(v).float())
                                        for k, v in sorted(edit[kind].items(), key=lambda i: int(i[0]))}
        return record

    def _donor_targets(self, model, edits):
        targets = {}
        for edit in edits:
            if "targets" in edit or "deltas" in edit:
                kind = "targets" if "targets" in edit else "deltas"
                targets[edit["id"]] = (kind, {int(k): torch.as_tensor(v).double().cpu()
                                              for k, v in edit[kind].items()})
                continue
            _, outs = capture(model, edit["donor_ids"], self.layers, {})
            targets[edit["id"]] = ("targets", {layer: outs[layer][int(edit["donor_position"])]
                                               for layer in self.layers})
        return targets

    def _rows(self, model, layer, edits, donors, preserve, preserve_offset=0):
        deltas = self._deltas(below=layer)
        groups = {}
        fmap = self.features[layer]
        for edit in edits:
            ins, outs = capture(model, edit["ids"], [layer], deltas)
            Phi, raw = fmap(ins[layer]).double().cpu(), outs[layer]
            T = torch.zeros(raw.shape, dtype=torch.float64)
            w = torch.ones(len(raw), dtype=torch.float64)
            kind, values = donors[edit["id"]]
            T[edit["position"]] = values[layer] - (raw[edit["position"]] if kind == "targets" else 0)
            w[edit["position"]] = self.edit_weight
            groups[f"edit:{edit['id']}"] = (Phi, T, w)
        for index, ids in enumerate(preserve):
            ins, outs = capture(model, ids, [layer], deltas)
            Phi = fmap(ins[layer]).double().cpu()
            groups[f"preserve:{preserve_offset + index}"] = (
                Phi, torch.zeros_like(outs[layer]), torch.ones(len(Phi), dtype=torch.float64))
        return groups

    # -- persistence ----------------------------------------------------------------------
    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        tensors, index = {}, {}
        for layer, fmap in self.features.items():
            for name, tensor in fmap.tensors.items():
                tensors[f"{layer}.feat.{name}"] = tensor.clone()
        for layer, r in self.readouts.items():
            tensors[f"{layer}.P"], tensors[f"{layer}.C"] = r.P.contiguous(), r.C.contiguous()
            tensors[f"{layer}.B"] = r.B.contiguous()
            index[str(layer)] = {"alpha": r.alpha, "groups": sorted(r.rows)}
            for n, group in enumerate(sorted(r.rows)):
                for name, tensor in zip("PTw", r.rows[group]):
                    tensors[f"{layer}.rows.{n}.{name}"] = tensor.contiguous()
        meta = {"manifest": self.manifest(), "index": index}
        save_file(tensors, str(path), metadata={"mdb_mlp_memory": json.dumps(meta, sort_keys=True)})
        return self.digest

    @classmethod
    def load(cls, path, *, expected_digest=None):
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as handle:
            meta = json.loads(handle.metadata()["mdb_mlp_memory"])
            manifest = meta["manifest"]
            if manifest.get("schema") != MEMORY_SCHEMA or manifest.get("features") != FEATURES:
                raise MLPMemoryError("MLP memory file schema changed")
            fmaps = {}
            for key, spec in manifest.get("feature_maps", {}).items():
                fmaps[int(key)] = FeatureMap.from_spec(spec["kind"], {
                    name: handle.get_tensor(f"{key}.feat.{name}") for name in spec["tensors"]})
            memory = cls(manifest["layers"], base_identity=manifest["base_identity"],
                         alpha=manifest["alpha"], edit_weight=manifest["edit_weight"], features=fmaps)
            memory.edits = manifest["edits"]
            for key, info in meta["index"].items():
                layer = int(key)
                r = LayerReadout(layer, info["alpha"], handle.get_tensor(f"{layer}.P"),
                                 handle.get_tensor(f"{layer}.C"), handle.get_tensor(f"{layer}.B"))
                r.rows = {g: tuple(handle.get_tensor(f"{layer}.rows.{n}.{name}") for name in "PTw")
                          for n, g in enumerate(info["groups"])}
                memory.readouts[layer] = r
        if memory.manifest() != manifest:
            raise MLPMemoryError("MLP memory file content does not match its manifest")
        if expected_digest is not None and memory.digest != expected_digest:
            raise MLPMemoryError("MLP memory digest mismatch")
        return memory


class Installation:
    """Forward hooks adding the readout to native MLP outputs. ``remove()`` is exact undo."""

    def __init__(self, model, deltas, digest):
        if getattr(model, "_mdb_mlp_memory", None) is not None:
            raise MLPMemoryError("another MLP memory overlay is already installed on this model")
        self.digest, self._model = digest, model
        self._handles = []
        layers = decoder_layers(model)
        cache = {}
        for layer, (fmap, B) in deltas.items():
            def hook(module, inputs, output, B=B, fmap=fmap, layer=layer):
                key = (output.device, output.dtype)
                if (layer, key) not in cache:
                    cache[(layer, key)] = B.to(device=output.device, dtype=torch.float32)
                delta = fmap(inputs[0]) @ cache[(layer, key)]
                return output + delta.to(output.dtype)
            self._handles.append(layers[layer].mlp.register_forward_hook(hook))
        model._mdb_mlp_memory = digest

    @property
    def active(self):
        return bool(self._handles)

    def remove(self):
        for handle in self._handles:
            handle.remove()
        if self._handles:
            self._model._mdb_mlp_memory = None
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


@contextmanager
def _hooks(model, layers, deltas, ins, outs):
    stack = decoder_layers(model)
    handles = []
    try:
        for layer in sorted(set(layers) | set(deltas)):
            def hook(module, inputs, output, layer=layer):
                if layer in layers:
                    ins[layer] = inputs[0][0].detach().double().cpu()
                    outs[layer] = output[0].detach().double().cpu()
                if layer in deltas:
                    fmap, B = deltas[layer]
                    B = B.to(device=output.device, dtype=torch.float32)
                    return output + (fmap(inputs[0]) @ B).to(output.dtype)
            handles.append(stack[layer].mlp.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def capture(model, ids, layers, deltas):
    """Run one prompt; return raw (pre-delta) MLP inputs and outputs at ``layers``."""
    if getattr(model, "_mdb_mlp_memory", None) is not None:
        raise MLPMemoryError("fit and capture need the native model; remove the live overlay first")
    ins, outs = {}, {}
    device = next(model.parameters()).device
    with _hooks(model, list(layers), deltas, ins, outs):
        model(input_ids=torch.tensor([list(ids)], device=device), use_cache=False)
    return ins, outs


@torch.inference_mode()
def probe_digest(model, ids):
    """Digest of final-position logits for a fixed probe; used to witness exact undo."""
    device = next(model.parameters()).device
    logits = model(input_ids=torch.tensor([list(ids)], device=device), use_cache=False).logits
    return _tensor_sha(logits[0, -1])


@torch.inference_mode()
def fit_address_readers(model, labelled, layers, *, k_subject=32, k_relation=None, kind="bound",
                        phasor_k=256, seed=0):
    """Fit subject and relation readers from a balanced subject x relation prompt grid.

    ``labelled``: ``{"ids", "position", "subject", "relation"}``, one prompt per (subject,
    relation) pair, read at ``position`` (normally the subject's last token). The readers are
    the leading principal directions of the centered subject means and relation means of the
    normalized MLP input (two-way ANOVA main effects), whitened to unit variance. Returns
    ``({layer: FeatureMap}, {layer: stats})``; stats split the key variance into subject,
    relation and interaction (non-additive) parts.
    """
    subjects = sorted({item["subject"] for item in labelled})
    relations = sorted({item["relation"] for item in labelled})
    cells = {(item["subject"], item["relation"]) for item in labelled}
    if len(cells) != len(labelled) or len(cells) != len(subjects) * len(relations):
        raise MLPMemoryError("address readers need exactly one prompt per subject x relation")
    k_relation = min(k_relation or len(relations) - 1, len(relations) - 1)
    k_subject = min(k_subject, len(subjects) - 1)
    rows = {layer: {} for layer in layers}
    for item in labelled:
        ins, _ = capture(model, item["ids"], layers, {})
        for layer in layers:
            rows[layer][(item["subject"], item["relation"])] = _features(ins[layer][item["position"]])
    maps, stats = {}, {}
    for layer in layers:
        X = torch.stack([torch.stack([rows[layer][(s, r)] for r in relations]) for s in subjects])
        mu = X.mean((0, 1))
        ms, mr = X.mean(1) - mu, X.mean(0) - mu
        inter = X - mu - ms[:, None] - mr[None]
        total = float(((X - mu) ** 2).sum())
        stats[layer] = {"subject": float((ms ** 2).sum() * len(relations)) / total,
                        "relation": float((mr ** 2).sum() * len(subjects)) / total,
                        "interaction": float((inter ** 2).sum()) / total,
                        "mean_key_norm_centered": float((X - mu).norm(dim=-1).mean()),
                        "mean_interaction_norm": float(inter.norm(dim=-1).mean())}

        def reader(M, k):
            _, S, Vt = torch.linalg.svd(M, full_matrices=False)
            sd = (S[:k] / M.shape[0] ** 0.5).clamp_min(1e-12)
            return Vt[:k] / sd[:, None]
        Ps, Pr = reader(ms, k_subject), reader(mr, k_relation)
        tensors = {"mu": mu, "Ps": Ps, "Pr": Pr}
        if kind == "bound":
            maps[layer] = BoundFeatures(**tensors)
        elif kind == "phasor":
            maps[layer] = PhasorFeatures(**tensors, **PhasorFeatures.expansions(
                k_subject, k_relation, phasor_k, seed + layer))
        else:
            raise MLPMemoryError(f"unknown reader feature kind: {kind}")
    return maps, stats


def attribute_delta(model, ids, position, target_token, keep=(), *, layers, steps=60, lr=0.05,
                    kl_weight=1.0, l2=1e-3, also=(), init=None):
    """Optimize per-layer MLP-output deltas at ``position`` that make ``target_token`` the next
    token of ``ids`` while keeping the next-token distribution of each ``keep`` prompt
    (``{"ids", "position"}``, the same deltas injected at its own position). ``also`` adds more
    ``{"ids", "position"}`` prompts whose next token must also become ``target_token`` (other
    phrasings of the same relation, for generalization). ``init`` (``{layer: delta}``) starts the
    optimization from given deltas, for example a donor transplant difference. Returns
    ``({layer: delta}, stats)``. Only the deltas receive gradients.
    """
    if getattr(model, "_mdb_mlp_memory", None) is not None:
        raise MLPMemoryError("attribute deltas need the native model; remove the live overlay first")
    stack = decoder_layers(model)
    device = next(model.parameters()).device
    d = model.config.hidden_size
    layers = sorted(int(layer) for layer in layers)
    start = (torch.stack([torch.as_tensor(init[layer]).float() for layer in layers]).to(device)
             if init is not None else torch.zeros(len(layers), d, device=device))
    delta = start.clone().requires_grad_(True)
    targets = [{"ids": ids, "position": position}, *also]
    state = {"pos": None}
    handles = []

    def hook(module, inputs, output, index):
        if state["pos"] is None:
            return None
        out = output.clone()
        out[:, state["pos"], :] = out[:, state["pos"], :] + delta[index].to(out.dtype)
        return out
    for i, layer in enumerate(layers):
        handles.append(stack[layer].mlp.register_forward_hook(
            lambda m, a, o, i=i: hook(m, a, o, i)))

    def last_logp(token_ids, pos):
        state["pos"] = pos
        logits = model(input_ids=torch.tensor([list(token_ids)], device=device), use_cache=False).logits
        return torch.log_softmax(logits[0, -1].float(), -1)
    try:
        with torch.no_grad():
            native = [last_logp(k["ids"], None) for k in keep]
        opt = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            with torch.enable_grad():
                nll = sum(-last_logp(t["ids"], t["position"])[target_token] for t in targets) / len(targets)
                kl = sum(torch.sum(n.exp() * (n - last_logp(k["ids"], k["position"])))
                         for n, k in zip(native, keep)) / max(1, len(keep))
                loss = nll + kl_weight * kl + l2 * (delta ** 2).sum()
                (grad,) = torch.autograd.grad(loss, [delta])
            opt.zero_grad()
            delta.grad = grad
            opt.step()
        with torch.no_grad():
            p_target = float(last_logp(ids, position)[target_token].exp())
            p_also = [float(last_logp(t["ids"], t["position"])[target_token].exp()) for t in also]
            kl_end = float(sum(torch.sum(n.exp() * (n - last_logp(k["ids"], k["position"])))
                               for n, k in zip(native, keep)) / max(1, len(keep))) if keep else 0.0
    finally:
        for h in handles:
            h.remove()
    out = {layer: delta[i].detach().double().cpu() for i, layer in enumerate(layers)}
    return out, {"p_target": p_target, "p_also": p_also, "keep_kl": kl_end,
                 "delta_norms": {str(layer): float(v.norm()) for layer, v in out.items()}}


NONE = "<none>"


def _softmax_regression(X, y, n_classes, weights, l2, steps):
    """Weighted multinomial logistic regression on z-scored features, folded back to raw ``X``.

    Unit-normalized keys share a large common component, so class differences are small in raw
    coordinates and an L2 penalty on raw weights collapses the classifier to its prior. Fitting
    on per-dimension standardized features avoids that; the returned ``W, b`` act on raw ``X``.
    """
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    X = (X - mu) / sd
    W = torch.zeros(n_classes, X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(n_classes, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")
    wsum = weights.sum()

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(X @ W.T + b, y, reduction="none") * weights).sum() / wsum
        loss = loss + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        loss = float(opt.step(closure))
    W, b = W.detach() / sd, b.detach() - (W.detach() / sd) @ mu
    return W, b, loss


def fit_spectral_readers(model, labelled, layers, *, k=256, mode="phasor", l2=1e-4, steps=300,
                         namespace="mdb-mlp-memory"):
    """Fit calibrated subject and role readers and return spectral ``AddressFeatures``.

    ``labelled``: ``{"ids", "position", "subject", "relation"}``. Several wordings per relation
    should be supplied so the role reader is trained to be invariant to wording; ``relation``
    may be ``None`` (an explicit "no role" example, e.g. a bare subject). Every other position of
    every labelled prompt is a ``<none>`` example for both readers, so the address abstains
    off the subject. Class weights balance ``<none>`` against labelled rows. Phases come from
    the spectral-address ABI hash of each class label; ``<none>`` has amplitude 0.

    Returns ``({layer: AddressFeatures}, {layer: diagnostics}, classes)``.
    """
    from mdb.spectral_address import _hash_phases
    subjects = sorted({i["subject"] for i in labelled if i["subject"] is not None}) + [NONE]
    roles = sorted({i["relation"] for i in labelled if i["relation"] is not None}) + [NONE]
    s_index, r_index = {c: n for n, c in enumerate(subjects)}, {c: n for n, c in enumerate(roles)}
    device = next(model.parameters()).device
    feats = {layer: [] for layer in layers}
    ys, yr, pos_flag = [], [], []
    for item in labelled:
        ins, _ = capture(model, item["ids"], layers, {})
        for layer in layers:
            feats[layer].append(_features(ins[layer]).float())
        for p in range(len(item["ids"])):
            at = p == item["position"]
            ys.append(s_index[item["subject"]] if at and item["subject"] is not None else s_index[NONE])
            yr.append(r_index[item["relation"]] if at and item["relation"] is not None else r_index[NONE])
            pos_flag.append(at)
    ys, yr = torch.tensor(ys, device=device), torch.tensor(yr, device=device)
    flag = torch.tensor(pos_flag, device=device)
    weights = torch.where(flag, torch.tensor(1.0, device=device),
                          torch.tensor(float(flag.sum()) / max(1, int((~flag).sum())), device=device))
    mode_id = {"phasor": 0, "unit": 1, "onehot": 2}[mode]
    theta_s = torch.tensor([_hash_phases(f"symbol:{namespace}:subject:{c}", k) for c in subjects])
    theta_r = torch.tensor([_hash_phases(f"role:{namespace}:{c}", k) for c in roles])
    amp_s = torch.tensor([0.0 if c == NONE else 1.0 for c in subjects])
    amp_r = torch.tensor([0.0 if c == NONE else 1.0 for c in roles])
    maps, diag = {}, {}
    for layer in layers:
        X = torch.cat(feats[layer]).to(device)
        Ws, bs, loss_s = _softmax_regression(X, ys, len(subjects), weights, l2, steps)
        Wr, br, loss_r = _softmax_regression(X, yr, len(roles), weights, l2, steps)
        with torch.no_grad():
            acc_s = float(((X @ Ws.T + bs).argmax(-1) == ys)[flag].float().mean())
            acc_r = float(((X @ Wr.T + br).argmax(-1) == yr)[flag].float().mean())
            none_r = float(((X @ Wr.T + br).argmax(-1) == r_index[NONE])[~flag].float().mean())
        maps[layer] = AddressFeatures(Ws=Ws, bs=bs, Wr=Wr, br=br, theta_s=theta_s, theta_r=theta_r,
                                      amp_s=amp_s, amp_r=amp_r, mode=torch.tensor(float(mode_id)))
        diag[layer] = {"train_acc_subject": acc_s, "train_acc_relation": acc_r,
                       "offsubject_none_rate": none_r, "rows": int(len(X)),
                       "final_loss_subject": loss_s, "final_loss_relation": loss_r}
    return maps, diag, {"subjects": subjects, "roles": roles}


def address_posteriors(fmap, x):
    """Subject and role posteriors of an ``AddressFeatures`` map for keys ``x`` (diagnostics)."""
    xh = _features(x)
    d = xh.device
    ps = torch.softmax(xh @ fmap._t("Ws", d).T + fmap._t("bs", d), -1)
    pr = torch.softmax(xh @ fmap._t("Wr", d).T + fmap._t("br", d), -1)
    return ps, pr
