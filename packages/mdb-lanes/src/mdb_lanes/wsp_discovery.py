"""Automatic bounded affine-intervention discovery and parameter synthesis.

The caller supplies an evaluator, never a selected site or target weights.
Discovery outcomes select a candidate; held-out evaluation belongs to a later
sealed study. This controller searches an explicit gain/channel grammar, not an
unrestricted space of programs or semantic mechanisms.
"""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .wsp import WeightStudy
from .wsp_weights import ParameterPatch, WeightSpaceError, digest, parameter_inventory


def _plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


@dataclass(frozen=True)
class AffineEdit:
    site: str
    mode: str
    gain: float
    channels: tuple[int, ...] = ()

    @property
    def identity(self):
        return digest(asdict(self))


def affine_sites(model):
    """Infer the affine universe; exclude the declared output embedding module."""
    import torch

    output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return {name: module for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and module is not output}


def _validate_edit(model, edit):
    import math

    sites = affine_sites(model)
    if edit.site not in sites or edit.mode not in {"write-gain", "channel-gain"}:
        raise WeightSpaceError("unknown affine site or intervention mode")
    if isinstance(edit.gain, bool) or not math.isfinite(edit.gain):
        raise WeightSpaceError("finite gain required")
    if (edit.mode == "write-gain" and edit.channels or edit.mode == "channel-gain" and
            (not edit.channels or len(set(edit.channels)) != len(edit.channels)
             or any(type(j) is not int or not 0 <= j < sites[edit.site].in_features for j in edit.channels))):
        raise WeightSpaceError("explicit valid channel support required")
    return sites[edit.site]


@contextmanager
def virtual_edit(model, edit):
    """Change an actual affine return without mutating any parameter."""
    module = _validate_edit(model, edit)

    def hook(layer, args, output):
        if edit.gain == 1:
            return output
        if edit.mode == "write-gain":
            return output * edit.gain
        x = args[0][..., list(edit.channels)]
        correction = x @ layer.weight[:, list(edit.channels)].T
        return output + (edit.gain - 1) * correction

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def synthesize_affine_patch(model, edit, *, model_identity):
    """Lower the selected state intervention into original-dtype weight bytes."""
    module = _validate_edit(model, edit)
    weight = module.weight.detach().clone()
    replacements = {}
    prefix = edit.site + "." if edit.site else ""
    if edit.mode == "write-gain":
        weight *= edit.gain
        if module.bias is not None:
            replacements[prefix + "bias"] = module.bias.detach() * edit.gain
    else:
        weight[:, list(edit.channels)] *= edit.gain
    replacements[prefix + "weight"] = weight
    return ParameterPatch(model, replacements, model_identity=model_identity)


def fit_output_delta(hidden, desired_change, preservation=None, *, preserve_weight=1., ridge=1e-6):
    """FP64 ridge solve: fit ΔD H≈R while penalizing ΔD K; no backprop.

    Samples are rows. This is a local fit to supplied desired changes, not an
    automatic source of desired changes or a guarantee of useful continuation.
    """
    import numpy as np

    h, r = np.asarray(hidden, dtype=np.float64), np.asarray(desired_change, dtype=np.float64)
    k = np.empty((0, h.shape[1])) if preservation is None and h.ndim == 2 else np.asarray(preservation, dtype=np.float64)
    if (h.ndim != 2 or r.ndim != 2 or k.ndim != 2 or h.shape[0] != r.shape[0]
            or not h.shape[0] or k.shape[1] != h.shape[1] or ridge <= 0 or preserve_weight < 0
            or not all(np.isfinite(v).all() for v in (h, r, k))
            or not np.isfinite([ridge, preserve_weight]).all()):
        raise ValueError("finite aligned sample matrices and positive ridge required")
    # Solve in sample space to avoid a hidden-width-squared inverse.
    design = np.concatenate((h, np.sqrt(preserve_weight) * k), axis=0)
    targets = np.concatenate((r, np.zeros((len(k), r.shape[1]))), axis=0)
    dual = np.linalg.solve(design @ design.T + ridge * np.eye(len(design)), targets)
    return dual.T @ design


class AffineDiscovery:
    """Three finite rounds: write screen, dose refinement, channel refinement.

    Evaluator returns {observations: {objective: finite scalar, ...}, work, state}.
    Larger objective is preferred; correctness, preservation and their tradeoff
    are declared by the caller. Rankings are observations, never null claims.
    """

    def __init__(self, model, root, *, binding, input_identity, source_identity,
                 max_sites=32, shortlist=3, channels_per_site=2, max_wall_ms=600000,
                 on_progress=None):
        if any(type(v) is not int or v < 1 for v in (max_sites, shortlist, channels_per_site, max_wall_ms)):
            raise ValueError("positive finite search geometry required")
        self.model, self.root = model, Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.binding, self.input_identity, self.source_identity = binding, input_identity, source_identity
        self.max_sites, self.shortlist, self.channels_per_site = max_sites, shortlist, channels_per_site
        self.max_wall_ms, self.on_progress = max_wall_ms, on_progress
        self.records, self.features = [], {}
        self._baseline = parameter_inventory(model)
        self.sites = affine_sites(model)
        if not self.sites:
            raise ValueError("no supported affine sites")
        universe = sorted(self.sites)
        # Deterministic coverage across the entire supplied model, not a hidden
        # layer preference. Unscanned sites stay explicit in the report.
        indices = sorted({i * len(universe) // min(max_sites, len(universe))
                          for i in range(min(max_sites, len(universe)))})
        self.scanned = [universe[i] for i in indices]
        self._save("universe.json", {"sites": universe, "scanned": self.scanned,
            "grammar": ["write-gain", "channel-gain"], "output_embedding_excluded": True,
            "binding": binding, "input_identity": input_identity, "source_identity": source_identity,
            "parameter_inventory": self._baseline})

    def _save(self, relative, value):
        with (self.root / relative).open("x") as stream:
            json.dump(_plain(value), stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    def _emit(self, value):
        if self.on_progress:
            self.on_progress(copy.deepcopy(value))

    def _round(self, name, edits, evaluator, *, collect_features=False):
        import math

        import torch

        hooks = []
        if collect_features:
            for site in self.scanned:
                def capture(module, args, site=site):
                    x = args[0].detach().float().reshape(-1, module.in_features)
                    sums = x.abs().sum(dim=0).cpu()
                    previous, count = self.features.get(site, (torch.zeros_like(sums), 0))
                    self.features[site] = (previous + sums, count + len(x))
                hooks.append(self.sites[site].register_forward_pre_hook(capture))
        try:
            with WeightStudy(self.root / name, model=self.model, binding=self.binding,
                input_identity=self.input_identity, source_identity=self.source_identity,
                budgets={"max_predictions": max(1, len(edits)), "max_executions": max(1, len(edits)),
                         "max_native_calls": 8192, "max_wall_ms": self.max_wall_ms,
                         "max_actions": max(512, 4 * len(edits) + 16)}) as study:
                for index, edit in enumerate(edits):
                    declaration = asdict(edit) if edit else None
                    study.add_arm("candidate" + str(index), role="edited" if edit else "native",
                        state_policy="restart", context={"edit": declaration, "round": name},
                        predictor=lambda context, e=declaration: {"predictions": {
                            "intervention": e, "local_law": "scaled affine write or selected-column contribution",
                            "consumer_outcome": "unknown before native experiment"}})
                study.seal()
                for index, edit in enumerate(edits):
                    def run(context, edit=edit):
                        if edit is None:
                            return evaluator(context.model)
                        with virtual_edit(context.model, edit):
                            return evaluator(context.model)
                    arm = "candidate" + str(index)
                    study.run(arm, run)
                    observation = study.tx.observation(arm)
                    result = observation["result"]
                    objective = result.get("observations", {}).get("objective")
                    valid = isinstance(objective, (float, int)) and not isinstance(objective, bool) and math.isfinite(objective)
                    row = {"round": name, "arm": arm, "edit": asdict(edit) if edit else None,
                           "objective": float(objective) if valid else None,
                           "instrument_status": observation["instrument_status"] if valid else "evidence-incomplete",
                           "observations": result.get("observations", {}), "error": result.get("error"),
                           "instrumented_work": observation["instrumented_work"]}
                    self.records.append(row)
                    self._emit({"round": name, "completed": index + 1, "total": len(edits),
                                "site": edit.site if edit else None, "objective": row["objective"]})
        finally:
            for hook in hooks:
                hook.remove()

    def discover(self, evaluator):
        self._round("baseline", [None], evaluator, collect_features=True)
        self._round("screen", [AffineEdit(site, "write-gain", .5) for site in self.scanned], evaluator)
        ranked = sorted([r for r in self.records if r["edit"] and r["objective"] is not None],
                        key=lambda r: (-r["objective"], r["edit"]["site"]))
        selected = [r["edit"]["site"] for r in ranked[:self.shortlist]]
        if selected:
            self._round("dose", [AffineEdit(site, "write-gain", gain)
                for site in selected for gain in (0., 1.5)], evaluator)
            import numpy as np

            proposals, curves = [], {}
            for site in selected:
                points = [(1., self.records[0]["objective"])] + [
                    (row["edit"]["gain"], row["objective"]) for row in self.records
                    if row["edit"] and row["edit"]["site"] == site
                    and row["edit"]["mode"] == "write-gain" and row["objective"] is not None]
                points = [(x, y) for x, y in points if y is not None]
                if len(points) >= 3:
                    a, b, c = np.polyfit(*np.asarray(points).T, 2).tolist()
                    gain = float(np.clip(-b / (2 * a), -2, 2)) if a < -1e-15 else None
                    curves[site] = {"points": points, "quadratic": [a, b, c], "proposed_gain": gain,
                                    "status": "surrogate proposal; native test remains authoritative"}
                    if gain is not None:
                        proposals.append(AffineEdit(site, "write-gain", gain))
            self._save("dose-proposals.json", curves)
            if proposals:
                self._round("adaptive-dose", proposals, evaluator)
            channel_edits, nominations = [], {}
            for site in selected:
                sums, count = self.features[site]
                weight = self.sites[site].weight.detach().float()
                score = sums / max(1, count) * weight.norm(dim=0).cpu()
                indices = score.argsort(descending=True, stable=True)[:self.channels_per_site].tolist()
                nominations[site] = {"channels": indices, "scores": score[indices].tolist(),
                                     "status": "magnitude nomination, not causal verdict"}
                channel_edits.extend(AffineEdit(site, "channel-gain", gain, (j,))
                                     for j in indices for gain in (0., 2.))
            self._save("channel-nominations.json", nominations)
            self._round("channels", channel_edits, evaluator)
        valid = [r for r in self.records if r["objective"] is not None]
        if not valid:
            raise WeightSpaceError("no applicable objective observations")
        # Native is first and wins ties; a candidate need not be installed merely
        # because it was the best intervention among harmful alternatives.
        best = max(valid, key=lambda row: row["objective"])
        chosen = AffineEdit(**{**best["edit"], "channels": tuple(best["edit"]["channels"])}) if best["edit"] else None
        candidate = synthesize_affine_patch(self.model, chosen, model_identity=self.binding["model"]) if chosen else None
        if parameter_inventory(self.model) != self._baseline:
            raise WeightSpaceError("discovery changed original parameters")
        if candidate:
            candidate.save(self.root / "selected-patch")
        result = {"schema": "mdb-affine-discovery-v1", "binding": self.binding,
            "universe_size": len(self.sites), "scanned_sites": self.scanned, "records": self.records,
            "selected": asdict(chosen) if chosen else None,
            "baseline_objective": self.records[0]["objective"], "selected_objective": best["objective"],
            "selection_scope": "discovery examples only; no held-out claim",
            "parameter_restoration": "exact inventory match", "terminal_status": "not-evaluated"}
        self._save("selection.json", result)
        return chosen, candidate, result
