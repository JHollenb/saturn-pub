"""Sealed finite candidate panels over one prepared native forward parent.

No callback or import path is loaded from a serialized panel. Model-specific
consumer/continuation adapters remain explicit runtime code owned by the caller.
"""
from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass

from saturn_pub.values import describe, digest

from .forward_interventions import (
    ForwardInterventionError,
    ForwardInterventionResult,
    PreparedForwardParent,
    _dose_value,
    _resolve_doses,
)


@dataclass(frozen=True)
class PanelRow:
    branch_id: str
    result: ForwardInterventionResult | None
    error: str | None
    wall_seconds: float


class ExperimentPanel:
    """Snapshot and seal every replacement before any candidate executes.

    A candidate is a mapping containing replacements, optional doses, source
    identity and metadata. Labels/approximate forecasts are metadata, with no
    authority to overwrite the native outcome or certify a scientific claim.
    """

    def __init__(self, parent: PreparedForwardParent, candidates: Mapping[str, Mapping],
                 *, observe_values=(), max_branches=64, metadata=None):
        if type(parent) is not PreparedForwardParent:
            raise ForwardInterventionError("panel requires a prepared native parent")
        if not isinstance(max_branches, int) or isinstance(max_branches, bool) or max_branches<1:
            raise ForwardInterventionError("panel needs a positive branch budget")
        if not isinstance(candidates, Mapping) or not candidates or len(candidates)>max_branches:
            raise ForwardInterventionError("candidate panel exceeds its finite branch budget")
        self.__parent = parent
        self.__observe = tuple(observe_values)
        self.__candidates = copy.deepcopy(dict(candidates))
        self.__ran = False
        descriptors = parent.value_descriptors()
        rows = {}
        for bid, candidate in self.__candidates.items():
            if not isinstance(bid,str) or not bid.strip() or not isinstance(candidate,Mapping):
                raise ForwardInterventionError("panel branch IDs and candidate mappings are required")
            if set(candidate)-{"replacements","dose","source_identity","metadata"}:
                raise ForwardInterventionError("unknown declarative candidate field")
            replacements = candidate.get("replacements")
            if not isinstance(replacements,Mapping) or not replacements:
                raise ForwardInterventionError("panel candidate requires replacements")
            doses = _resolve_doses(replacements,candidate.get("dose",1.))
            for vid,value in replacements.items():
                rec = descriptors.get(vid)
                if rec is None or rec["producer"] is None:
                    raise ForwardInterventionError("panel site must be a produced SSA value")
                if rec["producer"]<parent.through_event:
                    raise ForwardInterventionError("panel intervention precedes prepared boundary")
                import torch
                if not isinstance(value,torch.Tensor) or value.device.type!="cpu" or value.layout!=torch.strided:
                    raise ForwardInterventionError("panel replacement requires a dense CPU tensor")
                if list(value.shape)!=rec["shape"] or str(value.dtype)!=rec["dtype"]:
                    raise ForwardInterventionError("panel replacement geometry differs")
                if not bool(torch.isfinite(value).all()):
                    raise ForwardInterventionError("panel replacement must be finite")
                _dose_value(doses[vid],vid)
            rows[bid] = {"replacements":{vid:describe(value) for vid,value in replacements.items()},
                "doses":doses,"source_identity":copy.deepcopy(candidate.get("source_identity",{})),
                "metadata":copy.deepcopy(candidate.get("metadata",{}))}
        body = {"schema":"mdb-forward-experiment-panel-v1","parent":parent.fingerprint,
                "candidates":rows,"observe_values":list(self.__observe),
                "metadata":copy.deepcopy(metadata or {}),"max_branches":max_branches,
                "execution":"sequential scalar native continuations; no physical batching"}
        self.__manifest = {**body,"fingerprint":digest(body)}

    @property
    def fingerprint(self):
        return self.__manifest["fingerprint"]

    def manifest(self):
        return copy.deepcopy(self.__manifest)

    def run(self, *, continue_on_error=True):
        if self.__ran:
            raise ForwardInterventionError("panel already executed; create a new sealed panel to replay")
        self.__ran = True
        rows = []
        for bid,candidate in self.__candidates.items():
            started = time.perf_counter()
            try:
                result = self.__parent.branch(candidate["replacements"],
                    dose=candidate.get("dose",1.),observe_values=self.__observe,
                    source_identity={"panel_fingerprint":self.fingerprint,"branch_id":bid,
                                     "declared_source":candidate.get("source_identity",{})})
            except (ValueError,RuntimeError,TypeError,KeyError) as error:
                rows.append(PanelRow(bid,None,f"{type(error).__name__}: {error}",time.perf_counter()-started))
                if not continue_on_error:
                    break
            else:
                rows.append(PanelRow(bid,result,None,time.perf_counter()-started))
        return tuple(rows)
