"""Exact, removable parameter edits for weight-space investigations.

Torch is optional and imported only by numerical operations. Restoration covers
the declared parameter objects; caches, buffers, RNG and environments belong to
the caller's state adapter. No optimizer, gradient update or checkpoint write.
"""
from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


class WeightSpaceError(ValueError):
    """Parameter identity, finite-value or restoration contract was violated."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _descriptor(value):
    import torch

    from .tensor_hash import tensor_bytes_sha256

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise WeightSpaceError("dense strided Torch tensors required")
    if value.device.type == "meta":
        raise WeightSpaceError("materialized tensor required")
    if not bool(torch.isfinite(value).all()):
        raise WeightSpaceError("nonfinite parameter evidence")
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": tensor_bytes_sha256(value), "elements": value.numel()}


def weight_diff(before, after, *, support_limit=64, chunk_elements=1 << 18):
    """Compare named tensors, streaming one tensor/chunk at a time.

    Mappings can lazily load checkpoint tensors. Counts use *named tensor
    elements*, including declared aliases; they are not a logical parameter
    count. Byte changes include signed zero. No SVD or rank assumption is made.
    """
    import torch

    if type(support_limit) is not int or support_limit < 0:
        raise WeightSpaceError("nonnegative support limit required")
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise WeightSpaceError("positive chunk size required")
    if set(before) != set(after) or any(not isinstance(n, str) or not n for n in before):
        raise WeightSpaceError("matching named tensor inventories required")
    rows, total, changed = [], 0, 0
    for name in sorted(before):
        a, b = before[name], after[name]
        da, db = _descriptor(a), _descriptor(b)
        if (da["shape"], da["dtype"]) != (db["shape"], db["dtype"]):
            raise WeightSpaceError("shape/dtype changed: " + name)
        total += da["elements"]
        if da["sha256"] == db["sha256"]:
            rows.append({"name": name, "before": da, "after": db, "changed_elements": 0})
            continue
        af, bf = a.detach().reshape(-1), b.detach().reshape(-1)
        count, numeric, squares, max_abs = 0, 0, 0.0, 0.0
        row_support = torch.zeros(a.shape[0], dtype=torch.bool) if a.ndim == 2 else None
        col_support = torch.zeros(a.shape[1], dtype=torch.bool) if a.ndim == 2 else None
        for start in range(0, a.numel(), chunk_elements):
            x = af[start:start + chunk_elements].cpu().contiguous()
            y = bf[start:start + chunk_elements].cpu().contiguous()
            different = (x.view(torch.uint8).reshape(-1, x.element_size()) !=
                         y.view(torch.uint8).reshape(-1, y.element_size())).any(dim=1)
            count += int(different.sum())
            numeric += int((x != y).sum())
            if x.is_complex():
                delta = y.to(torch.complex128) - x.to(torch.complex128)
            else:
                delta = y.to(torch.float64) - x.to(torch.float64)
            squares += float(delta.abs().square().sum())
            if delta.numel():
                max_abs = max(max_abs, float(delta.abs().max()))
            if row_support is not None:
                indices = torch.nonzero(different).flatten() + start
                row_support[indices // a.shape[1]] = True
                col_support[indices % a.shape[1]] = True
        row = {"name": name, "before": da, "after": db, "changed_elements": count,
               "numerically_changed_elements": numeric, "delta_l2": squares ** 0.5,
               "delta_max_abs": max_abs}
        if row_support is not None:
            for axis, support in (("rows", row_support), ("columns", col_support)):
                indices = torch.nonzero(support).flatten().tolist()
                row[axis] = {"count": len(indices), "indices": indices[:support_limit],
                             "truncated": len(indices) > support_limit}
        changed += count
        rows.append(row)
    return {"schema": "mdb-weight-diff-v1", "tensors": rows,
            "before_identity": digest({r["name"]: r["before"] for r in rows}),
            "after_identity": digest({r["name"]: r["after"] for r in rows}),
            "denominator": {"kind": "named-tensor-elements-including-aliases", "elements": total},
            "changed_elements": changed, "changed_fraction": changed / total if total else None}


def _parameters(model):
    try:
        return dict(model.named_parameters(remove_duplicate=False))
    except (AttributeError, TypeError) as error:
        raise WeightSpaceError("model must expose named parameter aliases") from error


def parameter_inventory(model):
    """Hash unique parameter objects and retain every declared tied name."""
    parameters, seen, rows, byte_count = _parameters(model), set(), [], 0
    for name, parameter in sorted(parameters.items()):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        aliases = sorted(k for k, value in parameters.items() if value is parameter)
        rows.append({"name": name, "aliases": aliases, **_descriptor(parameter)})
        byte_count += parameter.numel() * parameter.element_size()
    return {"parameters": rows, "identity": digest(rows), "bytes_hashed": byte_count}


class ParameterPatch:
    """Frozen replacement tensors, bound to original bytes and alias topology.

    Exact tied aliases are supported. Distinct Parameter objects sharing touched
    storage are refused, including overlapping views. Install is single-writer;
    nested installs on the same parameter are refused by original-byte checks.
    """

    def __init__(self, model, replacements, *, model_identity):
        if not model_identity or not replacements:
            raise WeightSpaceError("model identity and nonempty replacements required")
        parameters = _parameters(model)
        if not set(replacements) <= set(parameters):
            raise WeightSpaceError("replacement names must be actual parameters")
        self._entries, self._active = [], False
        self.model_identity = json.loads(json.dumps(model_identity, allow_nan=False))
        selected = set()
        for name in sorted(replacements):
            original, replacement = parameters[name], replacements[name]
            descriptor = _descriptor(original)
            target_descriptor = _descriptor(replacement)
            if (descriptor["shape"], descriptor["dtype"]) != (
                    target_descriptor["shape"], target_descriptor["dtype"]):
                raise WeightSpaceError("replacement shape/dtype differs: " + name)
            if not original.is_floating_point() or original.is_complex():
                raise WeightSpaceError("patches require real floating-point parameters")
            if not original.is_contiguous():
                raise WeightSpaceError("patches require contiguous parameters")
            aliases = sorted(k for k, value in parameters.items() if value is original)
            for alias in aliases:
                if alias in replacements and _descriptor(replacements[alias]) != target_descriptor:
                    raise WeightSpaceError("conflicting tied parameter replacements")
            if id(original) in selected:
                continue
            selected.add(id(original))
            self._entries.append({"name": aliases[0], "aliases": aliases,
                                  "original": original.detach().cpu().clone(),
                                  "replacement": replacement.detach().cpu().clone(),
                                  "before": descriptor, "after": target_descriptor})
        self._entries.sort(key=lambda entry: entry["name"])
        self._validate_aliases(parameters)
        self._manifest = {"schema": "mdb-parameter-patch-v1", "model_identity": self.model_identity,
                          "parameters": [{k: copy.deepcopy(e[k]) for k in
                                          ("name", "aliases", "before", "after")} for e in self._entries],
                          "restoration_scope": "declared parameter bytes and objects only"}
        self._manifest["identity"] = digest(self._manifest)

    @property
    def manifest(self):
        return copy.deepcopy(self._manifest)

    def _validate_aliases(self, parameters):
        for entry in self._entries:
            name = entry["name"]
            if name not in parameters:
                raise WeightSpaceError("parameter disappeared: " + name)
            original = parameters[name]
            aliases = sorted(k for k, value in parameters.items() if value is original)
            if aliases != entry["aliases"]:
                raise WeightSpaceError("parameter alias topology changed")
            storage = original.untyped_storage().data_ptr()
            if any(value is not original and value.numel() and original.numel()
                   and value.device == original.device
                   and value.untyped_storage().data_ptr() == storage for value in parameters.values()):
                raise WeightSpaceError("distinct parameters share touched storage")

    def scaled(self, model, dose):
        """Scale from original bytes, then round once to the original dtype."""
        import math

        import torch

        if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(dose):
            raise WeightSpaceError("finite scalar dose required")
        replacements = {}
        for entry in self._entries:
            a, b = entry["original"], entry["replacement"]
            replacements[entry["name"]] = (a.clone() if dose == 0 else b.clone() if dose == 1 else
                (a.to(torch.float64) + dose * (b.to(torch.float64) - a.to(torch.float64))).to(a.dtype))
        result = ParameterPatch(model, replacements, model_identity=self.model_identity)
        if [e["before"] for e in result._entries] != [e["before"] for e in self._entries]:
            raise WeightSpaceError("dose requires the original receiving parameters")
        return result

    @contextmanager
    def installed(self, model):
        """Validate every write first; restore even when native execution raises."""
        import torch

        if self._active:
            raise WeightSpaceError("patch is already installed")
        parameters = _parameters(model)
        self._validate_aliases(parameters)
        for entry in self._entries:
            if _descriptor(parameters[entry["name"]]) != entry["before"]:
                raise WeightSpaceError("original parameter bytes differ: " + entry["name"])
            if _descriptor(entry["replacement"]) != entry["after"]:
                raise WeightSpaceError("frozen patch bytes differ")
        self._active = True
        written = []
        try:
            with torch.no_grad():
                for entry in self._entries:
                    parameter = parameters[entry["name"]]
                    written.append((entry, parameter))
                    parameter.copy_(entry["replacement"])
                    if _descriptor(parameter) != entry["after"]:
                        raise WeightSpaceError("installed parameter bytes differ")
            yield self.manifest
        finally:
            restoration_error = None
            with torch.no_grad():
                for entry, parameter in reversed(written):
                    try:
                        parameter.copy_(entry["original"])
                        current = _parameters(model)
                        if any(current.get(alias) is not parameter for alias in entry["aliases"]):
                            raise WeightSpaceError("parameter object replaced during execution")
                        if _descriptor(parameter) != entry["before"]:
                            raise WeightSpaceError("parameter restoration differs")
                    except Exception as error:
                        restoration_error = error
            self._active = False
            if restoration_error is not None:
                raise WeightSpaceError("parameter restoration failed") from restoration_error

    def save(self, root):
        """Persist edited tensors, never original parameter copies or code."""
        from .tensor_archive import write_tensor_archive

        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        payloads = {}
        for index, entry in enumerate(self._entries):
            value = entry["replacement"]
            # Other admitted floating types widen exactly; the original target
            # descriptor still binds bytes reconstructed by load's explicit cast.
            if str(value.dtype) not in {"torch.bfloat16", "torch.float32", "torch.float64"}:
                value = value.float()
            payloads[entry["name"]] = write_tensor_archive(
                root / "weights" / f"{index}.npy", value, root=root)
        record = {"manifest": self.manifest, "payloads": payloads}
        record["identity"] = digest(record)
        with (root / "patch.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return record

    @classmethod
    def load(cls, root, model, *, model_identity):
        import torch

        from .tensor_archive import load_tensor_archive

        record = inspect_patch(root)
        manifest = record["manifest"]
        if manifest["model_identity"] != model_identity:
            raise WeightSpaceError("patch checkpoint/model identity differs")
        replacements = {}
        for entry in manifest["parameters"]:
            array = load_tensor_archive(root, record["payloads"][entry["name"]])
            dtype = getattr(torch, entry["after"]["dtype"].removeprefix("torch."))
            replacements[entry["name"]] = torch.from_numpy(array.copy()).to(dtype)
        result = cls(model, replacements, model_identity=model_identity)
        if result.manifest != manifest:
            raise WeightSpaceError("loaded patch binding differs")
        return result


def inspect_patch(root):
    """Inspect manifest seals without importing a numerical framework.

    Tensor payload verification occurs in ParameterPatch.load using the existing
    archive reader. A recomputed digest is not an external authenticity proof.
    """
    record = json.loads((Path(root) / "patch.json").read_text())
    manifest = record["manifest"]
    if (manifest.get("schema") != "mdb-parameter-patch-v1"
            or record.get("identity") != digest({k: v for k, v in record.items() if k != "identity"})
            or manifest.get("identity") != digest({k: v for k, v in manifest.items() if k != "identity"})):
        raise WeightSpaceError("patch manifest seal differs")
    return record
