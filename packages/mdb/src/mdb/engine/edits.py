"""State-cell edits: a write into the key/value cache that the native decoder layers read.

An occurrence and a state cell are different things. ``t18.L20.v`` is the value row 18 wrote
when it ran layer 20 (edited at that occurrence with ``Run.set``). ``cache.L20.v[kv0,row=18]``
is the cell as later rows read it: ``Run.set_cell`` edits it at any cursor after the write, and
the edit takes effect for every read of the cell that has not happened yet. Reads that already
happened are not recomputed.

Realisation is native. A ``CellEdit`` rides on the next layer transition (``PortController``
``cells=``): its write lands in the ``DynamicCache`` object that transition hands the native
layer, by copy-on-write of that layer's key or value tensor (the cut a checkpoint holds keeps
the unedited tensor). ``timing`` says when, inside that transition:

* ``before``: at the layer's entry, before its attention reads the cache;
* ``after``: at the layer's exit, after its attention read. Used when the cursor sits in the
  cell's own layer at or past the read boundary (``CELL_BOUNDARY``): ``kr`` for a key cell
  (the key argument of the eager call holds the cached keys), ``p`` for a value cell (a stop at
  ``p`` computes the value mix natively).

Operators (``CELL_OPS``): ``replace``, ``add``, ``scale`` as for ports (the operand cast to the
cell dtype, one rounded op), and ``mix DOSE`` toward a value: ``cell + DOSE * (value - cell)`` in
float32, rounded once to the cell dtype; ``mix 1`` is the exact copy of the value's words.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .ports import _max_abs, _with_labels, changed_fraction, tensor_digest

CELL_OPS = ("replace", "add", "scale", "mix")
CELL_BOUNDARY = {"k": "kr", "v": "p"}
CELL_READ = {"k": "kr", "v": "A"}
TIMINGS = ("before", "after")
CELL_AXES = ("batch", "kv_head", "row", "head_dim")


def _picks(items: Sequence[int] | None, name: str) -> tuple[int, ...] | None:
    if items is None:
        return None
    out = tuple(int(i) for i in items)
    if not out:
        raise ValueError(f"cell {name} selection names no element")
    if any(i < 0 for i in out) or len(set(out)) != len(out):
        raise ValueError(f"cell {name} indices must be distinct nonnegative integers")
    return out


@dataclass(frozen=True)
class CellEdit:
    """One staged write into ``cache.L<layer>.<tensor>``.

    ``rows`` (required), ``kv`` and ``d`` (None: the whole axis) pick the elements. ``value`` is a
    finite number or a floating tensor of the picked block's shape ``(1, kv, rows, d)``. ``dose``
    is required for ``mix`` and refused otherwise. ``staged_at`` is the transition count of the
    boundary the cursor stood on when the edit was staged (which stops ``reverse`` re-stages it
    at). ``selection`` is the selector text and ``origin`` the pointer of a value taken from
    another run (``{origin, via, address, sha256, where, cut, recipe}``), both recorded verbatim.
    """

    layer: int
    tensor: str
    rows: tuple[int, ...]
    value: Any
    op: str = "replace"
    kv: tuple[int, ...] | None = None
    d: tuple[int, ...] | None = None
    dose: float | None = None
    timing: str = "before"
    staged_at: int = 0
    selection: str | None = None
    origin: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("cell layer must be a nonnegative integer")
        if self.tensor not in ("k", "v"):
            raise ValueError("a cache cell is k or v")
        if self.op not in CELL_OPS:
            raise ValueError(f"cell op must be one of {', '.join(CELL_OPS)}")
        if self.timing not in TIMINGS:
            raise ValueError(f"cell timing must be one of {', '.join(TIMINGS)}")
        object.__setattr__(self, "rows", _picks(self.rows, "row"))
        object.__setattr__(self, "kv", _picks(self.kv, "kv"))
        object.__setattr__(self, "d", _picks(self.d, "d"))
        if (self.op == "mix") != (self.dose is not None):
            raise ValueError("mix needs a dose, and only mix takes one")
        if self.dose is not None:
            dose = float(self.dose)
            if dose != dose or dose in (float("inf"), float("-inf")):
                raise ValueError("a mix dose must be finite")
            object.__setattr__(self, "dose", dose)
        value = self.value
        if isinstance(value, torch.Tensor):
            if not value.is_floating_point():
                raise ValueError("cell edit tensors must be floating point")
            if value.dim() != 4 or value.shape[0] != 1 or value.shape[2] != len(self.rows):
                raise ValueError(f"a cell value tensor has the block shape (1, kv, rows, d); got "
                                 f"{list(value.shape)} for {len(self.rows)} row(s)")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("cell edit values must be finite")
            object.__setattr__(self, "value", value.detach().to("cpu").clone())
        elif type(value) in (int, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("edit scalars must be finite")
            object.__setattr__(self, "value", float(value))
        else:
            raise ValueError("cell edit value must be a tensor or a finite number")
        if type(self.staged_at) is not int or self.staged_at < 0:
            raise ValueError("staged_at must be a nonnegative transition count")

    # --- what the edit names ------------------------------------------------------------

    @property
    def port(self) -> str:
        """The cell's name without a selector: ``cache.L<layer>.<tensor>``."""
        return f"cache.L{self.layer}.{self.tensor}"

    @property
    def text(self) -> str:
        """The cell address with its selector, as the user wrote it (canonical picks)."""
        return self.port + f"[{self.selection}]" if self.selection else self.port

    def block_shape(self, kv_heads: int, head_dim: int) -> tuple[int, int, int, int]:
        return (1, len(self.kv) if self.kv else kv_heads, len(self.rows),
                len(self.d) if self.d else head_dim)

    def check(self, shape: Sequence[int]) -> None:
        """Refuse an edit that does not fit a cache tensor of ``shape`` ``(1, KV, T, d_h)``."""
        _, kv_heads, length, head_dim = (int(s) for s in shape)
        if max(self.rows) >= length:
            raise ValueError(f"{self.port} holds rows 0..{length - 1} here; row {max(self.rows)} "
                             "is not written yet")
        if self.kv is not None and max(self.kv) >= kv_heads:
            raise ValueError(f"kv {max(self.kv)} is out of range ({kv_heads} kv heads)")
        if self.d is not None and max(self.d) >= head_dim:
            raise ValueError(f"d {max(self.d)} is out of range (head_dim {head_dim})")
        if isinstance(self.value, torch.Tensor):
            want = self.block_shape(kv_heads, head_dim)
            if tuple(self.value.shape) != want:
                raise ValueError(f"the value has shape {list(self.value.shape)}; the selected "
                                 f"cell block has shape {list(want)}")

    def index(self, shape: Sequence[int], device: Any = "cpu") -> tuple[Any, ...]:
        """Advanced index of the picked block in a cache tensor of ``shape``."""
        _, kv_heads, _, head_dim = (int(s) for s in shape)
        kv = torch.tensor(self.kv if self.kv else range(kv_heads), device=device)
        rows = torch.tensor(self.rows, device=device)
        d = torch.tensor(self.d if self.d else range(head_dim), device=device)
        return (0, kv.view(-1, 1, 1), rows.view(1, -1, 1), d.view(1, 1, -1))

    def mask(self, shape: Sequence[int]) -> torch.Tensor:
        """The picked elements as a bool mask over a cache tensor of ``shape``."""
        out = torch.zeros(tuple(int(s) for s in shape), dtype=torch.bool)
        out[self.index(shape)] = True
        return out

    @property
    def exact_copy(self) -> bool:
        """Whether every picked element becomes the cast of the supplied value (replace, mix 1)."""
        return self.op == "replace" or (self.op == "mix" and self.dose == 1.0)

    # --- realisation ----------------------------------------------------------------------

    def _operand(self, block: torch.Tensor) -> torch.Tensor:
        if isinstance(self.value, torch.Tensor):
            return self.value[0].to(block.device)
        return torch.tensor(self.value, dtype=torch.float64, device=block.device)

    def ideal(self, block: torch.Tensor) -> torch.Tensor:
        """The exact float64 result on the picked block ``block`` (the request, unrounded)."""
        x = block.detach().to("cpu", torch.float64)
        v = self._operand(block).detach().to("cpu", torch.float64).expand(x.shape)
        if self.exact_copy:
            return v.clone()
        if self.op == "add":
            return x + v
        if self.op == "scale":
            return x * v
        return x + self.dose * (v - x)

    def apply_block(self, block: torch.Tensor) -> torch.Tensor:
        """The written block: the native arithmetic of the operator on the picked block."""
        operand = self._operand(block)
        if self.exact_copy:
            return operand.to(block.dtype).expand(block.shape).clone()
        if self.op == "add":
            return block + operand.to(block.dtype)
        if self.op == "scale":
            return block * operand.to(block.dtype)
        x = block.float()
        return (x + self.dose * (operand.float() - x)).to(block.dtype)

    def realise(self, cache: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """``(new cache tensor, record)``: a copy of ``cache`` with the block written (the input
        is never modified). The record: ``realised_sha256`` of the written block, its dtype,
        ``cast_*`` (the operand against its cast), ``realised_*`` (the written block against
        the exact float64 request), ``changed_fraction`` and ``count``."""
        self.check(cache.shape)
        index = self.index(cache.shape, cache.device)
        block = cache[index]
        written = self.apply_block(block)
        out = cache.clone()
        out[index] = written
        operand = self._operand(block).detach().to("cpu", torch.float64)
        cast = operand.to(block.dtype).to(torch.float64)
        cast_error = _max_abs((cast - operand).abs())
        effect = _max_abs((written.detach().to("cpu", torch.float64) - self.ideal(block)).abs())
        record = {
            "realised_dtype": str(written.dtype),
            "realised_sha256": tensor_digest(written),
            "cast_exact": cast_error == 0.0,
            "cast_max_abs_error": cast_error,
            "realised_exact": effect == 0.0,
            "realised_max_abs_error": effect,
            "changed_fraction": changed_fraction(block, written),
            "count": int(block.numel()),
        }
        return out, _with_labels(record)

    # --- records ----------------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        """Plain JSON of the request (tensors by payload sha256): what checkpoints and replay
        recipes hold. ``port`` is ``cache.L<l>.<k|v>``, so an edit list stays one list."""
        value = self.value
        if isinstance(value, torch.Tensor):
            described: dict[str, Any] = {"kind": "tensor", "sha256": tensor_digest(value),
                                         "shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            described = {"kind": "scalar", "value": value}
        cell = {"layer": self.layer, "tensor": self.tensor, "rows": list(self.rows),
                "kv": None if self.kv is None else list(self.kv),
                "d": None if self.d is None else list(self.d), "timing": self.timing,
                "staged_at": self.staged_at}
        entry: dict[str, Any] = {"port": self.port, "op": self.op, "value": described,
                                 "cell": cell}
        if self.dose is not None:
            entry["dose"] = self.dose
        if self.selection is not None:
            entry["selection"] = self.selection
        if self.origin is not None:
            entry["origin"] = dict(self.origin)
        return entry

    @property
    def key(self) -> str:
        described = self.manifest()["value"]
        return described["sha256"] if described["kind"] == "tensor" else f"scalar:{described['value']}"

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], value: Any) -> CellEdit:
        """The edit a ``manifest()`` describes, with ``value`` (the tensor it names, or None for a
        scalar)."""
        cell = manifest["cell"]
        described = manifest["value"]
        return cls(
            int(cell["layer"]), cell["tensor"], tuple(cell["rows"]),
            described["value"] if described["kind"] == "scalar" else value,
            manifest["op"], kv=cell.get("kv"), d=cell.get("d"), dose=manifest.get("dose"),
            timing=cell.get("timing", "before"), staged_at=int(cell.get("staged_at", 0)),
            selection=manifest.get("selection"), origin=manifest.get("origin"),
        )


def is_cell(manifest: dict[str, Any]) -> bool:
    """Whether an edit manifest (checkpoint or recipe edit) is a state-cell edit."""
    return "cell" in manifest


def overlay(cache: torch.Tensor, cells: Sequence[CellEdit], layer: int, tensor: str
            ) -> torch.Tensor:
    """``cache`` (one layer's key or value tensor) with the ``before`` edits of ``cells`` that
    address it written in order, as the next native read of it will see them."""
    out = cache
    for edit in cells:
        if edit.layer == layer and edit.tensor == tensor and edit.timing == "before":
            out = edit.realise(out)[0]
    return out


__all__ = ["CELL_AXES", "CELL_BOUNDARY", "CELL_OPS", "CELL_READ", "CellEdit", "TIMINGS",
           "is_cell", "overlay"]
