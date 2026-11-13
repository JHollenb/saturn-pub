"""View records served by ``mdb serve``: the source of truth for the wire format.

TypeScript types in ``web/src/api/types.gen.ts`` are generated from these models
(``python -m mdb.serve.typegen --write``); a test fails when that file is stale.

Every number travels as ``Num``: a value with its evidence class. ``unresolved`` and ``refused``
carry no value and must say why. Field names are the wire names (camelCase where the UI reads
them), so the generated types need no mapping layer.
"""
from __future__ import annotations

import math
from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_PREFIX = "mdb-view-"
CAP_BYTES = 64 * 1024

Ev = Literal["measured", "derived-exact", "derived-local", "bound", "supplied", "unresolved",
             "refused"]
VALUED = ("measured", "derived-exact", "derived-local", "bound", "supplied")
# Keys whose own value is structure, not a measurement: a scalar int, a list of ints, or a flat
# name -> int map such as a count per kind (ids, indices, shapes, counts, clock positions, byte
# sizes). A key here exempts only ITS value, never a subtree (a float, a nested object or a list of
# objects under it is still checked);
# every other number in a response is a ``Num``/``NumList`` carrying an evidence class. The web
# client reads the same list (``STRUCTURAL_KEYS`` in the generated types) and the contract test
# walks every response with it.
STRUCTURAL_KEYS = (
    "at", "i", "x", "y", "x0", "x1", "y0", "y1", "w", "h", "level", "lane", "count", "bytes", "cap",
    "seq", "page", "pages", "total", "order", "mask", "k", "bins", "axis", "axes", "fixed",
    "shape", "outShape", "full", "index", "selector", "select", "item", "block", "head",
    "kvGroup", "step", "clock", "pos", "token", "layer", "nonfinite", "counts", "cuts",
    "receipts", "symbols", "records", "sessions", "rows", "id", "locationsMore", "readersMore",
    "usesMore", "evidenceMore", "limits", "kinds",
    # snake_case keys of stored record bodies, symbol bindings and state clocks (indices, ids and
    # counts; every one holds only non-negative ints in real data, never a measurement):
    "attention_selector", "contrast", "decision", "edges", "kv_group", "kv_head", "left", "right",
    "local_clock", "logical_step", "native_payload_reads", "native_transition_index", "neuron",
    "nodes", "own_tokens", "query_head", "receiver_row", "repeated_argument_groups", "row", "size",
    "source_row", "src", "steps", "stride", "token_id", "token_ids", "tokens",
    "torch_num_threads", "transition", "n",
)
State = Literal["resolved", "unresolved", "refused"]
Verdict = Literal["supported", "set_only", "not_supported", "unresolved", "control_missing",
                  "untested"]


class Model(BaseModel):
    # Every field is always serialised, so defaults are required in the generated types.
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Num(Model):
    """A number with its evidence class. ``ptr`` is the digest of the pointer it was read from."""
    v: float | None
    ev: Ev
    unit: str | None = None
    proof: str | None = None
    ptr: str | None = None
    reason: str | None = None
    note: str | None = None  # e.g. "as recorded; this record carries no per-number authority"
    recorded_authority: str | None = None  # the record-level authority string, shown as text

    @model_validator(mode="after")
    def _evidence(self):
        if self.ev in ("unresolved", "refused"):
            if self.v is not None:
                raise ValueError(f"an {self.ev} number carries no value")
            if not self.reason:
                raise ValueError(f"an {self.ev} number says why")
        else:
            if self.v is None or not math.isfinite(self.v):
                raise ValueError("a valued number needs a finite value")
        return self


class NumList(Model):
    """A labelled vector: every value shares one evidence class (e.g. histogram bin edges)."""
    vs: list[float]
    ev: Ev
    unit: str | None = None
    proof: str | None = None

    @model_validator(mode="after")
    def _evidence(self):
        if self.ev in ("unresolved", "refused"):
            raise ValueError("a NumList holds values; use Num for unresolved/refused")
        if not all(math.isfinite(v) for v in self.vs):
            raise ValueError("NumList values are finite")
        return self


def num(value, ev, **kw) -> Num:
    return Num(v=None if value is None else float(value), ev=ev, **kw)


def unresolved(reason, **kw) -> Num:
    return Num(v=None, ev="unresolved", reason=reason, **kw)


class Omitted(Model):
    """What a bounded list left out: never silent truncation."""
    field: str
    count: int
    sum: Num | None = None
    reason: str | None = None


class Refused(Model):
    """Sent with ``None`` fields dropped, so they are optional on the wire."""
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=False)

    refused: str  # the reason
    state: Literal["refused"] = "refused"
    detail: str | None = None
    bytes: int | None = None
    cap: int | None = None


T = TypeVar("T")


class Envelope(Model, Generic[T]):
    schema_: str = Field(alias="schema")
    state: State = "resolved"
    reason: str | None = None  # non-empty whenever state is not "resolved"
    data: T
    digest: str
    bytes: int
    omitted: list[Omitted] | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True,
                              json_schema_serialization_defaults_required=True)


# --------------------------------------------------------------------------- clocks / addresses

class AxisTick(Model):
    """One tick of a clock axis. ``addressable`` is False for a boundary that no occurrence
    address names (e.g. before the embedding, where no hidden state exists yet)."""
    i: int
    label: str
    frag: str = ""
    port: str | None = None
    detail: str | None = None
    addressable: bool = True
    inferred: bool = False  # True when the tick exists only because of an inferred layer count


class ClockAxis(Model):
    """One named clock axis, ordered. The renderer never assumes which axes exist."""
    id: str
    label: str
    kind: Literal["ordinal", "categorical"]
    ticks: list[AxisTick]
    note: str | None = None


class OccAddr(Model):
    run: str
    branch: str | None = None
    cut: str | None = None
    clock: dict[str, int | None]
    sel: str | None = None


class SlotPointer(Model):
    kind: Literal["slot"] = "slot"
    cut: str
    slot: str
    item: int | None = None
    select: list[int] = []


class BlobPointer(Model):
    kind: Literal["blob"] = "blob"
    ref: str


class OccurrencePointer(Model):
    kind: Literal["occurrence"] = "occurrence"
    address: str


PointerSpec = Annotated[Union[SlotPointer, BlobPointer, OccurrencePointer],
                        Field(discriminator="kind")]


class PointerRef(Model):
    digest: str
    addr: str | None = None
    spec: PointerSpec | None = None
    via: list[str] = []
    status: Literal["resolvable", "frontier"]
    reason: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None


class LiveState(Model):
    state: Literal["offline", "paused", "running"]
    frontier: list[str] = []


class FrameFocus(Model):
    cut: str | None
    point: dict[str, Any] | None
    receipts: list[str]
    symbols: list[str]
    records: list[str]
    pointers: list[PointerRef]
    note: str | None = None


class Frame(Model):
    """``addr`` is the canonical spelling of the requested address; ``key`` is its value key
    (``mdb.core.address.value_key``: ``tN.emb`` -> ``tN.L0.h``, ``tN.Lk.h_out`` ->
    ``tN.L{k+1}.h``), the identity views use to decide two addresses are one cell."""
    addr: str
    key: str | None = None
    occ: OccAddr
    axes: list[ClockAxis]
    pos: dict[str, int | None]
    live: LiveState
    focus: FrameFocus | None = None
    prev: str | None = None
    next: str | None = None
    stateClock: dict[str, Any] | None = None


# --------------------------------------------------------------------------- sessions

class SessionSummary(Model):
    id: str
    family: str | None
    adapter: str | None
    granularity: str | None
    model: str
    cuts: int
    receipts: int
    symbols: int
    records: int
    selected: str
    home: str | None
    state: State = "resolved"  # "refused": the session's stored rows cannot be read
    reason: str | None = None


class PortInfo(Model):
    name: str
    role: str | None
    kind: str | None
    writable: bool | None
    recorded: bool
    addr: str | None = None


class LensInfo(Model):
    id: str
    label: str
    unit: str
    hasData: bool
    note: str


class ViewAvail(Model):
    available: bool
    reason: str | None = None


class Capabilities(Model):
    """``layers`` is the decoder layer count: ``measured`` (the cut surface's layer clock),
    ``derived-local`` (a lower bound: the highest recorded layer index) or ``unresolved``."""
    session: str
    axes: list[ClockAxis]
    layers: Num
    ports: list[PortInfo]
    lenses: list[LensInfo]
    views: dict[str, ViewAvail]
    reduce: dict[str, Any]
    ops: ViewAvail


class Hello(Model):
    version: str
    schemas: dict[str, str]
    cap: int
    sessions: int
    readOnly: bool
    capabilities: dict[str, ViewAvail]
    web: ViewAvail


# --------------------------------------------------------------------------- timeline / branches

class TimelineCell(Model):
    x: int
    y: int
    v: float | None
    ev: Ev
    glyphs: list[str] = []


class TokenCell(Model):
    x: int
    id: int | None
    text: str | None = None
    ev: Ev
    reason: str | None = None


class Window(Model):
    x0: int
    x1: int
    y0: int
    y1: int


class TimelineBlock(Model):
    lens: str
    lenses: list[LensInfo]
    unit: str
    axes: list[ClockAxis]
    win: Window
    cells: list[TimelineCell]
    fill: Num
    tokens: list[TokenCell] | None = None
    branch: str | None = None


class BranchNode(Model):
    cut: str
    parent: str | None
    branch: str
    clock: dict[str, int | None]
    kind: str
    retained: bool
    receipt: str | None = None


class BranchInfo(Model):
    name: str
    head: str
    base: str | None
    cuts: int


class BranchTree(Model):
    run: str
    active: str
    nodes: list[BranchNode]
    branches: list[BranchInfo]


# --------------------------------------------------------------------------- symbols

class ReadSourceView(Model):
    name: str
    rows: list[int]
    assignment: str


class ReaderView(Model):
    block: int
    head: int
    kvGroup: int | None
    step: int | None
    receivers: str | None


class ReadEdge(Model):
    id: str
    source: ReadSourceView
    reader: ReaderView
    mass: Num
    massKind: str
    verdict: Verdict
    addr: str | None = None


class LifePoint(Model):
    cut: str | None
    clock: dict[str, int | None]
    label: str


class Lifetime(Model):
    defs: list[LifePoint]
    reads: list[LifePoint]


class UseView(Model):
    id: str
    consumer: str
    support: str
    closure: str | None


class SymbolRow(Model):
    id: str
    symbol: str
    context: str
    status: Literal["supplied", "observed", "validated", "abstained"]
    assignment: str | None
    claim: str
    confidence: Num | None
    abstainReason: str | None
    locations: list[str]
    locationsMore: int
    readers: list[ReadEdge]
    readersMore: int
    uses: list[UseView]
    usesMore: int
    verdict: str
    lifetime: Lifetime
    evidence: list[str]
    evidenceMore: int
    fingerprint: str
    lifecycles: list[str]


class SymbolPage(Model):
    rows: list[SymbolRow]
    page: int
    pages: int
    total: int
    at: str | None
    q: str | None
    unbound: Num
    coverage: str


class SymbolDetail(Model):
    row: SymbolRow
    binding: dict[str, Any]
    slices: list[str]
    statuses: dict[str, str]
    claimBoundary: str | None


# --------------------------------------------------------------------------- graph

GraphEdgeType = Literal["dataflow", "read", "kv_write", "attribution", "intervention", "declared",
                        "unknown"]


class GraphNode(Model):
    id: str
    kind: Literal["state", "action", "output", "source", "aggregate"]
    label: str
    role: str
    x: int | None = None
    y: int | None = None
    lane: int = 0
    ev: Ev | None = None
    count: int | None = None
    addr: str | None = None
    meta: dict[str, Any] = {}


class GraphEdge(Model):
    id: str
    source: str
    target: str
    type: GraphEdgeType
    relation: str
    evidence: str


class PrunedGroup(Model):
    group: str
    count: int
    node: str
    sum: Num | None = None


class CircuitOverlay(Model):
    id: str
    kind: str
    label: str  # text only: any number goes in ``interaction``
    interaction: Num | None = None
    nodes: list[str]
    verdict: str
    record: str


class SliceRef(Model):
    id: str
    component: str
    cut: str
    complete: bool


class GraphView(Model):
    zoom: Literal["slice", "session"]
    center: str | None
    slice: SliceRef | None
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    pruned: list[PrunedGroup]
    overlays: list[CircuitOverlay]
    slices: list[SliceRef]
    authority: str


# --------------------------------------------------------------------------- records

class NamedNum(Model):
    name: str
    n: Num


class RecordSummary(Model):
    id: str
    schema_: str = Field(alias="schema")
    kind: str
    title: str
    status: str
    authority: str
    cut: str | None = None
    focus: str | None = None
    addr: str | None = None
    order: int
    nums: list[NamedNum] = []

    model_config = ConfigDict(extra="forbid", populate_by_name=True,
                              json_schema_serialization_defaults_required=True)


class RecordPage(Model):
    rows: list[RecordSummary]
    page: int
    pages: int
    total: int
    kinds: dict[str, int]


class ArmRow(Model):
    mask: int
    label: str
    status: str
    value: Num
    effect: Num
    prediction: Num | None = None
    residual: Num | None = None
    decision: str | None = None
    record: str
    cut: str | None = None


class Link(Model):
    rel: str
    id: str


class RecordDetail(Model):
    summary: RecordSummary
    body: dict[str, Any]
    arms: list[ArmRow] | None = None
    pointers: list[PointerRef]
    links: list[Link]


# --------------------------------------------------------------------------- reduce / ops / events

ReduceOp = Literal["stats", "topk", "slice", "histogram", "tile", "project"]


class ReduceRequest(Model):
    pointer: PointerSpec
    op: ReduceOp
    params: dict[str, Any] = {}


class ReduceBase(Model):
    op: ReduceOp
    pointer: str
    shape: list[int]
    dtype: str
    note: str | None = None


class StatsResult(ReduceBase):
    op: Literal["stats"] = "stats"
    count: Num
    finite: Num
    min: Num
    max: Num
    mean: Num
    std: Num
    absmax: Num
    l2: Num


class TopkEntry(Model):
    index: list[int]
    v: Num


class TopkResult(ReduceBase):
    op: Literal["topk"] = "topk"
    axis: int | None
    by: Literal["abs", "signed"]
    k: int
    entries: list[TopkEntry]
    score: str


class SliceResult(ReduceBase):
    op: Literal["slice"] = "slice"
    selector: list[Any]
    outShape: list[int]
    encoding: Literal["f32-le-base64"]
    data: str
    nonfinite: int
    ev: Ev


class HistogramResult(ReduceBase):
    op: Literal["histogram"] = "histogram"
    bins: int
    edges: NumList
    counts: list[int]
    nonfinite: int
    ev: Ev


class TileResult(ReduceBase):
    op: Literal["tile"] = "tile"
    axes: list[int]
    fixed: list[int]
    x0: int
    y0: int
    level: int
    w: int
    h: int
    full: list[int]
    encoding: Literal["u8-base64"]
    data: str
    domain: NumList
    scale: Literal["linear", "diverging"]
    agg: str
    nonfinite: int
    ev: Ev


class ProjectResult(ReduceBase):
    op: Literal["project"] = "project"
    direction: str
    directionNorm: Num
    outShape: list[int]
    encoding: Literal["f32-le-base64"]
    data: str
    ev: Ev


ReduceResult = Annotated[Union[StatsResult, TopkResult, SliceResult, HistogramResult, TileResult,
                               ProjectResult], Field(discriminator="op")]


class OpRequest(Model):
    op: str
    args: dict[str, Any] = {}
    request_id: str


class Event(Model):
    type: Literal["hello", "heartbeat"]
    seq: int
    at: int  # epoch milliseconds (a clock position, not a measurement)
    data: dict[str, Any] = {}


# Wire schema ids, one per view record (the Envelope ``schema`` field).
VIEWS = {
    "hello": Hello,
    "sessions": SessionSummary,
    "capabilities": Capabilities,
    "frame": Frame,
    "timeline": TimelineBlock,
    "branches": BranchTree,
    "symbols": SymbolPage,
    "symbol": SymbolDetail,
    "graph": GraphView,
    "records": RecordPage,
    "record": RecordDetail,
    "reduce": None,  # ReduceResult union
    "op": None,  # registry-defined
    "event": Event,
}
SCHEMAS = {name: f"{SCHEMA_PREFIX}{name}-v1" for name in VIEWS}

# Models exported to TypeScript (typegen walks their JSON Schema).
EXPORTED = [Num, NumList, Omitted, Refused, AxisTick, ClockAxis, OccAddr, SlotPointer, BlobPointer,
            OccurrencePointer, PointerRef, LiveState, FrameFocus, Frame, SessionSummary, PortInfo,
            LensInfo, ViewAvail, Capabilities, Hello, TimelineCell, TokenCell, Window,
            TimelineBlock, BranchNode, BranchInfo, BranchTree, ReadSourceView, ReaderView,
            ReadEdge, LifePoint, Lifetime, UseView, SymbolRow, SymbolPage, SymbolDetail,
            GraphNode, GraphEdge, PrunedGroup, CircuitOverlay, SliceRef, GraphView, NamedNum,
            RecordSummary, RecordPage, ArmRow, Link, RecordDetail, ReduceRequest, StatsResult,
            TopkEntry, TopkResult, SliceResult, HistogramResult, TileResult, ProjectResult,
            OpRequest, Event]
