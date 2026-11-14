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
    """``live`` is the session id of the attached debugger (``mdb serve --tiny|--model``), None
    on a read-only server over a recorded project."""
    version: str
    schemas: dict[str, str]
    cap: int
    sessions: int
    readOnly: bool
    capabilities: dict[str, ViewAvail]
    web: ViewAvail
    live: str | None = None


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


class OpResult(Model):
    """One op of a live session. ``result`` is exactly the JSON ``python -m mdb.repl --json``
    prints for the same command (an op result, ``{error, state, reason}`` or ``{interrupted,
    state, reason}``); its numbers are registry-labelled (``{value, authority}``). ``seq``
    numbers ops and console lines in execution order; ``where`` is the cursor after it."""
    request_id: str
    op: str
    seq: int
    where: str | None
    result: dict[str, Any]


class ConsoleRequest(Model):
    line: str = Field(max_length=4096)


class ConsoleResult(Model):
    """One console line: ``text`` is what the REPL prints for it."""
    line: str
    seq: int
    where: str | None
    text: str


class Interrupt(Model):
    """``interrupted``: an op was running and Ctrl-C was delivered to it. ``cancelled``: jobs
    that were queued behind it and will not run (each answers as interrupted)."""
    interrupted: bool
    running: str | None
    cancelled: int
    reason: str | None


class LiveUpstream(Model):
    """One edit upstream of a value: ``selection`` is the selector text that limits it (``h6``
    for ``L22.A[h6]``, ``row=2`` for a state cell), None for the whole port."""
    row: int
    layer: int
    port: str
    op: str
    staged: bool
    selection: str | None = None


class LiveEdit(Model):
    """Provenance of a computed port value (``Run.read_info``): ``authority`` is None and
    ``state`` refused when the engine declined to label it."""
    state: State
    authority: Ev | None
    edited: bool
    kind: str
    by: str | None
    reason: str | None
    upstream: list[LiveUpstream]
    upstream_total: int


class LivePort(Model):
    """One port of the listed layer in program order (the ``list`` op). ``edit`` is None for a
    port the run has not computed yet (nothing to say, never "unedited")."""
    port: str
    addr: str
    sublayer: str | None
    operator: str
    proofs: list[str]
    editable: bool
    cursor: bool
    focus: bool
    computed: bool
    edit: LiveEdit | None = None


class LiveCheckpoint(Model):
    """A checkpoint name and where its latest checkpoint stands: ``durable`` (stored in the
    project; ``restart NAME`` replays it after a server restart) or ``memory`` (this process
    only, the ``checkpoints`` op; a server without a project)."""
    name: str
    where: str
    store: Literal["memory", "durable"]


class LiveFrame(Model):
    """The live cursor: ``where`` (None before ``start``), the ``where`` op's ``stop`` and
    tokens, the listed layer's ports, staged and committed edits (the ``info edits`` op), the
    breakpoint/watch table (``info breakpoints``), displays and checkpoints. ``limits``:
    ``layers``, ``max_positions``, ``heads``, ``kv_heads``. The frame always fits the response
    cap: each list shows its first items and ``more`` names every list it shortened (``field``:
    the list, or
    ``upstream:<addr>`` for a port's upstream edits; ``count``: items left out; ``reason``: the
    ``GET .../live/items`` request for the rest)."""
    session: str
    family: str
    live: LiveState
    seq: int
    where: str | None
    stop: dict[str, Any] | None
    row: int | None
    layer: int | None
    limits: dict[str, int]
    ports: list[LivePort]
    checks: dict[str, Any] | None = None
    tokens: list[dict[str, Any]]
    edits: dict[str, Any] | None
    breakpoints: list[dict[str, Any]]
    displays: list[dict[str, Any]]
    checkpoints: list[LiveCheckpoint]
    more: list[Omitted]


class LiveItems(Model):
    """One page of a live frame list (``breakpoints``, ``displays``, ``checkpoints``,
    ``staged``, ``committed`` edits, or ``upstream`` edits of the computed address ``addr``):
    the items from ``offset`` that fit one response, ``total`` items in all, ``next`` the offset
    of the following page (None at the end)."""
    list: str
    addr: str | None
    offset: int
    total: int
    items: list[dict[str, Any]]
    next: int | None


class LiveRow(Model):
    """One token row: ``token`` its id; ``count`` stops computed in it (all for a finished row,
    None for a prefilled row whose ports were never stepped); ``edits`` ports edited there,
    each with its selector (``L22.A[h6]``)."""
    row: int
    token: int | None
    count: int | None
    edits: list[str]


class LiveStop(Model):
    """A stop of the run's clock: token ``row`` and ``index`` into ``stops``."""
    row: int
    index: int
    label: str


class LiveTimeline(Model):
    """The run's clock: every row's stops (``stops``, one program per row), the cursor and the
    run root (the first stop the run can step back to), and per-row edit marks. ``limits``:
    ``layers``, ``max_positions``, ``heads``, ``kv_heads``."""
    session: str
    where: str | None
    stops: list[str]
    cursor: LiveStop | None
    root: LiveStop | None
    rows: list[LiveRow]
    limits: dict[str, int]


class LiveCircuit(Model):
    """One circuit of the run's ledger (the ``circuit`` op): its id, consumer, ``k`` (top) and
    ``depth``, node ``count``, the cursor it was built at, its sealed record id, how many times
    it was tested and the verdict text of the last test (None: never tested)."""
    id: int
    consumer: str
    k: int
    depth: int
    count: int
    where: str
    record: str
    tests: int
    verdict: str | None


class LiveForecast(Model):
    """One sealed forecast of the run (the ``forecast`` op), as ``analysis.forecast.view`` gives
    it: target, status (``open`` until ``verify`` closes it), where it was sealed, its record
    id, the predicted summary and bound (labelled numbers, premises left out), the number of
    identities in its method chain, and the last ``verify`` verdict (None before one)."""
    id: int
    target: str
    status: str
    sealed_at: str
    record: str
    steps: int
    predicted: dict[str, Any]
    bound: dict[str, Any]
    verdict: dict[str, Any] | None


class LiveSymbol(Model):
    """A user symbol declared on the run (the ``symbol`` op): its name, the addresses it was
    declared over and the token rows they cover."""
    name: str
    definitions: list[str]
    rows: list[int]


class LiveLedger(Model):
    """What this run has built: circuits, sealed forecasts and declared symbols, the latest of
    each (``more`` names what was left out). ``seq`` is the op count it was read at."""
    session: str
    seq: int
    where: str | None
    circuits: list[LiveCircuit]
    forecasts: list[LiveForecast]
    symbols: list[LiveSymbol]
    more: list[Omitted]


class Event(Model):
    """``op``: a live op or console line completed (``data``: op, seq, source, where, stopped,
    state, stale views). Events carry no tensors; clients re-fetch the stale views."""
    type: Literal["hello", "heartbeat", "op"]
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
    "op": OpResult,
    "console": ConsoleResult,
    "interrupt": Interrupt,
    "live": LiveFrame,
    "live-timeline": LiveTimeline,
    "live-items": LiveItems,
    "live-ledger": LiveLedger,
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
            OpRequest, OpResult, ConsoleRequest, ConsoleResult, Interrupt, LiveUpstream, LiveEdit,
            LivePort, LiveCheckpoint, LiveFrame, LiveItems, LiveRow, LiveStop, LiveTimeline,
            LiveCircuit, LiveForecast, LiveSymbol, LiveLedger, Event]
