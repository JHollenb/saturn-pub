// The circuit walk (the `circuit` op): build a circuit for a consumer, draw it as a layered
// graph (consumer left, then layer components, heads, source rows), walk it with the keyboard
// (up/down siblings, left/right along an edge, Enter asks to seek to the node), test it by
// native ablation, and list the circuits this run already built. Pure: actions go through `act`.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { LiveCircuit } from '../../api/types.gen';
import { EvidenceMark } from '../../chrome/EvidenceMark';
import { Count } from '../../chrome/NumView';
import { OpValue } from '../../chrome/OpValue';
import { markStyle } from '../../chrome/evidence';
import { at, int, isRec, seekable, srcOf, str, strs, valOf, type Rec } from '../../chrome/labelled';
import type { CircuitEntry, FocusRead, Slot } from '../../state/inspect';
import { SlotState } from '../Live/SlotState';
import {
  LABEL_W,
  LABEL_X,
  MIN_SCALE,
  NODE_R,
  ROW_H,
  familyOf,
  layoutCircuit,
  rootScroll,
  scaleFor,
  walk,
  type CircuitLayout,
  type Placed,
  type WalkKey,
} from './layout';
import { pagedCounts, spliceBody, usePages, type PageSource } from './paging';

export interface CircuitActions {
  build(consumer: string, top: number, depth: number): void;
  /** Select a circuit; one this page has not seen is fetched with `circuit show ID`. */
  select(id: number): void;
  test(id: number): void;
  node(id: string | null): void;
  seek(target: string, from: string): void;
  overlay(on: boolean): void;
  interrupt(): void;
}

export interface CircuitViewProps {
  ledger: LiveCircuit[];
  circuits: ReadonlyMap<number, CircuitEntry>;
  selected: number | null;
  node: string | null;
  focus: FocusRead | null;
  slot: Slot;
  testSlot: Slot;
  busy: boolean;
  overlay: boolean;
  act: CircuitActions;
}

export function CircuitView(props: CircuitViewProps) {
  const { circuits, selected, act } = props;
  const entry = selected !== null ? (circuits.get(selected) ?? null) : null;
  return (
    <div className="circuit-view" data-testid="circuit-view">
      <BuildForm busy={props.busy} act={act} focus={props.focus} entry={entry} />
      <SlotState
        slot={props.slot}
        what="circuit"
        onInterrupt={act.interrupt}
        idle="No circuit built in this page yet: name a consumer and press build, or show one from the list."
      />
      <Ledger {...props} />
      {props.focus && <FocusLine focus={props.focus} entry={entry} />}
      {entry ? (
        <>
          <CircuitHead entry={entry} />
          <Graph key={entry.id} entry={entry} source={{ kind: 'live', id: entry.id }} {...props} />
          <TestPanel entry={entry} {...props} />
        </>
      ) : (
        selected !== null && (
          <div className="na" data-testid="circuit-unseen">
            circuit {selected}: not loaded in this page; press show in the list.
          </div>
        )
      )}
    </div>
  );
}

/** A circuit record's sealed body drawn read-only (recorded mode): the head, the same layered
 *  graph and keyboard walk as the Live tab, no seek, no test. */
export function StoredCircuit({
  id,
  body,
  source = null,
}: {
  id: number;
  body: Rec;
  /** Where its child pages are read from (`/stored/{record}?node=`); None: the body is whole. */
  source?: PageSource | null;
}) {
  const [node, setNode] = useState<string | null>(null);
  const entry: CircuitEntry = useMemo(() => ({ id, body, test: null, ablated: null }), [id, body]);
  const circuits = useMemo(() => new Map([[id, entry]]), [id, entry]);
  const act: CircuitActions = useMemo(
    () => ({
      build: () => undefined,
      select: () => undefined,
      test: () => undefined,
      node: setNode,
      seek: () => undefined,
      overlay: () => undefined,
      interrupt: () => undefined,
    }),
    [],
  );
  return (
    <div className="circuit-view" data-testid="stored-circuit">
      <CircuitHead entry={entry} />
      <Graph
        key={`${id}:${source?.kind === 'stored' ? source.record : ''}`}
        entry={entry}
        source={source}
        ledger={[]}
        circuits={circuits}
        selected={id}
        node={node}
        focus={null}
        slot={{ state: 'idle' }}
        testSlot={{ state: 'idle' }}
        busy={false}
        overlay={false}
        act={act}
        readOnly
      />
    </div>
  );
}

// --------------------------------------------------------------------------- build

function BuildForm({
  busy,
  act,
  focus,
  entry,
}: {
  busy: boolean;
  act: CircuitActions;
  focus: FocusRead | null;
  entry: CircuitEntry | null;
}) {
  const [consumer, setConsumer] = useState('');
  const [top, setTop] = useState(2);
  const [depth, setDepth] = useState(3);
  const fallback = str(entry?.body.consumer) ?? '';
  const target = consumer.trim() || fallback;
  return (
    <form
      className="circuit-build"
      aria-label="build a circuit"
      onSubmit={(e) => {
        e.preventDefault();
        if (target) act.build(target, top, depth);
      }}
    >
      <span className="muted">circuit</span>
      <input
        className="mono"
        aria-label="circuit consumer"
        placeholder={fallback || 'M[4,9] | logits[7] | L1.o'}
        value={consumer}
        onChange={(e) => setConsumer(e.target.value)}
        size={14}
      />
      <label>
        top K{' '}
        <select aria-label="top K" value={top} onChange={(e) => setTop(Number(e.target.value))}>
          {[1, 2, 3, 4, 5, 6, 7, 8].map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </label>
      <label>
        depth{' '}
        <select aria-label="depth" value={depth} onChange={(e) => setDepth(Number(e.target.value))}>
          <option value={1}>1 layer terms</option>
          <option value={2}>2 heads, neurons</option>
          <option value={3}>3 source rows</option>
        </select>
      </label>
      <button type="submit" disabled={busy || !target}>
        build
      </button>
      {focus && <span className="faint">(depth 3 shows which source rows each head reads)</span>}
    </form>
  );
}

function Ledger({ ledger, circuits, selected, busy, overlay, act }: CircuitViewProps) {
  return (
    <section className="circuit-ledger" aria-label="circuits built in this run">
      <h3>Circuits built in this run</h3>
      {ledger.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>id</th>
                <th>consumer</th>
                <th>K</th>
                <th>D</th>
                <th>nodes</th>
                <th>built at</th>
                <th>last test</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {ledger.map((c) => (
                <tr
                  key={c.id}
                  className={c.id === selected ? 'selected-row' : undefined}
                  data-circuit={c.id}
                >
                  <td>
                    <Count n={c.id} />
                  </td>
                  <td className="mono">
                    {c.consumer}
                    {c.from_previous_run && (
                      <>
                        {' '}
                        <span
                          className="chip faint"
                          data-testid="previous-run"
                          title="built by an earlier run (before start/restart): its cursor and source rows are that run's prompt, not this one's"
                        >
                          previous run
                        </span>
                      </>
                    )}
                  </td>
                  <td>
                    <Count n={c.k} />
                  </td>
                  <td>
                    <Count n={c.depth} />
                  </td>
                  <td>
                    <Count n={c.count} />
                  </td>
                  <td className="mono faint">{c.where}</td>
                  <td>
                    {c.verdict ? (
                      <span className="chip verdict-chip">{c.verdict}</span>
                    ) : (
                      <span className="faint">not tested</span>
                    )}
                  </td>
                  <td>
                    <button
                      type="button"
                      disabled={busy && !circuits.has(c.id)}
                      aria-pressed={c.id === selected}
                      onClick={() => act.select(c.id)}
                      title={circuits.has(c.id) ? 'draw it' : 'circuit show ID (fetch and draw it)'}
                    >
                      {c.id === selected ? 'shown' : 'show'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="faint">none yet</div>
      )}
      <label className="muted">
        <input type="checkbox" checked={overlay} onChange={(e) => act.overlay(e.target.checked)} />{' '}
        highlight the selected circuit's nodes on the pipeline and the architecture strip
      </label>
    </section>
  );
}

function CircuitHead({ entry }: { entry: CircuitEntry }) {
  const b = entry.body;
  const closure = b.closure;
  return (
    <div className="circuit-head">
      circuit <Count n={entry.id} /> <code>{str(b.consumer) ?? '?'}</code> · top{' '}
      <Count n={int(b.k) ?? 0} /> depth <Count n={int(b.depth) ?? 0} /> ·{' '}
      <Count n={int(b.count) ?? 0} /> nodes · built at <code>{str(b.where) ?? '?'}</code>
      <div>
        closure <OpValue value={at(closure, 'value')} what="closure" />{' '}
        <span className="faint">within</span> <OpValue value={at(closure, 'bound')} what="bound" />
        {at(closure, 'within') === true
          ? ' within'
          : at(closure, 'within') === false
            ? ' OUTSIDE'
            : ''}
        {str(at(closure, 'meaning')) && (
          <span className="faint"> ({str(at(closure, 'meaning'))})</span>
        )}
      </div>
      {b.remainders_within !== undefined && (
        <div className="faint">
          remainders{' '}
          {b.remainders_within === true
            ? 'all within their rounding bounds'
            : b.remainders_within === false
              ? 'NOT all within their rounding bounds'
              : b.unexamined === true
                ? 'not all checked: the subtrees under the +N more groups were never expanded'
                : 'within: not stated (a bound was refused)'}
          {b.unexamined === true && b.unexamined_count !== undefined && (
            <>
              {' '}
              (<OpValue value={b.unexamined_count} what="unexamined_count" compact /> unexamined)
            </>
          )}
        </div>
      )}
      {isRec(b.edits) && (
        <div className="faint">
          edit nodes: <Count n={int(b.edits.count) ?? 0} />
          {str(b.edits.meaning) ? ` (${str(b.edits.meaning)})` : ''}
        </div>
      )}
    </div>
  );
}

/** Where a read the Symbols view pointed at sits in the selected circuit. */
function readNodes(
  entry: CircuitEntry | null,
  focus: FocusRead | null,
  layout: CircuitLayout | null,
) {
  if (!entry || !focus || !layout) return new Set<string>();
  const m = /^t(\d+)\.L(\d+)\.h(\d+)$/.exec(focus.reader);
  if (!m) return new Set<string>();
  const [, row, layer, head] = m;
  const out = new Set<string>();
  for (const p of layout.placed) {
    const a = p.node.address;
    if (a === `t${row}.L${layer}.A[h${head}]` && p.node.kind === 'head') out.add(p.node.id);
    const src = srcOf(a);
    if (src !== null && focus.rows.includes(src) && a.startsWith(`t${row}.L${layer}.p[h${head},`))
      out.add(p.node.id);
  }
  return out;
}

function FocusLine({ focus, entry }: { focus: FocusRead; entry: CircuitEntry | null }) {
  const layout = useMemo(() => (entry ? layoutCircuit(entry.body, [entry.body]) : null), [entry]);
  const found = readNodes(entry, focus, layout);
  return (
    <div className="circuit-focus" data-testid="circuit-focus">
      read <code>{focus.reader}</code> of <code>{focus.binding}</code> (rows{' '}
      {focus.rows.map((r) => `t${r}`).join(', ')}):{' '}
      {!entry ? (
        <span className="faint">
          no circuit shown; build one for the consumer to place this read.
        </span>
      ) : found.size ? (
        <span>
          highlighted in circuit {entry.id} ({found.size} node{found.size === 1 ? '' : 's'}).
        </span>
      ) : (
        <span className="faint">
          not among circuit {entry.id}'s listed nodes (it may sit in a +N more group or a paged
          group not expanded, or need depth 3 or a larger K).
        </span>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- graph

const KEYS: Record<string, WalkKey> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
};

function Graph({
  entry,
  circuits,
  node,
  focus,
  busy,
  act,
  readOnly = false,
  source,
}: {
  entry: CircuitEntry;
  readOnly?: boolean;
  /** Where child pages are fetched from; null: this body has none to fetch. */
  source: PageSource | null;
} & CircuitViewProps) {
  const consumer = str(entry.body.consumer);
  const paging = usePages(source ?? { kind: 'live', id: entry.id });
  const body = useMemo(() => spliceBody(entry.body, paging.pages), [entry.body, paging.pages]);
  const family = useMemo(
    () =>
      familyOf([...circuits.values()], consumer).map((c) => (c.id === entry.id ? body : c.body)),
    [circuits, consumer, entry.id, body],
  );
  const layout = useMemo(() => layoutCircuit(body, family), [body, family]);
  const ablated = useMemo(() => ablatedIds(entry), [entry]);
  const reads = readNodes(entry, focus, layout);
  const box = useRef<HTMLDivElement>(null);
  const [boxWidth, setBoxWidth] = useState<number | null>(null);
  const [zoom, setZoom] = useState<'fit' | number>('fit');
  useEffect(() => {
    const el = box.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const watch = new ResizeObserver(() => setBoxWidth(el.clientWidth));
    watch.observe(el);
    setBoxWidth(el.clientWidth);
    return () => watch.disconnect();
  }, []);
  const scale = scaleFor(zoom, layout.width, boxWidth);
  const width = Math.ceil(layout.width * scale);
  const height = Math.ceil(layout.height * scale);
  const overflows = boxWidth !== null && width > boxWidth + 1;
  const current = layout.placed.find((p) => p.node.id === node) ?? null;
  useEffect(() => {
    // the keyboard walk keeps the walked node (and its label) in view
    const el = box.current;
    if (!current || !el || typeof el.scrollTo !== 'function') return;
    const left = (current.x - NODE_R - 8) * scale;
    const right = (current.x + LABEL_X + LABEL_W + 8) * scale;
    const top = (current.y - ROW_H) * scale;
    const bottom = (current.y + ROW_H) * scale;
    let x = el.scrollLeft;
    let y = el.scrollTop;
    if (left < x) x = left;
    else if (right > x + el.clientWidth) x = right - el.clientWidth;
    if (top < y) y = top;
    else if (bottom > y + el.clientHeight) y = bottom - el.clientHeight;
    if (x !== el.scrollLeft || y !== el.scrollTop)
      el.scrollTo({ left: Math.max(0, x), top: Math.max(0, y) });
  }, [current, scale]);
  // Show (a new circuit in this pane) and fit bring the consumer into view: the pane is one DOM
  // element reused across circuits, so without this a circuit opens where the previous one was
  // scrolled (often its tail), and a zoom keeps a scroll offset that no longer holds the root.
  const [reveal, setReveal] = useState(0);
  const bar = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof el.scrollTo !== 'function') return;
    el.scrollTo(rootScroll(layout, scale));
    // and when the pane's scroll parent (the Live tab) has the root off screen, it scrolls the
    // graph's zoom bar to its top: the root sits just under it
    const root = el.querySelector('[data-kind="consumer"]');
    const outer = scrollParent(el.parentElement);
    if (root && outer && bar.current && typeof bar.current.scrollIntoView === 'function') {
      const r = root.getBoundingClientRect();
      const o = outer.getBoundingClientRect();
      const top = Math.max(o.top, 0);
      const bottom = Math.min(o.bottom, window.innerHeight);
      if (r.top < top || r.bottom > bottom) bar.current.scrollIntoView({ block: 'start' });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- on show and on fit/zoom only
  }, [entry.id, reveal]);
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const key = KEYS[e.key];
    if (key) {
      e.preventDefault();
      // walking into a paged group fetches its page (its children were examined, stored apart)
      if (key === 'right' && current?.node.paged && !current.node.children.length && source)
        void paging.load(current.node.id);
      act.node(walk(layout, node, key));
    } else if (e.key === 'Enter' && current && !readOnly) {
      e.preventDefault();
      const target = seekable(current.node.address);
      if (target && !busy) act.seek(target, `circuit ${entry.id} node ${current.node.name}`);
    } else if (e.key === 'Escape') {
      act.node(null);
    }
  };
  // a zoom step keeps the walked node in view (the walk effect above); with none, the root
  const zoomTo = (to: 'fit' | number) => {
    setZoom(to);
    if (to === 'fit' || !current) setReveal((n) => n + 1);
  };
  const step = (by: number) => zoomTo(Math.round(Math.max(0.4, Math.min(2, scale + by)) * 10) / 10);
  return (
    <div className="circuit-graph-wrap">
      <div ref={bar} className="circuit-zoom" role="group" aria-label="graph zoom">
        <button type="button" aria-pressed={zoom === 'fit'} onClick={() => zoomTo('fit')}>
          fit
        </button>
        <button type="button" onClick={() => step(-0.1)} aria-label="zoom out">
          −
        </button>
        <button type="button" onClick={() => step(0.1)} aria-label="zoom in">
          +
        </button>
        <button type="button" aria-pressed={zoom === 1} onClick={() => zoomTo(1)}>
          100%
        </button>
        <span className="faint" data-testid="graph-scale">
          {Math.round(scale * 100)}%
          {zoom === 'fit' && scale <= MIN_SCALE && overflows
            ? ` (smallest readable scale; the graph is wider than the pane)`
            : ''}
        </span>
        {overflows && (
          <span className="scroll-cue" data-testid="graph-scroll-cue">
            scroll sideways for more columns →
          </span>
        )}
      </div>
      <div
        ref={box}
        className={overflows ? 'circuit-graph circuit-graph-overflow' : 'circuit-graph'}
        tabIndex={0}
        data-own-keys=""
        role="tree"
        aria-label={`circuit ${entry.id} for ${consumer ?? '?'}: arrows walk${readOnly ? '' : ', Enter asks to seek'}`}
        aria-activedescendant={current ? `cnode-${entry.id}-${current.node.id}` : undefined}
        onKeyDown={onKey}
        onFocus={() => {
          if (!node && layout.placed[0]) act.node(layout.placed[0].node.id);
        }}
        data-testid="circuit-graph"
      >
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          role="presentation"
          style={{ display: 'block' }}
        >
          {layout.edges.map((e) => (
            <EdgeLine key={`${e.from.node.id}-${e.to.node.id}`} e={e} />
          ))}
          {layout.placed.map((p) => (
            <NodeMark
              key={p.node.id}
              id={`cnode-${entry.id}-${p.node.id}`}
              p={p}
              selected={p.node.id === node}
              read={reads.has(p.node.id)}
              ablated={ablated.get(p.node.id) ?? null}
              paged={p.node.paged !== null && p.node.children.length === 0}
              onSelect={() => act.node(p.node.id)}
            />
          ))}
        </svg>
      </div>
      <GraphLegend readOnly={readOnly} />
      {current && (
        <NodeLine
          p={current}
          entry={entry}
          busy={busy}
          act={act}
          readOnly={readOnly}
          paging={source ? paging : null}
        />
      )}
    </div>
  );
}

/** The nearest ancestor that scrolls vertically (or the document's scrolling element). */
function scrollParent(el: HTMLElement | null): HTMLElement | null {
  for (let at = el; at; at = at.parentElement) {
    const y = getComputedStyle(at).overflowY;
    if ((y === 'auto' || y === 'scroll') && at.scrollHeight > at.clientHeight) return at;
  }
  return (document.scrollingElement as HTMLElement | null) ?? null;
}

/** The set the last test ablated each node in (the op's `ablated` node ids), and the nodes it
 *  excluded from both sets (`excluded.nodes`). */
function ablatedIds(entry: CircuitEntry): Map<string, string> {
  const out = new Map<string, string>();
  for (const set of ['retained', 'complement']) {
    for (const id of strs(at(entry.ablated, set))) out.set(id, set);
  }
  for (const id of strs(at(entry.test, 'excluded', 'nodes'))) out.set(id, 'excluded');
  return out;
}

function EdgeLine({ e }: { e: CircuitLayout['edges'][number] }) {
  // an edge leaves its parent from the end of the parent's reserved label box, so it never
  // crosses a label
  const x1 = e.from.x + LABEL_X + LABEL_W;
  const x2 = e.to.x - NODE_R;
  const mid = (x1 + x2) / 2;
  const d = `M${x1},${e.from.y} C${mid},${e.from.y} ${mid},${e.to.y} ${x2},${e.to.y}`;
  const stroke =
    e.sign === 'pos' ? 'var(--sign-pos)' : e.sign === 'neg' ? 'var(--sign-neg)' : 'var(--ink-3)';
  return (
    <path
      d={d}
      fill="none"
      stroke={stroke}
      strokeWidth={e.width ?? 1}
      strokeDasharray={e.width === null ? '1 4' : e.sign === 'neg' ? '6 3' : undefined}
      opacity={0.8}
      data-sign={e.sign}
    >
      <title>{`${e.to.node.name} → ${e.from.node.name}: ${
        e.width === null
          ? 'no value'
          : e.sign === 'none'
            ? 'size (a norm: no sign)'
            : e.sign === 'neg'
              ? 'negative contribution'
              : 'positive contribution'
      }`}</title>
    </path>
  );
}

function NodeMark({
  id,
  p,
  selected,
  read,
  ablated,
  paged,
  onSelect,
}: {
  id: string;
  p: Placed;
  selected: boolean;
  read: boolean;
  ablated: string | null;
  /** Children examined but stored as a page, not fetched yet. */
  paged: boolean;
  onSelect: () => void;
}) {
  const { node, x, y } = p;
  const val = valOf(node.value);
  const ev =
    val.kind === 'num' ? val.n.ev : val.kind === 'summary' ? val.parts[0]![1].ev : val.n.ev;
  const hollowKind = node.kind === 'more' || node.kind === 'remainder';
  const style = hollowKind ? 'hollow' : markStyle(ev);
  const fill = style === 'solid' ? 'var(--ink-2)' : 'var(--panel)';
  const dash = hollowKind ? '3 2' : style === 'dashed-empty' ? '2 2' : undefined;
  const common = {
    stroke: 'var(--ink)',
    strokeWidth: selected ? 3 : 1.4,
    strokeDasharray: dash,
    fill,
  };
  return (
    <g
      id={id}
      role="treeitem"
      aria-selected={selected}
      aria-label={`${node.name} (${node.kind}) ${node.address}`}
      data-node={node.id}
      data-kind={node.kind}
      data-slot={`${p.col},${p.slot}`}
      onClick={onSelect}
      style={{ cursor: 'pointer' }}
    >
      <title>{`${node.name} · ${node.kind} · ${node.address}${node.note ? `\n${node.note}` : ''}${paged ? '\nchildren examined, stored as a page: walk right or expand to fetch it' : ''}`}</title>
      {(read || ablated) && (
        <rect
          x={x - NODE_R - 4}
          y={y - NODE_R - 4}
          width={2 * NODE_R + 8}
          height={2 * NODE_R + 8}
          fill="none"
          stroke={read ? 'var(--role-symbol)' : 'var(--role-intervention)'}
          strokeWidth={2}
          strokeDasharray={ablated === 'complement' && !read ? '2 2' : undefined}
        />
      )}
      {node.children.length > 0 && (
        // the node's label box is part of its edge: a faint rule from the node to where its
        // edges leave
        <line
          x1={x + NODE_R}
          x2={x + LABEL_X + LABEL_W}
          y1={y}
          y2={y}
          stroke="var(--ink-3)"
          strokeWidth={1}
          opacity={0.45}
        />
      )}
      {node.kind === 'consumer' ? (
        <rect x={x - NODE_R} y={y - NODE_R} width={2 * NODE_R} height={2 * NODE_R} {...common} />
      ) : node.kind === 'more' ? (
        <rect
          x={x - NODE_R - 2}
          y={y - NODE_R + 1}
          width={2 * NODE_R + 4}
          height={2 * NODE_R - 2}
          rx={4}
          {...common}
        />
      ) : node.kind === 'remainder' ? (
        <path
          d={`M${x},${y - NODE_R} L${x + NODE_R},${y} L${x},${y + NODE_R} L${x - NODE_R},${y}Z`}
          {...common}
        />
      ) : (
        <circle cx={x} cy={y} r={NODE_R} {...common} />
      )}
      {paged && (
        <text
          x={x - 4}
          y={y + NODE_R + 9}
          fontSize={9}
          fill="var(--ink-2)"
          data-testid="paged-mark"
        >
          ▸
        </text>
      )}
      <text
        x={x + LABEL_X}
        y={y - 3}
        fontSize={10}
        fill="var(--ink)"
        stroke="var(--panel)"
        strokeWidth={3}
        paintOrder="stroke"
      >
        {p.label}
      </text>
      <foreignObject x={x + LABEL_X - 2} y={y} width={LABEL_W} height={14}>
        <span className="cnode-value">
          <OpValue value={node.value} what="value" compact showReason={false} />
          {ablated === 'retained'
            ? ' ✕R'
            : ablated === 'complement'
              ? ' ✕C'
              : ablated
                ? ' excl.'
                : ''}
        </span>
      </foreignObject>
    </g>
  );
}

function GraphLegend({ readOnly = false }: { readOnly?: boolean }) {
  return (
    <div className="faint circuit-legend" data-testid="circuit-legend">
      <span>
        <svg width={28} height={8} aria-hidden="true">
          <line x1={0} x2={28} y1={4} y2={4} stroke="var(--sign-pos)" strokeWidth={3} />
        </svg>{' '}
        positive
      </span>{' '}
      <span>
        <svg width={28} height={8} aria-hidden="true">
          <line
            x1={0}
            x2={28}
            y1={4}
            y2={4}
            stroke="var(--sign-neg)"
            strokeWidth={3}
            strokeDasharray="6 3"
          />
        </svg>{' '}
        negative (dashed)
      </span>{' '}
      · width ∝ |exact contribution| · <EvidenceMark ev="measured" /> measured{' '}
      <EvidenceMark ev="derived-exact" /> derived · rounded box: +N more (their Σ) · diamond:
      remainder (measured, with its bound; never dropped) · ▸ paged: children examined, stored as
      a page (fetched on expand; not +N more, which is a sum never expanded) · square: consumer
      {readOnly
        ? ' · arrows walk (recorded: no seek, no test)'
        : ' · after a test: ✕R ablated with the retained set, ✕C with the complement · arrows walk, Enter asks to seek'}
    </div>
  );
}

function NodeLine({
  p,
  entry,
  busy,
  act,
  readOnly = false,
  paging,
}: {
  p: Placed;
  entry: CircuitEntry;
  busy: boolean;
  act: CircuitActions;
  readOnly?: boolean;
  paging: ReturnType<typeof usePages> | null;
}) {
  const target = readOnly ? null : seekable(p.node.address);
  const counts = p.node.children.length === 0 ? pagedCounts(p.node.paged) : null;
  const state = paging?.status.get(p.node.id);
  return (
    <div className="circuit-node-line" data-testid="circuit-node-line">
      <strong>{p.node.name}</strong> <span className="faint">{p.node.kind}</span>{' '}
      <code>{p.node.address}</code> <OpValue value={p.node.value} what="value" />
      {p.node.raw.bound !== undefined && (
        <>
          {' '}
          <span className="faint">bound</span> <OpValue value={p.node.raw.bound} what="bound" />
          {p.node.raw.within === true ? ' within' : p.node.raw.within === false ? ' OUTSIDE' : ''}
        </>
      )}
      {p.node.note && <span className="faint"> · {p.node.note}</span>}{' '}
      {counts && (
        <span data-testid="paged-node">
          <span className="chip">paged</span>{' '}
          <span className="faint">
            {counts.children} children examined ({counts.descendants} nodes below), stored as a
            page, not listed here
          </span>{' '}
          {paging && (
            <button
              type="button"
              disabled={state?.state === 'loading'}
              onClick={() => void paging.load(p.node.id)}
            >
              {state?.state === 'loading' ? 'fetching…' : 'expand'}
            </button>
          )}{' '}
          {state?.state === 'failed' && (
            <span className="na" data-testid="paged-failed">
              page refused: {state.reason}
            </span>
          )}
        </span>
      )}{' '}
      {target ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => act.seek(target, `circuit ${entry.id} node ${p.node.name}`)}
        >
          seek {target}…
        </button>
      ) : readOnly ? (
        <span className="faint">(recorded: no session to seek)</span>
      ) : (
        <span className="faint">(no occurrence to seek)</span>
      )}{' '}
      {!readOnly && <span className="faint">full record in the Inspector</span>}
    </div>
  );
}

// --------------------------------------------------------------------------- test

const SETS = ['retained', 'complement'] as const;
const TEST_KNOWN = new Set([
  'excluded',
  'base',
  'retained',
  'complement',
  'consumer',
  'id',
  'record',
  'rule',
  'tolerance',
  'verdict',
]);

const CIRCUIT_TEST_DISABLED =
  'circuit test is disabled: zero-ablating the complement says nothing on a real model (issue M4)';

function TestPanel({ entry, testSlot, act }: { entry: CircuitEntry } & CircuitViewProps) {
  const running = testSlot.state === 'running';
  const test = entry.test;
  return (
    <section className="circuit-test" aria-label="circuit test">
      <div>
        <button
          type="button"
          disabled
          onClick={() => act.test(entry.id)}
          title={CIRCUIT_TEST_DISABLED}
        >
          circuit test {entry.id}
        </button>{' '}
        <span className="faint" data-testid="test-disabled">
          {CIRCUIT_TEST_DISABLED}
        </span>
      </div>
      {(testSlot.state !== 'idle' || running) && (
        <SlotState slot={testSlot} what="test" onInterrupt={act.interrupt} idle="" />
      )}
      {test ? <TestResult test={test} /> : <div className="faint">not tested in this page</div>}
    </section>
  );
}

export function TestResult({ test }: { test: Rec }) {
  const notes = Object.entries(test).filter(
    ([k, v]) => !TEST_KNOWN.has(k) && (typeof v === 'string' || isRec(v)),
  );
  return (
    <div className="test-result" data-testid="test-result">
      <div>
        base <OpValue value={test.base} what="base" /> · tolerance{' '}
        <OpValue value={test.tolerance} what="tolerance" />
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>ablate</th>
              <th className="n">nodes</th>
              <th className="n">predicted Δ</th>
              <th className="n">measured Δ</th>
              <th className="n">consumer after</th>
              <th>edits</th>
            </tr>
          </thead>
          <tbody>
            {SETS.map((set) => {
              const s = isRec(test[set]) ? (test[set] as Rec) : null;
              if (!s) {
                return (
                  <tr key={set}>
                    <td>{set}</td>
                    <td colSpan={5}>
                      <OpValue value={undefined} what={`${set} ablation`} />
                    </td>
                  </tr>
                );
              }
              return (
                <tr key={set} data-set={set}>
                  <td>{set}</td>
                  <td className="n">{int(s.count) !== null ? <Count n={int(s.count)!} /> : '?'}</td>
                  <td className="n" title={str(s.prediction) ?? undefined}>
                    <OpValue value={s.predicted_delta} what="predicted_delta" />
                  </td>
                  <td className="n">
                    <OpValue value={s.delta} what="delta" />
                  </td>
                  <td className="n">
                    <OpValue value={s.measured} what="measured" />
                  </td>
                  <td className="mono faint edits">{strs(s.edits).join(', ')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {[
        ...new Set(SETS.map((set) => str(at(test, set, 'prediction'))).filter((p) => p !== null)),
      ].map((p) => (
        <div key={p} className="faint" data-testid="test-prediction">
          predicted Δ: {p}
        </div>
      ))}
      {isRec(test.excluded) && (
        <div data-testid="test-excluded">
          excluded from both sets: {strs(test.excluded.nodes).join(', ') || 'none'}
          {str(test.excluded.reason) && (
            <span className="faint"> — {str(test.excluded.reason)}</span>
          )}
        </div>
      )}
      <div>
        verdict:{' '}
        <span className="chip verdict-chip" data-testid="test-verdict">
          {str(test.verdict) ?? 'no verdict given'}
        </span>{' '}
        {str(test.rule) && <span className="faint">({str(test.rule)})</span>}
      </div>
      {notes.map(([k, v]) => (
        <div key={k} className="faint" data-field={k}>
          {k}: {typeof v === 'string' ? v : <OpValue value={v} what={k} />}
        </div>
      ))}
    </div>
  );
}
