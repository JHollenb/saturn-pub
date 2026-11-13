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
  COL_W,
  NODE_R,
  familyOf,
  layoutCircuit,
  walk,
  type CircuitLayout,
  type Placed,
  type WalkKey,
} from './layout';

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
          <Graph entry={entry} {...props} />
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
                  <td className="mono">{c.consumer}</td>
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
        <input
          type="checkbox"
          checked={overlay}
          onChange={(e) => act.overlay(e.target.checked)}
        />{' '}
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
        {at(closure, 'within') === true ? ' within' : at(closure, 'within') === false ? ' OUTSIDE' : ''}
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
              : 'within: not stated'}
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
function readNodes(entry: CircuitEntry | null, focus: FocusRead | null, layout: CircuitLayout | null) {
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
        <span className="faint">no circuit shown; build one for the consumer to place this read.</span>
      ) : found.size ? (
        <span>highlighted in circuit {entry.id} ({found.size} node{found.size === 1 ? '' : 's'}).</span>
      ) : (
        <span className="faint">
          not among circuit {entry.id}'s listed nodes (it may sit in a +N more group, or need depth
          3 or a larger K).
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
}: { entry: CircuitEntry } & CircuitViewProps) {
  const consumer = str(entry.body.consumer);
  const family = useMemo(
    () => familyOf([...circuits.values()], consumer).map((c) => c.body),
    [circuits, consumer],
  );
  const layout = useMemo(() => layoutCircuit(entry.body, family), [entry.body, family]);
  const ablated = useMemo(() => ablatedIds(entry), [entry]);
  const reads = readNodes(entry, focus, layout);
  const box = useRef<HTMLDivElement>(null);
  const current = layout.placed.find((p) => p.node.id === node) ?? null;
  useEffect(() => {
    // keep the walked node in view
    if (!current || !box.current || typeof box.current.scrollTo !== 'function') return;
    const el = box.current;
    if (current.x < el.scrollLeft || current.x > el.scrollLeft + el.clientWidth - 40)
      el.scrollTo({ left: Math.max(0, current.x - 40) });
  }, [current]);
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const key = KEYS[e.key];
    if (key) {
      e.preventDefault();
      act.node(walk(layout, node, key));
    } else if (e.key === 'Enter' && current) {
      e.preventDefault();
      const target = seekable(current.node.address);
      if (target && !busy) act.seek(target, `circuit ${entry.id} node ${current.node.name}`);
    } else if (e.key === 'Escape') {
      act.node(null);
    }
  };
  return (
    <div className="circuit-graph-wrap">
      <div
        ref={box}
        className="circuit-graph"
        tabIndex={0}
        data-own-keys=""
        role="tree"
        aria-label={`circuit ${entry.id} for ${consumer ?? '?'}: arrows walk, Enter asks to seek`}
        aria-activedescendant={current ? `cnode-${entry.id}-${current.node.id}` : undefined}
        onKeyDown={onKey}
        onFocus={() => {
          if (!node && layout.placed[0]) act.node(layout.placed[0].node.id);
        }}
        data-testid="circuit-graph"
      >
        <svg
          width={layout.width + COL_W}
          height={layout.height}
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
              onSelect={() => act.node(p.node.id)}
            />
          ))}
        </svg>
      </div>
      <GraphLegend />
      {current && <NodeLine p={current} entry={entry} busy={busy} act={act} />}
    </div>
  );
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
  const x1 = e.from.x + NODE_R;
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
  onSelect,
}: {
  id: string;
  p: Placed;
  selected: boolean;
  read: boolean;
  ablated: string | null;
  onSelect: () => void;
}) {
  const { node, x, y } = p;
  const val = valOf(node.value);
  const ev = val.kind === 'num' ? val.n.ev : val.kind === 'summary' ? val.parts[0]![1].ev : val.n.ev;
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
      <title>{`${node.name} · ${node.kind} · ${node.address}${node.note ? `\n${node.note}` : ''}`}</title>
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
      <text
        x={x + NODE_R + 4}
        y={y - 3}
        fontSize={10}
        fill="var(--ink)"
        stroke="var(--panel)"
        strokeWidth={3}
        paintOrder="stroke"
      >
        {p.label}
        {ablated === 'retained' ? ' ✕R' : ablated === 'complement' ? ' ✕C' : ablated ? ' (excluded)' : ''}
      </text>
      <foreignObject x={x + NODE_R + 2} y={y - 1} width={COL_W - NODE_R - 8} height={15}>
        <span className="cnode-value">
          <OpValue value={node.value} what="value" compact showReason={false} />
        </span>
      </foreignObject>
    </g>
  );
}

function GraphLegend() {
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
      remainder (measured, with its bound; never dropped) · square: consumer · after a test: ✕R
      ablated with the retained set, ✕C with the complement · arrows walk, Enter asks to seek
    </div>
  );
}

function NodeLine({
  p,
  entry,
  busy,
  act,
}: {
  p: Placed;
  entry: CircuitEntry;
  busy: boolean;
  act: CircuitActions;
}) {
  const target = seekable(p.node.address);
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
      {target ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => act.seek(target, `circuit ${entry.id} node ${p.node.name}`)}
        >
          seek {target}…
        </button>
      ) : (
        <span className="faint">(no occurrence to seek)</span>
      )}{' '}
      <span className="faint">full record in the Inspector</span>
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

function TestPanel({ entry, testSlot, busy, act }: { entry: CircuitEntry } & CircuitViewProps) {
  const running = testSlot.state === 'running';
  const test = entry.test;
  return (
    <section className="circuit-test" aria-label="circuit test">
      <div>
        <button
          type="button"
          disabled={busy}
          onClick={() => act.test(entry.id)}
          title="ablates the retained set and its complement natively (moves the session and restores it)"
        >
          circuit test {entry.id}
        </button>{' '}
        <span className="faint">
          native ablation of the retained nodes and of their complement; the session is restored
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
              <th>nodes</th>
              <th>predicted Δ</th>
              <th>measured Δ</th>
              <th>consumer after</th>
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
                  <td>{int(s.count) !== null ? <Count n={int(s.count)!} /> : '?'}</td>
                  <td title={str(s.prediction) ?? undefined}>
                    <OpValue value={s.predicted_delta} what="predicted_delta" />
                  </td>
                  <td>
                    <OpValue value={s.delta} what="delta" />
                  </td>
                  <td>
                    <OpValue value={s.measured} what="measured" />
                  </td>
                  <td className="mono faint">{strs(s.edits).join(', ')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {[...new Set(SETS.map((set) => str(at(test, set, 'prediction'))).filter((p) => p !== null))].map(
        (p) => (
          <div key={p} className="faint" data-testid="test-prediction">
            predicted Δ: {p}
          </div>
        ),
      )}
      {isRec(test.excluded) && (
        <div data-testid="test-excluded">
          excluded from both sets: {strs(test.excluded.nodes).join(', ') || 'none'}
          {str(test.excluded.reason) && <span className="faint"> — {str(test.excluded.reason)}</span>}
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
