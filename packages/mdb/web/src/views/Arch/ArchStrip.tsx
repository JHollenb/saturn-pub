// A compact, always-visible map of the model: token rows (columns) x the layer program (the row
// input, each layer, the output), with the current layer's port program expanded beside it.
// It shows where the cursor is, which cells are materialised (from the live clock: stops
// computed per row), where edits and checkpoints sit, and the selected circuit's nodes. A cell
// that is not materialised is drawn empty, never as zero. Clicking a cell asks to seek there.
import type { LiveCheckpoint, LiveFrame, LiveTimeline } from '../../api/types.gen';
import { srcOf } from '../../chrome/labelled';

export interface ArchProps {
  frame: LiveFrame;
  timeline: LiveTimeline | null;
  /** Addresses of the selected circuit's nodes (empty: no overlay). */
  circuit: string[];
  /** Pages of the checkpoint list fetched past the frame. */
  checkpoints?: LiveCheckpoint[];
  busy: boolean;
  seek(target: string, from: string): void;
}

/** Lanes of the strip: the row input (`emb`), each layer, the row output (`fn`, `logits`). */
type Lane = { key: string; label: string; layer: number | null; stops: string[] };

export function lanes(stops: string[], layers: number): Lane[] {
  const out: Lane[] = [{ key: 'in', label: 'in', layer: null, stops: [] }];
  for (let l = 0; l < layers; l += 1) out.push({ key: `L${l}`, label: `L${l}`, layer: l, stops: [] });
  out.push({ key: 'out', label: 'out', layer: null, stops: [] });
  const first = stops.findIndex((s) => /^L\d+\./.test(s));
  stops.forEach((stop, i) => {
    const m = /^L(\d+)\./.exec(stop);
    if (m) out[1 + Number(m[1])]?.stops.push(stop);
    else (first < 0 || i < first ? out[0]! : out[out.length - 1]!).stops.push(stop);
  });
  return out;
}

/** The lane key of an address: `L<n>` for a layer port, `in` for the row input (`start`,
 *  `emb`), `out` for the rest (`fn`, `logits`, `sample`, margins). Null for a non-row address. */
export function laneOf(addr: string): { row: number; lane: string } | null {
  const m = /^t(\d+)\.(?:L(\d+)\.|([A-Za-z_]+))/.exec(addr);
  if (!m) return null;
  if (m[2] !== undefined) return { row: Number(m[1]), lane: `L${m[2]}` };
  return { row: Number(m[1]), lane: m[3] === 'start' || m[3] === 'emb' ? 'in' : 'out' };
}

export type CellState = 'full' | 'partial' | 'cache' | 'empty';

/** How much of a lane a row has computed: every stop, some, none, or (a prefilled row whose
 *  ports were never stepped) its cache only. */
export function cellState(count: number | null, stops: string[], lane: Lane): CellState {
  if (count === null) return lane.layer === null ? 'empty' : 'cache';
  const done = new Set(stops.slice(0, count));
  const hit = lane.stops.filter((s) => done.has(s)).length;
  if (!lane.stops.length || hit === 0) return 'empty';
  return hit === lane.stops.length ? 'full' : 'partial';
}

function seekTarget(row: number, lane: Lane): string {
  if (lane.layer !== null) return `t${row}.L${lane.layer}.h`;
  return lane.key === 'in' ? `t${row}.emb` : `t${row}.logits`;
}

export function ArchStrip({ frame, timeline, circuit, checkpoints = [], busy, seek }: ArchProps) {
  if (!timeline || !timeline.cursor || frame.where === null) {
    return (
      <div className="arch na" data-testid="arch">
        architecture: no run yet
      </div>
    );
  }
  const layers = timeline.limits.layers ?? frame.limits.layers ?? 0;
  const all = lanes(timeline.stops, layers);
  const marks = new Map<string, string[]>();
  const mark = (at: { row: number; lane: string } | null, text: string) => {
    if (!at) return;
    const key = `${at.row}|${at.lane}`;
    marks.set(key, [...(marks.get(key) ?? []), text]);
  };
  for (const r of timeline.rows) for (const e of r.edits) mark(laneOf(`t${r.row}.${e}`), `edit ${e}`);
  for (const c of [...frame.checkpoints, ...checkpoints]) mark(laneOf(c.where), `checkpoint ${c.name}`);
  const nodes = new Set<string>();
  const reads = new Set<string>();
  for (const addr of circuit) {
    const at = laneOf(addr);
    if (!at) continue;
    nodes.add(`${at.row}|${at.lane}`);
    const src = srcOf(addr);
    if (src !== null && at.lane.startsWith('L')) reads.add(`${src}|${at.lane}`);
  }
  const here = timeline.cursor;
  const hereLane = laneOf(`t${here.row}.${here.label}`)?.lane ?? null;
  const circuitPorts = new Set(circuit.map((a) => a.split('[')[0]));
  return (
    <div className="arch" data-testid="arch" aria-label="architecture">
      <div className="arch-grid-wrap">
        <table className="arch-grid" role="grid" aria-label="token rows by layers">
          <thead>
            <tr>
              <th />
              {timeline.rows.map((r) => (
                <th key={r.row} scope="col" className={r.row === here.row ? 'arch-here-col' : undefined}>
                  t{r.row}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {all.map((lane) => (
              <tr key={lane.key}>
                <th scope="row" className="mono">
                  {lane.label}
                </th>
                {timeline.rows.map((r) => {
                  const state = cellState(r.count, timeline.stops, lane);
                  const key = `${r.row}|${lane.key}`;
                  const cursor = r.row === here.row && lane.key === hereLane;
                  const notes = marks.get(key) ?? [];
                  const inCircuit = nodes.has(key);
                  const readFrom = reads.has(key);
                  const target = seekTarget(r.row, lane);
                  const title = [
                    `t${r.row} ${lane.label}: ${STATE_TEXT[state]}`,
                    cursor ? 'cursor here' : '',
                    ...notes,
                    inCircuit ? 'a node of the selected circuit' : '',
                    readFrom ? 'read by the selected circuit' : '',
                    `click: seek ${target} (asks first)`,
                  ]
                    .filter(Boolean)
                    .join('\n');
                  return (
                    <td key={r.row} className="arch-td">
                      <button
                        type="button"
                        className={`arch-cell arch-${state}${cursor ? ' arch-cursor' : ''}${inCircuit ? ' arch-node' : ''}${readFrom ? ' arch-read' : ''}`}
                        disabled={busy}
                        title={title}
                        aria-label={title.replace(/\n/g, '; ')}
                        data-cell={key}
                        data-state={state}
                        onClick={() => seek(target, `architecture cell t${r.row} ${lane.label}`)}
                      >
                        {notes.some((n) => n.startsWith('edit')) ? '✎' : ''}
                        {notes.some((n) => n.startsWith('checkpoint')) ? '⚑' : ''}
                        {inCircuit ? '◆' : readFrom ? '○' : ''}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="arch-ports" aria-label={`port program of t${frame.row}.L${frame.layer}`}>
        <span className="faint mono">
          t{frame.row ?? '?'}.L{frame.layer ?? '?'}
        </span>
        {frame.ports.map((p) => {
          const inCircuit = circuitPorts.has(p.addr);
          return (
            <button
              key={p.port}
              type="button"
              className={`arch-port${p.computed ? ' arch-full' : ' arch-empty'}${p.cursor ? ' arch-cursor' : ''}${inCircuit ? ' arch-node' : ''}`}
              disabled={busy}
              title={`${p.addr}: ${p.computed ? 'computed' : 'not computed yet'}${p.cursor ? ' (cursor)' : ''}${inCircuit ? '; a node of the selected circuit' : ''}; click: seek (asks first)`}
              onClick={() => seek(p.addr, `port ${p.addr}`)}
            >
              {p.port}
              {inCircuit ? '◆' : ''}
            </button>
          );
        })}
      </div>
      <div className="faint arch-legend">
        <span className="arch-cell arch-full" /> computed <span className="arch-cell arch-partial" />{' '}
        part computed <span className="arch-cell arch-cache" /> cache only (prefilled){' '}
        <span className="arch-cell arch-empty" /> not computed (no value) · ✎ edit · ⚑ checkpoint · ◆
        circuit node · ○ read by the circuit · outlined: cursor
      </div>
    </div>
  );
}

const STATE_TEXT: Record<CellState, string> = {
  full: 'computed',
  partial: 'partly computed',
  cache: 'cache only (prefilled, not stepped)',
  empty: 'not computed (no value)',
};
