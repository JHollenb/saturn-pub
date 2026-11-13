// The live debugger, drawn from the live frame and clock. Pure: every action goes through `act`,
// which the container (`Live.tsx`) wires to registry ops. Numbers come only from op results
// (`{value, authority}`) and server reductions (`Num`), always through NumView; a port the run
// has not computed shows no value at all.
import { useEffect, useRef, useState, type ReactNode } from 'react';
import type {
  LiveCheckpoint,
  LiveEdit,
  LiveForecast,
  LiveFrame,
  LivePort,
  LiveTimeline,
  Num,
  Refused,
} from '../../api/types.gen';
import { EvidenceMark } from '../../chrome/EvidenceMark';
import { ViewLegend } from './ViewLegend';
import { RefusedView } from '../../chrome/NotAvailable';
import { Count, NumView } from '../../chrome/NumView';
import { asNum } from '../../chrome/evidence';
import { OpValue } from '../../chrome/OpValue';
import type { Tab } from '../../state/inspect';
import type { ReduceEntry } from '../../state/reduce';
import {
  moreOf,
  summaryKey,
  type ConsoleLine,
  type Extra,
  type Outcome,
  type Summary,
} from '../../state/live';

export interface LiveActions {
  op(name: string, args?: Record<string, unknown>): void;
  command(line: string): void;
  interrupt(): void;
  expand(addr: string | null): void;
  /** Fetch the next page of a list the frame shortened (`more` field). */
  more(field: string): void;
  /** Show another view of the live session. */
  tab?(tab: Tab): void;
  /** Seal a forecast of a port (the `forecast` op). */
  forecast?(target: string): void;
  /** Verify forecasts (the `verify` op). */
  verify?(ids: number[]): void;
}

const TABS: [Tab, string][] = [
  ['pipeline', 'Pipeline'],
  ['symbols', 'Symbols'],
  ['graph', 'Circuits'],
  ['predict', 'Predict'],
];

/** Report a pipeline row's element so the page can ask only for rows on screen; returns the
 *  cleanup. Absent (tests, no IntersectionObserver): every row counts as shown. */
export type WatchRow = (addr: string, el: Element) => () => void;

export interface LiveViewProps {
  frame: LiveFrame | null;
  timeline: LiveTimeline | null;
  running: string | null;
  last: Outcome | null;
  refused: Refused | null;
  summaries: ReadonlyMap<string, Summary>;
  console: ConsoleLine[];
  expanded: string | null;
  /** The URL's address when it is not the live cursor (a pasted link). */
  urlAddr: string | null;
  /** Pages of shortened lists fetched so far, by `more` field. */
  extra?: ReadonlyMap<string, Extra>;
  watchRow?: WatchRow;
  act: LiveActions;
  /** The view shown below the controls; the other views are `panes`. Absent: the pipeline. */
  tab?: Tab;
  panes?: Partial<Record<Exclude<Tab, 'pipeline'>, ReactNode>>;
  /** The architecture strip (always visible) and a pending-seek confirmation line. */
  strip?: ReactNode;
  seekLine?: ReactNode;
  /** The run's sealed forecasts: the pipeline's forecast column (absent: no column). */
  forecasts?: LiveForecast[];
  /** Addresses of the selected circuit's nodes, marked on the pipeline. */
  circuitPorts?: ReadonlySet<string>;
  /** The shown view's "show labels" setting (authority chips beside every number). */
  labels?: boolean;
  onLabels?(on: boolean): void;
}

export function LiveView(props: LiveViewProps) {
  const { frame, running, act } = props;
  const extra = props.extra ?? new Map<string, Extra>();
  if (!frame) {
    // No frame (refused, e.g. over the cap, or not loaded yet): the controls and the console
    // still work; every op answers for itself.
    return (
      <div
      className={`live ${props.labels ? 'labels-on' : 'labels-off'}`}
      data-testid="live"
      data-running={running ? 'yes' : 'no'}
    >
        <div className="live-status" role="status">
          <span className="faint">{props.refused ? 'no live frame' : 'loading live session…'}</span>
          <span className="live-state">
            {running ? (
              <span data-testid="state-running">running {running}…</span>
            ) : (
              <OutcomeText last={props.last} />
            )}
          </span>
        </div>
        <Controls frame={null} running={running} act={act} urlAddr={null} />
        {props.refused && <RefusedView what="live frame" refused={props.refused} />}
        <Console lines={props.console} disabled={!!running} act={act} />
      </div>
    );
  }
  const hasRun = frame.where !== null;
  const tab = props.tab ?? 'pipeline';
  const pane = tab === 'pipeline' ? null : props.panes?.[tab];
  return (
    <div
      className={`live ${props.labels ? 'labels-on' : 'labels-off'}`}
      data-testid="live"
      data-running={running ? 'yes' : 'no'}
    >
      <Status {...props} frame={frame} />
      <Controls frame={frame} running={running} act={act} urlAddr={props.urlAddr} />
      {props.refused && <RefusedView what="request" refused={props.refused} />}
      {props.seekLine}
      {hasRun && props.strip}
      {props.panes && act.tab && (
        <div className="live-tabs" role="tablist" aria-label="live views">
          {TABS.map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="tab"
              id={`live-tab-${key}`}
              aria-selected={tab === key}
              className={tab === key ? 'live-tab here' : 'live-tab'}
              onClick={() => act.tab?.(key)}
            >
              {label}
            </button>
          ))}
        </div>
      )}
      {hasRun && props.onLabels && (
        <ViewLegend labels={!!props.labels} onLabels={props.onLabels} />
      )}
      {pane ? (
        <div role="tabpanel" aria-labelledby={`live-tab-${tab}`} className="live-pane">
          {pane}
        </div>
      ) : hasRun ? (
        <>
          {props.timeline && <Clock timeline={props.timeline} disabled={!!running} act={act} />}
          <Pipeline
            frame={frame}
            summaries={props.summaries}
            expanded={props.expanded}
            act={act}
            watchRow={props.watchRow}
            forecasts={props.forecasts}
            circuitPorts={props.circuitPorts}
            disabled={!!running}
          />
        </>
      ) : (
        <div className="na" data-testid="no-run">
          No run yet: give token ids (or text, when the model has a tokenizer) and press start.
        </div>
      )}
      {!pane && (
        <div className="live-tables">
          <Displays
            frame={frame}
            extra={extra}
            summaries={props.summaries}
            disabled={!!running}
            act={act}
          />
          <Breakpoints frame={frame} extra={extra} disabled={!!running} act={act} />
          <Edits frame={frame} extra={extra} disabled={!!running || !hasRun} act={act} />
          <Checkpoints frame={frame} extra={extra} disabled={!!running} act={act} />
        </div>
      )}
      <Console lines={props.console} disabled={!!running} act={act} />
    </div>
  );
}

/** The address a display names at the cursor: a row-relative address (`L0.h`) is read in the
 *  cursor's row; an absolute one (`t3.L0.a`) as given. */
export function displayAt(address: string, row: number | null): string | null {
  if (/^t\d+\./.test(address)) return address;
  return row === null ? null : `t${row}.${address}`;
}

/** The frame's displays (`display ADDR`): each address, where it reads at the cursor, its value
 *  when the pipeline already reduced that port, and print / undisplay. */
function Displays({
  frame,
  extra,
  summaries,
  disabled,
  act,
}: {
  frame: LiveFrame;
  extra: ReadonlyMap<string, Extra>;
  summaries: ReadonlyMap<string, Summary>;
  disabled: boolean;
  act: LiveActions;
}) {
  const [address, setAddress] = useState('');
  const items = [...frame.displays, ...(extra.get('displays')?.items ?? [])] as {
    id?: unknown;
    address?: unknown;
  }[];
  return (
    <section aria-label="displays" data-testid="displays">
      <h3>Displays</h3>
      {items.length ? (
        <table>
          <tbody>
            {items.map((d, i) => {
              const id = typeof d.id === 'number' ? d.id : null;
              const addr = typeof d.address === 'string' ? d.address : '?';
              const here = displayAt(addr, frame.row);
              const summary = here ? (summaries.get(summaryKey(frame.seq, here)) ?? null) : null;
              return (
                <tr key={id ?? `d${i}`} data-display={id ?? ''}>
                  <td>{id !== null ? <Count n={id} /> : '?'}</td>
                  <td className="mono">{addr}</td>
                  <td className="mono faint">{here && here !== addr ? `→ ${here}` : ''}</td>
                  <td className="n">
                    {summary ? (
                      <Reduced
                        entry={summary.stats}
                        pick={(r) => ('l2' in r ? r.l2 : null)}
                      />
                    ) : (
                      <span className="faint" title="not a port of the layer on screen">
                        print to read
                      </span>
                    )}
                  </td>
                  <td>
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() => act.command(`print ${here ?? addr}`)}
                    >
                      print
                    </button>
                    {id !== null && (
                      <button
                        type="button"
                        disabled={disabled}
                        onClick={() => act.op('undisplay', { ids: [id] })}
                      >
                        undisplay
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <div className="faint">none</div>
      )}
      <More frame={frame} field="displays" extra={extra} act={act} noun="displays" />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!address.trim()) return;
          act.op('display', { address: address.trim() });
          setAddress('');
        }}
      >
        <input
          className="mono"
          aria-label="display address"
          placeholder="L0.h (each stop) or t3.L0.a"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          size={14}
        />
        <button type="submit" disabled={disabled || !address.trim()}>
          display
        </button>
      </form>
    </section>
  );
}

/** "N more" for a list the frame shortened, with the pages fetched so far; nothing when the
 *  frame showed it whole. */
function More({
  frame,
  field,
  extra,
  act,
  noun,
}: {
  frame: LiveFrame;
  field: string;
  extra: ReadonlyMap<string, Extra>;
  act: LiveActions;
  noun: string;
}) {
  const more = moreOf(frame, field);
  if (!more) return null;
  const got = extra.get(field);
  const left = more.count - (got?.items.length ?? 0);
  return (
    <div className="live-more" data-testid={`more-${field}`}>
      {left > 0 ? (
        <button type="button" className="linklike" onClick={() => act.more(field)}>
          <Count n={left} /> more {noun} not shown — show more
        </button>
      ) : (
        <span className="faint">all shown</span>
      )}
      {got?.refused && <RefusedView what={`more ${noun}`} refused={got.refused} />}
    </div>
  );
}

// --------------------------------------------------------------------------- status

function Status({ frame, running, last }: LiveViewProps & { frame: LiveFrame }) {
  const stop = frame.stop as { operator?: string; proofs?: string[] } | null;
  return (
    <div className="live-status" role="status">
      <code className="live-where">{frame.where ?? 'no run'}</code>
      {stop?.operator && <span className="mono muted"> {stop.operator}</span>}
      {stop?.proofs?.map((p) => (
        <span key={p} className="chip">
          {p}
        </span>
      ))}
      <span className="live-state">
        {running ? (
          <span data-testid="state-running">running {running}…</span>
        ) : (
          <OutcomeText last={last} />
        )}
      </span>
    </div>
  );
}

function OutcomeText({ last }: { last: Outcome | null }) {
  if (!last) return <span className="faint">paused</span>;
  switch (last.kind) {
    case 'ok':
      return (
        <span data-testid="state-stopped" data-stopped={last.stopped ?? ''}>
          {last.op}
          {last.stopped ? ` stopped: ${last.stopped}` : ' done'}
        </span>
      );
    case 'unresolved':
      return (
        <span data-testid="state-unresolved" title={last.reason}>
          {last.op}: partly unresolved ({last.reason})
        </span>
      );
    case 'interrupted':
      return (
        <span data-testid="state-interrupted">
          {last.op} interrupted at {last.where}
        </span>
      );
    case 'refused':
      return (
        <span className="num-refused" data-testid="state-refused" data-ev="refused">
          {last.op} refused{last.type ? ` (${last.type})` : ''}: {last.reason}
        </span>
      );
  }
}

// --------------------------------------------------------------------------- controls

function Controls({
  frame,
  running,
  act,
  urlAddr,
}: {
  frame: LiveFrame | null;
  running: string | null;
  act: LiveActions;
  urlAddr: string | null;
}) {
  const [prompt, setPrompt] = useState('');
  const [until, setUntil] = useState('');
  const busy = !!running;
  const noRun = frame !== null && frame.where === null; // no frame: let the op answer
  const motion = (label: string, op: string, title: string) => (
    <button type="button" disabled={busy || noRun} onClick={() => act.op(op)} title={title}>
      {label}
    </button>
  );
  return (
    <div className="live-controls" role="toolbar" aria-label="stepping controls">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const text = prompt.trim();
          if (!text) return;
          act.op('start', { prompt: tokensOrText(text) });
        }}
      >
        <input
          className="mono"
          aria-label="prompt"
          placeholder="token ids: 1 5 9 3"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button type="submit" disabled={busy || !prompt.trim()}>
          start
        </button>
      </form>
      {motion('step', 'step', 'step one port')}
      {motion('next', 'next', 'to the end of the sublayer')}
      {motion('finish', 'finish', 'to the end of the layer')}
      {motion('continue', 'continue', 'to the next breakpoint or watch hit')}
      {motion('reverse-step', 'reverse-step', 'step back one port')}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (until.trim()) act.op('until', { target: until.trim() });
        }}
      >
        <input
          className="mono"
          aria-label="until address"
          placeholder="until ADDR (t3.L1.p)"
          value={until}
          onChange={(e) => setUntil(e.target.value)}
        />
        <button type="submit" disabled={busy || noRun || !until.trim()}>
          until
        </button>
      </form>
      <button
        type="button"
        disabled={!busy}
        onClick={() => act.interrupt()}
        title="Ctrl-C the running op"
      >
        interrupt
      </button>
      {urlAddr && !busy && !noRun && (
        <span className="chip" data-testid="url-addr">
          link names <code>{urlAddr}</code>
          <button type="button" onClick={() => act.op('until', { target: urlAddr })}>
            until there
          </button>
        </span>
      )}
    </div>
  );
}

/** `1 5 9 3` -> token ids; anything else is prompt text (the model's tokenizer reads it). */
export function tokensOrText(text: string): number[] | string {
  const words = text.split(/[\s,]+/).filter(Boolean);
  return words.length && words.every((w) => /^\d+$/.test(w)) ? words.map(Number) : text;
}

// --------------------------------------------------------------------------- clock

function Clock({
  timeline,
  disabled,
  act,
}: {
  timeline: LiveTimeline;
  disabled: boolean;
  act: LiveActions;
}) {
  const cursor = timeline.cursor;
  if (!cursor) return null;
  const row = timeline.rows.find((r) => r.row === cursor.row);
  const edited = new Set((row?.edits ?? []).map((e) => e.replace(' (staged)', '')));
  return (
    <div className="live-clock" aria-label="clock">
      <div className="faint">
        rows{' '}
        {timeline.rows.map((r) => (
          <span
            key={r.row}
            className={r.row === cursor.row ? 'chip live-row-here' : 'chip'}
            title={r.count === null ? 'prefilled (not stepped)' : `${r.count} stops computed`}
          >
            t{r.row}
            {r.token !== null && <span className="faint"> #{r.token}</span>}
            {r.edits.length > 0 && <span title={r.edits.join(', ')}> ✎</span>}
          </span>
        ))}
      </div>
      <div className="live-stops" role="list">
        {timeline.stops.map((label, i) => {
          const here = i === cursor.index;
          const done = i < cursor.index;
          const addr = `t${cursor.row}.${label}`;
          return (
            <button
              key={label}
              type="button"
              role="listitem"
              className={`live-stop${here ? ' here' : ''}${done ? ' done' : ''}`}
              data-edited={edited.has(label) ? 'yes' : undefined}
              disabled={disabled || i <= cursor.index}
              title={here ? `${addr} (cursor)` : done ? `${addr} (computed)` : `until ${addr}`}
              onClick={() => act.op('until', { target: addr })}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- pipeline

function Pipeline({
  frame,
  summaries,
  expanded,
  act,
  watchRow,
  forecasts,
  circuitPorts,
  disabled,
}: {
  frame: LiveFrame;
  summaries: ReadonlyMap<string, Summary>;
  expanded: string | null;
  act: LiveActions;
  watchRow?: WatchRow;
  forecasts?: LiveForecast[];
  circuitPorts?: ReadonlySet<string>;
  disabled: boolean;
}) {
  return (
    <div className="table-scroll" aria-label="pipeline">
      <table className="live-pipeline">
        <thead>
          <tr>
            <th aria-label="cursor" />
            <th>
              t{frame.row ?? '?'}.L{frame.layer ?? '?'}
            </th>
            <th>operator</th>
            <th>proof</th>
            <th>‖value‖</th>
            <th>top |value|</th>
            <th>edit</th>
            {forecasts && (
              <th title="sealed prediction (hollow) beside the native value once verified">
                forecast
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {frame.ports.map((p) => (
            <PortRow
              key={p.port}
              port={p}
              summary={summaries.get(summaryKey(frame.seq, p.addr)) ?? null}
              open={p.cursor || p.addr === expanded}
              act={act}
              watchRow={watchRow}
              forecasts={forecasts}
              inCircuit={circuitPorts?.has(p.addr) ?? false}
              disabled={disabled}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PortRow({
  port,
  summary,
  open,
  act,
  watchRow,
  forecasts,
  inCircuit,
  disabled,
}: {
  port: LivePort;
  summary: Summary | null;
  open: boolean;
  act: LiveActions;
  watchRow?: WatchRow;
  forecasts?: LiveForecast[];
  inCircuit: boolean;
  disabled: boolean;
}) {
  const ref = useRef<HTMLTableRowElement>(null);
  const addr = port.addr;
  const computed = port.computed;
  useEffect(() => {
    if (!watchRow || !computed || !ref.current) return;
    return watchRow(addr, ref.current);
  }, [watchRow, addr, computed]);
  return (
    <tr
      ref={ref}
      className={port.cursor ? 'live-port here' : 'live-port'}
      data-port={port.port}
      data-computed={port.computed ? 'yes' : 'no'}
      data-circuit={inCircuit ? 'yes' : undefined}
      aria-current={port.cursor ? 'step' : undefined}
    >
      <td>
        {port.cursor ? '▶' : ''}
        {inCircuit && (
          <span className="circuit-mark" title="a node of the selected circuit">
            ◆
          </span>
        )}
      </td>
      <td className="mono">
        {port.computed ? (
          <button
            type="button"
            className="linklike"
            onClick={() => act.expand(open ? null : port.addr)}
          >
            {port.port}
          </button>
        ) : (
          port.port
        )}
      </td>
      <td className="mono muted">{port.operator}</td>
      <td>
        {port.proofs.map((p) => (
          <span key={p} className="chip">
            {p}
          </span>
        ))}
      </td>
      {port.computed ? (
        <>
          <td>
            <Reduced entry={summary?.stats ?? null} pick={(r) => ('l2' in r ? r.l2 : null)} />
          </td>
          <td>{open ? <TopEntries entry={summary?.topk ?? null} /> : null}</td>
        </>
      ) : (
        <td colSpan={2} className="faint" data-testid="not-computed">
          not computed yet
        </td>
      )}
      <td className="live-edit-cell">
        <EditMark edit={port.edit} />
      </td>
      {forecasts && (
        <td className="live-forecast" data-testid="forecast-cell">
          <ForecastCell port={port} forecasts={forecasts} act={act} disabled={disabled} />
        </td>
      )}
    </tr>
  );
}

/** A port's forecast: the latest sealed prediction (hollow: derived), and once `verify` closed
 *  it, the native value (solid: measured) with the residual and its bound. A port the forecast
 *  op can still predict (not computed yet) offers to seal one; nothing is predicted unasked. */
export function ForecastCell({
  port,
  forecasts,
  act,
  disabled,
}: {
  port: LivePort;
  forecasts: LiveForecast[];
  act: LiveActions;
  disabled: boolean;
}) {
  const f = forecasts
    .filter((x) => x.target === port.addr)
    .sort((a, b) => b.id - a.id)[0];
  if (!f) {
    return port.computed ? null : (
      <button
        type="button"
        className="linklike"
        disabled={disabled || !act.forecast}
        title={`forecast ${port.addr}: seal a prediction from values at the cursor`}
        onClick={() => act.forecast?.(port.addr)}
      >
        forecast
      </button>
    );
  }
  const v = f.verdict;
  return (
    <span className="forecast" data-forecast={f.id} data-status={f.status}>
      <span className="faint">#{f.id} pred</span> <OpValue value={f.predicted} compact />
      {v && v.native !== undefined ? (
        <>
          {' '}
          <span className="faint">native</span> <OpValue value={v.native} compact />{' '}
          <span className="faint">resid</span>{' '}
          <OpValue value={v.residual_max_abs} what="residual" compact /> ≤{' '}
          <OpValue value={v.bound_max} what="bound" compact />{' '}
          <span className="chip">{typeof v.verdict === 'string' ? v.verdict : f.status}</span>
        </>
      ) : port.computed ? (
        <>
          {' '}
          <button
            type="button"
            disabled={disabled || !act.verify}
            onClick={() => act.verify?.([f.id])}
            title="compare with the native value now that the cursor has passed it"
          >
            verify
          </button>
        </>
      ) : (
        <span className="faint" title="hollow until measured">
          {' '}
          awaiting native · bound <OpValue value={(f.bound as Record<string, unknown>).max} compact />
        </span>
      )}
    </span>
  );
}

function unresolvedNum(reason: string): Num {
  return {
    v: null,
    ev: 'unresolved',
    reason,
    unit: null,
    proof: null,
    ptr: null,
    note: null,
    recorded_authority: null,
  };
}

function Reduced({
  entry,
  pick,
}: {
  entry: ReduceEntry | null;
  pick: (r: Extract<ReduceEntry, { state: 'ok' }>['result']) => Num | null;
}) {
  if (!entry || entry.state === 'loading') return <span className="faint">…</span>;
  if (entry.state === 'refused') {
    return <NumView n={unresolvedNum(entry.refused.detail ?? entry.refused.refused)} showReason />;
  }
  const n = pick(entry.result);
  return n ? <NumView n={n} /> : <span className="faint">n/a</span>;
}

function TopEntries({ entry }: { entry: ReduceEntry | null }) {
  if (!entry || entry.state === 'loading') return <span className="faint">…</span>;
  if (entry.state === 'refused') {
    return <NumView n={unresolvedNum(entry.refused.detail ?? entry.refused.refused)} />;
  }
  const r = entry.result;
  if (r.op !== 'topk') return null;
  return (
    <span className="live-top">
      {r.entries.map((e) => (
        <span key={e.index.join(',')} className="chip">
          [{e.index.join(',')}] <NumView n={e.v} />
        </span>
      ))}
    </span>
  );
}

/** The edit status of a computed value (`Run.read_info`); nothing for an uncomputed port. */
export function EditMark({ edit }: { edit: LiveEdit | null }) {
  if (!edit) return null;
  if (edit.state !== 'resolved' || !edit.authority) {
    return (
      <span className="num-refused" data-ev="refused" title={edit.reason ?? ''}>
        label refused: {edit.reason ?? 'no reason given'}
      </span>
    );
  }
  const parts: ReactNode[] = [];
  if (edit.edited) {
    const label = `edited (${edit.kind}${edit.by ? ` by ${edit.by}` : ''}) · ${edit.authority}`;
    parts.push(
      <span
        key="e"
        className="chip live-mark"
        data-testid="edited"
        data-ev={edit.authority}
        title={label}
      >
        <EvidenceMark ev={edit.authority} /> {label}
      </span>,
    );
  }
  const total = edit.upstream_total;
  if (total) {
    const listed = edit.upstream.map((u) => `t${u.row}.L${u.layer}.${u.port} ${u.op}`);
    const label = `downstream of ${total} edit${total === 1 ? '' : 's'}: ${listed.join(', ')}${
      total > listed.length ? `, and ${total - listed.length} more` : ''
    }`;
    parts.push(
      <span key="u" className="chip live-mark" data-testid="downstream" title={label}>
        downstream of <Count n={total} /> edit
        {total === 1 ? '' : 's'}
      </span>,
    );
  }
  return parts.length ? <>{parts}</> : <span className="faint">native</span>;
}

// --------------------------------------------------------------------------- tables

interface TableItem {
  id: number;
  kind: string;
  pattern: string;
  condition?: string | null;
  predicate?: string | null;
  mode?: string | null;
  enabled: boolean;
  hits: number;
}

function Breakpoints({
  frame,
  extra,
  disabled,
  act,
}: {
  frame: LiveFrame;
  extra: ReadonlyMap<string, Extra>;
  disabled: boolean;
  act: LiveActions;
}) {
  const [kind, setKind] = useState<'break' | 'watch'>('break');
  const [pattern, setPattern] = useState('');
  const [condition, setCondition] = useState('');
  const items = [
    ...frame.breakpoints,
    ...(extra.get('breakpoints')?.items ?? []),
  ] as unknown as TableItem[];
  return (
    <section aria-label="breakpoints and watches">
      <h3>Breakpoints and watches</h3>
      {items.length ? (
        <table>
          <tbody>
            {items.map((b) => (
              <tr key={b.id} data-enabled={b.enabled ? 'yes' : 'no'}>
                <td>
                  <Count n={b.id} />
                </td>
                <td>{b.kind}</td>
                <td className="mono">{b.pattern}</td>
                <td className="mono faint live-clip" title={b.condition ?? b.predicate ?? ''}>
                  {b.condition ?? b.predicate ?? b.mode ?? ''}
                </td>
                <td title="hits">
                  <Count n={b.hits} /> hits
                </td>
                <td>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => act.op(b.enabled ? 'disable' : 'enable', { ids: [b.id] })}
                  >
                    {b.enabled ? 'disable' : 'enable'}
                  </button>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => act.op('delete', { ids: [b.id] })}
                  >
                    delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="faint">none</div>
      )}
      <More frame={frame} field="breakpoints" extra={extra} act={act} noun="breakpoints" />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!pattern.trim()) return;
          const cond = condition.trim();
          if (kind === 'break') {
            act.op(
              'break',
              cond ? { pattern: pattern.trim(), condition: cond } : { pattern: pattern.trim() },
            );
          } else {
            act.op(
              'watch',
              cond ? { address: pattern.trim(), predicate: cond } : { address: pattern.trim() },
            );
          }
          setPattern('');
          setCondition('');
        }}
      >
        <select
          aria-label="kind"
          value={kind}
          onChange={(e) => setKind(e.target.value as 'break' | 'watch')}
        >
          <option value="break">break</option>
          <option value="watch">watch</option>
        </select>
        <input
          className="mono"
          aria-label="pattern"
          placeholder={kind === 'break' ? 'L*.p' : 'L1.h'}
          value={pattern}
          onChange={(e) => setPattern(e.target.value)}
        />
        <input
          className="mono"
          aria-label="condition"
          placeholder={kind === 'break' ? 'if: norm > 3' : 'predicate (optional)'}
          value={condition}
          onChange={(e) => setCondition(e.target.value)}
        />
        <button type="submit" disabled={disabled || !pattern.trim()}>
          add
        </button>
      </form>
    </section>
  );
}

interface EditRecord {
  port: string;
  op: string;
  kind: string;
  layer?: number;
  row?: number;
  value?: unknown;
  fired?: boolean;
}

function EditLine({ e }: { e: EditRecord }) {
  const value = asNum(e.value);
  return (
    <span className="mono">
      {e.layer !== undefined ? `L${e.layer}.` : ''}
      {e.port} {e.op} {value ? <NumView n={value} /> : <span className="faint">{e.kind}</span>}
    </span>
  );
}

function Edits({
  frame,
  extra,
  disabled,
  act,
}: {
  frame: LiveFrame;
  extra: ReadonlyMap<string, Extra>;
  disabled: boolean;
  act: LiveActions;
}) {
  const editable = frame.ports.filter((p) => p.editable).map((p) => p.port);
  const [port, setPort] = useState('');
  const [op, setOp] = useState<'replace' | 'add' | 'scale'>('scale');
  const [value, setValue] = useState('');
  const edits = (frame.edits ?? {}) as {
    staged?: EditRecord[];
    committed?: EditRecord[];
    staged_total?: number;
  };
  const staged = [
    ...(edits.staged ?? []),
    ...((extra.get('staged')?.items ?? []) as unknown as EditRecord[]),
  ];
  const committed = [
    ...(edits.committed ?? []),
    ...((extra.get('committed')?.items ?? []) as unknown as EditRecord[]),
  ];
  const anyStaged = (edits.staged_total ?? staged.length) > 0;
  const chosen = port || editable[0] || '';
  const number = Number(value);
  const valid = value.trim() !== '' && Number.isFinite(number);
  return (
    <section aria-label="edits">
      <h3>Edits</h3>
      <div>
        staged:{' '}
        {staged.length ? (
          staged.map((e, i) => (
            <div key={i} data-testid="staged-edit">
              <EditLine e={e} />
            </div>
          ))
        ) : (
          <span className="faint">none</span>
        )}
        <More frame={frame} field="staged" extra={extra} act={act} noun="staged edits" />
      </div>
      {committed.length > 0 && (
        <div>
          committed:{' '}
          {committed.map((e, i) => (
            <div key={i} data-testid="committed-edit">
              <span className="faint">t{e.row ?? '?'} </span>
              <EditLine e={e} />
            </div>
          ))}
          <More frame={frame} field="committed" extra={extra} act={act} noun="committed edits" />
        </div>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!valid || !chosen || frame.layer === null) return;
          act.op('set', { target: `L${frame.layer}.${chosen}`, value: number, op });
          setValue('');
        }}
      >
        <select aria-label="edit port" value={chosen} onChange={(e) => setPort(e.target.value)}>
          {editable.map((p) => (
            <option key={p} value={p}>
              L{frame.layer ?? '?'}.{p}
            </option>
          ))}
        </select>
        <select
          aria-label="edit op"
          value={op}
          onChange={(e) => setOp(e.target.value as 'replace' | 'add' | 'scale')}
        >
          <option value="replace">replace</option>
          <option value="add">add</option>
          <option value="scale">scale</option>
        </select>
        <input
          className="mono"
          aria-label="scalar"
          inputMode="decimal"
          placeholder="scalar"
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <button type="submit" disabled={disabled || !valid || !chosen}>
          stage
        </button>
      </form>
      <button type="button" disabled={disabled || !anyStaged} onClick={() => act.op('commit')}>
        commit
      </button>
      <button type="button" disabled={disabled || !anyStaged} onClick={() => act.op('abort')}>
        abort
      </button>
    </section>
  );
}

function Checkpoints({
  frame,
  extra,
  disabled,
  act,
}: {
  frame: LiveFrame;
  extra: ReadonlyMap<string, Extra>;
  disabled: boolean;
  act: LiveActions;
}) {
  const [name, setName] = useState('');
  const rows = [
    ...frame.checkpoints,
    ...((extra.get('checkpoints')?.items ?? []) as unknown as LiveCheckpoint[]),
  ];
  return (
    <section aria-label="checkpoints">
      <h3>Checkpoints</h3>
      {rows.length ? (
        <table>
          <tbody>
            {rows.map((c: LiveCheckpoint) => (
              <tr key={c.name}>
                <td className="mono">{c.name}</td>
                <td className="mono faint">{c.where}</td>
                <td
                  className="faint"
                  title={c.store === 'durable' ? 'stored in the project' : 'this process'}
                >
                  {c.store}
                </td>
                <td>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => act.op('restart', { name: c.name })}
                  >
                    restart
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="faint">none</div>
      )}
      <More frame={frame} field="checkpoints" extra={extra} act={act} noun="checkpoints" />
      <form
        onSubmit={(e) => {
          e.preventDefault();
          act.op('checkpoint', name.trim() ? { name: name.trim() } : {});
          setName('');
        }}
      >
        <input
          className="mono"
          aria-label="checkpoint name"
          placeholder="name (optional)"
          maxLength={64}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button type="submit" disabled={disabled || frame.where === null}>
          checkpoint
        </button>
      </form>
    </section>
  );
}

// --------------------------------------------------------------------------- console

function Console({
  lines,
  disabled,
  act,
}: {
  lines: ConsoleLine[];
  disabled: boolean;
  act: LiveActions;
}) {
  const [text, setText] = useState('');
  return (
    <section className="live-console" aria-label="console">
      <h3>Console</h3>
      <pre className="mono" data-testid="console-log">
        {lines
          .slice(-40)
          .map((l) => `(mdb) ${l.line}\n${l.text}\n`)
          .join('')}
      </pre>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!text.trim()) return;
          act.command(text);
          setText('');
        }}
      >
        <input
          className="mono"
          aria-label="command"
          placeholder="any REPL command: print L0.q top=3 | info edits | help"
          value={text}
          onChange={(e) => setText(e.target.value)}
          style={{ width: '100%' }}
        />
        <button type="submit" disabled={disabled || !text.trim()}>
          run
        </button>
      </form>
    </section>
  );
}
