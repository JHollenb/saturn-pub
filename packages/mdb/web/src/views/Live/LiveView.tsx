// The live debugger, drawn from the live frame and clock. Pure: every action goes through `act`,
// which the container (`Live.tsx`) wires to registry ops. Numbers come only from op results
// (`{value, authority}`) and server reductions (`Num`), always through NumView; a port the run
// has not computed shows no value at all.
import { useState, type ReactNode } from 'react';
import type {
  LiveCheckpoint,
  LiveEdit,
  LiveFrame,
  LivePort,
  LiveTimeline,
  Num,
  Refused,
} from '../../api/types.gen';
import { EvidenceMark } from '../../chrome/EvidenceMark';
import { RefusedView } from '../../chrome/NotAvailable';
import { Count, NumView } from '../../chrome/NumView';
import { asNum } from '../../chrome/evidence';
import type { ReduceEntry } from '../../state/reduce';
import { summaryKey, type ConsoleLine, type Outcome, type Summary } from '../../state/live';

export interface LiveActions {
  op(name: string, args?: Record<string, unknown>): void;
  command(line: string): void;
  interrupt(): void;
  expand(addr: string | null): void;
}

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
  act: LiveActions;
}

export function LiveView(props: LiveViewProps) {
  const { frame, running, act } = props;
  if (!frame) {
    return props.refused ? (
      <RefusedView what="live session" refused={props.refused} />
    ) : (
      <div className="faint">loading live session…</div>
    );
  }
  const hasRun = frame.where !== null;
  return (
    <div className="live" data-testid="live" data-running={running ? 'yes' : 'no'}>
      <Status {...props} frame={frame} />
      <Controls frame={frame} running={running} act={act} urlAddr={props.urlAddr} />
      {props.refused && <RefusedView what="request" refused={props.refused} />}
      {hasRun ? (
        <>
          {props.timeline && <Clock timeline={props.timeline} disabled={!!running} act={act} />}
          <Pipeline frame={frame} summaries={props.summaries} expanded={props.expanded} act={act} />
        </>
      ) : (
        <div className="na" data-testid="no-run">
          No run yet: give token ids (or text, when the model has a tokenizer) and press start.
        </div>
      )}
      <div className="live-tables">
        <Breakpoints frame={frame} disabled={!!running} act={act} />
        <Edits frame={frame} disabled={!!running || !hasRun} act={act} />
        <Checkpoints frame={frame} disabled={!!running} act={act} />
      </div>
      <Console lines={props.console} disabled={!!running} act={act} />
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
  frame: LiveFrame;
  running: string | null;
  act: LiveActions;
  urlAddr: string | null;
}) {
  const [prompt, setPrompt] = useState('');
  const [until, setUntil] = useState('');
  const busy = !!running;
  const noRun = frame.where === null;
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
}: {
  frame: LiveFrame;
  summaries: ReadonlyMap<string, Summary>;
  expanded: string | null;
  act: LiveActions;
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
}: {
  port: LivePort;
  summary: Summary | null;
  open: boolean;
  act: LiveActions;
}) {
  return (
    <tr
      className={port.cursor ? 'live-port here' : 'live-port'}
      data-port={port.port}
      data-computed={port.computed ? 'yes' : 'no'}
      aria-current={port.cursor ? 'step' : undefined}
    >
      <td>{port.cursor ? '▶' : ''}</td>
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
      <td>
        <EditMark edit={port.edit} />
      </td>
    </tr>
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
    parts.push(
      <span key="e" className="chip" data-testid="edited" data-ev={edit.authority}>
        <EvidenceMark ev={edit.authority} /> edited ({edit.kind}
        {edit.by ? ` by ${edit.by}` : ''}) · {edit.authority}
      </span>,
    );
  }
  if (edit.upstream.length) {
    const where = edit.upstream.map((u) => `t${u.row}.L${u.layer}.${u.port} ${u.op}`).join(', ');
    parts.push(
      <span key="u" className="chip" data-testid="downstream" title={where}>
        downstream of <Count n={edit.upstream.length} /> edit
        {edit.upstream.length === 1 ? '' : 's'}
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
  disabled,
  act,
}: {
  frame: LiveFrame;
  disabled: boolean;
  act: LiveActions;
}) {
  const [kind, setKind] = useState<'break' | 'watch'>('break');
  const [pattern, setPattern] = useState('');
  const [condition, setCondition] = useState('');
  const items = frame.breakpoints as unknown as TableItem[];
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
                <td className="mono faint">{b.condition ?? b.predicate ?? b.mode ?? ''}</td>
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
  disabled,
  act,
}: {
  frame: LiveFrame;
  disabled: boolean;
  act: LiveActions;
}) {
  const editable = frame.ports.filter((p) => p.editable).map((p) => p.port);
  const [port, setPort] = useState('');
  const [op, setOp] = useState<'replace' | 'add' | 'scale'>('scale');
  const [value, setValue] = useState('');
  const edits = (frame.edits ?? {}) as {
    staged?: EditRecord[];
    committed?: { row: number; layer: number; edits: EditRecord[] }[];
  };
  const staged = edits.staged ?? [];
  const committed = edits.committed ?? [];
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
      </div>
      {committed.length > 0 && (
        <div>
          committed:{' '}
          {committed.map((g) => (
            <div key={`${g.row}.${g.layer}`}>
              t{g.row}.L{g.layer}:{' '}
              {g.edits.map((e, i) => (
                <EditLine key={i} e={e} />
              ))}
            </div>
          ))}
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
      <button type="button" disabled={disabled || !staged.length} onClick={() => act.op('commit')}>
        commit
      </button>
      <button type="button" disabled={disabled || !staged.length} onClick={() => act.op('abort')}>
        abort
      </button>
    </section>
  );
}

function Checkpoints({
  frame,
  disabled,
  act,
}: {
  frame: LiveFrame;
  disabled: boolean;
  act: LiveActions;
}) {
  const [name, setName] = useState('');
  return (
    <section aria-label="checkpoints">
      <h3>Checkpoints</h3>
      {frame.checkpoints.length ? (
        <table>
          <tbody>
            {frame.checkpoints.map((c: LiveCheckpoint) => (
              <tr key={`${c.store}:${c.name}:${c.where}`}>
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
