// The symbol table at the cursor (the `symbols` op) as a circuit map: one row per binding with
// its status, definition sites (click: seek there, after a confirmation), a lifetime strip
// (reader rows x layers, cursor marked), its readers with mass and exact effect side by side
// (and the causal verdict once `reads` measured it), and the always-present Unbound row.
// Pure: every action goes through `act`.
import { useState } from 'react';
import type { LiveSymbol } from '../../api/types.gen';
import { Count } from '../../chrome/NumView';
import { OpValue } from '../../chrome/OpValue';
import { at, cellOf, int, isRec, recs, str, strs, type Rec } from '../../chrome/labelled';
import type { FocusRead, Slot } from '../../state/inspect';
import { SlotState } from '../Live/SlotState';
import { DEFAULT_LEGEND, legendOf, markClass, parseStrip } from './strip';

export interface SymbolsActions {
  /** Re-run the `symbols` op with these arguments. */
  run(args: Record<string, unknown>): void;
  declare(name: string, addresses: string): void;
  /** Measure the causal verdicts of a source row's reads (`reads SOURCE`; moves and restores). */
  measure(source: string, consumer: string | null): void;
  openRead(focus: FocusRead): void;
  seek(target: string, from: string): void;
  interrupt(): void;
}

export interface LiveSymbolsProps {
  slot: Slot;
  reads: Slot;
  args: Record<string, unknown>;
  where: string | null;
  busy: boolean;
  declared: LiveSymbol[];
  act: SymbolsActions;
}

/** Keys of a binding this view draws itself; any other labelled field is listed generically. */
const KNOWN = new Set([
  'symbol',
  'kind',
  'rows',
  'definitions',
  'lifetime',
  'mass',
  'readers',
  'status',
  'token',
  'text',
  'to_consumer',
  'edited_by',
  'omitted',
  'count',
  'mean_mass',
]);

export function LiveSymbols(props: LiveSymbolsProps) {
  const { slot, act } = props;
  const result = slot.state === 'done' ? slot.result : null;
  return (
    <div className="live-symbols" data-testid="live-symbols">
      <ConsumerForm args={props.args} busy={props.busy} act={act} />
      <SlotState
        slot={slot}
        what="symbols"
        onInterrupt={act.interrupt}
        idle="No symbol table yet: choose a consumer (or none) and press show."
      />
      {result && <Table result={result} {...props} />}
      <DeclareForm busy={props.busy} declared={props.declared} act={act} />
    </div>
  );
}

// --------------------------------------------------------------------------- consumer

type ConsumerKind = 'none' | 'margin' | 'logit' | 'port';

/** The `consumer=` argument the form builds: `M[a,b]`, `logits[i]` or a port address. */
export function consumerText(kind: ConsumerKind, a: string, b: string, port: string): string | null {
  if (kind === 'margin') return a.trim() && b.trim() ? `M[${a.trim()},${b.trim()}]` : null;
  if (kind === 'logit') return a.trim() ? `logits[${a.trim()}]` : null;
  if (kind === 'port') return port.trim() || null;
  return null;
}

function ConsumerForm({
  args,
  busy,
  act,
}: {
  args: Record<string, unknown>;
  busy: boolean;
  act: SymbolsActions;
}) {
  const [kind, setKind] = useState<ConsumerKind>('none');
  const [a, setA] = useState('');
  const [b, setB] = useState('');
  const [port, setPort] = useState('');
  const consumer = consumerText(kind, a, b, port);
  const ready = kind === 'none' || consumer !== null;
  return (
    <form
      className="symbols-consumer"
      aria-label="symbols consumer"
      onSubmit={(e) => {
        e.preventDefault();
        if (!ready) return;
        act.run(consumer ? { consumer } : {});
      }}
    >
      <label>
        consumer{' '}
        <select
          aria-label="consumer kind"
          value={kind}
          onChange={(e) => setKind(e.target.value as ConsumerKind)}
        >
          <option value="none">none (per-layer o)</option>
          <option value="margin">margin M[a,b]</option>
          <option value="logit">logit</option>
          <option value="port">a port</option>
        </select>
      </label>
      {(kind === 'margin' || kind === 'logit') && (
        <input
          className="mono"
          aria-label={kind === 'margin' ? 'margin token a' : 'logit index'}
          placeholder={kind === 'margin' ? 'a (id)' : 'i'}
          value={a}
          onChange={(e) => setA(e.target.value)}
          size={6}
        />
      )}
      {kind === 'margin' && (
        <input
          className="mono"
          aria-label="margin token b"
          placeholder="b (id)"
          value={b}
          onChange={(e) => setB(e.target.value)}
          size={6}
        />
      )}
      {kind === 'port' && (
        <input
          className="mono"
          aria-label="consumer port"
          placeholder="L1.h_out"
          value={port}
          onChange={(e) => setPort(e.target.value)}
          size={12}
        />
      )}
      <button type="submit" disabled={busy || !ready}>
        show symbols
      </button>
      {typeof args.consumer === 'string' && (
        <span className="faint">
          last asked: <code>consumer={args.consumer}</code>
        </span>
      )}
    </form>
  );
}

// --------------------------------------------------------------------------- table

/** The readers' causal verdicts of the last `reads` result, by reader name, when it measured
 *  the given source row with the same consumer. */
function verdicts(reads: Slot, row: number, consumer: string | null): Map<string, Rec> {
  const out = new Map<string, Rec>();
  if (reads.state !== 'done') return out;
  const r = reads.result;
  if (int(at(r, 'source', 'row')) !== row) return out;
  if (consumer !== null && str(r.consumer) !== consumer) return out;
  for (const read of recs(r.reads)) {
    const name = str(read.name);
    if (name && isRec(read.causal)) out.set(name, read.causal);
  }
  return out;
}

function Table({
  result,
  reads,
  where,
  busy,
  act,
  args,
}: { result: Rec } & LiveSymbolsProps) {
  const bindings = recs(result.bindings);
  const consumer = str(result.consumer);
  const here = cellOf(str(result.where) ?? where ?? '');
  const stale = where !== null && str(result.where) !== null && str(result.where) !== where;
  const page = isRec(result.page) ? result.page : {};
  const offset = int(page.offset) ?? 0;
  const limit = int(page.limit) ?? bindings.length;
  const total = int(page.total);
  const legends = bindings.map((b) => legendOf(b.lifetime)).find(Boolean) ?? null;
  const legend = legends ?? DEFAULT_LEGEND;
  // disclosure per binding: open for the first one, any binding whose rows hold the cursor, and
  // the row whose verdicts were just measured, until the user toggles it
  const [toggled, setToggled] = useState<Map<string, boolean>>(new Map());
  return (
    <div className="symbols-result">
      <div className="symbols-head">
        symbols at <code>{str(result.where) ?? '?'}</code> · consumer{' '}
        <code>{consumer ?? 'none'}</code>
        {result.consumer_total !== undefined && (
          <>
            {' '}
            = <OpValue value={result.consumer_total} what="consumer_total" />
          </>
        )}{' '}
        · min_mass <OpValue value={result.min_mass} what="min_mass" />
        {stale && (
          <span className="chip" data-testid="symbols-stale" title="the cursor moved since">
            cursor moved to {where}: <button
              type="button"
              className="linklike"
              disabled={busy}
              onClick={() => act.run(args)}
            >
              refresh
            </button>
          </span>
        )}
      </div>
      <div className="table-scroll">
        <table className="symbols-table">
          <thead>
            <tr>
              <th>symbol</th>
              <th>status</th>
              <th title="reader rows x layers; the cursor's cell is outlined">lifetime</th>
              <th
                className="n"
                title="attention weight its rows receive, summed over heads and layers: a sum of weights, not a probability"
              >
                mass (Σ heads, layers)
              </th>
              <th className="n">reads</th>
              <th className="n">mean per read</th>
              <th className="n">to consumer</th>
            </tr>
          </thead>
          <tbody>
            {bindings.map((b, i) => {
              const name = str(b.symbol) ?? `#${i}`;
              const rows = rowsOf(b);
              const measuredRow =
                reads.state === 'done' ? int(at(reads.result, 'source', 'row')) : null;
              const auto =
                i === 0 ||
                (here !== null && rows.includes(here[0])) ||
                (measuredRow !== null && rows.length === 1 && rows[0] === measuredRow);
              const open = toggled.get(name) ?? auto;
              return (
                <BindingRows
                  key={name}
                  b={b}
                  open={open}
                  onToggle={() => setToggled(new Map(toggled).set(name, !open))}
                  here={here}
                  legend={legend}
                  reads={reads}
                  consumer={consumer}
                  busy={busy}
                  act={act}
                />
              );
            })}
            <UnboundRows unbound={result.unbound} />
          </tbody>
        </table>
      </div>
      <Legend legend={legend} fromOp={legends !== null} />
      <Accounting accounting={result.accounting} />
      {total !== null && (
        <div className="symbols-page">
          bindings <Count n={total === 0 ? 0 : offset + 1} />–<Count n={offset + bindings.length} /> of{' '}
          <Count n={total} />{' '}
          <button
            type="button"
            disabled={busy || offset <= 0}
            onClick={() => act.run({ ...args, offset: Math.max(0, offset - limit) })}
          >
            ‹ previous
          </button>{' '}
          <button
            type="button"
            disabled={busy || offset + bindings.length >= total}
            onClick={() => act.run({ ...args, offset: offset + limit })}
          >
            next ›
          </button>
        </div>
      )}
      <ReadsState reads={reads} act={act} />
    </div>
  );
}

function ReadsState({ reads, act }: { reads: Slot; act: SymbolsActions }) {
  // a finished measurement shows in its binding's reader lines; only the other states (running,
  // refused, interrupted) need a line of their own
  if (reads.state === 'idle' || (reads.state === 'done' && reads.partial === null)) return null;
  return (
    <div className="symbols-reads">
      verdicts: <SlotState slot={reads} what="reads" onInterrupt={act.interrupt} idle="" />
    </div>
  );
}

/** A definition site as the occurrence `seek` moves to: the first layer of a `L*` site, the
 *  layer-0 key of a bare token row, the row of a cache cell. */
export function siteTarget(site: string): string | null {
  const plain = site.replace(/\s*\(.*\)$/, '').trim();
  const cache = /^cache\.L(\d+)\.([kv])\[.*row=(\d+).*\]$/.exec(plain);
  if (cache) return `t${cache[3]}.L${cache[1]}.${cache[2]}`;
  const row = /^t(\d+)$/.exec(plain);
  if (row) return `t${row[1]}.L0.k`;
  const port = /^t(\d+)\.L(\*|\d+)\.([A-Za-z_]+)/.exec(plain);
  if (port) return `t${port[1]}.L${port[2] === '*' ? '0' : port[2]}.${port[3]}`;
  return null;
}

function rowsOf(b: Rec): number[] {
  return Array.isArray(b.rows)
    ? (b.rows as unknown[]).filter((r): r is number => typeof r === 'number')
    : [];
}

/** `text` shortened in the middle to `max` characters (the full text goes in a title). */
export function midTrunc(text: string, max = 32): string {
  if (text.length <= max) return text;
  const keep = max - 1;
  return `${text.slice(0, Math.ceil(keep / 2))}…${text.slice(text.length - Math.floor(keep / 2))}`;
}

/** One binding: a compact row (name, status, lifetime, mass, reads, mean per read, to-consumer)
 *  and, behind its disclosure, the definition sites and one line per reader. */
function BindingRows({
  b,
  open,
  onToggle,
  here,
  legend,
  reads,
  consumer,
  busy,
  act,
}: {
  b: Rec;
  open: boolean;
  onToggle: () => void;
  here: [number, number | null] | null;
  legend: Record<string, string>;
  reads: Slot;
  consumer: string | null;
  busy: boolean;
  act: SymbolsActions;
}) {
  const name = str(b.symbol) ?? '?';
  const rows = rowsOf(b);
  const measured = rows.length === 1 ? verdicts(reads, rows[0] ?? -1, consumer) : new Map();
  const token = int(b.token);
  const status = str(b.status) ?? 'unknown';
  const readers = recs(b.readers);
  const omitted = int(b.omitted);
  const extra = Object.entries(b).filter(([k]) => !KNOWN.has(k));
  const sub = `symbol-sub-${name.replace(/[^A-Za-z0-9_-]/g, '_')}`;
  return (
    <>
      <tr className="symbol-row" data-symbol={name} data-status={status} data-open={open ? 'yes' : 'no'}>
        <td className="symbol-name">
          <button
            type="button"
            className="disclosure"
            aria-expanded={open}
            aria-controls={sub}
            title={open ? 'hide sites and readers' : 'show sites and readers'}
            onClick={onToggle}
          >
            {open ? '▾' : '▸'}
          </button>{' '}
          <strong className="mono">{name}</strong>
          {token !== null && (
            <span className="faint">
              {' '}
              id <Count n={token} />
            </span>
          )}
          {str(b.text) !== null && <span className="mono"> “{str(b.text)}”</span>}{' '}
          <span className="faint">{str(b.kind) ?? ''}</span>
        </td>
        <td>
          <span className={`chip status-chip status-${status}`} data-testid="status-chip">
            {status}
          </span>
          {Array.isArray(b.edited_by) && (
            <span className="faint" title={JSON.stringify(b.edited_by).slice(0, 400)}>
              {' '}
              by <Count n={b.edited_by.length} /> edit{b.edited_by.length === 1 ? '' : 's'}
            </span>
          )}
        </td>
        <td>
          <Lifetime lifetime={b.lifetime} here={here} legend={legend} />
        </td>
        <td className="n" data-label="mass">
          <OpValue value={b.mass} what="mass" />
        </td>
        <td className="n" data-label="reads">
          {int(b.count) !== null ? <Count n={int(b.count)!} /> : <span className="faint">—</span>}
        </td>
        <td className="n" data-label="mean per read">
          {b.mean_mass !== undefined ? (
            <OpValue value={b.mean_mass} what="mean_mass" />
          ) : (
            <span className="faint">—</span>
          )}
        </td>
        <td className="n" data-label="to consumer">
          <OpValue value={b.to_consumer} what="to_consumer" />
        </td>
      </tr>
      {open && (
        <tr className="symbol-sub" id={sub} data-sub={name}>
          <td colSpan={7}>
            <div className="sub-line">
              <span className="faint">defined at</span>{' '}
              {strs(b.definitions).map((site) => {
                const target = siteTarget(site);
                return target ? (
                  <button
                    key={site}
                    type="button"
                    className="linklike mono site"
                    title={`${site}: seek ${target} (asks first: it moves the session)`}
                    onClick={() => act.seek(target, `definition ${site} of ${name}`)}
                  >
                    {midTrunc(site)}
                  </button>
                ) : (
                  <span key={site} className="mono site" title={site}>
                    {midTrunc(site)}
                  </span>
                );
              })}
              {extra.map(([k, v]) => (
                <span key={k} className="faint" data-field={k}>
                  {' '}
                  · {k}{' '}
                  {typeof v === 'number' && Number.isInteger(v) ? (
                    <Count n={v} />
                  ) : (
                    <OpValue value={v} what={k} />
                  )}
                </span>
              ))}
            </div>
            <div className="reader-lines" role="table" aria-label={`readers of ${name}`}>
              <div className="reader-line reader-head faint" role="row">
                <span role="columnheader">reader</span>
                <span role="columnheader" className="n">
                  mass
                </span>
                <span role="columnheader" className="n">
                  exact effect
                </span>
                <span role="columnheader">causal verdict</span>
              </div>
              {readers.map((r, i) => {
                const reader = str(r.name) ?? '?';
                const causal = measured.get(reader);
                return (
                  <div className="reader-line" role="row" key={`${reader}-${i}`} data-reader={reader}>
                    <span role="cell">
                      <button
                        type="button"
                        className="linklike mono"
                        title={`${reader}: open this read in the Circuits view`}
                        onClick={() => act.openRead({ reader, binding: name, rows })}
                      >
                        {midTrunc(reader, 24)}
                      </button>
                    </span>
                    <span role="cell" className="n">
                      <OpValue value={r.mass} what="mass" />
                    </span>
                    <span role="cell" className="n">
                      <OpValue value={r.effect} what="effect" />
                    </span>
                    <span role="cell">
                      {causal ? (
                        <Verdict causal={causal} />
                      ) : (
                        <span className="faint" title="press measure verdicts for a native verdict">
                          not measured
                        </span>
                      )}
                    </span>
                  </div>
                );
              })}
              {omitted !== null && omitted > 0 && (
                <div className="reader-line faint" role="row">
                  <span role="cell">
                    +<Count n={omitted} /> more reads
                  </span>
                </div>
              )}
            </div>
            {rows.length === 1 ? (
              <button
                type="button"
                disabled={busy}
                title="reads tN: measures each listed read natively (moves the session and restores it)"
                onClick={() => act.measure(`t${rows[0]}`, consumer)}
              >
                measure verdicts
              </button>
            ) : (
              <span className="faint">verdicts are measured per source row</span>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

/** A read's native causal verdict (`reads`): the op's verdict text, the measured change, the
 *  predicted one and the rounding bound. */
export function Verdict({ causal }: { causal: Rec }) {
  const verdict = str(causal.verdict) ?? 'no verdict';
  const reason = str(causal.reason);
  return (
    <span className="verdict" data-verdict={verdict}>
      <span className="chip verdict-chip" title={reason ?? undefined}>
        {verdict}
      </span>
      {reason && <span className="faint"> {reason}</span>} <span className="faint">Δ</span>{' '}
      <OpValue value={causal.delta} what="delta" /> <span className="faint">pred</span>{' '}
      <OpValue value={causal.predicted} what="predicted" /> <span className="faint">≤</span>{' '}
      <OpValue value={causal.rounding_bound} what="rounding_bound" />
    </span>
  );
}

function Lifetime({
  lifetime,
  here,
  legend,
}: {
  lifetime: unknown;
  here: [number, number | null] | null;
  legend: Record<string, string>;
}) {
  const rows = parseStrip(str(at(lifetime, 'strip')));
  if (!rows) return <OpValue value={undefined} what="lifetime strip" />;
  const first = str(at(lifetime, 'first'));
  const last = str(at(lifetime, 'last'));
  return (
    <div className="lifetime" title={`${first ?? '—'} .. ${last ?? '—'}`}>
      {rows.map((r) => (
        <div key={r.row} className="life-row">
          <span className="faint mono">t{r.row}</span>
          {r.marks.map((mark, layer) => {
            const cursor = here !== null && here[0] === r.row && here[1] === layer;
            return (
              <span
                key={layer}
                className={`life-cell ${markClass(mark, legend)}${cursor ? ' life-cursor' : ''}`}
                data-mark={mark}
                title={`t${r.row}.L${layer}: ${legend[mark] ?? `mark ${mark} (no legend)`}${cursor ? ' (cursor)' : ''}`}
                aria-label={`t${r.row}.L${layer} ${legend[mark] ?? mark}`}
              >
                {markClass(mark, legend) === 'life-other' ? mark : ''}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function Legend({ legend, fromOp }: { legend: Record<string, string>; fromOp: boolean }) {
  return (
    <div className="life-legend faint" data-testid="life-legend">
      lifetime ({fromOp ? 'legend from the op' : 'marks as the op documents them'}):{' '}
      {Object.entries(legend).map(([mark, meaning]) => (
        <span key={mark} className="life-key">
          <span className={`life-cell ${markClass(mark, legend)}`}>
            {markClass(mark, legend) === 'life-other' ? mark : ''}
          </span>{' '}
          <code>{mark}</code> {meaning}
        </span>
      ))}
      <span className="life-key">
        <span className="life-cell life-cursor" /> cursor
      </span>
    </div>
  );
}

function UnboundRows({ unbound }: { unbound: unknown }) {
  if (!isRec(unbound)) {
    return (
      <tr className="symbol-row unbound" data-symbol="Unbound">
        <td>
          <strong>Unbound</strong>
        </td>
        <td colSpan={6}>
          <OpValue value={undefined} what="Unbound row" />
        </td>
      </tr>
    );
  }
  const parts = isRec(unbound.parts) ? unbound.parts : {};
  const listed = Object.entries(parts).filter(([k]) => k !== 'meaning');
  const meaning = [str(unbound.meaning), str(parts.meaning)].filter(Boolean).join(' ');
  return (
    <>
      <tr className="symbol-row unbound" data-symbol="Unbound" data-status="unbound">
        <td className="symbol-name">
          <strong>Unbound</strong>{' '}
          <span className="faint" title={meaning || undefined}>
            what no listed binding carries
          </span>
        </td>
        <td />
        <td />
        <td className="n" data-label="mass">
          <OpValue value={unbound.mass} what="mass" />
        </td>
        <td />
        <td />
        <td className="n" data-label="to consumer">
          <OpValue value={unbound.readout} what="readout" />
        </td>
      </tr>
      <tr className="symbol-sub unbound" data-sub="Unbound">
        <td colSpan={7}>
          {listed.length ? (
            <span className="unbound-parts">
              {listed.map(([k, v]) => (
                <span key={k} data-part={k} className="unbound-part">
                  <span className="faint">{k.replace(/_/g, ' ')}</span> <OpValue value={v} what={k} />
                  {isRec(v) && v.bound !== undefined && (
                    <>
                      {' '}
                      <span className="faint">≤</span> <OpValue value={v.bound} what="bound" />
                      {v.within === true ? ' within' : v.within === false ? ' OUTSIDE' : ''}
                    </>
                  )}
                </span>
              ))}
            </span>
          ) : (
            <span className="faint">no parts given</span>
          )}
          {meaning && (
            <details className="unbound-meaning faint">
              <summary>what the parts are</summary>
              {meaning}
            </details>
          )}
        </td>
      </tr>
    </>
  );
}

function Accounting({ accounting }: { accounting: unknown }) {
  if (!isRec(accounting)) return null;
  return (
    <div className="faint symbols-accounting">
      accounting: <Count n={int(accounting.heads) ?? 0} /> heads · listed mass{' '}
      <OpValue value={accounting.listed_mass} what="listed_mass" /> · max |Σp − 1|{' '}
      <OpValue value={accounting.max_head_deviation} what="max_head_deviation" /> ≤{' '}
      <OpValue value={accounting.deviation_bound} what="deviation_bound" />{' '}
      {accounting.within_bound === true ? 'within bound' : accounting.within_bound === false ? 'OUTSIDE bound' : ''}
    </div>
  );
}

// --------------------------------------------------------------------------- declare

function DeclareForm({
  busy,
  declared,
  act,
}: {
  busy: boolean;
  declared: LiveSymbol[];
  act: SymbolsActions;
}) {
  const [name, setName] = useState('');
  const [addresses, setAddresses] = useState('');
  return (
    <section className="symbols-declare" aria-label="declare a symbol">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!name.trim() || !addresses.trim()) return;
          act.declare(name.trim(), addresses.trim());
          setName('');
          setAddresses('');
        }}
      >
        <span className="muted">symbol</span>
        <input
          className="mono"
          aria-label="symbol name"
          placeholder="NAME"
          value={name}
          onChange={(e) => setName(e.target.value)}
          size={8}
        />
        =
        <input
          className="mono"
          aria-label="symbol addresses"
          placeholder="t1, cache.L1.v[row=2]"
          value={addresses}
          onChange={(e) => setAddresses(e.target.value)}
          size={22}
        />
        <button type="submit" disabled={busy || !name.trim() || !addresses.trim()}>
          declare
        </button>
      </form>
      {declared.length > 0 && (
        <div className="faint">
          declared on this run:{' '}
          {declared.map((s) => (
            <span key={s.name} className="chip" title={s.definitions.join(', ')}>
              {s.name} = {s.definitions.join(', ')}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
