// What a recorded project's engine records hold, drawn with the Live tab's components, read-only:
// the symbol table of each `symbol-table` record and the circuit of each `circuit` record (the
// same layered graph and keyboard walk). A project written by `mdb engine --project --script`
// retains no cuts: the views that need cuts (the timeline, the recorded-history graph) say so
// plainly instead of drawing an empty pane.
import { useState, type ReactNode } from 'react';
import type { RecordPage, SessionSummary, StoredBody } from '../../api/types.gen';
import { Show } from '../../chrome/Loaded';
import { Count } from '../../chrome/NumView';
import { useView } from '../../state/records';
import { StoredCircuit } from '../Circuit/CircuitView';
import { ViewLegend } from '../Live/ViewLegend';
import { StoredSymbolTable } from '../LiveSymbols/LiveSymbols';

/** As in the Live tabs: the evidence mark carries the authority; "show labels" spells it out. */
function useLabels(): [boolean, (on: boolean) => void, string] {
  const [labels, setLabels] = useState(false);
  return [labels, setLabels, labels ? 'labels-on' : 'labels-off'];
}

export type DrawnKind = 'symbol-table' | 'circuit';

/** The session's record kinds and their counts (`records` view), or null while loading. */
export function useRecordKinds(sid: string | null): Record<string, number> | null {
  const loaded = useView('records', sid ? `/s/${encodeURIComponent(sid)}/records` : null, {
    page: 0,
  });
  return loaded.state === 'ok' ? loaded.env.data.kinds : null;
}

/** What a view that needs cuts says when the session retains none (after "no retained cuts"). */
export function noCutsText(session: SessionSummary | undefined, what: string): string {
  const records = session ? `${session.records} record${session.records === 1 ? '' : 's'}` : '';
  return (
    `this session holds ${records || 'records'} and no state cuts, so the ${what} has nothing ` +
    'to draw (a project written by `mdb engine --project --script` keeps records only)'
  );
}

/** A compact panel standing in for a view that needs cuts. */
export function NoCuts({
  title,
  what,
  session,
  children,
}: {
  title: string;
  what: string;
  session: SessionSummary | undefined;
  children?: ReactNode;
}) {
  return (
    <section className="panel" aria-label={title.toLowerCase()} data-testid={`no-cuts-${title}`}>
      <h2>{title}</h2>
      <div className="panel-body">
        <div className="na" data-testid="no-cuts">
          <strong>no retained cuts</strong>: {noCutsText(session, what)}
        </div>
        {children}
      </div>
    </section>
  );
}

/** One page of records of a kind, a picker over them, and the chosen record's body. */
function useStored(sid: string, kind: DrawnKind) {
  const base = `/s/${encodeURIComponent(sid)}`;
  const [page, setPage] = useState(0);
  const list = useView('records', `${base}/records`, { kind, page });
  const [chosen, setChosen] = useState<string | null>(null);
  const rows = list.state === 'ok' ? list.env.data.rows : [];
  const id = chosen ?? rows[0]?.id ?? null;
  const body = useView('stored', id ? `${base}/stored/${encodeURIComponent(id)}` : null);
  return { list, rows, id, setChosen, body, page, setPage };
}

function Picker({
  label,
  list,
  id,
  choose,
  page,
  setPage,
}: {
  label: string;
  list: RecordPage;
  id: string | null;
  choose: (id: string) => void;
  page: number;
  setPage: (n: number) => void;
}) {
  return (
    <div className="recorded-picker">
      <select aria-label={label} value={id ?? ''} onChange={(e) => choose(e.target.value)}>
        {list.rows.map((r) => (
          <option key={r.id} value={r.id}>
            {r.title} · {r.id.slice(0, 12)}
          </option>
        ))}
      </select>{' '}
      <span className="faint">
        <Count n={list.total} /> recorded
      </span>
      {list.pages > 1 && (
        <>
          {' '}
          <button type="button" disabled={page <= 0} onClick={() => setPage(page - 1)}>
            ‹
          </button>{' '}
          page <Count n={page + 1} /> of <Count n={list.pages} />{' '}
          <button type="button" disabled={page >= list.pages - 1} onClick={() => setPage(page + 1)}>
            ›
          </button>
        </>
      )}
    </div>
  );
}

function Origin({ stored }: { stored: StoredBody }) {
  return (
    <div className="faint recorded-origin">
      record <code title={stored.id}>{stored.id.slice(0, 16)}</code> · sealed {stored.authority} ·
      read-only (no live model: nothing here seeks or measures)
    </div>
  );
}

/** The symbol tables the session recorded (`symbol-table` records). */
export function RecordedSymbols({ sid }: { sid: string }) {
  const s = useStored(sid, 'symbol-table');
  const [labels, setLabels, cls] = useLabels();
  return (
    <section className="panel" aria-label="symbols" data-testid="recorded-symbols">
      <h2>Symbols (recorded tables)</h2>
      <div className={`panel-body ${cls}`}>
        <ViewLegend labels={labels} onLabels={setLabels} />
        <Show what="symbol-table records" loaded={s.list}>
          {(list) =>
            list.total === 0 ? (
              <div className="na">
                No <code>symbol-table</code> record in this session (run <code>symbols</code> with a
                project attached to record one).
              </div>
            ) : (
              <>
                <Picker
                  label="symbol table"
                  list={list}
                  id={s.id}
                  choose={s.setChosen}
                  page={s.page}
                  setPage={(n) => {
                    s.setPage(n);
                    s.setChosen(null);
                  }}
                />
                {s.id && (
                  <Show what="symbol table" loaded={s.body}>
                    {(stored) => (
                      <>
                        <Origin stored={stored} />
                        <StoredSymbolTable result={stored.body} />
                      </>
                    )}
                  </Show>
                )}
              </>
            )
          }
        </Show>
      </div>
    </section>
  );
}

/** The circuits the session recorded (`circuit` records), as the Live Circuits graph. */
export function RecordedCircuits({ sid }: { sid: string }) {
  const s = useStored(sid, 'circuit');
  const [labels, setLabels, cls] = useLabels();
  return (
    <section className="panel" aria-label="circuits" data-testid="recorded-circuits">
      <h2>Circuits (recorded)</h2>
      <div className={`panel-body ${cls}`}>
        <ViewLegend labels={labels} onLabels={setLabels} />
        <Show what="circuit records" loaded={s.list}>
          {(list) =>
            list.total === 0 ? (
              <div className="na">
                No <code>circuit</code> record in this session (run <code>circuit CONSUMER</code>{' '}
                with a project attached to record one).
              </div>
            ) : (
              <>
                <Picker
                  label="circuit"
                  list={list}
                  id={s.id}
                  choose={s.setChosen}
                  page={s.page}
                  setPage={(n) => {
                    s.setPage(n);
                    s.setChosen(null);
                  }}
                />
                {s.id && (
                  <Show what="circuit" loaded={s.body}>
                    {(stored) => (
                      <>
                        <Origin stored={stored} />
                        <StoredCircuit
                          id={
                            stored.ledger_id ??
                            list.rows.findIndex((r) => r.id === stored.id) + 1 + s.page * 25
                          }
                          body={stored.body}
                          source={{ kind: 'stored', sid, record: stored.id }}
                        />
                        {stored.ledger_id === null && (
                          <div className="faint" data-testid="no-ledger-id">
                            this record stores no ledger id: the number above is its position in
                            this list
                          </div>
                        )}
                      </>
                    )}
                  </Show>
                )}
              </>
            )
          }
        </Show>
      </div>
    </section>
  );
}
